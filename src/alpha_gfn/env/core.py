import random
from typing import List, Tuple, Optional, Union, Dict, Any
import copy
import torch
import numpy as np
import math
from torch import nn
from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates
from gfn.actions import Actions

from alphagen.data.tokens import *
from alphagen.data.tree import ExpressionBuilder, OutOfDataRangeError, InvalidExpressionException, ExpressionParser
from alphagen.data.expression import Expression
from ..config import *
from ..alpha_pool import AlphaPoolGFN
from ..preprocessors import IntegerPreprocessor


class GFNEnvCore(DiscreteEnv):
    def __init__(self, pool: AlphaPoolGFN,
                 encoder: nn.Module = None,
                 device: torch.device = torch.device('cuda:0'),
                 mask_dropout_prob: float = 0.1,
                 ssl_weight: float = 0.1,
                 nov_weight: float = 0.1,
                 builder_cache_size: int = 1024):
        self.pool = pool
        self.encoder = encoder
        self.mask_dropout_prob = mask_dropout_prob
        self.ssl_weight = ssl_weight
        self.nov_weight = nov_weight
        self.builder = ExpressionBuilder()
        self.parser = ExpressionParser()

        # Cache for ExpressionBuilder states to avoid redundant tree rebuilding
        # Key: tuple of token_ids, Value: (builder_stack_copy, type_stack, filter_depth_stack, is_valid)
        self._builder_cache: Dict[Tuple[int, ...], Any] = {}
        self._builder_cache_size = builder_cache_size
        self._builder_cache_hits = 0
        self._builder_cache_misses = 0

        # Per-batch caches for type inference (cleared at start of each update_masks call)
        # Key: id(expr), Value: result
        self._semantic_type_cache: Dict[int, Any] = {}
        self._range_property_cache: Dict[int, str] = {}
        self._structural_equality_cache: Dict[Tuple[int, int], bool] = {}

        # Diagnostic counters for debugging mask rejections
        self._diag_total_mask_calls = 0
        self._diag_forced_exit_curriculum = 0
        self._diag_forced_exit_invalid = 0
        self._diag_forced_exit_constant = 0
        self._diag_no_valid_actions = 0
        self._diag_sink_states = 0

        # Build action list (base tokens only - no macros for memory efficiency)
        self.beg_token = [BEG_TOKEN]
        self.operators = [OperatorToken(op) for op in OPERATORS]
        self.features = [FeatureToken(feat) for feat in FEATURES]
        self.delta_times = [DeltaTimeToken(dt) for dt in DELTA_TIMES]
        self.constants = [ConstantToken(c) for c in CONSTANTS]
        self.sep_token = [SEP_TOKEN]

        # Action list: base tokens (SEP is the exit action and must be last for gfn library)
        # Order: BEG, operators, features, delta_times, constants, SEP
        self.action_list: List[Token] = (
            self.beg_token + self.operators + self.features + self.delta_times +
            self.constants + self.sep_token
        )
        self.id_to_token_map = {i: token for i, token in enumerate(self.action_list)}
        n_actions = len(self.action_list)

        # Build string-to-id cache for fast token lookup
        self._str_to_id_cache = {str(token): i for token, i in self.token_to_id_map.items()}

        s0 = torch.tensor([self.token_to_id_map[BEG_TOKEN]] + [-1] * (MAX_EXPR_LENGTH - 1), dtype=torch.long, device=device)
        sf = torch.full((MAX_EXPR_LENGTH,), self.token_to_id_map[SEP_TOKEN], dtype=torch.long, device=device)
        preprocessor = IntegerPreprocessor(output_dim=MAX_EXPR_LENGTH)

        print(f"[Env] Initialized with {n_actions} base actions")

        super().__init__(
            n_actions=n_actions,
            s0=s0,
            sf=sf,
            state_shape=(MAX_EXPR_LENGTH,),
            dummy_action=torch.tensor([-1], dtype=torch.long, device=device),
            exit_action=torch.tensor([self.token_to_id_map[SEP_TOKEN]], dtype=torch.long, device=device),
            device_str=str(device),
            preprocessor=preprocessor
        )

        # Pre-calculate indices for vectorized validation
        self._precalculate_validation_indices()

    def _precalculate_validation_indices(self):
        from alphagen.data.expression import (
            UnaryOperator, BinaryOperator, RollingOperator, PairRollingOperator,
            Gt, Ge, Lt, Le, Having, NotHaving
        )

        # Offsets in the action list
        beg_offset = len(self.beg_token)
        n_ops = len(self.operators)
        n_features = len(self.features)
        n_dts = len(self.delta_times)

        # Helper to get absolute indices
        def get_op_indices(condition):
            return [beg_offset + i for i, op in enumerate(self.operators) if condition(op.operator)]

        # Daily operators only
        from alphagen.data.expression import UnaryOperator, BinaryOperator, RollingOperator, PairRollingOperator
        self.idx_unary = get_op_indices(lambda op: issubclass(op, UnaryOperator))
        self.idx_binary = get_op_indices(lambda op: issubclass(op, BinaryOperator))
        
        # Specific indices
        from alphagen.data.expression import TsQuantile, TsRelStrength, Quantile
        self.idx_ts_quantile = get_op_indices(lambda op: op == TsQuantile)
        self.idx_ts_relstrength = get_op_indices(lambda op: op == TsRelStrength)
        self.idx_quantile = get_op_indices(lambda op: op == Quantile)

        # Rolling ops (excluding specific ones handled separately)
        self.idx_rolling_2 = get_op_indices(lambda op: issubclass(op, RollingOperator) and op.n_args() == 2)
        
        # Rolling 3: Exclude TsQuantile and TsRelStrength (handled specially)
        self.idx_rolling_3 = get_op_indices(lambda op: issubclass(op, RollingOperator) and op.n_args() == 3 
                                            and op != TsQuantile and op != TsRelStrength)
        
        self.idx_pair_rolling = get_op_indices(lambda op: issubclass(op, PairRollingOperator))
        
        # Logical operators (output BOOLEAN type)
        from alphagen.data.expression import Gt, Ge, Lt, Le
        self.idx_logical = get_op_indices(lambda op: op in (Gt, Ge, Lt, Le))
        self.logical_ops = {Gt, Ge, Lt, Le}
        
        # Having/NotHaving operators (require BOOLEAN as second arg)
        from alphagen.data.expression import Having, NotHaving
        self.idx_having = get_op_indices(lambda op: op in (Having, NotHaving))
        self.having_ops = {Having, NotHaving}

        # Binary operators excluding Having/NotHaving, logical ops, and Quantile
        self.idx_binary_non_having_non_logical = [i for i in self.idx_binary if i not in self.idx_having and i not in self.idx_logical and i not in self.idx_quantile]

        # Allowed quantiles for Quantile and TsQuantile
        self.quantile_constants = {0.25, 0.33, 0.5, 0.67, 0.75}

        # Dispersion/Correlation ops that require window > 1 (std, var, ir, corr, cov, skew, kurt, rank)
        # TsRank(x, 1) is always 0.0 (constant), so we ban it too.
        from alphagen.data.expression import TsStd, TsVar, TsIr, TsSkew, TsKurt, TsRank, TsCorr, TsCov, TsMomRank
        dispersion_classes = {TsStd, TsVar, TsIr, TsSkew, TsKurt, TsRank, TsCorr, TsCov, TsMomRank}
        self.idx_ts_momrank = get_op_indices(lambda op: op == TsMomRank)
        
        self.idx_dispersion_ops = get_op_indices(lambda op: op in dispersion_classes)

        
        # ========== SEMANTIC TYPE SYSTEM ==========
        # Define semantic categories for features - only same-type comparisons allowed
        from enum import Enum
        class SemanticType(Enum):
            PRICE = 1    # open, high, low, close, vwap
            VOLUME = 2   # volume
            RATIO = 3    # sortino_ratio, ts_mom_rank, max_dd_ratio, rel_strength_ma, drift_factor
            STATIONARY = 4 # ret, delta, rank, quantile, div (signals)
            UNKNOWN = 0  # derived/computed expressions
        
        self.SemanticType = SemanticType
        
        # Map FeatureType to SemanticType
        self.feature_semantic_types = {
            FeatureType.OPEN: SemanticType.PRICE,
            FeatureType.CLOSE: SemanticType.PRICE,
            FeatureType.HIGH: SemanticType.PRICE,
            FeatureType.LOW: SemanticType.PRICE,
            FeatureType.VWAP: SemanticType.PRICE,
            FeatureType.VOLUME: SemanticType.VOLUME,
            FeatureType.DRIFT_FACTOR: SemanticType.RATIO,
            FeatureType.AMIHUD_MEAN: SemanticType.RATIO,
            FeatureType.AMIHUD_RANGE: SemanticType.RATIO,
            # LOG features (magnitude-aware)
            FeatureType.LOG_CLOSE: SemanticType.PRICE,
            FeatureType.LOG_VOLUME: SemanticType.VOLUME,
            FeatureType.LOG_MONEY: SemanticType.VOLUME,
            # NORM features (cross-sectionally normalized)
            FeatureType.NORM_OPEN: SemanticType.PRICE,
            FeatureType.NORM_CLOSE: SemanticType.PRICE,
            FeatureType.NORM_HIGH: SemanticType.PRICE,
            FeatureType.NORM_LOW: SemanticType.PRICE,
            FeatureType.NORM_VWAP: SemanticType.PRICE,
            FeatureType.NORM_VOLUME: SemanticType.VOLUME,
        }
        
        # Unary operators that preserve semantic type (for type propagation)
        from alphagen.data.expression import SLog1p, Inv as InvOp, Abs as AbsOp, Sqrt as SqrtOp, Ret
        self.type_preserving_unary_ops = {SLog1p, InvOp, AbsOp, SqrtOp}  # These preserve the "kind" of value
        # Note: Rank outputs RATIO type always, so not in this set
        
        # Operators allowed on BOOLEAN outputs from Ge/Gt/Le/Lt
        from alphagen.data.expression import TsSum, TsMean, Mul, TsCorr, TsCov
        self.ops_allowed_on_boolean = {
            TsSum, TsMean,  # Count/proportion
            Mul,            # Conditional weighting
            TsCorr, TsCov,  # Correlation
            Having, NotHaving  # Filtering
        }
        self.idx_ops_allowed_on_boolean = get_op_indices(lambda op: op in self.ops_allowed_on_boolean)
        
        # Cross-sectional constant operators - these produce same value for all stocks on each day
        # If used as root operator, the factor has IC=0 (useless)
        from alphagen.data.expression import Corr
        self.cross_sectional_constant_ops = {Corr, Quantile}

        # Sign should not feed into these operators (degenerate outputs)
        from alphagen.data.expression import Rank
        # Removed Less from exclusion list as it was removed from operators

        self.idx_features = list(range(beg_offset + n_ops, beg_offset + n_ops + n_features))
        self.idx_delta_times = list(range(beg_offset + n_ops + n_features, beg_offset + n_ops + n_features + n_dts))
        self.idx_constants = list(range(beg_offset + n_ops + n_features + n_dts, beg_offset + n_ops + n_features + n_dts + len(self.constants)))
        
        # Exit action index (SEP_TOKEN is last - required by gfn library)
        self.idx_exit = len(self.action_list) - 1

    @property
    def token_to_id_map(self):
        # The last action is the exit action
        mapping = {token: i for i, token in enumerate(self.action_list)}
        return mapping

    def step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        next_states_tensor = states.tensor.clone()
        for i, (state_tensor, action_id_tensor) in enumerate(zip(states.tensor, actions.tensor.squeeze(-1))):
            action_id = action_id_tensor.item()
            token = self.id_to_token_map[action_id]

            if token == SEP_TOKEN:  # Exit action - transition to sink state
                next_states_tensor[i] = self.sf
            else:  # Regular token
                non_padded_len = (state_tensor != -1).sum()
                if non_padded_len < MAX_EXPR_LENGTH:
                    next_states_tensor[i, non_padded_len] = action_id
        return next_states_tensor

    def backward_step(self, states: DiscreteStates, actions: Actions) -> torch.Tensor:
        # Implement backward step if needed
        raise NotImplementedError

    def _infer_semantic_type(self, expr):
        """Recursively infer the semantic type (PRICE, VOLUME, STATIONARY) of an expression.
        Results are cached per-batch to avoid redundant recursive traversals."""
        if expr is None:
            return self.SemanticType.UNKNOWN

        # Check cache first
        expr_id = id(expr)
        if expr_id in self._semantic_type_cache:
            return self._semantic_type_cache[expr_id]

        from alphagen.data.expression import Feature, Ret, Rank, Quantile, Div, Abs, SLog1p, Sqrt, Inv, Add, Sub, Mul
        from alphagen.data.expression import TsDelta, TsRank, TsQuantile, TsMomRank

        result = self.SemanticType.UNKNOWN

        # 1. Leaf: Feature
        if hasattr(expr, '_feature'):  # Feature
            result = self.feature_semantic_types.get(expr._feature, self.SemanticType.UNKNOWN)
        # 2. Stationary Ops (Creators)
        elif hasattr(expr, '_operand') or hasattr(expr, '_lhs'):
            type_checks = type(expr)
            if type_checks in {Ret, Rank, Quantile, Div, TsDelta, TsRank, TsQuantile, TsMomRank}:
                result = self.SemanticType.STATIONARY
            # 3. Preserving Ops (Propagators)
            elif hasattr(expr, '_operand'):  # Unary/Rolling
                operand_type = self._infer_semantic_type(expr._operand)
                result = operand_type  # Propagate (e.g. Log(Price) -> Price-ish Level)
            # Binary: Add, Sub, Mul
            elif hasattr(expr, '_lhs') and hasattr(expr, '_rhs'):
                t1 = self._infer_semantic_type(expr._lhs)
                t2 = self._infer_semantic_type(expr._rhs)
                if t1 == self.SemanticType.STATIONARY or t2 == self.SemanticType.STATIONARY:
                    result = self.SemanticType.UNKNOWN
                elif t1 == t2:
                    result = t1  # Add(Price, Price) -> Price

        # Cache the result
        self._semantic_type_cache[expr_id] = result
        return result

    def _are_structurally_equal(self, e1: Expression, e2: Expression) -> bool:
        """
        Check if two expressions are structurally identical recursively.
        Avoids string serialization for performance and robustness.
        Results are cached per-batch using (id(e1), id(e2)) as key.
        """
        if e1 is e2:
            return True
        if type(e1) != type(e2):
            return False

        # Check cache
        cache_key = (id(e1), id(e2))
        if cache_key in self._structural_equality_cache:
            return self._structural_equality_cache[cache_key]
        # Also check reverse key
        reverse_key = (id(e2), id(e1))
        if reverse_key in self._structural_equality_cache:
            return self._structural_equality_cache[reverse_key]

        result = False

        # 1. Leaves
        if hasattr(e1, '_feature'):  # Feature
            result = e1._feature == e2._feature
        elif hasattr(e1, '_value'):  # Constant
            result = abs(e1._value - e2._value) < 1e-9
        # 2. Unary / Rolling (single operand)
        elif hasattr(e1, '_operand'):
            if not self._are_structurally_equal(e1._operand, e2._operand):
                result = False
            else:
                # Check extra args for Rolling/PairRolling
                if hasattr(e1, '_delta_time') and e1._delta_time != e2._delta_time:
                    result = False
                elif hasattr(e1, '_quantile') and not self._are_structurally_equal(e1._quantile, e2._quantile):
                    result = False
                else:
                    result = True
        # 3. Binary
        elif hasattr(e1, '_lhs') and hasattr(e1, '_rhs'):
            result = (self._are_structurally_equal(e1._lhs, e2._lhs) and
                      self._are_structurally_equal(e1._rhs, e2._rhs))
        # 4. Variadic (if any)
        elif hasattr(e1, '_operands'):
            if len(e1._operands) != len(e2._operands):
                result = False
            else:
                result = all(self._are_structurally_equal(o1, o2) for o1, o2 in zip(e1._operands, e2._operands))
        else:
            result = True

        # Cache the result
        self._structural_equality_cache[cache_key] = result
        return result
    def _infer_range_properties(self, expr):
        """
        Infer range properties of an expression (e.g. is it strictly positive?).
        Returns: 'POSITIVE' if strictly positive, 'ANY' otherwise.
        Results are cached per-batch to avoid redundant recursive traversals.
        """
        if expr is None:
            return 'ANY'

        # Check cache first
        expr_id = id(expr)
        if expr_id in self._range_property_cache:
            return self._range_property_cache[expr_id]

        from alphagen.data.expression import Feature, Abs, Sqrt, Add, Mul, Div, Constant

        result = 'ANY'

        # 1. Base Features
        if hasattr(expr, '_feature'):
            # Prices and Volumes are strictly positive
            if expr.is_featured:  # It's a feature token
                ft = expr._feature
                if self.feature_semantic_types.get(ft) in [self.SemanticType.PRICE, self.SemanticType.VOLUME]:
                    result = 'POSITIVE'
        # 2. Constants
        elif hasattr(expr, '_value'):
            result = 'POSITIVE' if expr._value > 0 else 'ANY'
        else:
            # 3. Operators
            t = type(expr)

            # Always Positive Ops
            if t in {Abs, Sqrt}:
                result = 'POSITIVE'
            # Propagating Ops
            elif t == Add:
                l = self._infer_range_properties(expr._lhs)
                r = self._infer_range_properties(expr._rhs)
                if l == 'POSITIVE' and r == 'POSITIVE':
                    result = 'POSITIVE'
            elif t == Mul:
                l = self._infer_range_properties(expr._lhs)
                r = self._infer_range_properties(expr._rhs)
                if l == 'POSITIVE' and r == 'POSITIVE':
                    result = 'POSITIVE'
            elif t == Div:
                l = self._infer_range_properties(expr._lhs)
                r = self._infer_range_properties(expr._rhs)
                if l == 'POSITIVE' and r == 'POSITIVE':
                    result = 'POSITIVE'

        # Cache the result
        self._range_property_cache[expr_id] = result
        return result

    def _build_state_cached(self, token_ids: Tuple[int, ...]) -> Tuple[ExpressionBuilder, List[bool], List[int], bool]:
        """
        Build ExpressionBuilder state from token_ids, using cache when possible.

        Returns:
            (builder, type_stack, filter_depth_stack, invalid_state)

        This avoids O(n²) complexity when update_masks is called repeatedly
        for incrementally longer sequences.
        """
        # Check cache first
        if token_ids in self._builder_cache:
            self._builder_cache_hits += 1
            cached = self._builder_cache[token_ids]
            # Return deep copies to avoid mutation issues
            builder = ExpressionBuilder()
            builder.stack = copy.copy(cached['stack'])
            return builder, list(cached['type_stack']), list(cached['filter_depth_stack']), cached['invalid']

        self._builder_cache_misses += 1

        # O(1) parent lookup: only check immediate parent (token_ids[:-1])
        # This is the most common case when building incrementally
        parent_key = token_ids[:-1] if len(token_ids) > 1 else None

        # Initialize from parent or scratch
        if parent_key is not None and parent_key in self._builder_cache:
            cached = self._builder_cache[parent_key]
            builder = ExpressionBuilder()
            builder.stack = copy.copy(cached['stack'])
            type_stack = list(cached['type_stack'])
            filter_depth_stack = list(cached['filter_depth_stack'])
            start_idx = len(parent_key)
            invalid_state = cached['invalid']
        else:
            builder = ExpressionBuilder()
            type_stack = []
            filter_depth_stack = []
            start_idx = 1  # Skip BEG_TOKEN
            invalid_state = False

        # Build from start_idx
        from alphagen.data.expression import RollingOperator, PairRollingOperator
        from alphagen.data.expression import TsMean, TsStd, TsMax, TsMin, TsMed, TsEMA, TsRank, TsQuantile, TsSum, TsSkew, TsKurt
        FILTER_OPS = {TsMean, TsStd, TsMax, TsMin, TsMed, TsEMA, TsRank, TsQuantile, TsSum, TsSkew, TsKurt}

        for i in range(start_idx, len(token_ids)):
            if invalid_state:
                break
            token_id = token_ids[i]
            token = self.id_to_token_map[token_id]
            try:
                builder.add_token(token)
            except InvalidExpressionException:
                invalid_state = True
                break

            # Track Type and Filter Depth
            if hasattr(token, 'operator'):
                op = token.operator

                if op in self.logical_ops:
                    if len(type_stack) >= 2:
                        type_stack.pop(); type_stack.pop()
                    type_stack.append(True)
                elif op in self.having_ops:
                    if len(type_stack) >= 2:
                        type_stack.pop(); type_stack.pop()
                    type_stack.append(False)
                elif hasattr(op, 'n_args'):
                    n_args = op.n_args()
                    input_depths = []
                    for _ in range(n_args):
                        if type_stack: type_stack.pop()
                        if filter_depth_stack: input_depths.append(filter_depth_stack.pop())

                    base_depth = max(input_depths) if input_depths else 0
                    new_depth = base_depth + 1 if op in FILTER_OPS else base_depth

                    type_stack.append(False)
                    filter_depth_stack.append(new_depth)

            elif hasattr(token, 'feature') or hasattr(token, 'value'):
                type_stack.append(False)
                filter_depth_stack.append(0)

        # Cache the result (with LRU eviction)
        if len(self._builder_cache) >= self._builder_cache_size:
            # Remove oldest entry (first key)
            oldest_key = next(iter(self._builder_cache))
            del self._builder_cache[oldest_key]

        self._builder_cache[token_ids] = {
            'stack': copy.copy(builder.stack),
            'type_stack': list(type_stack),
            'filter_depth_stack': list(filter_depth_stack),
            'invalid': invalid_state
        }

        return builder, type_stack, filter_depth_stack, invalid_state

    def update_masks(self, states: DiscreteStates):
        from alphagen.data.expression import DeltaTime
        from alphagen.data.expression import Operator
        from alphagen.data.expression import TsMean, TsStd, TsMax, TsMin, TsMed, TsEMA, TsRank, TsQuantile, TsSum, TsSkew, TsKurt
        FILTER_OPS = {TsMean, TsStd, TsMax, TsMin, TsMed, TsEMA, TsRank, TsQuantile, TsSum, TsSkew, TsKurt}

        # Clear per-batch caches for type inference functions
        # These use id(expr) as keys, which are only valid within this batch
        self._semantic_type_cache.clear()
        self._range_property_cache.clear()
        self._structural_equality_cache.clear()

        def has_logical_op(expr: Expression) -> bool:
            """Recursively detect logical operators (Gt/Ge/Lt/Le) in the tree."""
            if expr is None:
                return False
            expr_type = type(expr)
            if issubclass(expr_type, Operator) and expr_type in self.logical_ops:
                return True

            # Walk common child attributes
            for child_attr in ("_operand", "_lhs", "_rhs"):
                child = getattr(expr, child_attr, None)
                if isinstance(child, Expression) and has_logical_op(child):
                    return True

            # Handle variadic/quantile-style operands if present
            operands = getattr(expr, "_operands", None)
            if operands:
                for child in operands:
                    if isinstance(child, Expression) and has_logical_op(child):
                        return True

            quantile = getattr(expr, "_quantile", None)
            if isinstance(quantile, Expression) and has_logical_op(quantile):
                return True

            return False

        def evaluates_to_cs_constant(e):
            """Recursively check if expression evaluates to a cross-sectional constant."""
            if e is None:
                return False
            e_type = type(e)

            # 1. Base Constancy Sources
            # Quantile and Corr produce Scalar (per day) from Vector inputs
            if e_type in self.cross_sectional_constant_ops:
                return True
            # Constants are constants
            if hasattr(e, '_value'):  # Constant Token expression
                return True

            # 2. Features: Inherently Cross-Sectional (Not Constant)
            if hasattr(e, '_feature'):
                return False

            # 3. Recursive Propagation
            # Unary Ops: Constant -> Constant
            operand = getattr(e, "_operand", None)
            if isinstance(operand, Expression) and evaluates_to_cs_constant(operand):
                return True

            # Rolling Ops (with DT): Constant -> Constant
            # (e.g. Mean of constants is constant)

            # Binary Ops: Constant ONLY if BOTH children are Constant
            lhs = getattr(e, "_lhs", None)
            rhs = getattr(e, "_rhs", None)
            if isinstance(lhs, Expression) and isinstance(rhs, Expression):
                return evaluates_to_cs_constant(lhs) and evaluates_to_cs_constant(rhs)

            # Logical filter (Having): Consumes boolean mask.
            # If mask is constant AND value is constant -> Constant?
            # Usually Having maps value -> value or NaN.
            # If value is Constant, Result is Constant (or NaN).
            # If value is NotConstant, Result is NotConstant.

            return False

        batch_masks = []
        MAX_FILTER_DEPTH = 3

        for state_tensor in states.tensor:
            self._diag_total_mask_calls += 1

            if torch.all(state_tensor == self.sf):  # This is a sink state
                self._diag_sink_states += 1
                valid_actions = [False] * self.n_actions
                valid_actions[self.idx_exit] = True
                batch_masks.append(valid_actions)
                continue

            token_ids_list = [tid.item() for tid in state_tensor if tid >= 0]

            # Force Curriculum Length Constraint
            if hasattr(self, 'max_len') and len(token_ids_list) >= self.max_len - 1:
                self._diag_forced_exit_curriculum += 1
                valid_actions = [False] * self.n_actions
                valid_actions[self.idx_exit] = True
                batch_masks.append(valid_actions)
                continue

            # Use cached builder state to avoid O(n²) rebuilding
            token_ids = tuple(token_ids_list)
            builder, type_stack, filter_depth_stack, invalid_state = self._build_state_cached(token_ids)

            if invalid_state:
                self._diag_forced_exit_invalid += 1
                valid_actions = [False] * self.n_actions
                valid_actions[self.idx_exit] = True
                batch_masks.append(valid_actions)
                continue
            
            # Initialize mask with all False
            valid_actions = [False] * self.n_actions
            
            # Check stack properties once
            stack = builder.stack
            stack_len = len(stack)

            # Lookahead Constraints
            # To finish safely (Stop at S=1), we need Remaining >= Stack.
            # Push: S->S+1, R->R-1. Need R-1 >= S+1 => R >= S+2
            # Unary: S->S, R->R-1. Need R-1 >= S => R >= S+1
            
            can_push_feature = True
            can_unary = True
            
            if hasattr(self, 'max_len'):
                rem_steps = self.max_len - 1 - len(token_ids)
                if rem_steps < stack_len + 2:
                    can_push_feature = False
                if rem_steps < stack_len + 1:
                    can_unary = False

            # Check stack properties once

            current_filter_depth = filter_depth_stack[-1] if filter_depth_stack else 0

            # Helper booleans
            top_is_featured = stack_len >= 1 and stack[-1].is_featured
            top_is_dt = stack_len >= 1 and isinstance(stack[-1], DeltaTime)
            
            # Check if top of stack is BOOLEAN type
            top_is_boolean = len(type_stack) >= 1 and type_stack[-1] == True

            second_is_featured = stack_len >= 2 and stack[-2].is_featured
            second_is_dt = stack_len >= 2 and isinstance(stack[-2], DeltaTime)
            second_is_boolean = len(type_stack) >= 2 and type_stack[-2] == True

            third_is_featured = stack_len >= 3 and stack[-3].is_featured

            # 1. Unary Operators: stack[-1].is_featured
            if top_is_featured and can_unary:
                # Filter Nesting Check (Part 1: Unary/Rolling filters)

                is_filter_saturated = current_filter_depth >= MAX_FILTER_DEPTH
                
                if top_is_boolean:
                    # Boolean outputs: only Rank makes sense
                    for idx in self.idx_unary:
                        if idx in self.idx_ops_allowed_on_boolean:
                            valid_actions[idx] = True
                else:
                    for idx in self.idx_unary:
                        op_class = self.action_list[idx].operator
                        
                        # Nesting Guard
                        if is_filter_saturated and op_class in FILTER_OPS:
                            continue

                        # SEMANTIC GUARD: Block Abs on Positive Values
                        from alphagen.data.expression import Abs as AbsOp, Sqrt as SqrtOp
                        if op_class == AbsOp:
                            range_prop = self._infer_range_properties(stack[-1])
                            if range_prop == 'POSITIVE':
                                continue # Abs(Pos) is redundant
                        
                        # Unary Chain Redundancy (Direct type checks)
                        # Avoid Abs(Abs)
                        top_type = type(stack[-1])
                        
                        if op_class == AbsOp and top_type == AbsOp: continue
                        # Log/Exp/Sign not in action space, so no need to check
                             
                        # SEMANTIC GUARD: Block Abs on Raw Features (Always Positive) - redundant with above but explicit for safety
                        from alphagen.data.expression import Feature
                        is_raw_feature = isinstance(stack[-1], Feature)
                        if op_class == AbsOp and is_raw_feature:
                            # Double check it's a price/volume feature
                            ft_type = self.feature_semantic_types.get(stack[-1]._feature)
                            if ft_type in [self.SemanticType.PRICE, self.SemanticType.VOLUME]:
                                continue
                            
                        # SEMANTIC GUARD: Block Abs(TsQuantile) - squashes signal
                        from alphagen.data.expression import TsQuantile
                        if op_class == AbsOp and isinstance(stack[-1], TsQuantile):
                            continue
                            
                        valid_actions[idx] = True

            # 2. Binary Operators (non-Having, non-Logical): top and second are featured, neither is DT
            if top_is_featured and second_is_featured and not top_is_dt and not second_is_dt:
                # Filter Nesting Check (Part 2: Binary filters?)
                # Usually binary ops aren't filters (Add, Sub), but TsCorr/TsCov are.
                is_filter_saturated = (current_filter_depth >= MAX_FILTER_DEPTH) 
                # Note: Binary ops merge branches. Depth is max(d1, d2). 
                # If either branch is deep, we might restrict.
                # Here we just check current stack top (which is one operand). 
                # Ideally check both from filter_depth_stack but accessing -1, -2 is easy.
                # Actually, if we apply a NEW binary filter (TsCorr), we check its result depth checks inputs.
                # Since we already computed depth in the stack loop, we just check if new_op would exceed.
                # But here we just want to know if we CAN apply it.
                
                # Check for Quantile Constant semantics
                top_is_constant = hasattr(stack[-1], '_value')
                top_val = stack[-1]._value if top_is_constant else None
                
                is_quantile_constant = top_is_constant and any(abs(top_val - q) < 1e-6 for q in self.quantile_constants)

                if is_quantile_constant:
                    # Only allow Quantile operator (cross-sectional)
                    for idx in self.idx_quantile: valid_actions[idx] = True
                else:
                    # Allow generic binary ops (excluding Quantile)
                    for idx in self.idx_binary_non_having_non_logical:
                        # SEMANTIC GUARD: Block Add(Feature, Feature) and Mul(Price, Price)
                        op_class = self.action_list[idx].operator
                        
                        # Nesting Guard
                        if is_filter_saturated and op_class in FILTER_OPS:
                            continue
                        
                        from alphagen.data.expression import Add, Mul, Sub, Div, Feature
                        
                        top_is_raw = isinstance(stack[-1], Feature)
                        second_is_raw = isinstance(stack[-2], Feature)
                        
                        if top_is_raw and second_is_raw:
                            # 1. Block Add(Feature, Feature) - usually meaningless
                            if op_class == Add:
                                continue
                            # 2. Block Mul(Price, Price)
                            if op_class == Mul:
                                type1 = self.feature_semantic_types.get(stack[-1]._feature, self.SemanticType.UNKNOWN)
                                type2 = self.feature_semantic_types.get(stack[-2]._feature, self.SemanticType.UNKNOWN)
                                if type1 == self.SemanticType.PRICE and type2 == self.SemanticType.PRICE:
                                    continue
                                    
                        # 3. Block Sub(X, X) and Div(X, X) - Redundant
                        # Check STRUCTURAL equality
                        if self._are_structurally_equal(stack[-1], stack[-2]):
                            if op_class == Sub:
                                continue # X - X = 0
                            if op_class == Div:
                                continue # X / X = 1 (or NaN)
                                
                        valid_actions[idx] = True

            # 2a. Logical Operators
            if top_is_featured and second_is_featured and not top_is_dt and not second_is_dt:
                if not top_is_boolean and not second_is_boolean:
                    # Avoid comparing with Quantile Constants
                    top_is_constant = hasattr(stack[-1], '_value')
                    top_val = stack[-1]._value if top_is_constant else None
                    if top_is_constant and any(abs(top_val - q) < 1e-6 for q in self.quantile_constants):
                        pass 
                    else:
                        type1 = self._infer_semantic_type(stack[-1])
                        type2 = self._infer_semantic_type(stack[-2])
                        
                        # Block Comparison of Identical Expressions: X > X is always False
                        if self._are_structurally_equal(stack[-1], stack[-2]):
                           idx_logical_set = set(self.idx_logical)
                           pass # Allow none
                        else:
                            # Allow comparison only if compatible
                            types_compatible = (
                                type1 == type2 or 
                                type1 == self.SemanticType.UNKNOWN or 
                                type2 == self.SemanticType.UNKNOWN
                            )
                            
                            if types_compatible:
                                for idx in self.idx_logical: valid_actions[idx] = True
            
            # 2b. Having/NotHaving: top must be BOOLEAN and featured (not DT), second must be CONTINUOUS and featured
            # Also ensure type_stack and builder.stack are in sync by checking top_is_featured
            if top_is_boolean and top_is_featured and not top_is_dt and second_is_featured and not second_is_boolean and not second_is_dt:
                for idx in self.idx_having: valid_actions[idx] = True

            # 3. Rolling Operators (2 args): top is DT, second is Featured
            if top_is_dt and second_is_featured and can_unary:
                # Check if DT is 1
                dt_val = stack[-1]._delta_time
                for idx in self.idx_rolling_2:
                    op_class = self.action_list[idx].operator
                    # Dispersion ops require minimum window of 10 to avoid constant outputs
                    if dt_val < 10 and idx in self.idx_dispersion_ops:
                        continue
                    # TsMomRank requires window >=20
                    if op_class == TsMomRank and dt_val < 20:
                        continue
                    # Still ban window=1 for all rolling ops (undefined/trivial)
                    if dt_val == 1:
                        continue
                    valid_actions[idx] = True

            # 4. Rolling Operators (3 args): top is DT, second not DT, third Featured
            # Only generic 3-arg ops (none left if TsQuantile and TsRelStrength are removed from this group)
            if top_is_dt and not second_is_dt and third_is_featured:
                # If there are any other 3-arg rolling ops
                dt_val = stack[-1]._delta_time
                for idx in self.idx_rolling_3: 
                    op_class = self.action_list[idx].operator
                    
                    # Dispersion ops require minimum window of 10
                    if dt_val < 10 and idx in self.idx_dispersion_ops:
                        continue
                    if dt_val == 1:
                        continue
                        
                    # SEMANTIC GUARD: TsCorr/TsCov inputs must be Stationary/Ratio (Signal)
                    # Block TsCorr(Level, Level) like TsCorr(Price, Vol) or TsCorr(Price, Price)
                    from alphagen.data.expression import TsCorr, TsCov
                    if op_class in {TsCorr, TsCov}:
                        # stack[-2] is second operand, stack[-3] is first operand
                        t1 = self._infer_semantic_type(stack[-3])
                        t2 = self._infer_semantic_type(stack[-2])
                        is_t1_level = (t1 == self.SemanticType.PRICE or t1 == self.SemanticType.VOLUME)
                        is_t2_level = (t2 == self.SemanticType.PRICE or t2 == self.SemanticType.VOLUME)
                        
                        # If BOTH are non-stationary Levels, block.
                        if is_t1_level and is_t2_level:
                            continue

                    valid_actions[idx] = True

                # TsQuantile specific check: Quantile arg (stack[-2]) must be valid
                if len(self.idx_ts_quantile) > 0:
                    quantile_arg = stack[-2]
                    is_valid_quantile = False
                    if hasattr(quantile_arg, '_value'): # It's a Constant expression node
                        val = quantile_arg._value
                        # Only allow specific quantiles
                        if any(abs(val - q) < 1e-6 for q in self.quantile_constants):
                            is_valid_quantile = True
                    
                    if is_valid_quantile:
                        for idx in self.idx_ts_quantile: valid_actions[idx] = True

            # 5. TsRelStrength (Special 3-ary): [Operand, Constant(fast), DeltaTime(slow)] -> TsRelStrength
            # Requires: top=DeltaTime(slow), second=Constant(fast), third=Featured
            # FIX: tree.py expects stack[-2] to be Constant, not DeltaTime!
            second_is_constant = stack_len >= 2 and hasattr(stack[-2], '_value')
            if top_is_dt and second_is_constant and third_is_featured:
                slow_dt = stack[-1]._delta_time
                fast_val = stack[-2]._value  # This is a Constant value, not DeltaTime

                # Structural rule: 0 < fast < slow (matching tree.py validation)
                if fast_val > 0 and fast_val < slow_dt:
                    for idx in self.idx_ts_relstrength:
                        valid_actions[idx] = True

            # 6. Pair Rolling (TsCorr, TsCov): top is DT, second Featured, third Featured
            if top_is_dt and second_is_featured and third_is_featured:
                dt_val = stack[-1]._delta_time
                for idx in self.idx_pair_rolling:
                    # Dispersion ops require minimum window of 10
                    if dt_val < 10 and idx in self.idx_dispersion_ops:
                        continue
                    # Ban window=1 for all rolling ops
                    if dt_val == 1:
                        continue
                    # FIX: Actually enable the action! (was missing)
                    valid_actions[idx] = True
            
            # Lookahead Constraints moved to top
            
            if not top_is_dt and can_push_feature:
                for idx in self.idx_features: valid_actions[idx] = True

            # 7. DeltaTime: valid if top is Featured OR (len>=2 and !top.featured and second.featured)
            dt_valid = False
            if top_is_featured:
                dt_valid = True
            elif stack_len >= 2 and not top_is_featured and second_is_featured:
                dt_valid = True

            if dt_valid:
                for idx in self.idx_delta_times: valid_actions[idx] = True

            # 8. Constants: valid if len==0 or top.featured
            if stack_len == 0 or top_is_featured:
                if can_push_feature: # Re-use the same flag as Features
                    for idx in self.idx_constants: valid_actions[idx] = True

            # Exit action
            # Prevent exit if expression is invalid:
            # 1. Root/any operator is a logical op (Gt, Ge, Lt, Le) - boolean output not useful as factor
            # 2. Expression contains cross-sectional constant (Corr, Quantile) anywhere - IC=0
            #    because any unary/rolling op on constant still produces constant
            expr_is_invalid = False
            expr_is_constant = False
            if stack_len == 1:
                try:
                    expr = builder.get_tree()
                    expr_is_constant = evaluates_to_cs_constant(expr)
                    expr_is_invalid = has_logical_op(expr) or expr_is_constant
                except InvalidExpressionException:
                    pass
            
            if len(token_ids) < MAX_EXPR_LENGTH:
                if builder.is_valid() and not expr_is_invalid:
                    valid_actions[self.idx_exit] = True
            else:
                # Still prevent invalid exit, even at max length
                if not expr_is_invalid:
                    valid_actions[self.idx_exit] = True

            # If the current tree is a cross-sectional constant, force immediate exit.
            # (Previously this forced backtrack; removing backtrack avoids cyclic trajectories.)
            if expr_is_constant:
                self._diag_forced_exit_constant += 1
                valid_actions = [False] * self.n_actions
                valid_actions[self.idx_exit] = True

            # Apply random mask dropout based on expression length
            # Longer expressions get higher dropout probability
            expr_length = len(token_ids)
            length_based_dropout_prob = self.mask_dropout_prob * (expr_length / MAX_EXPR_LENGTH)

            # Collect non-exit actions that are currently valid
            special_actions = {self.idx_exit}
            true_indices = [i for i in range(len(valid_actions)) if valid_actions[i] and i not in special_actions]

            # Only apply dropout if exit action is valid (terminal-eligible state)
            # This matches upstream behavior
            if valid_actions[self.idx_exit] == True and true_indices:
                if np.random.rand() < length_based_dropout_prob:
                    # Mask all non-exit actions
                    for idx in true_indices:
                        valid_actions[idx] = False

            # Safety guard: ensure at least one action is valid
            # This prevents all-False masks that cause -inf logits
            if not any(valid_actions):
                self._diag_no_valid_actions += 1
                valid_actions[self.idx_exit] = True

            batch_masks.append(valid_actions)

        states.forward_masks = torch.tensor(batch_masks, dtype=torch.bool, device=self.device)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic statistics for debugging performance issues."""
        total = max(self._diag_total_mask_calls, 1)
        return {
            'total_mask_calls': self._diag_total_mask_calls,
            'sink_states': self._diag_sink_states,
            'sink_states_pct': 100.0 * self._diag_sink_states / total,
            'forced_exit_curriculum': self._diag_forced_exit_curriculum,
            'forced_exit_curriculum_pct': 100.0 * self._diag_forced_exit_curriculum / total,
            'forced_exit_invalid': self._diag_forced_exit_invalid,
            'forced_exit_invalid_pct': 100.0 * self._diag_forced_exit_invalid / total,
            'forced_exit_constant': self._diag_forced_exit_constant,
            'forced_exit_constant_pct': 100.0 * self._diag_forced_exit_constant / total,
            'no_valid_actions': self._diag_no_valid_actions,
            'no_valid_actions_pct': 100.0 * self._diag_no_valid_actions / total,
            'builder_cache_hits': self._builder_cache_hits,
            'builder_cache_misses': self._builder_cache_misses,
            'builder_cache_hit_rate': 100.0 * self._builder_cache_hits / max(self._builder_cache_hits + self._builder_cache_misses, 1),
        }

    def reset_diagnostics(self):
        """Reset all diagnostic counters."""
        self._diag_total_mask_calls = 0
        self._diag_forced_exit_curriculum = 0
        self._diag_forced_exit_invalid = 0
        self._diag_forced_exit_constant = 0
        self._diag_no_valid_actions = 0
        self._diag_sink_states = 0
        self._builder_cache_hits = 0
        self._builder_cache_misses = 0

    def reward(self, final_states: DiscreteStates, return_components: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        rewards = []
        components_list = []
        # Profiling counters
        import time
        encoder_time = 0.0
        pool_time = 0.0
        
        for state_tensor in final_states.tensor:
            builder = ExpressionBuilder()
            token_ids = [tid.item() for tid in state_tensor if tid >= 0]
            
            # Reconstruct the expression for reward calculation
            history: List[Token] = []
            invalid_state = False
            
            for token_id in token_ids[1:]:
                token = self.id_to_token_map[token_id]
                
                history.append(token)
                try:
                    builder.add_token(token)
                except InvalidExpressionException:
                    invalid_state = True
                    break

            reward = 0.0
            env_reward = 0.0
            struct_reward = 0.0
            nov_reward = 0.0
            ssl_reward = 0.0
            log_r_tb = -10.0
            
            if invalid_state:
                reward = 1e-6
            else:
                try:
                    expr = builder.get_tree()
                    
                    # Compute embedding only for this valid expression
                    embedding = None
                    if self.encoder is not None:
                        t0 = time.time()
                        with torch.no_grad():
                            # Add batch dimension for single state
                            single_state = state_tensor.unsqueeze(0)
                            embedding = self.encoder(single_state).squeeze(0)
                        encoder_time += (time.time() - t0)
                    
                    t1 = time.time()
                    # Updated to unpack 5 components from refactored AlphaPool
                    env_reward, struct_reward, nov_reward, ssl_reward, log_r_tb = self.pool.try_new_expr_with_ssl(expr, embedding)
                    pool_time += (time.time() - t1)
                    
                    # Total Reward = Environment (Performance) + Weighted SSL + Weighted Novelty + Structure
                    # We treat this sum as log_R(x)
                    log_reward_val = env_reward + self.ssl_weight * ssl_reward + self.nov_weight * nov_reward + struct_reward
                    reward = math.exp(log_reward_val)
                except (OutOfDataRangeError, InvalidExpressionException):
                    # Invalid intraday structure or out-of-range data
                    reward = 1e-6
                    log_r_tb = -10.0 # Low value
                except Exception:
                    reward = 1e-6
                    log_r_tb = -10.0
            
            rewards.append(reward)  # Must be positive
            
            if return_components:
                components_list.append({
                    'env_reward': env_reward,
                    'struct_reward': struct_reward,
                    'nov_reward': nov_reward,
                    'ssl_reward': ssl_reward,
                    'log_R_tb': log_r_tb
                })
        
        # Store timing stats as class attributes for external access
        if not hasattr(self, 'reward_encoder_time'):
            self.reward_encoder_time = 0.0
            self.reward_pool_time = 0.0
            self.reward_calls = 0
        self.reward_encoder_time += encoder_time
        self.reward_pool_time += pool_time
        self.reward_calls += 1
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        
        if return_components:
            collated = {}
            if components_list:
                for k in components_list[0].keys():
                    collated[k] = torch.tensor([c.get(k, 0.0) for c in components_list], dtype=torch.float, device=self.device)
            return rewards_tensor, collated
            
        return rewards_tensor
