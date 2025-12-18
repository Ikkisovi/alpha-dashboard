from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.data.expression import (
    Expression,
    OutOfDataRangeError,
    Log,
    Less,
    Quantile,
    Rank,
    Sign,
    TsQuantile,
)
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen_qlib.stock_data import StockData
from .reward_utils import RewardUtil


class ExplorationPool:
    """
    A lightweight pool for the 'Structure-Only' phase.
    Stores 'mediocre but valid' factors (IC > 0.01) to serve as a 
    diversity baseline (stepping stones), preventing the 'nothing passes' issue.
    """
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        # We store expression strings to deduplicate and for debugging
        self.exprs: List[str] = []
        # We store values for correlation checks
        self.values: List[Tensor] = []
        # Score = IC * sqrt(len), used for replacement policy
        self.scores: List[float] = []

    def try_add(self, expr: Expression, value: Tensor, ic: float) -> None:
        # Basic filter: IC > 0.01 must be met before calling this
        expr_str = str(expr)
        
        # 1. Deduplication
        if expr_str in self.exprs:
            return

        # 2. Length check (Len >= 5 to avoid trivial T(x))
        # Rough proxy: count tokens or just string length? implementation_plan said len>=5
        # Let's use node counting if possible, or just string length as proxy
        # Expression doesn't easily give node count without traversal. 
        # Let's count '(' + '$' as a rough proxy like AlphaPoolGFN does.
        node_count = expr_str.count('(') + expr_str.count('$')
        if node_count < 5:
            return

        # 3. Score Calculation: Reward complexity + validity
        score = ic * math.sqrt(node_count)
        
        # 4. Admission
        if len(self.exprs) < self.capacity:
            self.exprs.append(expr_str)
            self.values.append(value)
            self.scores.append(score)
        else:
            # Replacement: Evict lowest score
            min_score = min(self.scores)
            if score > min_score:
                idx = self.scores.index(min_score)
                self.exprs[idx] = expr_str
                self.values[idx] = value
                self.scores[idx] = score

    def max_correlation(self, value: Tensor) -> float:
        """Calculate max correlation of `value` against all factors in this pool."""
        if not self.values:
            return 0.0
        
        max_corr = 0.0
        # We can optimize this with batch ops later if needed, but loop is fine for 50 items
        for other_val in self.values:
            # batch_pearsonr returns tensor, we take mean absolute?
            # Usually we care about absolute correlation for diversity
            c = batch_pearsonr(value, other_val).mean().abs().item()
            if c > max_corr:
                max_corr = c
        return max_corr


class AlphaPoolGFN(AlphaPool):
    def __init__(
        self,
        capacity: int,
        stock_data: StockData,
        target: Expression,
        ic_mut_threshold: float = 0.3,
        use_r_squared: bool = False,
        r_squared_threshold: float = 0.5,
        ssl_k: int = 3,
        ssl_tau: float = 0.1,
        test_data: Optional[StockData] = None,
        long_short_bins: int = 10,
        rank_ic_weight: float = 0.5,
        monotonic_weight: float = 0.4,
        directional_weight: float = 0.3,
        spread_weight: float = 0.2,
        complexity_weight: float = 0.1,
        structure_only_reward: bool = False,
        structure_only_disable_progress: float = 1.0,
        structure_only_full_guard_len: int = 20,
        min_train_ic: float = 0.02,  # Hard gate: reject below this
        min_test_ic: float = 0.0,
        min_rank_ic: float = 0.03,    # Hard gate: reject below this
        min_icir: float = 0.2,        # Soft gate: ICIR has larger units
        min_ic_hit_ratio: float = 0.60,
        min_excess_return_mean: float = 0.0006,
        min_monotonicity: float = 0.1,
        min_complexity_score: float = 0.1,
        min_expression_nodes: int = 1,
        spread_scale: float = 0.02,
        directional_scale: float = 0.02,
        target_complexity: int = 12,
        train_ic_t_threshold: float = 0.0,
        test_ic_t_threshold: float = 0.0,
        enforce_sign_consistency: bool = True,
        sign_consistency_epsilon: float = 0.005,
        allow_sign_flip: bool = False,
        constraint_penalty_scale: float = 0.1,
        icir_weight: float = 0.0,
        sortino_weight: float = 0.0,
        # Seasonality and Consistency parameters
        seasonality_ic_threshold: float = 0.02,      # |IC| threshold for seasonality counting
        consistency_ic_threshold: float = 0.01,       # IC threshold for consistency counting
        min_months_for_seasonality: int = 6,          # Minimum months to evaluate seasonality
        min_periods_for_consistency: int = 6,         # Minimum periods to evaluate consistency
        seasonality_weight: float = 0.3,              # Weight for seasonality bonus in reward
        consistency_weight: float = 0.3,              # Weight for consistency bonus in reward
    ):
        # CRITICAL: Save target expression BEFORE super().__init__ evaluates it
        # This is needed for _load_raw_target_returns() to work
        from alphagen.data.expression import Expression
        if isinstance(target, Expression):
            self.target_expr = target
        else:
            self.target_expr = None  # target is already a Tensor
        
        super().__init__(capacity, stock_data, target) # Original super().__init__
        # The provided edit's super().__init__ seems to be for AlphaPoolBase, not AlphaPool.
        # Keeping the original AlphaPool super().__init__ and adding the new parameters as instance attributes.
        # If the intention was to change the base class or its __init__ signature, that would be a larger change.

        self.ic_mut_threshold = ic_mut_threshold
        self.use_r_squared = use_r_squared
        self.r_squared_threshold = max(0.0, min(1.0, r_squared_threshold))
        self.ssl_k = ssl_k  # K for k-nearest neighbors
        self.ssl_tau = ssl_tau  # Temperature parameter for similarity weight
        self.test_data = test_data
        self.test_target = None
        if self.test_data is not None:
            try:
                self.test_target = self._normalize_by_day(self.target_expr.evaluate(self.test_data))
            except OutOfDataRangeError:
                self.test_target = None
        self.long_short_bins = long_short_bins
        self.rank_ic_weight = rank_ic_weight
        self.monotonic_weight = monotonic_weight
        self.directional_weight = directional_weight
        self.spread_weight = spread_weight
        self.complexity_weight = complexity_weight
        self.structure_only_reward = structure_only_reward
        self.structure_only_disable_progress = structure_only_disable_progress
        self.structure_only_full_guard_len = structure_only_full_guard_len
        self.structure_reward_ic_floor = 0.01
        self.structure_reward_lengths = [6, 10, 15, 20, 25]
        self.min_train_ic = min_train_ic
        self.min_test_ic = min_test_ic
        self.min_rank_ic = min_rank_ic
        self.min_icir = min_icir # New parameter
        self.min_ic_hit_ratio = min_ic_hit_ratio # New parameter
        self.min_excess_return_mean = min_excess_return_mean # New parameter
        self.min_monotonicity = min_monotonicity
        self.min_complexity_score = min_complexity_score
        self.min_expression_nodes = min_expression_nodes
        self.spread_scale = spread_scale
        self.directional_scale = directional_scale
        self.target_complexity_nodes = max(1, target_complexity)
        self.train_ic_t_threshold = max(0.0, train_ic_t_threshold)
        self.test_ic_t_threshold = max(0.0, test_ic_t_threshold)
        self.enforce_sign_consistency = enforce_sign_consistency
        self.sign_consistency_epsilon = max(0.0, sign_consistency_epsilon)
        self.allow_sign_flip = allow_sign_flip # New parameter
        self.constraint_penalty_scale = max(1e-4, constraint_penalty_scale)
        self.icir_weight = max(0.0, icir_weight)
        self.sortino_weight = max(0.0, sortino_weight)
        # Seasonality and Consistency parameters
        self.seasonality_ic_threshold = seasonality_ic_threshold
        self.consistency_ic_threshold = consistency_ic_threshold
        self.min_months_for_seasonality = min_months_for_seasonality
        self.min_periods_for_consistency = min_periods_for_consistency
        self.seasonality_weight = max(0.0, seasonality_weight)
        self.consistency_weight = max(0.0, consistency_weight)
        # Initialize embeddings storage with the same structure as other factor properties
        self.embeddings: List[Optional[Tensor]] = [None for _ in range(capacity + 1)]
        # Metadata for exporting factor performance
        self.factor_notes: List[Optional[Dict]] = [None for _ in range(capacity + 1)]
        self.best_insample_weights: List[float] = []
        self.data_start = getattr(stock_data, "_start_time", None)
        self.data_end = getattr(stock_data, "_end_time", None)
        self.recent_reward_stats = []  # Buffer for TensorBoard logging
        # Cache raw target returns for equal-weight baseline checks
        self.raw_target_returns: Optional[Tensor] = self._load_raw_target_returns()
        self.baseline_returns: Optional[Tensor] = self._compute_baseline_returns(self.raw_target_returns)
        self.raw_test_returns: Optional[Tensor] = self._load_raw_target_returns(self.test_data) if self.test_data is not None else None
        self.covariance: Optional[np.ndarray] = None
        self.diversity: Optional[float] = None
        self.verbose_sources = {"resume", "de_tune", "tuned_pool", "gfn", "seed", "seed_resume", "seed_resume_having", "having_expand"}
        # Debug counters for admission bottlenecks
        self.debug_eval = True
        self.debug_every = 1000
        self.debug_stats = defaultdict(int)
        self.debug_total = 0
        # Rule 8: Progressive reward schedule
        self.training_progress = 0.0  # 0.0 = start, 1.0 = end
        # Avoid noisy duplicate skip logs
        self._last_skip_log: Optional[Tuple[str, str, str]] = None
        # Cache to prevent duplicate expression evaluations within same step
        self._expr_cache: Dict[str, Tuple[float, float]] = {}
        # Exploration Pool for Structure-Only phase (stepping stone) - only active if structure_only_reward is True
        if structure_only_reward:
            self.exploration_pool = ExplorationPool(capacity=50)
        else:
            self.exploration_pool = None
        # Curriculum scheduler reference (set externally by train_gfn.py)
        self.curriculum = None
        
        # Sanity Check: Verify data and target are valid
        if self.target is not None:
             print(f"[Pool Debug] Target Shape: {self.target.shape}")
             try:
                 nan_count = torch.isnan(self.target).sum().item()
                 print(f"[Pool Debug] Target NaNs: {nan_count} / {self.target.numel()}")
                 
                 # Check basic feature IC (Close Price vs Target)
                 from alphagen.data.expression import Feature, FeatureType
                 # Note: FeatureType enum might need to be imported or accessed differently depending on setup
                 # We use Feature(FeatureType.CLOSE) assuming enum is available
                 from alphagen_qlib.stock_data import FeatureType as FT
                 close_expr = Feature(FT.CLOSE)
                 close_val = self._normalize_by_day(close_expr.evaluate(self.data))
                 
                 # Align shapes if needed (though init should match)
                 t_align = self.target
                 v_align = close_val
                 if v_align.shape[0] != t_align.shape[0]:
                     min_len = min(v_align.shape[0], t_align.shape[0])
                     v_align = v_align[-min_len:]
                     t_align = t_align[-min_len:]
                     
                 from alphagen.utils.correlation import batch_pearsonr
                 ic_val = batch_pearsonr(v_align, t_align).mean().item()
                 print(f"[Pool Debug] Sanity Check - Close Price IC: {ic_val:.4f}")
             except Exception as e:
                 print(f"[Pool Debug] Sanity Check Failed: {e}")
    
    def set_training_progress(self, progress: float):
        """Set training progress for Rule 8 progressive schedule. 0.0 = start, 1.0 = end."""
        self.training_progress = max(0.0, min(1.0, progress))

    def _safe_eval_and_normalize(
        self, expr: Expression, data: StockData
    ) -> Tuple[Optional[Tensor], Optional[str]]:
        """
        Evaluate an expression and normalize it by day, returning an error tag instead of raising.
        """
        try:
            return self._normalize_by_day(expr.evaluate(data)), None
        except OutOfDataRangeError:
            return None, "out_of_range"
        except (ValueError, NotImplementedError) as exc:
            return None, str(exc)

    def _log_skip(self, expr: Expression, source: str, reason: str) -> None:
        if source not in self.verbose_sources:
            return
        key = (source, str(expr), reason)
        if self._last_skip_log == key:
            return
        self._last_skip_log = key
        print(f"[Pool Skip:{source}] {expr} -- {reason}")



    def get_avg_ic(self) -> float:
        """Get average absolute IC of factors in the pool."""
        if self.size == 0:
            return 0.0
        # Return mean of absolute ICs (magnitude of signal)
        return float(np.mean(np.abs(self.single_ics[:self.size])))

    def try_new_expr(
        self,
        expr: Expression,
        embedding: Optional[Tensor] = None,
        source: str = "gfn",
        count_eval: bool = True,
    ) -> Tuple[Dict[str, float], bool]:
        """
        Evaluate expression, compute rewards via RewardUtil, and decide admission to pool.
        Returns: (reward_dict, is_admitted)
        """
        # Early deduplication: cache recent expr evaluations to avoid duplicate work
        expr_key = str(expr)
        if hasattr(self, '_expr_cache') and expr_key in self._expr_cache:
            # Rehydrate from cache (cached result is Dict, bool)
            return self._expr_cache[expr_key]
        
        def cache_and_return(reward_dict: Dict[str, float], admitted: bool) -> Tuple[Dict[str, float], bool]:
            """Store result in cache, notify curriculum, and return it."""
            if len(self._expr_cache) > 10000:
                self._expr_cache.clear()
            self._expr_cache[expr_key] = (reward_dict, admitted)
            # Notify curriculum scheduler of environment reward (for progress)
            if hasattr(self, 'curriculum') and self.curriculum is not None:
                self.curriculum.notify_reward(reward_dict.get('r_env', 0.0))
            return reward_dict, admitted
        
        def mark(reason: str) -> None:
            if not self.debug_eval: return
            self.debug_stats[reason] += 1
            if count_eval:
                self.debug_total += 1
                if self.debug_total % self.debug_every == 0:
                    summary = ", ".join(f"{k}:{v}" for k, v in sorted(self.debug_stats.items()))
                    print(f"[Pool Debug] evals={self.debug_total} :: {summary}", flush=True)

        expr_str = str(expr)
        
        # 1. Duplicate Check (Pool)
        if expr_str in {str(e) for e in self.exprs[:self.size] if e is not None}:
            mark("duplicate")
            # Return low reward
            return cache_and_return({'r_total': -1.0, 'r_env': -1.0, 'log_R_tb': -4.0}, False)
        
        # 2. Grammar/Simple Validation (Length count)
        token_count = expr_str.count('$') + expr_str.count('(')
        if token_count < self.min_expression_nodes:
            mark(f"too_simple")
            return cache_and_return({'r_total': -1.0, 'r_env': -1.0, 'log_R_tb': -4.0}, False)
        
        # 3. Invalid Sign Combo (if used)
        sign_reason = self._invalid_sign_combo(expr)
        if sign_reason:
            mark("invalid_sign_combo")
            return cache_and_return({'r_total': -1.0, 'r_env': -1.0, 'log_R_tb': -4.0}, False)
            
        # 4. Evaluation
        value, eval_err = self._safe_eval_and_normalize(expr, self.data)
        if value is None:
            mark(f"eval_fail")
            # High penalty for failure
            return cache_and_return({'r_total': -2.0, 'r_env': -2.0, 'log_R_tb': -8.0}, False)

        # 5. Metric Calculation
        # Align target
        target = self.target
        if value.shape[0] != target.shape[0]:
            min_len = min(value.shape[0], target.shape[0])
            value = value[-min_len:]
            target = target[-min_len:]
            
        # IC
        ic_series = batch_pearsonr(value, target)
        train_ic_mean = ic_series.mean().item()
        
        # Canonical Sign
        canonical_sign = 1.0
        if train_ic_mean < 0:
            canonical_sign = -1.0
            value = -value
            ic_series = -ic_series
            train_ic_mean = -train_ic_mean
            from alphagen.data.expression import Mul, Constant
            expr = Mul(Constant(-1.0), expr)
            
        train_ic_std = ic_series.std().item()
        icir = train_ic_mean / (train_ic_std + 1e-6)
        
        # Excess Return & Sortino
        excess_mean, excess_sortino, _ = self._compute_excess_return_mean(value)
        
        metrics = {
            'ic': train_ic_mean,
            'icir': icir,
            'excess_return': excess_mean if excess_mean is not None else 0.0,
            'sortino': excess_sortino if excess_sortino is not None else 0.0
        }
        
        # 6. Reward Calculation (via RewardUtil)
        # Compute avg sortino of current pool
        pool_avg_sortino = 0.0
        sortino_sum = 0.0
        count = 0
        for i in range(self.size):
            note = self.factor_notes[i]
            if note and 'sortino' in note:
                sortino_sum += note['sortino']
                count += 1
        if count > 0:
            pool_avg_sortino = sortino_sum / count
            
        reward_dict = RewardUtil.compute_rewards(metrics, value, expr_str, self.training_progress, pool_avg_sortino=pool_avg_sortino)
        # Structural novelty reward (diversity vs current pool)
        reward_dict['r_novelty'] = self.compute_novelty_reward(embedding)
        
        # 7. Admission Logic (Gates -> Storage)
        # Using the simplified "Club" logic:
        # A factor is admitted if it meets minimum standards.
        # It replaces an existing factor if it is better or provides diversity.
        
        is_admitted = False
        
        # Hard Gates
        passes_ic = train_ic_mean > self.min_train_ic
        passes_excess = (excess_mean is not None) and (excess_mean > 0.0)
        
        if passes_ic and passes_excess:
            is_admitted = True
        else:
             mark("rejected_gates")

        if is_admitted:
            # Diversity / Correlation check
            # Only calculate mutual ICs if we are considering adding it
            ic_mean_check, ic_mut = self._calc_ics(
                value,
                ic_mut_threshold=self.ic_mut_threshold,
                precomputed_single_ic=train_ic_mean
            )
            
            # Prepare metadata
            factor_metrics = {
                "expr": str(expr),
                "train_ic": float(train_ic_mean),
                "icir": float(icir),
                "excess_return_mean": float(excess_mean) if excess_mean is not None else 0.0,
                "sortino": float(excess_sortino) if excess_sortino is not None else 0.0,
                "source": source
            }
            
            # Add/Replace Logic (QFR Style)
            # 1. Add tentatively
            self._add_factor(expr, value, train_ic_mean, ic_mut, embedding, factor_metrics)
            
            # 2. Check if Full
            if self.size > self.capacity:
                # 3. Optimize Weights (Critical for QFR eviction)
                self.recompute_weights()
                
                # 4. Evict Smallest Absolute Weight
                # Find index of factor with min|w|
                # (Skip index 0 usually?? No, fairness.)
                # weights are stored in self.weights[:self.size]
                current_weights = self.weights[:self.size]
                abs_weights = np.abs(self.weights[:self.size])
                
                # Check for "do not remove self immediately if possible" or just pure competition?
                # Pure competition.
                
                worst_idx = int(np.argmin(abs_weights))
                
                self._remove_idx(worst_idx)
                
                if worst_idx == self.size: # We removed the one we just added (it was at end before swap?)
                    # Wait, _remove_idx swaps with last and decrements size.
                    # If worst_idx was the last one (the new one), it just gets popped.
                    mark("rejected_weight")
                    is_admitted = False # Effectively didn't stay
                else:
                    mark("replace_weight")
                    is_admitted = True
            else:
                mark("added")
                is_admitted = True


        if count_eval:
            self.eval_cnt += 1

        return cache_and_return(reward_dict, is_admitted)
    def _find_worst_factor_idx(self) -> int:
        """
        Identify the worst factor to remove, prioritizing those failing absolute goals.
        """
        # 1. Check for factors failing absolute goals (IC < 0.01, RankIC < 0.01, Excess < 0)
        # We iterate and find the "worst" among failures (e.g. lowest IC)
        worst_failure_idx = -1
        worst_failure_val = float('inf')
        
        for i, note in enumerate(self.factor_notes):
            if note is None or i >= self.size: continue
            
            # Check failure conditions
            failed = False
            metric_val = 0.0
            
            if note.get('train_ic', 0) < 0.01:
                failed = True
                metric_val = note.get('train_ic', 0)
            elif note.get('rank_ic_train', 0) < 0.01:
                failed = True
                metric_val = note.get('rank_ic_train', 0)
            elif note.get('excess_return_mean', 0) < 0.0:
                failed = True
                metric_val = note.get('excess_return_mean', 0)
                
            if failed:
                # If multiple fail, pick the one with lowest metric (heuristic)
                # or just pick the first one. Let's pick lowest IC for consistency.
                ic = note.get('train_ic', 0)
                if ic < worst_failure_val:
                    worst_failure_val = ic
                    worst_failure_idx = i
        
        if worst_failure_idx != -1:
            return worst_failure_idx
        
        # 2. Fallback to lowest score if all pass absolute goals
        current_scores = self._current_scores()
        if len(current_scores) == 0: return 0
        return int(np.argmin(current_scores))

    def force_load_exprs(self, exprs: List[Expression], source: str = "resume") -> None:
        """
        Warm start the pool with pre-existing expressions while reusing the same
        acceptance logic as new GFN samples (minus the generation step).
        """
        seen = set()
        for expr in exprs:
            expr_key = str(expr)
            if expr_key in seen:
                continue
            seen.add(expr_key)
            self.try_new_expr(expr, embedding=None, source=source, count_eval=False)

        if self.size > 0:
            self.recompute_weights()

    def _compute_multiple_correlation(self, value: Tensor) -> Optional[float]:
        """
        Estimate the multiple correlation (R^2) between a candidate factor and the existing pool.
        Returns None if disabled or insufficient historical samples.
        """
        if not self.use_r_squared or self.size == 0:
            return None
        existing_vectors: List[np.ndarray] = []
        for stored in self.values[:self.size]:
            if stored is None:
                continue
            existing_vectors.append(stored.detach().cpu().numpy().reshape(-1))
        if not existing_vectors:
            return None
        y = value.detach().cpu().numpy().reshape(-1)
        
        # Find minimum length across all vectors (including y)
        min_len = min(len(y), min(len(v) for v in existing_vectors))
        # Truncate all to minimum length (take tail - most recent data)
        y = y[-min_len:]
        existing_vectors = [v[-min_len:] for v in existing_vectors]
        
        X = np.stack(existing_vectors, axis=1)
        if X.shape[0] != y.shape[0]:
            return None
        valid_mask = np.isfinite(y)
        valid_mask &= np.all(np.isfinite(X), axis=1)
        if valid_mask.sum() <= X.shape[1] + 1:
            return None
        X = X[valid_mask]
        y = y[valid_mask]
        y_centered = y - y.mean()
        X_centered = X - X.mean(axis=0, keepdims=True)
        denom = np.dot(y_centered, y_centered)
        if denom <= 1e-9:
            return None
        try:
            beta, _, _, _ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
        except np.linalg.LinAlgError:
            return None
        y_pred = X_centered @ beta
        resid = y_centered - y_pred
        ss_res = float(np.dot(resid, resid))
        r_squared = 1.0 - ss_res / denom
        return float(np.clip(r_squared, 0.0, 1.0))
    
    def _add_factor(
        self,
        expr: Expression,
        value: Tensor,
        ic_ret: float,
        ic_mut: List[float],
        embedding: Optional[Tensor] = None,
        factor_metrics: Optional[Dict] = None
    ):
        # Call parent method to handle standard factor storage
        super()._add_factor(expr, value, ic_ret, ic_mut)
        # Store the embedding for the newly added factor
        n = self.size - 1  # size was incremented in parent method
        self.embeddings[n] = embedding
        self.factor_notes[n] = factor_metrics
        self.best_insample_weights = list(self.weights[:self.size])
    
    def _remove_idx(self, idx: int) -> None:
        """
        Remove the factor at the given index by swapping it with the tail element.
        """
        if self.size == 0 or idx >= self.size:
            return
        last = self.size - 1
        self._swap_idx(idx, last)
        self._clear_last()

    def _clear_last(self) -> None:
        if self.size == 0:
            return
        last = self.size - 1
        self.exprs[last] = None
        self.values[last] = None
        self.single_ics[last] = 0.0
        self.mutual_ics[last, :] = 0.0
        self.mutual_ics[:, last] = 0.0
        self.weights[last] = 0.0
        self.embeddings[last] = None
        self.factor_notes[last] = None
        self.size -= 1
        self.best_insample_weights = list(self.weights[:self.size])

    def _pop(self, score_based_idx: Optional[int] = None) -> None:
        # Pop the factor with the lowest score (default: lowest ic)
        if self.size <= self.capacity:
            return
        if score_based_idx is None:
            score_based_idx = int(np.argmin(np.abs(self.single_ics[:self.size])))
        self._swap_idx(score_based_idx, self.capacity)
        self.size = self.capacity
        self.best_insample_weights = list(self.weights[:self.size])
    
    def _swap_idx(self, i, j) -> None:
        if i == j:
            return
        # Call parent method to handle standard factor swapping
        super()._swap_idx(i, j)
        # Swap embeddings
        self.embeddings[i], self.embeddings[j] = self.embeddings[j], self.embeddings[i]
        self.factor_notes[i], self.factor_notes[j] = self.factor_notes[j], self.factor_notes[i]

    def _current_scores(self) -> np.ndarray:
        """
        Retrieve stored scores for comparison when the pool is full.
        Falls back to abs(single_ics) if ic_reward is not recorded.
        """
        scores: List[float] = []
        for i in range(self.size):
            note = self.factor_notes[i] or {}
            # Prefer stored ic_reward + excess if available
            ic_reward = note.get("ic_reward")
            excess = note.get("excess_return_mean", 0.0)
            if ic_reward is not None:
                score = float(ic_reward) + max(0.0, float(excess or 0.0))
            else:
                score = abs(self.single_ics[i])
            scores.append(float(score))
        return np.array(scores)

    def calculate_covariance_entropy(self) -> Optional[float]:
        """
        Compute covariance-based entropy of factor outputs.
        Flattens each factor's normalized values across time/stocks, drops rows with NaN,
        computes covariance, and entropy of eigenvalue distribution normalized by log(n_factors).
        """
        if self.size < 2:
            return None
        mats = []
        for v in self.values[:self.size]:
            if v is None:
                continue
            arr = v.detach().cpu().numpy().reshape(-1)
            mats.append(arr)
        if len(mats) < 2:
            return None
        mat = np.stack(mats, axis=1)  # samples x factors
        valid_rows = ~np.isnan(mat).any(axis=1)
        mat = mat[valid_rows]
        if mat.shape[0] < 2:
            raise ValueError("After discarding NaN, the number of samples is insufficient.")
        C = np.cov(mat, rowvar=False)
        self.covariance = C
        eigs = np.linalg.eigvalsh(C)
        eigs = np.clip(eigs, a_min=0, a_max=None)
        total = eigs.sum()
        if total <= 0:
            return None
        p = eigs / total
        p = p[p > 0]
        diversity = -(p * np.log(p)).sum()
        diversity = diversity / np.log(len(mats))
        self.diversity = float(diversity)
        return self.diversity

    def _optimize_internal(self, ics_ret: Tensor, ics_mut: Tensor, init_weights: Tensor, alpha: float = 5e-3, lr: float = 5e-4, n_iter: int = 100) -> float:
        """
        Stateless optimization of weights to maximize Ensemble IC.
        Returns the optimized Ensemble IC.
        """
        weights = init_weights.clone().requires_grad_()
        optim = torch.optim.Adam([weights], lr=lr)
        
        best_loss = 1e9
        
        for _ in range(n_iter):
            ret_ic_sum = (weights * ics_ret).sum()
            mut_ic_sum = (torch.outer(weights, weights) * ics_mut).sum()
            loss_ic = mut_ic_sum - 2 * ret_ic_sum + 1
            loss_l1 = torch.norm(weights, p=1)
            loss = loss_ic + alpha * loss_l1
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
        
        # Calculate final Ensemble IC
        with torch.no_grad():
            # Ensemble IC = (w.T @ IC_ret) / sqrt(w.T @ IC_mut @ w)
            # But the loss minimized (w.T Q w - 2 w.T b + 1) is related to MSE of prediction.
            # The actual IC is roughly proportional to the optimized objective if we assume linearity.
            # Let's use the standard formula for Ensemble IC:
            # IC_ens = Cov(Sum w_i f_i, y) / Std(Sum w_i f_i)
            # IC_ens = (w.T @ IC_ret) / sqrt(w.T @ IC_mut @ w)
            # Note: IC_mut is correlation matrix of factors.
            
            numerator = (weights * ics_ret).sum()
            denominator = torch.sqrt((torch.outer(weights, weights) * ics_mut).sum())
            if denominator < 1e-6:
                return 0.0
            ensemble_ic = numerator / denominator
            return ensemble_ic.item()

    def calculate_ensemble_improvement(self, ic_ret: float, ic_mut: np.ndarray) -> float:
        """
        Calculate how much the candidate factor improves the Ensemble IC.
        """
        if self.size == 0:
            return ic_ret # First factor improvement is its IC
            
        device = self.device
        
        try:
            # Current state - ensure float type for gradients
            curr_ics_ret = torch.from_numpy(self.single_ics[:self.size]).float().to(device)
            curr_ics_mut = torch.from_numpy(self.mutual_ics[:self.size, :self.size]).float().to(device)
            curr_weights = torch.from_numpy(self.weights[:self.size]).float().to(device)
            
            # Calculate current Ensemble IC
            curr_ensemble_ic = self._optimize_internal(curr_ics_ret, curr_ics_mut, curr_weights, n_iter=50)
        except Exception:
            # If optimization fails, just return the IC improvement
            return max(0.0, ic_ret)
        
        try:
            # Augmented state
            # Append new IC
            new_ics_ret = torch.cat([curr_ics_ret, torch.tensor([ic_ret], device=device, dtype=torch.float32)])
            
            # Append new correlations
            # ic_mut is (N,), correlations with existing factors
            # We need to construct (N+1, N+1)
            # [[M, c], [c.T, 1]]
            
            # Note: ic_mut passed from try_new_expr might be (N,) or (capacity,).
            # We need the first N elements.
            cand_corr = torch.from_numpy(ic_mut[:self.size]).float().to(device)
        
            new_ics_mut = torch.eye(self.size + 1, device=device, dtype=torch.float32)
            new_ics_mut[:self.size, :self.size] = curr_ics_mut
            new_ics_mut[:self.size, self.size] = cand_corr
            new_ics_mut[self.size, :self.size] = cand_corr
            
            # Init weights: [old_weights, 0]
            new_weights = torch.cat([curr_weights, torch.tensor([0.0], device=device, dtype=torch.float32)])
            
            # Optimize
            new_ensemble_ic = self._optimize_internal(new_ics_ret, new_ics_mut, new_weights, n_iter=50)
            
            return new_ensemble_ic - curr_ensemble_ic
        except Exception:
            # If optimization fails, return a small improvement estimate
            return max(0.0, ic_ret * 0.1)

    def recompute_weights(self, alpha: float = 5e-3, lr: float = 5e-4, n_iter: int = 500) -> np.ndarray:
        """
        Refit ensemble weights on the current pool content.
        """
        if self.size == 0:
            self.best_insample_weights = []
            return np.array([])
        new_weights = self._optimize(alpha=alpha, lr=lr, n_iter=n_iter)
        self.weights[:self.size] = new_weights
        self.best_insample_weights = list(new_weights)
        self.best_ic_ret = self.evaluate_ensemble()
        return new_weights

    def to_dict(self) -> dict:
        factor_notes: List[Dict] = []
        for i in range(self.size):
            note = self.factor_notes[i] or {}
            sanitized = {}
            for k, v in note.items():
                if isinstance(v, (np.floating, np.integer)):
                    sanitized[k] = float(v)
                else:
                    sanitized[k] = v
            sanitized["expr"] = sanitized.get("expr", str(self.exprs[i]))
            sanitized["weight"] = float(self.weights[i])
            factor_notes.append(sanitized)

        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "weights": list(self.weights[:self.size]),
            "record_reward": self.record_reward,
            "best_in_sample_ic": float(self.best_ic_ret) if self.best_ic_ret is not None else 0.0,
            "best_in_sample_weights": list(self.best_insample_weights),
            "factor_notes": factor_notes,
            "eval_cnt": int(self.eval_cnt),
            "train_window": {"start": self.data_start, "end": self.data_end},
        }

    def _combine_scores(self, train_score: float, test_score: Optional[float]) -> float:
        if test_score is None:
            return train_score
        return min(train_score, test_score)

    def _compute_ic_statistics(self, factor: Tensor, target_tensor: Tensor) -> Tuple[float, float]:
        # Align shapes if they differ
        if factor.shape[0] != target_tensor.shape[0]:
            min_len = min(factor.shape[0], target_tensor.shape[0])
            factor = factor[-min_len:]
            target_tensor = target_tensor[-min_len:]
        ic_series = batch_pearsonr(factor, target_tensor)
        ic_mean = ic_series.mean().item()
        ic_std = ic_series.std().item()
        return ic_mean, ic_std

    def _aggregate_ic_by_month(self, ic_series: Tensor) -> Tuple[List[float], List[str]]:
        """
        Aggregate daily IC values into monthly means.

        Args:
            ic_series: Tensor of shape [days] containing daily IC values

        Returns:
            Tuple of (monthly_ics, month_labels) where:
            - monthly_ics: List of mean IC for each month
            - month_labels: List of month strings (e.g., "2023-01")
        """
        import pandas as pd

        # Get dates from StockData - align with IC series
        # ic_series corresponds to evaluation window (after backtrack, before future)
        dates = pd.to_datetime(self.data._dates)

        # Trim dates to match IC series length (accounting for backtrack/future days)
        start_idx = self.data.max_backtrack_days
        end_idx = len(dates) - self.data.max_future_days if self.data.max_future_days > 0 else len(dates)
        eval_dates = dates[start_idx:end_idx]

        # Ensure lengths match
        ic_len = len(ic_series)
        if len(eval_dates) != ic_len:
            # Fallback: use last ic_len dates from eval window
            if ic_len <= len(eval_dates):
                eval_dates = eval_dates[-ic_len:]
            else:
                # Not enough dates, return empty
                return [], []

        # Convert IC series to numpy
        ic_np = ic_series.detach().cpu().numpy()

        # Create DataFrame for grouping
        df = pd.DataFrame({
            'date': eval_dates,
            'ic': ic_np
        })
        df['month'] = df['date'].dt.to_period('M')

        # Group by month and compute mean IC
        monthly = df.groupby('month')['ic'].mean()

        monthly_ics = monthly.values.tolist()
        month_labels = [str(m) for m in monthly.index]

        return monthly_ics, month_labels

    def calculate_seasonality_score(self, ic_series: Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Calculate seasonality score based on sign-switching high |IC| months.

        Seasonality captures factors that work in opposite directions in different
        market regimes but are stable (not noise). High seasonality factors exhibit:
        - High absolute IC in individual months (above threshold)
        - Regular sign flips between adjacent qualifying months
        - Consistent flip intervals (low variance = more regular)

        Args:
            ic_series: Tensor of shape [days] containing daily IC values

        Returns:
            Tuple of (seasonal_score, debug_dict) where:
            - seasonal_score: Float in [0, 1], higher = more seasonal
            - debug_dict: Dictionary with intermediate values for logging
        """
        monthly_ics, month_labels = self._aggregate_ic_by_month(ic_series)

        debug = {
            'n_months': len(monthly_ics),
            'qualifying_months': 0,
            'sign_flips': 0,
            'sign_flip_rate': 0.0,
            'regularity_bonus': 0.0,
            'avg_abs_ic': 0.0,
            'seasonal_score': 0.0
        }

        # Check minimum data requirement
        if len(monthly_ics) < self.min_months_for_seasonality:
            return 0.0, debug

        # Identify qualifying months (|monthly_IC| >= threshold)
        threshold = self.seasonality_ic_threshold
        qualifying_indices = []
        qualifying_ics = []

        for i, ic in enumerate(monthly_ics):
            if abs(ic) >= threshold:
                qualifying_indices.append(i)
                qualifying_ics.append(ic)

        debug['qualifying_months'] = len(qualifying_indices)

        # Need at least 2 qualifying months for sign flip analysis
        if len(qualifying_indices) < 2:
            return 0.0, debug

        # Count sign flips between adjacent qualifying months
        sign_flips = 0
        flip_intervals = []  # Distance between flips for regularity
        last_flip_idx = None

        for i in range(1, len(qualifying_ics)):
            prev_sign = 1 if qualifying_ics[i-1] > 0 else -1
            curr_sign = 1 if qualifying_ics[i] > 0 else -1

            if prev_sign != curr_sign:
                sign_flips += 1
                # Track interval since last flip
                if last_flip_idx is not None:
                    flip_intervals.append(qualifying_indices[i] - last_flip_idx)
                last_flip_idx = qualifying_indices[i]

        debug['sign_flips'] = sign_flips

        # Calculate sign flip rate
        max_possible_flips = len(qualifying_indices) - 1
        sign_flip_rate = sign_flips / max_possible_flips if max_possible_flips > 0 else 0.0
        debug['sign_flip_rate'] = sign_flip_rate

        # Calculate regularity bonus (lower variance in flip intervals = more regular)
        regularity_bonus = 1.0
        if len(flip_intervals) >= 2:
            interval_std = float(np.std(flip_intervals))
            interval_mean = float(np.mean(flip_intervals))
            # Coefficient of variation: std/mean, lower = more regular
            if interval_mean > 0:
                cv = interval_std / interval_mean
                # Map CV to [0, 1]: cv=0 -> 1.0, cv>=2 -> 0.0
                regularity_bonus = max(0.0, 1.0 - cv / 2.0)
        debug['regularity_bonus'] = regularity_bonus

        # Average absolute IC of qualifying months
        avg_abs_ic = sum(abs(ic) for ic in qualifying_ics) / len(qualifying_ics)
        debug['avg_abs_ic'] = avg_abs_ic

        # Final seasonality score
        # Score = sign_flip_rate * regularity_bonus * avg_abs_ic
        # Scale avg_abs_ic to ~1 by dividing by typical threshold
        scaled_avg_ic = min(1.0, avg_abs_ic / 0.05)  # IC of 0.05 maps to 1.0
        seasonal_score = sign_flip_rate * regularity_bonus * scaled_avg_ic
        debug['seasonal_score'] = seasonal_score

        return seasonal_score, debug

    def calculate_consistency_score(self, ic_series: Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Calculate consistency score based on sub-period IC sampling.

        Consistency rewards factors with stable positive contributions across
        all time periods, avoiding factors that only work in specific regimes.

        Args:
            ic_series: Tensor of shape [days] containing daily IC values

        Returns:
            Tuple of (consistency_score, debug_dict) where:
            - consistency_score: Float in [0, 1], higher = more consistent
            - debug_dict: Dictionary with intermediate values for logging
        """
        monthly_ics, month_labels = self._aggregate_ic_by_month(ic_series)

        debug = {
            'n_periods': len(monthly_ics),
            'passing_periods': 0,
            'pass_ratio': 0.0,
            'consistency_bonus': 0.0,
            'consistency_score': 0.0
        }

        # Check minimum data requirement
        if len(monthly_ics) < self.min_periods_for_consistency:
            return 0.0, debug

        # Count periods meeting IC threshold
        threshold = self.consistency_ic_threshold
        passing_periods = sum(1 for ic in monthly_ics if ic > threshold)

        debug['passing_periods'] = passing_periods

        # Calculate pass ratio
        pass_ratio = passing_periods / len(monthly_ics)
        debug['pass_ratio'] = pass_ratio

        # Consistency bonus: linear scale from 0.5 to 1.0
        # pass_ratio < 0.5 -> bonus = 0
        # pass_ratio = 0.5 -> bonus = 0
        # pass_ratio = 1.0 -> bonus = 1.0
        consistency_bonus = max(0.0, (pass_ratio - 0.5) / 0.5)
        debug['consistency_bonus'] = consistency_bonus

        # Final consistency score (same as bonus for now, can add IC magnitude later)
        consistency_score = consistency_bonus
        debug['consistency_score'] = consistency_score

        return consistency_score, debug

    def _ic_sign(self, value: float) -> int:
        if abs(value) < self.sign_consistency_epsilon:
            return 0
        return 1 if value > 0 else -1
    
    def _align_ic(self, ic_value: float) -> float:
        """Return absolute IC magnitude."""
        return abs(float(ic_value))

    def _compute_rank_ic(self, value: Tensor, target_tensor: Tensor) -> float:
        # Align shapes if they differ
        if value.shape[0] != target_tensor.shape[0]:
            min_len = min(value.shape[0], target_tensor.shape[0])
            value = value[-min_len:]
            target_tensor = target_tensor[-min_len:]
        rank_ic = batch_spearmanr(value, target_tensor).mean().item()
        if math.isnan(rank_ic):
            return 0.0
        return abs(rank_ic)

    def _calc_ics(
        self,
        value: Tensor,
        ic_mut_threshold: Optional[float] = None,
        precomputed_single_ic: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Override parent to use configured ic_mut_threshold and allow precomputed single_ic.

        Args:
            value: Factor values tensor
            ic_mut_threshold: Max correlation threshold for mutual ICs
            precomputed_single_ic: If provided, skip single_ic calculation (optimization)

        Returns:
            Tuple of (single_ic, mutual_ics) or (None, None) if rejected
        """
        if ic_mut_threshold is None:
            ic_mut_threshold = self.ic_mut_threshold

        # Use precomputed single_ic if available (optimization: avoid duplicate batch_pearsonr)
        if precomputed_single_ic is not None:
            single_ic = precomputed_single_ic
        else:
            # Align shapes before batch_pearsonr
            if value.shape[0] != self.target.shape[0]:
                min_len = min(value.shape[0], self.target.shape[0])
                value_for_ic = value[-min_len:]
                target_for_ic = self.target[-min_len:]
            else:
                value_for_ic = value
                target_for_ic = self.target
            single_ic = batch_pearsonr(value_for_ic, target_for_ic).mean().item()

        # Check threshold
        thres = self.ic_lower_bound if self.ic_lower_bound is not None else 0.
        if not (self.size > 1 or self._under_thres_alpha) and abs(single_ic) < thres:
            return None, None

        # Compute mutual ICs (correlation with existing factors)
        mutual_ics = []
        for i in range(self.size):
            existing_value = self.values[i]
            # Align shapes if they differ (different rolling windows produce different lengths)
            if value.shape[0] != existing_value.shape[0]:
                min_len = min(value.shape[0], existing_value.shape[0])
                value_aligned = value[-min_len:]
                existing_aligned = existing_value[-min_len:]
            else:
                value_aligned = value
                existing_aligned = existing_value
            mutual_ic = batch_pearsonr(value_aligned, existing_aligned).mean().item()
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return None, None
            mutual_ics.append(mutual_ic)

        return single_ic, np.array(mutual_ics) if mutual_ics else np.array([])

    def _compute_long_short_metrics(self, value: Tensor, target_tensor: Tensor) -> Dict[str, float]:
        if value is None or target_tensor is None:
            return {"monotonicity": 0.0, "directional": 0.0, "spread": 0.0}

        factor = value.detach().cpu().numpy()
        returns = target_tensor.detach().cpu().numpy()
        bucket_returns: List[List[float]] = [[] for _ in range(self.long_short_bins)]

        for day in range(factor.shape[0]):
            f_day = factor[day]
            r_day = returns[day]
            mask = np.isfinite(f_day) & np.isfinite(r_day)
            if mask.sum() < self.long_short_bins:
                continue
            f_sel = f_day[mask]
            r_sel = r_day[mask]
            order = np.argsort(f_sel)
            bucket_edges = np.linspace(0, len(order), self.long_short_bins + 1, dtype=int)
            for bucket in range(self.long_short_bins):
                start = bucket_edges[bucket]
                end = bucket_edges[bucket + 1]
                if end - start <= 0:
                    continue
                members = order[start:end]
                bucket_returns[bucket].append(r_sel[members].mean())

        avg_returns = np.array([np.mean(bucket) if bucket else 0.0 for bucket in bucket_returns])
        if np.allclose(avg_returns, 0.0):
            return {"monotonicity": 0.0, "directional": 0.0, "spread": 0.0}

        # Monotonicity: Correlation between bucket index and return
        if len(avg_returns) > 1:
            monotonicity = np.corrcoef(np.arange(len(avg_returns)), avg_returns)[0, 1]
            if np.isnan(monotonicity): monotonicity = 0.0
        else:
            monotonicity = 0.0

        directional = min(max(avg_returns[-1], 0.0), max(-avg_returns[0], 0.0))
        directional_score = math.tanh(directional / max(1e-6, self.directional_scale))
        spread = max(0.0, avg_returns[-1] - avg_returns[0])
        spread_score = math.tanh(spread / max(1e-6, self.spread_scale))

        return {
            "monotonicity": float(monotonicity),
            "directional": float(directional_score),
            "spread": float(spread_score)
        }

    def _penalized_reward(self, value: float, threshold: float) -> float:
        if threshold <= 0:
            return max(1e-6, float(value))
        normalized = max(0.0, float(value)) / max(threshold, 1e-8)
        normalized = min(normalized, 1.0)
        return max(1e-6, normalized * self.constraint_penalty_scale)

    def _load_raw_target_returns(self, data: Optional[StockData] = None) -> Optional[Tensor]:
        if data is None:
            data = self.data
        if data is None:
            return None
        target_expr = getattr(self, "target_expr", None)
        if target_expr is None:
            return None
        
        try:
            # Use the expression form so we get the raw, unnormalized target
            target_val = target_expr.evaluate(data)
            
            # Align to same length as self.target (which was truncated by parent class)
            expected_len = self.target.shape[0] if self.target is not None else target_val.shape[0]
            if target_val.shape[0] != expected_len:
                # Take the tail (most recent days) to align with evaluated target
                target_val = target_val[-expected_len:]
            
            # --- Safety Fix for Outliers ---
            # 1. Clamp extremely large returns (e.g. from penny stocks or data artifacts)
            # -1.0 is max loss, 10.0 is 1000% gain (reasonable cap for 20 days)
            target_val = torch.clamp(target_val, min=-1.0, max=10.0)

            # 2. Mask periods where price is near zero (0.0 fill artifact or missing data)
            # This prevents infinite returns created by 0.0 denominators
            from alphagen_qlib.stock_data import FeatureType
            close_idx = -1
            for i, f in enumerate(data._features):
                if f == FeatureType.CLOSE:
                    close_idx = i
                    break
            
            if close_idx >= 0:
                # data.data shape is (Time, Features, Stocks)
                close_prices = data.data[:, close_idx, :]
                # Align close_prices to target_val length
                if close_prices.shape[0] != target_val.shape[0]:
                    close_prices = close_prices[-target_val.shape[0]:]
                # Mask where Close < 1e-4 (effectively 0.0 or tiny)
                invalid_mask = close_prices < 1e-4
                target_val = torch.where(invalid_mask, torch.tensor(float('nan'), device=target_val.device), target_val)
                
            return target_val
        except Exception as e:
            print(f"Error loading raw target returns: {e}")
            return None

    def _compute_baseline_returns(self, returns: Optional[Tensor]) -> Optional[Tensor]:
        if returns is None:
            return None
        valid_mask = torch.isfinite(returns)
        if valid_mask.dim() < 2:
            return None
        filled = torch.where(valid_mask, returns, torch.zeros_like(returns))
        counts = valid_mask.sum(dim=1).clamp(min=1)
        baseline = filled.sum(dim=1) / counts
        baseline = torch.where(torch.isfinite(baseline), baseline, torch.zeros_like(baseline))
        return baseline

    def _compute_excess_return_mean(self, value: Tensor) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if self.raw_target_returns is None or self.baseline_returns is None:
            return None, None, None
        factor = value.detach().cpu().numpy()
        returns_np = self.raw_target_returns.detach().cpu().numpy()
        baseline_np = self.baseline_returns.detach().cpu().numpy()
        n_days = min(factor.shape[0], returns_np.shape[0], baseline_np.shape[0])
        if n_days == 0:
            return None, None, None
            
        # Align to the tail (most recent days)
        # Tensors might differ in length due to rolling window consumption at the start
        factor = factor[-n_days:]
        returns_np = returns_np[-n_days:]
        baseline_np = baseline_np[-n_days:]
        
        excess_long_only_list = []
        excess_long_short_list = []
        
        for day in range(n_days):
            f_day = factor[day]
            r_day = returns_np[day]
            mask = np.isfinite(f_day) & np.isfinite(r_day)
            n_valid = mask.sum()
            if n_valid < self.long_short_bins or not np.isfinite(baseline_np[day]):
                continue
            f_sel = f_day[mask]
            r_sel = r_day[mask]
            order = np.argsort(f_sel)

            # Long-Only: Always top 20% of stocks (constant)
            top_20_pct_count = max(1, int(n_valid * 0.2))
            members_long_only = order[-top_20_pct_count:]
            long_only_return = r_sel[members_long_only].mean()

            # Long-Short: Top 10% vs Bottom 10% (independent of bin count)
            top_10_pct_count = max(1, int(n_valid * 0.1))
            bottom_10_pct_count = max(1, int(n_valid * 0.1))
            members_long = order[-top_10_pct_count:]
            members_short = order[:bottom_10_pct_count]
            long_return = r_sel[members_long].mean()
            short_return = r_sel[members_short].mean()
            
            # Calculate Excess Returns
            # Long-Only: Active Return vs Benchmark
            excess_long_only = long_only_return - baseline_np[day]
            # Long-Short: Pure Spread (not subtracting benchmark, as it's market neutral)
            excess_long_short = long_return - short_return
            
            excess_long_only_list.append(excess_long_only)
            excess_long_short_list.append(excess_long_short)
            
        if not excess_long_only_list:
            return None, None, None
        
        # Calculate mean excess returns
        excess_long_only_mean = float(np.mean(excess_long_only_list))
        excess_long_short_mean = float(np.mean(excess_long_short_list))
        
        # Calculate baseline mean for context (though not used for LS threshold anymore)
        baseline_mean = float(np.mean(baseline_np[np.isfinite(baseline_np)]))
        
        # Acceptance logic:
        # - Long-Short: Pass if Spread > 0 (Positive Alpha)
        # - Long-Only: Pass if Active Return > 0 (Beat Market)
        long_short_passes = excess_long_short_mean > 0
        long_only_passes = excess_long_only_mean > 0
        
        # Factor passes if EITHER condition is met
        if long_short_passes or long_only_passes:
            # Use the better of the two for reward calculation
            final_excess = max(excess_long_only_mean, excess_long_short_mean)
        else:
            # Return the max but indicate failure by returning None later
            final_excess = max(excess_long_only_mean, excess_long_short_mean)
        
        # Calculate downside metrics using final excess
        all_excess = np.array(excess_long_only_list if excess_long_only_mean > excess_long_short_mean else excess_long_short_list)
        downside = all_excess[all_excess < 0]
        downside_std = float(np.sqrt(np.mean(np.square(downside)))) if downside.size > 0 else None
        sortino = None
        if downside_std and downside_std > 0:
            sortino = float(final_excess / downside_std)
        
        # Store metadata for validation in try_new_expr
        self._last_long_short_passes = long_short_passes
        self._last_long_only_passes = long_only_passes
        self._last_baseline_mean = baseline_mean
        
        return float(final_excess), sortino, downside_std

    def _compute_downside_metrics(self, value: Tensor, raw_target: Optional[Tensor]) -> Dict[str, float]:
        """
        Measure how well a factor surfaces persistent losers (bottom bucket).
        Returns magnitudes that are positive when the factor captures downside.
        """
        if raw_target is None:
            return {"mean_loss": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "hit_ratio": 0.0}
        factor = value.detach().cpu().numpy()
        returns_np = raw_target.detach().cpu().numpy()
        n_days = min(factor.shape[0], returns_np.shape[0])
        if n_days == 0:
            return {"mean_loss": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "hit_ratio": 0.0}

        bottom_bucket_returns: List[float] = []
        for day in range(n_days):
            f_day = factor[day]
            r_day = returns_np[day]
            mask = np.isfinite(f_day) & np.isfinite(r_day)
            if mask.sum() < self.long_short_bins:
                continue
            f_sel = f_day[mask]
            r_sel = r_day[mask]
            order = np.argsort(f_sel)
            bucket_edges = np.linspace(0, len(order), self.long_short_bins + 1, dtype=int)
            start = bucket_edges[0]
            end = bucket_edges[1]
            if end - start <= 0:
                continue
            members = order[start:end]
            bottom_bucket_returns.append(r_sel[members].mean())

        if not bottom_bucket_returns:
            return {"mean_loss": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "hit_ratio": 0.0}

        bottom_arr = np.array(bottom_bucket_returns, dtype=float)
        negative = bottom_arr[bottom_arr < 0]
        mean_loss = float(np.mean(-np.minimum(bottom_arr, 0.0)))
        hit_ratio = float((bottom_arr < 0).mean())
        downside_std = float(np.sqrt(np.mean(np.square(negative)))) if negative.size > 0 else 0.0
        sortino = float(mean_loss / (downside_std + 1e-6)) if mean_loss > 0 else 0.0
        max_dd = self._max_drawdown_from_returns(bottom_arr)
        return {
            "mean_loss": mean_loss,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "hit_ratio": hit_ratio,
        }

    def _max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        if returns.size == 0:
            return 0.0
        equity = np.cumprod(1.0 + returns)
        if equity.size == 0:
            return 0.0
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / np.where(peaks == 0, 1.0, peaks)
        min_dd = drawdowns.min() if drawdowns.size > 0 else 0.0
        return float(-min_dd) if min_dd < 0 else 0.0

    def _expression_complexity(self, expr: Expression) -> Tuple[int, int]:
        children = self._expression_children(expr)
        if not children:
            return 1, 1
        max_depth = 0
        node_count = 1
        for child in children:
            depth, nodes = self._expression_complexity(child)
            node_count += nodes
            max_depth = max(max_depth, depth)
        return max_depth + 1, node_count

    def _expression_children(self, expr: Expression) -> List[Expression]:
        children: List[Expression] = []
        for value in vars(expr).values():
            if isinstance(value, Expression):
                children.append(value)
            elif isinstance(value, (list, tuple)):
                children.extend([item for item in value if isinstance(item, Expression)])
        return children
    
    def _invalid_sign_combo(self, expr: Expression) -> Optional[str]:
        """
        Guardrail: forbid Sign feeding into degenerate consumers.
        Examples we want to block:
        - Less(Sign(...), x) clips to {-1,0,1} and is dominated by the other operand.
        - Log(Sign(...)) hits log(0/-1) => -inf/nan.
        - Rank/Quantile(TsQuantile) on Sign is uninformative (all ties).
        """
        forbidden_consumers = (Less, Log, Rank, Quantile, TsQuantile)

        def contains_sign(node: Expression) -> bool:
            if isinstance(node, Sign):
                return True
            return any(contains_sign(child) for child in self._expression_children(node))

        def dfs(node: Expression) -> Optional[str]:
            if isinstance(node, forbidden_consumers):
                if any(contains_sign(child) for child in self._expression_children(node)):
                    return f"Sign under {type(node).__name__} is disallowed"
            for child in self._expression_children(node):
                reason = dfs(child)
                if reason:
                    return reason
            return None

        return dfs(expr)
    
    def _find_k_nearest_neighbors(self, query_embedding: Tensor, k: int, exclude_self: bool = True, distance_threshold: float = 1e-6) -> List[int]:
        """
        Find k nearest neighbors based on embedding similarity
        
        Args:
            query_embedding: The embedding to find neighbors for
            k: Number of neighbors to find
            exclude_self: Whether to exclude identical embeddings (distance ≈ 0)
            distance_threshold: Minimum distance to consider as different embeddings
            
        Returns:
            List of indices of k nearest neighbors
        """
        if self.size <= 1:
            # print(f"[SSL Debug] Pool size <= 1 ({self.size}), no neighbors available")
            return []
        
        distances = []
        valid_indices = []
        
        for i in range(self.size):
            if self.embeddings[i] is not None:
                # Calculate L2 distance
                dist = torch.norm(query_embedding - self.embeddings[i]).item()
                
                # Skip if this is likely the same embedding (distance ≈ 0)
                if exclude_self and dist < distance_threshold:
                    # print(f"[SSL Debug] Factor {i}: distance = {dist:.6f} (SKIPPED - too similar/identical)")
                    continue
                    
                distances.append(dist)
                valid_indices.append(i)
                # print(f"[SSL Debug] Factor {i}: distance = {dist:.6f}")
        
        if len(distances) == 0:
            # print(f"[SSL Debug] No valid embeddings found in pool (after excluding self)")
            return []
        
        # Get k smallest distances (nearest neighbors)
        k = min(k, len(distances))
        _, indices = torch.topk(torch.tensor(distances), k, largest=False)
        
        neighbor_indices = [valid_indices[idx] for idx in indices.tolist()]
        neighbor_distances = [distances[idx] for idx in indices.tolist()]
        
        # print(f"[SSL Debug] Found {len(neighbor_indices)} neighbors: {neighbor_indices}")
        # print(f"[SSL Debug] Neighbor distances: {[f'{d:.6f}' for d in neighbor_distances]}")
        
        return neighbor_indices
    
    def _compute_similarity_weights(self, query_embedding: Tensor, neighbor_indices: List[int]) -> Tensor:
        """
        Compute similarity weights using softmax with temperature
        
        Args:
            query_embedding: The query embedding
            neighbor_indices: Indices of neighbor factors
            
        Returns:
            Normalized similarity weights
        """
        if not neighbor_indices:
            # print(f"[SSL Debug] No neighbor indices provided")
            return torch.tensor([])
        
        # Calculate squared L2 distances
        distances = []
        similarity_scores = []
        for idx in neighbor_indices:
            if self.embeddings[idx] is not None:
                dist_squared = torch.norm(query_embedding - self.embeddings[idx]) ** 2
                similarity_score = -dist_squared / self.ssl_tau  # Negative for similarity
                distances.append(dist_squared.item())
                similarity_scores.append(similarity_score)
                # print(f"[SSL Debug] Factor {idx}: dist²={dist_squared.item():.4f}, similarity_score={similarity_score.item():.4f}")
        
        if not similarity_scores:
            # print(f"[SSL Debug] No valid similarity scores computed")
            return torch.tensor([])
        
        # Apply softmax to get normalized weights
        weights = F.softmax(torch.tensor(similarity_scores), dim=0)
        
        # print(f"[SSL Debug] Similarity weights: {[f'{w.item():.4f}' for w in weights]}")
        # print(f"[SSL Debug] Weight sum: {weights.sum().item():.4f}")
        
        return weights
    
    def _compute_consistency_loss(self, query_value: Tensor, neighbor_indices: List[int], weights: Tensor) -> float:
        """
        Compute consistency loss between query factor and its neighbors
        
        Args:
            query_value: Normalized factor values for the query factor
            neighbor_indices: Indices of neighbor factors
            weights: Similarity weights
            
        Returns:
            Consistency loss value
        """
        if not neighbor_indices or len(weights) == 0:
            # print(f"[SSL Debug] No neighbors or weights for consistency loss")
            return 0.0
        
        total_loss = 0.0
        individual_losses = []
        
        # print(f"[SSL Debug] Computing consistency loss with {len(neighbor_indices)} neighbors")
        # print(f"[SSL Debug] Query value shape: {query_value.shape}")
        
        for i, idx in enumerate(neighbor_indices):
            if idx < len(self.values) and self.values[idx] is not None:
                neighbor_value = self.values[idx]
                
                # Align shapes if they differ (different rolling windows produce different lengths)
                if query_value.shape[0] != neighbor_value.shape[0]:
                    min_len = min(query_value.shape[0], neighbor_value.shape[0])
                    query_aligned = query_value[-min_len:]
                    neighbor_aligned = neighbor_value[-min_len:]
                else:
                    query_aligned = query_value
                    neighbor_aligned = neighbor_value
                
                # Calculate MSE loss per cross-section and average
                diff_squared = (query_aligned - neighbor_aligned) ** 2
                mse_per_section = diff_squared.mean(dim=1)  # Average across stocks in each day
                avg_mse = mse_per_section.mean().item()  # Average across days
                
                weighted_loss = weights[i].item() * avg_mse
                total_loss += weighted_loss
                individual_losses.append(avg_mse)
                
                # print(f"[SSL Debug] Neighbor {idx}: MSE={avg_mse:.6f}, weight={weights[i].item():.4f}, weighted_loss={weighted_loss:.6f}")
        
        # print(f"[SSL Debug] Individual MSE losses: {[f'{loss:.6f}' for loss in individual_losses]}")
        # print(f"[SSL Debug] Total consistency loss: {total_loss:.6f}")
        
        return total_loss
    
    def compute_ssl_reward(self, expr: Expression, embedding: Optional[Tensor] = None) -> float:
        """
        Compute SSL (Self-Supervised Learning) reward based on structural consistency
        Note: This now also incorporates the structure-aware behavioral reward (SA) described in Eq. (9)-(10).
        
        Args:
            expr: The expression to compute SSL reward for
            embedding: The structural embedding of the expression
            
        Returns:
            SSL+SA reward (positive value, higher is better)
        """
        # print(f"\n[SSL Debug] ===== Computing SSL Reward =====")
        # print(f"[SSL Debug] Pool size: {self.size}, K: {self.ssl_k}, τ: {self.ssl_tau}")
        
        if embedding is None:
            # print(f"[SSL Debug] No embedding provided, SSL reward = 0.0")
            return 0.0
            
        if self.size <= 1:
            # print(f"[SSL Debug] Pool size <= 1, SSL reward = 0.0")
            return 0.0
        
        # print(f"[SSL Debug] Query embedding shape: {embedding.shape}")
        # print(f"[SSL Debug] Query embedding norm: {torch.norm(embedding).item():.4f}")
        
        # Find k nearest neighbors based on embedding similarity (excluding self)
        neighbor_indices = self._find_k_nearest_neighbors(embedding, self.ssl_k, exclude_self=True)
        
        if not neighbor_indices:
            # print(f"[SSL Debug] No neighbors found, SSL reward = 0.0")
            return 0.0
        
        # Compute similarity weights
        weights = self._compute_similarity_weights(embedding, neighbor_indices)
        
        if len(weights) == 0:
            # print(f"[SSL Debug] No weights computed, SSL reward = 0.0")
            return 0.0
        
        # Get normalized factor values
        query_value, eval_err = self._safe_eval_and_normalize(expr, self.data)
        if query_value is None:
            return 0.0
        
        # Compute consistency loss
        consistency_loss = self._compute_consistency_loss(query_value, neighbor_indices, weights)
        
        # Transform consistency loss to SSL reward
        ssl_reward = np.exp(-consistency_loss)
        # print(f"[SSL Debug] SSL reward: {ssl_reward:.6f}")

        # ------------------------------------------------------------------
        # Structure-Aware behavioral reward (Eq. 9-10)
        # w_ij ∝ exp(-||e_i - e_j||^2), d_behav = mean MSE over days/stocks
        # R_SA = exp( - Σ_j w_ij * d_behav )
        # ------------------------------------------------------------------
        sa_reward = self.compute_sa_reward(expr, embedding)
        ssl_reward = float(ssl_reward) + float(sa_reward)
        # print(f"[SSL Debug] SA reward: {sa_reward:.6f} | Combined SSL+SA: {ssl_reward:.6f}")
        # print(f"[SSL Debug] ===== SSL Reward Computation Done =====\n")
        
        return ssl_reward

    def compute_sa_reward(self, expr: Expression, embedding: Optional[Tensor] = None) -> float:
        """
        Structure-Aware reward per Eq. (9)-(10):
        w_ij = softmax_j( -||e_i - e_j||^2 ), d_behav = mean MSE of factor outputs,
        R_SA = exp( - Σ_j w_ij * d_behav ).
        """
        if embedding is None or self.size <= 1:
            return 0.0

        # Neighbors by structural embedding (exclude self)
        neighbor_indices = self._find_k_nearest_neighbors(embedding, self.ssl_k, exclude_self=True)
        if not neighbor_indices:
            return 0.0

        # Softmax weights with temperature τ=1 to match Eq. (9)
        sims = []
        for idx in neighbor_indices:
            if self.embeddings[idx] is None:
                sims.append(None)
                continue
            dist_sq = torch.norm(embedding - self.embeddings[idx]) ** 2
            sims.append(-dist_sq)
        valid_scores = [s for s in sims if s is not None]
        if not valid_scores:
            return 0.0
        weights = F.softmax(torch.tensor(valid_scores), dim=0)

        # Evaluate query factor (normalized)
        query_value, eval_err = self._safe_eval_and_normalize(expr, self.data)
        if query_value is None:
            return 0.0

        # Compute behavioral distance (MSE) with weighted sum
        total_loss = 0.0
        w_idx = 0
        for i, idx in enumerate(neighbor_indices):
            if self.embeddings[idx] is None:
                continue
            if idx >= len(self.values) or self.values[idx] is None:
                continue
            neighbor_value = self.values[idx]
            # Align lengths
            if query_value.shape[0] != neighbor_value.shape[0]:
                min_len = min(query_value.shape[0], neighbor_value.shape[0])
                qv = query_value[-min_len:]
                nv = neighbor_value[-min_len:]
            else:
                qv = query_value
                nv = neighbor_value
            diff_squared = (qv - nv) ** 2
            mse_per_day = diff_squared.mean(dim=1)  # average over stocks
            avg_mse = mse_per_day.mean().item()      # average over days
            total_loss += weights[w_idx].item() * avg_mse
            w_idx += 1

        sa_reward = math.exp(-total_loss)
        return float(sa_reward)

    def compute_novelty_reward(self, embedding: Optional[Tensor] = None) -> float:
        """
        Compute a novelty reward based on structural embedding distance to the pool.

        Uses cosine distance: novelty = clamp(1 - max_cos_sim, 0, 1).
        """
        if embedding is None or self.size == 0:
            return 0.0

        sims: List[float] = []
        for i in range(self.size):
            emb_i = self.embeddings[i]
            if emb_i is None:
                continue
            try:
                emb_i = emb_i.to(embedding.device)
                sim = F.cosine_similarity(embedding.unsqueeze(0), emb_i.unsqueeze(0)).item()
                sims.append(sim)
            except Exception:
                continue

        if not sims:
            return 0.0

        novelty = 1.0 - max(sims)
        if novelty < 0.0:
            novelty = 0.0
        if novelty > 1.0:
            novelty = 1.0
        return float(novelty)
    
    def try_new_expr_with_ssl(self, expr: Expression, embedding: Optional[Tensor] = None) -> Tuple[float, float, float, float, float]:
        """
        Compute env, struct, nov, and ssl rewards.
        Compatible wrapper for GFNEnvCore.
        Returns: (env, struct, nov, ssl, log_R_tb)
        """
        reward_dict, admitted = self.try_new_expr(expr, embedding)
        
        env_reward = reward_dict.get('r_env', 0.0)
        struct_reward = reward_dict.get('r_struct', 0.0)
        nov_reward = reward_dict.get('r_novelty', 0.0)
        log_r_tb = reward_dict.get('log_R_tb', 0.0)
        
        # Get SSL reward
        ssl_reward = 0.0
        if embedding is not None:
            ssl_reward = self.compute_ssl_reward(expr, embedding)
        
        return env_reward, struct_reward, nov_reward, ssl_reward, log_r_tb
    
    def debug_embedding_similarities(self, query_embedding: Tensor) -> None:
        """
        Debug method to analyze embedding similarities in the pool
        
        Args:
            query_embedding: The embedding to compare against pool embeddings
        """
        # print(f"\n[SSL Debug] ===== Embedding Similarity Analysis =====")
        # print(f"[SSL Debug] Pool size: {self.size}")
        # print(f"[SSL Debug] Query embedding shape: {query_embedding.shape}")
        # print(f"[SSL Debug] Query embedding norm: {torch.norm(query_embedding).item():.6f}")
        
        if self.size == 0:
            # print(f"[SSL Debug] Empty pool")
            return
            
        for i in range(self.size):
            if self.embeddings[i] is not None:
                dist = torch.norm(query_embedding - self.embeddings[i]).item()
                cosine_sim = F.cosine_similarity(query_embedding.unsqueeze(0), self.embeddings[i].unsqueeze(0)).item()
                
                status = "IDENTICAL" if dist < 1e-6 else "DIFFERENT"
                # print(f"[SSL Debug] Factor {i}: L2_dist={dist:.6f}, cosine_sim={cosine_sim:.6f} [{status}]")
            else:
                # print(f"[SSL Debug] Factor {i}: No embedding stored")
                pass
                
        # print(f"[SSL Debug] ===== Embedding Analysis Done =====\n")
