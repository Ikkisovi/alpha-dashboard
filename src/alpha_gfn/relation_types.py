from enum import IntEnum
from typing import List, Tuple

class RelationType(IntEnum):
    """
    Simplified RGCN Relation Types - Matching Upstream AlphaSAGE

    Rolling ops are split into coarse sub-groups to give the encoder
    a hint about the kind of temporal transformation being applied
    (smoother/variance, extrema/quantile, diff/change, order stats).
    """

    # Unary operators: Abs, Sign, Log, Rank, etc.
    UNARY_OPERAND = 0

    # Binary operators (left and right operands)
    BINARY_LEFT = 1
    BINARY_RIGHT = 2

    # Rolling window operators: TsMean, TsStd, etc.
    ROLLING_WINDOW = 3
    ROLLING_OPERAND = 4        # Generic fallback for rolling operands
    ROLLING_MEAN_VAR = 5       # Smoothers/dispersion: mean/var/std/ema/wma/mad
    ROLLING_EXTREMA = 6        # Max/Min/ArgMax/ArgMin/Quantile
    ROLLING_DIFF = 7           # Delta, pct change, min/max diff variants
    ROLLING_ORDER = 8          # Order statistics / moments: rank, skew, kurt, median

    # Pair rolling operators: TsCorr, TsCov (3 arguments)
    PAIR_ROLLING = 9

    # Filter operators: Having, NotHaving
    FILTER_OPERAND = 10   # The data being filtered
    FILTER_CONDITION = 11 # The boolean condition

    # Operator-specific relation for Quantile parameter
    ROLLING_QUANTILE = 12

    @classmethod
    def num_relations(cls) -> int:
        return 13  # 0-12


def get_operator_relation_types(op_name: str) -> List[Tuple[int, RelationType]]:
    """
    Get relation types for operator children (simplified, daily-only).
    Returns list of (child_index, RelationType) tuples.
    """

    # Unary operators
    if op_name in {'Abs', 'Sign', 'Log', 'SLog1p', 'Inv', 'Neg', 'Sqrt', 'Rank', 'Ret'}:
        return [(0, RelationType.UNARY_OPERAND)]

    # Binary operators
    if op_name in {'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Greater', 'Less', 'Corr', 'Quantile',
                   'Gt', 'Ge', 'Lt', 'Le'}:
        return [
            (0, RelationType.BINARY_LEFT),
            (1, RelationType.BINARY_RIGHT),
        ]

    # Filter operators
    if op_name in {'Having', 'NotHaving'}:
        return [
            (0, RelationType.FILTER_OPERAND),
            (1, RelationType.FILTER_CONDITION),
        ]

    # Rolling operators (2 args: operand, window)
    if op_name in {'Ref', 'TsDelta', 'TsDelay', 'TsPctChange', 'TsMinMaxDiff', 'TsMaxDiff', 'TsMinDiff'}:
        return [
            (0, RelationType.ROLLING_DIFF),
            (1, RelationType.ROLLING_WINDOW),
        ]
    if op_name in {'TsMean', 'TsSum', 'TsStd', 'TsVar', 'TsMad', 'TsWMA', 'TsEMA', 'TsIr', 'TsDiv',
                   'TsSortino', 'TsMomRank', 'TsMaxDd'}:
        return [
            (0, RelationType.ROLLING_MEAN_VAR),
            (1, RelationType.ROLLING_WINDOW),
        ]
    # TsRelStrength is a 3-arg rolling op: operand, fast_window, slow_window
    if op_name in {'TsRelStrength'}:
        return [
            (0, RelationType.ROLLING_MEAN_VAR),
            (1, RelationType.ROLLING_WINDOW),  # fast window
            (2, RelationType.ROLLING_WINDOW),  # slow window
        ]
    if op_name in {'TsMax', 'TsMin', 'TsArgMax', 'TsArgMin'}:
        return [
            (0, RelationType.ROLLING_EXTREMA),
            (1, RelationType.ROLLING_WINDOW),
        ]
    if op_name in {'TsQuantile'}:
        return [
            (0, RelationType.ROLLING_EXTREMA),
            (1, RelationType.ROLLING_QUANTILE),
            (2, RelationType.ROLLING_WINDOW),
        ]
    if op_name in {'TsMedian', 'TsMed', 'TsRank', 'TsSkew', 'TsKurt'}:
        return [
            (0, RelationType.ROLLING_ORDER),
            (1, RelationType.ROLLING_WINDOW),
        ]

    # Pair rolling operators (3 args: left, right, window)
    if op_name in {'TsCorr', 'TsCov', 'TsBeta'}:
        # These are pair rolling operators with 3 args
        return [
            (0, RelationType.ROLLING_OPERAND),
            (1, RelationType.ROLLING_OPERAND),
            (2, RelationType.PAIR_ROLLING),
        ]

    # Default fallback: unary
    return [(0, RelationType.UNARY_OPERAND)]
