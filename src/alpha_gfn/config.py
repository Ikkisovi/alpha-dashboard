from alphagen.data.expression import *
from alphagen_qlib.stock_data import FeatureType
from .relation_types import RelationType

# GFN Task Hyperparameters
MAX_EXPR_LENGTH = 20

# Relation configuration (daily-only, matching upstream)
NUM_RELATIONS = RelationType.num_relations()  # Includes intraday edges

# GFN Model Hyperparameters
HIDDEN_DIM = 128
NUM_ENCODER_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Action Space (reduced to save memory)
OPERATORS = [
    # Unary
    Abs, SLog1p, Inv, Rank, Sqrt, Ret,
    # Binary
    Add, Sub, Mul, Div,  Corr, Quantile, # Greater, Less removed
    # Logical / Transformational
    # Having, NotHaving, Gt, Ge, Lt, Le,
    # Rolling (daily) - reduced set
    Ref, TsMean, TsIr, TsVar, TsSkew, TsKurt, TsMax, TsMin, TsDelta, TsRet,
    TsMed, TsMad, TsRank, 
    # TsArgMax, TsArgMin, TsPctChange, 
    TsWMA, TsEMA, TsQuantile,
    TsSortino, TsMomRank, TsMaxDd, TsRelStrength,
    # Pair rolling (daily) - keep only TsCorr
    TsCorr,
]

# Base features: Raw OHLCV + VWAP (Normalization reverted per user request)
BASE_FEATURES = [
    FeatureType.OPEN, FeatureType.CLOSE, FeatureType.HIGH, FeatureType.LOW,
    FeatureType.VOLUME,
    FeatureType.LOG_VOLUME, FeatureType.LOG_MONEY, FeatureType.LOG_CLOSE, FeatureType.VWAP,
]

# Risk features exposed directly (others should be composed via ops)
RISK_FEATURES = [
    FeatureType.DRIFT_FACTOR,
    FeatureType.AMIHUD_MEAN,
    FeatureType.AMIHUD_RANGE,
]

# Default feature set used by GFN env/tokenizer
FEATURES = BASE_FEATURES + RISK_FEATURES

DELTA_TIMES = [10, 15, 20, 30, 40, 50]  # Original: 6 windows

CONSTANTS = [
    -1.0, 
    0.25, 0.33, 0.5, 0.67, 0.75,
    10.0, 15.0, 20.0, 30.0, 50.0, 100.0
]
