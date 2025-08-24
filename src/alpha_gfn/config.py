from alphagen.data.expression import *
from alphagen_qlib.stock_data import FeatureType

# GFN Task Hyperparameters
MAX_EXPR_LENGTH = 20

# GFN Model Hyperparameters
HIDDEN_DIM = 128
NUM_ENCODER_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.1

# Training Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Action Space
OPERATORS = [
    # Unary
    Abs, SLog1p, Inv, Sign, Log, Rank,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less,
    # Rolling
    Ref, TsMean, TsSum, TsStd, TsIr, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsVar, TsSkew, TsKurt, TsMax, TsMin,
    TsMed, TsMad, TsRank, TsDelta, TsDiv, TsPctChange, TsWMA, TsEMA,
    # Pair rolling
    TsCov, TsCorr
]

FEATURES = [
    FeatureType.OPEN,
    FeatureType.CLOSE,
    FeatureType.HIGH,
    FeatureType.LOW,
    FeatureType.VOLUME,
    FeatureType.VWAP
]

DELTA_TIMES = [10, 20, 30, 40, 50]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]
