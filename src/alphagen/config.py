from alphagen.data.expression import *
from alphagen_qlib.stock_data import FeatureType

MAX_EXPR_LENGTH = 20
MAX_EPISODE_LENGTH = 256

FEATURES = [FeatureType.OPEN,
            FeatureType.CLOSE,
            FeatureType.HIGH,
            FeatureType.LOW,
            FeatureType.VOLUME,
            FeatureType.VWAP
            ]

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

DELTA_TIMES = [10, 20, 30, 40, 50]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.
