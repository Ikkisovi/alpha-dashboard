from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType


high = High = HIGH = Feature(FeatureType.HIGH)
low = Low = LOW = Feature(FeatureType.LOW)
volume = Volume = VOLUME = Feature(FeatureType.VOLUME)
open_ = Open = OPEN = Feature(FeatureType.OPEN)
close = Close = CLOSE = Feature(FeatureType.CLOSE)
vwap = Vwap = VWAP = Feature(FeatureType.VWAP)

# Advanced features
sortino_ratio = Feature(FeatureType.SORTINO_RATIO)
ts_mom_rank = Feature(FeatureType.TS_MOM_RANK)
max_dd_ratio = Feature(FeatureType.MAX_DD_RATIO)
rel_strength_ma = Feature(FeatureType.REL_STRENGTH_MA)
drift_factor = Feature(FeatureType.DRIFT_FACTOR)
amihud_mean = Feature(FeatureType.AMIHUD_MEAN)
amihud_range = Feature(FeatureType.AMIHUD_RANGE)
log_close = Feature(FeatureType.LOG_CLOSE)
log_volume = Feature(FeatureType.LOG_VOLUME)
log_money = Feature(FeatureType.LOG_MONEY)

target = Ref(close, -20) / close - 1