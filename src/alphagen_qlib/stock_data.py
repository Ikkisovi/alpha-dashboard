from typing import List, Union, Optional, Tuple, Dict
from enum import IntEnum
import numpy as np
import pandas as pd
import torch

class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5
    # Advanced technical features
    SORTINO_RATIO = 6
    TS_MOM_RANK = 7
    MAX_DD_RATIO = 8
    REL_STRENGTH_MA = 9
    # Intraday features
    DRIFT_FACTOR = 10
    AMIHUD_MEAN = 11
    AMIHUD_RANGE = 12
    # Magnitude features (Log transform)
    LOG_CLOSE = 13
    LOG_VOLUME = 14
    # Cross-sectional Normalized features [0, 1]
    NORM_OPEN = 15
    NORM_CLOSE = 16
    NORM_HIGH = 17
    NORM_LOW = 18
    NORM_VOLUME = 19
    NORM_VWAP = 20
    LOG_MONEY = 21

def change_to_raw_min(features):
    result = []
    for feature in features:
        if feature in ['$vwap']:
            result.append(f"$money/$volume")
        elif feature in ['$volume']:
            result.append(f"{feature}/100000")
            # result.append('$close')
        elif feature in ['$log_close']:
            result.append(f"$close") # We calculate log later
        elif feature in ['$log_volume']:
            result.append(f"$volume/100000") # We calculate log later
        else:
            result.append(feature)
    return result

def change_to_raw(features):
    result = []
    for feature in features:
        if feature in ['$open','$close','$high','$low','$vwap']:
            result.append(f"{feature}*$factor")
        elif feature in ['$volume']:
            result.append(f"{feature}/$factor/1000000")
            # result.append('$close')
        elif feature in ['$log_close']:
            # For log features, we need the raw value first
            result.append(f"$close*$factor")
        elif feature in ['$log_volume']:
            result.append(f"$volume/$factor/1000000")
        elif feature in [
            '$sortino_ratio',
            '$ts_mom_rank',
            '$max_dd_ratio',
            '$rel_strength_ma',
            '$drift_factor',
            '$amihud_mean',
            '$amihud_range',
        ]:
            # Derived daily factors do not require raw price/volume scaling
            result.append(feature)
        else:
            raise ValueError(f"feature {feature} not supported")
    return result

class StockData:
    _qlib_initialized: bool = False

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: Optional[torch.device] = None,
                 raw:bool = False,
                 qlib_path:Union[str,Dict] = "",
                 freq:str = 'day',
                 region: Optional[str] = None,
                 load_minute_data: bool = False,
                 minutes_per_day: int = 390,
                 minute_qlib_path: Optional[Union[str,Dict]] = None,
                 ) -> None:
        # Auto-detect device if not specified
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._init_qlib(qlib_path, region)
        self.df_bak = None
        self.raw = raw
        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.freq = freq
        self.load_minute_data = load_minute_data
        self.minutes_per_day = minutes_per_day
        self.minute_qlib_path = minute_qlib_path if minute_qlib_path is not None else qlib_path
        self._region = region
        
        # Store daily qlib config for restoration after loading minute data
        self._daily_qlib_path = qlib_path
        self._daily_region = region

        # Load daily data (always loaded)
        self.data, self._dates, self._stock_ids = self._get_data()

        # Load minute data (optional, lazy-loaded)
        self.minute_data: Optional[torch.Tensor] = None
        if load_minute_data:
            self.minute_data = self._get_minute_data()

    @classmethod
    def _init_qlib(cls,qlib_path, region: Optional[str] = None) -> None:
        if cls._qlib_initialized:
            return
        import qlib
        from qlib.config import REG_CN, REG_US
        if isinstance(qlib_path, dict):
            cfg = dict(qlib_path)
        else:
            cfg = {"provider_uri": qlib_path}

        def _resolve_region(region_value: Optional[str]):
            if region_value is None:
                return REG_CN
            return REG_US if region_value.lower() == 'us' else REG_CN

        if "region" in cfg:
            cfg_region = cfg["region"]
            if isinstance(cfg_region, str):
                cfg["region"] = _resolve_region(cfg_region)
        else:
            inferred = None
            if region is not None:
                inferred = region
            elif isinstance(qlib_path, str) and 'us' in qlib_path.lower():
                inferred = 'us'
            cfg["region"] = _resolve_region(inferred)

        print(cfg.get("provider_uri"), cfg["region"])
        qlib.init(**cfg)
        cls._qlib_initialized = True

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        cal: np.ndarray = D.calendar(freq=self.freq)
        start_index = cal.searchsorted(pd.Timestamp(self._start_time))  # type: ignore
        end_index = cal.searchsorted(pd.Timestamp(self._end_time))  # type: ignore
        # Clamp indices to valid calendar range to prevent negative indexing wrap-around
        backtrack_index = max(0, start_index - self.max_backtrack_days)
        real_start_time = cal[backtrack_index]
        # Clamp end_index to valid range before checking calendar value
        if end_index >= len(cal):
            end_index = len(cal) - 1
        elif cal[end_index] != pd.Timestamp(self._end_time):
            end_index -= 1
        future_index = min(len(cal) - 1, end_index + self.max_future_days)
        real_end_time = cal[future_index]
        result =  (QlibDataLoader(config=exprs,freq=self.freq)  # type: ignore
                .load(self._instrument, real_start_time, real_end_time))
        return result

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        # 1. Map requested FeatureType enums to the actual Qlib expression strings
        # We need this mapping to reconstruct the tensor later
        feature_map = [] # List of expression strings corresponding to self._features
        
        for f in self._features:
            if f == FeatureType.LOG_VOLUME:
                # Log(Volume + 1)
                expr = 'Log($volume+1)'
            elif f == FeatureType.LOG_MONEY:
                # Log(Money + 1), where Money approx = Volume * Typical Price
                expr = 'Log($volume*($close+$high+$low)/3+1)'
            elif f == FeatureType.LOG_CLOSE:
                expr = 'Log($close+1)'
                
            else:
                # Default: lowercase enum name (e.g. OPEN -> $open)
                feat_name = '$' + f.name.lower()
                expr = feat_name
            
            # Raw mode or Reverted mode: use the utils to transform names or direct mapping
            # Since we reverted to raw features (OPEN, CLOSE...), direct mapping works
            feature_map.append(expr)
        
        # Determine actual expressions to load
        if self.raw:
             # Legacy logic for raw mode (mostly used in backtesting/dumping)
             # We assume raw requests don't have the LOG/NORM overlap issue usually
             features_to_load = ['$' + f.name.lower() for f in self._features]
             if self.freq == 'day':
                 features_to_load = change_to_raw(features_to_load)
             else:
                 features_to_load = change_to_raw_min(features_to_load)
             feature_map = features_to_load # In raw mode, map is identity 1-to-1 to load list
        
        # 2. Identify UNIQUE expressions to load from Qlib
        # This prevents "ValueError: Columns with duplicate values"
        unique_exprs = sorted(list(set(feature_map)))
        
        # 3. Load Data
        df = self._load_exprs(unique_exprs)
        self.df_bak = df

        # 4. Fill NaNs
        # ffill then bfill for prices (prevents 0.0)
        # fillna(0.0) for anything remaining
        df = df.ffill().bfill().fillna(0.0)

        # 5. Extract dimensions
        # df index is (Time, Instrument)
        dates = df.index.levels[0]
        # In Qlib, unstacking might drop some columns if they are all NaN? No.
        # But we need to be careful about stock_ids.
        # Let's verify by unstacking ONE column to get the stock_ids
        sample_unstack = df.iloc[:, 0].unstack(level=1)
        stock_ids = sample_unstack.columns
        n_dates = len(dates)
        n_stocks = len(stock_ids)
        
        # 6. Construct Tensor
        vals_list = []
        for i, expr in enumerate(feature_map):
            # Select the column corresponding to the feature
            # df[expr] gives a Series with (Time, Instrument) index
            series = df[expr]
            # Unstack to get (Time, Stock)
            feat_df = series.unstack(level=1)
            # Re-ensure fillna just in case unstack introduced NaNs (e.g. missing instrument-time pairs)
            # though ffill/bfill above should have handled it if index was complete.
            # QlibDF index is usually complete for loaded instruments.
            feat_df = feat_df.fillna(0.0) 
            vals_list.append(feat_df.values)
            
        values = np.stack(vals_list, axis=1) # (T, F, N)

        # Verify dimensions
        expected_shape = (n_dates, len(self._features), n_stocks)
        if values.shape != expected_shape:
             # Try to be helpful with error msg
             raise ValueError(f"Shape mismatch: {values.shape} vs {expected_shape}")

        tensor_data = torch.tensor(values, dtype=torch.float, device=self.device)

        # --- Post-Processing Logic Removed (User Request) ---
        # Tensor contains raw OHLCV values now.

        return tensor_data, dates, stock_ids

    def _get_minute_data(self) -> torch.Tensor:
        """
        Load minute-level data and reshape to (n_dates, minutes_per_day, n_features, n_stocks).

        This method loads 1-minute frequency data from qlib and organizes it by trading day.
        Each day should have `minutes_per_day` bars (390 for US markets, 240 for CN markets).

        Returns:
            torch.Tensor: Minute data with shape (n_dates, minutes_per_day, n_features, n_stocks)

        Raises:
            ValueError: If minute data shape doesn't match expected dimensions
        """
        # Filter to base features only for minute data to save memory/time
        # We assume derived features (Sortino etc) are not needed at minute level
        # Define locally to avoid circular imports / path issues with src.alpha_gfn.config
        minute_features = [
            '$' + f.name.lower() for f in [
                FeatureType.OPEN,
                FeatureType.CLOSE,
                FeatureType.HIGH,
                FeatureType.LOW,
                FeatureType.VOLUME,
                FeatureType.VWAP,
            ]
        ]
        
        if self.raw:
            minute_features = change_to_raw_min(minute_features)

        # Reinitialize qlib with minute data path temporarily
        import qlib
        from qlib.config import REG_CN, REG_US
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D

        # Build config for minute data
        if isinstance(self.minute_qlib_path, dict):
            minute_cfg = dict(self.minute_qlib_path)
        else:
            minute_cfg = {"provider_uri": self.minute_qlib_path}

        # Set region
        if "region" not in minute_cfg:
            if self._region is not None:
                minute_cfg["region"] = REG_US if self._region.lower() == 'us' else REG_CN
            elif isinstance(self.minute_qlib_path, str) and 'us' in self.minute_qlib_path.lower():
                minute_cfg["region"] = REG_US
            else:
                minute_cfg["region"] = REG_CN

        # Reinitialize qlib for minute data temporarily
        qlib.init(**minute_cfg)

        try:
            # Get minute calendar
            cal: np.ndarray = D.calendar(freq='1min')
            start_index = cal.searchsorted(pd.Timestamp(self._start_time))
            end_index = cal.searchsorted(pd.Timestamp(self._end_time))

            # Clamp indices to valid calendar range
            backtrack_index = max(0, start_index - self.max_backtrack_days * self.minutes_per_day)
            real_start_time = cal[backtrack_index]

            if end_index >= len(cal):
                end_index = len(cal) - 1
            elif cal[end_index] != pd.Timestamp(self._end_time):
                end_index -= 1

            future_index = min(len(cal) - 1, end_index + self.max_future_days * self.minutes_per_day)
            real_end_time = cal[future_index]

            # Load minute-frequency data from qlib
            df_minute = (QlibDataLoader(config=minute_features, freq='1min')
                        .load(self._instrument, real_start_time, real_end_time))
        finally:
            # CRITICAL: Restore original daily qlib config
            # This ensures subsequent daily data operations work correctly
            if isinstance(self._daily_qlib_path, dict):
                daily_cfg = dict(self._daily_qlib_path)
            else:
                daily_cfg = {"provider_uri": self._daily_qlib_path}
            
            # Set region for daily config
            if "region" not in daily_cfg:
                if self._daily_region is not None:
                    daily_cfg["region"] = REG_US if self._daily_region.lower() == 'us' else REG_CN
                elif isinstance(self._daily_qlib_path, str) and 'us' in self._daily_qlib_path.lower():
                    daily_cfg["region"] = REG_US
                else:
                    daily_cfg["region"] = REG_CN
            
            # Reinitialize qlib back to daily data
            qlib.init(**daily_cfg)

        # Fill NaN values
        df_minute = df_minute.fillna(0.0)

        # Optimized reshaping: (Time, Stock) -> (Time, Stock, Feature)
        # Qlib returns MultiIndex (Time, Stock) with columns as Features
        # We want (Time, Feature, Stock) eventually
        
        # 1. Unstack to move Stock to columns: (Time, Feature * Stock)
        # This is much faster than stack().unstack()
        df_unstacked = df_minute.unstack(level=1, fill_value=0.0)
        
        # 2. Reindex to ensure we have all expected timestamps
        # Construct target minute index based on daily dates
        target_dates = pd.to_datetime(self._dates).normalize().unique()
        target_dates = np.sort(target_dates)
        
        # Create a full minute index for all target days
        # We assume standard trading hours (minutes_per_day) for each day
        # Note: This assumes the minute data in Qlib aligns with this simple generation
        # If Qlib has irregular minutes, we might need a different approach, but for standard US/CN this works
        # However, a safer way is to use the actual minute calendar we got from Qlib, filtered by our target days
        
        # Let's try to just reindex to the range we care about
        # But we need to align exactly to (n_days, minutes_per_day)
        
        # Alternative: Iterate days but use efficient slicing on the unstacked dataframe
        # The unstacked dataframe index is DatetimeIndex (minutes)
        
        n_features = len(minute_features)
        n_stocks = len(self._stock_ids)
        
        # Prepare output array
        n_days = len(target_dates)
        values_per_day = np.zeros((n_days, self.minutes_per_day, n_features, n_stocks), dtype=np.float32)
        
        # Get the stock mapping
        # df_unstacked columns are MultiIndex (Feature, Stock) or just (Stock) if 1 feature?
        # QlibDataLoader returns columns as Features. So unstack(level=1) makes columns (Feature, Stock)
        
        # We need to map df_unstacked columns to our self._stock_ids
        # df_unstacked.columns is (Feature, Stock)
        # Let's reshape df_unstacked to (Time, Feature, Stock_in_Data) first
        
        # It's easier to work with values if we sort columns
        df_unstacked = df_unstacked.sort_index(axis=1)
        
        # The columns are (Feature, Stock). We have n_features.
        # Let's assume the stock order in level 1 matches sorted(stock_ids_minute)
        stock_ids_minute = df_minute.index.get_level_values(1).unique().sort_values()
        
        # Create a mapping from stock_id to index in our target array
        stock_map = {sid: i for i, sid in enumerate(self._stock_ids)}
        
        # Identify which stocks in data are valid and where they go
        valid_stock_indices = [] # indices in data
        target_stock_indices = [] # indices in target
        
        for i, sid in enumerate(stock_ids_minute):
            if sid in stock_map:
                valid_stock_indices.append(i)
                target_stock_indices.append(stock_map[sid])
                
        # Convert to numpy for fast indexing
        # Shape: (TotalMinutes, N_Features * N_Stocks_Data)
        raw_values = df_unstacked.values
        # Reshape to (TotalMinutes, N_Features, N_Stocks_Data)
        raw_values = raw_values.reshape(raw_values.shape[0], n_features, len(stock_ids_minute))
        
        # Select only valid stocks and reorder to match self._stock_ids
        # Shape: (TotalMinutes, N_Features, N_Stocks_Target)
        # Initialize with zeros (for missing stocks)
        aligned_values = np.zeros((raw_values.shape[0], n_features, n_stocks), dtype=np.float32)
        if valid_stock_indices:
            aligned_values[:, :, target_stock_indices] = raw_values[:, :, valid_stock_indices]
            
        # Now align to days
        # We can use searchsorted to find start/end of each day in the index
        minute_index = df_unstacked.index
        
        # Create a 'date' array for the minute index
        minute_dates = minute_index.normalize()
        
        # Find start/end indices for each target day
        # This is still a loop but over days (e.g. 1000), which is fast enough if operations inside are slices
        
        # Optimization: Pre-calculate start/end indices
        # We can use searchsorted on the dates if they are sorted (they should be)
        
        # But minute_dates has repeats. searchsorted works on sorted arrays.
        # It will find the first occurrence.
        
        # Let's iterate target dates and slice
        # To make it faster, we can group the minute_index by date once
        
        # Actually, since we have the minute calendar from Qlib, we can use that?
        # But df_minute might have missing rows.
        
        # Let's stick to the loop over target_dates, it's robust and 1000 iterations is negligible
        # compared to the massive stack/unstack we removed.
        
        for i, date in enumerate(target_dates):
            # Find rows for this date
            # Using searchsorted on dates is tricky because of repeats.
            # But we can just use boolean masking or slice if we know the range.
            # Boolean mask is O(N), might be slow if N is large (1M minutes).
            
            # Better: use slice of the index
            try:
                # loc is reasonably fast on DatetimeIndex
                day_slice = aligned_values[minute_dates == date]
                
                if len(day_slice) >= self.minutes_per_day:
                    values_per_day[i] = day_slice[:self.minutes_per_day]
                elif len(day_slice) > 0:
                    # Partial day, fill what we have
                    values_per_day[i, :len(day_slice)] = day_slice
            except KeyError:
                pass # Missing date, leave as zeros

        return torch.tensor(values_per_day, dtype=torch.float, device=self.device)

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    # =========================================================================
    # Rolling Window Cache for Speedup
    # =========================================================================
    def initialize_rolling_cache(
        self,
        delta_times: List[int] = None,
        stats: Optional[List[str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Precompute common rolling statistics for base features.
        This significantly speeds up expression evaluation by avoiding redundant calculations.

        Args:
            delta_times: List of window sizes to precompute (default: [10, 15, 20, 30, 40, 50])
            stats: Which rolling stats to cache. Options: ['mean','std','min','max'].
                   Default caches only ['mean','std'] to reduce memory footprint.
            dtype: Optional dtype to store cached tensors (e.g., torch.float16) to save memory.
        """
        if delta_times is None:
            delta_times = [10, 15, 20, 30, 40, 50]
        if stats is None:
            stats = ["mean", "std"]
        stats = [s.lower() for s in stats if s]

        if not stats:
            print("[StockData] Rolling cache disabled (no stats requested)")
            return

        if not hasattr(self, "_rolling_cache"):
            self._rolling_cache = {}

        dtype_str = str(dtype).replace("torch.", "") if dtype is not None else str(self.data.dtype).replace("torch.", "")
        print(
            f"[StockData] Precomputing rolling cache for {len(self._features)} features x "
            f"{len(delta_times)} windows (stats={stats}, dtype={dtype_str})..."
        )

        for feat_idx, _feat_type in enumerate(self._features):
            feat_data = self.data[:, feat_idx, :]  # (T, N)

            # Precompute cumulative sums for memory-efficient mean/std.
            need_mean_or_std = ("mean" in stats) or ("std" in stats)
            cumsum_pad = None
            cumsum_sq_pad = None
            if need_mean_or_std:
                cumsum = feat_data.cumsum(dim=0)
                zero = torch.zeros((1,) + cumsum.shape[1:], device=feat_data.device, dtype=feat_data.dtype)
                cumsum_pad = torch.cat([zero, cumsum], dim=0)  # (T+1, N)
                if "std" in stats:
                    cumsum_sq = (feat_data ** 2).cumsum(dim=0)
                    cumsum_sq_pad = torch.cat([zero, cumsum_sq], dim=0)

            for dt in delta_times:
                if dt > feat_data.shape[0]:
                    continue

                # Rolling mean/std via cumsum (forward-looking windows)
                sum_dt = None
                if need_mean_or_std and cumsum_pad is not None:
                    sum_dt = cumsum_pad[dt:] - cumsum_pad[:-dt]  # (T-dt+1, N)

                    if "mean" in stats:
                        mean_dt = sum_dt / float(dt)
                        if dtype is not None:
                            mean_dt = mean_dt.to(dtype)
                        self._rolling_cache[(feat_idx, dt, "mean")] = mean_dt

                    if "std" in stats:
                        sum_sq_dt = cumsum_sq_pad[dt:] - cumsum_sq_pad[:-dt]  # type: ignore
                        mean_dt = sum_dt / float(dt)
                        # Unbiased variance to match torch.std default unbiased=True
                        if dt > 1:
                            var_dt = (sum_sq_dt - (sum_dt ** 2) / float(dt)) / float(dt - 1)
                        else:
                            var_dt = torch.zeros_like(mean_dt)
                        std_dt = var_dt.clamp_min(0.0).sqrt()
                        if dtype is not None:
                            std_dt = std_dt.to(dtype)
                        self._rolling_cache[(feat_idx, dt, "std")] = std_dt

                # Rolling min/max if requested (use unfold view; outputs are cached)
                if ("min" in stats) or ("max" in stats):
                    windows = feat_data.unfold(0, dt, 1)  # (T-dt+1, N, dt) view
                    if "min" in stats:
                        min_dt = windows.min(dim=-1)[0]
                        if dtype is not None:
                            min_dt = min_dt.to(dtype)
                        self._rolling_cache[(feat_idx, dt, "min")] = min_dt
                    if "max" in stats:
                        max_dt = windows.max(dim=-1)[0]
                        if dtype is not None:
                            max_dt = max_dt.to(dtype)
                        self._rolling_cache[(feat_idx, dt, "max")] = max_dt
            
            # Aggressive cleanup of intermediates to prevent OOM
            del feat_data
            if cumsum_pad is not None: del cumsum_pad
            if cumsum_sq_pad is not None: del cumsum_sq_pad
            
            # Explicit GC
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                try: torch.mps.empty_cache()
                except: pass

        print(f"[StockData] Rolling cache initialized with {len(self._rolling_cache)} entries")

    def get_cached_rolling(self, feat_idx: int, delta_time: int, stat: str, period: slice = None) -> Optional[torch.Tensor]:
        """
        Retrieve precomputed rolling statistic from cache.

        Args:
            feat_idx: Feature index
            delta_time: Rolling window size
            stat: Statistic type ('mean', 'std', 'min', 'max')
            period: Optional slice for time period

        Returns:
            Cached tensor or None if not available
        """
        if not hasattr(self, '_rolling_cache'):
            return None

        cache_key = (feat_idx, delta_time, stat)
        cached = self._rolling_cache.get(cache_key)

        if cached is None:
            return None

        if period is not None:
            # FIX: Adjust for backward-looking windows
            # The cache stores windows[i] = data[i:i+dt] (forward-looking)
            # But RollingOperator.evaluate() computes backward-looking windows:
            #   result[j] = data[j-dt+1:j+1] for output index j
            # To align: cached[i] corresponds to data ending at index (i + dt - 1)
            # For output day d (raw index = max_backtrack_days + d), we need
            # the window ending at that index, which is cached[max_backtrack_days + d - dt + 1]
            offset = delta_time - 1
            start = (period.start if period.start is not None else 0) + self.max_backtrack_days - offset
            stop = (period.stop if period.stop is not None else 1) + self.max_backtrack_days + self.n_days - 1 - offset

            # Ensure we don't exceed bounds
            start = max(0, min(start, cached.shape[0]))
            stop = max(start, min(stop, cached.shape[0]))

            return cached[start:stop]

        return cached

    def clear_rolling_cache(self) -> None:
        """Clear the rolling cache to free memory."""
        if hasattr(self, '_rolling_cache'):
            self._rolling_cache.clear()
            print("[StockData] Rolling cache cleared")

    def verify_cache_correctness(self, delta_times: List[int] = None, rtol: float = 1e-3, atol: float = 1e-5) -> bool:
        """
        Verify rolling cache produces same results as standard evaluation.

        This is critical for ensuring no data leakage - the cache must return
        backward-looking windows that match RollingOperator.evaluate().

        Note: Small numerical differences are expected because the cache uses cumsum-based
        computation while standard evaluation uses direct unfold+mean. Data leakage would
        manifest as HUGE differences (completely different values), not small precision errors.

        Args:
            delta_times: Window sizes to verify (default: [10, 20, 40])
            rtol: Relative tolerance (default 0.1% = 1e-3)
            atol: Absolute tolerance for values near zero (default 1e-5)

        Returns:
            True if cache is correct, raises AssertionError otherwise
        """
        from alphagen.data.expression import TsMean, TsStd, TsMax, TsMin, Feature

        if delta_times is None:
            delta_times = [10, 20, 40]

        if not hasattr(self, '_rolling_cache') or not self._rolling_cache:
            print("[StockData] No cache to verify")
            return True

        print(f"[StockData] Verifying cache correctness for delta_times={delta_times}...")

        # Test with first feature (CLOSE is typical)
        test_feat = self._features[0]
        feat = Feature(test_feat)

        for dt in delta_times:
            if dt not in [k[1] for k in self._rolling_cache.keys()]:
                continue

            # Save cache state
            saved_cache = self._rolling_cache.copy()

            # Clear cache to force standard evaluation
            self._rolling_cache.clear()

            # Standard evaluation (no cache)
            ts_mean = TsMean(feat, dt)
            standard_result = ts_mean.evaluate(self)

            # Restore cache
            self._rolling_cache = saved_cache

            # Cached evaluation
            cached_result = ts_mean.evaluate(self)

            # Compare using relative + absolute tolerance (like torch.allclose)
            # Data leakage would cause massive differences (e.g., 100x), not small precision errors
            diff = (standard_result - cached_result).abs()
            tolerance = atol + rtol * standard_result.abs()
            within_tol = diff <= tolerance

            if not within_tol.all():
                # Find worst violation
                rel_diff = diff / (standard_result.abs() + atol)
                worst_idx = rel_diff.argmax().item()
                day_idx = worst_idx // standard_result.shape[1]
                stock_idx = worst_idx % standard_result.shape[1]
                max_rel_diff = rel_diff[day_idx, stock_idx].item()
                abs_diff = diff[day_idx, stock_idx].item()

                raise AssertionError(
                    f"Cache mismatch for TsMean(dt={dt}): rel_diff={max_rel_diff:.6f} ({max_rel_diff*100:.4f}%) at "
                    f"day={day_idx}, stock={stock_idx}\n"
                    f"  Standard: {standard_result[day_idx, stock_idx].item():.6f}\n"
                    f"  Cached:   {cached_result[day_idx, stock_idx].item():.6f}\n"
                    f"  Abs diff: {abs_diff:.6f}\n"
                    f"  Note: If rel_diff > 10%, this indicates data leakage (wrong window direction).\n"
                    f"        Small diffs (<1%) are numerical precision differences (expected)."
                )

        print(f"[StockData] Cache verification PASSED for all delta_times")
        return True

    def add_data(self,data:torch.Tensor,dates:pd.Index):
        data = data.to(self.device)
        self.data = torch.cat([self.data,data],dim=0)
        self._dates = pd.Index(self._dates.append(dates))


    def make_dataframe(
        self,
        data: Union[torch.Tensor, List[torch.Tensor]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
            Parameters:
            - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or
            a list of tensors of size `(n_days, n_stocks)`
            - `columns`: an optional list of column names
            """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
