#!/usr/bin/env python3
"""
Compute custom factors for the legacy universe and dump them into Qlib.

Custom factors:
- sortino_ratio
- ts_mom_rank
- max_dd_ratio
- rel_strength_ma

The formulas match the existing feature files:
- Rolling windows are on closing prices.
- Z-scores / percentiles use a 252-day window with warmup.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from loguru import logger

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
# Ensure project root is on the path for absolute imports
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.qlib_dump_bin import DumpDataBase

FEATURE_COLUMNS = ["sortino_ratio", "ts_mom_rank", "max_dd_ratio", "rel_strength_ma"]


def _rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def _pct(x: pd.Series) -> float:
        x = x.dropna()
        if x.empty:
            return np.nan
        # rank relative to the window
        return (rankdata(x)[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5

    # rolling apply is slow, but necessary for exact rank in window
    return series.rolling(window=window, min_periods=min_periods).apply(_pct, raw=False)


def rankdata(a):
    """Compute rank of values in array 1-based."""
    if len(a) == 0:
        return np.array([])
    return pd.Series(a).rank().values


def compute_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily rolling custom features (no drift factor - that's calculated from intraday data).
    
    Args:
        daily_df: DataFrame with 'close' and 'volume'.
    """
    close = daily_df["close"].astype(float)

    # Sortino ratio: 20-day avg return / 20-day downside std, raw (no z-score), needs ~20 bars.
    returns = close.pct_change()
    avg_ret = returns.rolling(window=20).mean()
    downside_std = returns.clip(upper=0).rolling(window=20).std()
    sortino_ratio = avg_ret / (downside_std + 1e-8)  # Raw

    # TS mom rank: momentum over horizons 5/10/20 averaged,
    # then 20-day rolling percentile rank of that average (min 20 bars).
    m5 = close / close.shift(5) - 1
    m10 = close / close.shift(10) - 1
    m20 = close / close.shift(20) - 1
    mom_avg = (m5 + m10 + m20) / 3
    
    # Rolling TsRank (percentile of the latest value within the window)
    ts_mom_rank = _rolling_percentile(mom_avg, window=20, min_periods=20)

    # Max drawdown ratio: 20-day rolling high vs close -> (close / roll_max - 1), raw drawdown.
    roll_max = close.rolling(window=20).max()
    max_dd_ratio = close / roll_max - 1

    # Rel strength MA: short/long MAs (10/20) -> short/long - 1, raw.
    ma10 = close.rolling(window=10).mean()
    ma20 = close.rolling(window=20).mean()
    rel_strength_ma = ma10 / (ma20 + 1e-8) - 1

    return pd.DataFrame(
        {
            "sortino_ratio": sortino_ratio,
            "ts_mom_rank": ts_mom_rank,
            "max_dd_ratio": max_dd_ratio,
            "rel_strength_ma": rel_strength_ma,
        }
    )


def _load_instruments(path: Path) -> List[str]:
    df = pd.read_csv(path, sep="\t", header=None, names=["symbol", "start", "end"], dtype=str).fillna("")
    return df["symbol"].str.strip().str.upper().tolist()


class FeatureBinWriter(DumpDataBase):
    """Thin wrapper to reuse DumpDataBase bin dumping."""

    def dump(self):  # pragma: no cover - not used
        raise NotImplementedError


def dump_bins(
    features_csv_dir: Path,
    qlib_dir: Path,
    symbols: Iterable[str],
    calendar_path: Path,
    max_workers: int,
) -> None:
    dumper = FeatureBinWriter(
        csv_path=str(features_csv_dir),
        qlib_dir=str(qlib_dir),
        freq="day",
        max_workers=max_workers,
        date_field_name="date",
        file_suffix=".csv",
        symbol_field_name="symbol",
        include_fields=",".join(FEATURE_COLUMNS),
    )
    calendars = dumper._read_calendars(calendar_path)

    for symbol in symbols:
        csv_path = features_csv_dir / f"{symbol.lower()}.csv"
        if not csv_path.exists():
            logger.warning(f"Feature CSV missing for {symbol}; skipping dump")
            continue
        df = pd.read_csv(csv_path, parse_dates=["date"])
        dumper._dump_bin(df, calendars)
        logger.info(f"Dumped features for {symbol}")


def process_symbol(symbol: str, raw_csv_dir: Path, features_csv_dir: Path) -> bool:
    raw_path = raw_csv_dir / f"{symbol.lower()}.csv"
    if not raw_path.exists():
        logger.error(f"Raw CSV missing for {symbol} at {raw_path}")
        return False

    df = pd.read_csv(raw_path, parse_dates=["date"])
    if df.empty:
        logger.warning(f"No data for {symbol}")
        return False

    df = df.sort_values("date")
    # compute_features now takes the full daily_df to access 'volume'
    features = compute_features(df)
    out_df = pd.concat(
        [
            pd.Series(symbol, index=features.index, name="symbol"),
            df["date"].dt.strftime("%Y-%m-%d"),
            features,
        ],
        axis=1,
    )

    features_csv_dir.mkdir(parents=True, exist_ok=True)
    out_path = features_csv_dir / f"{symbol.lower()}.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved features CSV for {symbol} to {out_path} ({len(out_df)} rows)")
    return True


def run_custom_features(
    instrument_file: Path,
    qlib_dir: Path,
    raw_csv_dir: Path | None = None,
    features_csv_dir: Path | None = None,
    max_workers: int = 8,
) -> None:
    """End-to-end computation and dumping of custom features."""
    raw_csv_dir = raw_csv_dir or qlib_dir / "raw_csv"
    features_csv_dir = features_csv_dir or qlib_dir / "features_csv"
    calendar_path = qlib_dir / "calendars" / "day.txt"

    symbols = _load_instruments(instrument_file)
    logger.info(f"Loaded {len(symbols)} symbols from {instrument_file}")

    success = 0
    for sym in symbols:
        if process_symbol(sym, raw_csv_dir, features_csv_dir):
            success += 1
    logger.info(f"Computed features for {success}/{len(symbols)} symbols")

    dump_bins(features_csv_dir, qlib_dir, symbols, calendar_path, max_workers)
    logger.success(f"Finished dumping custom features to {qlib_dir / 'features'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute legacy custom features and dump to Qlib.")
    parser.add_argument("--instrument_file", type=Path, default=Path("data/spy100_qlib/instruments/legacy.txt"))
    parser.add_argument("--qlib_dir", type=Path, default=Path("data/spy100_qlib"))
    parser.add_argument("--raw_csv_dir", type=Path, default=None, help="Defaults to <qlib_dir>/raw_csv")
    parser.add_argument("--features_csv_dir", type=Path, default=None, help="Defaults to <qlib_dir>/features_csv")
    parser.add_argument("--max_workers", type=int, default=8)
    args = parser.parse_args()

    run_custom_features(
        instrument_file=args.instrument_file,
        qlib_dir=args.qlib_dir,
        raw_csv_dir=args.raw_csv_dir,
        features_csv_dir=args.features_csv_dir,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
