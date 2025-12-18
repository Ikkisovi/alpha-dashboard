"""
Lightweight strategy computation service.
Only computes portfolio returns given:
- Pre-computed factor signals (from parent compute_metrics call)
- Close prices (loaded directly from binary files)
- Strategy parameters (top_pct, strat_type, rebalance_days)

This is ~10x faster than full compute_service.py
"""
import sys
import os
import json
import struct
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def load_close_prices(qlib_path: str, start_date: str, end_date: str) -> tuple:
    """
    Load close prices directly from binary files (bypassing Qlib).
    Returns: (dates, stock_ids, close_prices as numpy array)
    """
    import pandas as pd

    qlib_dir = Path(qlib_path)

    # 1. Load calendar
    calendar_file = qlib_dir / "calendars" / "day.txt"
    with open(calendar_file, 'r') as f:
        all_dates = [pd.Timestamp(line.strip()) for line in f if line.strip()]

    # Filter to date range with padding
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Find indices
    start_idx = next((i for i, d in enumerate(all_dates) if d >= start_ts), 0)
    end_idx = next((i for i, d in enumerate(all_dates) if d > end_ts), len(all_dates))

    # Add padding for backtrack/future
    backtrack = 100
    future = 30
    start_idx = max(0, start_idx - backtrack)
    end_idx = min(len(all_dates), end_idx + future)

    dates = all_dates[start_idx:end_idx]
    n_days = len(dates)

    # 2. Load instrument list
    instruments_file = qlib_dir / "instruments" / "all.txt"
    stock_ids = []
    with open(instruments_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                stock_ids.append(parts[0].lower())

    n_stocks = len(stock_ids)

    # 3. Load close prices from binary files
    close_prices = np.full((n_days, n_stocks), np.nan, dtype=np.float32)

    features_dir = qlib_dir / "features"
    for s_idx, stock_id in enumerate(stock_ids):
        close_file = features_dir / stock_id / "close.day.bin"
        if close_file.exists():
            with open(close_file, 'rb') as f:
                # Qlib binary format: float32 array
                data = np.fromfile(f, dtype='<f')
                # Slice to our date range
                if len(data) >= end_idx:
                    close_prices[:, s_idx] = data[start_idx:end_idx]
                elif len(data) > start_idx:
                    available = min(len(data) - start_idx, n_days)
                    close_prices[:available, s_idx] = data[start_idx:start_idx + available]

    return dates, stock_ids, close_prices


def compute_strategy_returns(
    close_prices: np.ndarray,
    factor_signals: np.ndarray,
    dates: list,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    top_pct: float = 0.2,
    strat_type: str = "long_short",
    rebalance_days: int = 1
) -> dict:
    """
    Compute portfolio returns given factor signals and close prices.

    Args:
        close_prices: (T, N) array of close prices
        factor_signals: (T, N) array of factor signals (combined z-score)
        dates: list of Timestamps
        train_start/end, test_start/end: date strings
        top_pct: fraction of stocks to go long/short
        strat_type: 'long_short' or 'long_only'
        rebalance_days: rebalancing frequency

    Returns:
        dict with daily returns, metrics, etc.
    """
    import pandas as pd

    dates = pd.to_datetime(dates)
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    # Compute daily stock returns: r_t = close_t / close_{t-1} - 1
    stock_returns = np.zeros_like(close_prices)
    stock_returns[1:] = close_prices[1:] / np.maximum(close_prices[:-1], 1e-8) - 1
    stock_returns = np.clip(stock_returns, -0.9, 0.5)  # Cap extreme returns
    stock_returns = np.nan_to_num(stock_returns, nan=0.0)

    T, N = close_prices.shape
    pct = max(min(top_pct, 0.5), 0.01)
    step = max(int(rebalance_days), 1)

    # Portfolio returns
    port_ret = np.zeros(T)
    benchmark_ret = np.zeros(T)
    rebalance_ids = np.zeros(T, dtype=int)

    # Rank signals
    ranks = np.argsort(np.argsort(factor_signals, axis=1), axis=1).astype(float)

    rebalance_idx = 0
    for start in range(0, T, step):
        end = min(start + step, T)

        # 1. Determine Weights at START of window
        row_ranks = ranks[start]
        long_mask = row_ranks >= (N * (1 - pct))
        short_mask = row_ranks < (N * pct)

        weights = np.zeros(N)
        long_count = max(long_mask.sum(), 1)
        short_count = max(short_mask.sum(), 1)

        weights[long_mask] = 1.0 / long_count
        if strat_type == 'long_short':
            weights[short_mask] = -1.0 / short_count
        elif strat_type == 'equal_weight':
            weights[:] = 1.0 / N

        # 2. Compute returns with buy-and-hold drift
        window_rets = stock_returns[start:end]
        stock_cum_factors = np.cumprod(1.0 + window_rets, axis=0)

        # Portfolio NAV
        portfolio_nav = 1.0 + np.sum(weights * (stock_cum_factors - 1.0), axis=1)

        # Daily returns from NAV
        prev_nav = np.concatenate([[1.0], portfolio_nav[:-1]])
        daily_rets = portfolio_nav / prev_nav - 1.0

        port_ret[start:end] = daily_rets
        rebalance_ids[start:end] = rebalance_idx
        rebalance_idx += 1

    # Benchmark: Equal-weight buy-and-hold
    valid_mask = ~np.isnan(close_prices[0])
    v_count = max(valid_mask.sum(), 1)
    init_weights = np.zeros(N)
    init_weights[valid_mask] = 1.0 / v_count

    bench_cum = np.cumprod(1.0 + stock_returns, axis=0)
    bench_nav = np.sum(init_weights * bench_cum, axis=1)
    prev_bench = np.concatenate([[1.0], bench_nav[:-1]])
    benchmark_ret = bench_nav / prev_bench - 1.0

    # Split by period
    train_mask = (dates >= train_start_ts) & (dates <= train_end_ts)
    test_mask = (dates >= test_start_ts) & (dates <= test_end_ts)

    # For dashboard, we need to offset dates to match factor tensor indexing
    # The factor tensor excludes backtrack/future days
    backtrack = 100
    future = 30
    if len(dates) > backtrack + future:
        factor_dates = dates[backtrack:-future] if future > 0 else dates[backtrack:]
        factor_port_ret = port_ret[backtrack:-future] if future > 0 else port_ret[backtrack:]
        factor_bench_ret = benchmark_ret[backtrack:-future] if future > 0 else benchmark_ret[backtrack:]
        factor_reb_ids = rebalance_ids[backtrack:-future] if future > 0 else rebalance_ids[backtrack:]

        train_mask = (factor_dates >= train_start_ts) & (factor_dates <= train_end_ts)
        test_mask = (factor_dates >= test_start_ts) & (factor_dates <= test_end_ts)
    else:
        factor_dates = dates
        factor_port_ret = port_ret
        factor_bench_ret = benchmark_ret
        factor_reb_ids = rebalance_ids

    # Build daily returns output
    daily_returns = []

    def calc_metrics(ret_slice, period_name):
        if len(ret_slice) == 0:
            return None
        avg_ret = np.mean(ret_slice)
        std_ret = np.std(ret_slice)
        sharpe = (avg_ret / (std_ret + 1e-9)) * np.sqrt(252)

        cum = np.cumprod(1 + ret_slice)
        run_max = np.maximum.accumulate(cum)
        dd = (cum - run_max) / run_max
        mdd = np.min(dd)

        return {
            "factor_id": -1,
            "factor_name": f"Strategy ({strat_type})",
            "horizon_days": 1,
            "period": period_name,
            "ic": 0,
            "ic_std": 0,
            "icir": 0,
            "ric": 0,
            "sharpe": float(sharpe),
            "annual_return": float(avg_ret * 252),
            "daily_return_mean": float(avg_ret),
            "max_drawdown": float(mdd)
        }

    metrics = []

    # Train metrics
    if train_mask.any():
        train_ret = factor_port_ret[train_mask]
        m = calc_metrics(train_ret, "train")
        if m:
            metrics.append(m)

        train_dates = factor_dates[train_mask]
        train_bench = factor_bench_ret[train_mask]
        train_reb = factor_reb_ids[train_mask]
        for i, (d, r, b, rid) in enumerate(zip(train_dates, train_ret, train_bench, train_reb)):
            daily_returns.append({
                "date": d.strftime('%Y-%m-%d'),
                "factor_id": -1,
                "horizon_days": 1,
                "return": float(r),
                "benchmark": float(b),
                "rebalance_id": int(rid)
            })

    # Test metrics
    if test_mask.any():
        test_ret = factor_port_ret[test_mask]
        m = calc_metrics(test_ret, "test")
        if m:
            metrics.append(m)

        test_dates = factor_dates[test_mask]
        test_bench = factor_bench_ret[test_mask]
        test_reb = factor_reb_ids[test_mask]
        for i, (d, r, b, rid) in enumerate(zip(test_dates, test_ret, test_bench, test_reb)):
            daily_returns.append({
                "date": d.strftime('%Y-%m-%d'),
                "factor_id": -1,
                "horizon_days": 1,
                "return": float(r),
                "benchmark": float(b),
                "rebalance_id": int(rid)
            })

    return {
        "metrics": metrics,
        "daily_returns": daily_returns,
        "computation_time_ms": 0
    }


def main():
    """Read request from stdin, compute strategy returns, output JSON."""
    import time

    try:
        input_data = sys.stdin.read()
        if not input_data.strip():
            request = {}
        else:
            request = json.loads(input_data)

        t0 = time.time()

        # Parse request
        qlib_path = str(PROJECT_ROOT / "data" / "1555_qlib")
        train_start = request.get('train_start', '2022-01-01')
        train_end = request.get('train_end', '2023-12-31')
        test_start = request.get('test_start', '2024-01-01')
        test_end = request.get('test_end', '2024-12-31')

        strategy_cfg = request.get('strategy', {})
        strat_type = strategy_cfg.get('type', 'long_short')
        top_pct = strategy_cfg.get('top_pct', 0.2)
        rebalance_days = strategy_cfg.get('rebalance_days', 1)

        # Get factor signals from request (pre-computed by parent call)
        factor_signals = request.get('factor_signals')

        # Load close prices (fast - direct binary read)
        dates, stock_ids, close_prices = load_close_prices(
            qlib_path, train_start, test_end
        )

        if factor_signals is None:
            # If no factor signals provided, use equal weight (benchmark)
            factor_signals = np.ones_like(close_prices)
        else:
            factor_signals = np.array(factor_signals)

        # Ensure shapes match
        T, N = close_prices.shape
        if factor_signals.shape != (T, N):
            # Try to reshape or use simple signal
            print(f"Warning: factor_signals shape {factor_signals.shape} != close {(T, N)}", file=sys.stderr)
            factor_signals = np.ones_like(close_prices)

        # Compute strategy returns
        result = compute_strategy_returns(
            close_prices=close_prices,
            factor_signals=factor_signals,
            dates=dates,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            top_pct=top_pct,
            strat_type=strat_type,
            rebalance_days=rebalance_days
        )

        result['computation_time_ms'] = (time.time() - t0) * 1000

        print(json.dumps(result))

    except Exception as e:
        import traceback
        print(json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()
