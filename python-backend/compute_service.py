
import sys
import os
import json
import traceback
from pathlib import Path
import logging
import io
import re

# Add src to python path to allow imports from AlphaSAGE/src
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT / "src"))

# Suppress all logging to stdout (Qlib logs)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)
logging.root.addHandler(stderr_handler)

# Capture and discard any print statements during imports
_stdout_capture = io.StringIO()
_original_stdout = sys.stdout
sys.stdout = _stdout_capture

import torch
import pandas as pd
import numpy as np

try:
    from alphagen_qlib.stock_data import StockData
    # Import commonly used operators for eval()
    from alphagen.data.expression import *
    # Import feature variables (close, volume, vwap, etc.) for eval() to work
    from alphagen_generic.features import *
    from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret, batch_sharpe_ratio, batch_max_drawdown
    from gan.utils.builder import exprs2tensor
except ImportError as e:
    sys.stdout = _original_stdout
    print(json.dumps({"error": f"Failed to import modules: {str(e)}", "traceback": traceback.format_exc()}))
    sys.exit(1)

# Restore stdout after imports
sys.stdout = _original_stdout

# Global cache for StockData to avoid reloading if the process stays alive (though unlikely in one-shot spawn)
_data_cache = {}

def export_factor_values_csv(path: Path,
                             factor_tensor: torch.Tensor,
                             dates,
                             stock_ids,
                             factor_ids: list[int]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    values = factor_tensor.detach().cpu().numpy().astype(np.float32)  # (T, N, F)
    t_count, n_count, f_count = values.shape
    if len(dates) != t_count or len(stock_ids) != n_count:
        raise RuntimeError(
            f"Factor export shape mismatch: tensor {values.shape}, dates {len(dates)}, stocks {len(stock_ids)}"
        )
    if len(factor_ids) != f_count:
        factor_ids = list(range(f_count))
    date_strings = [d.strftime("%Y-%m-%d") for d in dates]
    factor_cols = [f"factor_{fid}" for fid in factor_ids]
    index = pd.MultiIndex.from_product([date_strings, list(stock_ids)], names=["date", "ticker"])
    df = pd.DataFrame(values.reshape(t_count * n_count, f_count), index=index, columns=factor_cols)
    df.to_csv(path)


def get_stock_data(start_date: str, end_date: str, qlib_path: str) -> StockData:
    cache_key = f"{start_date}_{end_date}"
    if cache_key not in _data_cache:
        # Suppress print statements during StockData initialization (Qlib logs)
        _temp_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            _data_cache[cache_key] = StockData(
                instrument='all',  # Use 'all' instrument (available in instruments/all.txt)
                start_time=start_date,
                end_time=end_date,
                max_backtrack_days=100,
                max_future_days=30,  # Sufficient for 20d horizon
                qlib_path=qlib_path,
                region='us'  # US market data
            )
        finally:
            sys.stdout = _temp_stdout
    return _data_cache[cache_key]


def _fix_tsquantile_args(expr: str) -> str:
    """Insert a default window if TsQuantile is missing the 3rd argument."""
    needle = "TsQuantile("
    idx = 0
    while True:
        start = expr.find(needle, idx)
        if start == -1:
            break
        i = start + len(needle)
        depth = 0
        while i < len(expr):
            ch = expr[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    inner = expr[start + len(needle):i]
                    # Count top-level commas
                    commas = 0
                    d2 = 0
                    for c in inner:
                        if c == "(":
                            d2 += 1
                        elif c == ")":
                            d2 -= 1
                        elif c == "," and d2 == 0:
                            commas += 1
                    if commas == 1:
                        expr = expr[:i] + ",20)" + expr[i + 1:]
                        i += len(",20)") - 1
                    break
                else:
                    depth -= 1
            i += 1
        idx = start + len(needle)
    return expr


def _clean_expression(expr_str: str) -> str:
    """
    Normalize feature tokens so eval() can resolve them.
    - $open -> open_
    - Preserve existing open_ tokens
    - Strip remaining '$'
    """
    expr = expr_str.strip()
    expr = re.sub(r"\$open_", "open_", expr)
    expr = re.sub(r"\$open\b", "open_", expr)
    expr = re.sub(r"\$", "", expr)
    expr = _fix_tsquantile_args(expr)
    return expr


def load_pool(pool_path: str) -> tuple[list, list[str]]:
    """Load pool JSON and return (expr tuples, parse warnings)."""
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    with open(pool_path, 'r') as f:
        data = json.load(f)

    exprs_raw = data.get('exprs', [])
    # weights = data.get('weights', []) # Not using weights for now, evaluating individual factors

    parse_warnings: list[str] = []
    # Pre-process expressions (replace tokens if needed)
    exprs = []
    for idx, expr_str in enumerate(exprs_raw):
        clean_expr = _clean_expression(expr_str)
        try:
            # Evaluate string to Expression object
            expr_obj = eval(clean_expr)
            exprs.append((expr_str, expr_obj))
        except Exception as e:
            # Second-chance: if TsQuantile was malformed, patch with default window and retry
            retry_expr = _fix_tsquantile_args(clean_expr)
            try:
                expr_obj = eval(retry_expr)
                exprs.append((expr_str, expr_obj))
                parse_warnings.append(f"Auto-fixed TsQuantile window for expression #{idx}")
            except Exception as e2:
                warning = f"Failed to parse expression #{idx}: {expr_str}. Error: {e2}"
                parse_warnings.append(warning)
                print(f"Stderr: {warning}", file=sys.stderr)

    return exprs, parse_warnings


def build_rebalanced_returns(signal: torch.Tensor,
                             target_returns: torch.Tensor,
                             top_pct: float,
                             strat_type: str,
                             rebalance_days: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Turn a daily signal into portfolio returns while respecting rebalance cadence.
    Returns (daily portfolio return, rebalance_id per day).
    """
    pct = float(max(min(top_pct, 0.5), 0.01))
    step = max(int(rebalance_days), 1)

    # Rank signals once and reuse for each window
    ranks = signal.argsort(dim=1).argsort(dim=1).float()
    n_stocks = signal.shape[1]

    port_ret = torch.zeros(signal.shape[0], device=signal.device)
    rebalance_ids = torch.zeros(signal.shape[0], dtype=torch.long, device=signal.device)

    rebalance_idx = 0
    for start in range(0, signal.shape[0], step):
        end = min(start + step, signal.shape[0])

        # 1. Determine Weights at the START of the window
        row_ranks = ranks[start]
        long_mask = row_ranks >= (n_stocks * (1 - pct))
        short_mask = row_ranks < (n_stocks * pct)

        weights = torch.zeros(n_stocks, device=signal.device)
        long_count = long_mask.sum().clamp(min=1).float()
        short_count = short_mask.sum().clamp(min=1).float()

        weights[long_mask] = 1.0 / long_count
        if strat_type == 'long_short':
            weights[short_mask] = -1.0 / short_count
        elif strat_type == 'equal_weight':
            weights[:] = 1.0 / n_stocks
        
        # 2. Daily Returns with Buy-and-Hold (weight drift)
        # Cap daily returns at 50% to prevent data anomally / split errors from exploding
        window_rets = torch.nan_to_num(target_returns[start:end], nan=0.0).clamp(min=-0.9, max=0.5)
        
        # NAV_t = Sum( w_i * Prod(1 + r_i) )
        # Using cumulative product of (1 + r) to drift weights
        stock_cum_factors = (1.0 + window_rets).cumprod(dim=0)
        
        # Portfolio NAV (assuming 1.0 start)
        # If L/S (sum weights = 0), NAV starts at 1.0 and adds PnL
        # If Long (sum weights = 1), NAV starts at 1.0 and follows market
        portfolio_nav = 1.0 + (weights.unsqueeze(0) * (stock_cum_factors - 1.0)).sum(dim=1)
        
        # Derive daily returns from NAV
        prev_nav = torch.cat([torch.tensor([1.0], device=signal.device), portfolio_nav[:-1]])
        daily_rets = portfolio_nav / prev_nav - 1.0
        
        # Store
        port_ret[start:end] = daily_rets
        rebalance_ids[start:end] = rebalance_idx
        rebalance_idx += 1

    return port_ret, rebalance_ids

def compute_metrics(request):
    try:
        # 1. Setup Paths
        qlib_data_path = PROJECT_ROOT / "data" / "1555_qlib"
        if not qlib_data_path.exists():
            raise RuntimeError(f"Qlib data directory not found at {qlib_data_path}")

        # 2. Parse Request
        pool_id = request.get('pool_id', 'pool_100000.json')

        def resolve_pool_file(pid: str) -> str:
            if os.path.exists(pid):
                return pid
            candidates = list((PROJECT_ROOT / "data").rglob(pid))
            if not candidates:
                raise FileNotFoundError(f"Could not locate pool file: {pid}")
            if len(candidates) == 1:
                return str(candidates[0])
            # Choose the richest pool (most expressions, then largest file)
            best_path = None
            best_len = -1
            best_size = -1
            for c in candidates:
                try:
                    data = json.load(open(c))
                    expr_len = len(data.get("exprs", []))
                except Exception:
                    expr_len = -1
                size = os.path.getsize(c)
                if (expr_len, size) > (best_len, best_size):
                    best_len, best_size, best_path = expr_len, size, c
            return str(best_path or candidates[0])

        pool_file = resolve_pool_file(pool_id)

        factor_ids = request.get('factor_ids') # list of indices or None for all
        train_start = request.get('train_start', '2022-01-01')
        train_end = request.get('train_end', '2023-12-31')
        test_start = request.get('test_start', '2024-01-01')
        test_end = request.get('test_end', '2024-12-31')
        target_horizons = request.get('target_horizons', [20])
        factor_id_offset = request.get('factor_id_offset')
        offset_val = None

        # Skip heavy analytics for strategy-only mode (faster)
        skip_analytics = request.get('skip_analytics', False)
        combined_signal_output = request.get('combined_signal_output')
        export_factor_values_csv_path = request.get('export_factor_values_csv')

        parse_warnings: list[str] = []

        # 3. Load Data
        # We need data covering train_start to test_end
        # Plus backtracking and future days
        # StockData handles start/end.
        # However, to support train AND test separately, we can load one big chunk or two.
        # One big chunk is more efficient if they are contiguous.
        # Let's load comprehensive range.
        
        full_start = train_start
        full_end = test_end
        
        data = get_stock_data(full_start, full_end, str(qlib_data_path))
        device = data.device

        # 4. Load Factors
        all_exprs, parse_warnings = load_pool(pool_file)

        if len(all_exprs) == 0:
            return {
                "error": f"No valid factors parsed from {pool_id}",
                "parse_errors": parse_warnings
            }

        # Filter factors if requested
        if factor_ids:
            selected_exprs = [all_exprs[i] for i in factor_ids if i < len(all_exprs)]
            selected_factor_ids = [fid for fid in factor_ids if fid < len(all_exprs)]
        else:
            selected_exprs = all_exprs
            selected_factor_ids = list(range(len(selected_exprs)))

        if factor_id_offset is not None:
            try:
                offset_val = int(factor_id_offset)
            except (TypeError, ValueError):
                offset_val = 0
            output_factor_ids = [offset_val + fid for fid in selected_factor_ids]
        else:
            output_factor_ids = list(selected_factor_ids)

        if len(selected_exprs) == 0:
            return {
                "error": f"Requested factor_ids {factor_ids} not found in pool {pool_id}",
                "parse_errors": parse_warnings
            }

        # 5. Evaluate Factors (Convert to Tensor)
        # exprs2tensor expects list of Expression objects
        expr_objects = [e[1] for e in selected_exprs]
        expr_strings = [e[0] for e in selected_exprs]
        
        # This returns tensor of shape (Time, Stocks, Factors) ? No, (Time, Stocks, Factors) usually
        # Check builder.py: return torch.stack(values, dim=-1) -> (T, N, F)
        # Note: exprs2tensor arguments: (exprs, data)
        factor_tensor = exprs2tensor(expr_objects, data) # (T, N, F)
        
        # 6. Compute Targets for each Horizon
        # We need Targets: Ref(Close, -d)/Close - 1
        targets = {}
        feature_close = Feature(FeatureType.CLOSE) # used for target calc
        close_feat = feature_close # alias used below
        # IMPORTANT: targets are *returns* and must NOT be normalized by day.
        # exprs2tensor(normalize=True) would turn returns into z-scores and
        # blow up strategy/benchmark performance.
        for h in target_horizons:
            target_expr = Ref(close_feat, -h) / close_feat - 1
            # exprs2tensor handles single expr? Yes, list.
            target_t = exprs2tensor([target_expr], data, normalize=False)  # (T, N, 1)
            targets[h] = target_t.squeeze(-1)  # (T, N)

        # 7. Compute Metrics
        metrics_results = []
        daily_returns = []

        # Define split indices
        # Note: data._dates includes backtrack and future days, but factor tensors only have n_days entries
        # Get the proper date slice that matches the factor tensor
        if data.max_future_days == 0:
            dates = data._dates[data.max_backtrack_days:]
        else:
            dates = data._dates[data.max_backtrack_days:-data.max_future_days]

        train_mask = (dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))
        test_mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))

        exported_factor_values_csv = None
        if export_factor_values_csv_path:
            export_path = Path(export_factor_values_csv_path)
            if not export_path.is_absolute():
                export_path = PROJECT_ROOT / export_path
            export_factor_values_csv(
                export_path,
                factor_tensor,
                dates,
                data._stock_ids,
                output_factor_ids
            )
            exported_factor_values_csv = str(export_path)
        
        # Helper to compute for a specific period mask
        def calc_period_metrics(period_name, mask, f_tensor_slice, f_name, f_id, h_days, t_slice):
            if not mask.any(): return None
            
            # Slice tensors
            f_slice = f_tensor_slice[mask]
            
            # Compute stats
            ic = batch_pearsonr(f_slice, t_slice).mean().item()
            ic_std = batch_pearsonr(f_slice, t_slice).std().item()
            ric = batch_spearmanr(f_slice, t_slice).mean().item()
            
            # Returns
            ret = batch_ret(f_slice, t_slice).mean().item() 
            
            sharpe = batch_sharpe_ratio(batch_ret(f_slice, t_slice)).item()
            mdd = batch_max_drawdown(batch_ret(f_slice, t_slice)).item()
            
            # Annualize based on horizon (e.g. if 20-day return, multiply by 242/20)
            annual_scale = 242.0 / max(h_days, 1)
            
            return {
               "factor_id": f_id,
               "factor_name": f_name,
               "horizon_days": h_days,
               "period": period_name,
               "ic": ic,
               "ic_std": ic_std,
               "icir": ic / (ic_std + 1e-9),
               "ric": ric,
               "sharpe": sharpe,
               "annual_return": ret * annual_scale,
               "daily_return_mean": ret / max(h_days, 1),
               "max_drawdown": mdd
            }

        # Metrics for Individual Factors
        for i in range(len(selected_exprs)):
            fid = output_factor_ids[i] if i < len(output_factor_ids) else i
            for h in target_horizons:
                targets_h = targets[h]
                train_target_slice = targets_h[train_mask, :]
                test_target_slice = targets_h[test_mask, :]
                
                # Train Metrics
                m_train = calc_period_metrics("train", train_mask, factor_tensor[:, :, i], expr_strings[i], fid, h, train_target_slice)
                if m_train: metrics_results.append(m_train)
                
                # Test Metrics
                m_test = calc_period_metrics("test", test_mask, factor_tensor[:, :, i], expr_strings[i], fid, h, test_target_slice)
                if m_test: metrics_results.append(m_test)
                
                # Daily Returns (Train)
                if train_mask.any():
                    f_slice = factor_tensor[train_mask, :, i]
                    t_slice = targets_h[train_mask, :]
                    daily_rets = batch_ret(f_slice, t_slice)
                    train_dates = dates[train_mask]
                    for d_idx, ret_val in enumerate(daily_rets.cpu().numpy()):
                         daily_returns.append({
                            "date": train_dates[d_idx].strftime('%Y-%m-%d'),
                            "factor_id": fid,
                            "horizon_days": h,
                            "return": float(ret_val)
                        })

                # Daily Returns (Test)
                if test_mask.any():
                    f_slice = factor_tensor[test_mask, :, i]
                    t_slice = targets_h[test_mask, :]
                    daily_rets = batch_ret(f_slice, t_slice)
                    test_dates = dates[test_mask]
                    for d_idx, ret_val in enumerate(daily_rets.cpu().numpy()):
                         daily_returns.append({
                            "date": test_dates[d_idx].strftime('%Y-%m-%d'),
                            "factor_id": fid,
                            "horizon_days": h,
                            "return": float(ret_val)
                        })

        # Combined Strategy Metrics
        # Portfolio backtest for 1+ factors (single factor uses that factor as the signal).
        if len(selected_exprs) >= 1:
             strategy_cfg = request.get('strategy', {})
             strat_type = strategy_cfg.get('type', 'long_short')
             top_pct = strategy_cfg.get('top_pct', 0.2)
             rebalance_days = strategy_cfg.get('rebalance_days', 1)

             # Calculate Combined Signal (Equal Weight of Z-Scores)
             batch_mean = factor_tensor.mean(dim=1, keepdim=True)
             batch_std = factor_tensor.std(dim=1, keepdim=True)
             z_scores = (factor_tensor - batch_mean) / (batch_std + 1e-9)
             combined_signal = z_scores.mean(dim=2) # (T, N)

             # Ensure 1d targets exist for daily equity curve
             if 1 not in targets:
                 target_1d_expr = Ref(feature_close, -1) / feature_close - 1
                 targets[1] = exprs2tensor([target_1d_expr], data, normalize=False).squeeze(-1)
             target_1d_t = targets[1]

             if combined_signal_output:
                 try:
                     os.makedirs(os.path.dirname(combined_signal_output), exist_ok=True)
                     np.savez(
                         combined_signal_output,
                         signal=combined_signal.detach().cpu().numpy().astype(np.float32),
                         target_returns=target_1d_t.detach().cpu().numpy().astype(np.float32),
                         dates=np.array([d.strftime('%Y-%m-%d') for d in dates])
                     )
                 except Exception as e:
                     raise RuntimeError(f"Failed to save combined signal to {combined_signal_output}: {e}") from e

             # Debug: Check target returns distribution
             valid_target = target_1d_t[~torch.isnan(target_1d_t)]
             print(f"[DEBUG] Target returns - mean: {valid_target.mean().item():.6f}, std: {valid_target.std().item():.6f}", file=sys.stderr)
             print(f"[DEBUG] Target returns - min: {valid_target.min().item():.6f}, max: {valid_target.max().item():.6f}", file=sys.stderr)
             print(f"[DEBUG] Target returns shape: {target_1d_t.shape}, n_stocks: {target_1d_t.shape[1]}", file=sys.stderr)

             # Build portfolio with specified rebalance cadence
             port_ret, rebalance_ids = build_rebalanced_returns(
                 combined_signal, target_1d_t, top_pct, strat_type, rebalance_days
             )

             # Compute Equal-Weight Buy-and-Hold Benchmark (TRUE buy-and-hold, NO rebalancing)
             n_stocks = target_1d_t.shape[1]  # Number of stocks
             benchmark_ret = torch.zeros(target_1d_t.shape[0], device=device)

             # Initial equal weights at t=0 for all valid stocks
             valid_at_start = ~torch.isnan(target_1d_t[0])
             v_count = valid_at_start.sum().clamp(min=1).float()
             init_weights = torch.zeros(n_stocks, device=device)
             init_weights[valid_at_start] = 1.0 / v_count

             # Clean returns: replace NaN with 0, clamp extreme values
             clean_returns = torch.nan_to_num(target_1d_t, nan=0.0).clamp(min=-0.9, max=0.5)

             # Compute cumulative return factor for each stock: (1+r1)*(1+r2)*...
             stock_cum_factors = (1.0 + clean_returns).cumprod(dim=0)

             # Portfolio NAV over time (buy-and-hold, weights drift naturally)
             # NAV_t = sum(initial_weight_i * cumulative_factor_i)
             portfolio_nav = (init_weights.unsqueeze(0) * stock_cum_factors).sum(dim=1)

             # Compute daily returns from NAV
             prev_nav = torch.cat([torch.tensor([1.0], device=device), portfolio_nav[:-1]])
             benchmark_ret = portfolio_nav / prev_nav - 1.0

             # Debug logging for benchmark
             print(f"[DEBUG] Benchmark NAV range: [{portfolio_nav.min().item():.4f}, {portfolio_nav.max().item():.4f}]", file=sys.stderr)
             print(f"[DEBUG] Benchmark return range: [{benchmark_ret.min().item():.6f}, {benchmark_ret.max().item():.6f}]", file=sys.stderr)
             print(f"[DEBUG] Benchmark mean daily return: {benchmark_ret.mean().item():.6f}", file=sys.stderr)
             print(f"[DEBUG] Strategy return range: [{port_ret.min().item():.6f}, {port_ret.max().item():.6f}]", file=sys.stderr)
             print(f"[DEBUG] Strategy mean daily return: {port_ret.mean().item():.6f}", file=sys.stderr)

             for h in target_horizons:
                 targets_h = targets[h]

                 train_ret_slice = port_ret[train_mask]
                 test_ret_slice = port_ret[test_mask]

                 def calc_daily_based_metrics(period_name, ret_slice, f_id):
                     if ret_slice.numel() == 0: return None

                     avg_ret = ret_slice.mean().item()
                     std_ret = ret_slice.std().item()
                     sharpe = (avg_ret / (std_ret + 1e-9)) * (242 ** 0.5)

                     # Max Drawdown from cumulative
                     cum = (1 + ret_slice).cumprod(dim=0)
                     run_max = cum.cummax(dim=0)[0]
                     dd = (cum - run_max) / run_max
                     mdd = dd.min().item()

                     # IC of signal vs horizon target
                     mask = train_mask if period_name == 'train' else test_mask
                     sig_slice = combined_signal[mask]
                     tgt_slice = targets_h[mask]
                     ic = batch_pearsonr(sig_slice, tgt_slice).mean().item()
                     ic_std = batch_pearsonr(sig_slice, tgt_slice).std().item()

                     return {
                        "factor_id": f_id,
                        "factor_name": f"Strategy ({strat_type})",
                        "horizon_days": h,
                        "period": period_name,
                        "ic": ic,
                        "ic_std": ic_std,
                        "icir": ic / (ic_std + 1e-9),
                        "ric": 0, # Skip for combined
                        "sharpe": sharpe,
                        "annual_return": avg_ret * 252,
                        "daily_return_mean": avg_ret,
                        "max_drawdown": mdd
                     }

                 m_train = calc_daily_based_metrics("train", train_ret_slice, -1)
                 if m_train: metrics_results.append(m_train)

                 m_test = calc_daily_based_metrics("test", test_ret_slice, -1)
                 if m_test: metrics_results.append(m_test)

                 # Append Daily Returns for Train Period (if requested/available)
                 if train_mask.any():
                    train_dates = dates[train_mask]
                    train_rets_np = train_ret_slice.cpu().numpy()
                    train_reb_ids = rebalance_ids[train_mask].cpu().numpy()
                    train_bench_rets = benchmark_ret[train_mask].cpu().numpy()
                    for d_idx, ret_val in enumerate(train_rets_np):
                        daily_returns.append({
                           "date": train_dates[d_idx].strftime('%Y-%m-%d'),
                           "factor_id": -1, # STRATEGY ID
                           "horizon_days": h,
                           "return": float(ret_val),
                           "benchmark": float(train_bench_rets[d_idx]),
                           "rebalance_id": int(train_reb_ids[d_idx])
                       })

                 # Append Daily Returns for Test Period (used for Chart)
                 if test_mask.any():
                    test_dates = dates[test_mask]
                    test_rets_np = test_ret_slice.cpu().numpy()
                    test_reb_ids = rebalance_ids[test_mask].cpu().numpy()
                    test_bench_rets = benchmark_ret[test_mask].cpu().numpy()

                    for d_idx, ret_val in enumerate(test_rets_np):
                        daily_returns.append({
                           "date": test_dates[d_idx].strftime('%Y-%m-%d'),
                           "factor_id": -1, # STRATEGY ID
                           "horizon_days": h,
                           "return": float(ret_val),
                           "benchmark": float(test_bench_rets[d_idx]),
                           "rebalance_id": int(test_reb_ids[d_idx])
                       })

        # --- EXTENDED ANALYTICS ---
        # Skip if skip_analytics=True (for faster strategy-only computation)

        corr_matrix = []
        vif_scores = []
        rolling_ic_data = []
        grid_search_results = []
        factor_dictionary = []

        # Helper: Generate factor name
        def generate_factor_name(expr_str: str, idx: int) -> str:
            ops = re.findall(r'(Ts\w+|Rank|Log|Abs|Ret)', expr_str)
            features = re.findall(r'\$(\w+)', expr_str)
            if features and ops:
                return f"{features[0]}_{ops[0].lower()}_{idx}"
            elif features:
                return f"{features[0]}_{idx}"
            return f"factor_{idx}"

        # Helper: Classify factor type
        def classify_factor_type(expr_str: str) -> str:
            expr_lower = expr_str.lower()
            if any(t in expr_lower for t in ['tsret', 'tsmomrank', 'ret', 'momentum']): return 'momentum'
            if any(t in expr_lower for t in ['volume', 'log_volume']): return 'volume'
            if any(t in expr_lower for t in ['tsstd', 'tsvar', 'tsskew', 'tskurt']): return 'volatility'
            if any(t in expr_lower for t in ['tscorr', 'corr']): return 'correlation'
            if any(t in expr_lower for t in ['rank', 'quantile']): return 'mean_reversion'
            return 'composite'

        if not skip_analytics:
            # 1. Correlation Matrix
            # Flatten (T, N, F) -> (T*N, F)
            T, N, F = factor_tensor.shape
            flat = factor_tensor.reshape(-1, F)
            valid_mask = ~torch.isnan(flat).any(dim=1)
            flat_clean = flat[valid_mask]
            if flat_clean.shape[0] > 0:
                 corr = torch.corrcoef(flat_clean.T)
                 corr = torch.nan_to_num(corr, nan=0.0)
                 corr_matrix = corr.cpu().tolist()

            # 2. VIF Scores
            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                flat_np = flat_clean.cpu().numpy()
                # Subsample if too large
                if len(flat_np) > 10000:
                    indices = np.random.choice(len(flat_np), 10000, replace=False)
                    flat_np = flat_np[indices]

                for i in range(F):
                    try:
                        vif = variance_inflation_factor(flat_np, i)
                        vif = min(vif, 1000.0)
                    except:
                        vif = 0.0
                    vif_scores.append({
                        "factor_id": selected_factor_ids[i] if i < len(selected_factor_ids) else i,
                        "vif": float(vif),
                        "multicollinear": bool(vif > 10.0)
                    })
            except ImportError:
                print("statsmodels not installed, skipping VIF", file=sys.stderr)

            # 3. Rolling IC (Test Period, Primary Horizon)
            primary_horizon = target_horizons[0]
            targets_h = targets[primary_horizon]
            if test_mask.any():
                test_dates_list = [d.strftime('%Y-%m-%d') for d in dates[test_mask]]
                for i in range(len(selected_exprs)):
                    # batch_pearsonr returns (T,)
                    daily_ic = batch_pearsonr(factor_tensor[test_mask, :, i], targets_h[test_mask, :])
                    for d_idx, ic_val in enumerate(daily_ic.cpu().numpy()):
                        rolling_ic_data.append({
                            "date": test_dates_list[d_idx],
                            "factor_id": selected_factor_ids[i] if i < len(selected_factor_ids) else i,
                            "ic": float(ic_val)
                        })

            # 4. Greedy Grid Search (Forward Selection)
            if test_mask.any():
                max_factors = min(10, len(selected_exprs))
                flat_test = factor_tensor[test_mask].reshape(-1, F)
                flat_target = targets_h[test_mask].reshape(-1)
                valid = ~(torch.isnan(flat_test).any(dim=1) | torch.isnan(flat_target))
                X = flat_test[valid].cpu().numpy()
                y = flat_target[valid].cpu().numpy()

                selected = []
                remaining = list(range(F))

                for n in range(1, max_factors + 1):
                    best_r2 = -np.inf
                    best_idx = None

                    for idx in remaining:
                        trial = selected + [idx]
                        X_trial = X[:, trial]
                        # Add constant
                        X_const = np.column_stack([np.ones(len(X_trial)), X_trial])
                        try:
                            coef, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
                            y_pred = X_const @ coef
                            ss_res = np.sum((y - y_pred) ** 2)
                            ss_tot = np.sum((y - y.mean()) ** 2)
                            r2 = 1 - ss_res / ss_tot
                        except:
                            r2 = 0.0

                        if r2 > best_r2:
                            best_r2 = r2
                            best_idx = idx

                    if best_idx is not None:
                        selected.append(best_idx)
                        remaining.remove(best_idx)
                        real_ids = [selected_factor_ids[j] if j < len(selected_factor_ids) else j for j in selected]
                        grid_search_results.append({
                            "n_factors": n,
                            "r_squared": float(best_r2),
                            "selected_factors": real_ids
                        })

            # 5. Factor Dictionary
            for i, (expr_str, _) in enumerate(selected_exprs):
                fid = output_factor_ids[i] if i < len(output_factor_ids) else i
                test_metrics = [m for m in metrics_results if m['factor_id'] == fid and m['period'] == 'test' and m['horizon_days'] == primary_horizon]
                tm = test_metrics[0] if test_metrics else {}

                factor_dictionary.append({
                    "factor_id": fid,
                    "name": generate_factor_name(expr_str, fid),
                    "expression": expr_str,
                    "type": classify_factor_type(expr_str),
                    "description": "", # User requested empty description
                    "ic": tm.get('ic', 0),
                    "icir": tm.get('icir', 0),
                    "sharpe": tm.get('sharpe', 0),
                    "mdd": tm.get('max_drawdown', 0),
                    "source": os.path.basename(pool_file)
                })

        response = {
            "metrics": metrics_results,
            "daily_returns": daily_returns,
            "correlation_matrix": corr_matrix,
            "vif_scores": vif_scores,
            "rolling_ic": rolling_ic_data,
            "grid_search_results": grid_search_results,
            "factor_dictionary": factor_dictionary,
            "parse_warnings": parse_warnings,
            "pool_path": pool_file,
            "computation_time_ms": 0 # Placeholder, can measure wrap
        }
        if offset_val is not None:
            response["factor_id_offset"] = offset_val
        if exported_factor_values_csv:
            response["factor_values_csv"] = exported_factor_values_csv
        return response

    except Exception as e:
        # Include full stack trace in error
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    # Read from stdin
    try:
        if sys.stdin.isatty():
             # Debug mode if running manually without pipe
             print("Waiting for JSON input on stdin...", file=sys.stderr)
             
        input_data = sys.stdin.read()
        if not input_data.strip():
            # If empty, maybe run a default test?
            # Or just exit
            request = {}
        else:
            request = json.loads(input_data)
        
        import time
        t0 = time.time()
        
        result = compute_metrics(request)
        
        result['computation_time_ms'] = (time.time() - t0) * 1000
        
        # Print JSON to stdout
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": f"Fatal execution error: {str(e)}", "traceback": traceback.format_exc()}))
        sys.exit(1)
