import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution
import torch

from src.alphagen.data.tree import ExpressionParser
from src.alphagen_qlib.stock_data import StockData
from src.alpha_gfn.alpha_pool import AlphaPoolGFN


def _top_bottom_long_short_returns(factor: torch.Tensor, returns: torch.Tensor, pct: float = 0.2) -> np.ndarray:
    """
    Build a simple equal-weight long-short return series from factor cross-sections.
    factor, returns: (T, N)
    """
    factor_np = factor.detach().cpu().numpy()
    ret_np = returns.detach().cpu().numpy()
    n_days, n_stocks = factor_np.shape
    k = max(1, int(n_stocks * pct))
    ls_ret: List[float] = []
    for t in range(n_days):
        f = factor_np[t]
        r = ret_np[t]
        mask = np.isfinite(f) & np.isfinite(r)
        if mask.sum() < k * 2:
            continue
        f_sel = f[mask]
        r_sel = r[mask]
        order = np.argsort(f_sel)
        bottom = order[:k]
        top = order[-k:]
        top_ret = np.mean(r_sel[top])
        bottom_ret = np.mean(r_sel[bottom])
        ls_ret.append(top_ret - bottom_ret)
    return np.array(ls_ret, dtype=float)


def _sortino_ratio(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    mean = returns.mean()
    downside = returns[returns < 0]
    if downside.size == 0:
        return float("inf")
    downside_std = downside.std()
    if downside_std < 1e-9:
        return 0.0
    return mean / downside_std


def _calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if returns.size == 0:
        return 0.0
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    max_dd = np.min(drawdown)
    years = max(1e-6, returns.size / periods_per_year)
    cagr = (cum[-1]) ** (1 / years) - 1
    if max_dd == 0:
        return 0.0
    return cagr / abs(max_dd)


class ExpressionDETuner:
    """
    Lightweight DE tuner that perturbs numeric constants in-place for a single expression.
    """

    def __init__(
        self,
        pool: AlphaPoolGFN,
        stock_data: StockData,
        output_dir: str,
        popsize: int = 5,
        maxiter: int = 5,
        metric_pct: float = 0.2,
    ):
        self.pool = pool
        self.data = stock_data
        self.output_dir = output_dir
        self.popsize = popsize
        self.maxiter = maxiter
        self.metric_pct = metric_pct
        os.makedirs(self.output_dir, exist_ok=True)
        self.parser = ExpressionParser()

    def _extract_constants(self, expr_str: str) -> Tuple[str, List[float]]:
        pattern = r'(?<![A-Za-z_\$])(-?\d+(?:\.\d+)?)'
        constants = [float(m.group(0)) for m in re.finditer(pattern, expr_str)]
        template = re.sub(pattern, lambda m, c=[0]: f"{{{c.__setitem__(0, c[0]+1) or c[0]-1}}}", expr_str)
        return template, constants

    def _metric_bundle(self, expr_str: str) -> Optional[Dict[str, float]]:
        try:
            expr = self.parser.parse(expr_str)
            factor = expr.evaluate(self.data)
            returns = self.pool.raw_target_returns
            if returns is None:
                return None
            # Align shapes: returns: (T, N), factor maybe (T, N)
            ls_ret = _top_bottom_long_short_returns(self.pool._normalize_by_day(factor), returns, pct=self.metric_pct)
            if ls_ret.size == 0:
                return None
            mean_excess = float(ls_ret.mean())
            sortino = float(_sortino_ratio(ls_ret))
            calmar = float(_calmar_ratio(ls_ret))
            return {"mean_excess": mean_excess, "sortino": sortino, "calmar": calmar}
        except Exception:
            return None

    def _build_bounds(self, constants: List[float]) -> List[Tuple[float, float]]:
        bounds = []
        for c in constants:
            span = max(1.0, abs(c) * 2.0)
            bounds.append((c - span, c + span))
        return bounds

    def tune(self, expr_str: str) -> Optional[Dict[str, str]]:
        template, consts = self._extract_constants(expr_str)
        if not consts:
            return None

        baseline = self._metric_bundle(expr_str)
        if baseline is None:
            return None

        bounds = self._build_bounds(consts)

        def objective(params: List[float]) -> float:
            candidate = template.format(*params)
            metrics = self._metric_bundle(candidate)
            if metrics is None:
                return 1e6
            # Guard: all metrics must improve
            if (
                metrics["mean_excess"] <= baseline["mean_excess"]
                or metrics["sortino"] <= baseline["sortino"]
                or metrics["calmar"] <= baseline["calmar"]
            ):
                return 1e3
            score = metrics["mean_excess"] + metrics["sortino"] + metrics["calmar"]
            return -score

        result = differential_evolution(
            objective,
            bounds=bounds,
            popsize=self.popsize,
            maxiter=self.maxiter,
            recombination=0.7,
            polish=False,
            disp=False,
        )

        best_params = result.x
        best_expr = template.format(*best_params)
        best_metrics = self._metric_bundle(best_expr)
        if best_metrics is None:
            return None
        if (
            best_metrics["mean_excess"] <= baseline["mean_excess"]
            or best_metrics["sortino"] <= baseline["sortino"]
            or best_metrics["calmar"] <= baseline["calmar"]
        ):
            return None

        record = {
            "original_expr": expr_str,
            "tuned_expr": best_expr,
            "baseline": baseline,
            "tuned": best_metrics,
            "params": [float(p) for p in best_params],
        }
        self._write_record(record)
        return record

    def _write_record(self, record: Dict) -> None:
        path = os.path.join(self.output_dir, "tuned_records.json")
        data: List = []
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append(record)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
