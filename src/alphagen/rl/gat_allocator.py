import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.nn import GATConv
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch_geometric is required for the GAT allocator") from exc

from alphagen.data.tree import ExpressionParser
from alphagen.data.expression import Expression, Feature, Ref
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen_qlib.stock_data import FeatureType, StockData


def _ensure_device(t: Tensor, device: torch.device) -> Tensor:
    return t if t.device == device else t.to(device)


def _daily_ic(factors: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """
    factors: (n_stocks, n_factors)
    target:  (n_stocks,)
    """
    factors = factors - factors.mean(dim=0, keepdim=True)
    target = target - target.mean()
    cov = (factors * target.unsqueeze(-1)).mean(dim=0)
    std_f = torch.clamp(factors.std(dim=0), min=eps)
    std_t = torch.clamp(target.std(), min=eps)
    return cov / (std_f * std_t)


def _corr_edges(flat_factors: Tensor, threshold: float) -> Tuple[Tensor, Tensor]:
    """
    flat_factors: (n_samples, n_factors) collapsed over time x assets
    """
    if flat_factors.ndim != 2:
        raise ValueError(f"flat_factors must be 2D, got {flat_factors.shape}")
    corr = torch.corrcoef(flat_factors.T).clamp(-1.0, 1.0)
    mask = (corr.abs() > threshold) & ~torch.eye(corr.size(0), dtype=torch.bool, device=corr.device)
    if not torch.any(mask):
        # Fallback to fully connected with zero weights to keep GAT happy
        num = corr.size(0)
        edge_index = torch.stack(torch.meshgrid(torch.arange(num, device=corr.device),
                                                torch.arange(num, device=corr.device),
                                                indexing="ij"), dim=0).reshape(2, -1)
        edge_weight = corr.reshape(-1)
    else:
        edge_index = mask.nonzero(as_tuple=False).T.contiguous()
        edge_weight = corr[mask]
    return edge_index, edge_weight


def _long_short_return(signal: Tensor, future_ret: Tensor, temp: float = 5.0) -> Tensor:
    """
    Differentiable long-short return using softmax over positive/negative signals.
    signal: (n_stocks,)
    future_ret: (n_stocks,)
    """
    long_w = torch.softmax(signal * temp, dim=0)
    short_w = torch.softmax(-signal * temp, dim=0)
    return (long_w * future_ret).sum() - (short_w * future_ret).sum()


def _long_short_hold_return(signal: Tensor, future_rets: Tensor, temp: float = 5.0) -> Tensor:
    """
    Multi-day variant that fixes long/short weights on rebalance day and holds
    through the provided window of future returns.
    signal: (n_stocks,)
    future_rets: (horizon, n_stocks)
    """
    long_w = torch.softmax(signal * temp, dim=0)
    short_w = torch.softmax(-signal * temp, dim=0)
    pnl = torch.tensor(0.0, device=signal.device)
    for day_ret in future_rets:
        pnl = pnl + (long_w * day_ret).sum() - (short_w * day_ret).sum()
    return pnl


@dataclass
class FactorState:
    node_features: Tensor
    edge_index: Tensor
    regime_features: Tensor
    edge_weight: Optional[Tensor] = None


class FactorStateBuilder:
    """
    Builds per-timestep factor graph state with rolling IC/turnover stats and
    correlation-based edges.
    """

    def __init__(
        self,
        factors_norm: Tensor,      # (T, N, F)
        target_ret: Tensor,        # (T, N)
        state_window: int = 20,
        corr_threshold: float = 0.3,
    ):
        self.factors_norm = factors_norm
        self.target_ret = target_ret
        self.state_window = state_window
        self.corr_threshold = corr_threshold

    def build(self, t: int, regime_feat: Tensor) -> FactorState:
        """
        Build state ending at time t (uses returns at t+1 for IC targets).
        """
        start = max(0, t - self.state_window + 1)
        fac_hist = self.factors_norm[start : t + 1]  # (L, N, F)
        tgt_hist = self.target_ret[start + 1 : t + 2]  # align next-day returns

        curr_fac = self.factors_norm[t]          # (N, F)
        curr_target = self.target_ret[t + 1]     # (N,)

        # Rolling IC and IC-IR
        ics = []
        for i in range(fac_hist.shape[0] - 1):
            ics.append(_daily_ic(fac_hist[i], tgt_hist[i]))
        if len(ics) == 0:
            ic_mean = torch.zeros(curr_fac.size(1), device=curr_fac.device)
            ic_ir = torch.zeros_like(ic_mean)
        else:
            ic_stack = torch.stack(ics, dim=0)                  # (L-1, F)
            ic_mean = ic_stack.mean(dim=0)
            # Use unbiased=False to avoid NaN when there's only 1 element
            ic_std = ic_stack.std(dim=0, unbiased=False) if ic_stack.shape[0] > 1 else torch.ones_like(ic_mean) * 1e-4
            ic_ir = ic_stack.mean(dim=0) / (ic_std + 1e-4)

        # Turnover (day-over-day rank change)
        if t == 0:
            turnover = torch.zeros(curr_fac.size(1), device=curr_fac.device)
        else:
            prev_rank = torch.argsort(torch.argsort(self.factors_norm[t - 1], dim=0), dim=0).float()
            curr_rank = torch.argsort(torch.argsort(curr_fac, dim=0), dim=0).float()
            turnover = (curr_rank - prev_rank).abs().mean(dim=0)

        # Current IC against next-day returns
        curr_ic = _daily_ic(curr_fac, curr_target)
        vol = curr_fac.std(dim=0, unbiased=False)

        node_features = torch.stack([curr_ic, ic_mean, ic_ir, turnover, vol], dim=1)  # (F, 5)
        # Ensure no NaN values in node features
        node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Edge construction from rolling correlation
        flat = fac_hist.reshape(-1, fac_hist.size(-1))
        edge_index, edge_weight = _corr_edges(flat, self.corr_threshold)

        return FactorState(
            node_features=node_features,
            edge_index=edge_index,
            regime_features=regime_feat,
            edge_weight=edge_weight,
        )


class RegimeFeatureBuilder:
    """
    Builds regime descriptors from universe returns and optional VIX/SPY/VTV series.
    """

    def __init__(
        self,
        target_ret: Tensor,                  # (T, N)
        vix_series: Optional[Tensor] = None, # (T,)
        spy_series: Optional[Tensor] = None, # (T,)
        vtv_series: Optional[Tensor] = None, # (T,)
        window: int = 20,
    ):
        self.target_ret = target_ret
        self.vix_series = vix_series
        self.spy_series = spy_series
        self.vtv_series = vtv_series
        self.window = window
        self.eps = 1e-6

        def _prep_returns(series: Optional[Tensor]) -> Optional[Tensor]:
            if series is None or series.numel() < 2:
                return None
            log_ret = torch.log(series[1:] / (series[:-1] + self.eps))
            return torch.cat([log_ret.new_zeros(1), log_ret])

        self.spy_returns = _prep_returns(spy_series)
        self.vtv_returns = _prep_returns(vtv_series)

        if self.spy_series is not None and self.vtv_series is not None:
            ratio = torch.log(self.vtv_series / (self.spy_series + self.eps))
            self.vtv_spy_ratio = ratio
        else:
            self.vtv_spy_ratio = None

    def build(self, t: int) -> Tensor:
        start = max(0, t - self.window + 1)
        market = self.target_ret[start:t+1].mean(dim=1)  # (L,)
        vol = market.std(unbiased=False) if market.numel() > 0 else torch.tensor(0.0, device=market.device)
        dispersion = self.target_ret[t].std(unbiased=False)
        breadth = (self.target_ret[t] > 0).float().mean()

        feats = [vol, dispersion, breadth]
        if self.vix_series is not None:
            feats.append(self.vix_series[min(t, self.vix_series.shape[0]-1)])

        def _win_std(series: Optional[Tensor]) -> Tensor:
            if series is None or series.numel() == 0:
                return torch.tensor(0.0, device=self.target_ret.device)
            return series[start:t+1].std(unbiased=False)

        def _safe_get(series: Optional[Tensor]) -> Tensor:
            if series is None:
                return torch.tensor(0.0, device=self.target_ret.device)
            idx = min(t, series.shape[0]-1)
            return series[idx]

        spy_ret = _safe_get(self.spy_returns)
        vtv_ret = _safe_get(self.vtv_returns)
        ret_mean = self.target_ret[t].mean()

        spy_vol = _win_std(self.spy_returns)
        vtv_vol = _win_std(self.vtv_returns)

        feats.extend([
            spy_vol,
            spy_vol - vol,
            spy_ret,
            spy_ret - ret_mean,
            vtv_vol,
            vtv_vol - vol,
            vtv_ret,
            vtv_ret - ret_mean,
        ])

        if self.vtv_spy_ratio is not None:
            ratio_win = self.vtv_spy_ratio[start:t+1]
            ratio_z = (ratio_win[-1] - ratio_win.mean()) / (ratio_win.std(unbiased=False) + self.eps)
            feats.append(ratio_z)

        regime = torch.stack(feats)
        regime = torch.nan_to_num(regime, nan=0.0, posinf=0.0, neginf=0.0)
        return regime


class FactorGATPolicy(nn.Module):
    """
    Graph attention policy producing factor weights conditioned on regime features.
    """

    def __init__(
        self,
        node_dim: int,
        regime_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_norm = nn.LayerNorm(node_dim)
        self.regime_norm = nn.LayerNorm(regime_dim) if regime_dim > 0 else None
        self.gat1 = GATConv(node_dim + regime_dim, hidden_dim, heads=heads,
                            concat=True, dropout=dropout, add_self_loops=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1,
                            concat=False, dropout=dropout, add_self_loops=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        regime_features: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        n = node_features.size(0)
        h = self.node_norm(node_features)
        if regime_features is not None and regime_features.numel() > 0:
            r = regime_features
            if self.regime_norm is not None:
                r = self.regime_norm(r)
            r = r.unsqueeze(0).expand(n, -1)
            h = torch.cat([h, r], dim=-1)

        # edge_weight is currently unused because GATConv does not accept it;
        # correlation strength is implicitly handled via learned attention.
        h = F.elu(self.gat1(h, edge_index))
        h = F.elu(self.gat2(h, edge_index))
        logits = self.out(h).squeeze(-1)
        weights = torch.softmax(logits, dim=0)
        return weights, logits


def load_pool_expressions(pool_path: Path, parser: ExpressionParser, limit: Optional[int] = None) -> List[Expression]:
    pool = json.loads(pool_path.read_text())
    expr_strs = pool.get("exprs", pool)  # tolerate raw list
    if limit is not None:
        expr_strs = expr_strs[:limit]
    return [parser.parse(e) for e in expr_strs]


def evaluate_factors(
    expressions: List[Expression],
    data: StockData,
) -> Tensor:
    """
    Returns normalized factor tensor with shape (T, N, F)
    """
    values: List[Tensor] = []
    min_len = None
    for expr in expressions:
        v = expr.evaluate(data)
        if min_len is None:
            min_len = v.shape[0]
        min_len = min(min_len, v.shape[0])
        values.append(v)
    assert min_len is not None
    trimmed = [v[-min_len:] for v in values]
    stacked = torch.stack(trimmed, dim=-1)  # (T, N, F)
    # Normalize each factor per day to remove scale differences
    normed_factors = []
    for i in range(stacked.shape[-1]):
        normed_factors.append(AlphaPoolBase._normalize_by_day(stacked[..., i]))
    normed = torch.stack(normed_factors, dim=-1)
    normed = torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)
    return normed


def compute_target_returns(data: StockData, horizon: int = 1) -> Tensor:
    close = Feature(FeatureType.CLOSE)
    target_expr = Ref(close, -horizon) / close - 1
    target = target_expr.evaluate(data)
    return target


def _load_single_series(
    symbol: str,
    qlib_dir: str,
    start: str,
    end: str,
    device: torch.device,
    minutes_per_day: int = 390,
) -> Optional[Tensor]:
    """
    Load a single-instrument close series as tensor (T,).
    """
    try:
        data = StockData(
            instrument=[symbol],
            start_time=start,
            end_time=end,
            qlib_path=qlib_dir,
            device=device,
            max_backtrack_days=5,
            max_future_days=5,
            features=[FeatureType.CLOSE],
            load_minute_data=False,
            minutes_per_day=minutes_per_day,
        )
    except Exception:
        return None
    if data.data.shape[2] == 0:
        return None
    # Only CLOSE was requested, so it resides at index 0
    close = data.data[:, 0, 0]
    return close[-len(data._dates):]


def build_vix_series(
    qlib_dir: str,
    start: str,
    end: str,
    device: torch.device,
    minutes_per_day: int = 390,
) -> Optional[Tensor]:
    series = _load_single_series("^VIX", qlib_dir, start, end, device, minutes_per_day)
    if series is None:
        return None
    log_vix = torch.log(series + 1e-6)
    norm = (log_vix - log_vix.mean()) / (log_vix.std() + 1e-6)
    return norm


def compute_regime_series(
    target_ret: Tensor,
    vix_series: Optional[Tensor],
    window: int,
    t: int,
) -> Tensor:
    builder = RegimeFeatureBuilder(target_ret, vix_series, window)
    return builder.build(t)


def portfolio_step(
    policy: FactorGATPolicy,
    state: FactorState,
    factor_signals: Tensor,
    future_returns: Tensor,
    temp: float = 5.0,
    slippage_bps: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """
    Single step forward + portfolio return.
    """
    weights, logits = policy(
        state.node_features,
        state.edge_index,
        state.regime_features,
    )
    signal = factor_signals @ weights  # (N,)
    if future_returns.ndim == 1:
        step_ret = _long_short_return(signal, future_returns, temp=temp)
    else:
        step_ret = _long_short_hold_return(signal, future_returns, temp=temp)

    if slippage_bps > 0:
        # Apply round-trip cost once per rebalance (long + short legs).
        cost = (slippage_bps / 10000.0) * 2.0
        step_ret = step_ret - cost
    return step_ret, weights
