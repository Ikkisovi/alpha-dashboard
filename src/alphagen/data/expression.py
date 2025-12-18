from abc import ABCMeta, abstractmethod
from typing import List, Type, Union

import torch
from torch import Tensor

from alphagen_qlib.stock_data import StockData, FeatureType


class OutOfDataRangeError(IndexError):
    pass


class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str: return str(self)

    def __add__(self, other: Union["Expression", float]) -> "Add":
        if isinstance(other, Expression):
            return Add(self, other)
        else:
            return Add(self, Constant(other))

    def __radd__(self, other: float) -> "Add": return Add(Constant(other), self)

    def __sub__(self, other: Union["Expression", float]) -> "Sub":
        if isinstance(other, Expression):
            return Sub(self, other)
        else:
            return Sub(self, Constant(other))

    def __rsub__(self, other: float) -> "Sub": return Sub(Constant(other), self)

    def __mul__(self, other: Union["Expression", float]) -> "Mul":
        if isinstance(other, Expression):
            return Mul(self, other)
        else:
            return Mul(self, Constant(other))

    def __rmul__(self, other: float) -> "Mul": return Mul(Constant(other), self)

    def __truediv__(self, other: Union["Expression", float]) -> "Div":
        if isinstance(other, Expression):
            return Div(self, other)
        else:
            return Div(self, Constant(other))

    def __rtruediv__(self, other: float) -> "Div": return Div(Constant(other), self)

    def __pow__(self, other: Union["Expression", float]) -> "Pow":
        if isinstance(other, Expression):
            return Pow(self, other)
        else:
            return Pow(self, Constant(other))

    def __rpow__(self, other: float) -> "Pow": return Pow(Constant(other), self)

    def __pos__(self) -> "Expression": return self
    def __neg__(self) -> "Sub": return Sub(Constant(0), self)
    def __abs__(self) -> "Abs": return Abs(self)

    @property
    def is_featured(self): raise NotImplementedError


class Feature(Expression):
    def __init__(self, feature: FeatureType, is_minute: bool = False) -> None:
        self._feature = feature
        self._is_minute = is_minute

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        start = period.start + data.max_backtrack_days
        stop = period.stop + data.max_backtrack_days + data.n_days - 1
        
        # Determine feature index mapping dynamically
        # StockData features might be a subset or reordered relative to FeatureType enum
        if not hasattr(data, '_feature_index_map'):
            data._feature_index_map = {f: i for i, f in enumerate(data._features)}
        
        feature_idx = data._feature_index_map.get(self._feature)
        if feature_idx is None:
             raise ValueError(f"Feature {self._feature} not present in StockData loaded features.")
             
        return data.data[start:stop, feature_idx, :]

    def __str__(self) -> str:
        prefix = '$m_' if getattr(self, "_is_minute", False) else '$'
        return prefix + self._feature.name.lower()

    @property
    def is_featured(self): return True


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self._value = value

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert period.step == 1 or period.step is None
        if (period.start < -data.max_backtrack_days or
                period.stop - 1 > data.max_future_days):
            raise OutOfDataRangeError()
        device = data.data.device
        dtype = data.data.dtype
        days = period.stop - period.start - 1 + data.n_days
        return torch.full(size=(days, data.n_stocks),
                          fill_value=self._value, dtype=dtype, device=device)

    # def __str__(self) -> str: return f'Constant({str(self._value)})'
    def __str__(self) -> str: return f'{str(self._value)}'

    @property
    def is_featured(self): return False


class DeltaTime(Expression):
    # This is not something that should be in the final expression
    # It is only here for simplicity in the implementation of the tree builder
    def __init__(self, delta_time: int) -> None:
        self._delta_time = delta_time

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        assert False, "Should not call evaluate on delta time"

    def __str__(self) -> str: return str(self._delta_time)

    @property
    def is_featured(self): return False


# Operator base classes

class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int: ...

    @classmethod
    @abstractmethod
    def category_type(cls) -> Type['Operator']: ...


class UnaryOperator(Operator):
    def __init__(self, operand: Union[Expression, float]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)

    @classmethod
    def n_args(cls) -> int: return 1

    @classmethod
    def category_type(cls) -> Type['Operator']: return UnaryOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(data, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"

    @property
    def is_featured(self): return self._operand.is_featured


class BinaryOperator(Operator):
    def __init__(self, lhs: Union[Expression, float], rhs: Union[Expression, float]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return BinaryOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        lhs_val = self._lhs.evaluate(data, period)
        rhs_val = self._rhs.evaluate(data, period)
        if lhs_val.shape[0] != rhs_val.shape[0]:
            min_len = min(lhs_val.shape[0], rhs_val.shape[0])
            if min_len <= 0:
                raise ValueError(f"Time alignment failed for {type(self).__name__} operands.")
            # Keep the most recent overlapping window to avoid shape mismatches from different rolling windows
            lhs_val = lhs_val[-min_len:]
            rhs_val = rhs_val[-min_len:]
        return self._apply(lhs_val, rhs_val)

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs})"

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


class RollingOperator(Operator):
    def __init__(self, operand: Union[Expression, float], delta_time: Union[int, DeltaTime]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return RollingOperator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = self._operand.evaluate(data, slice(start, stop))   # (L+W-1, S)
        values = values.unfold(0, self._delta_time, 1)              # (L, S, W)
        return self._apply(values)                                  # (L, S)

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._delta_time})"

    @property
    def is_featured(self): return self._operand.is_featured


class PairRollingOperator(Operator):
    def __init__(self,
                 lhs: Expression, rhs: Expression,
                 delta_time: Union[int, DeltaTime]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)
        if isinstance(delta_time, DeltaTime):
            delta_time = delta_time._delta_time
        self._delta_time = delta_time

    @classmethod
    def n_args(cls) -> int: return 3

    @classmethod
    def category_type(cls) -> Type['Operator']: return PairRollingOperator

    def _unfold_one(self, expr: Expression,
                    data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        # L: period length (requested time window length)
        # W: window length (dt for rolling)
        # S: stock count
        values = expr.evaluate(data, slice(start, stop))            # (L+W-1, S)
        return values.unfold(0, self._delta_time, 1)                # (L, S, W)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        lhs = self._unfold_one(self._lhs, data, period)
        rhs = self._unfold_one(self._rhs, data, period)
        
        # Align lengths (handle mismatches from nested lookbacks/truncation)
        min_len = min(lhs.shape[0], rhs.shape[0])
        if lhs.shape[0] != min_len:
            lhs = lhs[-min_len:]
        if rhs.shape[0] != min_len:
            rhs = rhs[-min_len:]
            
        return self._apply(lhs, rhs)                                # (L, S)

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs},{self._delta_time})"

    @property
    def is_featured(self): return self._lhs.is_featured or self._rhs.is_featured


# Operator implementations

class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.abs()

    
class SLog1p(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign() * operand.abs().log1p()
    
class Inv(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return 1/operand


class Ret(UnaryOperator):
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        # Optimized robust evaluation: fetch extended window once to avoid shape mismatch
        # between current and prev slices if data is truncated at boundaries.
        start = period.start - 1
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        
        # values[1:] corresponds to 'current' (start..stop)
        # values[:-1] corresponds to 'prev' (start-1..stop-1)
        # They are guaranteed to have the same length (len(values) - 1)
        return values[1:] / values[:-1] - 1

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the UnaryOperator interface if called directly
        # But evaluate() overrides the main logic.
        # If _apply is called, it implies operand is already evaluated.
        # We can't do time-shift here easily without the time dimension context or assuming structure.
        # So we rely on evaluate() being called.
        return operand


class Sign(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sign()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.log()


class Rank(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        """
        Cross-sectional percentile rank with proper tie handling.
        Uses 'average' ranking: tied values get the mean of their ranks.
        This ensures constant inputs produce constant outputs (no spurious variation).

        Optimized O(n log n) implementation using argsort with tie correction.

        Input shape: (T, N) where T=time, N=stocks
        Output shape: (T, N) with ranks in [0, 1]
        """
        # operand shape: (T, N) - rank across N (stocks) for each T (day)
        nan_mask = operand.isnan()
        T, N = operand.shape

        # Replace NaN with -inf for sorting (will be at the start)
        neg_inf = torch.tensor(float('-inf'), device=operand.device, dtype=operand.dtype)
        safe_operand = torch.where(nan_mask, neg_inf, operand)

        # Get sort indices and inverse (O(n log n))
        sorted_indices = safe_operand.argsort(dim=-1)  # (T, N)

        # Create ranks: for each position, what rank does it have?
        # Use scatter to invert the argsort
        ranks = torch.zeros_like(operand)
        batch_idx = torch.arange(T, device=operand.device).unsqueeze(-1).expand(T, N)
        ranks[batch_idx, sorted_indices] = torch.arange(N, device=operand.device, dtype=operand.dtype).unsqueeze(0).expand(T, N)

        # Handle ties: gather sorted values and find where consecutive values are equal
        sorted_vals = torch.gather(safe_operand, -1, sorted_indices)  # (T, N)

        # Find ties: where sorted_vals[t, i] == sorted_vals[t, i+1]
        # For each group of ties, assign the average rank
        # This is done efficiently by finding group boundaries
        ties = torch.zeros_like(sorted_vals, dtype=torch.bool)
        ties[:, 1:] = (sorted_vals[:, 1:] == sorted_vals[:, :-1])

        # If there are ties, we need to average ranks within each tie group
        if ties.any():
            # Find start of each tie group (where value changes or is first)
            group_start = ~ties  # True at start of each group

            # Cumulative group ID for each position
            group_id = group_start.cumsum(dim=-1) - 1  # (T, N), 0-indexed group IDs

            # Calculate average rank per group using scatter_add
            # Sum of ranks in each group
            max_groups = N
            rank_sum = torch.zeros(T, max_groups, device=operand.device, dtype=operand.dtype)
            raw_ranks = torch.arange(N, device=operand.device, dtype=operand.dtype).unsqueeze(0).expand(T, N)
            rank_sum.scatter_add_(1, group_id, raw_ranks)

            # Count of elements in each group
            group_count = torch.zeros(T, max_groups, device=operand.device, dtype=operand.dtype)
            group_count.scatter_add_(1, group_id, torch.ones_like(raw_ranks))

            # Average rank for each group
            avg_rank = rank_sum / group_count.clamp(min=1)

            # Assign average rank back to each position in sorted order
            sorted_avg_ranks = torch.gather(avg_rank, 1, group_id)  # (T, N)

            # Scatter back to original positions
            ranks.scatter_(1, sorted_indices, sorted_avg_ranks)

        # Normalize to [0, 1] based on valid (non-NaN) count
        n_valid = (~nan_mask).sum(dim=-1, keepdim=True).clamp(min=1).float()
        ranks = ranks / (n_valid - 1).clamp(min=1)
        ranks = ranks.clamp(0, 1)
        ranks[nan_mask] = torch.nan
        return ranks


class Sqrt(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.abs().sqrt()


class IntraRef(Operator):
    """
    Minute-level lag operator. Shifts minute data within each day by delta_minutes.

    Should only be used inside intraday aggregators. Returns minute-level output.
    """
    def __init__(self, operand: Union[Expression, float], delta_minutes: Union[int, DeltaTime]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)
        if isinstance(delta_minutes, DeltaTime):
            delta_minutes = delta_minutes._delta_time
        self._delta_minutes = max(0, int(delta_minutes))

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return IntraRef

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        raise ValueError("IntraRef is a minute-level operator and must be used inside an intraday aggregator.")

    def _apply(self, minute_values: Tensor) -> Tensor:
        """
        minute_values: (n_days, minutes_per_day, n_stocks)
        """
        if self._delta_minutes == 0:
            return minute_values
        leading = minute_values[:, 0:1, :].expand(-1, self._delta_minutes, -1)
        shifted = torch.cat([leading, minute_values[:, :-self._delta_minutes, :]], dim=1)
        return shifted

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._delta_minutes})"

    @property
    def is_featured(self): return self._operand.is_featured


class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs + rhs


class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs - rhs


class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs * rhs


class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        # Guard against zero/near-zero denominators to avoid inf/nan cascades.
        eps = torch.tensor(1e-8, device=rhs.device, dtype=rhs.dtype)
        safe_rhs = torch.where(rhs.abs() < eps, eps, rhs)
        return lhs / safe_rhs


class Pow(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs ** rhs


class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.max(rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured and self._rhs.is_featured


class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: return lhs.min(rhs)

    @property
    def is_featured(self):
        return self._lhs.is_featured and self._rhs.is_featured


class Corr(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        """
        Cross-sectional correlation per day, broadcast back to factor shape.
        If either std is near zero, return 0 for that day.
        """
        lhs_mean = lhs.mean(dim=-1, keepdim=True)
        rhs_mean = rhs.mean(dim=-1, keepdim=True)
        lhs_centered = lhs - lhs_mean
        rhs_centered = rhs - rhs_mean
        cov = (lhs_centered * rhs_centered).mean(dim=-1)
        lhs_std = lhs_centered.pow(2).mean(dim=-1).sqrt()
        rhs_std = rhs_centered.pow(2).mean(dim=-1).sqrt()
        denom = lhs_std * rhs_std
        corr = torch.where(denom > 1e-8, cov / denom, torch.zeros_like(cov))
        return corr.unsqueeze(-1).expand_as(lhs)


def _compute_sorted_quantile(sorted_vals: Tensor, quantiles: Tensor) -> Tensor:
    """
    Helper for 1D quantile interpolation given sorted values.
    sorted_vals: (batch, n)
    quantiles: (batch,)
    """
    if sorted_vals.ndim != 2:
        sorted_vals = sorted_vals.reshape(-1, sorted_vals.shape[-1])
    n = sorted_vals.shape[-1]
    if n <= 0:
        raise ValueError("Cannot compute quantile over empty dimension.")
    quantiles = quantiles.to(dtype=sorted_vals.dtype)
    if n == 1:
        return sorted_vals[:, 0]
    idx = quantiles * (n - 1)
    lower_idx = torch.floor(idx).long().clamp_(0, n - 1)
    upper_idx = torch.ceil(idx).long().clamp_(0, n - 1)
    frac = (idx - lower_idx.to(dtype=sorted_vals.dtype))
    batch_indices = torch.arange(sorted_vals.shape[0], device=sorted_vals.device)
    lower = sorted_vals[batch_indices, lower_idx]
    upper = sorted_vals[batch_indices, upper_idx]
    return lower + (upper - lower) * frac


class Quantile(BinaryOperator):
    def _apply(self, operand: Tensor, quantile: Tensor) -> Tensor:
        n = operand.shape[-1]
        if n <= 0:
            raise ValueError("Quantile operator requires a non-empty cross-sectional dimension.")

        operand_shape = operand.shape[:-1]
        flat_values = operand.reshape(-1, n)

        q_vals = quantile
        if q_vals.dim() == operand.dim():
            q_vals = q_vals.mean(dim=-1)
        q_vals = q_vals.reshape(-1).clamp(0.0, 1.0)

        sorted_vals = flat_values.sort(dim=-1).values
        quantile_values = _compute_sorted_quantile(sorted_vals, q_vals)
        broadcast_shape = operand_shape + (1,)
        return quantile_values.view(*broadcast_shape).expand_as(operand)


class Gt(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs > rhs).to(dtype=lhs.dtype)


class Ge(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs >= rhs).to(dtype=lhs.dtype)


class Lt(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs < rhs).to(dtype=lhs.dtype)


class Le(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return (lhs <= rhs).to(dtype=lhs.dtype)


class Having(BinaryOperator):
    """
    Filters 'lhs' where 'rhs' (condition) is True (1.0).
    Returns NaN where condition is False.
    """
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        # rhs is expected to be a boolean mask (0.0 or 1.0)
        # We treat > 0.5 as True to be safe with float precision
        mask = rhs > 0.5
        return torch.where(mask, lhs, torch.tensor(float('nan'), device=lhs.device, dtype=lhs.dtype))


class NotHaving(BinaryOperator):
    """
    Filters 'lhs' where 'rhs' (condition) is False (0.0).
    Returns NaN where condition is True.
    """
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        # rhs is expected to be a boolean mask (0.0 or 1.0)
        # We treat <= 0.5 as True (condition is False)
        mask = rhs <= 0.5
        return torch.where(mask, lhs, torch.tensor(float('nan'), device=lhs.device, dtype=lhs.dtype))


class Ref(RollingOperator):
    # Ref is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Ref only deal with the values
    # at -dt. Nonetheless, it should be classified as rolling since it modifies
    # the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop - self._delta_time
        return self._operand.evaluate(data, slice(start, stop))

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class TsMean(RollingOperator):
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        # Check if operand is a simple Feature and cache is available
        if isinstance(self._operand, Feature) and hasattr(data, '_rolling_cache') and hasattr(data, '_feature_index_map'):
            feat_idx = data._feature_index_map.get(self._operand._feature)
            if feat_idx is not None:
                cached = data.get_cached_rolling(feat_idx, self._delta_time, 'mean', period)
                if cached is not None:
                    return cached
        # Fall back to standard evaluation
        return super().evaluate(data, period)

    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1)


class TsSum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.sum(dim=-1)


class TsStd(RollingOperator):
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        # Check if operand is a simple Feature and cache is available
        if isinstance(self._operand, Feature) and hasattr(data, '_rolling_cache') and hasattr(data, '_feature_index_map'):
            feat_idx = data._feature_index_map.get(self._operand._feature)
            if feat_idx is not None:
                cached = data.get_cached_rolling(feat_idx, self._delta_time, 'std', period)
                if cached is not None:
                    return cached
        # Fall back to standard evaluation
        return super().evaluate(data, period)

    def _apply(self, operand: Tensor) -> Tensor: return operand.std(dim=-1)

class TsIr(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.mean(dim=-1) / operand.std(dim=-1)

class TsMinMaxDiff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0] - operand.min(dim=-1)[0]
class TsMaxDiff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand[..., -1] - operand.max(dim=-1)[0]
class TsMinDiff(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand[..., -1] - operand.min(dim=-1)[0]

class TsVar(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.var(dim=-1)


class TsSkew(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # skew = m3 / m2^(3/2)
        central = operand - operand.mean(dim=-1, keepdim=True)
        m3 = (central ** 3).mean(dim=-1)
        m2 = (central ** 2).mean(dim=-1)
        return m3 / m2 ** 1.5


class TsKurt(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        # kurt = m4 / var^2 - 3
        central = operand - operand.mean(dim=-1, keepdim=True)
        m4 = (central ** 4).mean(dim=-1)
        var = operand.var(dim=-1)
        return m4 / var ** 2 - 3


class TsMax(RollingOperator):
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        # Check if operand is a simple Feature and cache is available
        if isinstance(self._operand, Feature) and hasattr(data, '_rolling_cache') and hasattr(data, '_feature_index_map'):
            feat_idx = data._feature_index_map.get(self._operand._feature)
            if feat_idx is not None:
                cached = data.get_cached_rolling(feat_idx, self._delta_time, 'max', period)
                if cached is not None:
                    return cached
        # Fall back to standard evaluation
        return super().evaluate(data, period)

    def _apply(self, operand: Tensor) -> Tensor: return operand.max(dim=-1)[0]


class TsMin(RollingOperator):
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        # Check if operand is a simple Feature and cache is available
        if isinstance(self._operand, Feature) and hasattr(data, '_rolling_cache') and hasattr(data, '_feature_index_map'):
            feat_idx = data._feature_index_map.get(self._operand._feature)
            if feat_idx is not None:
                cached = data.get_cached_rolling(feat_idx, self._delta_time, 'min', period)
                if cached is not None:
                    return cached
        # Fall back to standard evaluation
        return super().evaluate(data, period)

    def _apply(self, operand: Tensor) -> Tensor: return operand.min(dim=-1)[0]


class TsMed(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor: return operand.median(dim=-1)[0]


class TsMad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        central = operand - operand.mean(dim=-1, keepdim=True)
        return central.abs().mean(dim=-1)


class TsRank(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        last = operand[:, :, -1, None]
        left = (last < operand).count_nonzero(dim=-1)
        right = (last <= operand).count_nonzero(dim=-1)
        result = (right + left + (right > left)) / (2 * n)
        return result


class TsDelta(RollingOperator):
    # Delta is not *really* a rolling operator, in that other rolling operators
    # deal with the values in (-dt, 0], while Delta only deal with the values
    # at -dt and 0. Nonetheless, it should be classified as rolling since it
    # modifies the time window.

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] - values[:-self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        # This is just for fulfilling the RollingOperator interface
        ...


class TsRet(RollingOperator):
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop))
        return values[self._delta_time:] / values[:-self._delta_time] - 1

    def _apply(self, operand: Tensor) -> Tensor:
        ...


class TsDiv(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        return operand[:, :, -1] / operand.mean(dim=-1)


class TsArgMax(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        argmax_idx = operand.argmax(dim=-1).to(dtype=operand.dtype)
        # Normalize to [0, 1]: 0 = max at oldest point, 1 = max at newest point
        return argmax_idx / (n - 1) if n > 1 else torch.zeros_like(argmax_idx)


class TsArgMin(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        argmin_idx = operand.argmin(dim=-1).to(dtype=operand.dtype)
        # Normalize to [0, 1]: 0 = min at oldest point, 1 = min at newest point
        return argmin_idx / (n - 1) if n > 1 else torch.zeros_like(argmin_idx)


class TsPctChange(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return (operand[:, :, -1] - operand[:, :, 0]) / operand[:, :, 0]


class TsWMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class TsEMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1 - 2 / (1 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha ** power
        weights /= weights.sum()
        return (weights * operand).sum(dim=-1)


class TsSortino(RollingOperator):
    """
    Rolling Sortino ratio: mean return / downside std over the window.
    Operand is typically a price; returns are computed within the window.
    """

    def _apply(self, operand: Tensor) -> Tensor:
        # operand: (L, S, W)
        ret = operand[..., 1:] / operand[..., :-1] - 1
        avg_ret = ret.mean(dim=-1)
        downside_std = ret.clamp(max=0).std(dim=-1)
        return avg_ret / (downside_std + 1e-8)


class TsMomRank(RollingOperator):
    """
    Rolling momentum rank: average of 5/10/20-day momentum, ranked within the window.
    Assumes window >= 20.
    """

    def _apply(self, operand: Tensor) -> Tensor:
        # operand: (L, S, W)
        W = operand.shape[-1]
        # Require minimum window to compute 20-day momentum (inclusive window)
        if W < 20:
            return torch.full(operand.shape[:2], float('nan'), device=operand.device, dtype=operand.dtype)
        last = operand[..., -1]
        m5 = last / operand[..., -6] - 1
        m10 = last / operand[..., -11] - 1
        # Use the earliest available point for the 20-day leg
        m20 = last / operand[..., -min(20, W)] - 1
        mom_avg = (m5 + m10 + m20) / 3
        # Rank within window: compare last value to all values in window
        # Reuse TsRank logic: count of <= / < against window samples of mom_avg proxy
        # Build a surrogate window of mom_avg by shifting last against history of last?
        # Approximate by ranking last vs historical last values using rolling average of window ends
        # Simpler: percent rank of mom_avg across stocks only (not time) is not intended.
        # Instead, compute window-wise rank treating mom_avg of each position within window:
        # Create momentum series for each offset k using current/lagged prices.
        # For efficiency, approximate rank by comparing last momentum to momentum of previous (W-1) offsets.
        mom_series = []
        for k in (5, 10, 20):
            if W - k - 1 >= 0:
                mom_k = operand[..., k:] / operand[..., :-k] - 1
                mom_series.append(mom_k[..., :-1])  # exclude last to avoid overlap with current
        if not mom_series:
            return torch.full(operand.shape[:2], float('nan'), device=operand.device, dtype=operand.dtype)
        # Use 20-lag momentum series for rank if available; else fall back to combined
        mom_hist = mom_series[-1]  # shape (L, S, W-1)
        last_mom = m20
        # Rank: percentile of last_mom within historical window
        last_expanded = last_mom.unsqueeze(-1).expand_as(mom_hist)
        left = (last_expanded < mom_hist).count_nonzero(dim=-1)
        right = (last_expanded <= mom_hist).count_nonzero(dim=-1)
        rank = (right + left + (right > left)) / (2 * mom_hist.shape[-1])
        return rank


class TsMaxDd(RollingOperator):
    """
    Rolling max drawdown ratio: close / rolling max(close) - 1 over the window.
    """

    def _apply(self, operand: Tensor) -> Tensor:
        # operand: (L, S, W)
        roll_max = operand.max(dim=-1).values
        last = operand[..., -1]
        return last / roll_max - 1


class TsRelStrength(RollingOperator):
    """
    Rolling relative strength: MA(fast) / MA(slow) - 1.
    Takes operand, fast_window (const), slow_window (DeltaTime).
    fast_window must be < slow_window.
    """

    def __init__(self, operand: Union[Expression, float], fast_window: Union[Expression, float], slow_window: Union[int, DeltaTime]) -> None:
        # Normalize fast_window
        if isinstance(fast_window, DeltaTime):
            fast_window = fast_window._delta_time
        elif isinstance(fast_window, Expression):
            # Accept Constant expressions only
            if hasattr(fast_window, "_value"):
                fast_window = fast_window._value
            else:
                raise ValueError("fast_window must be a numeric constant")
        # Normalize slow_window
        if isinstance(slow_window, DeltaTime):
            slow_window = slow_window._delta_time
        elif isinstance(slow_window, Expression) and hasattr(slow_window, "_value"):
            slow_window = slow_window._value
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window for TsRelStrength")
        self._fast_window = int(fast_window)
        super().__init__(operand, slow_window)

    @classmethod
    def n_args(cls) -> int: return 3

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        # Evaluate with slow window; compute fast window from the same series
        start = period.start - self._delta_time + 1
        stop = period.stop
        series = self._operand.evaluate(data, slice(start, stop))  # (L+slow-1, S)
        slow = series.unfold(0, self._delta_time, 1)               # (L, S, slow)
        fast = series.unfold(0, self._fast_window, 1)              # (L+slow-fast, S, fast)
        # Align fast to last L windows
        offset = self._delta_time - self._fast_window
        fast = fast[offset:]
        if fast.shape[0] != slow.shape[0]:
            # Shape mismatch; return NaNs
            return torch.full(slow.shape[:2], float('nan'), device=slow.device, dtype=slow.dtype)
        ma_fast = fast.mean(dim=-1)
        ma_slow = slow.mean(dim=-1)
        return ma_fast / (ma_slow + 1e-8) - 1

    def _apply(self, operand: Tensor) -> Tensor:
        # Not used; evaluate overrides.
        raise NotImplementedError("TsRelStrength uses evaluate; _apply should not be called.")

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._fast_window},{self._delta_time})"


class TsQuantile(RollingOperator):
    def __init__(self, operand: Union[Expression, float], quantile: Union[Expression, float], delta_time: Union[int, DeltaTime]) -> None:
        self._quantile = quantile if isinstance(quantile, Expression) else Constant(quantile)
        super().__init__(operand, delta_time)

    @classmethod
    def n_args(cls) -> int: return 3

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        values = self._operand.evaluate(data, slice(start, stop)).unfold(0, self._delta_time, 1)
        quantiles = self._quantile.evaluate(data, period)
        return self._apply_quantile(values, quantiles)

    def _apply_quantile(self, operand: Tensor, quantiles: Tensor) -> Tensor:
        l, s, w = operand.shape
        
        # Align quantiles to operand length (time dimension)
        if quantiles.shape[0] != l:
            min_len = min(quantiles.shape[0], l)
            quantiles = quantiles[-min_len:]
            operand = operand[-min_len:]
            l = min_len
            
        flat_values = operand.reshape(-1, w)
        if quantiles.dim() > 1:
            q_vals = quantiles.mean(dim=-1)
        else:
            q_vals = quantiles
        q_vals = q_vals.reshape(-1, 1).expand(-1, s).reshape(-1).clamp(0.0, 1.0)
        sorted_vals = flat_values.sort(dim=-1).values
        quantile_values = _compute_sorted_quantile(sorted_vals, q_vals)
        return quantile_values.view(l, s)

    def _apply(self, operand: Tensor) -> Tensor:
        raise NotImplementedError("TsQuantile requires the quantile parameter; call evaluate() instead.")

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._quantile},{self._delta_time})"


class TsCov(PairRollingOperator):
    @staticmethod
    def _rolling_sum_2d(x: Tensor, window: int) -> Tensor:
        """
        Rolling sum over time for 2D tensors.

        Args:
            x: (T, S)
            window: window length W

        Returns:
            (T-W+1, S)
        """
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        if x.shape[0] < window:
            raise OutOfDataRangeError()

        prefix = torch.zeros((1, x.shape[1]), device=x.device, dtype=x.dtype)
        csum = torch.cat([prefix, x.cumsum(dim=0)], dim=0)  # (T+1, S)
        return csum[window:] - csum[:-window]

    @staticmethod
    def _rolling_cov_2d(lhs: Tensor, rhs: Tensor, window: int) -> Tensor:
        """
        Memory-efficient rolling covariance using cumulative sums (avoids (T,S,W) temporaries).
        """
        if lhs.ndim != 2 or rhs.ndim != 2:
            raise ValueError(f"Expected 2D lhs/rhs, got {tuple(lhs.shape)} / {tuple(rhs.shape)}")
        if lhs.shape != rhs.shape:
            min_len = min(lhs.shape[0], rhs.shape[0])
            lhs = lhs[-min_len:]
            rhs = rhs[-min_len:]

        lhs_f = torch.nan_to_num(lhs, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)
        rhs_f = torch.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)

        sum_x = TsCov._rolling_sum_2d(lhs_f, window)
        sum_y = TsCov._rolling_sum_2d(rhs_f, window)
        sum_xy = TsCov._rolling_sum_2d(lhs_f * rhs_f, window)

        n = float(window)
        cov_sum = sum_xy - (sum_x * sum_y) / n
        denom = float(max(window - 1, 1))
        return (cov_sum / denom).to(dtype=lhs.dtype)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        lhs = self._lhs.evaluate(data, slice(start, stop))
        rhs = self._rhs.evaluate(data, slice(start, stop))

        min_len = min(lhs.shape[0], rhs.shape[0])
        if lhs.shape[0] != min_len:
            lhs = lhs[-min_len:]
        if rhs.shape[0] != min_len:
            rhs = rhs[-min_len:]

        return self._rolling_cov_2d(lhs, rhs, self._delta_time)

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        """
        Fallback (used by PairRollingOperator.evaluate): still avoid allocating centered (T,S,W) buffers.
        """
        n = lhs.shape[-1]
        lhs_f = torch.nan_to_num(lhs, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)
        rhs_f = torch.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)

        sum_x = lhs_f.sum(dim=-1)
        sum_y = rhs_f.sum(dim=-1)
        sum_xy = (lhs_f * rhs_f).sum(dim=-1)

        cov_sum = sum_xy - (sum_x * sum_y) / float(n)
        return (cov_sum / float(max(n - 1, 1))).to(dtype=lhs.dtype)


class TsCorr(PairRollingOperator):
    @staticmethod
    def _rolling_sum_2d(x: Tensor, window: int) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}")
        if window <= 0:
            raise ValueError(f"window must be > 0, got {window}")
        if x.shape[0] < window:
            raise OutOfDataRangeError()

        prefix = torch.zeros((1, x.shape[1]), device=x.device, dtype=x.dtype)
        csum = torch.cat([prefix, x.cumsum(dim=0)], dim=0)  # (T+1, S)
        return csum[window:] - csum[:-window]

    @staticmethod
    def _rolling_corr_2d(lhs: Tensor, rhs: Tensor, window: int) -> Tensor:
        """
        Memory-efficient rolling correlation using cumulative sums.

        This avoids allocating large (T,S,W) intermediate tensors, which can OOM on
        long histories and large stock universes.
        """
        if lhs.ndim != 2 or rhs.ndim != 2:
            raise ValueError(f"Expected 2D lhs/rhs, got {tuple(lhs.shape)} / {tuple(rhs.shape)}")
        if lhs.shape != rhs.shape:
            min_len = min(lhs.shape[0], rhs.shape[0])
            lhs = lhs[-min_len:]
            rhs = rhs[-min_len:]

        lhs_f = torch.nan_to_num(lhs, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)
        rhs_f = torch.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)

        sum_x = TsCorr._rolling_sum_2d(lhs_f, window)
        sum_y = TsCorr._rolling_sum_2d(rhs_f, window)
        sum_x2 = TsCorr._rolling_sum_2d(lhs_f * lhs_f, window)
        sum_y2 = TsCorr._rolling_sum_2d(rhs_f * rhs_f, window)
        sum_xy = TsCorr._rolling_sum_2d(lhs_f * rhs_f, window)

        n = float(window)
        cov_sum = sum_xy - (sum_x * sum_y) / n
        var_x = (sum_x2 - (sum_x * sum_x) / n).clamp_min(0.0)
        var_y = (sum_y2 - (sum_y * sum_y) / n).clamp_min(0.0)

        denom = torch.sqrt(var_x * var_y)
        denom = torch.where((var_x < 1e-6) | (var_y < 1e-6), torch.ones_like(denom), denom)
        return (cov_sum / denom).to(dtype=lhs.dtype)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        start = period.start - self._delta_time + 1
        stop = period.stop
        lhs = self._lhs.evaluate(data, slice(start, stop))
        rhs = self._rhs.evaluate(data, slice(start, stop))

        min_len = min(lhs.shape[0], rhs.shape[0])
        if lhs.shape[0] != min_len:
            lhs = lhs[-min_len:]
        if rhs.shape[0] != min_len:
            rhs = rhs[-min_len:]

        return self._rolling_corr_2d(lhs, rhs, self._delta_time)

    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        """
        Fallback (used by PairRollingOperator.evaluate): avoid centered (T,S,W) temporaries.
        """
        n = lhs.shape[-1]
        lhs_f = torch.nan_to_num(lhs, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)
        rhs_f = torch.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)

        sum_x = lhs_f.sum(dim=-1)
        sum_y = rhs_f.sum(dim=-1)
        sum_x2 = (lhs_f * lhs_f).sum(dim=-1)
        sum_y2 = (rhs_f * rhs_f).sum(dim=-1)
        sum_xy = (lhs_f * rhs_f).sum(dim=-1)

        cov_sum = sum_xy - (sum_x * sum_y) / float(n)
        var_x = (sum_x2 - (sum_x * sum_x) / float(n)).clamp_min(0.0)
        var_y = (sum_y2 - (sum_y * sum_y) / float(n)).clamp_min(0.0)

        denom = torch.sqrt(var_x * var_y)
        denom = torch.where((var_x < 1e-6) | (var_y < 1e-6), torch.ones_like(denom), denom)
        return (cov_sum / denom).to(dtype=lhs.dtype)


# ============================================================================
# Intraday Aggregation Operators
# ============================================================================
# These operators aggregate minute-level data into daily values.
# They enable alpha mining on intraday patterns while maintaining daily IC evaluation.


class IntradayAggregator(Operator):
    """
    Base class for operators that aggregate minute-level data to daily values.

    Unlike RollingOperator which operates across days, IntradayAggregator
    operates WITHIN each day to create a single daily value per stock.

    Expression structure:
        DailyMean(Pow($volume, -5.0))

    Data flow:
        1. Evaluate operand on minute data: (n_days, 390, n_stocks)
        2. Aggregate within each day: (n_days, n_stocks)
        3. Use for daily IC calculation
    """

    def __init__(self, operand: Expression) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)

    @classmethod
    def n_args(cls) -> int:
        return 1

    @classmethod
    def category_type(cls) -> Type['Operator']:
        return IntradayAggregator

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        """
        Evaluate the operand on minute data and aggregate to daily values.

        Args:
            data: StockData with minute_data attribute
            period: Time period for evaluation (in days)

        Returns:
            Tensor of shape (n_days, n_stocks) - daily aggregated values

        Raises:
            ValueError: If minute_data is not available
        """
        if data.minute_data is None:
            raise ValueError(
                f"{type(self).__name__} requires minute-level data. "
                "Set load_minute_data=True when creating StockData."
            )

        # Guard against invalid trees that may bypass build-time validation (e.g. resumed pools)
        self._validate_minute_operand(self._operand)

        # Get the period indices for minute data
        # period is in days, but we need to map to minute data days
        # For now, assume minute data covers the same period as daily data
        start_day = period.start + data.max_backtrack_days
        stop_day = period.stop + data.max_backtrack_days + data.n_days - 1

        # Clamp to available minute data
        n_days_minute = data.minute_data.shape[0]
        start_day_clamped = max(0, min(start_day, n_days_minute))
        stop_day_clamped = max(start_day_clamped, min(stop_day, n_days_minute))

        # Evaluate operand on minute data
        # We need to temporarily make minute_data look like daily data for evaluation
        minute_values = self._evaluate_on_minutes(data, start_day_clamped, stop_day_clamped)
        # Shape: (n_days_actual, minutes_per_day, n_stocks)

        # Apply aggregation function within each day
        daily_values = self._apply(minute_values)  # Shape: (n_days_actual, n_stocks)

        # Ensure the result matches the expected number of days
        # This is consistent with how Feature.evaluate() works
        expected_days = period.stop - period.start - 1 + data.n_days
        actual_days = daily_values.shape[0]

        if actual_days != expected_days:
            # Pad or trim to match expected days
            if actual_days < expected_days:
                # Pad with NaN at the beginning (earlier dates missing)
                pad_size = expected_days - actual_days
                pad = torch.full(
                    (pad_size, daily_values.shape[1]),
                    float('nan'),
                    dtype=daily_values.dtype,
                    device=daily_values.device
                )
                daily_values = torch.cat([pad, daily_values], dim=0)
            else:
                # Trim from the beginning (keep most recent)
                daily_values = daily_values[-expected_days:]

        return daily_values

    def _validate_minute_operand(self, expr: Expression) -> None:
        """Enforce intraday layer rules: forbid daily rolling or nested intraday aggs."""
        layer = expression_output_layer(expr)
        if layer == "daily":
            raise ValueError(
                f"{type(self).__name__} expects minute-level input, got daily output from {type(expr).__name__}."
            )
        if isinstance(expr, (RollingOperator, PairRollingOperator)):
            raise ValueError(
                f"{type(self).__name__} expects minute-level operands; got rolling operator {type(expr).__name__}."
            )
        if isinstance(expr, IntradayAggregator):
            raise ValueError(f"Nested intraday aggregation is not supported: {type(expr).__name__}.")
        if isinstance(expr, IntraRef):
            # IntraRef is a minute-level lag; validate its operand recursively.
            self._validate_minute_operand(expr._operand)
            return
        if isinstance(expr, UnaryOperator):
            self._validate_minute_operand(expr._operand)
        elif isinstance(expr, BinaryOperator):
            self._validate_minute_operand(expr._lhs)
            self._validate_minute_operand(expr._rhs)
        elif isinstance(expr, (Feature, Constant)):
            return
        elif isinstance(expr, DeltaTime):
            raise ValueError("DeltaTime is not valid inside intraday aggregation operands.")
        elif isinstance(expr, Expression):
            # Any other operator types should be explicitly handled before use.
            raise NotImplementedError(
                f"Unsupported expression inside intraday aggregation: {type(expr).__name__}"
            )

    def _evaluate_on_minutes(self, data: StockData, start_day: int, stop_day: int) -> Tensor:
        """
        Evaluate the operand expression on minute-level data.

        This creates a temporary view where minute_data is treated as the primary data source.
        The operand is evaluated for each (day, minute) combination.

        Args:
            data: StockData with minute_data
            start_day: Starting day index in minute_data
            stop_day: Ending day index in minute_data

        Returns:
            Tensor of shape (n_days, minutes_per_day, n_stocks)
        """
        # Extract the relevant days from minute data
        minute_data_subset = data.minute_data[start_day:stop_day]
        # Shape: (n_days, minutes_per_day, n_features, n_stocks)

        n_days, minutes_per_day, n_features, n_stocks = minute_data_subset.shape
        dtype = minute_data_subset.dtype
        device = minute_data_subset.device

        def _eval_minutes(expr: Expression) -> Tensor:
            # Feature -> pull directly from minute tensor
            if isinstance(expr, Feature):
                feature_type = expr._feature
                try:
                    feature_idx = data._features.index(feature_type)
                except ValueError:
                    raise ValueError(
                        f"Feature {feature_type.name} not found in loaded features. "
                        f"Available features: {[f.name for f in data._features]}"
                    )
                return minute_data_subset[:, :, feature_idx, :]  # (n_days, minutes, n_stocks)

            # Constant -> broadcast to minute grid
            if isinstance(expr, Constant):
                return torch.full(
                    (n_days, minutes_per_day, n_stocks),
                    expr._value,
                    dtype=dtype,
                    device=device,
                )

            # Unary -> recurse then apply
            if isinstance(expr, UnaryOperator):
                operand = _eval_minutes(expr._operand)
                return expr._apply(operand)

            # Binary -> recurse then apply with alignment
            if isinstance(expr, BinaryOperator):
                lhs = _eval_minutes(expr._lhs)
                rhs = _eval_minutes(expr._rhs)
                # Align tensors if they have different time dimensions
                # This can happen when operands have different rolling windows
                if lhs.shape[0] != rhs.shape[0]:
                    min_len = min(lhs.shape[0], rhs.shape[0])
                    if min_len <= 0:
                        raise ValueError(
                            f"Time alignment failed for {type(expr).__name__} in intraday evaluation: "
                            f"lhs.shape={lhs.shape}, rhs.shape={rhs.shape}"
                        )
                    # Keep the most recent overlapping window
                    lhs = lhs[-min_len:]
                    rhs = rhs[-min_len:]
                return expr._apply(lhs, rhs)

            if isinstance(expr, IntraRef):
                operand = _eval_minutes(expr._operand)
                return expr._apply(operand)

            # Rolling operators work on daily time windows, so they should not be
            # used directly on minute-level expressions.
            if isinstance(expr, (RollingOperator, PairRollingOperator)):
                raise NotImplementedError(
                    f"{type(expr).__name__} operates on daily rolling windows. "
                    "Wrap minute-level computations with an intraday aggregator first."
                )

            # Nested intraday aggregation would reduce to daily and cannot be re-expanded here.
            if isinstance(expr, IntradayAggregator):
                raise NotImplementedError(
                    "Nested intraday aggregation is not supported inside minute-level evaluation."
                )

            if isinstance(expr, DeltaTime):
                raise ValueError("DeltaTime tokens are not valid inside intraday minute evaluation.")

            if isinstance(expr, Expression):
                raise NotImplementedError(
                    f"Unsupported expression type inside intraday evaluation: {type(expr).__name__}"
                )
            raise TypeError(f"Unexpected operand type: {type(expr).__name__}")

        return _eval_minutes(self._operand)

    @abstractmethod
    def _apply(self, minute_values: Tensor) -> Tensor:
        """
        Aggregate minute values to daily values.

        Args:
            minute_values: (n_days, minutes_per_day, n_stocks)

        Returns:
            daily_values: (n_days, n_stocks)
        """
        ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"

    @property
    def is_featured(self):
        return self._operand.is_featured


# Concrete Intraday Aggregation Operators

class IntraMean(IntradayAggregator):
    """Average of all minute values within each day"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        return minute_values.mean(dim=1)  # Average over minutes dimension


class IntraStd(IntradayAggregator):
    """Standard deviation of minute values within each day (intraday volatility)"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        return minute_values.std(dim=1)


class IntraLast(IntradayAggregator):
    """Last minute value of each day (close auction value)"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        return minute_values[:, -1, :]  # Last minute


class IntraFirst(IntradayAggregator):
    """First minute value of each day (open auction value)"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        return minute_values[:, 0, :]  # First minute


class IntraMax(IntradayAggregator):
    """Maximum minute value within each day"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        return minute_values.max(dim=1).values


class IntraMin(IntradayAggregator):
    """Minimum minute value within each day"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        return minute_values.min(dim=1).values


class IntraRange(IntradayAggregator):
    """Intraday range: max - min"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        return minute_values.max(dim=1).values - minute_values.min(dim=1).values


class IntraSum(IntradayAggregator):
    """Sum of all minute values (useful for volume, returns)"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        return minute_values.sum(dim=1)


class IntraMedian(IntradayAggregator):
    """Median of all minute values within each day"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        return minute_values.median(dim=1).values


class IntraVar(IntradayAggregator):
    """Variance of minute values within each day"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        return minute_values.var(dim=1)


class IntraSkew(IntradayAggregator):
    """Skewness of minute values (distribution asymmetry)"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        # skew = E[(X - μ)³] / σ³
        mean = minute_values.mean(dim=1, keepdim=True)
        central = minute_values - mean
        m3 = (central ** 3).mean(dim=1)
        m2 = (central ** 2).mean(dim=1)
        return m3 / (m2 ** 1.5 + 1e-8)


class IntraKurt(IntradayAggregator):
    """Kurtosis of minute values (tail risk measure)"""
    def _apply(self, minute_values: Tensor) -> Tensor:
        # kurt = E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
        mean = minute_values.mean(dim=1, keepdim=True)
        central = minute_values - mean
        m4 = (central ** 4).mean(dim=1)
        var = minute_values.var(dim=1)
        return m4 / (var ** 2 + 1e-8) - 3


class IntraSumRatio(Operator):
    """
    Ratio of two intraday sums: IntraSum(numerator) / IntraSum(denominator)
    Example: IntraSumRatio(Gt($return, 0), Constant(1)) -> Count of positive returns
    """
    def __init__(self, numerator: Expression, denominator: Expression) -> None:
        self._numerator = numerator if isinstance(numerator, Expression) else Constant(numerator)
        self._denominator = denominator if isinstance(denominator, Expression) else Constant(denominator)

    @classmethod
    def n_args(cls) -> int: return 2

    @classmethod
    def category_type(cls) -> Type['Operator']: return IntraSumRatio

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        # We reuse IntraSum logic but need to handle two operands
        # This is a bit tricky because IntradayAggregator expects 1 operand
        # So we implement it manually or compose it
        
        # Composition: IntraSum(num) / IntraSum(denom)
        num_agg = IntraSum(self._numerator)
        denom_agg = IntraSum(self._denominator)
        
        return num_agg.evaluate(data, period) / (denom_agg.evaluate(data, period) + 1e-8)
    
    def __str__(self) -> str:
        return f"IntraSumRatio({self._numerator},{self._denominator})"
    
    @property
    def is_featured(self):
        return self._numerator.is_featured or self._denominator.is_featured


def expression_output_layer(expr: Expression) -> str:
    """
    Best-effort frequency inference: returns 'minute' or 'daily'.
    Intraday aggregators and rolling ops produce daily outputs.
    Base ops inherit the max frequency of their children.
    """
    if isinstance(expr, (IntradayAggregator, IntraSumRatio, RollingOperator, PairRollingOperator)):
        return "daily"
    if isinstance(expr, UnaryOperator):
        return expression_output_layer(expr._operand)
    if isinstance(expr, BinaryOperator):
        left = expression_output_layer(expr._lhs)
        right = expression_output_layer(expr._rhs)
        if "daily" in (left, right):
            return "daily"
        return "minute"
    # Features/Constants default to minute to allow intraday usage; actual layer resolved by context.
    return "minute"



Operators: List[Type[Expression]] = [
    # Unary
    Abs, SLog1p, Inv, Sign, Log, Rank, Sqrt, IntraRef,
    # Binary
    Add, Sub, Mul, Div, Pow, Greater, Less, Quantile,
    # Rolling
    Ref, TsMean, TsSum, TsStd, TsIr, TsMinMaxDiff, TsMaxDiff, TsMinDiff, TsVar, TsSkew, TsKurt, TsMax, TsMin,
    TsMed, TsMad, TsRank, TsDelta, TsDiv, TsArgMax, TsArgMin, TsPctChange, TsWMA, TsEMA, TsQuantile,
    # Pair rolling
    TsCov, TsCorr,
    # Intraday aggregation
    # Intraday aggregation
    IntraMean, IntraStd, IntraLast, IntraFirst, IntraMax, IntraMin, IntraRange,
    IntraSum, IntraMedian, IntraVar, IntraSkew, IntraKurt,
    # Intraday Ratio
    IntraSumRatio
]
