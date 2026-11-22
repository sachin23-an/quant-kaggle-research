"""
Standard Quantitative Evaluation Metrics
- Optiver: Mean Absolute Error (MAE)
- Jane Street: Weighted Zero-Mean R²
- Alpha Signal Quality: Spearman Rank Information Coefficient (Rank IC)
- Portfolio & Strategy: Annualized Sharpe, Information Ratio, Maximum Drawdown
"""

import math
from typing import List, Optional, Sequence, Union

try:
    import numpy as np
except ImportError:
    np = None


def mean_absolute_error(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """Mean Absolute Error (used in Optiver Trading at the Close)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def weighted_zero_mean_r2(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> float:
    """
    Weighted Zero-Mean R² (Jane Street Market Data Forecasting evaluation metric):
    R² = 1 - sum(w * (y - y_hat)^2) / sum(w * y^2)
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    if weights is None:
        weights = [1.0] * n

    ss_res = sum(w * (yt - yp) ** 2 for yt, yp, w in zip(y_true, y_pred, weights))
    ss_tot = sum(w * (yt ** 2) for yt, w in zip(y_true, weights))

    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def calc_rank_ic(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    """
    Spearman Rank Information Coefficient (Rank IC).
    Evaluates monotonic ranking quality of alpha forecasts.
    """
    n = len(y_true)
    if n < 2:
        return 0.0

    def get_ranks(seq: Sequence[float]) -> List[float]:
        sorted_pairs = sorted(enumerate(seq), key=lambda x: x[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and sorted_pairs[j][1] == sorted_pairs[j + 1][1]:
                j += 1
            avg_rank = (i + j + 2) / 2.0  # 1-based ranking
            for k in range(i, j + 1):
                ranks[sorted_pairs[k][0]] = avg_rank
            i = j + 1
        return ranks

    rank_true = get_ranks(y_true)
    rank_pred = get_ranks(y_pred)

    mean_t = sum(rank_true) / n
    mean_p = sum(rank_pred) / n

    cov = sum((rt - mean_t) * (rp - mean_p) for rt, rp in zip(rank_true, rank_pred))
    var_t = sum((rt - mean_t) ** 2 for rt in rank_true)
    var_p = sum((rp - mean_p) ** 2 for rp in rank_pred)

    denom = math.sqrt(var_t * var_p)
    if denom == 0:
        return 0.0
    return cov / denom


def annualized_sharpe_ratio(
    returns: Sequence[float],
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Annualized Sharpe Ratio of excess returns.
    """
    n = len(returns)
    if n < 2:
        return 0.0

    excess = [r - (risk_free_rate / periods_per_year) for r in returns]
    mean_exc = sum(excess) / n
    var = sum((r - mean_exc) ** 2 for r in excess) / (n - 1)
    std = math.sqrt(var)

    if std == 0:
        return 0.0
    return (mean_exc / std) * math.sqrt(periods_per_year)


def information_ratio(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Information Ratio (Alpha / Tracking Error).
    """
    n = len(strategy_returns)
    if n < 2:
        return 0.0

    diff = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
    mean_diff = sum(diff) / n
    var_diff = sum((d - mean_diff) ** 2 for d in diff) / (n - 1)
    std_diff = math.sqrt(var_diff)

    if std_diff == 0:
        return 0.0
    return (mean_diff / std_diff) * math.sqrt(periods_per_year)


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """
    Maximum peak-to-trough decline of an equity curve.
    """
    if len(equity_curve) < 2:
        return 0.0

    peak = equity_curve[0]
    mdd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (peak - val) / peak
            if dd > mdd:
                mdd = dd
    return mdd


if __name__ == "__main__":
    print("Testing Quantitative Metrics...")
    y_t = [0.01, -0.02, 0.015, -0.005, 0.03]
    y_p = [0.008, -0.015, 0.02, -0.002, 0.025]
    w = [1.0, 1.2, 0.8, 1.0, 1.5]

    mae = mean_absolute_error(y_t, y_p)
    r2_zm = weighted_zero_mean_r2(y_t, y_p, w)
    ric = calc_rank_ic(y_t, y_p)
    sharpe = annualized_sharpe_ratio(y_t)
    mdd = max_drawdown([100, 105, 102, 98, 110, 107])

    print(f"MAE: {mae:.5f}")
    print(f"Weighted Zero-Mean R²: {r2_zm:.4f}")
    print(f"Rank IC: {ric:.4f}")
    print(f"Annualized Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {mdd:.2%}")

    assert ric > 0.9, "Rank IC failed correlation check!"
    assert mae > 0, "MAE check failed!"
    print("All quantitative metric checks passed successfully!")
