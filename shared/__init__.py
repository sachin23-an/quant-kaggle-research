"""
Shared quantitative research infrastructure:
- Purged & embargoed group time-series cross-validation
- Automated look-ahead & data leakage audit engine
- Standard quantitative performance metrics (Weighted Zero-Mean R², Vol-Adjusted Sharpe, Rank IC)
"""

from .validation import PurgedGroupTimeSeriesSplit
from .lookahead_audit import LookAheadAuditor
from .metrics import (
    weighted_zero_mean_r2,
    mean_absolute_error,
    calc_rank_ic,
    annualized_sharpe_ratio,
    information_ratio,
    max_drawdown,
)

__all__ = [
    "PurgedGroupTimeSeriesSplit",
    "LookAheadAuditor",
    "weighted_zero_mean_r2",
    "mean_absolute_error",
    "calc_rank_ic",
    "annualized_sharpe_ratio",
    "information_ratio",
    "max_drawdown",
]
