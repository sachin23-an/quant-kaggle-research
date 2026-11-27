"""Analyze the relationship between target and returns across all periods."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)

print("=" * 65)
print("CORRELATION BETWEEN TARGET AND RETURNS ACROSS 31 VALIDATION PERIODS")
print("=" * 65)

corrs = []
for item in harness._cached_val_data:
    p = item["period"]
    ret = item["val_returns"]
    tgt = item["val_target"]
    tickers = list(ret.columns)

    # Cross-sectional correlation at each timestamp
    cs_dot = (ret[tickers] * tgt[tickers]).sum(axis=1)
    norm_r = np.sqrt((ret[tickers]**2).sum(axis=1))
    norm_t = np.sqrt((tgt[tickers]**2).sum(axis=1))
    cs_corr = cs_dot / (norm_r * norm_t + 1e-9)

    m_corr = cs_corr.mean()
    corrs.append(m_corr)
    # print(f"Period {p}: mean cross-sectional corr(target, returns) = {m_corr:+.4f}")

print(f"Overall Mean Corr(Target, Returns): {np.mean(corrs):+.4f} (min={np.min(corrs):+.4f}, max={np.max(corrs):+.4f})")
print(f"Positive periods: {sum(1 for c in corrs if c > 0)}/31 ({sum(1 for c in corrs if c > 0)/31*100:.1f}%)")
