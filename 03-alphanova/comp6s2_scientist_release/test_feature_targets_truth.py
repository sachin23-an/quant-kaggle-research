"""Measure exact cross-sectional correlation of features with returns and target."""

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

def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

all_corr_ret = {f"Feature.{i}": [] for i in range(1, 7)}
all_corr_tgt = {f"Feature.{i}": [] for i in range(1, 7)}

for item in harness._cached_val_data:
    feat = item["val_features"]
    ret = item["val_returns"]
    tgt = item["val_target"]
    tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))

    for i in range(1, 7):
        f = csrank(feat[f"Feature.{i}"][tickers])
        r = ret[tickers]
        t = tgt[tickers]

        # Correlation with returns (PnL driver)
        # Note: signal at t-1 earns return at t: P(t-1) * X(t)
        # So contemporaneous F(t) predicting return at t+1:
        f_prev = f.shift(1)
        ok_r = f_prev.notna() & r.notna()
        # Dot product of normalized ranks / returns
        f_norm = f_prev.sub(f_prev.mean(axis=1), axis=0)
        r_norm = r.sub(r.mean(axis=1), axis=0)
        dot_r = (f_norm * r_norm).sum(axis=1) / (np.sqrt((f_norm**2).sum(axis=1)) * np.sqrt((r_norm**2).sum(axis=1)) + 1e-9)
        all_corr_ret[f"Feature.{i}"].append(dot_r.mean())

        # Correlation with target (IC driver)
        # Target is contemporaneous T(t) vs signal P(t) or P(t-1)
        t_norm = t.sub(t.mean(axis=1), axis=0)
        f_curr = f.sub(f.mean(axis=1), axis=0)
        dot_t = (f_curr * t_norm).sum(axis=1) / (np.sqrt((f_curr**2).sum(axis=1)) * np.sqrt((t_norm**2).sum(axis=1)) + 1e-9)
        all_corr_tgt[f"Feature.{i}"].append(dot_t.mean())

print("=" * 65)
print(f"{'Feature':<15} | {'Corr with Returns X(t)':<22} | {'Corr with Target T(t)':<20}")
print("=" * 65)
for i in range(1, 7):
    fn = f"Feature.{i}"
    cr = np.nanmean(all_corr_ret[fn])
    ct = np.nanmean(all_corr_tgt[fn])
    print(f"{fn:<15} | {cr:>+22.4f} | {ct:>+20.4f}")
