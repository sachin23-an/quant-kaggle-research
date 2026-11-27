"""Inspect learned weights of AlignedTargetRidge."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from test_aligned_target_alpha import AlignedTargetRidge

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)

# Inspect period 031 and average across all periods
feature_names = [
    "inv_f1", "inv_f2", "inv_f3", "inv_f4", "inv_core",
    "f6", "f6_f1", "f6_f2", "f6_f3", "f6_f4", "f6_f5",
    "d12", "d34"
]

all_coefs = []
for p_key in harness.periods:
    from data_loader import load_period
    d = load_period(p_key, str(harness.data_dir), harness.universe)
    n_total = len(d["features"])
    n_val = min(800, int(0.2 * n_total))
    n_train = n_total - n_val
    tr_feat = d["features"].iloc[:n_train]
    tr_tgt = d["target"].iloc[:n_train]
    cols = list(tr_tgt.columns)
    tickers = sorted(cols, key=lambda c: int(c.split(".")[-1]))

    model = AlignedTargetRidge(alpha=1500.0, span=260)
    X, _ = model._design(tr_feat, tickers)
    y = tr_tgt[tickers].values.reshape(-1)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)

    r = Ridge(alpha=1500.0, fit_intercept=False, positive=True)
    r.fit(X[ok], y[ok])
    all_coefs.append(r.coef_)

mean_coefs = np.mean(all_coefs, axis=0)
print("=" * 60)
print("MEAN LEARNED WEIGHTS ACROSS 31 PERIODS (alpha=1500, positive=True):")
print("=" * 60)
for fn, w in zip(feature_names, mean_coefs):
    print(f"{fn:<15}: {w:.6f}")
