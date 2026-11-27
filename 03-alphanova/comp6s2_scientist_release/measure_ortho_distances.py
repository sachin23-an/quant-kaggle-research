"""Measure exact distance between Orthogonal High-IC and Attempt 4 & 6."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from city_tools import compute_gauge_fixed, compute_city

def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)

def get_city(func):
    all_gf = []
    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        tickers = sorted(list(item["val_returns"].columns), key=lambda c: int(c.split(".")[-1]))
        g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 7)}
        sig = func(g, tickers)
        smoothed = sig.ewm(span=200, min_periods=1).mean()
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            p_vals = smoothed[tickers].values - smoothed[tickers].values.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(p_vals))
            all_gf.append(compute_gauge_fixed(p_vals[:n_use], g_mat[:n_use]))
    return compute_city(np.concatenate(all_gf, axis=0))

# 1. Attempt 4
c_att4 = get_city(lambda g, t: -(csrank(g[1]) + csrank(g[2]) + csrank(g[3]) + csrank(g[4])) / 4.0)

# 2. Attempt 6
c_att6 = get_city(lambda g, t: -(csrank(g[1]) + csrank(g[3])) / 2.0)

# 3. Orthogonal candidate
c_ortho = get_city(lambda g, t: -(csrank(g[1]) + csrank(g[3])) / 2.0 + 0.60 * (csrank(g[6] - g[1]) - csrank(g[5] - g[6])))

d4 = np.degrees(np.arccos(np.clip(np.dot(c_ortho, c_att4), -1, 1)))
d6 = np.degrees(np.arccos(np.clip(np.dot(c_ortho, c_att6), -1, 1)))

print(f"Distance between Orthogonal Candidate and Attempt #4: {d4:.2f}°")
print(f"Distance between Orthogonal Candidate and Attempt #6: {d6:.2f}°")
