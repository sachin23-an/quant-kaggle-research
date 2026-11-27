"""Analyze the projection of features and transforms onto S^18."""

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

# Let's project each raw feature and simple transforms into S^18
transforms = {
    "-F1": lambda g, t: -csrank(g[1][t]),
    "-F2": lambda g, t: -csrank(g[2][t]),
    "-F3": lambda g, t: -csrank(g[3][t]),
    "-F4": lambda g, t: -csrank(g[4][t]),
    "-F5": lambda g, t: -csrank(g[5][t]),
    "+F6": lambda g, t: csrank(g[6][t]),
    "F6-F1": lambda g, t: csrank(g[6][t] - g[1][t]),
    "F6-F2": lambda g, t: csrank(g[6][t] - g[2][t]),
    "F6-F3": lambda g, t: csrank(g[6][t] - g[3][t]),
    "F6-F5": lambda g, t: csrank(g[6][t] - g[5][t]),
    "F1-F2": lambda g, t: csrank(g[1][t] - g[2][t]),
    "F3-F4": lambda g, t: csrank(g[3][t] - g[4][t]),
}

cities = {}
for name, func in transforms.items():
    all_gf = []
    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        ret = item["val_returns"]
        tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
        g = {i: feat[f"Feature.{i}"] for i in range(1, 7)}

        sig = func(g, tickers)
        sig = sig.sub(sig.mean(axis=1), axis=0)

        smoothed = sig.ewm(span=200, min_periods=1).mean()
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)

        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            pred_vals = smoothed.values
            pred_vals = pred_vals - pred_vals.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(pred_vals))
            gf = compute_gauge_fixed(pred_vals[:n_use], g_mat[:n_use])
            all_gf.append(gf)

    concat_gf = np.concatenate(all_gf, axis=0)
    city = compute_city(concat_gf)
    cities[name] = city

print("Feature Cities on S^18 (Top 6 Dims):")
print("-" * 80)
for name, city in cities.items():
    top_dims = np.argsort(np.abs(city))[::-1][:6]
    dim_str = ", ".join([f"d{d}:{city[d]:+.3f}" for d in top_dims])
    print(f"{name:<10} | Dim0 (IC): {city[0]:+.4f} | {dim_str}")
