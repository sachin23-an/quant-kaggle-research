"""Check angular distance between candidate submission and our own previous submissions."""

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

# Let's compute the city for:
# 1. Attempt #4 (Pure -(F1..F4)/4, span=150)
# 2. Attempt #6 (Pure -(F1+F3)/2, span=180)
# 3. New candidate (-(F1..F4)/4 + 0.2*(F6-F1), span=200)

models = {
    "Attempt_4_signal_ff972547ec0de88c": lambda g, t: (-(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0).ewm(span=150, min_periods=1).mean(),
    "Attempt_6_signal_3838a1c9d3773201": lambda g, t: (-(csrank(g[1][t]) + csrank(g[3][t])) / 2.0).ewm(span=180, min_periods=1).mean(),
    "New_Candidate": lambda g, t: (-(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0 + 0.20 * csrank(g[6][t] - g[1][t])).ewm(span=200, min_periods=1).mean(),
}

cities = {}
for name, func in models.items():
    all_gf = []
    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        ret = item["val_returns"]
        tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
        g = {i: feat[f"Feature.{i}"] for i in range(1, 7)}

        sig = func(g, tickers)
        sig = sig.sub(sig.mean(axis=1), axis=0)

        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            pred_vals = sig[tickers].values
            pred_vals = pred_vals - pred_vals.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(pred_vals))
            gf = compute_gauge_fixed(pred_vals[:n_use], g_mat[:n_use])
            all_gf.append(gf)

    concat_gf = np.concatenate(all_gf, axis=0)
    city = compute_city(concat_gf)
    cities[name] = city

c1 = cities["Attempt_4_signal_ff972547ec0de88c"]
c2 = cities["Attempt_6_signal_3838a1c9d3773201"]
c3 = cities["New_Candidate"]

dot_4_new = np.clip(np.dot(c1, c3), -1.0, 1.0)
deg_4_new = np.degrees(np.arccos(dot_4_new))

dot_6_new = np.clip(np.dot(c2, c3), -1.0, 1.0)
deg_6_new = np.degrees(np.arccos(dot_6_new))

print(f"Angular distance to Attempt #4 (signal_ff972547ec0de88c): {deg_4_new:.2f}°")
print(f"Angular distance to Attempt #6 (signal_3838a1c9d3773201): {deg_6_new:.2f}°")
