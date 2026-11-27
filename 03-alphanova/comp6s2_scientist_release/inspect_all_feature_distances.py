"""Find the maximally novel direction on S^18 that is farthest from all competitors and prior submissions, then map it to features."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities
from evaluation import backtest

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")
df = pd.read_parquet(city_db_path)
dim_cols = [f"city_dim_{i}" for i in range(19)]
db_cities = df[dim_cols].values
db_norms = np.linalg.norm(db_cities, axis=1, keepdims=True)
db_cities = db_cities / np.maximum(db_norms, 1e-8)

def csrank(df):
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

# Let's project key candidate features into S^18
feature_cities = {}
features_tested = {
    "F6-F1": lambda g, t: csrank(g[6][t] - g[1][t]),
    "F6-F2": lambda g, t: csrank(g[6][t] - g[2][t]),
    "F6-F3": lambda g, t: csrank(g[6][t] - g[3][t]),
    "F6-F4": lambda g, t: csrank(g[6][t] - g[4][t]),
    "F6-F5": lambda g, t: csrank(g[6][t] - g[5][t]),
    "F3-F5": lambda g, t: csrank(g[3][t] - g[5][t]),
    "F2-F5": lambda g, t: csrank(g[2][t] - g[5][t]),
    "F5-F4": lambda g, t: csrank(g[5][t] - g[4][t]),
    "F5-F1": lambda g, t: csrank(g[5][t] - g[1][t]),
    "-F6":   lambda g, t: -csrank(g[6][t]),
    "-F5":   lambda g, t: -csrank(g[5][t]),
    "-F1":   lambda g, t: -csrank(g[1][t]),
    "-F2":   lambda g, t: -csrank(g[2][t]),
    "-F3":   lambda g, t: -csrank(g[3][t]),
    "-F4":   lambda g, t: -csrank(g[4][t]),
}

f_cities = []
f_names = list(features_tested.keys())

for name in f_names:
    func = features_tested[name]
    all_gf = []
    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        tickers = sorted(list(item["val_returns"].columns), key=lambda c: int(c.split(".")[-1]))
        g = {i: feat[f"Feature.{i}"] for i in range(1, 7)}
        sig = func(g, tickers)
        smoothed = sig.ewm(span=200, min_periods=1).mean().sub(sig.mean(axis=1), axis=0)
        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            p_vals = smoothed[tickers].values - smoothed[tickers].values.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat), len(p_vals))
            all_gf.append(compute_gauge_fixed(p_vals[:n_use], g_mat[:n_use]))
    c = compute_city(np.concatenate(all_gf, axis=0))
    f_cities.append(c)

f_cities_mat = np.array(f_cities) # (M, 19)

# For each feature, check its distance to the nearest database city and Dim0
print("=" * 80)
print(f"{'Feature / Spread':<15} | {'City Dim0 (IC)':<15} | {'Min Dist to DB':<15} | {'Nearest ID':<20}")
print("=" * 80)
for idx, name in enumerate(f_names):
    c = f_cities_mat[idx]
    dots = db_cities @ c
    min_dist_deg = np.degrees(np.arccos(np.clip(np.max(dots), -1, 1)))
    near_idx = np.argmax(dots)
    near_id = df["predictor_id"].iloc[near_idx]
    print(f"{name:<15} | {c[0]:>+15.4f} | {min_dist_deg:>13.1f}° | {near_id:<20}")
