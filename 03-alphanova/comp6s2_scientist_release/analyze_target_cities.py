"""Analyze how to construct signals that align with Rank 2 and Rank 3 on S^18."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities

def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")
df = pd.read_parquet(city_db_path)
dim_cols = [f"city_dim_{i}" for i in range(19)]

r2 = df[df["predictor_id"] == "signal_1d4528f4db1c33c1"][dim_cols].values[0]
r3 = df[df["predictor_id"] == "signal_9ebffdb75b09eb12"][dim_cols].values[0]
r4 = df[df["predictor_id"] == "signal_1be5e43160de3010"][dim_cols].values[0]

# Compute our Attempt #4 city
all_gf_att4 = []
for idx, item in enumerate(harness._cached_val_data):
    feat = item["val_features"]
    tickers = sorted(list(item["val_returns"].columns), key=lambda c: int(c.split(".")[-1]))
    g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 5)}
    sig = -(csrank(g[1]) + csrank(g[2]) + csrank(g[3]) + csrank(g[4])) / 4.0
    sig = sig.ewm(span=150, min_periods=1).mean().sub(sig.mean(axis=1), axis=0)
    g_mat = harness._cached_gauge[idx]
    if g_mat is not None:
        p_vals = sig[tickers].values - sig[tickers].values.mean(axis=1, keepdims=True)
        n_use = min(len(g_mat), len(p_vals))
        all_gf_att4.append(compute_gauge_fixed(p_vals[:n_use], g_mat[:n_use]))

city_att4 = compute_city(np.concatenate(all_gf_att4, axis=0))

print(f"Angle between Attempt #4 and Rank 2: {np.degrees(np.arccos(np.clip(np.dot(city_att4, r2), -1, 1))):.1f}°")
print(f"Angle between Attempt #4 and Rank 3: {np.degrees(np.arccos(np.clip(np.dot(city_att4, r3), -1, 1))):.1f}°")
print(f"Angle between Attempt #4 and Rank 4: {np.degrees(np.arccos(np.clip(np.dot(city_att4, r4), -1, 1))):.1f}°")
