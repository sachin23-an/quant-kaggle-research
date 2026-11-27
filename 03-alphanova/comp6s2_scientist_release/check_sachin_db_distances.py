"""Check distance from Orthogonal High-IC alpha to ALL of Sachin's previous 7 submissions."""

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

city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")
df = pd.read_parquet(city_db_path)
dim_cols = [f"city_dim_{i}" for i in range(19)]

# Check if Sachin's signal IDs exist in the database
sachin_ids = [
    "signal_def684e15a30afcb",
    "signal_2079134745ebebd4",
    "signal_951c06b21f1b1587",
    "signal_ff972547ec0de88c",
    "signal_2ca39a1fca36a2c8",
    "signal_3838a1c9d3773201",
    "signal_ac2051ab4596ae7f",
]

found_sachin = df[df["predictor_id"].isin(sachin_ids)]
print(f"Sachin's signals found in city database: {len(found_sachin)}")
for _, row in found_sachin.iterrows():
    print(f"  {row['predictor_id']}: Dim0={row['city_dim_0']:.4f}")

# Compute Orthogonal High-IC Alpha city
all_gf = []
for idx, item in enumerate(harness._cached_val_data):
    feat = item["val_features"]
    ret = item["val_returns"]
    tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
    g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 7)}

    # Orthogonal High-IC: inv_core + 0.60 * ( (F6-F1) - (F5-F6) )
    inv_core = -(csrank(g[1]) + csrank(g[3])) / 2.0
    f6_spread = csrank(g[6] - g[1]) - csrank(g[5] - g[6])
    sig = inv_core + 0.60 * f6_spread

    smoothed = sig.ewm(span=200, min_periods=1).mean()
    smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)

    g_mat = harness._cached_gauge[idx]
    if g_mat is not None:
        p_vals = smoothed[tickers].values - smoothed[tickers].values.mean(axis=1, keepdims=True)
        n_use = min(len(g_mat), len(p_vals))
        all_gf.append(compute_gauge_fixed(p_vals[:n_use], g_mat[:n_use]))

city_ortho = compute_city(np.concatenate(all_gf, axis=0))

if not found_sachin.empty:
    for _, row in found_sachin.iterrows():
        c_prev = row[dim_cols].values
        d = np.degrees(np.arccos(np.clip(np.dot(city_ortho, c_prev), -1, 1)))
        print(f"Distance to {row['predictor_id']}: {d:.2f}°")
