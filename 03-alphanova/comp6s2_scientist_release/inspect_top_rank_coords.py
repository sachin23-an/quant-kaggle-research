"""Inspect coordinates of top leaderboard signals and find what feature combinations reconstruct them."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")
df = pd.read_parquet(city_db_path)

dim_cols = [f"city_dim_{i}" for i in range(19)]

r2 = df[df["predictor_id"] == "signal_1d4528f4db1c33c1"][dim_cols].values[0]
r3 = df[df["predictor_id"] == "signal_9ebffdb75b09eb12"][dim_cols].values[0]
r4 = df[df["predictor_id"] == "signal_1be5e43160de3010"][dim_cols].values[0]

print("Rank 2 (cookedjay, Sharpe +0.0215, IC 0.0573):")
print(f"Dim 0 (IC): {r2[0]:+.4f}")
for i, v in enumerate(r2):
    if abs(v) > 0.15:
        print(f"  dim_{i:02d}: {v:+.4f}")

print("\nRank 3 (jinfanhahaha, Sharpe +0.0214, IC 0.0569):")
print(f"Dim 0 (IC): {r3[0]:+.4f}")
for i, v in enumerate(r3):
    if abs(v) > 0.15:
        print(f"  dim_{i:02d}: {v:+.4f}")

print("\nRank 4 (devnikhilmishra, Sharpe +0.0172, IC 0.0502):")
print(f"Dim 0 (IC): {r4[0]:+.4f}")
for i, v in enumerate(r4):
    if abs(v) > 0.15:
        print(f"  dim_{i:02d}: {v:+.4f}")
