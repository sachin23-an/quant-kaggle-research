"""Reverse-engineer the top leaderboard signal coordinates on S^18."""

import sys
from pathlib import Path
import zipfile, io
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

with zipfile.ZipFile("D:/Projects/Quant/quant-competition-research/03-alphanova/city_coords_s2.zip") as z:
    with z.open("signal_cities_ambient_coords.parquet") as f:
        df = pd.read_parquet(io.BytesIO(f.read()))

top_signals = [
    ("Rank 2", "signal_1d4528f4db1c33c1"),
    ("Rank 3", "signal_9ebffdb75b09eb12"),
    ("Rank 4", "signal_1be5e43160de3010"),
    ("Rank 6", "signal_75dc15bae7f7fd18"),
]

coord_cols = [f"city_dim_{i}" for i in range(19)]

print("=" * 80)
print("TOP LEADERBOARD SIGNALS: AMBIENT COORDINATES ON S^18")
print("=" * 80)

top_vectors = {}
for label, sig_id in top_signals:
    row = df[df["predictor_id"] == sig_id]
    if not row.empty:
        vec = row[coord_cols].values[0]
        top_vectors[sig_id] = vec
        print(f"{label:<8} | ID: {sig_id} | Dim0 (IC): {vec[0]:+.4f} | Top 5 Dims: {np.round(vec[:5], 3)}")
        # Print non-zero dimensions with high magnitude
        top_dims = np.argsort(np.abs(vec))[::-1][:6]
        top_dim_str = ", ".join([f"dim_{d}={vec[d]:+.3f}" for d in top_dims])
        print(f"         Dominant dimensions: {top_dim_str}")
    else:
        print(f"{label:<8} | ID: {sig_id} NOT FOUND in archive")

# Angular distance between these top signals
sig_ids = list(top_vectors.keys())
print("\n--- Angular Distance Between Top Signals ---")
for i in range(len(sig_ids)):
    for j in range(i + 1, len(sig_ids)):
        v1 = top_vectors[sig_ids[i]]
        v2 = top_vectors[sig_ids[j]]
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        ang = np.arccos(dot) * 180.0 / np.pi
        print(f"{sig_ids[i]} vs {sig_ids[j]}: {ang:.1f}°")
