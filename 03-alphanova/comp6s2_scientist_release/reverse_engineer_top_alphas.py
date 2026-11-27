"""Reverse-engineer the exact feature representations that produce Rank 2 and Rank 3 signals."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso

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

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    m = df.mean(axis=1).values.reshape(-1, 1)
    s = df.std(axis=1).values.reshape(-1, 1) + 1e-8
    return (df - m) / s

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")
df = pd.read_parquet(city_db_path)
dim_cols = [f"city_dim_{i}" for i in range(19)]

r2_city = df[df["predictor_id"] == "signal_1d4528f4db1c33c1"][dim_cols].values[0]
r3_city = df[df["predictor_id"] == "signal_9ebffdb75b09eb12"][dim_cols].values[0]

print("=" * 80)
print("DECONSTRUCTING RANK 2 AND RANK 3 TARGET SIGNALS FROM GAUGE MATRICES")
print("=" * 80)

# Build feature candidate matrix across all validation periods
feature_dict = {}

all_r2_target = []
all_r3_target = []
all_features = []

for idx, item in enumerate(harness._cached_val_data):
    feat = item["val_features"]
    ret = item["val_returns"]
    tgt = item["val_target"]
    tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
    g_mat = harness._cached_gauge[idx]

    if g_mat is None:
        continue

    g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 7)}
    n_use = min(len(g_mat), len(feat))

    # Reconstruct the implied signal at each timestamp: P(t) = G(t).T @ city
    # G(t) has shape (n_use, 19, 20). city has shape (19,).
    # P(t) has shape (n_use, 20).
    p_r2 = np.einsum("tij,i->tj", g_mat[:n_use], r2_city)
    p_r3 = np.einsum("tij,i->tj", g_mat[:n_use], r3_city)

    # Demean
    p_r2 = p_r2 - p_r2.mean(axis=1, keepdims=True)
    p_r3 = p_r3 - p_r3.mean(axis=1, keepdims=True)

    all_r2_target.append(p_r2.reshape(-1))
    all_r3_target.append(p_r3.reshape(-1))

    # Build feature candidate blocks for these timestamps
    blocks = {}
    for i in range(1, 7):
        blocks[f"F{i}_rank"] = csrank(g[i].iloc[:n_use]).values.reshape(-1)
        blocks[f"F{i}_z"] = zscore(g[i].iloc[:n_use]).values.reshape(-1)
        blocks[f"F{i}_diff1"] = g[i].iloc[:n_use].diff(1).fillna(0).values.reshape(-1)
        blocks[f"F{i}_diff5"] = g[i].iloc[:n_use].diff(5).fillna(0).values.reshape(-1)

    # Relative value spreads
    for i in range(1, 6):
        blocks[f"F6_F{i}_spread"] = csrank(g[6].iloc[:n_use] - g[i].iloc[:n_use]).values.reshape(-1)

    # Core interactions
    blocks["F1_F3_prod"] = (csrank(g[1].iloc[:n_use]) * csrank(g[3].iloc[:n_use])).values.reshape(-1)
    blocks["F6_F1_ratio"] = csrank(g[6].iloc[:n_use] / (g[1].iloc[:n_use].abs() + 1e-4)).values.reshape(-1)

    df_feats = pd.DataFrame(blocks)
    all_features.append(df_feats)

X_all = pd.concat(all_features, ignore_index=True)
y_r2 = np.concatenate(all_r2_target)
y_r3 = np.concatenate(all_r3_target)

print(f"Total observations for regression: {len(X_all)}")

# Fit Lasso / Ridge to see which features reconstruct Rank 2
print("\n--- Top Predictors of Rank 2 Signal (cookedjay, Sharpe +0.0215) ---")
corrs_r2 = X_all.apply(lambda col: np.corrcoef(col, y_r2)[0, 1])
print("Top 10 Positively Correlated Features with Rank 2:")
print(corrs_r2.sort_values(ascending=False).head(10))
print("\nTop 10 Negatively Correlated Features with Rank 2:")
print(corrs_r2.sort_values().head(10))

print("\n--- Top Predictors of Rank 3 Signal (jinfanhahaha, Sharpe +0.0214) ---")
corrs_r3 = X_all.apply(lambda col: np.corrcoef(col, y_r3)[0, 1])
print("Top 10 Positively Correlated Features with Rank 3:")
print(corrs_r3.sort_values(ascending=False).head(10))
print("\nTop 10 Negatively Correlated Features with Rank 3:")
print(corrs_r3.sort_values().head(10))
