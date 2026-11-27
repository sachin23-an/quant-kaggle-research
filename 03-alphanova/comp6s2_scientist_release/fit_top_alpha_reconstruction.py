"""Fit optimal linear combination to reconstruct Rank 2 and Rank 3 signals."""

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

# Compute Attempt 4 city
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

c_att4 = compute_city(np.concatenate(all_gf_att4, axis=0))

# Collect X and y_R2 across all periods
X_blocks = []
y_r2_list = []
y_r3_list = []
ret_list = []
tgt_list = []

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

    p_r2 = np.einsum("tij,i->tj", g_mat[:n_use], r2_city)
    p_r2 = p_r2 - p_r2.mean(axis=1, keepdims=True)
    y_r2_list.append(p_r2.reshape(-1))

    p_r3 = np.einsum("tij,i->tj", g_mat[:n_use], r3_city)
    p_r3 = p_r3 - p_r3.mean(axis=1, keepdims=True)
    y_r3_list.append(p_r3.reshape(-1))

    # Candidate feature definitions
    blk = {}
    for i in range(1, 7):
        blk[f"rank_F{i}"] = csrank(g[i].iloc[:n_use]).values.reshape(-1)
        blk[f"z_F{i}"] = zscore(g[i].iloc[:n_use]).values.reshape(-1)
    
    # All pairs of spreads
    for i in range(1, 7):
        for j in range(i + 1, 7):
            blk[f"spread_{i}_{j}"] = csrank(g[i].iloc[:n_use] - g[j].iloc[:n_use]).values.reshape(-1)

    # Cross ratios
    blk["ratio_6_1"] = csrank(g[6].iloc[:n_use] / (g[1].iloc[:n_use].abs() + 0.1)).values.reshape(-1)
    blk["ratio_6_core"] = csrank(g[6].iloc[:n_use] / ((g[1].iloc[:n_use] + g[3].iloc[:n_use]).abs() + 0.1)).values.reshape(-1)

    # Volatility / dispersion
    disp = np.std([csrank(g[1].iloc[:n_use]).values, csrank(g[2].iloc[:n_use]).values, csrank(g[3].iloc[:n_use]).values, csrank(g[4].iloc[:n_use]).values], axis=0)
    blk["core_disp"] = disp.reshape(-1)

    X_blocks.append(pd.DataFrame(blk))
    ret_list.append(ret[tickers].iloc[:n_use])
    tgt_list.append(tgt[tickers].iloc[:n_use])

X_df = pd.concat(X_blocks, ignore_index=True)
y_r2_arr = np.concatenate(y_r2_list)
y_r3_arr = np.concatenate(y_r3_list)

feature_cols = X_df.columns.tolist()

# 1. Fit Ridge on Rank 2 target
ridge_r2 = Ridge(alpha=100.0, fit_intercept=False)
ridge_r2.fit(X_df.values, y_r2_arr)
weights_r2 = pd.Series(ridge_r2.coef_, index=feature_cols)

print("=" * 80)
print("FITTED WEIGHTS TO RECONSTRUCT RANK 2 TARGET (cookedjay):")
print(weights_r2[weights_r2.abs() > 0.005].sort_values(ascending=False))

# 2. Fit Ridge on Rank 3 target
ridge_r3 = Ridge(alpha=100.0, fit_intercept=False)
ridge_r3.fit(X_df.values, y_r3_arr)
weights_r3 = pd.Series(ridge_r3.coef_, index=feature_cols)

print("\n" + "=" * 80)
print("FITTED WEIGHTS TO RECONSTRUCT RANK 3 TARGET (jinfanhahaha):")
print(weights_r3[weights_r3.abs() > 0.005].sort_values(ascending=False))
