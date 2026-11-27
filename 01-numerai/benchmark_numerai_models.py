"""
=============================================================================
Numerai Classic Offline Validation Benchmark Engine
=============================================================================
Evaluates models across all historical validation eras in v6.0_validation.parquet:
- Mean Era CORR (CORR20)
- Correlation Sharpe (Mean / Std)
- Max Era Drawdown
- % Positive Eras
=============================================================================
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import joblib

DATA_DIR = Path("D:/Projects/Quant/quant-competition-research/data")

print("=" * 80)
print("NUMERAI OFFLINE VALIDATION BENCHMARK — v6.0")
print("=" * 80)

# 1. Load Feature Metadata
with open(DATA_DIR / "v6.0_features.json") as f:
    feature_meta = json.load(f)

small_features = feature_meta["feature_sets"]["small"]
numeric_features = [c for c in small_features if c != "feature_country"]
print(f"Loaded 'small' feature set: {len(small_features)} features ({len(numeric_features)} numeric)")

# 2. Inspect Validation Dataset Metadata
val_path = DATA_DIR / "v6.0_validation.parquet"
print(f"Validation dataset: {val_path}")

cols_to_load = ["date", "target"] + small_features
print(f"Loading {len(cols_to_load)} columns from validation parquet...")
t0 = time.time()
val_df = pd.read_parquet(val_path, columns=cols_to_load)
print(f"Loaded {len(val_df):,} validation rows across {val_df['date'].nunique()} dates in {time.time()-t0:.1f}s")

val_df = val_df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
unique_dates = sorted(val_df["date"].unique())
print(f"Dates range: {unique_dates[0]} to {unique_dates[-1]} (Total: {len(unique_dates)} dates)")

# 3. Load Current Live Model (Model 1: Baseline LGBM on 101 features)
model_path = DATA_DIR / "v6.0_lgbm_model.joblib"
print(f"\nLoading live deployed model: {model_path}...")
model_lgbm = joblib.load(model_path)
print("Model loaded successfully.")

# Convert categorical columns
for c in small_features:
    if not pd.api.types.is_numeric_dtype(val_df[c]):
        val_df[c] = val_df[c].astype("category")

# 4. Generate Predictions
print("\nGenerating baseline predictions on validation set...")
t_pred = time.time()
val_df["pred_raw"] = model_lgbm.predict(val_df[small_features])
print(f"Predictions generated in {time.time()-t_pred:.1f}s.")

# Evaluation helper
def evaluate_predictions(df, pred_col, target_col="target"):
    def era_corr(sub):
        r_pred = rankdata(sub[pred_col]) / len(sub)
        return np.corrcoef(r_pred, sub[target_col])[0, 1]

    corrs = df.groupby("date").apply(era_corr, include_groups=False)
    mean_corr = corrs.mean()
    std_corr = corrs.std()
    sharpe = mean_corr / (std_corr + 1e-9)
    pos_pct = (corrs > 0).mean() * 100
    
    cum_returns = corrs.cumsum()
    peak = cum_returns.cummax()
    drawdown = cum_returns - peak
    max_dd = drawdown.min()
    
    return {
        "mean_corr": mean_corr,
        "std_corr": std_corr,
        "sharpe": sharpe,
        "pos_pct": pos_pct,
        "max_dd": max_dd,
        "corrs": corrs
    }

metrics_baseline = evaluate_predictions(val_df, "pred_raw")

print("\n" + "=" * 80)
print(f"{'MODEL / STRATEGY':<35} | {'MEAN CORR':<10} | {'SHARPE':<8} | {'POS %':<7} | {'MAX DD':<9}")
print("-" * 80)
print(f"{'1. Baseline LGBM (Raw 101 feats)':<35} | {metrics_baseline['mean_corr']:+.5f}   | {metrics_baseline['sharpe']:+.4f}  | {metrics_baseline['pos_pct']:.1f}%  | {metrics_baseline['max_dd']:+.5f}")

# 5. Fast Date-by-Date Feature Neutralization
print("\nComputing Feature Projections per era...")
t_neut = time.time()

# Precompute projection component for each date group
projections = np.zeros(len(val_df), dtype=np.float64)

# Group indices
date_groups = val_df.groupby("date").indices

for date, idxs in date_groups.items():
    sub_scores = val_df["pred_raw"].iloc[idxs].values.astype(np.float64)
    sub_feats = val_df[numeric_features].iloc[idxs].values.astype(np.float64)
    # Mean-center features & scores
    sub_feats_mean = sub_feats.mean(axis=0, keepdims=True)
    sub_feats_c = sub_feats - sub_feats_mean
    sub_scores_c = sub_scores - sub_scores.mean()
    
    # Solve (X^T X + lambda I) w = X^T y
    XtX = sub_feats_c.T @ sub_feats_c
    reg = 1e-4 * np.trace(XtX) / XtX.shape[0] * np.eye(XtX.shape[0])
    w = np.linalg.solve(XtX + reg, sub_feats_c.T @ sub_scores_c)
    proj = sub_feats_c @ w
    projections[idxs] = proj

print(f"Computed projections across {len(date_groups)} dates in {time.time()-t_neut:.2f}s.")

# Evaluate different neutralization strengths
for prop in [0.25, 0.50, 0.75, 1.00]:
    col_name = f"pred_neut_{int(prop*100)}"
    val_df[col_name] = val_df["pred_raw"] - prop * projections
    m = evaluate_predictions(val_df, col_name)
    name = f"2. Feature Neutralized ({int(prop*100)}%)"
    print(f"{name:<35} | {m['mean_corr']:+.5f}   | {m['sharpe']:+.4f}  | {m['pos_pct']:.1f}%  | {m['max_dd']:+.5f}")

print("=" * 80)
