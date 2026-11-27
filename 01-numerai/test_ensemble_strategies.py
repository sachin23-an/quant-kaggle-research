"""
=============================================================================
Numerai Classic: Fast Ensemble & Model Diversity Benchmark
=============================================================================
Evaluates whether ensembling non-linear LightGBM with an L2 regularized
linear factor model (Ridge) improves:
1. Out-of-Sample Sharpe (target > 0.9243)
2. Mean Date Correlation
3. Max Drawdown reduction (target < -0.2272)
=============================================================================
"""

import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
import joblib

DATA_DIR = Path("D:/Projects/Quant/quant-competition-research/data")

print("=" * 80)
print("NUMERAI ENSEMBLE BENCHMARK (LightGBM + Ridge)")
print("=" * 80)

# 1. Load feature names
with open(DATA_DIR / "v6.0_features.json") as f:
    feature_meta = json.load(f)

small_features = feature_meta["feature_sets"]["small"]
numeric_features = [c for c in small_features if c != "feature_country"]

# 2. Train Ridge on train.parquet (downsampled every 4th date for lightweight execution)
train_path = DATA_DIR / "v6.0_train.parquet"
print(f"Loading training data (subsampled every 4th date for lightweight execution)...")
t0 = time.time()
train_cols = ["date", "target"] + numeric_features

# Read train dates first
train_meta = pd.read_parquet(train_path, columns=["date"])
all_dates = sorted(train_meta["date"].unique())
sampled_dates = set(all_dates[::4])  # Every 4th date
print(f"Sampled {len(sampled_dates)} dates out of {len(all_dates)} total train dates.")

train_df = pd.read_parquet(
    train_path,
    columns=train_cols,
    filters=[("date", "in", list(sampled_dates))]
)
train_df = train_df.dropna(subset=["target"])
print(f"Loaded {len(train_df):,} training rows in {time.time()-t0:.1f}s.")

# Train Ridge model with alpha=1000
print("Fitting L2 Regularized Ridge model (alpha=1000)...")
t_ridge = time.time()
ridge = Ridge(alpha=1000.0)
ridge.fit(train_df[numeric_features].astype(np.float32), train_df["target"].astype(np.float32))
print(f"Ridge model trained in {time.time()-t_ridge:.2f}s.")

# Free training RAM immediately
del train_df, train_meta
gc.collect()

# 3. Load Validation data
val_path = DATA_DIR / "v6.0_validation.parquet"
print(f"\nLoading validation data ({val_path})...")
t_val = time.time()
val_cols = ["date", "target"] + small_features
val_df = pd.read_parquet(val_path, columns=val_cols)
val_df = val_df.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
print(f"Loaded {len(val_df):,} validation rows across {val_df['date'].nunique()} dates in {time.time()-t_val:.1f}s.")

# 4. Load Pretrained LightGBM model
print("Loading live LightGBM baseline model...")
for c in small_features:
    if not pd.api.types.is_numeric_dtype(val_df[c]):
        val_df[c] = val_df[c].astype("category")

model_lgbm = joblib.load(DATA_DIR / "v6.0_lgbm_model.joblib")

print("Generating predictions...")
t_p = time.time()
pred_lgbm = model_lgbm.predict(val_df[small_features])
pred_ridge = ridge.predict(val_df[numeric_features].astype(np.float32))
print(f"Predictions generated in {time.time()-t_p:.1f}s.")

# Normalize predictions to percentile ranks per date
def rank_per_era(df, scores):
    df_temp = pd.DataFrame({"date": df["date"], "score": scores})
    return df_temp.groupby("date")["score"].rank(pct=True).values

r_lgbm = rank_per_era(val_df, pred_lgbm)
r_ridge = rank_per_era(val_df, pred_ridge)

def evaluate_era_scores(df, scores, target_col="target"):
    df_eval = pd.DataFrame({"date": df["date"], "score": scores, "target": df[target_col]})
    def era_corr(sub):
        return np.corrcoef(sub["score"], sub["target"])[0, 1]
    
    corrs = df_eval.groupby("date").apply(era_corr, include_groups=False)
    mean_corr = corrs.mean()
    std_corr = corrs.std()
    sharpe = mean_corr / (std_corr + 1e-9)
    pos_pct = (corrs > 0).mean() * 100
    cum_returns = corrs.cumsum()
    peak = cum_returns.cummax()
    drawdown = cum_returns - peak
    max_dd = drawdown.min()
    return mean_corr, sharpe, pos_pct, max_dd

print("\n" + "=" * 80)
print(f"{'MODEL / BLEND':<35} | {'MEAN CORR':<10} | {'SHARPE':<8} | {'POS %':<7} | {'MAX DD':<9}")
print("-" * 80)

# Evaluate Individual Models
m_lgbm, s_lgbm, p_lgbm, dd_lgbm = evaluate_era_scores(val_df, r_lgbm)
print(f"{'1. LightGBM (101 feats)':<35} | {m_lgbm:+.5f}   | {s_lgbm:+.4f}  | {p_lgbm:.1f}%  | {dd_lgbm:+.5f}")

m_ridge, s_ridge, p_ridge, dd_ridge = evaluate_era_scores(val_df, r_ridge)
print(f"{'2. Ridge L2 (alpha=1000)':<35} | {m_ridge:+.5f}   | {s_ridge:+.4f}  | {p_ridge:.1f}%  | {dd_ridge:+.5f}")

# Cross-prediction correlation
corr_models = np.corrcoef(r_lgbm, r_ridge)[0, 1]
print(f"\nModel-to-Model Rank Correlation: {corr_models:.4f}")

# Evaluate Blends
print("\n" + "-" * 80)
print(f"{'ENSEMBLE RATIO':<35} | {'MEAN CORR':<10} | {'SHARPE':<8} | {'POS %':<7} | {'MAX DD':<9}")
print("-" * 80)

for w_lgbm in [0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
    w_ridge = 1.0 - w_lgbm
    blend_score = w_lgbm * r_lgbm + w_ridge * r_ridge
    m_b, s_b, p_b, dd_b = evaluate_era_scores(val_df, blend_score)
    label = f"{int(w_lgbm*100)}% LGBM + {int(w_ridge*100)}% Ridge"
    print(f"{label:<35} | {m_b:+.5f}   | {s_b:+.4f}  | {p_b:.1f}%  | {dd_b:+.5f}")

print("=" * 80)
