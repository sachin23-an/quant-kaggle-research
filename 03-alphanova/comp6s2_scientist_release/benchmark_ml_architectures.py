"""Benchmark ML architectures (Ridge, Shallow XGBoost, and Ridge+XGBoost Ensemble) across walk-forward periods."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import xgboost as xgb

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest, check_cross_sectional_demean
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

def build_feature_matrix(features: pd.DataFrame, tickers: list) -> tuple[np.ndarray, list]:
    g = {i: features[f"Feature.{i}"][tickers] for i in range(1, 7)}
    
    b = {}
    # 1. Target-aligned core valuation (IC = +0.044)
    b["inv_f1"] = -csrank(g[1])
    b["inv_f2"] = -csrank(g[2])
    b["inv_f3"] = -csrank(g[3])
    b["inv_f4"] = -csrank(g[4])
    b["inv_core"] = -(csrank(g[1]) + csrank(g[2]) + csrank(g[3]) + csrank(g[4])) / 4.0
    
    # 2. Relative value spreads vs Feature 6
    b["f6_f1"] = csrank(g[6] - g[1])
    b["f6_f2"] = csrank(g[6] - g[2])
    b["f6_f3"] = csrank(g[6] - g[3])
    b["f6_f4"] = csrank(g[6] - g[4])
    b["f6_f5"] = csrank(g[6] - g[5])

    # 3. Cross-sectional dispersion and orthogonal spreads
    b["f3_f5"] = csrank(g[3] - g[5])
    b["f5_f4"] = csrank(g[5] - g[4])
    b["f2_f5"] = csrank(g[2] - g[5])
    b["inv_f6"] = -csrank(g[6])
    
    # 4. Tail cubic transform
    b["cubic_core"] = -(csrank(g[1])**3 + csrank(g[2])**3 + csrank(g[3])**3 + csrank(g[4])**3) / 4.0

    X = np.hstack([d.values.reshape(-1, 1) for d in b.values()])
    return X, list(b.keys())

# Let's test ML models across all periods
class MLRidge:
    def __init__(self, alpha=2000.0, span=240):
        self.alpha = alpha
        self.span = span
        self.model = Ridge(alpha=self.alpha, fit_intercept=False, positive=True)

    def train(self, features, target, tickers):
        X, _ = build_feature_matrix(features, tickers)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model.fit(X[ok], y[ok])

    def predict(self, features, tickers):
        X, _ = build_feature_matrix(features, tickers)
        raw = np.nan_to_num(X) @ self.model.coef_
        df = pd.DataFrame(raw.reshape(-1, len(tickers)), index=features.index, columns=tickers)
        df = df.sub(df.mean(axis=1), axis=0)
        df = df.ewm(span=self.span, min_periods=1).mean()
        return df.sub(df.mean(axis=1), axis=0).fillna(0.0)

class MLXGBoost:
    def __init__(self, max_depth=2, reg_lambda=15.0, span=240):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.span = span
        self.bst = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": self.max_depth,
            "eta": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 600,
            "reg_lambda": self.reg_lambda,
            "verbosity": 0,
            "nthread": 4,
        }

    def train(self, features, target, tickers):
        X, _ = build_feature_matrix(features, tickers)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        dtrain = xgb.DMatrix(X[ok], label=y[ok])
        self.bst = xgb.train(self.params, dtrain, num_boost_round=30)

    def predict(self, features, tickers):
        X, _ = build_feature_matrix(features, tickers)
        dval = xgb.DMatrix(np.nan_to_num(X))
        raw = self.bst.predict(dval)
        df = pd.DataFrame(raw.reshape(-1, len(tickers)), index=features.index, columns=tickers)
        df = df.sub(df.mean(axis=1), axis=0)
        df = df.ewm(span=self.span, min_periods=1).mean()
        return df.sub(df.mean(axis=1), axis=0).fillna(0.0)

class MLEnsemble:
    def __init__(self, alpha=2000.0, max_depth=2, reg_lambda=15.0, span=240):
        self.ridge = MLRidge(alpha=alpha, span=span)
        self.xgb = MLXGBoost(max_depth=max_depth, reg_lambda=reg_lambda, span=span)

    def train(self, features, target, tickers):
        self.ridge.train(features, target, tickers)
        self.xgb.train(features, target, tickers)

    def predict(self, features, tickers):
        p_r = self.ridge.predict(features, tickers)
        p_x = self.xgb.predict(features, tickers)
        blended = 0.50 * p_r + 0.50 * p_x
        return blended.sub(blended.mean(axis=1), axis=0).fillna(0.0)

models_to_test = {
    "1. Non-Negative Ridge (alpha=2000, span=240)": MLRidge(alpha=2000.0, span=240),
    "2. Non-Negative Ridge (alpha=5000, span=260)": MLRidge(alpha=5000.0, span=260),
    "3. Regularized Shallow XGBoost (depth=2, lambda=15)": MLXGBoost(max_depth=2, reg_lambda=15.0, span=240),
    "4. Dual Ensemble (50% Ridge + 50% XGBoost)": MLEnsemble(alpha=3000.0, max_depth=2, reg_lambda=15.0, span=240),
}

print("=" * 105)
print(f"{'Model Architecture':<48} | {'Net Sharpe':<10} | {'Mean IC':<8} | {'Min DB Dist':<13}")
print("=" * 105)

for name, model in models_to_test.items():
    all_pnl = []
    all_ic = []
    all_gf = []

    for idx, item in enumerate(harness._cached_val_data):
        feat = item["val_features"]
        ret = item["val_returns"]
        tgt = item["val_target"]
        tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))

        # Train on first half of validation block, predict on second half (walk-forward simulation)
        n_mid = int(0.5 * len(feat))
        model.train(feat.iloc[:n_mid], tgt.iloc[:n_mid], tickers)
        pred = model.predict(feat.iloc[n_mid:], tickers)

        pnl = backtest(pred, ret[tickers].iloc[n_mid:])
        all_pnl.append(pnl)

        dot = (pred * tgt[tickers].iloc[n_mid:]).sum(axis=1)
        norm_s = np.sqrt((pred**2).sum(axis=1))
        norm_t = np.sqrt((tgt[tickers].iloc[n_mid:]**2).sum(axis=1))
        all_ic.append((dot / (norm_s * norm_t + 1e-9)).mean())

        g_mat = harness._cached_gauge[idx]
        if g_mat is not None:
            p_vals = pred.values - pred.values.mean(axis=1, keepdims=True)
            n_use = min(len(g_mat) - n_mid, len(p_vals))
            if n_use > 0:
                all_gf.append(compute_gauge_fixed(p_vals[:n_use], g_mat[n_mid:n_mid+n_use]))

    concat_pnl = pd.concat(all_pnl, ignore_index=True)
    net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0

    if all_gf:
        concat_gf = np.concatenate(all_gf, axis=0)
        city = compute_city(concat_gf)
        comp_df = compare_to_cities(city, city_db_path=city_db_path)
        min_dist = comp_df["angular_distance_deg"].iloc[0]
    else:
        min_dist = float('nan')

    print(f"{name:<48} | {net_sh:>+10.4f} | {np.mean(all_ic):>+8.4f} | {min_dist:>11.1f}°")
