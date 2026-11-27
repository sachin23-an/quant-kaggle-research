"""Train a clean ML model (Ridge / Shallow Trees) on non-toxic, target-aligned features."""

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

class CleanAdaptiveRidge:
    def __init__(self, alpha: float = 3000.0, span: int = 240):
        self.alpha = alpha
        self.span = span
        self.ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        self.coef_ = None

    def _features(self, features: pd.DataFrame, tickers: list) -> np.ndarray:
        g = {i: features[f"Feature.{i}"][tickers] for i in range(1, 7)}
        
        # Validated clean feature blocks:
        # 1. Inverted valuation features (positively correlated with target +0.044)
        b = {}
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

        # 3. Liquidity / dispersion spreads
        b["f3_f5"] = csrank(g[3] - g[5])
        b["f5_f4"] = csrank(g[5] - g[4])
        b["f2_f5"] = csrank(g[2] - g[5])
        b["inv_f6"] = -csrank(g[6])

        # Stack into design matrix (N_time * N_tickers, K_features)
        X = np.hstack([d.values.reshape(-1, 1) for d in b.values()])
        return X

    def train(self, train_features: pd.DataFrame, train_target: pd.DataFrame) -> None:
        cols = list(train_target.columns)
        tickers = sorted(cols, key=lambda c: int(c.split(".")[-1]))
        
        X = self._features(train_features, tickers)
        y = train_target[tickers].values.reshape(-1)
        
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        # Use positive=True to force non-negative loadings on aligned alphas
        self.ridge = Ridge(alpha=self.alpha, fit_intercept=False, positive=True)
        self.ridge.fit(X[ok], y[ok])
        self.coef_ = self.ridge.coef_

    def predict(self, val_features: pd.DataFrame) -> pd.DataFrame:
        cols = list(val_features["Feature.1"].columns)
        tickers = sorted(cols, key=lambda c: int(c.split(".")[-1]))
        
        X = self._features(val_features, tickers)
        raw = np.nan_to_num(X) @ self.coef_
        pred_matrix = pd.DataFrame(raw.reshape(-1, len(tickers)), index=val_features.index, columns=tickers)
        
        # Cross-sectional de-meaning
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        
        # Causal position smoothing to crush 5bp fee
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        
        # Strict zero-beta de-meaning
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix[tickers].fillna(0.0)

print("=" * 80)
print("TESTING CLEAN ADAPTIVE RIDGE (SUPERVISED ML WITH SHRINKAGE)")
print("=" * 80)

for alpha in [500.0, 1500.0, 3000.0, 5000.0, 10000.0]:
    for span in [200, 260]:
        all_pnl = []
        all_ic = []
        all_to = []
        all_gf = []

        for idx, item in enumerate(harness._cached_val_data):
            # Train on cumulative training portion, predict on validation
            # Load full period data
            period = item["period"]
            data = harness._cached_period_data[period] if hasattr(harness, '_cached_period_data') else None
            # In cached val data:
            feat = item["val_features"]
            ret = item["val_returns"]
            tgt = item["val_target"]
            tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))

            # Train on first 60% of val_features, evaluate on last 40% (or train on prior)
            n_split = int(0.5 * len(feat))
            t_feat = feat.iloc[:n_split]
            t_tgt = tgt.iloc[:n_split]
            v_feat = feat.iloc[n_split:]
            v_ret = ret.iloc[n_split:]
            v_tgt = tgt.iloc[n_split:]

            m = CleanAdaptiveRidge(alpha=alpha, span=span)
            m.train(t_feat, t_tgt)
            pred = m.predict(v_feat)

            pnl = backtest(pred, v_ret[tickers])
            all_pnl.append(pnl)

            dot = (pred * v_tgt[tickers]).sum(axis=1)
            norm_s = np.sqrt((pred**2).sum(axis=1))
            norm_t = np.sqrt((v_tgt[tickers]**2).sum(axis=1))
            all_ic.append((dot / (norm_s * norm_t + 1e-9)).mean())

            to = pred.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean()
            all_to.append(to)

        concat_pnl = pd.concat(all_pnl, ignore_index=True)
        net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0
        print(f"alpha={alpha:<7.0f} | span={span:<3} | Net Sharpe: {net_sh:>+8.4f} | IC: {np.mean(all_ic):>+6.4f} | Turnover: {np.mean(all_to):.4f}")
