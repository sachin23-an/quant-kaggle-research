"""
Test upgrading demo_engineered with discovered top-alpha spreads:
- F2 - F3 (+0.4800)
- F4 - F1 (+0.3983)
- F2 - F1 (+0.3750)
- F4 - F3 (+0.2911)
- F6 - F4 (+0.2865)
with expanding mean tilt removal and Ridge + XGBoost ensemble.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")

class UpgradedAlphaPredictor(Predictor):
    def __init__(self, ridge_weight=0.70, span=150, alpha=2000.0):
        self.ridge_weight = ridge_weight
        self.xgb_weight = 1.0 - ridge_weight
        self.span = span
        self.alpha = alpha
        self.ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        self.ridge_coef_ = None
        self.bst_ = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 500,
            "reg_lambda": 10.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 30

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        # Core individual ranks
        b["f6"] = self._csrank(g[6])
        b["f1"] = self._csrank(g[1])
        b["f5"] = self._csrank(g[5])
        
        # Discovered high-alpha pairwise spreads
        b["f2_f3"] = self._csrank(g[2] - g[3])  # Dim0 = +0.4800
        b["f4_f1"] = self._csrank(g[4] - g[1])  # Dim0 = +0.3983
        b["f2_f1"] = self._csrank(g[2] - g[1])  # Dim0 = +0.3750
        b["f4_f3"] = self._csrank(g[4] - g[3])  # Dim0 = +0.2911
        b["f6_f4"] = self._csrank(g[6] - g[4])  # Dim0 = +0.2865
        b["f6_f1"] = self._csrank(g[6] - g[1])  # Dim0 = +0.2811
        b["f3_f5"] = self._csrank(g[3] - g[5])  # Dim0 = +0.1793
        return b

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = sorted(list(b[next(iter(b))].columns), key=lambda c: int(c.split(".")[-1]))
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b[next(iter(b))].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_clean, y_clean = X[ok], y[ok]
        
        # 1. Fit Ridge (unconstrained sign, L2 regularized)
        self.ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        self.ridge.fit(X_clean, y_clean)
        self.ridge_coef_ = self.ridge.coef_
        
        # 2. Fit Shallow XGBoost
        dtrain = xgb.DMatrix(X_clean, label=y_clean)
        self.bst_ = xgb.train(self.params, dtrain, num_boost_round=self.n_rounds)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self.ridge_coef_ is not None and self.bst_ is not None, "Model must be trained before predict()"
        X, tickers, index = self._design(features)
        X_safe = np.nan_to_num(X)
        
        # Predictions
        raw_ridge = X_safe @ self.ridge_coef_
        raw_xgb = self.bst_.predict(xgb.DMatrix(X_safe))
        raw_blend = self.ridge_weight * raw_ridge + self.xgb_weight * raw_xgb
        
        pred_matrix = pd.DataFrame(
            raw_blend.reshape(-1, len(tickers)), index=index, columns=tickers
        )
        
        # Causal position smoothing
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        
        # Causal removal of static per-instrument tilt
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        
        # Cross-sectional zero-beta de-meaning
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix[tickers].fillna(0.0)
