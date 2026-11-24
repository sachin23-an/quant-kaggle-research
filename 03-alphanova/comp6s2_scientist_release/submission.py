"""
Bi-weekly Cross-Sectional Signal Forecasting — Season 2
AlphaNova Competition Submission
Model: AlphaSpreadPredictor
Author: Sachin Kumar (sachin23-an)
"""

# /// script
# dependencies = [
#   "xgboost>=3.2.0",
#   "scikit-learn>=1.3.0",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import xgboost as xgb

from predictor import Predictor

warnings.filterwarnings("ignore")


class AlphaSpreadPredictor(Predictor):
    """
    Cross-Sectional Alpha Predictor for AlphaNova Season 2.
    
    Architecture:
    1. Contemporaneous cross-sectional rank normalization [-1, 1] across all 6 features.
    2. Spread feature engineering: captures orthogonal cross-asset momentum & relative value.
    3. Dual linear & tree regularized model (Ridge + Shallow XGBoost blend).
    4. Causal turnover control via Exponential Weighted Moving Average (span=300).
    5. Causal expanding-mean tilt subtraction to eliminate static asset bias.
    6. Cross-sectional de-meaning to guarantee zero net market exposure.
    """

    def __init__(self):
        self.ridge = Ridge(alpha=800.0, fit_intercept=False)
        self.xgb_model = None
        self.xgb_params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 600,
            "reg_lambda": 10.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 35
        self.span = 300

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank normalized to [-1, 1], NaN-aware per row."""
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features: pd.DataFrame) -> dict:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        r = {i: self._csrank(g[i]) for i in range(1, 7)}
        
        b = {}
        # Core directional features
        b["r6_inv"] = -r[6]
        b["r5"] = r[5]
        b["r1"] = r[1]
        
        # Primary cross-sectional spreads
        b["d56"] = r[5] - r[6]
        b["d26"] = r[2] - r[6]
        b["d46"] = r[4] - r[6]
        b["d16"] = r[1] - r[6]
        
        # Relative value differences
        b["d12"] = r[1] - r[2]
        b["d34"] = r[3] - r[4]
        b["d13"] = r[1] - r[3]
        return b

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b[next(iter(b))].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b[next(iter(b))].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        
        X_clean, y_clean = X[ok], y[ok]
        
        # Train Ridge model
        self.ridge.fit(X_clean, y_clean)
        
        # Train XGBoost model
        dtrain = xgb.DMatrix(X_clean, label=y_clean)
        self.xgb_model = xgb.train(self.xgb_params, dtrain, num_boost_round=self.n_rounds)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self.xgb_model is not None, "Model must be trained before predict()"
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X)
        
        # Blend Ridge and XGBoost predictions
        pred_ridge = self.ridge.predict(X_clean)
        pred_xgb = self.xgb_model.predict(xgb.DMatrix(X_clean))
        raw = 0.50 * pred_ridge + 0.50 * pred_xgb
        
        pred_matrix = pd.DataFrame(
            raw.reshape(-1, len(tickers)), index=index, columns=tickers
        )
        
        # 1. Causal position smoothing to suppress 5 bp transaction costs
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        
        # 2. Causal removal of static per-instrument tilt
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        
        # 3. Cross-sectional de-meaning (strictly mandatory)
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)
