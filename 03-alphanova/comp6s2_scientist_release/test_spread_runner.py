"""
Bi-weekly Cross-Sectional Signal Forecasting — Season 2
AlphaNova Competition Submission
Model: SpreadEngineeredXGBoost
Author: Sachin Kumar (sachin23-an)
"""

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class SpreadEngineeredXGBoost(Predictor):
    """
    Spread-Engineered Low-Capacity XGBoost Predictor for AlphaNova Season 2.
    
    Architecture:
    1. Contemporaneous cross-sectional spreads against Feature 6 (the dominant factor):
       - f6, f5, core (f1..f4 average)
       - f5 - f6, f4 - f6, f2 - f6, f1 - f6, f3 - f6
    2. Low-capacity regularized XGBoost (max_depth=2, min_child_weight=500, reg_lambda=5.0, 30 rounds)
       fitted walk-forward to learn regime-dependent spread weights.
    3. Causal EWM position smoothing (span=120).
    4. Causal expanding-mean tilt subtraction.
    5. Zero-beta cross-sectional de-meaning.
    """

    def __init__(self):
        self.model = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 500,
            "reg_lambda": 5.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 30
        self.span = 120

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        """Contemporaneous cross-sectional rank normalized to [-1, 1], NaN-aware."""
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _design(self, features: pd.DataFrame):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = list(g[1].columns)
        
        b = {}
        # Core directional signals
        b["f6"] = self._csrank(g[6][tickers])
        b["f5"] = self._csrank(g[5][tickers])
        b["f_core"] = (
            self._csrank(g[1][tickers])
            + self._csrank(g[2][tickers])
            + self._csrank(g[3][tickers])
            + self._csrank(g[4][tickers])
        ) / 4.0
        
        # Primary alpha spreads against F6 (momentum & valuation spreads)
        b["f5_f6"] = self._csrank(g[5][tickers] - g[6][tickers])
        b["f4_f6"] = self._csrank(g[4][tickers] - g[6][tickers])
        b["f2_f6"] = self._csrank(g[2][tickers] - g[6][tickers])
        b["f1_f6"] = self._csrank(g[1][tickers] - g[6][tickers])
        b["f3_f6"] = self._csrank(g[3][tickers] - g[6][tickers])
        
        X = np.hstack([v[tickers].values.reshape(-1, 1) for v in b.values()])
        return X, tickers, b["f6"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(
            self.params,
            xgb.DMatrix(X[ok], label=y[ok]),
            num_boost_round=self.n_rounds,
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None, "Must call train() before predict()"
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred_matrix = pd.DataFrame(
            raw.reshape(-1, len(tickers)), index=index, columns=tickers
        )
        
        # Causal position smoothing to crush 5bp transaction cost
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        
        # Causal removal of static per-instrument tilt
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        
        # Cross-sectional de-meaning (zero net market exposure)
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)
