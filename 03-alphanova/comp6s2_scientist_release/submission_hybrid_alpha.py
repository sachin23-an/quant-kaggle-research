"""Bi-weekly Cross-Sectional Signal Forecasting — Season 2
AlphaNova Competition Submission
Model: HybridAdaptiveAlpha
Author: Sachin Kumar (sachin23-an)
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from predictor import Predictor

warnings.filterwarnings("ignore")


class HybridAdaptiveAlpha(Predictor):
    """
    Hybrid Adaptive Alpha for AlphaNova Season 2.
    
    Architecture:
    1. Core Valuation Reversion: Contemporaneous negation of cross-sectional 
       cluster features (F1..F4), capturing structural asset mispricings.
    2. Adaptive ML Spread Overlay: Walk-forward regularized Ridge regression (L2) 
       trained dynamically each period to learn optimal loadings across 
       Feature 6 relative-value spreads.
    3. Optimal Causal Position Smoothing: Exponential decay (span=180) to 
       crush 5 bps hourly rebalancing friction.
    4. Causal Per-Instrument Tilt Removal: Subtracts expanding historical mean 
       of predictions to eliminate static long/short bias and satisfy the overfitting gate.
    5. Cross-Sectional Zero-Beta Neutrality: Demeans predictions across all 
       instruments at every timestamp.
    """

    def __init__(self, core_weight: float = 0.65, span: int = 180, ridge_alpha: float = 1000.0):
        self.core_weight = core_weight
        self.span = span
        self.ridge_alpha = ridge_alpha
        self.coef_ = None

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        """Contemporaneous cross-sectional rank normalized to [-1, 1], NaN-aware."""
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _spread_blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        b["f6"] = self._csrank(g[6])
        b["f5_f6"] = self._csrank(g[5] - g[6])
        b["f4_f6"] = self._csrank(g[4] - g[6])
        b["f2_f6"] = self._csrank(g[2] - g[6])
        b["f1_f6"] = self._csrank(g[1] - g[6])
        b["f5_f1"] = self._csrank(g[5] - g[1])
        return b

    def _design(self, features: pd.DataFrame):
        b = self._spread_blocks(features)
        tickers = list(b[next(iter(b))].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b[next(iter(b))].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = -target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)

        ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        ridge.fit(X[ok], y[ok])
        self.coef_ = ridge.coef_

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        tickers = list(g[1].columns)
        index = g[1].index

        # 1. Core cross-sectional reversion signal
        r1 = self._csrank(g[1][tickers])
        r2 = self._csrank(g[2][tickers])
        r3 = self._csrank(g[3][tickers])
        r4 = self._csrank(g[4][tickers])
        core_signal = -(r1 + r2 + r3 + r4) / 4.0

        # 2. Adaptive ML spread signal
        if self.coef_ is not None:
            X, _, _ = self._design(features)
            raw_ml = np.nan_to_num(X) @ self.coef_
            ml_signal = pd.DataFrame(raw_ml.reshape(-1, len(tickers)), index=index, columns=tickers)
            ml_signal = self._csrank(ml_signal)
        else:
            ml_signal = pd.DataFrame(0.0, index=index, columns=tickers)

        # 3. Dynamic Blend
        w = self.core_weight
        raw_composite = w * core_signal + (1.0 - w) * ml_signal

        # 4. Causal position smoothing to crush 5bp transaction fee
        pred_matrix = raw_composite.ewm(span=self.span, min_periods=1).mean()

        # 5. Causal per-instrument tilt removal (expanding historical mean)
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)

        # 6. Strictly cross-sectional de-meaning (zero beta)
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)
