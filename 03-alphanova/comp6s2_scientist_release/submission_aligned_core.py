"""
Bi-weekly Cross-Sectional Signal Forecasting — Season 2
AlphaNova Competition Submission
Model: AlignedValuationReversion
Author: Sachin Kumar (sachin23-an)
"""

import warnings
import numpy as np
import pandas as pd
from predictor import Predictor

warnings.filterwarnings("ignore")


class AlignedValuationReversion(Predictor):
    """
    Aligned Valuation Reversion Model for AlphaNova Season 2.
    
    Architecture:
    1. Valuation Reversion Core: Contemporaneous negation of cross-sectional 
       cluster features (Features 1..4), capturing structural asset mispricings.
    2. Relative Value Overlay: Feature 6 spread vs valuation anchor (F6 - F1).
    3. Optimal Position Smoothing: Exponential decay (span=200) to crush 
       5 bps hourly rebalancing friction.
    4. Strict Zero-Beta Neutrality: Demeans predictions across all instruments 
       at every timestamp.
    5. Numeric Ticker Preservation: Strictly aligns and outputs ticker columns 
       in canonical numeric order (ticker.1 .. ticker.20) to guarantee bit-for-bit
       parity with the competition's positional evaluation matrix.
    """

    def __init__(self, span: int = 200, spread_weight: float = 0.20):
        self.span = span
        self.spread_weight = spread_weight

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        """Contemporaneous cross-sectional rank normalized to [-1, 1], NaN-aware."""
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Stationary cross-sectional factor alpha; no retraining needed."""
        pass

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = list(features["Feature.1"].columns)
        # Canonical numeric sorting to eliminate lexicographic positional misalignment
        tickers = sorted(cols, key=lambda c: int(c.split(".")[-1]))

        g = {i: features[f"Feature.{i}"][tickers] for i in range(1, 7)}

        # 1. Valuation reversion core: contemporaneous negation of Features 1..4
        core = -(self._csrank(g[1]) + self._csrank(g[2]) + self._csrank(g[3]) + self._csrank(g[4])) / 4.0

        # 2. Relative value overlay: Feature 6 spread vs valuation anchor
        spread = self._csrank(g[6] - g[1])

        # 3. Composite signal
        sig = core + self.spread_weight * spread

        # 4. Strict cross-sectional de-meaning (zero net market beta)
        sig = sig.sub(sig.mean(axis=1), axis=0)

        # 5. Causal position smoothing to crush 5 bps hourly rebalancing turnover
        smoothed = sig.ewm(span=self.span, min_periods=1).mean()

        # 6. Final de-meaning and column preservation
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        return smoothed[tickers].fillna(0.0)
