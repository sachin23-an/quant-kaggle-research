"""
Bi-weekly Cross-Sectional Signal Forecasting — Season 2
AlphaNova Competition Submission
Model: PureInverseAlpha
Author: Sachin Kumar (sachin23-an)
"""

import warnings
import numpy as np
import pandas as pd
from predictor import Predictor

warnings.filterwarnings("ignore")


class PureInverseAlpha(Predictor):
    """
    Inverse Contemporaneous Cross-Sectional Alpha.
    
    Architecture:
    1. Mean-reversion on the primary contemporaneous cluster (Feature 1 and Feature 3).
    2. Causal position smoothing with span=180 to minimize the 5bp turnover drag.
    3. Strict zero-beta cross-sectional de-meaning per timestamp.
    """

    def __init__(self):
        self.span = 180

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        pass

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        f1 = features["Feature.1"]
        f3 = features["Feature.3"]
        core = (f1 + f3) / 2.0
        
        # Cross-sectional rank normalized to [-1, 1]
        r = core.rank(axis=1)
        n = core.notna().sum(axis=1).values.reshape(-1, 1)
        ranked = (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))
        
        # Mean reversion: inverse of the contemporaneous cluster
        signal = -ranked
        
        # Causal position smoothing (span=180) to control 5bp transaction fee
        signal = signal.ewm(span=self.span, min_periods=1).mean()
        
        # Zero-beta cross-sectional de-meaning
        signal = signal.sub(signal.mean(axis=1), axis=0)
        return signal.fillna(0.0)
