"""Aligned Target Alpha: Every feature strictly aligned with the target.

Architecture:
1. All input features oriented to have positive correlation with target.
2. Dynamic walk-forward learning via Ridge with heavy L2 shrinkage.
3. Position smoothing (span=240 to 300) to crush turnover below 0.05.
4. Causal expanding-mean tilt subtraction.
5. Strict zero-beta cross-sectional de-meaning.
6. Strictly sorted numeric ticker columns.
"""

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
from predictor import Predictor


class AlignedTargetRidge(Predictor):
    def __init__(self, alpha: float = 1500.0, span: int = 260):
        self.alpha = alpha
        self.span = span
        self.coef_ = None

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features: pd.DataFrame, tickers: list) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"][tickers] for i in range(1, 7)}
        b = {}
        # 1. Inverted core cluster features (all have +0.044 IC with target)
        b["inv_f1"] = -self._csrank(g[1])
        b["inv_f2"] = -self._csrank(g[2])
        b["inv_f3"] = -self._csrank(g[3])
        b["inv_f4"] = -self._csrank(g[4])
        b["inv_core"] = -(self._csrank(g[1]) + self._csrank(g[2]) + self._csrank(g[3]) + self._csrank(g[4])) / 4.0

        # 2. Spreads against Feature 6 (relative value)
        b["f6"] = self._csrank(g[6])
        b["f6_f1"] = self._csrank(g[6] - g[1])
        b["f6_f2"] = self._csrank(g[6] - g[2])
        b["f6_f3"] = self._csrank(g[6] - g[3])
        b["f6_f4"] = self._csrank(g[6] - g[4])
        b["f6_f5"] = self._csrank(g[6] - g[5])

        # 3. Relative valuation differences within core
        b["d12"] = -self._csrank(g[1] - g[2])
        b["d34"] = -self._csrank(g[3] - g[4])
        return b

    def _design(self, features: pd.DataFrame, tickers: list):
        b = self._blocks(features, tickers)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, b[next(iter(b))].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # Strictly sort tickers by numeric index: ticker.1 .. ticker.20
        cols = list(target.columns)
        tickers = sorted(cols, key=lambda c: int(c.split(".")[-1]))

        X, _ = self._design(features, tickers)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)

        ridge = Ridge(alpha=self.alpha, fit_intercept=False, positive=True)
        ridge.fit(X[ok], y[ok])
        self.coef_ = ridge.coef_

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = list(features["Feature.1"].columns)
        tickers = sorted(cols, key=lambda c: int(c.split(".")[-1]))

        X, index = self._design(features, tickers)
        if self.coef_ is not None and np.any(self.coef_ > 0):
            raw = np.nan_to_num(X) @ self.coef_
        else:
            # Fallback to equal-weighted inverted core
            g = {i: features[f"Feature.{i}"][tickers] for i in range(1, 5)}
            fallback = -(self._csrank(g[1]) + self._csrank(g[2]) + self._csrank(g[3]) + self._csrank(g[4])) / 4.0
            raw = fallback.values.reshape(-1)

        pred_matrix = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Causal position smoothing to crush 5bp transaction cost
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()

        # Causal expanding-mean tilt subtraction
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)

        # Cross-sectional zero-beta de-mean
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix[tickers].fillna(0.0)


class AlignedTargetXGBoost(Predictor):
    def __init__(self, span: int = 260, max_depth: int = 2, reg_lambda: float = 25.0):
        self.span = span
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.model = None

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features: pd.DataFrame, tickers: list) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"][tickers] for i in range(1, 7)}
        b = {}
        b["inv_f1"] = -self._csrank(g[1])
        b["inv_f2"] = -self._csrank(g[2])
        b["inv_f3"] = -self._csrank(g[3])
        b["inv_f4"] = -self._csrank(g[4])
        b["inv_core"] = -(self._csrank(g[1]) + self._csrank(g[2]) + self._csrank(g[3]) + self._csrank(g[4])) / 4.0

        b["f6"] = self._csrank(g[6])
        b["f6_f1"] = self._csrank(g[6] - g[1])
        b["f6_f2"] = self._csrank(g[6] - g[2])
        b["f6_f3"] = self._csrank(g[6] - g[3])
        b["f6_f4"] = self._csrank(g[6] - g[4])
        b["f6_f5"] = self._csrank(g[6] - g[5])
        return b

    def _design(self, features: pd.DataFrame, tickers: list):
        b = self._blocks(features, tickers)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, b[next(iter(b))].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        cols = list(target.columns)
        tickers = sorted(cols, key=lambda c: int(c.split(".")[-1]))

        X, _ = self._design(features, tickers)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)

        params = {
            "objective": "reg:squarederror",
            "max_depth": self.max_depth,
            "eta": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 600,
            "reg_lambda": self.reg_lambda,
            "verbosity": 0,
            "nthread": 4,
        }
        self.model = xgb.train(params, xgb.DMatrix(X[ok], label=y[ok]), num_boost_round=30)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = list(features["Feature.1"].columns)
        tickers = sorted(cols, key=lambda c: int(c.split(".")[-1]))

        X, index = self._design(features, tickers)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred_matrix = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix[tickers].fillna(0.0)


def evaluate_models():
    harness = ResearchHarness()
    harness.preload_validation_cache(verbose=False)

    print("=" * 80)
    print("WALK-FORWARD EVALUATION OF UNIDIRECTIONALLY ALIGNED ARCHITECTURES")
    print("=" * 80)

    # 1. AlignedTargetRidge with non-negative shrinkage (positive=True)
    print("\n1. Testing AlignedTargetRidge (positive L2 shrinkage)...")
    m_ridge = harness.run_walkforward(AlignedTargetRidge, params={"alpha": 1500.0, "span": 260}, train_full=True, gauge_fix=True, verbose=True)

    # 2. AlignedTargetXGBoost
    print("\n2. Testing AlignedTargetXGBoost (heavy L2, depth=2)...")
    m_xgb = harness.run_walkforward(AlignedTargetXGBoost, params={"span": 260, "max_depth": 2, "reg_lambda": 25.0}, train_full=True, gauge_fix=True, verbose=True)


if __name__ == "__main__":
    evaluate_models()
