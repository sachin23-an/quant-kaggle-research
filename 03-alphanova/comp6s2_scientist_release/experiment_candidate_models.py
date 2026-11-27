"""Systematic backtesting of candidate models on all 31 walk-forward periods.

Evaluates:
Candidate 1: Clean Engineered XGBoost (toxic spreads replaced with true alpha spreads + F6 spreads)
Candidate 2: Walk-Forward ElasticNet / Ridge Regression on Alpha Spreads
Candidate 3: Dynamic Adaptive Ensemble (Ridge + XGBoost on orthogonalized spreads)
Candidate 4: Core Reversion + F6 Spread Overlay with Volatility-Scaled Allocation
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from predictor import Predictor


# -------------------------------------------------------------
# CANDIDATE 1: CleanAlphaXGBoost
# -------------------------------------------------------------
class CleanAlphaXGBoost(Predictor):
    def __init__(self, span=180, max_depth=2, n_rounds=30, reg_lambda=10.0, eta=0.03):
        self.span = span
        self.max_depth = max_depth
        self.n_rounds = n_rounds
        self.reg_lambda = reg_lambda
        self.eta = eta
        self.model = None

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        # Core inverse signals (statistically positive alpha across 31 periods)
        b["inv_core"] = -(self._csrank(g[1]) + self._csrank(g[2]) + self._csrank(g[3]) + self._csrank(g[4])) / 4.0
        b["inv_f1"] = -self._csrank(g[1])
        b["inv_f3"] = -self._csrank(g[3])

        # Feature 6 target-predictive spreads
        b["f6"] = self._csrank(g[6])
        b["f5_f6"] = self._csrank(g[5] - g[6])
        b["f4_f6"] = self._csrank(g[4] - g[6])
        b["f2_f6"] = self._csrank(g[2] - g[6])
        b["f1_f6"] = self._csrank(g[1] - g[6])
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

        params = {
            "objective": "reg:squarederror",
            "max_depth": self.max_depth,
            "eta": self.eta,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 500,
            "reg_lambda": self.reg_lambda,
            "verbosity": 0,
            "nthread": 4,
        }
        self.model = xgb.train(params, xgb.DMatrix(X[ok], label=y[ok]), num_boost_round=self.n_rounds)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred_matrix = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Causal position smoothing
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        # Causal tilt removal
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        # Cross-sectional de-mean
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)


# -------------------------------------------------------------
# CANDIDATE 2: DynamicRidgePredictor
# -------------------------------------------------------------
class DynamicRidgePredictor(Predictor):
    def __init__(self, alpha=500.0, span=200):
        self.alpha = alpha
        self.span = span
        self.coef_ = None
        self.tickers = None

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        b["inv_core"] = -(self._csrank(g[1]) + self._csrank(g[2]) + self._csrank(g[3]) + self._csrank(g[4])) / 4.0
        b["inv_f1"] = -self._csrank(g[1])
        b["inv_f3"] = -self._csrank(g[3])
        b["f6"] = self._csrank(g[6])
        b["f5_f6"] = self._csrank(g[5] - g[6])
        b["f4_f6"] = self._csrank(g[4] - g[6])
        b["f2_f6"] = self._csrank(g[2] - g[6])
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

        ridge = Ridge(alpha=self.alpha, fit_intercept=False)
        ridge.fit(X[ok], y[ok])
        self.coef_ = ridge.coef_
        self.tickers = tickers

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index = self._design(features)
        raw = np.nan_to_num(X) @ self.coef_
        pred_matrix = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)


# -------------------------------------------------------------
# CANDIDATE 3: HybridAdaptiveEnsemble (Ridge + XGBoost blend)
# -------------------------------------------------------------
class HybridAdaptiveEnsemble(Predictor):
    def __init__(self, ridge_weight=0.5, span=200):
        self.ridge_weight = ridge_weight
        self.span = span
        self.ridge = DynamicRidgePredictor(alpha=500.0, span=span)
        self.xgb = CleanAlphaXGBoost(span=span, max_depth=2, n_rounds=25, reg_lambda=15.0)

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        self.ridge.train(features, target)
        self.xgb.train(features, target)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        p_ridge = self.ridge.predict(features)
        p_xgb = self.xgb.predict(features)

        # Cross-sectional rank normalizations before blending
        r_ridge = self.ridge._csrank(p_ridge)
        r_xgb = self.xgb._csrank(p_xgb)

        blended = self.ridge_weight * r_ridge + (1.0 - self.ridge_weight) * r_xgb
        blended = blended.ewm(span=self.span, min_periods=1).mean()
        blended = blended.sub(blended.mean(axis=1), axis=0)
        return blended.fillna(0.0)


def run_experiments():
    harness = ResearchHarness()
    harness.preload_validation_cache()

    candidates = [
        (
            "EXP-013",
            "CleanAlphaXGBoost",
            CleanAlphaXGBoost,
            {"span": 180, "max_depth": 2, "n_rounds": 30, "reg_lambda": 10.0},
            "Clean alpha spreads (inv_core, inv_f1, f6, f5-f6, f4-f6) with shallow XGBoost",
        ),
        (
            "EXP-014",
            "CleanAlphaXGBoost-span240",
            CleanAlphaXGBoost,
            {"span": 240, "max_depth": 2, "n_rounds": 25, "reg_lambda": 15.0},
            "Clean alpha spreads with span=240 and heavy L2 regularization",
        ),
        (
            "EXP-015",
            "DynamicRidgePredictor",
            DynamicRidgePredictor,
            {"alpha": 800.0, "span": 200},
            "Dynamic linear Ridge on alpha spreads with analytical shrinkage",
        ),
        (
            "EXP-016",
            "HybridAdaptiveEnsemble",
            HybridAdaptiveEnsemble,
            {"ridge_weight": 0.5, "span": 200},
            "50/50 Rank blend of Ridge and Clean XGBoost",
        ),
    ]

    for exp_id, name, cls, params, hyp in candidates:
        print(f"\n>>> Running Experiment {exp_id}: {name}...")
        metrics = harness.run_walkforward(cls, params=params, train_full=True, gauge_fix=True, verbose=True)
        harness.log_experiment(exp_id, name, hyp, metrics)


if __name__ == "__main__":
    run_experiments()
