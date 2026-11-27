"""Test Pure Valuation Spread Model across all 31 walk-forward periods."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from predictor import Predictor


class PureValuationSpreadAlpha(Predictor):
    """
    Pure Valuation Spread Alpha for AlphaNova Season 2.
    
    Exploits structural mispricings within the cross-sectional valuation cluster (F1..F4):
    - Negative spread differences: -(F1-F2), -(F1-F3), -(F1-F4), -(F2-F4), -(F2-F3)
    - Inverse cluster level: -(F1+F2+F3+F4)/4
    - Dynamic walk-forward non-negative Ridge regression
    - Heavy position smoothing (span=240) to crush turnover friction
    - Causal tilt removal and zero-beta de-meaning
    """

    def __init__(self, alpha: float = 800.0, span: int = 240):
        self.alpha = alpha
        self.span = span
        self.coef_ = None

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _spread_blocks(self, features: pd.DataFrame, tickers: list) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"][tickers] for i in range(1, 5)}
        b = {}
        # Core level
        b["inv_core"] = -(self._csrank(g[1]) + self._csrank(g[2]) + self._csrank(g[3]) + self._csrank(g[4])) / 4.0
        b["inv_f1"] = -self._csrank(g[1])
        b["inv_f3"] = -self._csrank(g[3])

        # Pairwise spreads with proven positive empirical alpha
        b["neg_d12"] = -self._csrank(g[1] - g[2])
        b["neg_d13"] = -self._csrank(g[1] - g[3])
        b["neg_d14"] = -self._csrank(g[1] - g[4])
        b["neg_d24"] = -self._csrank(g[2] - g[4])
        b["neg_d23"] = -self._csrank(g[2] - g[3])
        return b

    def _design(self, features: pd.DataFrame, tickers: list):
        b = self._spread_blocks(features, tickers)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, b[next(iter(b))].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
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
            g = {i: features[f"Feature.{i}"][tickers] for i in range(1, 5)}
            fallback = -(self._csrank(g[1]) + self._csrank(g[2]) + self._csrank(g[3]) + self._csrank(g[4])) / 4.0
            raw = fallback.values.reshape(-1)

        pred_matrix = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Causal position smoothing
        pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()

        # Causal expanding-mean tilt subtraction
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)

        # Cross-sectional zero-beta de-mean
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix[tickers].fillna(0.0)


if __name__ == "__main__":
    harness = ResearchHarness()
    harness.preload_validation_cache(verbose=False)

    print("=" * 80)
    print("TESTING PURE VALUATION SPREAD ALPHA ACROSS ALL 31 PERIODS")
    print("=" * 80)

    for sp in [180, 240, 300]:
        for alp in [200.0, 500.0, 1000.0]:
            m = harness.run_walkforward(PureValuationSpreadAlpha, params={"alpha": alp, "span": sp}, train_full=True, gauge_fix=True)
            print(f"Span={sp:3d} | Alpha={alp:6.1f} | Net Sharpe={m['net_sharpe']:+.4f} | Gross={m['gross_sharpe']:+.4f} | WinRate={m['win_rate_pct']:.1f}% | IC={m.get('ic', 0.0):+.4f} | Novelty={m.get('city_novelty_deg', 0.0):.1f}° (Nearest: {m.get('nearest_city', 'N/A')})")
