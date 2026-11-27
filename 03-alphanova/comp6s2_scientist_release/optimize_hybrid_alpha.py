"""Systematic optimization of Hybrid Alpha: Blending Core Reversion with Adaptive ML Spreads."""

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


class HybridCompositeAlpha(Predictor):
    def __init__(self, core_weight=0.6, span=200, ridge_alpha=1000.0, use_vol_scaling=True):
        self.core_weight = core_weight
        self.span = span
        self.ridge_alpha = ridge_alpha
        self.use_vol_scaling = use_vol_scaling
        self.coef_ = None

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _spread_blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        b = {}
        # Distinct, novel spread features
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
        X, _, _ = self._design(features)
        raw_ml = np.nan_to_num(X) @ self.coef_
        ml_signal = pd.DataFrame(raw_ml.reshape(-1, len(tickers)), index=index, columns=tickers)
        ml_signal = self._csrank(ml_signal)

        # 3. Blend Core + Adaptive ML
        w = self.core_weight
        raw_composite = w * core_signal + (1.0 - w) * ml_signal

        # 4. Volatility / Dispersion scaling (if enabled)
        if self.use_vol_scaling:
            # Measure cross-sectional dispersion across the 20 assets
            cs_disp = raw_composite.std(axis=1).replace(0.0, np.nan).fillna(1.0)
            target_disp = cs_disp.median()
            scale_factor = np.clip(cs_disp / target_disp, 0.5, 1.8)
            raw_composite = raw_composite.multiply(scale_factor, axis=0)

        # 5. Causal position smoothing
        pred_matrix = raw_composite.ewm(span=self.span, min_periods=1).mean()

        # 6. Causal removal of static per-instrument tilt
        expanding_mean = pred_matrix.shift(1).expanding(min_periods=1).mean()
        pred_matrix = (pred_matrix - expanding_mean).fillna(0.0)

        # 7. Strictly cross-sectional de-meaning (zero beta)
        pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
        return pred_matrix.fillna(0.0)


def sweep():
    harness = ResearchHarness()
    harness.preload_validation_cache(verbose=False)

    print("=" * 80)
    print("HYBRID COMPOSITE ALPHA PARAMETER SWEEP ACROSS ALL 31 PERIODS")
    print("=" * 80)

    weights = [0.3, 0.5, 0.7, 0.85]
    spans = [150, 180, 220]

    results = []
    for w in weights:
        for sp in spans:
            for vol_scale in [True, False]:
                params = {
                    "core_weight": w,
                    "span": sp,
                    "ridge_alpha": 1000.0,
                    "use_vol_scaling": vol_scale,
                }
                m = harness.run_walkforward(HybridCompositeAlpha, params=params, train_full=True, gauge_fix=True)
                results.append((w, sp, vol_scale, m))
                print(
                    f"CoreWeight={w:.2f} | Span={sp:3d} | VolScale={str(vol_scale):<5} | "
                    f"Net Sharpe={m['net_sharpe']:+.4f} | Gross={m['gross_sharpe']:+.4f} | "
                    f"WinRate={m['win_rate_pct']:.1f}% | IC={m.get('ic', 0.0):+.4f} | "
                    f"City Novelty={m.get('city_novelty_deg', 0.0):.1f}° (Nearest: {m.get('nearest_city', 'N/A')})"
                )

    # Sort by Net Sharpe
    results.sort(key=lambda x: x[3]["net_sharpe"], reverse=True)
    best = results[0]
    print("\n" + "=" * 80)
    print(f"BEST CONFIGURATION: CoreWeight={best[0]}, Span={best[1]}, VolScale={best[2]}")
    print(f"Net Sharpe: {best[3]['net_sharpe']:+.4f}, Gross: {best[3]['gross_sharpe']:+.4f}, Win Rate: {best[3]['win_rate_pct']:.1f}%")
    print(f"IC: {best[3].get('ic', 0.0):+.4f}, City Novelty: {best[3].get('city_novelty_deg', 0.0):.1f}°")
    print("=" * 80)

    harness.log_experiment(
        "EXP-020",
        "HybridCompositeAlpha-Best",
        f"Optimized Hybrid (CoreWeight={best[0]}, Span={best[1]}, VolScale={best[2]})",
        best[3],
    )


if __name__ == "__main__":
    sweep()
