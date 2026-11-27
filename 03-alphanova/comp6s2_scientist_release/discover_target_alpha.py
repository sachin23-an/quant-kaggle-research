"""Systematic search for feature representations that maximize correlation with TARGET."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest, TRANSACTION_COST
from city_tools import compute_gauge_fixed, compute_city, compute_ic


def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))


def analyze():
    harness = ResearchHarness()
    harness.preload_validation_cache(verbose=False)

    print("=" * 80)
    print("DECODING TARGET PREDICTABILITY ACROSS ALL 31 PERIODS")
    print("=" * 80)

    # 1. First, test simple rank correlations of individual features with target
    # Feature 1..6 raw, squared, signs, and all pairs
    feats_raw_ic = {f"F{i}": [] for i in range(1, 7)}
    for item in harness._cached_val_data:
        feat = item["val_features"]
        tgt = item["val_target"]
        tickers = list(tgt.columns)

        for i in range(1, 7):
            rf = csrank(feat[f"Feature.{i}"][tickers])
            # Pearson correlation with target per row
            dot = (rf * tgt[tickers]).sum(axis=1)
            norm_r = np.sqrt((rf**2).sum(axis=1))
            norm_t = np.sqrt((tgt[tickers]**2).sum(axis=1))
            ic = dot / (norm_r * norm_t + 1e-9)
            feats_raw_ic[f"F{i}"].append(ic.mean())

    print("Individual Feature Mean IC vs Target:")
    for f_name, ics in feats_raw_ic.items():
        print(f"  {f_name}: {np.mean(ics):+.4f} (std={np.std(ics):.4f})")

    # 2. Test multi-feature interactions
    # Let's test a linear Ridge model trained on expanding windows predicting target
    print("\n--- Testing Machine Learning Models Predicting Target Directly ---")

    for model_type in ["ridge_linear", "ridge_poly", "lgbm_shallow"]:
        all_val_ic = []
        all_val_sharpe = []
        all_city_dim0 = []

        for p_idx, p_key in enumerate(harness.periods):
            # Load full training data for this period
            data = harness.data_dir
            from data_loader import load_period

            d = load_period(p_key, str(harness.data_dir), harness.universe)
            n_total = len(d["features"])
            n_val = min(800, int(0.2 * n_total))
            n_train = n_total - n_val

            tr_feat = d["features"].iloc[:n_train]
            tr_tgt = d["target"].iloc[:n_train]
            va_feat = d["features"].iloc[n_train:]
            va_tgt = d["target"].iloc[n_train:]
            va_ret = d["returns"].iloc[n_train:]
            tickers = list(tr_tgt.columns)

            def build_design_matrix(f_df):
                g = {i: f_df[f"Feature.{i}"][tickers] for i in range(1, 7)}
                cols = []
                # Ranks
                for i in range(1, 7):
                    cols.append(csrank(g[i]))
                # Pairwise ratios/differences
                for i in range(1, 7):
                    for j in range(i + 1, 7):
                        cols.append(csrank(g[i] - g[j]))
                        if model_type == "ridge_poly":
                            cols.append(csrank(g[i] * g[j]))
                X = np.hstack([c.values.reshape(-1, 1) for c in cols])
                return X

            X_tr = build_design_matrix(tr_feat)
            y_tr = tr_tgt[tickers].values.reshape(-1)
            ok_tr = np.isfinite(y_tr) & np.isfinite(X_tr).all(axis=1)

            X_va = build_design_matrix(va_feat)
            y_va = va_tgt[tickers].values.reshape(-1)

            if model_type.startswith("ridge"):
                model = Ridge(alpha=1000.0, fit_intercept=False)
                model.fit(X_tr[ok_tr], y_tr[ok_tr])
                raw_pred = np.nan_to_num(X_va) @ model.coef_
            elif model_type == "lgbm_shallow":
                import lightgbm as lgb

                ds = lgb.Dataset(X_tr[ok_tr], label=y_tr[ok_tr])
                params = {
                    "objective": "regression",
                    "learning_rate": 0.05,
                    "num_leaves": 7,
                    "max_depth": 3,
                    "min_child_samples": 200,
                    "feature_fraction": 0.8,
                    "verbose": -1,
                    "n_jobs": 4,
                }
                bst = lgb.train(params, ds, num_boost_round=30)
                raw_pred = bst.predict(np.nan_to_num(X_va))

            pred_df = pd.DataFrame(raw_pred.reshape(-1, len(tickers)), index=va_feat.index, columns=tickers)
            # Demean
            pred_df = pred_df.sub(pred_df.mean(axis=1), axis=0)

            # Compute IC
            dot = (pred_df * va_tgt[tickers]).sum(axis=1)
            norm_p = np.sqrt((pred_df**2).sum(axis=1))
            norm_t = np.sqrt((va_tgt[tickers]**2).sum(axis=1))
            ic = (dot / (norm_p * norm_t + 1e-9)).mean()
            all_val_ic.append(ic)

            # Smooth for Sharpe
            sig = pred_df.ewm(span=240, min_periods=1).mean()
            sig = sig.sub(sig.mean(axis=1), axis=0)
            pnl = backtest(sig, va_ret[tickers])
            sh = pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
            all_val_sharpe.append(sh)

        print(f"Model: {model_type:<15} | Mean IC: {np.mean(all_val_ic):+.4f} (min={np.min(all_val_ic):+.4f}, max={np.max(all_val_ic):+.4f}) | Mean Net Sharpe: {np.mean(all_val_sharpe):+.4f}")


if __name__ == "__main__":
    analyze()
