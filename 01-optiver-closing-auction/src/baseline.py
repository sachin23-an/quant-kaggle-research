"""
Baseline Leak-Free Cross-Validation Pipeline for Optiver Trading at the Close
Integrates:
- PurgedGroupTimeSeriesSplit by date_id with 1-day purge window
- Automated LookAheadAuditor checks
- Microstructure feature engineering & market neutralization
- Out-of-Sample MAE tracking across folds
"""

import sys
import os
import math
from typing import Dict, List, Tuple

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.validation import PurgedGroupTimeSeriesSplit
from shared.lookahead_audit import LookAheadAuditor
from shared.metrics import mean_absolute_error, calc_rank_ic
from features import compute_microstructure_features, compute_cross_sectional_neutralization


class LinearRidgeModel:
    """Zero-dependency pure-python closed-form Ridge Regression for reproducible baselines."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.weights: List[float] = []
        self.intercept: float = 0.0

    def fit(self, X: List[List[float]], y: List[float]) -> "LinearRidgeModel":
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0
        if n_samples == 0 or n_features == 0:
            return self

        # Center data
        y_mean = sum(y) / n_samples
        x_means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]

        # Simple coordinate descent / regularized univariate fit for baseline
        self.weights = [0.0] * n_features
        for j in range(n_features):
            cov = sum((X[i][j] - x_means[j]) * (y[i] - y_mean) for i in range(n_samples))
            var = sum((X[i][j] - x_means[j]) ** 2 for i in range(n_samples)) + self.alpha
            self.weights[j] = cov / var if var > 0 else 0.0

        self.intercept = y_mean - sum(w * xm for w, xm in zip(self.weights, x_means))
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        return [self.intercept + sum(w * x for w, x in zip(self.weights, row)) for row in X]


def run_pipeline():
    print("=================================================================")
    print("  Optiver Closing Auction: Leak-Free Baseline Validation Pipeline")
    print("=================================================================\n")

    auditor = LookAheadAuditor(verbose=True)

    # 1. Generate Synthetic Panel (20 days, 20 stocks per day, 5 time buckets: 2,000 rows)
    # Allows self-contained validation without raw Kaggle files
    print("Generating validation panel...")
    records = []
    dates = []
    targets = []

    for d in range(1, 21):
        for s in range(10):
            for t in [300, 360, 420, 480, 540]:
                row = {
                    "stock_id": s,
                    "date_id": d,
                    "seconds_in_bucket": t,
                    "imbalance_size": 10000.0 * (1.0 + 0.1 * s),
                    "imbalance_buy_sell_flag": 1 if s % 2 == 0 else -1,
                    "reference_price": 100.0 + 0.05 * s,
                    "matched_size": 80000.0,
                    "near_price": 100.2 + 0.05 * s,
                    "far_price": 100.4 + 0.05 * s,
                    "bid_price": 99.9 + 0.05 * s,
                    "bid_size": 4000.0,
                    "ask_price": 100.1 + 0.05 * s,
                    "ask_size": 3500.0,
                    "wap": 100.02 + 0.05 * s,
                }
                # Target: basis point return with microstructure signal + noise
                true_signal = (row["bid_size"] - row["ask_size"]) / (row["bid_size"] + row["ask_size"]) * 5.0
                target_bps = true_signal + (0.5 if d % 2 == 0 else -0.5)
                records.append(row)
                dates.append(d)
                targets.append(target_bps)

    print(f"Total samples: {len(records)} across {len(set(dates))} unique dates.\n")

    # 2. Extract Microstructure Features
    print("Extracting microstructure features...")
    extracted_feats = [compute_microstructure_features(r) for r in records]
    feature_keys = list(extracted_feats[0].keys())

    # Form feature matrix
    X_matrix = [[feats[k] for k in feature_keys] for feats in extracted_feats]

    # 3. Run Pre-Flight Look-Ahead Audit
    print("\nRunning Pre-Flight Look-Ahead Audits...")
    auditor.audit_target_correlation_leakage(X_matrix, targets, feature_names=feature_keys)

    # 4. Cross-Validation: PurgedGroupTimeSeriesSplit
    print("\nExecuting Purged Group Time-Series Walk-Forward Splits...")
    cv = PurgedGroupTimeSeriesSplit(n_splits=4, purge_groups=1, embargo_groups=1)

    fold_maes = []
    fold_rics = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_matrix, groups=dates)):
        train_dates = [dates[i] for i in train_idx]
        test_dates = [dates[i] for i in test_idx]

        # Audit temporal monotonicity and group isolation per fold
        auditor.audit_temporal_monotonicity(train_dates, test_dates, min_purge_gap=1.0)
        auditor.audit_group_isolation(train_dates, test_dates)

        # Slice data
        X_train = [X_matrix[i] for i in train_idx]
        y_train = [targets[i] for i in train_idx]
        X_test = [X_matrix[i] for i in test_idx]
        y_test = [targets[i] for i in test_idx]

        # Fit model strictly on training data
        model = LinearRidgeModel(alpha=10.0)
        model.fit(X_train, y_train)

        # Out-of-sample prediction
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        ric = calc_rank_ic(y_test, preds)

        fold_maes.append(mae)
        fold_rics.append(ric)

        print(
            f"Fold {fold}: Train Dates [{min(train_dates)}..{max(train_dates)}] "
            f"-> Test Dates [{min(test_dates)}..{max(test_dates)}] "
            f"| OOS MAE: {mae:.4f} | Rank IC: {ric:.4f}"
        )

    mean_mae = sum(fold_maes) / len(fold_maes)
    mean_ric = sum(fold_rics) / len(fold_rics)
    print("\n-----------------------------------------------------------------")
    print(f"Overall OOS Results: Mean MAE = {mean_mae:.4f} | Mean Rank IC = {mean_ric:.4f}")
    print("-----------------------------------------------------------------\n")

    summary = auditor.summary()
    print(f"Audit Status: {summary['passed']}/{summary['total_checks']} checks PASSED.")
    assert summary["all_passed"], "Validation audit failed!"


if __name__ == "__main__":
    run_pipeline()
