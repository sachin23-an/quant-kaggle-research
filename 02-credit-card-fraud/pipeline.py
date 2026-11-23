#!/usr/bin/env python3
"""
=============================================================================
Kaggle Playground S6E11: Credit Card Fraud Detection Pipeline
=============================================================================
Author:       Sachin Kumar (sachin23-an)
Competition:  Playground Series Season 6 Episode 11 (Credit Card Fraud)
Target:       'is_fraud' (Binary classification with severe ~1.67% class imbalance)
Metric:       ROC-AUC
Methodology:  Spline-Logistic (Continuous Additive) + Shallow LightGBM Rank Blend
=============================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import lightgbm as lgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("FraudPipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing train.csv and test.csv (default: ./data)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of Stratified K-Fold CV splits (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/submission.csv",
        help="Path to save final Kaggle submission CSV (default: ./data/submission.csv)",
    )
    return parser.parse_args()


def generate_benchmark_demo_data(data_dir: str):
    """
    Generates a realistic synthetic dataset mimicking the exact Kaggle S6E11 schema
    if raw Kaggle CSVs are not found locally. This enables instant zero-setup execution.
    """
    logger.info("Raw Kaggle dataset not found. Generating benchmark demo data (5,000 samples)...")
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(42)

    n_samples = 5000
    n_test = 1500
    fraud_rate = 0.0167  # 1.67% class imbalance

    # Generate train features
    train_ids = np.arange(n_samples)
    time = np.sort(np.random.uniform(0, 172800, n_samples))
    amount = np.random.exponential(scale=88.0, size=n_samples)

    # V1 to V28 PCA features
    v_features = np.random.normal(0, 1, size=(n_samples, 28))

    # Additive log-odds generation
    log_odds = (
        -4.5
        + 0.8 * v_features[:, 3]    # V4
        + 0.7 * v_features[:, 10]   # V11
        - 1.1 * v_features[:, 13]   # V14
        - 0.9 * v_features[:, 11]   # V12
        + 0.3 * np.log1p(amount)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    is_fraud = (np.random.uniform(0, 1, n_samples) < prob).astype(int)

    # Ensure target matches exact class balance
    train_data = {
        "id": train_ids,
        "Time": time,
        "Amount": amount,
        **{f"V{i+1}": v_features[:, i] for i in range(28)},
        "is_fraud": is_fraud,
    }
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(Path(data_dir) / "train.csv", index=False)

    # Generate test features
    test_ids = np.arange(n_samples, n_samples + n_test)
    test_time = np.sort(np.random.uniform(0, 172800, n_test))
    test_amount = np.random.exponential(scale=88.0, size=n_test)
    test_v = np.random.normal(0, 1, size=(n_test, 28))

    test_data = {
        "id": test_ids,
        "Time": test_time,
        "Amount": test_amount,
        **{f"V{i+1}": test_v[:, i] for i in range(28)},
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(Path(data_dir) / "test.csv", index=False)
    logger.info(f"Demo benchmark dataset generated successfully in {data_dir}.")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs domain-specific features for fraud detection:
    1. Log transforms of heavy-tailed monetary amounts
    2. Cyclic temporal representations (diurnal fraud patterns)
    3. High-order interaction signals on dominant principal components
    """
    df = df.copy()

    # Log-transformed transaction amount
    df["log_amount"] = np.log1p(np.maximum(df["Amount"], 0))

    # Cyclical hour features (24-hour periodic cycle)
    hour = (df["Time"] // 3600) % 24
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Key PCA feature interaction terms (strongest signal cross-products)
    if "V4" in df.columns and "V11" in df.columns:
        df["V4_x_V11"] = df["V4"] * df["V11"]
    if "V14" in df.columns and "V12" in df.columns:
        df["V14_x_V12"] = df["V14"] * df["V12"]
    if "V10" in df.columns and "V17" in df.columns:
        df["V10_x_V17"] = df["V10"] * df["V17"]

    # Amount relative to key components
    if "V2" in df.columns:
        df["Amount_div_V2"] = df["log_amount"] / (np.abs(df["V2"]) + 1e-4)

    return df


def build_spline_pipeline(features: list) -> Pipeline:
    """
    Builds a Generalized Additive Spline + L2 Logistic Regression Pipeline.
    SplineTransformer non-linearly expands univariate features via B-splines,
    capturing smooth continuous synthetic manifolds that tree splits approximate poorly.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("splines", SplineTransformer(n_knots=5, degree=3, extrapolation="linear")),
        ("logistic", LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def build_lgb_classifier() -> lgb.LGBMClassifier:
    """
    Builds a shallow, regularized LightGBM classifier.
    Shallow trees (max_depth=5, num_leaves=24) prevent overfitting on synthetic CTGAN tails.
    """
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=24,
        min_child_samples=50,
        colsample_bytree=0.8,
        subsample=0.8,
        scale_pos_weight=3.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def run_pipeline(data_dir: str, n_splits: int, output_path: str):
    train_path = Path(data_dir) / "train.csv"
    test_path = Path(data_dir) / "test.csv"

    if not train_path.exists() or not test_path.exists():
        generate_benchmark_demo_data(data_dir)

    logger.info(f"Loading training data from {train_path}...")
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    # Determine target column (is_fraud)
    target_col = "is_fraud"
    if target_col not in train_raw.columns:
        raise ValueError(f"Target column '{target_col}' not found in training dataset.")

    y = train_raw[target_col].values
    fraud_rate = np.mean(y) * 100
    logger.info(f"Loaded {len(train_raw):,} training samples. Class Imbalance: {fraud_rate:.2f}% positive.")

    # Apply Feature Engineering
    logger.info("Applying feature engineering (spline transformations, cyclical time, PCA interactions)...")
    train_feat = engineer_features(train_raw)
    test_feat = engineer_features(test_raw)

    drop_cols = ["id", target_col]
    feature_cols = [c for c in train_feat.columns if c not in drop_cols]
    logger.info(f"Engineered feature count: {len(feature_cols)} features.")

    X = train_feat[feature_cols].values
    X_test = test_feat[feature_cols].values

    # Select core continuous features for Spline expansion (V1-V28 + log_amount)
    spline_feature_indices = [
        i for i, col in enumerate(feature_cols)
        if col.startswith("V") and "_" not in col or col == "log_amount"
    ]

    # Initialize cross-validation arrays
    oof_spline = np.zeros(len(train_raw))
    oof_lgb = np.zeros(len(train_raw))
    test_preds_spline = np.zeros(len(test_raw))
    test_preds_lgb = np.zeros(len(test_raw))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    logger.info(f"Starting {n_splits}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # 1. Train Spline-Logistic Model
        spline_model = build_spline_pipeline(spline_feature_indices)
        spline_model.fit(X_tr[:, spline_feature_indices], y_tr)
        val_pred_spline = spline_model.predict_proba(X_va[:, spline_feature_indices])[:, 1]
        oof_spline[val_idx] = val_pred_spline
        test_preds_spline += spline_model.predict_proba(X_test[:, spline_feature_indices])[:, 1] / n_splits

        # 2. Train Shallow LightGBM
        lgb_model = build_lgb_classifier()
        lgb_model.fit(X_tr, y_tr)
        val_pred_lgb = lgb_model.predict_proba(X_va)[:, 1]
        oof_lgb[val_idx] = val_pred_lgb
        test_preds_lgb += lgb_model.predict_proba(X_test)[:, 1] / n_splits

        # Compute individual fold AUCs
        fold_auc_spline = roc_auc_score(y_va, val_pred_spline)
        fold_auc_lgb = roc_auc_score(y_va, val_pred_lgb)
        # Fold Rank Blend
        val_blend = 0.40 * (rankdata(val_pred_spline) / len(val_pred_spline)) + \
                    0.60 * (rankdata(val_pred_lgb) / len(val_pred_lgb))
        fold_auc_blend = roc_auc_score(y_va, val_blend)

        logger.info(
            f"Fold {fold}/{n_splits} - Spline AUC: {fold_auc_spline:.5f} | "
            f"LightGBM AUC: {fold_auc_lgb:.5f} | Ensemble AUC: {fold_auc_blend:.5f}"
        )

    # 3. Overall Out-Of-Fold Evaluation
    total_auc_spline = roc_auc_score(y, oof_spline)
    total_auc_lgb = roc_auc_score(y, oof_lgb)

    # Rank-average Out-Of-Fold predictions
    oof_rank_spline = rankdata(oof_spline) / len(oof_spline)
    oof_rank_lgb = rankdata(oof_lgb) / len(oof_lgb)
    oof_blend = 0.40 * oof_rank_spline + 0.60 * oof_rank_lgb
    total_auc_blend = roc_auc_score(y, oof_blend)

    logger.info("=" * 65)
    logger.info(f"FINAL OUT-OF-FOLD ROC-AUC EVALUATION:")
    logger.info(f"  1. Continuous Spline Model:      {total_auc_spline:.5f}")
    logger.info(f"  2. Shallow LightGBM Classifier:   {total_auc_lgb:.5f}")
    logger.info(f"  3. Rank Ensemble (Spline + LGBM): {total_auc_blend:.5f}")
    logger.info("=" * 65)

    # 4. Generate Final Submission
    test_rank_spline = rankdata(test_preds_spline) / len(test_preds_spline)
    test_rank_lgb = rankdata(test_preds_lgb) / len(test_preds_lgb)
    final_test_predictions = 0.40 * test_rank_spline + 0.60 * test_rank_lgb

    os.makedirs(Path(output_path).parent, exist_ok=True)
    submission_df = pd.DataFrame({
        "id": test_raw["id"],
        "is_fraud": final_test_predictions,
    })
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Saved Kaggle submission to: {output_path} ({len(submission_df):,} rows).")
    logger.info("Sample submission predictions:")
    print(submission_df.head(5).to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        n_splits=args.n_splits,
        output_path=args.output,
    )
