#!/usr/bin/env python3
"""
=============================================================================
Numerai Classic Hedge Fund Alpha Submission Pipeline
=============================================================================
Author:       Sachin Kumar (sachin23-an)
Model Name:   sachin23_an
Dataset:      Numerai v6.0 ("Unity")
Architecture: LightGBM Regressor on Small Orthogonal Feature Set (101 Features)
Target:       Market-neutral 20-day equity returns (CORR20 / MMC20)
=============================================================================
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import lightgbm as lgb
from numerapi import NumerAPI
from dotenv import load_dotenv

# Load credentials from .env in project root or current directory
load_dotenv()
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("NumeraiPipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Numerai Live Alpha Submission Pipeline")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sachin23_an",
        help="Your registered Numerai model name (default: sachin23_an)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to store downloaded tournament parquet files (default: ./data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train model and generate submission.parquet without uploading to Numerai",
    )
    return parser.parse_args()


def get_authenticated_napi():
    """
    Initializes NumerAPI with public and secret credentials.
    Looks for environment variables NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY,
    or falls back to public API client if credentials are not configured.
    """
    public_id = os.environ.get("NUMERAI_PUBLIC_ID")
    secret_key = os.environ.get("NUMERAI_SECRET_KEY")

    if not public_id or not secret_key:
        logger.warning(
            "NUMERAI_PUBLIC_ID and/or NUMERAI_SECRET_KEY environment variables not found."
        )
        logger.warning("Attempting unauthenticated NumerAPI instance (download-only).")
        return NumerAPI()

    return NumerAPI(public_id=public_id, secret_key=secret_key)


def get_active_dataset_version(napi):
    """
    Dynamically identifies the current active dataset version (e.g. v6.0).
    """
    datasets = napi.list_datasets()
    # Filter for active version folders (v6.0 preferred)
    v6_files = [f for f in datasets if f.startswith("v6.0/")]
    if v6_files:
        return "v6.0"
    v5_files = [f for f in datasets if f.startswith("v5.0/")]
    if v5_files:
        return "v5.0"
    return "v6.0"


def load_dataset_features(napi, version, data_dir):
    """
    Downloads and inspects features.json to select the 'small' orthogonal feature set (~101 features).
    This ensures sub-2-minute training time and low memory footprint (< 2 GB RAM).
    """
    features_filename = f"{version}/features.json"
    local_features_path = Path(data_dir) / f"{version}_features.json"

    if not local_features_path.exists():
        logger.info(f"Downloading {features_filename}...")
        napi.download_dataset(features_filename, str(local_features_path))

    with open(local_features_path, "r") as f:
        feature_metadata = json.load(f)

    # Numerai v6.0 provides feature_stats and feature_sets dict
    if "feature_sets" in feature_metadata and "small" in feature_metadata["feature_sets"]:
        feature_cols = feature_metadata["feature_sets"]["small"]
        logger.info(f"Selected 'small' feature set containing {len(feature_cols)} features.")
    else:
        # Fallback to all features beginning with 'feature_'
        all_features = feature_metadata.get("feature_stats", {})
        feature_cols = [k for k in all_features.keys() if k.startswith("feature_")][:100]
        logger.info(f"Fallback: selected first {len(feature_cols)} features.")

    return feature_cols


def run_pipeline(model_name: str, data_dir: str, dry_run: bool = False):
    os.makedirs(data_dir, exist_ok=True)
    napi = get_authenticated_napi()

    # 1. Dataset Discovery
    version = get_active_dataset_version(napi)
    logger.info(f"Active Numerai Tournament Dataset Version: {version}")

    # 2. Extract Feature Columns
    feature_cols = load_dataset_features(napi, version, data_dir)

    # 3. Download & Load Training Dataset
    train_file = f"{version}/train.parquet"
    local_train_path = Path(data_dir) / f"{version}_train.parquet"
    if not local_train_path.exists():
        logger.info(f"Downloading training data: {train_file} (this may take a minute on first run)...")
        napi.download_dataset(train_file, str(local_train_path))

    logger.info(f"Loading training data from {local_train_path}...")
    # Load only necessary columns to minimize RAM
    cols_to_load = ["date", "target"] + [col for col in feature_cols if col != "target"]
    train_df = pd.read_parquet(local_train_path, columns=cols_to_load)
    logger.info(f"Train dataset loaded: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns.")

    # Convert non-numeric columns (e.g. feature_country string) to category for LightGBM
    cat_cols = [
        col for col in feature_cols
        if col in train_df.columns and not pd.api.types.is_numeric_dtype(train_df[col])
    ]
    for col in cat_cols:
        train_df[col] = train_df[col].astype("category")

    # Drop rows where target is NaN (if any)
    train_df = train_df.dropna(subset=["target"])
    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    # 4. Model Training: LightGBM Regressor
    logger.info("Training LightGBM Regressor (350 trees, max_depth=5, lr=0.03)...")
    model = lgb.LGBMRegressor(
        n_estimators=350,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
        n_jobs=-1,
        importance_type="gain",
    )
    model.fit(X_train, y_train)
    logger.info("Model training completed successfully.")

    # Clean up train memory
    del train_df, X_train, y_train

    # 5. Download & Load Live Data
    live_file = f"{version}/live.parquet"
    local_live_path = Path(data_dir) / f"{version}_live.parquet"
    logger.info(f"Downloading current round live data: {live_file}...")
    napi.download_dataset(live_file, str(local_live_path))

    live_cols = ["numerai_ticker"] + feature_cols
    live_df = pd.read_parquet(local_live_path, columns=live_cols)
    logger.info(f"Live dataset loaded: {len(live_df):,} live global assets.")

    # Convert categorical columns
    for col in cat_cols:
        if col in live_df.columns:
            live_df[col] = live_df[col].astype("category")

    # 6. Generate Predictions & Rank Normalization
    logger.info("Generating alpha predictions on live equity universe...")
    raw_preds = model.predict(live_df[feature_cols])

    # Rank-transform predictions to uniform distribution [0, 1] as required by Numerai
    ranked_signals = rankdata(raw_preds) / len(raw_preds)

    # 7. Construct Submission DataFrame
    submission_df = pd.DataFrame({
        "numerai_ticker": live_df["numerai_ticker"],
        "signal": ranked_signals.astype(np.float32),
    })

    sub_path = Path(data_dir) / "submission.parquet"
    submission_df.to_parquet(sub_path, index=False)
    logger.info(f"Saved formatted submission to {sub_path} ({len(submission_df)} rows).")

    # Preview sample signals
    logger.info("Sample live submission rows:")
    print(submission_df.head(5).to_string(index=False))

    # 8. Upload to Numerai
    if dry_run:
        logger.info("[DRY RUN] Prediction generated successfully. Skipped upload.")
        return

    logger.info(f"Connecting to Numerai to submit live signals for model: '{model_name}'...")
    try:
        models = napi.get_models()
        model_id = models.get(model_name)
        if not model_id:
            logger.error(
                f"Model '{model_name}' not found in your account models: {list(models.keys())}."
            )
            return

        submission_id = napi.upload_predictions(str(sub_path), model_id=model_id)
        logger.info(f"SUCCESS: Submission uploaded to Numerai! Submission ID: {submission_id}")
        logger.info(f"Track live scoring at: https://numer.ai/{model_name}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        logger.info(
            "Verify your NUMERAI_PUBLIC_ID and NUMERAI_SECRET_KEY have 'Upload submissions' permission."
        )


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        model_name=args.model_name,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
    )
