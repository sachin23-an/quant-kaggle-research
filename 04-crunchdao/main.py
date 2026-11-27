import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def get_model_path(model_directory_path: str):
    return os.path.join(model_directory_path, "model.joblib")


def get_feature_columns(X: pd.DataFrame):
    return [c for c in X.columns if c.startswith("Feature_")]


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_directory_path: str,
) -> None:
    print(f"train() called: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
    feature_cols = get_feature_columns(X_train)
    
    # Subsample recent moons or train on fast regularized Ridge
    # Use last 100 moons for fast high-quality recent signal
    unique_moons = sorted(X_train["moon"].unique())
    recent_moons = unique_moons[-100:] if len(unique_moons) > 100 else unique_moons
    mask = X_train["moon"].isin(recent_moons)
    
    X_sub = X_train.loc[mask, feature_cols]
    y_sub = y_train.loc[mask, "target"]
    
    # Light L2 regularization
    model = Ridge(alpha=100.0, fit_intercept=False)
    model.fit(X_sub, y_sub)
    
    os.makedirs(model_directory_path, exist_ok=True)
    model_path = get_model_path(model_directory_path)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
) -> pd.DataFrame:
    prediction = X_test[["id", "moon"]].copy()
    
    model_path = get_model_path(model_directory_path)
    model = joblib.load(model_path)
    
    feature_cols = get_feature_columns(X_test)
    raw = model.predict(X_test[feature_cols])
    
    # Cross-sectional percentile rank normalization (Numerai style)
    # Per-moon percentile ranking
    prediction["prediction"] = raw
    prediction["prediction"] = (
        prediction.groupby("moon")["prediction"].rank(pct=True).fillna(0.5)
    )
    
    return prediction
