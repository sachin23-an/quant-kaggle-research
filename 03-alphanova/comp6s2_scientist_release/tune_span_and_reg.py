"""
Test span and concentration scaling across all 31 walk-forward periods.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPT_DIR = Path("D:/Projects/Quant/quant-competition-research/03-alphanova/comp6s2_scientist_release")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest, check_cross_sectional_demean
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

def build_features(features: pd.DataFrame, tickers: list):
    g = {i: features[f"Feature.{i}"][tickers] for i in range(1, 7)}
    b = {}
    b["inv_f1"] = -csrank(g[1])
    b["inv_f2"] = -csrank(g[2])
    b["inv_f3"] = -csrank(g[3])
    b["inv_f4"] = -csrank(g[4])
    b["inv_core"] = -(csrank(g[1]) + csrank(g[2]) + csrank(g[3]) + csrank(g[4])) / 4.0
    b["f6_f1"] = csrank(g[6] - g[1])
    b["f6_f2"] = csrank(g[6] - g[2])
    b["f6_f3"] = csrank(g[6] - g[3])
    b["f6_f4"] = csrank(g[6] - g[4])
    b["f6_f5"] = csrank(g[6] - g[5])
    b["f3_f5"] = csrank(g[3] - g[5])
    b["f5_f4"] = csrank(g[5] - g[4])
    b["f2_f5"] = csrank(g[2] - g[5])
    b["inv_f6"] = -csrank(g[6])
    
    X = np.hstack([d.values.reshape(-1, 1) for d in b.values()])
    return X

for span in [60, 100, 150, 200, 250, 300]:
    for alpha in [1000.0, 2000.0, 4000.0]:
        val_pnls = []
        all_ics = []
        for period in harness.periods:
            val_features = harness.val_features[period]
            val_returns = harness.val_returns[period]
            val_target = harness.val_target[period]
            tickers = sorted(list(val_returns.columns), key=lambda c: int(c.split(".")[-1]))
            
            # Train Ridge
            X_train = build_features(val_features, tickers)  # In real walk-forward train is pre-val
            # Let's use harness to get proper train split
            pass
        break
    break

print("Testing harness...")
