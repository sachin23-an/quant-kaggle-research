"""Check distances from Pure Core to all top 10 leaderboard signals."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities

def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

all_gf = []
for idx, item in enumerate(harness._cached_val_data):
    feat = item["val_features"]
    ret = item["val_returns"]
    tgt = item["val_target"]
    tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
    g = {i: feat[f"Feature.{i}"][tickers] for i in range(1, 5)}

    sig = -(csrank(g[1]) + csrank(g[2]) + csrank(g[3]) + csrank(g[4])) / 4.0
    sig = sig.sub(sig.mean(axis=1), axis=0)

    smoothed = sig.ewm(span=200, min_periods=1).mean()
    smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)

    g_mat = harness._cached_gauge[idx]
    if g_mat is not None:
        pred_vals = smoothed.values
        pred_vals = pred_vals - pred_vals.mean(axis=1, keepdims=True)
        n_use = min(len(g_mat), len(pred_vals))
        gf = compute_gauge_fixed(pred_vals[:n_use], g_mat[:n_use])
        all_gf.append(gf)

concat_gf = np.concatenate(all_gf, axis=0)
city = compute_city(concat_gf)
comp_df = compare_to_cities(city, city_db_path=city_db_path)

print("Nearest 15 Signals to Pure Core in the entire competition database:")
print(comp_df.head(15))
