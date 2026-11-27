"""Check Attempt 7 HybridAdaptiveAlpha city distance."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from submission import HybridAdaptiveAlpha
from evaluation import backtest
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

all_gf = []
for idx, item in enumerate(harness._cached_val_data):
    feat = item["val_features"]
    ret = item["val_returns"]
    tgt = item["val_target"]
    
    m = HybridAdaptiveAlpha()
    m.train(feat, tgt)
    pred = m.predict(feat)
    
    g_mat = harness._cached_gauge[idx]
    if g_mat is not None:
        pred_vals = pred.values
        pred_vals = pred_vals - pred_vals.mean(axis=1, keepdims=True)
        n_use = min(len(g_mat), len(pred_vals))
        gf = compute_gauge_fixed(pred_vals[:n_use], g_mat[:n_use])
        all_gf.append(gf)

concat_gf = np.concatenate(all_gf, axis=0)
city = compute_city(concat_gf)
comp_df = compare_to_cities(city, city_db_path=city_db_path)
print("Attempt #7 Nearest Competitors:")
print(comp_df.head(5))
