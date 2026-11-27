import sys
from pathlib import Path
import numpy as np, pandas as pd
SCRIPT_DIR = Path("D:/Projects/Quant/quant-competition-research/03-alphanova/comp6s2_scientist_release")
sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from submission_top10_novel import AlphaNovaTop10NovelPredictor
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

all_gf = []
for idx, item in enumerate(harness._cached_val_data):
    pred = AlphaNovaTop10NovelPredictor()
    val_feat = item["val_features"]
    tickers = sorted(list(item["val_returns"].columns), key=lambda c: int(c.split(".")[-1]))
    pred.ridge_coef_ = np.ones(11)
    res_df = pred.predict(val_feat)
    g_mat = harness._cached_gauge[idx]
    if g_mat is not None:
        p_v = res_df.values - res_df.values.mean(axis=1, keepdims=True)
        n = min(len(g_mat), len(p_v))
        gf = compute_gauge_fixed(p_v[:n], g_mat[:n])
        all_gf.append(gf)

city = compute_city(np.concatenate(all_gf, axis=0))
comp = compare_to_cities(city, city_db_path=db_path)
print("=" * 60)
print("NOVELTY VERIFICATION:")
print(f"Angular distance to nearest database signal: {comp.iloc[0]['angular_distance_deg']:.2f}°")
print(f"Nearest signal ID: {comp.iloc[0]['predictor_id']}")
print(f"Dim0 (Aligned Alpha IC): {city[0]:+.4f}")
print("=" * 60)
