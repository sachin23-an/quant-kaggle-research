import sys
from pathlib import Path
SCRIPT_DIR = Path("D:/Projects/Quant/quant-competition-research/03-alphanova/comp6s2_scientist_release")
sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from submission_top10_novel import AlphaNovaTop10NovelPredictor
from city_tools import compare_to_cities

harness = ResearchHarness()
res = harness.run_walkforward(AlphaNovaTop10NovelPredictor, train_full=True, gauge_fix=True, verbose=False)

db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")
comp = compare_to_cities(res["city"], city_db_path=db_path)

print("=" * 60)
print("ALPHANOVA TOP-10 NOVEL PREDICTOR — 31-PERIOD WALK-FORWARD:")
print("=" * 60)
print(f"Net Sharpe:        {res['net_sharpe']:+.5f}")
print(f"Gross Sharpe:      {res['gross_sharpe']:+.5f}")
print(f"Dim0 (Aligned IC): {res['ic']:+.5f}")
print(f"Nearest Database:  {comp.iloc[0]['angular_distance_deg']:.2f}° to Predictor ID {comp.iloc[0]['signal_id']}")
print(f"Second Nearest:    {comp.iloc[1]['angular_distance_deg']:.2f}° to Predictor ID {comp.iloc[1]['signal_id']}")
print(f"Third Nearest:     {comp.iloc[2]['angular_distance_deg']:.2f}° to Predictor ID {comp.iloc[2]['signal_id']}")
print("=" * 60)
