import sys
from pathlib import Path
SCRIPT_DIR = Path("D:/Projects/Quant/quant-competition-research/03-alphanova/comp6s2_scientist_release")
sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from test_upgraded_alpha import UpgradedAlphaPredictor

harness = ResearchHarness()
res = harness.run_walkforward(UpgradedAlphaPredictor, train_full=True, gauge_fix=True, verbose=False)

print("=" * 60)
print("FULL 31-PERIOD WALK-FORWARD METRICS:")
print("=" * 60)
for k, v in res.items():
    if k != "per_period_sharpe" and k != "city":
        print(f"{k:<22}: {v}")
if "city" in res and res["city"] is not None:
    print(f"Dim0 (Aligned Alpha):   {res['city'][0]:+.4f}")
print("=" * 60)
