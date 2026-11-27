"""Fine-tune the Pareto frontier between Net Sharpe and City Novelty."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from optimize_hybrid_alpha import HybridCompositeAlpha

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)

print("=" * 80)
print("PARETO FRONTIER EXPLORATION: Net Sharpe vs City Novelty (Span=180)")
print("=" * 80)

weights = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]

for w in weights:
    params = {
        "core_weight": w,
        "span": 180,
        "ridge_alpha": 1000.0,
        "use_vol_scaling": False,
    }
    m = harness.run_walkforward(HybridCompositeAlpha, params=params, train_full=True, gauge_fix=True)
    print(
        f"CoreWeight={w:.2f} | Net Sharpe={m['net_sharpe']:+.4f} | Gross={m['gross_sharpe']:+.4f} | "
        f"WinRate={m['win_rate_pct']:.1f}% | IC={m.get('ic', 0.0):+.4f} | "
        f"City Novelty={m.get('city_novelty_deg', 0.0):.1f}° (Nearest: {m.get('nearest_city', 'N/A')})"
    )
