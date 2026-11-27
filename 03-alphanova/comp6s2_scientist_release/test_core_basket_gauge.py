"""Evaluate PureCrossSectionalReversion full metrics and novelty."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from submission_clean_basket import PureCrossSectionalReversion

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)

print(">>> Evaluating PureCrossSectionalReversion (Attempt #4 architecture)...")
m = harness.run_walkforward(PureCrossSectionalReversion, train_full=False, gauge_fix=True, verbose=True)
harness.log_experiment("EXP-019", "PureCrossSectionalReversion", "Pure core negation -(F1..F4)/4 with span=150", m)
