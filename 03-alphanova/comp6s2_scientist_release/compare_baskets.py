"""Compare submission_clean_basket.py against Pure -(F1..F4)/4 exactly."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from submission_clean_basket import PureCrossSectionalReversion
from evaluation import backtest


def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))


harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)

print("=" * 80)
print("COMPARING SUBMISSION_CLEAN_BASKET VS TRUE TARGET ALPHA ON PERIOD 031")
print("=" * 80)

# Check period 031
item = harness._cached_val_data[-1]  # period 031
feat = item["val_features"]
ret = item["val_returns"]
tgt = item["val_target"]

# 1. From submission_clean_basket.py
m1 = PureCrossSectionalReversion()
p1 = m1.predict(feat)

# 2. From test_true_target_alphas.py (with span=240, numeric sorted tickers)
cols = list(feat["Feature.1"].columns)
num_tickers = sorted(cols, key=lambda c: int(c.split(".")[-1]))
lex_tickers = list(feat["Feature.1"].columns)

print("Lexicographic tickers:", lex_tickers[:5])
print("Numeric tickers:      ", num_tickers[:5])

# What did submission_clean_basket use?
# In submission_clean_basket.py line 44:
# tickers = list(features["Feature.1"].columns)  <-- THIS WAS LEXICOGRAPHIC!
print("\np1 columns:", list(p1.columns)[:5])

# Evaluate PnL with label alignment
pnl1_label = backtest(p1, ret)
print(f"p1 (submission_clean_basket) Sharpe on Period 031: {pnl1_label.mean() / pnl1_label.std():+.4f}")

# What if evaluated positionally (by values)?
# ret is in NUMERIC order: ticker.1, ticker.2, ..., ticker.20
# p1 is in LEXICOGRAPHIC order: ticker.1, ticker.10, ticker.11, ..., ticker.2
pnl1_positional = (p1.values[1:] * ret[num_tickers].values[1:]).sum(axis=1) - 0.0005 * np.abs(np.diff(p1.values, axis=0)).sum(axis=1)
print(f"p1 Positional (values without label match) Sharpe on Period 031: {pnl1_positional.mean() / pnl1_positional.std():+.4f}")

# Multi-period positional vs label check
all_pnl_label = []
all_pnl_pos = []

for it in harness._cached_val_data:
    f = it["val_features"]
    r = it["val_returns"]
    pred = m1.predict(f)

    # Label backtest
    all_pnl_label.append(backtest(pred, r))

    # Positional backtest (if server matches numpy values directly)
    to_p = np.abs(np.diff(pred.values, axis=0)).sum(axis=1)
    r_pos = (pred.values[1:] * r.values[1:]).sum(axis=1) - 0.0005 * to_p
    all_pnl_pos.append(pd.Series(r_pos))

tot_label = pd.concat(all_pnl_label, ignore_index=True)
tot_pos = pd.concat(all_pnl_pos, ignore_index=True)

print(f"\nAcross ALL 31 Periods:")
print(f"Label alignment Sharpe (Pandas):      {tot_label.mean() / tot_label.std():+.4f}")
print(f"Positional alignment Sharpe (NumPy):  {tot_pos.mean() / tot_pos.std():+.4f}")
