"""Full performance breakdown of candidate submission model."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest, check_cross_sectional_demean
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities

def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

class CandidateSubmission:
    def __init__(self, span: int = 200):
        self.span = span

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        pass

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        cols = list(features["Feature.1"].columns)
        # Strict numeric sort
        tickers = sorted(cols, key=lambda c: int(c.split(".")[-1]))

        g = {i: features[f"Feature.{i}"][tickers] for i in range(1, 7)}
        
        # 1. Valuation reversion core: contemporaneous negation of Features 1..4
        core = -(csrank(g[1]) + csrank(g[2]) + csrank(g[3]) + csrank(g[4])) / 4.0
        
        # 2. Relative value overlay: Feature 6 spread vs valuation anchor
        spread = csrank(g[6] - g[1])
        
        # 3. Composite signal
        sig = core + 0.20 * spread
        
        # 4. Strict cross-sectional de-meaning (zero net market beta)
        sig = sig.sub(sig.mean(axis=1), axis=0)
        
        # 5. Causal position smoothing to crush 5 bps hourly rebalancing turnover
        smoothed = sig.ewm(span=self.span, min_periods=1).mean()
        
        # 6. Final de-meaning and column preservation
        smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
        return smoothed[tickers].fillna(0.0)

model = CandidateSubmission(span=200)

all_pnl = []
all_ic = []
all_turnover = []
per_period = []

for idx, item in enumerate(harness._cached_val_data):
    feat = item["val_features"]
    ret = item["val_returns"]
    tgt = item["val_target"]
    period_name = item["period"]
    tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))

    pred = model.predict(feat)
    dm_ok = check_cross_sectional_demean(pred)
    assert dm_ok, f"Demean check failed on {period_name}"

    pnl = backtest(pred, ret[tickers])
    all_pnl.append(pnl)

    # Hourly turnover
    to = pred.shift(1).fillna(0.0).diff().abs().sum(axis=1).mean()
    all_turnover.append(to)

    # IC vs Target
    dot = (pred * tgt[tickers]).sum(axis=1)
    norm_s = np.sqrt((pred**2).sum(axis=1))
    norm_t = np.sqrt((tgt[tickers]**2).sum(axis=1))
    ic = (dot / (norm_s * norm_t + 1e-9)).mean()
    all_ic.append(ic)

    p_sh = pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
    per_period.append((period_name, p_sh, ic, to))

concat_pnl = pd.concat(all_pnl, ignore_index=True)
net_sh = concat_pnl.mean() / concat_pnl.std()
mean_ic = np.mean(all_ic)
mean_to = np.mean(all_turnover)
win_rate = sum(1 for _, sh, _, _ in per_period if sh > 0) / len(per_period)

print("=" * 75)
print("COMPREHENSIVE AUDIT OF CANDIDATE SUBMISSION MODEL")
print("=" * 75)
print(f"Full Walk-Forward Net Sharpe (after 5bp fee): {net_sh:+.4f}")
print(f"Full Walk-Forward Mean IC:                    {mean_ic:+.4f}")
print(f"Mean Hourly L1 Turnover:                      {mean_to:.4f}")
print(f"Profitable Periods (Win Rate):                {win_rate:.1%} ({sum(1 for _, sh, _, _ in per_period if sh > 0)}/{len(per_period)})")
print(f"Cross-Sectional De-mean Check:                100% PASS")
print("=" * 75)
print("\nFirst 10 Periods & Last 10 Periods breakdown:")
for p, s, c, t in per_period[:5]:
    print(f"  Period {p}: Sharpe={s:+.4f}, IC={c:+.4f}, Turnover={t:.4f}")
print("  ...")
for p, s, c, t in per_period[-5:]:
    print(f"  Period {p}: Sharpe={s:+.4f}, IC={c:+.4f}, Turnover={t:.4f}")
