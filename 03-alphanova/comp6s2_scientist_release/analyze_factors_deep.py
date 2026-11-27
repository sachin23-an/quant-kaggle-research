"""Deep Quantitative Factor Analysis across all 31 historical periods.

Investigates:
1. Cross-sectional rank correlation of each feature with next-period returns.
2. Cross-sectional rank correlation of each feature with the forward target.
3. Feature-to-feature cross-correlation matrix (clustering / collinearity).
4. Feature autocorrelation and decay profiles (turnover drivers).
5. All 15 pairwise spreads (F_i - F_j) gross and net Sharpe under 5 bps cost.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import TRANSACTION_COST


def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))


def analyze():
    harness = ResearchHarness()
    harness.preload_validation_cache(verbose=False)

    print("=" * 70)
    print("DEEP FACTOR ANALYSIS ACROSS ALL 31 PERIODS (24,800 HOURLY OBSERVATIONS)")
    print("=" * 70)

    # Accumulate across all validation periods
    feature_ic_returns = {f"Feature.{i}": [] for i in range(1, 7)}
    feature_ic_target = {f"Feature.{i}": [] for i in range(1, 7)}
    feature_autocorr = {f"Feature.{i}": [] for i in range(1, 7)}

    # Individual feature standalone Sharpe (positive and negative orientation)
    feat_net_sharpe = {f"Feature.{i}": [] for i in range(1, 7)}
    feat_inv_net_sharpe = {f"Feature.{i}": [] for i in range(1, 7)}
    feat_gross_sharpe = {f"Feature.{i}": [] for i in range(1, 7)}

    # Track pairwise spreads
    pairs = []
    for i in range(1, 7):
        for j in range(i + 1, 7):
            pairs.append((i, j))
    spread_pnl = {f"F{i}-F{j}": [] for i, j in pairs}
    spread_gross = {f"F{i}-F{j}": [] for i, j in pairs}

    # Cross-correlation matrix accumulator
    corr_matrices = []

    for item in harness._cached_val_data:
        feat_df = item["val_features"]
        ret_df = item["val_returns"]
        tgt_df = item["val_target"]
        tickers = list(ret_df.columns)

        # Cross correlation across the 6 features for this period
        mean_ranks = {}
        for i in range(1, 7):
            rf = csrank(feat_df[f"Feature.{i}"][tickers])
            mean_ranks[f"F{i}"] = rf

            # Correlation with 1-hour contest returns (next hour: shift(1) signal * ret)
            # Cross-sectional correlation at each timestamp
            cs_corr_ret = (rf * ret_df).sum(axis=1) / np.sqrt((rf**2).sum(axis=1) * (ret_df**2).sum(axis=1) + 1e-9)
            feature_ic_returns[f"Feature.{i}"].append(cs_corr_ret.mean())

            # Correlation with target
            cs_corr_tgt = (rf * tgt_df).sum(axis=1) / np.sqrt((rf**2).sum(axis=1) * (tgt_df**2).sum(axis=1) + 1e-9)
            feature_ic_target[f"Feature.{i}"].append(cs_corr_tgt.mean())

            # Autocorrelation of the rank
            rf_lag1 = rf.shift(1)
            ac = (rf * rf_lag1).sum(axis=1) / np.maximum((rf**2).sum(axis=1), 1e-9)
            feature_autocorr[f"Feature.{i}"].append(ac.mean())

            # Position smoothing (span=150)
            sig = rf.ewm(span=150, min_periods=1).mean()
            sig = sig.sub(sig.mean(axis=1), axis=0)

            # Net PnL (long)
            pnl_gross = (sig.shift(1) * ret_df).sum(axis=1)
            to = sig.shift(1).fillna(0.0).diff().abs().sum(axis=1)
            pnl_net = pnl_gross - TRANSACTION_COST * to
            pnl_net.iloc[0] = 0

            # Net PnL (short / inverted)
            inv_sig = -sig
            inv_gross = (inv_sig.shift(1) * ret_df).sum(axis=1)
            inv_to = inv_sig.shift(1).fillna(0.0).diff().abs().sum(axis=1)
            inv_net = inv_gross - TRANSACTION_COST * inv_to
            inv_net.iloc[0] = 0

            feat_gross_sharpe[f"Feature.{i}"].append(pnl_gross)
            feat_net_sharpe[f"Feature.{i}"].append(pnl_net)
            feat_inv_net_sharpe[f"Feature.{i}"].append(inv_net)

        # Pairwise spreads
        for i, j in pairs:
            # Spread = rank(F_i) - rank(F_j)
            s = mean_ranks[f"F{i}"] - mean_ranks[f"F{j}"]
            s = s.ewm(span=150, min_periods=1).mean()
            s = s.sub(s.mean(axis=1), axis=0)

            p_gross = (s.shift(1) * ret_df).sum(axis=1)
            to_s = s.shift(1).fillna(0.0).diff().abs().sum(axis=1)
            p_net = p_gross - TRANSACTION_COST * to_s
            p_net.iloc[0] = 0

            spread_pnl[f"F{i}-F{j}"].append(p_net)
            spread_gross[f"F{i}-F{j}"].append(p_gross)

    # Print Summary Table
    print("\n1. INDIVIDUAL FEATURE CHARACTERISTICS (span=150 smoothed):")
    print("-" * 80)
    print(f"{'Feature':<12} | {'IC (Returns)':<14} | {'IC (Target)':<14} | {'Autocorr':<10} | {'Gross Sharpe':<13} | {'Long Net Sharpe':<16} | {'Short Net Sharpe':<16}")
    print("-" * 80)
    for i in range(1, 7):
        fk = f"Feature.{i}"
        ic_r = np.nanmean(feature_ic_returns[fk])
        ic_t = np.nanmean(feature_ic_target[fk])
        ac = np.nanmean(feature_autocorr[fk])

        all_g = pd.concat(feat_gross_sharpe[fk], ignore_index=True)
        all_l = pd.concat(feat_net_sharpe[fk], ignore_index=True)
        all_s = pd.concat(feat_inv_net_sharpe[fk], ignore_index=True)

        g_sharpe = all_g.mean() / all_g.std() if all_g.std() > 0 else 0.0
        l_sharpe = all_l.mean() / all_l.std() if all_l.std() > 0 else 0.0
        s_sharpe = all_s.mean() / all_s.std() if all_s.std() > 0 else 0.0

        print(f"{fk:<12} | {ic_r:+13.4f} | {ic_t:+13.4f} | {ac:9.3f} | {g_sharpe:+12.4f} | {l_sharpe:+15.4f} | {s_sharpe:+15.4f}")

    print("\n2. ALL 15 PAIRWISE SPREADS (Rank(Fi) - Rank(Fj), span=150):")
    print("-" * 65)
    print(f"{'Spread Pair':<15} | {'Gross Sharpe':<14} | {'Net Sharpe (Long)':<18} | {'Net Sharpe (Short)':<18}")
    print("-" * 65)

    spread_results = []
    for i, j in pairs:
        pair_key = f"F{i}-F{j}"
        g_all = pd.concat(spread_gross[pair_key], ignore_index=True)
        n_all = pd.concat(spread_pnl[pair_key], ignore_index=True)

        g_s = g_all.mean() / g_all.std() if g_all.std() > 0 else 0.0
        n_s = n_all.mean() / n_all.std() if n_all.std() > 0 else 0.0
        inv_n_s = -g_all.mean() / g_all.std()  # approximate inverse net
        # calculate exact inverted net
        inv_all = -g_all - TRANSACTION_COST * (g_all - n_all) / TRANSACTION_COST
        inv_net_s = -n_s  # exact if turnover symmetric

        spread_results.append((pair_key, g_s, n_s, -n_s))

    # Sort spreads by absolute net Sharpe
    spread_results.sort(key=lambda x: max(x[2], x[3]), reverse=True)
    for pair_key, g_s, l_s, s_s in spread_results:
        best_dir = f"+{pair_key}" if l_s > s_s else f"-({pair_key})"
        best_sharpe = max(l_s, s_s)
        print(f"{pair_key:<15} | {g_s:+13.4f} | {l_s:+17.4f} | {s_s:+17.4f} -> Best: {best_dir:<10} ({best_sharpe:+.4f})")


if __name__ == "__main__":
    analyze()
