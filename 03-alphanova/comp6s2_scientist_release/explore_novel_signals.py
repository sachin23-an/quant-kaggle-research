"""Explore signal space to find high-Sharpe, high-IC alphas that achieve >60 deg novelty."""

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

def test_signal_generator(name, gen_func, spans=[180, 240]):
    for span in spans:
        all_ic = []
        all_pnl = []
        all_gf = []

        for idx, item in enumerate(harness._cached_val_data):
            feat = item["val_features"]
            ret = item["val_returns"]
            tgt = item["val_target"]
            tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
            g = {i: feat[f"Feature.{i}"] for i in range(1, 7)}

            sig = gen_func(g, tickers, feat)
            sig = sig.sub(sig.mean(axis=1), axis=0)

            dot = (sig * tgt[tickers]).sum(axis=1)
            norm_s = np.sqrt((sig**2).sum(axis=1))
            norm_t = np.sqrt((tgt[tickers]**2).sum(axis=1))
            ic = (dot / (norm_s * norm_t + 1e-9)).mean()
            all_ic.append(ic)

            smoothed = sig.ewm(span=span, min_periods=1).mean()
            smoothed = smoothed.sub(smoothed.mean(axis=1), axis=0)
            pnl = backtest(smoothed, ret[tickers])
            all_pnl.append(pnl)

            if harness._cached_gauge[idx] is not None:
                g_mat = harness._cached_gauge[idx]
                pred_vals = smoothed[tickers].values
                pred_vals = pred_vals - pred_vals.mean(axis=1, keepdims=True)
                n_use = min(len(g_mat), len(pred_vals))
                gf = compute_gauge_fixed(pred_vals[:n_use], g_mat[:n_use])
                all_gf.append(gf)

        concat_pnl = pd.concat(all_pnl, ignore_index=True)
        net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0

        concat_gf = np.concatenate(all_gf, axis=0)
        city = compute_city(concat_gf)
        comp_df = compare_to_cities(city, city_db_path=city_db_path)
        min_dist_deg = comp_df["angular_distance_deg"].iloc[0]
        nearest_id = comp_df["signal_id"].iloc[0]

        print(f"{name:<45} (span={span:<3}) | Sharpe: {net_sh:>+7.4f} | IC: {np.mean(all_ic):>+6.4f} | Dist: {min_dist_deg:>5.1f}° | Nearest: {nearest_id}")

print("=" * 115)
print("EXPLORING NOVEL DIMENSIONS IN SIGNAL SPACE")
print("=" * 115)

# Explore various non-linear, cross-sectional, and interaction signals
test_signal_generator("Raw Core -(F1..F4)/4", lambda g, t, f: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0)

# 1. Non-linear power transform (cubed ranks emphasize tails / extreme mispricings)
test_signal_generator("Cubic Tail Reversion -(r^3)", lambda g, t, f: -(csrank(g[1][t])**3 + csrank(g[2][t])**3 + csrank(g[3][t])**3 + csrank(g[4][t])**3) / 4.0)

# 2. Asymmetric tail reversion (only short the most overpriced, long the most underpriced)
test_signal_generator("Signed Tail Reversion (|r|>0.5)", lambda g, t, f: -(
    np.sign(csrank(g[1][t])) * np.maximum(np.abs(csrank(g[1][t])) - 0.3, 0) +
    np.sign(csrank(g[2][t])) * np.maximum(np.abs(csrank(g[2][t])) - 0.3, 0) +
    np.sign(csrank(g[3][t])) * np.maximum(np.abs(csrank(g[3][t])) - 0.3, 0) +
    np.sign(csrank(g[4][t])) * np.maximum(np.abs(csrank(g[4][t])) - 0.3, 0)
) / 4.0)

# 3. Ratio-based relative value (F6 / Core)
test_signal_generator("Cross-Ratio F6 / (1 + |Core|)", lambda g, t, f: csrank(g[6][t] / (1.0 + np.abs(g[1][t] + g[2][t] + g[3][t] + g[4][t]))))

# 4. Momentum / Trend of Reversion (delta of rank over lookback)
test_signal_generator("Core Reversion + Short-term Delta (diff 5)", lambda g, t, f: -(csrank(g[1][t]) + csrank(g[2][t]) + csrank(g[3][t]) + csrank(g[4][t])) / 4.0 - 0.3 * (csrank(g[1][t]).diff(5).fillna(0) + csrank(g[3][t]).diff(5).fillna(0)) / 2.0)

# 5. Volatility / Dispersion scaled Core (scale each asset by inverse cross-feature dispersion)
test_signal_generator("Dispersion-Weighted Core Reversion", lambda g, t, f: (
    lambda r1, r2, r3, r4: -(r1 + r2 + r3 + r4) / (np.std([r1.values, r2.values, r3.values, r4.values], axis=0) + 0.1)
)(csrank(g[1][t]), csrank(g[2][t]), csrank(g[3][t]), csrank(g[4][t])))

# 6. Interaction F6 * -Core
test_signal_generator("Interaction F6 * (-Core)", lambda g, t, f: csrank(-g[6][t] * (g[1][t] + g[2][t] + g[3][t] + g[4][t])))

# 7. Residualized Core against F5 / F6
test_signal_generator("F6 Spread Orthogonal Core", lambda g, t, f: -(csrank(g[1][t]) + csrank(g[3][t])) / 2.0 + 0.4 * (csrank(g[6][t] - g[1][t]) - csrank(g[5][t] - g[1][t])))

