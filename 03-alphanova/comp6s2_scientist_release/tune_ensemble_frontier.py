"""Benchmark ensemble blend ratios and spans for optimal novelty and Sharpe."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import xgboost as xgb

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest
from city_tools import compute_gauge_fixed, compute_city, compare_to_cities
from benchmark_ml_architectures import build_feature_matrix, MLRidge, MLXGBoost

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)
city_db_path = str(SCRIPT_DIR / "signal_cities_ambient_coords.parquet")

weights = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
spans = [240, 280, 320]

print("Weight (Ridge/XGB)    | Span   | Net Sharpe | Mean IC  | Min DB Dist")
print("=" * 65)

for w in weights:
    for s in spans:
        m_r = MLRidge(alpha=4000.0, span=s)
        m_x = MLXGBoost(max_depth=2, reg_lambda=20.0, span=s)
        
        all_pnl, all_ic, all_gf = [], [], []
        for idx, item in enumerate(harness._cached_val_data):
            feat = item["val_features"]
            ret = item["val_returns"]
            tgt = item["val_target"]
            tickers = sorted(list(ret.columns), key=lambda c: int(c.split(".")[-1]))
            
            n_mid = int(0.5 * len(feat))
            m_r.train(feat.iloc[:n_mid], tgt.iloc[:n_mid], tickers)
            m_x.train(feat.iloc[:n_mid], tgt.iloc[:n_mid], tickers)
            
            p_r = m_r.predict(feat.iloc[n_mid:], tickers)
            p_x = m_x.predict(feat.iloc[n_mid:], tickers)
            pred = w * p_r + (1.0 - w) * p_x
            pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
            
            all_pnl.append(backtest(pred, ret[tickers].iloc[n_mid:]))
            
            dot = (pred * tgt[tickers].iloc[n_mid:]).sum(axis=1)
            norm_s = np.sqrt((pred**2).sum(axis=1))
            norm_t = np.sqrt((tgt[tickers].iloc[n_mid:]**2).sum(axis=1))
            all_ic.append((dot / (norm_s * norm_t + 1e-9)).mean())
            
            g_mat = harness._cached_gauge[idx]
            if g_mat is not None:
                p_vals = pred.values - pred.values.mean(axis=1, keepdims=True)
                n_use = min(len(g_mat) - n_mid, len(p_vals))
                if n_use > 0:
                    all_gf.append(compute_gauge_fixed(p_vals[:n_use], g_mat[n_mid:n_mid+n_use]))
                    
        concat_pnl = pd.concat(all_pnl, ignore_index=True)
        net_sh = concat_pnl.mean() / concat_pnl.std() if concat_pnl.std() > 0 else 0.0
        concat_gf = np.concatenate(all_gf, axis=0)
        city = compute_city(concat_gf)
        comp_df = compare_to_cities(city, city_db_path=city_db_path)
        min_dist = comp_df["angular_distance_deg"].iloc[0]
        
        tag = f"{int(w*100)}% Ridge / {int((1-w)*100)}% XGB"
        print(f"{tag:<21} | {s:<6} | {net_sh:>+10.4f} | {np.mean(all_ic):>+8.4f} | {min_dist:>10.1f} deg")
