import os
import numpy as np
import pandas as pd
from data_loader import list_periods, load_period
from evaluation import backtest, sharpe_ratio

DATA_DIR = "data/train"
UNIVERSE = "small"

periods = list_periods(DATA_DIR, UNIVERSE)
print(f"Total periods: {len(periods)}")

summary = []

for p in periods:
    data = load_period(p, DATA_DIR, UNIVERSE)
    feat = data["features"]
    target = data["target"]
    returns = data["returns"]
    print(f"Period {p}: rows={len(returns)}", flush=True)
    
    # Analyze Feature 1 to 6 contemporaneous rank IC with target and returns
    row_info = {"period": p, "rows": len(returns)}
    
    tickers = list(returns.columns)
    
    # Hourly returns direction vs target
    for f_idx in range(1, 7):
        f = feat[f"Feature.{f_idx}"][tickers]
        
        # Cross-sectional rank
        r_f = f.rank(axis=1)
        r_f = (r_f - r_f.mean(axis=1).values.reshape(-1, 1)) / (r_f.std(axis=1).values.reshape(-1, 1) + 1e-8)
        
        # IC with target
        # align by dot product
        t_norm = target[tickers] - target[tickers].mean(axis=1).values.reshape(-1, 1)
        ic_target = (r_f * t_norm).mean(axis=1).mean()
        
        # PnL with returns (5bp cost)
        pf = backtest(r_f, returns[tickers])
        sh = pf.mean() / (pf.std() + 1e-8)
        
        row_info[f"F{f_idx}_ic_target"] = ic_target
        row_info[f"F{f_idx}_sharpe_ret"] = sh
        
    summary.append(row_info)

df_summary = pd.DataFrame(summary)
print("\n--- Mean across all 31 periods ---")
for f_idx in range(1, 7):
    print(f"F{f_idx}: mean IC(target)={df_summary[f'F{f_idx}_ic_target'].mean():.4f}, mean Sharpe(returns)={df_summary[f'F{f_idx}_sharpe_ret'].mean():.4f}")

print("\n--- Period by Period details for F6 & F5 ---")
print(df_summary[["period", "F5_ic_target", "F5_sharpe_ret", "F6_ic_target", "F6_sharpe_ret"]].tail(10))
print(df_summary[["period", "F5_ic_target", "F5_sharpe_ret", "F6_ic_target", "F6_sharpe_ret"]].head(10))
