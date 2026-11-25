import os
import numpy as np
import pandas as pd
from data_loader import list_periods, load_period
from evaluation import backtest, sharpe_ratio

DATA_DIR = "data/train"
UNIVERSE = "small"

periods = list_periods(DATA_DIR, UNIVERSE)

# Let's test different spans of EWM smoothing on Feature 6 (inverted), Feature 5, and differences
# across all periods

for span in [20, 60, 120, 240]:
    sh_list_inv6 = []
    sh_list_f1_f2 = []
    sh_list_f1_f3 = []
    sh_list_demo_xgboost = []
    
    for p in periods:
        data = load_period(p, DATA_DIR, UNIVERSE)
        feat = data["features"]
        returns = data["returns"]
        tickers = list(returns.columns)
        
        # Test 1: Inverted F6
        f6 = feat["Feature.6"][tickers]
        r6 = f6.rank(axis=1)
        r6 = (r6 - r6.mean(axis=1).values.reshape(-1, 1)) / (r6.std(axis=1).values.reshape(-1, 1) + 1e-8)
        sig_inv6 = (-r6).ewm(span=span, min_periods=1).mean()
        sig_inv6 = sig_inv6.sub(sig_inv6.mean(axis=1), axis=0)
        sh_inv6 = sharpe_ratio(sig_inv6, returns[tickers])
        sh_list_inv6.append(sh_inv6)
        
        # Test 2: (F1 - F2)
        diff_1_2 = feat["Feature.1"][tickers] - feat["Feature.2"][tickers]
        r_1_2 = diff_1_2.rank(axis=1)
        r_1_2 = (r_1_2 - r_1_2.mean(axis=1).values.reshape(-1, 1)) / (r_1_2.std(axis=1).values.reshape(-1, 1) + 1e-8)
        sig_1_2 = r_1_2.ewm(span=span, min_periods=1).mean()
        sig_1_2 = sig_1_2.sub(sig_1_2.mean(axis=1), axis=0)
        sh_1_2 = sharpe_ratio(sig_1_2, returns[tickers])
        sh_list_f1_f2.append(sh_1_2)

    print(f"Span={span}: Mean Sharpe inv(F6) = {np.mean(sh_list_inv6):.4f}, Mean Sharpe (F1-F2) = {np.mean(sh_list_f1_f2):.4f}")
