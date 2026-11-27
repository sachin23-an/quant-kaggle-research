"""Test PureCrossSectionalReversion with strict numeric sorting on exact shifted positional evaluation."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from research_harness import ResearchHarness
from evaluation import backtest, TRANSACTION_COST

def csrank(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    n = df.notna().sum(axis=1).values.reshape(-1, 1)
    return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

harness = ResearchHarness()
harness.preload_validation_cache(verbose=False)

all_pnl_numeric_pos = []
all_pnl_lexico_pos = []
all_pnl_label = []

for item in harness._cached_val_data:
    feat = item["val_features"]
    ret = item["val_returns"]
    
    lex_cols = list(feat["Feature.1"].columns)
    num_cols = sorted(lex_cols, key=lambda c: int(c.split(".")[-1]))
    
    # 1. Prediction built in numeric order
    g_num = {i: feat[f"Feature.{i}"][num_cols] for i in range(1, 5)}
    sig_num = -(csrank(g_num[1]) + csrank(g_num[2]) + csrank(g_num[3]) + csrank(g_num[4])) / 4.0
    sig_num = sig_num.ewm(span=200, min_periods=1).mean()
    sig_num = sig_num.sub(sig_num.mean(axis=1), axis=0)
    pred_num = sig_num[num_cols].fillna(0.0)

    # 2. Prediction built in lexico order (Attempt #4 style)
    g_lex = {i: feat[f"Feature.{i}"][lex_cols] for i in range(1, 5)}
    sig_lex = -(csrank(g_lex[1]) + csrank(g_lex[2]) + csrank(g_lex[3]) + csrank(g_lex[4])) / 4.0
    sig_lex = sig_lex.ewm(span=200, min_periods=1).mean()
    sig_lex = sig_lex.sub(sig_lex.mean(axis=1), axis=0)
    pred_lex = sig_lex[lex_cols].fillna(0.0)

    ret_num = ret[num_cols]

    # Exact evaluation matching evaluation.py:
    # pf_returns = (predictions.shift(1) * returns).sum(axis=1)
    # turnover = predictions.shift(1).fillna(0.0).diff().abs().sum(axis=1)
    # With numeric values:
    pred_shifted_num = np.vstack([np.zeros((1, 20)), pred_num.values[:-1]])
    to_num = np.abs(np.diff(pred_shifted_num, axis=0)).sum(axis=1)
    to_num = np.append([np.abs(pred_shifted_num[0]).sum()], to_num)
    # Actually matching diff() on shift(1).fillna(0.0):
    p_s_num = np.vstack([np.zeros((1, 20)), pred_num.values[:-1]])
    diff_num = np.diff(p_s_num, axis=0, prepend=np.zeros((1, 20)))
    to_num = np.abs(diff_num).sum(axis=1)
    pnl_num_pos = (p_s_num * ret_num.values).sum(axis=1) - 0.0005 * to_num
    pnl_num_pos[0] = 0
    all_pnl_numeric_pos.append(pd.Series(pnl_num_pos))

    p_s_lex = np.vstack([np.zeros((1, 20)), pred_lex.values[:-1]])
    diff_lex = np.diff(p_s_lex, axis=0, prepend=np.zeros((1, 20)))
    to_lex = np.abs(diff_lex).sum(axis=1)
    pnl_lex_pos = (p_s_lex * ret_num.values).sum(axis=1) - 0.0005 * to_lex
    pnl_lex_pos[0] = 0
    all_pnl_lexico_pos.append(pd.Series(pnl_lex_pos))

    all_pnl_label.append(backtest(pred_num, ret_num))

pnl_num_pos_series = pd.concat(all_pnl_numeric_pos, ignore_index=True)
pnl_lex_pos_series = pd.concat(all_pnl_lexico_pos, ignore_index=True)
pnl_label_series = pd.concat(all_pnl_label, ignore_index=True)

sh_num_pos = pnl_num_pos_series.mean() / pnl_num_pos_series.std()
sh_lex_pos = pnl_lex_pos_series.mean() / pnl_lex_pos_series.std()
sh_label = pnl_label_series.mean() / pnl_label_series.std()

print("=" * 70)
print(f"Numeric Order Shifted Positional Sharpe:  {sh_num_pos:+.4f}")
print(f"Lexicographic Shifted Positional Sharpe: {sh_lex_pos:+.4f}")
print(f"Label Alignment Sharpe (Pandas Backtest): {sh_label:+.4f}")
print("=" * 70)
