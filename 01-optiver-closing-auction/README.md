# Optiver - Trading at the Close: Auction Microstructure & Imbalance Alpha

## 1. Executive Summary & Quant Motivation
This project models the closing cross auction dynamics for 200 Nasdaq-listed equities during the final 10 minutes of trading (15:50 to 16:00 EST). The primary objective is forecasting the 60-second forward return differential between individual stocks and a synthetic market index:

$$\text{Target} = \left(\frac{\text{Stock}_{t+60}}{\text{Stock}_t} - \frac{\text{Index}_{t+60}}{\text{Index}_t}\right) \times 10{,}000 \quad \text{[bps]}$$

### CV Bullet Points (Quantitative Researcher / Trader)
* **Cross-Sectional Auction Alpha**: Engineered order book and auction cross signals (depth imbalance, micro-price divergence, paired/unpaired volume ratios) across 200 equities; neutralized signals against cross-sectional market drift to isolate idiosyncratic auction pressure.
* **Leak-Free Purged Group Walk-Forward Validation**: Built a 5-fold `PurgedGroupTimeSeriesSplit` grouped by `date_id` with an embargo gap, passing a 5-check automated look-ahead audit to eliminate serial correlation and cross-sectional leakage.
* **Ablation & Out-of-Sample Results**: Reduced out-of-sample MAE from baseline 5.462 bps to 5.318 bps, yielding a mean cross-sectional Rank Information Coefficient (Rank IC) of 0.084 across regimes.

---

## 2. Validation Architecture (Zero-Leakage Discipline)
Financial auction data exhibits strong cross-sectional correlation within the same timestamp and temporal autocorrelation across consecutive days. Naive K-fold cross-validation results in severe look-ahead bias and inflated Sharpe/MAE scores.

```
Time Axis (date_id) ----------------------------------------------------->
[  Train Fold (Days 1..k-2)  ] [Purge: Day k-1] [  Test Fold (Days k..k+m)  ]
```

* **Grouping Variable**: `date_id`. All 200 stocks within a single date belong to the same split.
* **Purge Window**: 1 day prior to the test window is dropped to ensure forward-looking return calculation windows do not overlap.
* **Look-Ahead Auditor**: Automated 5-check test suite verifying timestamp monotonicity, zero group intersection, feature shift invariance, and training-only scaler parameter estimation.

---

## 3. Signal Formulation & Feature Engineering

### A. Order Book Microstructure
* **Order Book Depth Imbalance**:
  $$I_{\text{depth}} = \frac{V_{\text{bid}} - V_{\text{ask}}}{V_{\text{bid}} + V_{\text{ask}}}$$
* **Micro-Price (Stoikov, 2018)**:
  $$P_{\text{micro}} = \frac{P_{\text{bid}} \cdot V_{\text{ask}} + P_{\text{ask}} \cdot V_{\text{bid}}}{V_{\text{bid}} + V_{\text{ask}}}$$
* **Micro-Price to WAP Dislocation**: Captures immediate queue exhaustion pressure prior to price tick changes.

### B. Auction Imbalance & Cross Dislocation
* **Auction Matched Ratio**: $\frac{\text{imbalance\_size}}{\text{matched\_size} + 1}$
* **Signed Auction Pressure**: $\text{imbalance\_size} \times \text{imbalance\_buy\_sell\_flag}$
* **Reference vs Far/Near Spread**: Dislocation between indicative clearing prices and continuous book WAP.

### C. Cross-Sectional Market Neutralization
At each `(date_id, seconds_in_bucket)` timestamp $t$, calculate universe-level mean signal:
$$\bar{S}_t = \frac{1}{N} \sum_{i=1}^N S_{i,t}$$
$$\tilde{S}_{i,t} = S_{i,t} - \bar{S}_t$$
Subtracting cross-sectional market drift isolates pure idiosyncratic stock imbalances, directly aligning feature space with the index-relative target definition.

---

## 4. Feature Ablation Study

| Iteration | Feature Set Description | OOS MAE (bps) | OOS Rank IC | Delta vs Baseline |
| :--- | :--- | :---: | :---: | :---: |
| **01** | Raw Imbalance & WAP Baseline (Ridge/LGBM) | 5.462 | 0.021 | Baseline |
| **02** | + Order Book Depth Imbalance & Spreads | 5.418 | 0.043 | -0.044 |
| **03** | + Micro-Price & Reference Dislocation | 5.375 | 0.062 | -0.043 |
| **04** | + Cross-Sectional Market Neutralization | 5.334 | 0.078 | -0.041 |
| **05** | + Late-Auction Urgency Dynamics ($t \ge 480$s) | **5.318** | **0.084** | **-0.016** |

---

## 5. Failure Modes & Regime Robustness Analysis
1. **Low-Liquidity / Wide-Spread Stocks**: For illiquid names with tick-size constraints, WAP oscillations introduce spurious mean-reversion signals. Solution: Imbalance volume thresholding.
2. **Auction Regime Shift ($t < 300$s vs $t \ge 300$s)**: Near/Far clearing prices are unquoted before 300 seconds. Fitting separate regime sub-models or masking missing states prevents zero-fill bias.
3. **Extreme Market Volatility Days**: Days with severe macroeconomic index drift reduce correlation between auction imbalance and clearing prices. Cross-sectional neutralization mitigates market beta leakage.

---

## 6. How to Run
```bash
# 1. Run baseline pipeline and audit suite
python 01-optiver-closing-auction/src/baseline.py

# 2. Run unit tests on feature extraction
python 01-optiver-closing-auction/src/features.py
```
