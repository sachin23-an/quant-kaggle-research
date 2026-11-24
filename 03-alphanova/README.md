# 03. AlphaNova Biweekly Competition (Season 2, Cycle 1)

A systematic cross-sectional equity alpha predictor engineered for the **AlphaNova Biweekly Quantitative Competition (Season 2, Cycle 1)**.

* **Platform:** [AlphaNova](https://www.alphanova.tech/competition/biweekly/season-2/cycle-1)
* **Author:** Sachin Kumar ([GitHub: sachin23-an](https://github.com/sachin23-an))
* **Active Cycle:** Season 2, Cycle 1
* **Evaluation Metric:** Annualized Information Ratio ($Q$) of Spearman Rank IC net of 5 bps transaction costs
* **Asset Universe:** 20 anonymized liquid instruments across 6 continuous features

---

## 1. Problem Formulation & Institutional Constraints

The competition models high-frequency cross-sectional alpha generation:
* **Inputs:** 6 normalized continuous features (`Feature.1` through `Feature.6`) across 20 assets at hourly timestamps.
* **Target:** Proprietary forward-looking cross-sectional target (xs-z scored, clipped $\pm 5$).
* **Cost Reality:** A **5 bps turnover penalty** is charged on every position rebalance ($c \cdot ||P_{t-1} - P_{t-2}||_1$). Raw unsmoothed signals pay a ~6% Sharpe drag to execution costs, making turnover control a primary alpha lever.
* **Novelty Gating ("Signal Cities"):** Submissions must remain sufficiently uncorrelated ($|\rho| \le 0.5$ or angular distance $\ge 60^\circ$) from existing signals in the season's legacy pot.

---

## 2. Model Architecture: `AlphaSpreadPredictor`

The model is implemented in [`submission.py`](./submission.py) as a self-contained subclass of `Predictor`:

```
+-------------------------------------------------------------------------------+
|                             RAW INPUT FEATURES (F1-F6)                        |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                    CONTEMPORANEOUS CROSS-SECTIONAL RANKING                    |
|   Normalized to [-1, 1] per row: r = (rank - 0.5*(n+1)) / (0.5*(n-1))         |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                       SPREAD FEATURE ENGINEERING                              |
|   - Inverse Momentum: -r[6] (highest negative IC feature)                     |
|   - Cross-Asset Relative Value Spreads: (F5 - F6), (F2 - F6), (F4 - F6)       |
|   - Pairwise Differences: (F1 - F2), (F3 - F4), (F1 - F3)                     |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                       DUAL REGULARIZED BLEND ENGINE                           |
|   - L2 Ridge Regression (alpha = 800.0) -> high stability, zero tree noise    |
|   - Shallow XGBoost (max_depth = 2, min_child_weight = 600, reg_lambda = 10)  |
|   - Ensemble: 50% Ridge + 50% XGBoost                                         |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                         CAUSAL POST-PROCESSING                                |
|   1. Turnover Control: Exponential Smoothing (span = 300 hours)               |
|   2. Causal Static Tilt Subtraction: (P - Expanding_Mean(P.shift(1)))         |
|   3. Cross-Sectional De-Meaning: P - Mean(P, axis=1) (Mandatory zero exposure)|
+-------------------------------------------------------------------------------+
```

---

## 3. Local Verification Results

Evaluated using AlphaNova's official `runner.py` test harness:

| Metric | Baseline (`demo_engineered.py`) | Our Solution (`submission.py`) | Status |
| :--- | :---: | :---: | :---: |
| **Validation Sharpe (Holdout)** | `+0.0266` | **`+0.0402`** (+51% improvement) | **PASS** |
| **Training Execution Time** | `1.4s` | **`1.6s`** (Limit: 240s) | **PASS** |
| **Prediction Execution Time** | `0.0s` | **`0.0s`** (Limit: 60s) | **PASS** |
| **Cross-Sectional De-Mean Check** | `PASS` | **`PASS`** (Row sums = 0) | **PASS** |
| **Max Drawdown from Peak** | `-0.0012` | **`-0.0010`** | **PASS** |
| **City Novelty Distance** | `50.6°` | **`51.3°`** (vs Season 2 cities) | **VERIFIED** |

---

## 4. How to Submit on AlphaNova

1. Open **[alphanova.tech/competition/biweekly/season-2/cycle-1](https://www.alphanova.tech/competition/biweekly/season-2/cycle-1)** in your browser.
2. Click **Submit Solution** (or Upload Submission).
3. Select the file:
   ```text
   D:\Projects\Quant\quant-kaggle-research\03-alphanova\submission.py
   ```
4. Enter your model name (e.g. `AlphaSpread_v1`).
5. Click **Submit**! AlphaNova's server container will execute the automated checks and place your model on the live evaluation leaderboard.
