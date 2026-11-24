# 02. Credit Card Fraud Detection (Kaggle Playground S6E11)

A systematic machine learning solution for credit card fraud detection on synthetic transactional data, achieving **0.95567 ROC-AUC (Rank 81)** on the Kaggle public leaderboard through multi-model logit-space rank fusion, domain heuristic clean-zone filtering, and constrained diversity search.

* **Competition:** [Kaggle Playground Series Season 6 Episode 11](https://www.kaggle.com/competitions/playground-series-s6e11)
* **Author:** Sachin Kumar ([GitHub: sachin23-an](https://github.com/sachin23-an))
* **Primary Metric:** ROC-AUC
* **Class Imbalance:** ~1.67% Positive Class (Fraud)
* **Current Standing:** **Rank 81** (Score: `0.95567`) across 6 iterations

---

## 1. Solution Architecture & Key Methodologies

Standard probability averaging across heterogeneous model families (deep neural nets, tree ensembles, tabular transformers) often distorts calibrated thresholds. The solution employs a four-stage rank-space ensembling and optimization methodology:

```
+---------------------------------------------------------------------------------+
|                       STAGE 1: MULTI-MODEL DIVERSITY POOL                       |
|   Member models: RealMLP, TabM, LightGBM, CatBoost, XGBoost, GPT-2 Tabular      |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                       STAGE 2: LOGIT-RANK PROJECTION                            |
|   Percentile Rank:  R(x) = (rank(x) - 0.5) / N  in (0, 1)                       |
|   Log-Odds Space:   L(x) = logit(R(x)) in (-inf, +inf)                          |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                   STAGE 3: DOMAIN "CLEAN ZONE" MASKING                          |
|   Identifies transactions with: cvv_retry == 0, prior_disputes == 0,            |
|   clean security flags (no VPN/mismatch), and low merchant risk (< 13.1).       |
|   Down-shifts clean ranks: L_adjusted = L - 2 * zones                           |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                STAGE 4: CONSTRAINED DIVERSITY BLEND SEARCH                      |
|   Evaluates blend partners under strict stability constraints:                  |
|   Mean percentile shift <= 0.001  AND  Reordered rows >= 10%                    |
|   Result: Score 0.95567 (Rank 81)                                               |
+---------------------------------------------------------------------------------+
```

---

## 2. Mathematical Details

### A. Uniform Percentile Rank & Logit Space
Because ROC-AUC depends strictly on relative ordering rather than absolute probabilities, raw predictions are transformed into uniform ranks:
$$R(x) = \frac{\text{rankdata}(x) - 0.5}{N}$$

Predictions are then mapped into unbounded log-odds space using the logit function:
$$L(x) = \ln \left( \frac{R(x)}{1 - R(x)} \right)$$
In logit space, linear combinations of model outputs behave additively without boundary compression artifacts at 0 or 1.

### B. High-Confidence "Clean Zone" Mask
In credit card fraud detection, false positives in high-volume, low-risk transactions severely degrade ROC-AUC. Transactions meeting the following conditions have virtually zero empirical fraud probability:
* `cvv_retry_count == 0`
* `prior_disputes == 0`
* Zero security warning flags (no VPN, no IP/country mismatch, no billing mismatch, no AI scam flag)
* `merchant_risk_score < 13.1` or (`velocity_score <= 4.1` and `merchant_risk_score <= 20.0`)

Applying a downward rank shift (`- 2 * zones`) ensures these low-risk transactions are pushed to the lowest ranking percentiles, eliminating false-positive boundary leakage.

### C. Constrained Partner Search
To avoid over-fitting public leaderboard noise, blend candidates are searched around the dominant anchor prediction under two strict criteria:
1. **Bounded Mean Shift:** $\frac{1}{N} \sum |R_{\text{blend}} - R_{\text{anchor}}| \le 0.001$ (preserves the anchor's core calibration).
2. **Minimum Reordering:** $\frac{1}{N} \sum \mathbb{I}(R_{\text{blend}} \ne R_{\text{anchor}}) \ge 0.10$ (guarantees meaningful rank diversity).

---

## 3. Repository Contents

```text
02-credit-card-fraud/
├── credit_card_fraud_rank_fusion.ipynb   # Complete, structured 22-cell reproducible notebook
├── pipeline.py                           # Standalone offline cross-validation & spline pipeline
├── README.md                             # Methodology, mathematical formulation, and competition strategy
└── data/                                 # Local benchmark data and submission artifacts (gitignored)
```

---

## 4. Final-Week Competition Strategy (Protecting Top 100 & Targeting Top 50)

With approximately one week remaining in the competition, public leaderboard positions are vulnerable to **Private Leaderboard Shakeup** (the private test set evaluates a separate 50% split). 

To ensure our position remains solid in the Top 100 and moves toward Top 50, the following rules apply:

1. **Avoid Over-Tuning on Public Leaderboard Noise:**
   * Making small micro-shifts that gain $+0.00001$ on the public board often overfits the public sample.
   * Stability on Out-Of-Fold cross-validation is the most reliable predictor of private test performance.
2. **Inject Orthogonal Diversity (2%–5%):**
   * Most public top models share high correlation ($> 0.99$). Blending in a modest weight ($2\%$ to $5\%$) of an independently trained model (e.g., our continuous B-Spline or regularized CatBoost model) reduces ensemble variance on unseen private test data.
3. **Strategic 2-Submission Selection:**
   * Kaggle allows selecting exactly **2 final submissions** for private scoring:
     * **Selection 1 (The Leaderboard Anchor):** The highest scoring verified public blend (`0.95567+`).
     * **Selection 2 (The Robust Hedge):** A conservative blend combining the anchor with our independent out-of-fold feature model to hedge against public split variance.
