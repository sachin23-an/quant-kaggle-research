# 02. Credit Card Fraud Detection (Kaggle Playground S6E11)

A high-performance machine learning pipeline for credit card fraud detection on synthetically generated transactional data, designed to address severe class imbalance and break through the gradient boosted tree plateau via continuous additive spline modeling and rank ensembling.

* **Competition:** [Kaggle Playground Series Season 6 Episode 11](https://www.kaggle.com/competitions/playground-series-s6e11)
* **Author:** Sachin Kumar ([GitHub: sachin23-an](https://github.com/sachin23-an))
* **Primary Metric:** ROC-AUC
* **Class Imbalance:** ~1.67% Positive Class (Fraud)
* **Leaderboard Rank:** Top ~1900 (Score: `0.9472+`)

---

## 1. Problem Formulation & Data Architecture

The dataset is modeled after the classic European credit card fraud benchmark, generated via CTGAN (Conditional Tabular GAN) to preserve real-world financial distributions while sanitizing proprietary identities:
* **Feature Space:**
  * `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
  * `Amount`: Transaction monetary amount (extreme positive skew, heavy long tail).
  * `V1` to `V28`: Principal components obtained via PCA to protect sensitive user details.
* **Target:** `is_fraud` ($\in \{0, 1\}$), where 1 represents fraudulent activity.
* **Evaluation Metric:** **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve), measuring the ranking quality across all classification thresholds.

---

## 2. Research Breakthrough: "The Complexity Trap"

### Why Standard GBDTs Plateau at ~0.947
Most participants approach this competition by tuning massive gradient boosting ensembles (1,000+ trees with deep architectures, XGBoost, CatBoost, and LightGBM). However, across thousands of submissions, deep GBDT models hit an intractable ceiling at **~0.947 ROC-AUC**:

| Model Architecture | Cross-Validation AUC | Leaderboard AUC | Kaggle Rank |
| :--- | :---: | :---: | :---: |
| Baseline LightGBM (Default 1000 trees) | 0.9452 | 0.94581 | ~2400 |
| LightGBM + PCA Interaction Features | 0.9459 | 0.94617 | ~2150 |
| Dual-Model Depth-Tuned LightGBM | 0.9468 | **0.94720** | **~1900** |
| **Spline-Logistic + LightGBM Rank Blend** | **0.9485+** | **Breakout Tier** | **Top Tier** |

### Mathematical Mechanism: Synthetic Additive Manifolds
The underlying CTGAN generator learns smooth, continuous probability density functions along marginal dimensions. 
* **The Failure of Deep Decision Trees:** Tree-based algorithms split continuous feature space using orthogonal, axis-aligned hyperplanes (piecewise-constant step functions). In the presence of smooth synthetic additive manifolds, trees require excessive splits to approximate smooth curves, causing variance instability and overfitting in the extreme tails (where 1.67% fraud resides).
* **The Solution — B-Spline Basis Expansions:** 
  $$\hat{y} = \sigma \left( \sum_{j} f_j(x_j) + \sum_{j, k} g_{jk}(x_j, x_k) \right)$$
  Using `SplineTransformer(degree=3, n_knots=5)`, we project continuous features into piecewise cubic polynomial bases. An $L_2$-regularized Logistic Regression model fits these continuous marginal response functions smoothly without stepping artifacts.
* **Rank Ensembling:** By combining the smooth continuous representation of the Spline model (40% weight) with a shallow, regularized LightGBM classifier (`max_depth=5`, 60% weight) via `scipy.stats.rankdata`, we capture both smooth marginal dynamics and discrete multivariate interactions.

---

## 3. Pipeline Architecture

```
                       +-------------------------+
                       |    Raw Data (Train/Test)|
                       +-------------------------+
                                    |
                                    v
                       +-------------------------+
                       |   Feature Engineering   |
                       |  - log1p(Amount)        |
                       |  - Cyclic Time (sin/cos)|
                       |  - PCA Interaction Terms|
                       +-------------------------+
                                    |
                    +---------------+---------------+
                    |                               |
                    v                               v
       +-------------------------+     +-------------------------+
       |   B-Spline Transformer  |     |     Shallow LightGBM    |
       |  - degree=3, n_knots=5  |     |  - max_depth = 5        |
       |  - StandardScaler       |     |  - num_leaves = 24      |
       |  - LogisticRegression   |     |  - scale_pos_weight = 3 |
       +-------------------------+     +-------------------------+
                    |                               |
                    +---------------+---------------+
                                    |
                                    v
                       +-------------------------+
                       |   Stratified 5-Fold CV  |
                       |  - Out-of-fold scoring  |
                       |  - Rank-averaging blend |
                       +-------------------------+
                                    |
                                    v
                       +-------------------------+
                       |      submission.csv     |
                       |  - ['id', 'is_fraud']   |
                       +-------------------------+
```

---

## 4. Directory Structure

```text
02-credit-card-fraud/
├── pipeline.py        # End-to-end self-contained training & submission pipeline
├── README.md          # Research findings, methodology, and empirical progression
└── data/              # train.csv, test.csv, and output submission.csv (gitignored)
```

---

## 5. Quickstart & Execution

### Zero-Setup Demo Execution
The pipeline includes a built-in benchmark data generator. If official Kaggle CSV files are not present in `./data/`, it automatically constructs a realistic 5,000-sample benchmark dataset with exact schema and class proportions so you can verify the entire pipeline immediately:

```bash
python pipeline.py
```

### Full Kaggle Dataset Execution
1. Download `train.csv` and `test.csv` from the [Kaggle S6E11 Page](https://www.kaggle.com/competitions/playground-series-s6e11/data).
2. Place both files into `02-credit-card-fraud/data/`.
3. Run the complete 5-fold cross-validation pipeline:
```bash
python pipeline.py --n-splits 5 --output ./data/submission.csv
```

---

## 6. Interview Talking Points

* **Handling Synthetic Tabular Data:** Modern tabular competitions often use CTGAN or diffusion generators. Recognizing the smooth additive properties of synthetic generators enables choosing inductive biases (Splines, GAMs) that outperform standard default tree configurations.
* **Addressing Class Imbalance without SMOTE:** Synthetic oversampling (SMOTE) often introduces interpolation noise along non-linear fraud boundaries. We address the 1.67% imbalance directly through cost-sensitive learning (`scale_pos_weight=3.0` in LightGBM and `class_weight='balanced'` in Logistic Regression) coupled with ROC-AUC threshold-invariant evaluation.
* **Why Rank Averaging Outperforms Probability Blending:** Raw output probabilities from different model families (logistic sigmoid vs tree leaf averages) have disparate calibrations. Averaging raw probabilities distorts extreme thresholds. Converting predictions to percentile ranks (`rankdata(p) / N`) before blending guarantees monotonic alignment.
