# 01. Numerai Classic Hedge Fund Alpha Engine

Automated, production-grade alpha generation and submission pipeline for the **Numerai Classic** tournament. The pipeline downloads the latest tournament data (`v6.0 "Unity"`), trains a regularized gradient boosted decision tree on an orthogonal feature subset, normalizes predictions via quantile ranking, and uploads daily signals to the live hedge fund.

* **Live Model:** [`sachin23_an`](https://numer.ai/sachin23_an)
* **Author:** Sachin Kumar ([GitHub: sachin23-an](https://github.com/sachin23-an))
* **Verified Round:** Round 1404 live submission accepted

---

## 1. Problem Formulation & Hedge Fund Context

[Numerai](https://numer.ai) is a crowd-sourced quantitative hedge fund that aggregates market-neutral machine learning signals across thousands of data scientists into a single meta-model that trades real capital in global equity markets.

### The Quant Challenge
* **Asset Universe:** Over 7,000 global equities identified by synthetic `numerai_ticker` hashes.
* **Feature Space:** Over 2,000 pre-computed financial features grouped into correlation clusters, normalized to uniform bins $[0, 0.25, 0.5, 0.75, 1.0]$.
* **Target:** Market-neutralized, factor-neutralized residual stock returns over a forward 20-day holding period.
* **Evaluation Metrics:**
  * **CORR20 (Spearman Rank Correlation):** Correlation between submitted rank predictions and realized returns, neutralized against country, sector, and risk factors.
  * **MMC20 (Meta-Model Contribution):** The orthogonal alpha that your model provides relative to the existing consensus of all other staked models in the fund.
  * **FNC (Factor Neutral Correlation):** Residual correlation after orthogonalizing predictions against ~40 known systematic risk factors.

In quantitative equity trading, the signal-to-noise ratio is exceptionally low. A model achieving a consistent CORR of **0.02 to 0.04** with high Sharpe ratio generates substantial risk-adjusted compounding alpha.

---

## 2. Architecture & Pipeline Design

```
+-------------------------------------------------------------+
|                 NumerAPI (v6.0 "Unity")                     |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|        Feature Selection: 'small' Set (101 Features)        |
|  - Eliminates collinear noise                               |
|  - Sub-2-minute training & < 2 GB RAM footprint             |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|               LightGBM Regressor (Tree-Constrained)         |
|  - max_depth = 5, num_leaves = 31                           |
|  - colsample_bytree = 0.8, subsample = 0.8                  |
|  - Prevents overfitting on low SNR financial data           |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|               Quantile Rank Transformation                  |
|  - rankdata(predictions) / N -> Uniform [0, 1]              |
|  - Formats: ['numerai_ticker', 'signal']                    |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|       Direct Live Submission via NumerAPI (Round Daily)     |
+-------------------------------------------------------------+
```

### Why the 'Small' Feature Set?
The full dataset contains 2,300+ features and 4.3M rows (~15 GB parquet). Training all features on standard machines risks out-of-memory crashes and severe feature collinearity. 

By utilizing Numerai's curated `'small'` feature set (101 features), we:
1. Capture the primary orthogonal principal directions across all major factor clusters.
2. Reduce memory footprint to **< 1.8 GB RAM**.
3. Enable fast, deterministic daily retraining in **under 90 seconds**.

---

## 3. Directory Structure

```text
01-numerai/
├── submit.py          # Standalone, automated training & live submission script
├── README.md          # Project documentation, quant theory, and operational guide
└── data/              # Downloaded tournament data and submission artifacts (gitignored)
```

---

## 4. Quickstart & Execution

### Prerequisites
Install the required packages from the project root:
```bash
pip install -r ../requirements.txt
```

### Step 1: Set Your API Credentials
Generate an API key with **Upload Submissions** permission in your [Numerai Account Settings](https://numer.ai/account):

```bash
# In Windows PowerShell:
$env:NUMERAI_PUBLIC_ID = "YOUR_PUBLIC_ID"
$env:NUMERAI_SECRET_KEY = "YOUR_SECRET_KEY"

# Or in Linux / macOS / WSL:
export NUMERAI_PUBLIC_ID="YOUR_PUBLIC_ID"
export NUMERAI_SECRET_KEY="YOUR_SECRET_KEY"
```

### Step 2: Dry Run (Train & Generate Predictions Without Uploading)
Test the entire pipeline locally without sending data to Numerai:
```bash
python submit.py --dry-run
```
Output:
* Downloads active feature metadata (`v6.0/features.json`).
* Loads training data and trains the LightGBM alpha model.
* Generates live predictions for all 7,000+ assets in the current round.
* Saves `data/submission.parquet` with verified columns `['numerai_ticker', 'signal']`.

### Step 3: Live Submission
Execute a live upload to your model:
```bash
python submit.py --model-name sachin23_an
```

---

## 5. Automated Daily Scheduling

Numerai releases new live market data every weekday at **13:30 UTC (7:00 PM IST)** with a 1-hour submission deadline.

### Windows Task Scheduler Setup:
1. Open **Task Scheduler** in Windows.
2. Click **Create Basic Task** -> Name it `Numerai_Daily_Submission`.
3. Set Trigger: **Daily at 7:00 PM**.
4. Set Action: **Start a program**:
   * **Program/script:** `python.exe` (or path to your virtualenv python)
   * **Add arguments:** `D:\Projects\Quant\quant-competition-research\01-numerai\submit.py --model-name sachin23_an`
   * **Start in:** `D:\Projects\Quant\quant-competition-research\01-numerai`

### Linux / Crontab Setup:
```cron
# Runs every weekday at 13:35 UTC
35 13 * * 1-5 cd /path/to/quant-competition-research/01-numerai && NUMERAI_PUBLIC_ID=xxx NUMERAI_SECRET_KEY=yyy python3 submit.py >> numerai.log 2>&1
```

---

## 6. Interview Talking Points

* **Risk Management & Neutralization:** Unlike retail trading models that fit raw price trajectories, Numerai targets are pre-neutralized against market beta, sector tilts, and volatility factors. Success requires discovering true residual alpha, not disguised beta.
* **Avoiding Overfitting in Financial Data:** High-capacity deep neural networks or deep tree models quickly overfit noise when predicting residual stock returns. Restricting tree depth (`max_depth=5`) and applying feature column subsampling (`colsample_bytree=0.8`) regularizes the hypothesis space.
* **Production Reliability:** The pipeline dynamically discovers active tournament versions and feature schemas via `napi.list_datasets()`, preventing silent failures when Numerai updates data schemas.
