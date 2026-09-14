# The Air Jordan Resale Market — EDA & Price Prediction

An exploratory data analysis and predictive modeling project on 5,000 synthetic Air Jordan resale
transactions (2023–2025): how model, colorway, condition, and sales channel drive resale price, profit
margin, and inventory turnover time — and a regression model that predicts resale price from those
features.

**By Darius Follins**

## Project Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── jordan_market_dataset_2026.csv
├── jordan_resale_eda.ipynb        # cleaning & EDA
├── jordan_resale_modeling.ipynb   # price-prediction modeling + in-notebook widget
├── jordan_price_predictor.html    # standalone, no-install predictor (runs in-browser)
├── train_model.py                 # trains & saves the model used by app.py
├── model.pkl                      # pre-trained model (already included)
├── app.py                         # Streamlit predictor app
└── .streamlit/
    └── config.toml                # Streamlit theme
```

## Dataset

- **Source:** [Jordan Market Dataset 2026](https://www.kaggle.com/datasets/abdullahmeo/air-jordan-sneaker-market-and-resale-data2023-2026/data) (Kaggle, synthetic)
- **Size:** 5,000 transactions, 10 columns
- **Time range:** January 2023 – September 2025
- Columns: `Transaction_ID`, `Sale_Date`, `Shoe_Model`, `Colorway`, `Condition`, `Retail_Price_USD`,
  `Resale_Price_USD`, `Sales_Channel`, `Days_in_Inventory`, `Profit_Margin_USD`

Raw data lives in [`data/jordan_market_dataset_2026.csv`](data/jordan_market_dataset_2026.csv).

## Notebooks

[`jordan_resale_eda.ipynb`](jordan_resale_eda.ipynb) — cleaning & exploratory analysis:

1. Loading the data (with a data dictionary)
2. Data cleaning & validation
3. Univariate distributions (price, margin, inventory time)
4. Resale value & margin by model, colorway, and condition
5. Inventory turnover speed by sales channel, condition, and their interaction
6. Market trends over the 2023–2025 window, including monthly seasonality
7. Correlation analysis
8. Recommendations
9. Summary of findings

[`jordan_resale_modeling.ipynb`](jordan_resale_modeling.ipynb) — price-prediction modeling, building on
the EDA findings:

1. Feature engineering & explicit leakage decisions
2. Time-based train/test split (train 2023–2024, test 2025)
3. Naive baselines (mean price, assume-MSRP)
4. Linear Regression
5. Random Forest (default settings)
6. Hyperparameter tuning via grid search + cross-validation
7. Gradient Boosting
8. Model comparison
9. Prediction-accuracy plots (actual vs. predicted)
10. Feature importance (tuned Random Forest + Gradient Boosting permutation importance)
11. Summary of findings
12. Interactive price predictor (ipywidgets — requires a live Jupyter session)

## Try it live

Two interactive tools, both running the **exact same tuned Random Forest** documented in the modeling
notebook — verified to produce identical predictions for identical inputs.

- [`jordan_price_predictor.html`](jordan_price_predictor.html) — a standalone, no-install tool. The
  200-tree model was exported to JSON and re-implemented in plain JavaScript, so it runs entirely in
  your browser with no server, no Python, and no data upload. Just open the file directly.
- `app.py` — a [Streamlit](https://streamlit.io) app with the same design, but calling the real
  scikit-learn model directly (no JS re-implementation needed). Run locally with `streamlit run app.py`,
  or deploy for free via [Streamlit Community Cloud](https://streamlit.io/cloud) to get a shareable
  live link. Also includes an out-of-range warning if you enter a retail price a given model never
  actually sold for in the training data — Random Forests don't extrapolate well past what they've seen.

## Key findings — EDA

- The dataset is clean: no missing values, duplicates, or invalid ranges.
- `Condition` is the strongest simple driver of price — Deadstock pairs reliably resell above retail,
  while Used pairs typically sell at a loss. Both profit margin and days-in-inventory are **bimodal**
  distributions, and those two splits map onto `Condition` and `Sales_Channel` respectively.
- Certain colorways and models sustain meaningfully higher resale premiums than others.
- Turnover speed is a binary fast/slow effect: StockX and GOAT clear listings in ~8 days median vs.
  ~33–34 days for every other channel — notably including Stadium Goods, a real-world global platform
  that patterns with the "slow" group here rather than with StockX/GOAT.
- **Condition and channel do independent jobs**: condition drives price, channel drives turnover speed,
  and neither affects the other's outcome — this holds true within every channel individually.
- The overall market shows no strong time trend at any resolution checked (quarterly, half-year, or
  month-of-year) — no holiday-season spike, and average price/volume are both roughly flat throughout.

See the notebook's Recommendations section for how these findings translate into practical seller
advice (which channel to use, when condition vs. speed matters, etc.).

## Key findings — Modeling

- Hyperparameter tuning (grid search + 3-fold cross-validation) improved the Random Forest slightly:
  **MAE $57.52 → $56.90** on the held-out 2025 test set, with R² holding at **0.74**. Gradient Boosting
  (MAE $58.75) lands between Linear Regression (**MAE $62.07 / R² 0.72**) and the Random Forest
  versions, but doesn't beat either. All four comfortably beat naive baselines (MAE $121–130, R² ≈ 0).
- `Condition` dominates feature importance regardless of model or method: **83% of the tuned Random
  Forest's** built-in importance, and by far the largest error increase in Gradient Boosting's
  permutation importance — two different model families, two different importance measurement methods,
  same conclusion.
- All four models under-predict the high-price tail (actual sales above ~$400 get predicted around
  $450–550) — this persists across every model and after tuning, since it's a data/feature limitation,
  not an undertuned model.

## Next steps

Possible extensions: tune Gradient Boosting's hyperparameters, a wider Random Forest hyperparameter
search, a log-transformed target to reduce the tail-prediction gap.

## Setup

The two notebooks and the Streamlit app need a Python environment:

```bash
pip install -r requirements.txt
jupyter notebook jordan_resale_eda.ipynb   # or jordan_resale_modeling.ipynb
```

To run the Streamlit app:

```bash
streamlit run app.py
```

This loads the pre-trained `model.pkl` (already included — no need to retrain). If you want to
regenerate it yourself (e.g. after changing the data), run `python train_model.py` first, which
retrains the same tuned Random Forest and overwrites `model.pkl`.

The Air Jordan Resale Market/jordan_price_predictor.html needs no setup at all — just open the
file directly in any browser.
