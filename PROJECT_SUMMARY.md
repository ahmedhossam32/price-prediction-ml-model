# Project Summary — E-Commerce Price Prediction

## What This Project Does

Predicts product prices on the Olist Brazilian e-commerce platform using machine learning. Given product characteristics, shipping info, payment details, and seller/customer context — the model outputs an estimated price.

**Dataset:** 112,650 real orders from Olist (Brazil, 2016–2018)
**Final Result:** R² = 0.8332, MAE = $18.23

---

## The Journey: How We Got Here

### Step 1 — Algorithm Comparison

Four algorithms were trained and compared to find the best fit:

| Algorithm | R² | MAE |
|---|---|---|
| Linear Regression | 0.24 | $65.55 |
| Decision Tree | — | — |
| Random Forest | 0.45 | $50.63 |
| **XGBoost** | **0.69** | **$30.68** |

XGBoost won. Gradient boosting builds trees sequentially — each tree corrects what the previous got wrong — making it consistently stronger than the others on this type of structured data.

---

### Step 2 — Improving XGBoost: 0.69 → 0.78

File: `train_baseline.py`

Three changes made the biggest difference:

1. **Log-transform the price target** — Prices are right-skewed (most items R$50, some R$5000). Converting to `log(price)` makes the distribution normal. This single change was responsible for most of the jump from 0.69 to 0.78.

2. **Remove outliers** — Top and bottom 1% of prices removed. Extreme values were forcing the model off-track.

3. **Hyperparameter tuning** — RandomizedSearchCV with 15 iterations searched over learning rate, depth, and subsampling to find better settings than defaults.

---

### Step 3 — Adding New Data Sources: 0.78 → 0.83

File: `train.py` (current best)

Four new CSV tables were integrated that weren't used before:

| New Table | What We Extracted | Why It Helped |
|---|---|---|
| olist_order_payments_dataset.csv | Max installments per order | In Brazil, expensive items are paid in 10–12 installments. Cheapitems in 1. Strongest new signal. |
| olist_order_reviews_dataset.csv | Avg review per product/seller | Quality and reputation proxy. Partial substitute for missing brand data. |
| olist_customers_dataset.csv | Customer state | São Paulo buyers pay more than Northeast states. Geographic price variation. |
| olist_sellers_dataset.csv | Seller state | SP/RJ sellers tend to sell premium products. |

Additionally:
- Added `seller_avg_price` — if a seller consistently prices high, they sell premium goods (best available brand proxy)
- Fixed a data leakage issue: `category_avg_price` is now computed from training data only, not the full dataset

---

## Final Model Performance

```
R²   = 0.8332   (explains 83% of price variance)
MAE  = $18.23   (predictions within ~R$18 on average)
MAPE = 16.13%
```

The remaining ~17% is largely unexplained due to missing brand information — a fundamental dataset limitation.

---

## File Structure

```
ML_Price_Prediction_Model/
│
├── main.py                 — FastAPI REST API
├── train.py                — Best model training script (run this to retrain)
├── train_baseline.py       — Old baseline (R²=0.78, kept as rollback reference)
│
├── data/                   — 9 Olist CSV files
│
├── models/                 — One script per algorithm (for comparison)
│   ├── linear_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── xgboost_model.py
│
├── results/                — JSON metrics for every model
│   ├── best_model_result.json        (R²=0.8332)
│   ├── xgboost_baseline_result.json  (R²=0.6876)
│   ├── random_forest_result.json     (R²=0.4512)
│   ├── decision_tree_result.json
│   └── linear_regression_result.json
│
└── saved_models/
    ├── best_model.pkl        — Production model (used by API)
    └── best_model_baseline.pkl — Previous version (rollback)
```

---

## Quick Start

**Run the API:**
```bash
uvicorn main:app --reload
```

**Retrain the model:**
```bash
python train.py
```

**Run a comparison model (example):**
```bash
python models/random_forest.py
```

---

## API Usage

`POST /predict` with JSON body:

```json
{
  "freight_value": 15.50,
  "product_photos_qty": 3,
  "product_weight_g": 800,
  "product_length_cm": 20,
  "product_height_cm": 10,
  "product_width_cm": 15,
  "product_description_lenght": 500,
  "product_name_lenght": 45,
  "product_category_name_english": "electronics",
  "purchase_month": 6,
  "max_installments": 3,
  "customer_state": "SP",
  "seller_state": "SP"
}
```

Response:
```json
{ "predicted_price": 189.50 }
```

All fields except the first 10 are optional with sensible defaults.

---

## Known Limitations

- **No brand data** — cannot distinguish Apple from generic. `seller_avg_price` is the best available proxy.
- **Brazil-specific** — geographic and economic patterns are Brazil-specific.
- **Static** — model does not update automatically as prices change over time.

---

## Technologies

Python · XGBoost · scikit-learn · pandas · FastAPI · joblib
