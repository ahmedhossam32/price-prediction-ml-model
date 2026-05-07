# E-Commerce Price Prediction Model

## Overview

This project predicts product prices on the **Olist Brazilian e-commerce platform** using machine learning. It covers the full ML pipeline: data loading and merging from multiple sources, feature engineering, algorithm comparison, iterative model improvement, and deployment as a REST API.

**Final model: XGBoost — R² = 0.8332, MAE = $18.23**

---

## Project Structure

```
ML_Price_Prediction_Model/
│
├── main.py                          — FastAPI REST API for predictions
├── train.py                         — Main training script (best model)
├── train_baseline.py                — Baseline training script (kept for reference)
├── analyze_categories.py            — Generates per-category stats & SQL seed file
├── category_stats_seed.sql          — SQL INSERT seed for category_stats table (71 categories)
│
├── data/                            — Raw Olist dataset (9 CSV files)
│
├── models/                          — Algorithm comparison scripts
│   ├── linear_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── xgboost_model.py
│
├── results/                         — Evaluation metrics for all models
│   ├── best_model_result.json       — Final model results (R²=0.8332)
│   ├── xgboost_baseline_result.json
│   ├── random_forest_result.json
│   ├── decision_tree_result.json
│   └── linear_regression_result.json
│
└── saved_models/
    ├── best_model.pkl               — Current best model (production)
    └── best_model_baseline.pkl      — Previous baseline (rollback)
```

---

## Dataset

The Olist Brazilian E-Commerce dataset — a real-world dataset of 112,000+ orders. Nine CSV files were merged to build a rich feature set:

| File | Records | Used For |
|---|---|---|
| olist_order_items_dataset.csv | 112,650 | Prices, freight, seller/product IDs |
| olist_products_dataset.csv | 32,951 | Dimensions, weight, photos |
| olist_orders_dataset.csv | 99,441 | Timestamps, customer IDs |
| olist_order_payments_dataset.csv | 103,886 | Installments, payment type |
| olist_order_reviews_dataset.csv | 99,224 | Review scores |
| olist_customers_dataset.csv | 99,441 | Customer state |
| olist_sellers_dataset.csv | 3,095 | Seller state |
| product_category_name_translation.csv | 71 | Portuguese → English category names |
| olist_geolocation_dataset.csv | — | Available, not used (marginal gain) |

---

## Algorithm Comparison

Four algorithms were trained and compared on identical data to select the best approach.

| Algorithm | R² Score | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Linear Regression | 0.2402 | $65.55 | 168.54 | 62.66% |
| Decision Tree | — | — | — | — |
| Random Forest | 0.4512 | $50.63 | 143.24 | 45.11% |
| **XGBoost** | **0.6876** | **$30.68** | **108.06** | **23.84%** |

**XGBoost was selected** because gradient boosting — building trees sequentially where each tree corrects the errors of the previous — consistently outperforms bagging methods (Random Forest) on structured/tabular data.

---

## Iterative Improvement: 0.69 → 0.78 → 0.83

Once XGBoost was selected, the model was improved through multiple iterations.

### Iteration 1 — Baseline XGBoost: R² = 0.69
`models/xgboost_model.py`

Raw XGBoost with 500 estimators, no tuning, no preprocessing of the target variable. The main limitation was the price distribution: most items cost R$20–100 but some cost R$5,000+. This heavy right-skew forced the model to split capacity between cheap and expensive items.

### Iteration 2 — train_baseline.py: R² = 0.78

Key changes and why they worked:

| Change | Effect |
|---|---|
| `y = np.log1p(price)` as target | Biggest single gain. Converts skewed prices into a normal distribution XGBoost can learn cleanly. Predictions are converted back with `np.expm1()`. |
| Outlier removal (1st–99th percentile) | Removes extreme prices that acted as noise |
| Hyperparameter tuning (RandomizedSearchCV, 15 iterations, 3-fold CV) | Found better learning rate, depth, and subsampling than defaults |
| Additional engineered features | log_weight, log_volume, density, shipping_ratio, size_category |

### Iteration 3 — train.py (current best): R² = 0.83

Key changes and why they worked:

| Change | Effect |
|---|---|
| Payment installments (`max_installments`) | Strongest new signal. In Brazil, cheap items are paid in 1 installment, expensive items in 10–12. Directly encodes price tier. |
| Review scores (`product_avg_review`, `seller_avg_review`) | Quality proxy. High-rated products/sellers tend to be premium. Partial substitute for missing brand data. |
| Geographic features (`customer_state`, `seller_state`) | São Paulo buyers pay more than Northeast states. Encodes purchasing power and market segment. |
| `seller_avg_price` (brand proxy) | If a seller's average price is R$2000+, they sell premium goods. Best available substitute for missing brand column. |
| Fixed `category_avg_price` leakage | Previously computed from full dataset. Now computed from training data only — honest evaluation. |
| Removed noisy temporal features | `purchase_hour`, `day_of_week`, `is_weekend` did not help — prices don't vary by time of day |
| 25 tuning iterations | Wider search, better hyperparameters |

---

## Final Model: Results

```
R²   = 0.8332
MAE  = $18.23
MAPE = 16.13%
Features used: 26
```

The remaining ~17% unexplained variance is largely due to missing brand data (Apple vs. generic, Nike vs. unknown). This is a fundamental dataset limitation — no ML technique can recover information that isn't in the data.

---

## Features Used (26 total)

**Numeric (21):**
- Physical: `freight_value`, `product_weight_g`, `product_volume_cm3`, `product_photos_qty`, `product_length_cm`, `product_height_cm`, `product_width_cm`
- Derived: `density`, `shipping_ratio`, `log_weight`, `log_volume`, `complexity_score`
- Text: `product_description_lenght`, `product_name_lenght`
- Temporal: `purchase_month`, `is_peak_season`
- Demand: `product_sales_count`, `seller_sales_count`, `seller_product_diversity`
- Quality: `product_avg_review`, `seller_avg_review`
- Payment: `max_installments`
- Aggregates (train-only): `category_avg_price`, `seller_avg_price`

**Categorical (5):**
- `product_category_name_english`, `size_category`, `payment_type_mode`, `customer_state`, `seller_state`

---

## Running the API

```bash
uvicorn main:app --reload
```

**Endpoint:** `POST /predict`

**Minimal request (required fields only):**
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
  "purchase_month": 6
}
```

All other fields (review scores, installments, state, etc.) have sensible defaults and are optional.

**Response:**
```json
{
  "predicted_price": 189.50
}
```

---

## Retraining the Model

```bash
python train.py
```

Saves the retrained model to `saved_models/best_model.pkl` and results to `results/best_model_result.json`.

---

## Category Statistics (Database Seed)

`analyze_categories.py` reads the raw Olist CSVs and computes per-category aggregate stats used to seed a backend database:

| Column | Description |
|---|---|
| `category` | English product category name |
| `avg_price` | Mean item price across all orders in that category |
| `avg_review` | Mean review score (order-level, deduplicated) |
| `median_sales_count` | Median number of times a product in this category was ordered |
| `most_common_payment_type` | Dominant payment method (`credit_card` across all 71 categories) |
| `default_max_installments` | Installment ceiling derived from avg price (1 / 3 / 6 / 12) |

Running the script regenerates `category_stats_seed.sql` — a single `INSERT ... ON CONFLICT DO NOTHING` statement covering all 71 categories. Import it into a PostgreSQL database:

```bash
python analyze_categories.py        # regenerate the SQL
psql -d your_db -f category_stats_seed.sql
```

---

## Known Limitations

1. **No brand data** — The dataset has no brand column. An iPhone and a generic phone in the same category look identical to the model. `seller_avg_price` and `seller_avg_review` are the best available proxies.

2. **Brazil-specific** — Trained on 2016–2018 Brazilian e-commerce. Purchasing power, category pricing, and geographic patterns are Brazil-specific.

3. **Static model** — Prices change over time. The model does not retrain automatically.

---

## Technologies

| Library | Purpose |
|---|---|
| XGBoost | Core model |
| scikit-learn | Pipeline, preprocessing, evaluation |
| pandas / numpy | Data manipulation |
| FastAPI | REST API |
| joblib | Model serialization |
