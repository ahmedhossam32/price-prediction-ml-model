"""
MAIN TRAINING SCRIPT  —  Full dataset, best model
R²=0.8332, MAE=$18.23

Features used:
  + payment installments & type  (olist_order_payments_dataset.csv)
  + customer state               (olist_customers_dataset.csv)
  + seller state                 (olist_sellers_dataset.csv)
  + product & seller review avg  (olist_order_reviews_dataset.csv)
  + raw description/name length
  + seller_avg_price             (brand proxy, train-only)
  + category_avg_price           (train-only, no leakage)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

print("=" * 70)
print("EXPERIMENTAL MODEL v4  —  Full dataset")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# 1. LOAD ALL TABLES
# ─────────────────────────────────────────────────────────────
items      = pd.read_csv(os.path.join(DATA, "olist_order_items_dataset.csv"))
products   = pd.read_csv(os.path.join(DATA, "olist_products_dataset.csv"))
orders     = pd.read_csv(os.path.join(DATA, "olist_orders_dataset.csv"))
categories = pd.read_csv(os.path.join(DATA, "product_category_name_translation.csv"))
reviews    = pd.read_csv(os.path.join(DATA, "olist_order_reviews_dataset.csv"))
payments   = pd.read_csv(os.path.join(DATA, "olist_order_payments_dataset.csv"))
customers  = pd.read_csv(os.path.join(DATA, "olist_customers_dataset.csv"))
sellers    = pd.read_csv(os.path.join(DATA, "olist_sellers_dataset.csv"))

print(f"Items: {len(items):,} | Products: {len(products):,} | Reviews: {len(reviews):,}")
print(f"Payments: {len(payments):,} | Customers: {len(customers):,}")

# ─────────────────────────────────────────────────────────────
# 2. AGGREGATE FEATURES FROM NEW TABLES
# ─────────────────────────────────────────────────────────────

# — Payment: max installments + dominant payment type per order
#   NOTE: payment_value is NOT used — it equals price+freight (target leakage)
order_pay = (
    payments
    .groupby("order_id")
    .agg(
        max_installments=("payment_installments", "max"),
        payment_type_mode=("payment_type", lambda x: x.mode().iloc[0])
    )
    .reset_index()
)

# — Reviews: one row per order (keep latest), then avg per product/seller
reviews_clean = (
    reviews
    .sort_values("review_creation_date")
    .drop_duplicates("order_id", keep="last")
    [["order_id", "review_score"]]
)
review_items = reviews_clean.merge(
    items[["order_id", "product_id", "seller_id"]], on="order_id", how="left"
)
product_avg_review = (
    review_items.groupby("product_id")["review_score"]
    .mean().reset_index(name="product_avg_review")
)
seller_avg_review = (
    review_items.groupby("seller_id")["review_score"]
    .mean().reset_index(name="seller_avg_review")
)

# — Demand features
product_demand   = items.groupby("product_id").size().reset_index(name="product_sales_count")
seller_sales     = items.groupby("seller_id").size().reset_index(name="seller_sales_count")
seller_diversity = (
    items.groupby("seller_id")["product_id"]
    .nunique().reset_index(name="seller_product_diversity")
)

# ─────────────────────────────────────────────────────────────
# 3. BUILD MAIN DATAFRAME
# ─────────────────────────────────────────────────────────────
data = (
    items
    .merge(products,          on="product_id",   how="left")
    .merge(categories,        on="product_category_name", how="left")
    .merge(product_demand,    on="product_id",   how="left")
    .merge(seller_sales,      on="seller_id",    how="left")
    .merge(seller_diversity,  on="seller_id",    how="left")
    .merge(product_avg_review,on="product_id",   how="left")
    .merge(seller_avg_review, on="seller_id",    how="left")
    .merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
    .merge(orders[["order_id", "order_purchase_timestamp", "customer_id"]],
           on="order_id", how="left")
    .merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    .merge(order_pay,         on="order_id",     how="left")
)

# ─────────────────────────────────────────────────────────────
# 4. CLEAN
# ─────────────────────────────────────────────────────────────
data = data[data["price"].notna()].copy()
data["order_purchase_timestamp"] = pd.to_datetime(data["order_purchase_timestamp"])

# Fill review medians before feature engineering
med_prod_review   = data["product_avg_review"].median()
med_seller_review = data["seller_avg_review"].median()
data["product_avg_review"] = data["product_avg_review"].fillna(med_prod_review)
data["seller_avg_review"]  = data["seller_avg_review"].fillna(med_seller_review)

# Fill installments median (orders with no payment row)
data["max_installments"] = data["max_installments"].fillna(data["max_installments"].median())

# ─────────────────────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
data["product_volume_cm3"] = (
    data["product_length_cm"] * data["product_height_cm"] * data["product_width_cm"]
)
data["density"]        = data["product_weight_g"] / (data["product_volume_cm3"] + 1)
data["shipping_ratio"] = data["freight_value"]    / (data["product_weight_g"]   + 1)
data["log_weight"]     = np.log1p(data["product_weight_g"])
data["log_volume"]     = np.log1p(data["product_volume_cm3"])

data["purchase_month"] = data["order_purchase_timestamp"].dt.month
data["is_peak_season"] = data["purchase_month"].isin([11, 12]).astype(int)

data["complexity_score"] = (
    data["product_photos_qty"]          * 0.5 +
    data["product_description_lenght"]  * 0.01 +
    data["product_name_lenght"]         * 0.02
)

data["size_category"] = pd.cut(
    data["product_volume_cm3"],
    bins=[0, 1000, 10000, 100000, np.inf],
    labels=["small", "medium", "large", "very_large"]
)

# ─────────────────────────────────────────────────────────────
# 6. FEATURE LIST
# ─────────────────────────────────────────────────────────────
numeric_base = [
    "freight_value", "product_photos_qty",
    "product_weight_g", "product_volume_cm3",
    "product_description_lenght", "product_name_lenght",
    "density", "shipping_ratio", "complexity_score",
    "log_weight", "log_volume",
    "purchase_month", "is_peak_season",
    "product_avg_review", "seller_avg_review",
    "max_installments",
    "product_sales_count", "seller_sales_count", "seller_product_diversity",
]

categorical_base = [
    "product_category_name_english",
    "size_category",
    "payment_type_mode",
    "customer_state",
    "seller_state",
]

all_features = numeric_base + categorical_base

# ─────────────────────────────────────────────────────────────
# 7. CLEAN ROWS + OUTLIERS
# ─────────────────────────────────────────────────────────────
cols_needed = all_features + ["price", "seller_id"]  # product_category_name_english already in all_features
data_clean = data[cols_needed].dropna(subset=numeric_base + ["price"]).copy()

q_low  = data_clean["price"].quantile(0.01)
q_high = data_clean["price"].quantile(0.99)
data_clean = data_clean[(data_clean["price"] >= q_low) & (data_clean["price"] <= q_high)]

# Fill remaining categorical NaN with "unknown"
for col in categorical_base:
    data_clean[col] = data_clean[col].fillna("unknown").astype(str)

print(f"Clean rows after outlier removal: {len(data_clean):,}")

# ─────────────────────────────────────────────────────────────
# 8. SPLIT FIRST  (prevents any aggregate leakage into test set)
# ─────────────────────────────────────────────────────────────
X_raw = data_clean[all_features].copy()
y     = np.log1p(data_clean["price"])

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────────────────────
# 9. TRAIN-ONLY AGGREGATES  (no leakage)
# ─────────────────────────────────────────────────────────────
global_avg_price = float(np.expm1(y_train.mean()))

# category_avg_price
train_prices = data_clean.loc[X_train_raw.index, ["product_category_name_english"]].copy()
train_prices["price"] = np.expm1(y_train.values)
cat_avg_map = train_prices.groupby("product_category_name_english")["price"].mean().to_dict()

# seller_avg_price  (brand proxy: premium sellers consistently price high)
train_sellers = data_clean.loc[X_train_raw.index, "seller_id"]
seller_avg_map = (
    pd.Series(np.expm1(y_train.values), index=train_sellers.values)
    .groupby(level=0).mean()
    .to_dict()
)

def add_train_only_features(X, index):
    X = X.copy()
    cats   = data_clean.loc[index, "product_category_name_english"]
    sids   = data_clean.loc[index, "seller_id"]
    X["category_avg_price"] = cats.map(cat_avg_map).fillna(global_avg_price).values
    X["seller_avg_price"]   = sids.map(seller_avg_map).fillna(global_avg_price).values
    return X

X_train = add_train_only_features(X_train_raw, X_train_raw.index)
X_test  = add_train_only_features(X_test_raw,  X_test_raw.index)

numeric_final      = numeric_base + ["category_avg_price", "seller_avg_price"]
categorical_final  = categorical_base
features_final     = numeric_final + categorical_final

X_train = X_train[features_final]
X_test  = X_test[features_final]

print(f"Total features: {len(features_final)}")
print(f"  Numeric: {len(numeric_final)} | Categorical: {len(categorical_final)}")

# ─────────────────────────────────────────────────────────────
# 10. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", "passthrough", numeric_final),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_final),
])

# ─────────────────────────────────────────────────────────────
# 11. MODEL + TUNING
# ─────────────────────────────────────────────────────────────
xgb = XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb),
])

param_grid = {
    "model__n_estimators":     [500, 800, 1000, 1200],
    "model__max_depth":        [5, 6, 8, 10],
    "model__learning_rate":    [0.01, 0.03, 0.05, 0.08],
    "model__subsample":        [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.85, 1.0],
    "model__min_child_weight": [1, 3, 5],
    "model__gamma":            [0, 0.1, 0.3],
}

search = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=25,
    scoring="r2",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)

print("\nTraining with tuning (25 iterations × 3 folds = 75 fits)...")
search.fit(X_train, y_train)

best_pipeline = search.best_estimator_
print(f"\nBest hyperparameters:")
for k, v in search.best_params_.items():
    print(f"  {k.replace('model__', '')}: {v}")

# ─────────────────────────────────────────────────────────────
# 12. EVALUATION
# ─────────────────────────────────────────────────────────────
pred   = best_pipeline.predict(X_test)
y_true = np.expm1(y_test)
y_pred = np.expm1(pred)

r2   = r2_score(y_true, y_pred)
mae  = mean_absolute_error(y_true, y_pred)
mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

BASELINE_R2  = 0.7831
BASELINE_MAE = 20.50

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"{'Metric':<12} {'v4 (new)':>12} {'Baseline':>12} {'Delta':>10}")
print("-" * 60)
print(f"{'R²':<12} {r2:>12.4f} {BASELINE_R2:>12.4f} {r2 - BASELINE_R2:>+10.4f}")
print(f"{'MAE':<12} {mae:>11.2f}$ {BASELINE_MAE:>11.2f}$ {mae - BASELINE_MAE:>+9.2f}$")
print(f"{'MAPE':<12} {mape:>11.2f}%")
print(f"{'Features':<12} {len(features_final):>12}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 13. SAVE
# ─────────────────────────────────────────────────────────────
root_dir = os.path.dirname(__file__)

joblib.dump(best_pipeline, os.path.join(root_dir, "saved_models", "best_model.pkl"))

result = {
    "algorithm": "XGBoost",
    "r2":  float(r2),
    "mae": float(mae),
    "mape": mape,
    "features_count": len(features_final),
    "features_numeric": numeric_final,
    "features_categorical": categorical_final,
    "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
    "vs_baseline": {
        "r2_delta":  float(r2 - BASELINE_R2),
        "mae_delta": float(mae - BASELINE_MAE),
    },
    "new_tables_used": [
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_customers_dataset.csv",
        "olist_sellers_dataset.csv",
    ],
    "leakage_fixed": True,
}

with open(os.path.join(root_dir, "results", "best_model_result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"\nModel saved → saved_models/best_model.pkl")
print(f"Results saved → results/best_model_result.json")

if r2 > BASELINE_R2:
    print(f"\nIMPROVED over baseline by +{r2 - BASELINE_R2:.4f} R²")
else:
    print(f"\nDid not beat baseline ({r2 - BASELINE_R2:+.4f} R²)")
