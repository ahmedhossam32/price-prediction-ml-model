import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

print("="*70)
print("FINAL ML MODEL (BEST VERSION)")
print("="*70)

# =========================
# LOAD DATA
# =========================
items = pd.read_csv("data/olist_order_items_dataset.csv")
products = pd.read_csv("data/olist_products_dataset.csv")
orders = pd.read_csv("data/olist_orders_dataset.csv")
categories = pd.read_csv("data/product_category_name_translation.csv")

# =========================
# DEMAND FEATURES
# =========================
product_demand = items.groupby('product_id').size(
).reset_index(name='product_sales_count')
seller_sales = items.groupby('seller_id').size(
).reset_index(name='seller_sales_count')
seller_diversity = items.groupby('seller_id')['product_id'].nunique(
).reset_index(name='seller_product_diversity')

# =========================
# MERGE
# =========================
data = items.merge(products, on='product_id', how='left')
data = data.merge(categories, on='product_category_name', how='left')
data = data.merge(product_demand, on='product_id', how='left')
data = data.merge(seller_sales, on='seller_id', how='left')
data = data.merge(seller_diversity, on='seller_id', how='left')

data = data.merge(
    orders[['order_id', 'order_purchase_timestamp']],
    on='order_id',
    how='left'
)

# =========================
# CLEAN
# =========================
data = data[data['price'].notna()].copy()
data['order_purchase_timestamp'] = pd.to_datetime(
    data['order_purchase_timestamp'])

# =========================
# FEATURE ENGINEERING
# =========================
data['product_volume_cm3'] = (
    data['product_length_cm'] *
    data['product_height_cm'] *
    data['product_width_cm']
)

data['purchase_month'] = data['order_purchase_timestamp'].dt.month
data['purchase_day_of_week'] = data['order_purchase_timestamp'].dt.dayofweek
data['purchase_hour'] = data['order_purchase_timestamp'].dt.hour

data['density'] = data['product_weight_g'] / (data['product_volume_cm3'] + 1)
data['shipping_ratio'] = data['freight_value'] / (data['product_weight_g'] + 1)

data['is_weekend'] = data['purchase_day_of_week'].isin([5, 6]).astype(int)
data['is_peak_season'] = data['purchase_month'].isin([11, 12]).astype(int)

data['complexity_score'] = (
    data['product_photos_qty'] * 0.5 +
    data['product_description_lenght'] * 0.01 +
    data['product_name_lenght'] * 0.02
)

data['log_weight'] = np.log1p(data['product_weight_g'])
data['log_volume'] = np.log1p(data['product_volume_cm3'])

data['size_category'] = pd.cut(
    data['product_volume_cm3'],
    bins=[0, 1000, 10000, 100000, np.inf],
    labels=['small', 'medium', 'large', 'very_large']
)

# =========================
# CATEGORY AVG PRICE
# =========================
category_price = data.groupby('product_category_name_english')[
    'price'].mean().to_dict()
data['category_avg_price'] = data['product_category_name_english'].map(
    category_price)

# =========================
# FEATURES
# =========================
features = [
    'freight_value', 'product_photos_qty', 'product_weight_g', 'product_volume_cm3',
    'purchase_month', 'purchase_day_of_week', 'purchase_hour',
    'density', 'shipping_ratio', 'is_weekend', 'is_peak_season',
    'complexity_score', 'log_weight', 'log_volume',
    'product_sales_count', 'seller_sales_count', 'seller_product_diversity',
    'category_avg_price', 'product_category_name_english', 'size_category'
]

# 🔥 IMPORTANT (DO NOT REMOVE)
data_clean = data[features + ['price']].dropna()

# =========================
# OUTLIERS
# =========================
q_low = data_clean['price'].quantile(0.01)
q_high = data_clean['price'].quantile(0.99)
data_clean = data_clean[(data_clean['price'] >= q_low)
                        & (data_clean['price'] <= q_high)]

# =========================
# TARGET
# =========================
X = data_clean[features]
y = np.log1p(data_clean['price'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# PREPROCESSING
# =========================
categorical_cols = ['product_category_name_english', 'size_category']
numeric_cols = [col for col in features if col not in categorical_cols]

preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# =========================
# MODEL
# =========================
model = XGBRegressor(random_state=42, n_jobs=-1)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# =========================
# TUNING (KEEP THIS 🔥)
# =========================
param_grid = {
    'model__n_estimators': [500, 800, 1000],
    'model__max_depth': [6, 8, 10],
    'model__learning_rate': [0.01, 0.03, 0.05],
    'model__subsample': [0.7, 0.85, 1.0],
    'model__colsample_bytree': [0.7, 0.85, 1.0],
}

search = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=15,
    scoring='r2',
    cv=3,
    verbose=1,
    n_jobs=-1
)

print("\n🚀 Training with tuning...")
search.fit(X_train, y_train)

pipeline = search.best_estimator_

# =========================
# EVALUATION
# =========================
pred = pipeline.predict(X_test)

y_true = np.expm1(y_test)
y_pred = np.expm1(pred)

print("\n🎯 FINAL RESULT")
print(f"R² = {r2_score(y_true, y_pred):.4f}")
print(f"MAE = ${mean_absolute_error(y_true, y_pred):.2f}")

# =========================
# SAVE
# =========================
joblib.dump(pipeline, "best_model.pkl")

print("\n💾 Model saved!")
