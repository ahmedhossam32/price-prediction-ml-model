from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("saved_models/best_model.pkl")


# =========================
# INPUT SCHEMA
# =========================
class ProductInput(BaseModel):
    # Product physical properties
    freight_value: float = Field(..., ge=0)
    product_photos_qty: int = Field(..., ge=0)
    product_weight_g: float = Field(..., gt=0)
    product_length_cm: float = Field(..., gt=0)
    product_height_cm: float = Field(..., gt=0)
    product_width_cm: float = Field(..., gt=0)
    product_description_lenght: int = Field(..., ge=0)
    product_name_lenght: int = Field(..., ge=0)

    # Category
    product_category_name_english: str

    # Temporal
    purchase_month: int = Field(..., ge=1, le=12)

    # Demand / market context (optional — use 0 if unknown)
    product_sales_count: int = 0
    seller_sales_count: int = 0
    seller_product_diversity: int = 0

    # Review scores (optional — defaults to typical dataset median)
    product_avg_review: float = Field(default=4.0, ge=1.0, le=5.0)
    seller_avg_review: float = Field(default=4.0, ge=1.0, le=5.0)

    # Payment (optional)
    max_installments: int = Field(default=1, ge=1, le=24)
    payment_type_mode: str = "credit_card"   # credit_card | boleto | voucher | debit_card

    # Geography (optional — defaults to SP, most common)
    customer_state: str = "SP"
    seller_state: str = "SP"

    # Pre-computed averages (optional — use dataset-level defaults if unknown)
    category_avg_price: float = 100.0
    seller_avg_price: float = 100.0


# =========================
# ROOT
# =========================
@app.get("/")
def home():
    return {"message": "Price Prediction API v4 — R²=0.8332, MAE=$18.23"}


# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(data: ProductInput):
    try:
        d = data.dict()

        # ── Feature engineering (must match train_model_v4.py exactly) ──
        vol = d["product_length_cm"] * d["product_height_cm"] * d["product_width_cm"]

        d["product_volume_cm3"] = vol
        d["density"]            = d["product_weight_g"] / (vol + 1)
        d["shipping_ratio"]     = d["freight_value"] / (d["product_weight_g"] + 1)
        d["log_weight"]         = np.log1p(d["product_weight_g"])
        d["log_volume"]         = np.log1p(vol)
        d["is_peak_season"]     = int(d["purchase_month"] in [11, 12])
        d["complexity_score"]   = (
            d["product_photos_qty"]         * 0.5 +
            d["product_description_lenght"] * 0.01 +
            d["product_name_lenght"]        * 0.02
        )

        size_bins   = [0, 1000, 10000, 100000, np.inf]
        size_labels = ["small", "medium", "large", "very_large"]
        d["size_category"] = size_labels[
            next(i for i, b in enumerate(size_bins[1:]) if vol < b)
        ]

        # ── Build DataFrame in the exact column order the pipeline expects ──
        feature_order = [
            "freight_value", "product_photos_qty",
            "product_weight_g", "product_volume_cm3",
            "product_description_lenght", "product_name_lenght",
            "density", "shipping_ratio", "complexity_score",
            "log_weight", "log_volume",
            "purchase_month", "is_peak_season",
            "product_avg_review", "seller_avg_review",
            "max_installments",
            "product_sales_count", "seller_sales_count", "seller_product_diversity",
            "category_avg_price", "seller_avg_price",
            "product_category_name_english", "size_category",
            "payment_type_mode", "customer_state", "seller_state",
        ]

        df = pd.DataFrame([{k: d[k] for k in feature_order}])

        pred  = model.predict(df)
        price = float(np.expm1(pred[0]))

        return {"predicted_price": round(price, 2)}

    except Exception as e:
        return {"error": str(e)}
