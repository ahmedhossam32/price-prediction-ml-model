import pandas as pd
import numpy as np

# Load CSVs
items = pd.read_csv("data/olist_order_items_dataset.csv")
products = pd.read_csv("data/olist_products_dataset.csv")
reviews = pd.read_csv("data/olist_order_reviews_dataset.csv")
translation = pd.read_csv("data/product_category_name_translation.csv")

# Merge items with products to get category per item
items_products = items.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left"
)

# Merge with translation to get english names
items_products = items_products.merge(
    translation,
    on="product_category_name",
    how="left"
)

# Drop rows with no english category
items_products = items_products.dropna(subset=["product_category_name_english"])

# --- avg_price per category ---
avg_price = (
    items_products.groupby("product_category_name_english")["price"]
    .mean()
    .round(2)
    .rename("avg_price")
)

# --- avg_review per category ---
# reviews join to items via order_id, then items to products -> category
reviews_deduped = reviews.groupby("order_id")["review_score"].mean().reset_index()

items_with_reviews = items_products.merge(reviews_deduped, on="order_id", how="left")

avg_review = (
    items_with_reviews.groupby("product_category_name_english")["review_score"]
    .mean()
    .round(2)
    .rename("avg_review")
)

# --- median_sales_count per category ---
# Count how many times each product was ordered (number of order_item rows per product)
product_order_counts = (
    items_products.groupby("product_id")["order_id"]
    .count()
    .rename("sales_count")
    .reset_index()
)

# Attach category to each product
product_with_cat = product_order_counts.merge(
    products[["product_id", "product_category_name"]],
    on="product_id",
    how="left"
).merge(translation, on="product_category_name", how="left")

product_with_cat = product_with_cat.dropna(subset=["product_category_name_english"])

median_sales = (
    product_with_cat.groupby("product_category_name_english")["sales_count"]
    .median()
    .apply(lambda x: int(round(x)))
    .rename("median_sales_count")
)

# --- Combine all stats ---
stats = pd.concat([avg_price, avg_review, median_sales], axis=1).reset_index()
stats.columns = ["category", "avg_price", "avg_review", "median_sales_count"]

# Fill any missing avg_review with overall mean
overall_avg_review = stats["avg_review"].mean()
stats["avg_review"] = stats["avg_review"].fillna(round(overall_avg_review, 2))

# --- most_common_payment_type ---
stats["most_common_payment_type"] = "credit_card"

# --- default_max_installments ---
def max_installments(avg_p):
    if avg_p < 50:
        return 1
    elif avg_p < 100:
        return 3
    elif avg_p < 200:
        return 6
    else:
        return 12

stats["default_max_installments"] = stats["avg_price"].apply(max_installments)

# Sort alphabetically
stats = stats.sort_values("category").reset_index(drop=True)

# --- Build SQL ---
lines = []
for _, row in stats.iterrows():
    cat = row["category"].replace("'", "''")
    line = (
        f"  ('{cat}', {row['avg_price']}, {row['avg_review']}, "
        f"{row['median_sales_count']}, 'credit_card', {row['default_max_installments']})"
    )
    lines.append(line)

sql = (
    "INSERT INTO category_stats "
    "(category, avg_price, avg_review, median_sales_count, most_common_payment_type, default_max_installments)\n"
    "VALUES\n"
    + ",\n".join(lines)
    + "\nON CONFLICT (category) DO NOTHING;\n"
)

with open("category_stats_seed.sql", "w", encoding="utf-8") as f:
    f.write(sql)

print(f"Generated {len(stats)} categories -> category_stats_seed.sql\n")

# --- Top 10 by avg_price ---
top10 = stats.nlargest(10, "avg_price")[
    ["category", "avg_price", "avg_review", "median_sales_count", "default_max_installments"]
]
print("Top 10 categories by avg_price:")
print(top10.to_string(index=False))

# --- Bottom 5 sanity check ---
print("\nBottom 5 categories by avg_price:")
bot5 = stats.nsmallest(5, "avg_price")[
    ["category", "avg_price", "avg_review", "median_sales_count", "default_max_installments"]
]
print(bot5.to_string(index=False))

print(f"\nTotal categories: {len(stats)}")
print(f"avg_price range: {stats['avg_price'].min()} – {stats['avg_price'].max()}")
print(f"avg_review range: {stats['avg_review'].min()} – {stats['avg_review'].max()}")
print(f"median_sales_count range: {stats['median_sales_count'].min()} – {stats['median_sales_count'].max()}")
