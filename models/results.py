import json

files = [
    "models/linear_regression_result.json",
    "models/decision_tree_result.json",
    "models/random_forest_result.json",
    "models/xgboost_result.json"
]

results = []

for file in files:
    with open(file, "r") as f:
        results.append(json.load(f))

print("\nAlgorithm           R²      MAE     RMSE     MAPE(%)")
print("------------------------------------------------------")

for r in results:
    print(f"{r['algorithm']:<18} {r['r2']:.2f}   {r['mae']:.2f}   {r['rmse']:.2f}   {r['mape']:.2f}%")
