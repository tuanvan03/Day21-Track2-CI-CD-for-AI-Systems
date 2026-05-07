"""
Chay tu dong nhieu thi nghiem voi cac tham so khac nhau de so sanh 2 model.
Ket qua duoc ghi vao MLflow va in ra bang so sanh.
"""

import json
import mlflow
from src.train import train

# Tat cac config de thi nghiem
EXPERIMENTS = [
    # ===== RandomForest =====
    {"model_type": "random_forest", "n_estimators": 50,  "max_depth": 5,  "min_samples_split": 2},
    {"model_type": "random_forest", "n_estimators": 100, "max_depth": 10, "min_samples_split": 5},
    {"model_type": "random_forest", "n_estimators": 200, "max_depth": 15, "min_samples_split": 5},
    {"model_type": "random_forest", "n_estimators": 300, "max_depth": 20, "min_samples_split": 2},
    # ===== LogisticRegression =====
    {"model_type": "logistic_regression", "C": 0.1,   "solver": "lbfgs", "max_iter": 2000},
    {"model_type": "logistic_regression", "C": 1.0,   "solver": "lbfgs", "max_iter": 2000},
    {"model_type": "logistic_regression", "C": 10.0,  "solver": "lbfgs", "max_iter": 2000},
    {"model_type": "logistic_regression", "C": 100.0, "solver": "lbfgs", "max_iter": 2000},
]

results = []

print(f"{'Model':<25} {'Params':<35} {'Accuracy':<10} {'F1':<10}")
print("=" * 80)

for cfg in EXPERIMENTS:
    acc = train(cfg.copy())
    # Lay lai F1 tu file da ghi
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    f1 = metrics["f1_score"]

    params_str = ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "model_type")
    print(f"{cfg['model_type']:<25} {params_str:<35} {acc:<10.4f} {f1:<10.4f}")
    results.append({"model": cfg["model_type"], "params": params_str, "accuracy": acc, "f1": f1})

print("=" * 80)

# Tim config tot nhat
best = max(results, key=lambda r: r["accuracy"])
print(f"\nBest model: {best['model']} | {best['params']}")
print(f"Best accuracy: {best['accuracy']:.4f} | Best F1: {best['f1']:.4f}")

print("\nMo MLflow UI de xem them chi tiet:")
print("  mlflow ui --backend-store-uri sqlite:///mlflow.db")
