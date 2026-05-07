import os

# Set MLflow tracking URI truoc khi import mlflow de tranh file store fallback
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

EVAL_THRESHOLD = 0.70

# Dam bao tracking URI duoc set (cho ca truong hop import)
TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(TRACKING_URI)

# Tao / kiem tra experiment va artifact directory
_exp_name = "track2-experiment"
if mlflow.get_experiment_by_name(_exp_name) is None:
    mlflow.create_experiment(_exp_name)
mlflow.set_experiment(_exp_name)


MODEL_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}


def _build_model(model_type: str, params: dict):
    """
    Khoi tao mo hinh tuong ung voi model_type.
    Loai bo cac tham so khong phu hop voi tung loai mo hinh.
    """
    if model_type == "random_forest":
        # Chi lay cac tham so RandomForest can
        rf_params = {
            k: params[k] for k in ("n_estimators", "max_depth", "min_samples_split")
            if k in params
        }
        return RandomForestClassifier(random_state=42, **rf_params)

    elif model_type == "logistic_regression":
        lr_params = {
            k: params[k] for k in ("C", "solver", "max_iter", "penalty")
            if k in params and k != "penalty"  # penalty chi ap dung voi solver moi
        }
        # LogisticRegression mac dinh dung 'l2' penalty, khong can truyen explicit
        return LogisticRegression(random_state=42, **lr_params)

    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"Supported: {list(MODEL_REGISTRY.keys())}")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so va model_type.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: Kiem tra phan phoi nhan
    label_counts = y_train.value_counts().sort_index()
    total = len(y_train)
    label_dist = {str(k): float(v / total) for k, v in label_counts.items()}
    print("\n[Bonus 5] Phan phoi nhan tren tap huan luyen:")
    for k, v in label_dist.items():
        pct = v * 100
        warn = " *** CANH BAO: < 10% ***" if v < 0.10 else ""
        print(f"  Lop {k}: {pct:.2f}%{warn}")
    # Kiem tra neu co lop < 10%
    low_classes = [k for k, v in label_dist.items() if v < 0.10]

    with mlflow.start_run():

        # Lay model_type va ghi nhan vao MLflow
        model_type = params.pop("model_type", "random_forest")
        mlflow.log_param("model_type", model_type)

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen mo hinh theo model_type
        model = _build_model(model_type, params)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # Bonus 3: Confusion matrix + precision/recall tung lop
        cm = confusion_matrix(y_eval, preds)
        precision_per_class = precision_score(y_eval, preds, average=None)
        recall_per_class = recall_score(y_eval, preds, average=None)

        # Ghi report.txt
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w") as f:
            f.write("=" * 60 + "\n")
            f.write("BANG BAO CAO HIEU SUAT MO HINH\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model type: {model_type}\n")
            f.write(f"Accuracy:   {acc:.4f}\n")
            f.write(f"F1 Score:   {f1:.4f}\n\n")

            f.write("-" * 40 + "\n")
            f.write("Confusion Matrix:\n")
            f.write("-" * 40 + "\n")
            f.write("        " + " ".join(f"Du doan {i}" for i in range(len(cm))) + "\n")
            for i, row in enumerate(cm):
                f.write(f"Thuc {i}:  " + " ".join(f"{v:>8}" for v in row) + "\n")
            f.write("\n")

            f.write("-" * 40 + "\n")
            f.write("Chi tiet tung lop:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Lop':<6} {'Precision':<12} {'Recall':<12} {'So mau':<8}\n")
            f.write("-" * 40 + "\n")
            for i in range(len(precision_per_class)):
                count = int((y_eval == i).sum())
                f.write(f"{i:<6} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} {count:<8}\n")

            # Bonus 5: Phan phoi nhan
            f.write("\n")
            f.write("-" * 40 + "\n")
            f.write("Phan phoi nhan (tap huan luyen):\n")
            f.write("-" * 40 + "\n")
            for k, v in sorted(label_dist.items()):
                f.write(f"  Lop {k}: {v*100:.2f}%\n")

            f.write("\n" + "=" * 60 + "\n")

        print("\n[Bonus 3] Da ghi bao cao vao outputs/report.txt")

        # TODO 8: Luu metrics ra file outputs/metrics.json
        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "accuracy": acc,
                "f1_score": f1,
                "label_distribution": label_dist,
                "precision_per_class": [float(p) for p in precision_per_class],
                "recall_per_class": [float(r) for r in recall_per_class],
                "confusion_matrix": cm.tolist(),
            }, f, indent=2)

        # TODO 9: Luu mo hinh ra file models/model.pkl
        # File nay duoc upload len GCS o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        # pass  # xoa dong nay sau khi hoan thanh tat ca TODO ben tren

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
