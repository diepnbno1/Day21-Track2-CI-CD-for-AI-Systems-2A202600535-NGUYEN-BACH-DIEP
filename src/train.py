import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

EVAL_THRESHOLD = 0.70
CLASS_LABELS = [0, 1, 2]
MODEL_CONFIG_KEYS = {"random_forest", "gradient_boosting", "logistic_regression"}


def _flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_params(value, name))
        else:
            flattened[name] = value
    return flattened


def _select_model_config(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    model_type = params.get("model_type", "random_forest")
    if model_type not in MODEL_CONFIG_KEYS:
        supported = ", ".join(sorted(MODEL_CONFIG_KEYS))
        raise ValueError(f"Unsupported model_type={model_type!r}. Supported: {supported}")

    nested_config = params.get(model_type)
    if isinstance(nested_config, dict):
        model_params = dict(nested_config)
    else:
        model_params = {
            key: value
            for key, value in params.items()
            if key != "model_type" and key not in MODEL_CONFIG_KEYS
        }

    if model_type == "random_forest":
        model_params.setdefault("random_state", 42)
        model_params.setdefault("n_jobs", -1)
    elif model_type == "gradient_boosting":
        model_params.setdefault("random_state", 42)
    elif model_type == "logistic_regression":
        model_params.setdefault("random_state", 42)
        model_params.setdefault("max_iter", 1000)
        model_params.setdefault("n_jobs", -1)

    return model_type, model_params


def build_model(params: dict[str, Any]):
    model_type, model_params = _select_model_config(params)

    if model_type == "random_forest":
        return model_type, model_params, RandomForestClassifier(**model_params)
    if model_type == "gradient_boosting":
        return model_type, model_params, GradientBoostingClassifier(**model_params)
    if model_type == "logistic_regression":
        return model_type, model_params, LogisticRegression(**model_params)

    raise AssertionError("model_type validation should have caught this.")


def compute_label_distribution(y_train: pd.Series) -> dict[str, float]:
    ratios = y_train.value_counts(normalize=True).reindex(CLASS_LABELS, fill_value=0.0)
    return {str(int(label)): float(ratio) for label, ratio in ratios.items()}


def drift_warnings(label_distribution: dict[str, float], min_ratio: float = 0.10) -> list[str]:
    warnings = []
    for label, ratio in label_distribution.items():
        if ratio < min_ratio:
            warnings.append(
                f"WARNING: class {label} is {ratio:.2%} of training data, below {min_ratio:.0%}."
            )
    return warnings


def write_report(
    y_true: pd.Series,
    preds,
    metrics: dict[str, Any],
    model_params: dict[str, Any],
    path: str = "outputs/report.txt",
) -> None:
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        preds,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, preds, labels=CLASS_LABELS)

    lines = [
        "Model performance report",
        "========================",
        f"model_type: {metrics['model_type']}",
        f"accuracy: {metrics['accuracy']:.6f}",
        f"f1_score_weighted: {metrics['f1_score']:.6f}",
        "",
        "Model parameters",
        "----------------",
    ]
    lines.extend(f"{key}: {value}" for key, value in sorted(model_params.items()))
    lines.extend(
        [
            "",
            "Confusion matrix (rows=true label, columns=predicted label)",
            "----------------------------------------------------------",
            "labels: 0 1 2",
        ]
    )
    lines.extend(" ".join(str(int(value)) for value in row) for row in matrix)
    lines.extend(
        [
            "",
            "Per-class metrics",
            "-----------------",
            "class precision recall f1_score support",
        ]
    )
    for index, label in enumerate(CLASS_LABELS):
        lines.append(
            f"{label} {precision[index]:.6f} {recall[index]:.6f} "
            f"{f1_values[index]:.6f} {int(support[index])}"
        )
    lines.extend(
        [
            "",
            "Training label distribution",
            "---------------------------",
        ]
    )
    for label, ratio in metrics["label_distribution"].items():
        lines.append(f"class_{label}: {ratio:.6f}")

    if metrics["drift_warnings"]:
        lines.extend(["", "Data drift warnings", "-------------------"])
        lines.extend(metrics["drift_warnings"])

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME") or "wine-quality")

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    label_distribution = compute_label_distribution(y_train)
    data_warnings = drift_warnings(label_distribution)
    for warning in data_warnings:
        print(warning)

    model_type, model_params, model = build_model(params)

    with mlflow.start_run():

        mlflow.log_params(_flatten_params(params))
        mlflow.log_param("selected_model_type", model_type)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")
        metrics = {
            "model_type": model_type,
            "accuracy": float(acc),
            "f1_score": float(f1),
            "label_distribution": label_distribution,
            "drift_warnings": data_warnings,
        }

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Model: {model_type} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        write_report(y_eval, preds, metrics, model_params)
        mlflow.log_artifact("outputs/report.txt")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return float(acc)


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
