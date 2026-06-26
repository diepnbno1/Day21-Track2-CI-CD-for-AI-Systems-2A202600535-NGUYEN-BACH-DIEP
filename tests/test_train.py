import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.train import build_model, drift_warnings, train


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Wine Quality de su dung trong test.

    pytest cung cap `tmp_path` la mot thu muc tam thoi, tu dong xoa sau khi test ket thuc.
    Ham nay dung du lieu ngau nhien nen khong can ket noi GCS hay tai file CSV thuc.
    """
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)

    acc = train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiem tra file outputs/metrics.json duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "label_distribution" in metrics
    assert set(metrics["label_distribution"]) == {"0", "1", "2"}


def test_model_file_created(tmp_path):
    """Kiem tra file models/model.pkl duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.pkl")


def test_report_file_created(tmp_path):
    """Kiem tra report.txt co confusion matrix va per-class metrics."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/report.txt")
    report = open("outputs/report.txt", encoding="utf-8").read()
    assert "Confusion matrix" in report
    assert "Per-class metrics" in report


def test_build_model_supports_multiple_algorithms():
    """Kiem tra logic chon thuat toan theo model_type."""
    _, _, random_forest = build_model(
        {"model_type": "random_forest", "random_forest": {"n_estimators": 5}}
    )
    _, _, gradient_boosting = build_model(
        {"model_type": "gradient_boosting", "gradient_boosting": {"n_estimators": 5}}
    )
    _, _, logistic_regression = build_model(
        {"model_type": "logistic_regression", "logistic_regression": {"max_iter": 50}}
    )

    assert isinstance(random_forest, RandomForestClassifier)
    assert isinstance(gradient_boosting, GradientBoostingClassifier)
    assert isinstance(logistic_regression, LogisticRegression)


def test_drift_warning_for_small_class():
    """Kiem tra canh bao khi mot lop chiem duoi 10% tap train."""
    warnings = drift_warnings({"0": 0.95, "1": 0.05, "2": 0.0})

    assert any("class 1" in warning for warning in warnings)
    assert any("class 2" in warning for warning in warnings)
