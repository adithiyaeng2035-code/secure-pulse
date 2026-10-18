"""Cardiometabolic risk model training and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from modules.data_processor import FEATURE_COLUMNS, prepare_patient_record, validate_patient_data

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "risk_model.joblib"


def train_risk_model(
    data: pd.DataFrame, *, random_state: int = 42
) -> tuple[RandomForestClassifier, dict[str, float]]:
    """Fit a classifier on labeled records and report holdout metrics."""
    if "risk_label" not in data.columns:
        raise ValueError("Training data must include a risk_label column.")
    features = validate_patient_data(data)
    labels = pd.to_numeric(data["risk_label"], errors="coerce")
    if labels.isna().any() or not labels.isin((0, 1)).all():
        raise ValueError("risk_label must contain only 0 or 1.")
    if labels.nunique() != 2:
        raise ValueError("Training data must contain both risk_label classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels.astype(int),
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, model.predict(X_test))),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    return model, metrics


def save_risk_model(model: Any, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def load_risk_model(path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Risk model not found at {model_path}. Run `python train_model.py` first."
        )
    return joblib.load(model_path)


def predict_risk(
    record: dict[str, Any] | pd.DataFrame, *, model: Any | None = None
) -> dict[str, float | str]:
    """Predict the positive-class probability and map it to a simple risk band."""
    if isinstance(record, dict):
        features = prepare_patient_record(record)
    else:
        features = validate_patient_data(record)
    predictor = model if model is not None else load_risk_model()
    if not hasattr(predictor, "predict_proba"):
        raise TypeError("Risk model must implement predict_proba.")
    probabilities = predictor.predict_proba(features.loc[:, FEATURE_COLUMNS])
    classes = list(predictor.classes_)
    if 1 not in classes:
        raise ValueError("Risk model does not contain the positive risk class (1).")
    probability = float(probabilities[0, classes.index(1)])
    band = "Low" if probability < 0.35 else "Moderate" if probability < 0.65 else "High"
    return {"risk_probability": probability, "risk_band": band}
