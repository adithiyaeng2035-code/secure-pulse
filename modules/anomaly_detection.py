"""Outlier detection for patient measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from modules.data_processor import FEATURE_COLUMNS, validate_patient_data

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "anomaly_model.joblib"


def train_anomaly_model(
    data: pd.DataFrame, *, contamination: float = 0.03, random_state: int = 42
) -> IsolationForest:
    features = validate_patient_data(data)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    model.fit(features.loc[:, FEATURE_COLUMNS])
    return model


def save_anomaly_model(
    model: Any, path: str | Path = DEFAULT_MODEL_PATH
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def load_anomaly_model(path: str | Path = DEFAULT_MODEL_PATH) -> Any:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Anomaly model not found at {model_path}. Run `python train_model.py` first."
        )
    return joblib.load(model_path)


def detect_anomaly(
    record: dict[str, Any] | pd.DataFrame, *, model: Any | None = None
) -> dict[str, bool | float]:
    features = (
        validate_patient_data(pd.DataFrame([record]))
        if isinstance(record, dict)
        else validate_patient_data(record)
    )
    detector = model if model is not None else load_anomaly_model()
    prediction = int(detector.predict(features.loc[:, FEATURE_COLUMNS])[0])
    score = float(detector.decision_function(features.loc[:, FEATURE_COLUMNS])[0])
    return {"is_anomaly": prediction == -1, "anomaly_score": score}
