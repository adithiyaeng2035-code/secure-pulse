"""Validation and preparation for SecurePulse patient records."""

from __future__ import annotations

from typing import Any

import pandas as pd

FEATURE_COLUMNS = [
    "age",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol",
    "glucose",
    "bmi",
    "smoking",
    "family_history",
]

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 110),
    "systolic_bp": (70, 250),
    "diastolic_bp": (40, 150),
    "cholesterol": (80, 500),
    "glucose": (40, 500),
    "bmi": (10, 80),
    "smoking": (0, 1),
    "family_history": (0, 1),
}


class DataValidationError(ValueError):
    """Raised when a patient record is missing or contains invalid data."""


def validate_patient_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate required numeric features and return them in model column order."""
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("Patient data must be provided as a pandas DataFrame.")
    if data.empty:
        raise DataValidationError("At least one patient record is required.")

    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise DataValidationError(f"Missing required features: {', '.join(missing)}.")

    features = data.loc[:, FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    if features.isna().any().any():
        invalid_columns = features.columns[features.isna().any()].tolist()
        raise DataValidationError(
            f"Features must contain valid, non-missing numbers: {', '.join(invalid_columns)}."
        )

    for column, (minimum, maximum) in FEATURE_RANGES.items():
        invalid = ~features[column].between(minimum, maximum)
        if invalid.any():
            raise DataValidationError(
                f"{column} must be between {minimum:g} and {maximum:g}."
            )
        if column in ("smoking", "family_history") and not features[column].isin((0, 1)).all():
            raise DataValidationError(f"{column} must be 0 (no) or 1 (yes).")

    if (features["diastolic_bp"] >= features["systolic_bp"]).any():
        raise DataValidationError("Diastolic blood pressure must be lower than systolic blood pressure.")

    return features.astype(float)


def prepare_patient_record(record: dict[str, Any]) -> pd.DataFrame:
    """Validate one mapping and return a single-row model input."""
    return validate_patient_data(pd.DataFrame([record]))
