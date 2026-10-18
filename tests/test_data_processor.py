import pandas as pd
import pytest

from modules.data_processor import DataValidationError, FEATURE_COLUMNS, validate_patient_data


def valid_record():
    return {
        "age": 45,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "cholesterol": 190,
        "glucose": 95,
        "bmi": 25.0,
        "smoking": 0,
        "family_history": 1,
    }


def test_validate_patient_data_returns_ordered_numeric_features():
    data = pd.DataFrame([{**valid_record(), "patient_id": "SYN-00001"}])

    result = validate_patient_data(data)

    assert result.columns.tolist() == FEATURE_COLUMNS
    assert result.iloc[0]["age"] == 45.0


def test_validate_patient_data_rejects_missing_features():
    with pytest.raises(DataValidationError, match="Missing required features"):
        validate_patient_data(pd.DataFrame([{"age": 45}]))


@pytest.mark.parametrize(
    ("field", "value"),
    [("age", 17), ("smoking", 2), ("bmi", 100), ("diastolic_bp", 130)],
)
def test_validate_patient_data_rejects_invalid_values(field, value):
    record = valid_record()
    record[field] = value

    with pytest.raises(DataValidationError):
        validate_patient_data(pd.DataFrame([record]))


def test_validate_patient_data_rejects_missing_values():
    record = valid_record()
    record["glucose"] = None

    with pytest.raises(DataValidationError, match="non-missing"):
        validate_patient_data(pd.DataFrame([record]))
