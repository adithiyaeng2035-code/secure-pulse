import pandas as pd
import pytest

from generate_data import generate_synthetic_data
from modules.risk_prediction import predict_risk, train_risk_model


@pytest.fixture(scope="module")
def trained_model():
    data = generate_synthetic_data(600)
    model, metrics = train_risk_model(data)
    return model, metrics


def test_training_produces_metrics_and_two_classes(trained_model):
    model, metrics = trained_model

    assert set(model.classes_) == {0, 1}
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1


def test_predict_risk_returns_probability_and_band(trained_model):
    model, _ = trained_model
    record = {
        "age": 50,
        "systolic_bp": 130,
        "diastolic_bp": 82,
        "cholesterol": 205,
        "glucose": 100,
        "bmi": 27.5,
        "smoking": 0,
        "family_history": 1,
    }

    result = predict_risk(record, model=model)

    assert 0 <= result["risk_probability"] <= 1
    assert result["risk_band"] in {"Low", "Moderate", "High"}


def test_training_requires_both_target_classes():
    data = generate_synthetic_data(200)
    data["risk_label"] = 0

    with pytest.raises(ValueError, match="both risk_label classes"):
        train_risk_model(data)
