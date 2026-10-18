"""Train SecurePulse models on generated synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from modules.anomaly_detection import save_anomaly_model, train_anomaly_model
from modules.risk_prediction import save_risk_model, train_risk_model

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "synthetic_patient_data.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {args.data}. Run `python generate_data.py` first."
        )

    data = pd.read_csv(args.data)
    risk_model, metrics = train_risk_model(data)
    anomaly_model = train_anomaly_model(data)
    risk_path = save_risk_model(risk_model)
    anomaly_path = save_anomaly_model(anomaly_model)
    print(f"Risk model saved to {risk_path}")
    print(f"Holdout accuracy: {metrics['accuracy']:.3f}")
    print(f"Holdout ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"Anomaly model saved to {anomaly_path}")


if __name__ == "__main__":
    main()
