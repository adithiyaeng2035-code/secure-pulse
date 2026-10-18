"""Generate reproducible synthetic patient measurements for SecurePulse."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from modules.data_processor import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "synthetic_patient_data.csv"


def generate_synthetic_data(
    n_samples: int = 10_000, *, random_state: int = 42
) -> pd.DataFrame:
    """Create synthetic measurements and a probabilistic risk label."""
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100 for a useful synthetic dataset.")

    rng = np.random.default_rng(random_state)
    age = rng.integers(18, 91, size=n_samples)
    systolic = np.clip(rng.normal(118 + (age - 45) * 0.35, 16), 85, 210).round()
    diastolic = np.minimum(
        np.clip(rng.normal(76 + (age - 45) * 0.12, 10), 50, 125),
        systolic - 5,
    ).round()
    cholesterol = np.clip(rng.normal(185 + (age - 45) * 0.5, 32), 100, 350).round()
    glucose = np.clip(rng.normal(92 + (age - 45) * 0.25, 18), 55, 220).round()
    bmi = np.clip(rng.normal(26, 5, size=n_samples), 16, 48).round(1)
    smoking = rng.binomial(1, 0.2, size=n_samples)
    family_history = rng.binomial(1, 0.28, size=n_samples)

    risk_logit = (
        -4.2
        + (age - 45) * 0.035
        + (systolic - 120) * 0.018
        + (cholesterol - 190) * 0.009
        + (glucose - 95) * 0.012
        + (bmi - 25) * 0.045
        + smoking * 0.75
        + family_history * 0.55
    )
    risk_probability = 1 / (1 + np.exp(-risk_logit))
    labels = rng.binomial(1, risk_probability)

    frame = pd.DataFrame(
        {
            "patient_id": [f"SYN-{i:05d}" for i in range(1, n_samples + 1)],
            "age": age,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "cholesterol": cholesterol,
            "glucose": glucose,
            "bmi": bmi,
            "smoking": smoking,
            "family_history": family_history,
            "risk_label": labels,
        }
    )
    return frame[["patient_id", *FEATURE_COLUMNS, "risk_label"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dataset = generate_synthetic_data(args.samples, random_state=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset)} synthetic records to {args.output}")
    print(f"Synthetic positive-label rate: {dataset['risk_label'].mean():.1%}")


if __name__ == "__main__":
    main()
