"""SecurePulse synthetic patient risk dashboard."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from ipaddress import ip_address
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.anomaly_detection import detect_anomaly, load_anomaly_model
from modules.data_processor import (
    FEATURE_COLUMNS,
    DataValidationError,
    prepare_patient_record,
)
from modules.risk_prediction import load_risk_model, predict_risk
from modules.synthetic_audit import (
    AuditChainIntegrityError,
    append_synthetic_prediction_event,
)

st.set_page_config(page_title="SecurePulse", page_icon="🫀", layout="wide")

STYLESHEET_PATH = Path(__file__).resolve().parent / "assets" / "securepulse.css"
DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_patient_data.csv"


def is_loopback_bind_address(address: str | None) -> bool:
    if not address:
        return False
    try:
        return ip_address(address).is_loopback
    except ValueError:
        return False


def load_stylesheet() -> None:
    stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


def clear_demo_name() -> None:
    st.session_state["live_demo_name"] = ""


@st.cache_resource
def get_models():
    return load_risk_model(), load_anomaly_model()


def main() -> None:
    bind_address = st.get_option("server.address")
    if not is_loopback_bind_address(bind_address):
        st.error(
            "SecurePulse is configured for a non-local network address and will not "
            "start. Set `server.address` to a loopback IP such as `127.0.0.1`."
        )
        st.stop()

    load_stylesheet()
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    st.title("SecurePulse")
    st.subheader("Cardiometabolic risk screening demo")
    st.warning(
        "This educational demo uses synthetic data. It is not a diagnosis or a substitute "
        "for advice from a qualified health professional. Do not enter real patient data. "
        "Local demo only; it has no authentication or production-grade data protection."
    )

    try:
        risk_model, anomaly_model = get_models()
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"Models could not be loaded: {error}")
        st.code("python generate_data.py\npython train_model.py\nstreamlit run app.py")
        return

    patients_tab, live_tab, history_tab = st.tabs(
        ["Synthetic Patients", "Live", "History"]
    )

    with patients_tab:
        st.subheader("Synthetic patient profiles")
        st.caption(
            "These rows are generated demo data, not real people. Only the first "
            "20 records are shown."
        )
        if not DATA_PATH.is_file():
            st.error(f"Synthetic dataset not found at {DATA_PATH}. Run `python generate_data.py`.")
        else:
            try:
                patient_data = pd.read_csv(DATA_PATH, usecols=["patient_id", *FEATURE_COLUMNS])
            except (OSError, ValueError, pd.errors.ParserError) as error:
                st.error(f"Could not load the synthetic patient sample: {error}")
            else:
                patient_data = patient_data.head(20).rename(
                    columns={"patient_id": "Synthetic demo ID"}
                )
                st.dataframe(patient_data, use_container_width=True, hide_index=True)

    with live_tab:
        st.subheader("Live synthetic screening")
        st.markdown(
            "**New reading → AI risk estimate → anomaly detection → "
            "privacy-aware result → dashboard update.**"
        )
        with st.form("patient_screening"):
            st.markdown("#### Enter a synthetic screening profile")
            st.text_input(
                "Fictional demo patient name (optional)",
                key="live_demo_name",
                max_chars=40,
                help="Use a made-up name only. It is cleared on submission and excluded from results, history, and audit.",
            )
            left, middle, right = st.columns(3)
            with left:
                age = st.number_input(
                    "Age", min_value=18, max_value=110, value=45, key="live_age"
                )
                systolic_bp = st.number_input(
                    "Systolic blood pressure",
                    min_value=70,
                    max_value=250,
                    value=120,
                    key="live_systolic_bp",
                )
                diastolic_bp = st.number_input(
                    "Diastolic blood pressure",
                    min_value=40,
                    max_value=150,
                    value=80,
                    key="live_diastolic_bp",
                )
            with middle:
                cholesterol = st.number_input(
                    "Total cholesterol (mg/dL)",
                    min_value=80,
                    max_value=500,
                    value=190,
                    key="live_cholesterol",
                )
                glucose = st.number_input(
                    "Fasting glucose (mg/dL)",
                    min_value=40,
                    max_value=500,
                    value=95,
                    key="live_glucose",
                )
                bmi = st.number_input(
                    "BMI",
                    min_value=10.0,
                    max_value=80.0,
                    value=25.0,
                    step=0.1,
                    key="live_bmi",
                )
            with right:
                smoking = st.selectbox(
                    "Current smoking", ["No", "Yes"], key="live_smoking"
                )
                family_history = st.selectbox(
                    "Family history of cardiovascular disease",
                    ["No", "Yes"],
                    key="live_family_history",
                )

            submitted = st.form_submit_button(
                "Estimate risk", on_click=clear_demo_name
            )

        if submitted:
            record = {
                "age": age,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "cholesterol": cholesterol,
                "glucose": glucose,
                "bmi": bmi,
                "smoking": int(smoking == "Yes"),
                "family_history": int(family_history == "Yes"),
            }
            if diastolic_bp >= systolic_bp:
                st.error("Diastolic blood pressure must be lower than systolic blood pressure.")
            else:
                with st.status("Processing synthetic reading", expanded=True) as pipeline:
                    st.write("1. Validate the submitted measurements.")
                    try:
                        features = prepare_patient_record(record)
                    except DataValidationError as error:
                        pipeline.update(label="Reading validation failed", state="error")
                        st.error(f"Please correct the screening values: {error}")
                    else:
                        st.write("2. Calculate the synthetic-model risk estimate.")
                        risk = predict_risk(features, model=risk_model)

                        st.write("3. Check for patterns unusual in synthetic training data.")
                        anomaly = detect_anomaly(features, model=anomaly_model)

                        st.write("4. Record a privacy-minimized local demo audit event.")
                        audit_event = None
                        audit_error = None
                        try:
                            audit_event = append_synthetic_prediction_event()
                        except (AuditChainIntegrityError, OSError, sqlite3.Error) as error:
                            audit_error = str(error)

                        st.write("5. Refresh the session dashboard with prediction outputs only.")
                        history = st.session_state.prediction_history
                        history.append(
                            {
                                "submitted_at": datetime.now(timezone.utc).isoformat(
                                    timespec="seconds"
                                ),
                                "risk_band": risk["risk_band"],
                                "risk_probability": risk["risk_probability"],
                                "is_anomaly": anomaly["is_anomaly"],
                                "anomaly_score": anomaly["anomaly_score"],
                                "audit_sequence": (
                                    audit_event.sequence if audit_event else None
                                ),
                                "audit_error": audit_error,
                            }
                        )
                        pipeline.update(
                            label="Processing complete", state="complete", expanded=True
                        )

        st.divider()
        st.subheader("Live dashboard")
        history = st.session_state.prediction_history
        if not history:
            st.info("Submit a synthetic reading to run the processing pipeline.")
        else:
            latest_result = history[-1]
            st.metric("Synthetic submissions this session", str(len(history)))
            if latest_result["audit_error"]:
                st.error(
                    "Predictions completed, but the local synthetic audit chain could "
                    f"not be verified or updated: {latest_result['audit_error']}"
                )
            else:
                st.success(
                    f"Local synthetic audit event recorded (entry "
                    f"{latest_result['audit_sequence']}). No readings or prediction "
                    "values are included."
                )

            risk_col, anomaly_col = st.columns(2)
            risk_col.metric("Model risk band", str(latest_result["risk_band"]))
            risk_col.metric(
                "Synthetic-model estimate",
                f"{latest_result['risk_probability']:.1%}",
            )
            risk_col.caption(
                "This is the model's estimate for a positive label in synthetic data, "
                "not a real-world probability of disease."
            )
            if latest_result["is_anomaly"]:
                anomaly_col.warning(
                    "This profile is unusual relative to the synthetic training data."
                )
            else:
                anomaly_col.info(
                    "This profile was not flagged as unusual compared with the synthetic training data."
                )
            anomaly_col.caption(
                "An unusual-pattern flag means the measurements differ from patterns the "
                "model learned; it does not indicate a diagnosis."
            )
            st.caption(
                "These outputs come from models trained on synthetic data and are not clinically validated."
            )
            with st.expander("How to understand these results"):
                st.markdown(
                    """
                    - **Low** means the synthetic-model estimate is below 35%.
                    - **Moderate** means it is from 35% up to (but not including) 65%.
                    - **High** means it is 65% or higher.
                    - Those cutoffs are software display bands, not medical guideline thresholds.
                    - The anomaly check compares this profile with synthetic training examples.
                      A flag is not a diagnosis, and no flag does not mean a person is healthy.
                    - The model cannot explain the medical cause of a result or recommend treatment.
                      These estimates are not validated for real people.
                    """
                )

    with history_tab:
        st.subheader("Session history")
        history = st.session_state.prediction_history
        if not history:
            st.info("No synthetic prediction results in this browser session yet.")
        else:
            history_data = pd.DataFrame(
                [
                    {
                        "Submission": index,
                        "Submitted (UTC)": result["submitted_at"],
                        "Risk band": result["risk_band"],
                        "Synthetic estimate": result["risk_probability"],
                        "Anomaly flagged": result["is_anomaly"],
                    }
                    for index, result in enumerate(history, start=1)
                ]
            )
            st.dataframe(history_data, use_container_width=True, hide_index=True)
            if len(history_data) > 1:
                st.line_chart(
                    history_data,
                    x="Submission",
                    y="Synthetic estimate",
                    height=240,
                )
            else:
                st.info("Submit another synthetic reading to see the estimate trend.")
            st.caption(
                "History contains prediction summaries only and stays in memory for this "
                "browser session. Streamlit retains active form-widget values in session "
                "memory while the session is open; raw readings are not copied into history "
                "or the audit chain."
            )

    with st.expander("Privacy notes"):
            st.write(
                "The included dataset and all profiles in the Patients tab are synthetic. "
                "The optional fictional demo name is cleared when the form is submitted and "
                "is not used in predictions or copied to results, history, or audit. Streamlit "
                "holds active form-widget values in session memory while the session is open. "
                "Submitted measurements are used for the current calculation "
                "but are not copied into session history or the audit chain. Session history "
                "contains prediction summaries only. A local audit chain stores only the submission "
            "timestamp and a fixed demo event type. The form does not collect patient "
            "identifiers. This demo is not designed, certified, or configured for "
            "protected health information."
        )


if __name__ == "__main__":
    main()
