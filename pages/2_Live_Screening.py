"""Run a synthetic screening through the configured FastAPI backend."""

import httpx
import streamlit as st

from app import check_backend, configure_page, request_prediction


def clear_demo_name() -> None:
    st.session_state["live_demo_name"] = ""


configure_page("Live Screening | SecurePulse")
st.title("Live Synthetic Screening")
backend_error = check_backend()
if backend_error:
    st.error(
        f"Prediction backend disconnected: {backend_error}. Start it with "
        "`python -m modules.api`."
    )
else:
    st.success("Prediction backend connected — synthetic engine is ready.")
st.caption(
    "Enter a fictional profile only. Readings are sent to the configured prediction "
    "API for this request and are not saved in session history. This demo uses "
    "synthetic profiles and does not collect patient identifiers."
)

with st.form("patient_screening"):
    st.text_input(
        "Fictional demo name (optional)",
        key="live_demo_name",
        max_chars=40,
        help="Use a made-up name only. It is cleared on submission and never sent to the API.",
    )
    left, middle, right = st.columns(3)
    with left:
        age = st.number_input("Age", min_value=18, max_value=110, value=45)
        systolic_bp = st.number_input(
            "Systolic blood pressure", min_value=70, max_value=250, value=120
        )
        diastolic_bp = st.number_input(
            "Diastolic blood pressure", min_value=40, max_value=150, value=80
        )
    with middle:
        cholesterol = st.number_input(
            "Total cholesterol (mg/dL)", min_value=80, max_value=500, value=190
        )
        glucose = st.number_input(
            "Fasting glucose (mg/dL)", min_value=40, max_value=500, value=95
        )
        bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, step=0.1)
    with right:
        smoking = st.selectbox("Current smoking", ["No", "Yes"])
        family_history = st.selectbox(
            "Family history of cardiovascular disease", ["No", "Yes"]
        )
    submitted = st.form_submit_button("Estimate risk", on_click=clear_demo_name)

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
        try:
            result = request_prediction(record)
        except httpx.HTTPStatusError as error:
            st.error(f"Prediction backend error: {error.response.text}")
        except httpx.HTTPError as error:
            st.error(f"Could not reach prediction backend: {error}")
        except ValueError as error:
            st.error(f"Backend configuration error: {error}")
        else:
            st.session_state.prediction_history.append(result)
            if result["audit_error"]:
                st.error(f"Local demo audit issue: {result['audit_error']}")
            else:
                st.success(
                    "Prediction completed. Local synthetic audit event recorded "
                    f"(entry {result['audit_sequence']})."
                )

if st.session_state.prediction_history:
    latest = st.session_state.prediction_history[-1]
    st.divider()
    st.subheader("Latest result")
    count_col, risk_col, anomaly_col = st.columns(3)
    count_col.metric("Submissions this session", str(len(st.session_state.prediction_history)))
    risk_col.metric("Model risk band", str(latest["risk_band"]))
    risk_col.metric("Synthetic-model estimate", f"{latest['risk_probability']:.1%}")
    if latest["is_anomaly"]:
        anomaly_col.warning(
            "This profile is unusual relative to the synthetic training data."
        )
    else:
        anomaly_col.info("This profile was not flagged as unusual.")
    st.caption(
        "The estimate and anomaly flag are based on synthetic training data. "
        "They are not diagnoses or real-world disease probabilities."
    )
