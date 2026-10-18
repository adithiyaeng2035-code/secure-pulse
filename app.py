"""SecurePulse synthetic patient risk dashboard."""

from __future__ import annotations

import os
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pandas as pd
import streamlit as st

from modules.chatbot import DEFAULT_OLLAMA_MODEL, MAX_QUERY_LENGTH
from modules.data_processor import FEATURE_COLUMNS

st.set_page_config(page_title="SecurePulse", page_icon="🫀", layout="wide")

STYLESHEET_PATH = Path(__file__).resolve().parent / "assets" / "securepulse.css"
DATA_PATH = Path(__file__).resolve().parent / "data" / "synthetic_patient_data.csv"
REMOTE_CHAT_DISABLED_MESSAGE = (
    "Chat is disabled for remote prediction APIs. Configure "
    "SECUREPULSE_API_URL to a loopback address to keep chat on this device."
)


def is_loopback_bind_address(address: str | None) -> bool:
    if not address:
        return False
    try:
        return ip_address(address).is_loopback
    except ValueError:
        return False


def is_bind_address_allowed(address: str | None, *, allow_remote: bool = False) -> bool:
    return is_loopback_bind_address(address) or allow_remote


def get_api_url() -> str:
    api_url = os.environ.get(
        "SECUREPULSE_API_URL", "http://127.0.0.1:8000"
    ).rstrip("/")
    parsed_url = urlsplit(api_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("SECUREPULSE_API_URL must be an absolute HTTP or HTTPS URL.")
    if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
        raise ValueError("SECUREPULSE_API_URL cannot contain credentials, query, or fragment.")
    try:
        parsed_url.port
    except ValueError as error:
        raise ValueError("SECUREPULSE_API_URL contains an invalid port.") from error
    hostname = parsed_url.hostname.lower()
    is_loopback = hostname == "localhost" or is_loopback_bind_address(hostname)
    if parsed_url.scheme == "http" and not is_loopback:
        raise ValueError("Remote prediction APIs must use HTTPS.")
    if parsed_url.path not in {"", "/"}:
        raise ValueError("SECUREPULSE_API_URL must point to the API root.")
    return api_url


def get_chat_api_url() -> str:
    api_url = get_api_url()
    hostname = urlsplit(api_url).hostname or ""
    if hostname != "localhost" and not is_loopback_bind_address(hostname):
        raise ValueError(REMOTE_CHAT_DISABLED_MESSAGE)
    return api_url


def load_stylesheet() -> None:
    stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


def clear_demo_name() -> None:
    st.session_state["live_demo_name"] = ""


def check_backend() -> str | None:
    try:
        response = httpx.get(f"{get_api_url()}/health", timeout=2)
        response.raise_for_status()
        if response.json().get("engine") != "synthetic-risk-api":
            return "The backend responded with an unexpected engine."
    except (httpx.HTTPError, ValueError) as error:
        return str(error)
    return None


def request_prediction(record: dict[str, int | float]) -> dict[str, object]:
    response = httpx.post(f"{get_api_url()}/predict", json=record, timeout=15)
    response.raise_for_status()
    return response.json()


def check_chatbot() -> dict[str, str | bool]:
    try:
        response = httpx.get(f"{get_chat_api_url()}/chat/health", timeout=3)
        response.raise_for_status()
        status = response.json()
        if not isinstance(status, dict) or "available" not in status:
            return {
                "available": False,
                "model": "",
                "detail": "The backend returned an invalid chatbot status.",
            }
        return status
    except (httpx.HTTPError, ValueError) as error:
        return {"available": False, "model": "", "detail": str(error)}


def request_chat_answer(query: str) -> dict[str, str]:
    response = httpx.post(
        f"{get_chat_api_url()}/chat",
        json={"query": query},
        timeout=95,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    bind_address = st.get_option("server.address")
    allow_remote = os.environ.get("SECUREPULSE_ALLOW_REMOTE") == "1"
    if not is_bind_address_allowed(bind_address, allow_remote=allow_remote):
        st.error(
            "SecurePulse is configured for a non-local network address and will not "
            "start. Set `server.address` to a loopback IP such as `127.0.0.1`, or "
            "explicitly allow remote binding only behind deployment authentication."
        )
        st.stop()

    load_stylesheet()
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.title("SecurePulse")
    st.subheader("Cardiometabolic risk screening demo")
    st.warning(
        "This educational demo uses synthetic data. It is not a diagnosis or a substitute "
        "for advice from a qualified health professional. Do not enter real patient data. "
        "The app has no built-in user authentication; only use it locally or behind "
        "verified deployment access controls."
    )

    backend_error = check_backend()
    if backend_error:
        st.error(
            f"Prediction backend disconnected: {backend_error}. Start it in a "
            "separate terminal with `python -m modules.api`."
        )
    else:
        st.success("Prediction backend connected — synthetic engine is ready.")

    patients_tab, live_tab, history_tab, chatbot_tab = st.tabs(
        ["Synthetic Patients", "Live", "History", "Chatbot"]
    )

    with patients_tab:
        st.subheader("Synthetic patient profiles")
        if not DATA_PATH.is_file():
            st.error(f"Synthetic dataset not found at {DATA_PATH}. Run `python generate_data.py`.")
        else:
            try:
                patient_data = pd.read_csv(DATA_PATH, usecols=["patient_id", *FEATURE_COLUMNS])
            except (OSError, ValueError, pd.errors.ParserError) as error:
                st.error(f"Could not load the synthetic patient sample: {error}")
            else:
                st.caption(
                    f"Dataset contains {len(patient_data):,} synthetic profiles; "
                    "showing the first 20. These are not real people."
                )
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
                with st.status(
                    "Sending synthetic reading to prediction backend", expanded=True
                ) as pipeline:
                    st.write("1. Send measurements to the configured prediction API.")
                    st.write("2. Validate and process measurements in the backend engine.")
                    st.write("3. Return risk and anomaly outputs; audit stores no readings.")
                    try:
                        result = request_prediction(record)
                    except httpx.HTTPStatusError as error:
                        pipeline.update(label="Backend rejected the request", state="error")
                        st.error(f"Prediction backend error: {error.response.text}")
                    except httpx.HTTPError as error:
                        pipeline.update(label="Backend connection failed", state="error")
                        st.error(f"Could not reach prediction backend: {error}")
                    except ValueError as error:
                        pipeline.update(label="Backend configuration error", state="error")
                        st.error(str(error))
                    else:
                        st.write("4. Update this browser session with prediction outputs only.")
                        st.session_state.prediction_history.append(result)
                        pipeline.update(
                            label="Backend prediction complete",
                            state="complete",
                            expanded=True,
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
                "browser session. Readings are sent to the configured prediction API; "
                "Streamlit retains active widget values in memory while the session is open. "
                "Raw readings are not copied into history or the audit chain."
            )

    with chatbot_tab:
        st.subheader("SecurePulse assistant")
        st.caption(
            "Ask about this project, its synthetic dataset, setup, or demo outputs. "
            "Do not enter patient names, readings, or other personal information. "
            "The assistant cannot provide medical advice. Each question is answered "
            "independently; previous messages are not sent again."
        )
        chat_status = check_chatbot()
        if chat_status["available"]:
            st.success(f"Local Ollama connected — model {chat_status['model']}.")
        else:
            st.warning(
                f"Local AI chatbot unavailable: {chat_status['detail']}"
            )
            if chat_status["detail"] != REMOTE_CHAT_DISABLED_MESSAGE:
                st.info(
                    "Install Ollama, start its local service, and pull the model "
                    "to enable answers."
                )
                st.code(f"ollama pull {DEFAULT_OLLAMA_MODEL}")

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input(
            "Ask about SecurePulse",
            max_chars=MAX_QUERY_LENGTH,
            disabled=not bool(chat_status["available"]),
        )
        if prompt:
            st.session_state.chat_messages.append(
                {"role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Asking the local Ollama model..."):
                    try:
                        answer = request_chat_answer(prompt)
                    except httpx.HTTPStatusError as error:
                        st.error(
                            f"Chat backend error: {error.response.text}"
                        )
                    except httpx.HTTPError as error:
                        st.error(f"Could not reach the chat backend: {error}")
                    else:
                        st.markdown(answer["answer"])
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": answer["answer"]}
                        )
        if st.session_state.chat_messages and st.button(
            "Clear chat", key="clear_chat"
        ):
            st.session_state.chat_messages.clear()
            st.rerun()

    with st.expander("Privacy notes"):
            st.write(
                "The included dataset and all profiles in the Patients tab are synthetic. "
                "The optional fictional demo name is cleared when the form is submitted and "
                "is not used in predictions or copied to results, history, or audit. Streamlit "
                "holds active form-widget values in session memory while the session is open. "
                "Submitted measurements are sent to the configured prediction API "
                "for the current calculation, but are not copied into session history or the "
                "audit chain. Session history "
                "contains prediction summaries only. A local audit chain stores only the submission "
            "timestamp and a fixed demo event type. The form does not collect patient "
            "identifiers. This demo is not designed, certified, or configured for "
            "protected health information."
        )


if __name__ == "__main__":
    main()
