"""SecurePulse synthetic patient risk dashboard."""

from __future__ import annotations

import os
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import streamlit as st

from modules.chatbot import MAX_QUERY_LENGTH

ROOT = Path(__file__).resolve().parent
STYLESHEET_PATH = ROOT / "assets" / "securepulse.css"
DATA_PATH = ROOT / "data" / "synthetic_patient_data.csv"
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
    api_url = os.environ.get("SECUREPULSE_API_URL", "http://127.0.0.1:8000").rstrip("/")
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


def get_api_headers() -> dict[str, str]:
    api_url = get_api_url()
    hostname = urlsplit(api_url).hostname or ""
    is_remote = hostname != "localhost" and not is_loopback_bind_address(hostname)
    token = os.environ.get("SECUREPULSE_API_TOKEN", "")
    if is_remote and not token:
        raise ValueError(
            "SECUREPULSE_API_TOKEN must be configured for remote prediction APIs."
        )
    return {"Authorization": f"Bearer {token}"} if token else {}


def load_stylesheet() -> None:
    stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


def ensure_session_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, page_icon="🫀", layout="wide")
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
    ensure_session_state()


def check_backend() -> str | None:
    try:
        api_url = get_api_url()
        get_api_headers()
        response = httpx.get(f"{api_url}/health", timeout=2)
        response.raise_for_status()
        if response.json().get("engine") != "synthetic-risk-api":
            return "The backend responded with an unexpected engine."
    except (httpx.HTTPError, ValueError) as error:
        return str(error)
    return None


def request_prediction(record: dict[str, int | float]) -> dict[str, object]:
    response = httpx.post(
        f"{get_api_url()}/predict",
        json=record,
        headers=get_api_headers(),
        timeout=15,
    )
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
        headers=get_api_headers(),
        timeout=95,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    configure_page("SecurePulse | Overview")
    st.title("SecurePulse")
    st.subheader("Cardiometabolic risk screening demo")
    st.warning(
        "Educational demo using synthetic data only. It is not a diagnosis or a "
        "substitute for advice from a qualified health professional. Do not enter "
        "real patient data."
    )

    backend_error = check_backend()
    if backend_error:
        st.error(
            f"Prediction backend disconnected: {backend_error}. Start it in a "
            "separate terminal with `python -m modules.api`."
        )
    else:
        st.success("Prediction backend connected — synthetic engine is ready.")

    history = st.session_state.prediction_history
    overview_cols = st.columns(3)
    overview_cols[0].metric("Submissions this session", str(len(history)))
    if history:
        latest = history[-1]
        overview_cols[1].metric("Latest model risk band", str(latest["risk_band"]))
        overview_cols[2].metric(
            "Latest synthetic estimate", f"{latest['risk_probability']:.1%}"
        )
    else:
        overview_cols[1].metric("Latest model risk band", "—")
        overview_cols[2].metric("Latest synthetic estimate", "—")

    st.markdown("### Explore SecurePulse")
    st.page_link("pages/1_Synthetic_Patients.py", label="Browse synthetic profiles", icon="🧪")
    st.page_link("pages/2_Live_Screening.py", label="Run a synthetic screening", icon="🩺")
    st.page_link("pages/3_Patient_History.py", label="View session history", icon="📈")
    st.page_link("pages/4_Chatbot.py", label="Ask the SecurePulse assistant", icon="💬")
    st.caption(
        "History is session-only and contains prediction summaries, not raw readings. "
        "The synthetic model is not clinically validated."
    )


if __name__ == "__main__":
    main()
