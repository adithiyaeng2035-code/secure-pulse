from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import (
    REMOTE_CHAT_DISABLED_MESSAGE,
    check_chatbot,
    get_api_url,
    get_chat_api_url,
    is_bind_address_allowed,
    is_loopback_bind_address,
    request_chat_answer,
)

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def mock_backend(monkeypatch, *, audit_error=None):
    request = httpx.Request("GET", "http://127.0.0.1")
    health_response = httpx.Response(
        200,
        json={"status": "ok", "engine": "synthetic-risk-api"},
        request=request,
    )
    chat_health_response = httpx.Response(
        200,
        json={
            "available": False,
            "model": "llama3.2:3b",
            "detail": "Ollama is not running at http://127.0.0.1:11434.",
        },
        request=httpx.Request("GET", "http://127.0.0.1/chat/health"),
    )
    sequences = iter(range(1, 100))

    def post_prediction(url, *, json, **kwargs):
        is_unusual = json["age"] >= 100
        return httpx.Response(
            200,
            json={
                "submitted_at": "2026-10-18T00:00:00+00:00",
                "risk_band": "High" if is_unusual else "Low",
                "risk_probability": 0.9 if is_unusual else 0.2,
                "is_anomaly": is_unusual,
                "anomaly_score": -0.5 if is_unusual else 0.1,
                "audit_sequence": next(sequences),
                "audit_error": audit_error,
            },
            request=httpx.Request("POST", url),
        )

    def get_status(url, **kwargs):
        return chat_health_response if url.endswith("/chat/health") else health_response

    monkeypatch.setattr(httpx, "get", Mock(side_effect=get_status))
    monkeypatch.setattr(httpx, "post", Mock(side_effect=post_prediction))


def run_app(monkeypatch, tmp_path):
    mock_backend(monkeypatch)
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def test_loopback_bind_address_check():
    assert is_loopback_bind_address("127.0.0.1")
    assert is_loopback_bind_address("::1")
    assert not is_loopback_bind_address("0.0.0.0")
    assert not is_loopback_bind_address("")
    assert not is_loopback_bind_address(None)
    assert not is_loopback_bind_address("securepulse.example")
    assert is_bind_address_allowed("127.0.0.1")
    assert not is_bind_address_allowed("0.0.0.0")
    assert not is_bind_address_allowed("0.0.0.0", allow_remote=False)
    assert is_bind_address_allowed("0.0.0.0", allow_remote=True)


def test_remote_backend_requires_https(monkeypatch):
    monkeypatch.setenv("SECUREPULSE_API_URL", "http://api.example.com")
    try:
        get_api_url()
    except ValueError as error:
        assert "HTTPS" in str(error)
    else:
        raise AssertionError("Remote HTTP API URLs must be rejected.")

    monkeypatch.setenv("SECUREPULSE_API_URL", "https://api.example.com")
    assert get_api_url() == "https://api.example.com"


def test_chatbot_is_disabled_for_remote_backend(monkeypatch):
    monkeypatch.setenv("SECUREPULSE_API_URL", "https://api.example.com")

    with pytest.raises(ValueError, match="remote prediction APIs"):
        get_chat_api_url()

    status = check_chatbot()

    assert status["available"] is False
    assert status["detail"] == REMOTE_CHAT_DISABLED_MESSAGE


def test_chat_request_does_not_send_to_remote_backend(monkeypatch):
    monkeypatch.setenv("SECUREPULSE_API_URL", "https://api.example.com")
    post = Mock()
    monkeypatch.setattr(httpx, "post", post)

    with pytest.raises(ValueError, match="remote prediction APIs"):
        request_chat_answer("What is SecurePulse?")

    post.assert_not_called()


def test_app_stops_when_configured_for_non_loopback(monkeypatch, tmp_path):
    original_get_option = st.get_option
    monkeypatch.setattr(
        st,
        "get_option",
        lambda name: (
            "0.0.0.0" if name == "server.address" else original_get_option(name)
        ),
    )

    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not app.exception
    assert any(
        "non-local network address" in element.value for element in app.error
    )
    assert not app.button


def test_screening_submission_returns_results_and_logs_demo_event(
    monkeypatch, tmp_path
):
    app = run_app(monkeypatch, tmp_path)

    app.button[0].click().run()

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert set(metrics) == {
        "Synthetic submissions this session",
        "Model risk band",
        "Synthetic-model estimate",
    }
    assert metrics["Synthetic submissions this session"] == "1"
    assert metrics["Model risk band"] in {"Low", "Moderate", "High"}
    assert 0 <= float(metrics["Synthetic-model estimate"].rstrip("%")) <= 100
    assert any(
        "not flagged as unusual" in element.value for element in app.info
    )
    assert any(
        "Local synthetic audit event recorded (entry 1)" in element.value
        for element in app.success
    )
    assert any("backend connected" in element.value.lower() for element in app.success)


def test_unusual_reading_updates_results_and_logs_second_demo_event(
    monkeypatch, tmp_path
):
    app = run_app(monkeypatch, tmp_path)
    app.button[0].click().run()
    initial_metrics = [(metric.label, metric.value) for metric in app.metric]

    for index, value in {
        0: 109,
        1: 240,
        2: 140,
        3: 490,
        4: 490,
        5: 79.0,
    }.items():
        app.number_input[index].set_value(value)
    app.selectbox[0].set_value("Yes")
    app.selectbox[1].set_value("Yes")
    app.button[0].click().run()

    updated_metrics = [(metric.label, metric.value) for metric in app.metric]
    assert not app.exception
    assert len(updated_metrics) == 3
    assert updated_metrics != initial_metrics
    assert any(
        "unusual relative to the synthetic training data" in element.value
        for element in app.warning
    )
    assert any(
        "Local synthetic audit event recorded (entry 2)" in element.value
        for element in app.success
    )


def test_app_does_not_collect_patient_identifiers_or_display_secrets(
    monkeypatch, tmp_path
):
    app = run_app(monkeypatch, tmp_path)

    input_labels = [
        widget.label.lower()
        for widget in [*app.number_input, *app.selectbox]
    ]
    assert not any(
        identifier in label
        for label in input_labels
        for identifier in ("patient id", "name", "email", "phone", "ssn")
    )
    rendered_markdown = " ".join(element.value for element in app.markdown)
    assert "does not collect patient identifiers" in rendered_markdown
    assert "SYN-" not in rendered_markdown
    assert not app.exception


def test_synthetic_patients_page_shows_dataset_size(monkeypatch, tmp_path):
    app = run_app(monkeypatch, tmp_path)

    assert any(
        "Dataset contains 50,000 synthetic profiles" in item.value
        for item in app.caption
    )


def test_chatbot_page_explains_ollama_is_not_running(monkeypatch, tmp_path):
    app = run_app(monkeypatch, tmp_path)

    assert any(
        "Local AI chatbot unavailable" in item.value for item in app.warning
    )
    assert any("Install Ollama" in item.value for item in app.info)


def test_chatbot_page_explains_remote_api_restriction(monkeypatch, tmp_path):
    monkeypatch.setenv("SECUREPULSE_API_URL", "https://api.example.com")
    app = run_app(monkeypatch, tmp_path)

    assert any(
        REMOTE_CHAT_DISABLED_MESSAGE in item.value for item in app.warning
    )
    assert not any("Install Ollama" in item.value for item in app.info)


def test_app_surfaces_audit_chain_corruption(monkeypatch, tmp_path):
    mock_backend(monkeypatch, audit_error="Local demo audit chain integrity check failed.")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.button[0].click().run()

    assert len(app.metric) == 3
    assert any(
        "audit chain integrity check failed" in element.value.lower()
        for element in app.error
    )


def test_session_history_contains_prediction_summaries_only(monkeypatch, tmp_path):
    app = run_app(monkeypatch, tmp_path)
    app.button[0].click().run()

    history = app.session_state["prediction_history"]

    assert len(history) == 1
    assert set(history[0]) == {
        "submitted_at",
        "risk_band",
        "risk_probability",
        "is_anomaly",
        "anomaly_score",
        "audit_sequence",
        "audit_error",
    }
    assert not {"age", "systolic_bp", "diastolic_bp", "cholesterol", "glucose", "bmi"} & set(
        history[0]
    )


def test_demo_name_is_cleared_and_not_saved_in_history_or_audit(monkeypatch, tmp_path):
    app = run_app(monkeypatch, tmp_path)
    app.text_input[0].set_value("Fictional Demo Person")
    app.button[0].click().run()

    assert not app.exception
    assert app.session_state["live_demo_name"] == ""
    assert "Fictional Demo Person" not in " ".join(
        element.value for element in app.markdown
    )
    history = app.session_state["prediction_history"]
    assert len(history) == 1
    assert "name" not in history[0]

    posted_payload = httpx.post.call_args.kwargs["json"]
    assert "Fictional Demo Person" not in str(posted_payload)
    assert "name" not in posted_payload


def test_backend_disconnect_is_reported_and_does_not_add_history(monkeypatch, tmp_path):
    def fail_health(*args, **kwargs):
        raise httpx.ConnectError("backend is offline")

    monkeypatch.setattr(httpx, "get", fail_health)
    monkeypatch.setattr(httpx, "post", fail_health)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert any("backend disconnected" in item.value.lower() for item in app.error)
    app.button[0].click().run()
    assert any("could not reach prediction backend" in item.value.lower() for item in app.error)
    assert app.session_state["prediction_history"] == []
