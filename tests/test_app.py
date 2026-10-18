import sqlite3
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from app import is_loopback_bind_address
import modules.synthetic_audit as synthetic_audit

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def run_app(monkeypatch, tmp_path):
    monkeypatch.setattr(
        synthetic_audit, "DEFAULT_LEDGER_PATH", tmp_path / "demo_audit.sqlite3"
    )
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def test_loopback_bind_address_check():
    assert is_loopback_bind_address("127.0.0.1")
    assert is_loopback_bind_address("::1")
    assert not is_loopback_bind_address("0.0.0.0")
    assert not is_loopback_bind_address("")
    assert not is_loopback_bind_address(None)
    assert not is_loopback_bind_address("securepulse.example")


def test_app_stops_when_configured_for_non_loopback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        synthetic_audit, "DEFAULT_LEDGER_PATH", tmp_path / "demo_audit.sqlite3"
    )
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
    status = synthetic_audit.verify_synthetic_audit_chain()
    assert status.valid
    assert status.event_count == 1


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
    status = synthetic_audit.verify_synthetic_audit_chain()
    assert status.valid
    assert status.event_count == 2


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


def test_app_surfaces_audit_chain_corruption(monkeypatch, tmp_path):
    audit_path = tmp_path / "demo_audit.sqlite3"
    monkeypatch.setattr(synthetic_audit, "DEFAULT_LEDGER_PATH", audit_path)
    synthetic_audit.append_synthetic_prediction_event()

    with sqlite3.connect(audit_path) as connection:
        connection.execute(
            "UPDATE audit_events SET event_hash = ? WHERE sequence = 1",
            ("corrupted",),
        )

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

    audit_path = tmp_path / "demo_audit.sqlite3"
    with sqlite3.connect(audit_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(audit_events)")
        }
        stored_values = str(
            connection.execute(
                "SELECT timestamp, event_type, previous_hash, event_hash "
                "FROM audit_events"
            ).fetchall()
        )
    assert "Fictional Demo Person" not in stored_values
    assert "name" not in columns
