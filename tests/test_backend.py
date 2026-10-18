import asyncio
import sqlite3

import httpx

import modules.api as api
import modules.chatbot as chatbot
import modules.synthetic_audit as synthetic_audit

SYNTHETIC_READING = {
    "age": 45,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "cholesterol": 190,
    "glucose": 95,
    "bmi": 25.0,
    "smoking": 0,
    "family_history": 0,
}


def api_request(method, path, *, json=None):
    async def send_request():
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://securepulse.test",
        ) as client:
            return await client.request(method, path, json=json)

    return asyncio.run(send_request())


def test_health_reports_working_prediction_engine():
    response = api_request("GET", "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "engine": "synthetic-risk-api",
    }


def test_prediction_runs_models_and_audits_without_storing_reading(
    monkeypatch, tmp_path
):
    audit_path = tmp_path / "audit.sqlite3"
    monkeypatch.setattr(synthetic_audit, "DEFAULT_LEDGER_PATH", audit_path)
    api.get_models.cache_clear()

    response = api_request("POST", "/predict", json=SYNTHETIC_READING)

    assert response.status_code == 200
    prediction = response.json()
    assert prediction["risk_band"] in {"Low", "Moderate", "High"}
    assert 0 <= prediction["risk_probability"] <= 1
    assert isinstance(prediction["is_anomaly"], bool)
    assert isinstance(prediction["anomaly_score"], float)
    assert prediction["audit_sequence"] == 1
    assert prediction["audit_error"] is None
    assert "age" not in prediction
    with sqlite3.connect(audit_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(audit_events)")
        }
    assert "age" not in columns


def test_prediction_rejects_invalid_or_extra_fields_without_audit(
    monkeypatch, tmp_path
):
    audit_path = tmp_path / "audit.sqlite3"
    monkeypatch.setattr(synthetic_audit, "DEFAULT_LEDGER_PATH", audit_path)
    invalid_reading = {
        **SYNTHETIC_READING,
        "diastolic_bp": 120,
        "patient_name": "Not accepted",
    }

    response = api_request("POST", "/predict", json=invalid_reading)

    assert response.status_code == 422
    assert not audit_path.exists()


def test_chat_endpoint_returns_local_model_answer(monkeypatch):
    monkeypatch.setattr(api, "ask_ollama", lambda query: f"Answer: {query}")
    monkeypatch.setattr(
        api, "get_ollama_config", lambda: ("http://127.0.0.1:11434", "llama3.2:3b")
    )

    response = api_request("POST", "/chat", json={"query": "What is SecurePulse?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Answer: What is SecurePulse?",
        "model": "llama3.2:3b",
    }


def test_chat_endpoint_surfaces_missing_ollama(monkeypatch):
    monkeypatch.setattr(api, "ask_ollama", chatbot.ask_ollama)
    monkeypatch.setattr(
        chatbot,
        "check_ollama",
        lambda: {
            "available": False,
            "model": "llama3.2:3b",
            "detail": "Ollama not running",
        },
    )

    response = api_request("POST", "/chat", json={"query": "What is SecurePulse?"})

    assert response.status_code == 503
    assert "Ollama not running" in response.json()["detail"]
