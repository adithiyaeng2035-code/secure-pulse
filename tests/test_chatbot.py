from unittest.mock import Mock

import httpx
import pytest

import modules.chatbot as chatbot


def test_ollama_configuration_is_loopback_only(monkeypatch):
    monkeypatch.delenv("SECUREPULSE_OLLAMA_URL", raising=False)
    monkeypatch.delenv("SECUREPULSE_OLLAMA_MODEL", raising=False)

    assert chatbot.get_ollama_config() == (
        "http://127.0.0.1:11434",
        "llama3.2:3b",
    )

    monkeypatch.setenv("SECUREPULSE_OLLAMA_URL", "http://ollama.example")
    with pytest.raises(ValueError, match="loopback"):
        chatbot.get_ollama_config()


def test_check_ollama_reports_unavailable_service(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        Mock(side_effect=httpx.ConnectError("Ollama not running")),
    )

    status = chatbot.check_ollama()

    assert status["available"] is False
    assert "Ollama not running" in status["detail"]


def test_ask_ollama_sends_only_scoped_query_to_local_model(monkeypatch):
    tags_response = httpx.Response(
        200,
        json={"models": [{"name": "llama3.2:3b"}]},
        request=httpx.Request("GET", "http://127.0.0.1:11434/api/tags"),
    )
    chat_response = httpx.Response(
        200,
        json={"message": {"content": "SecurePulse uses synthetic data."}},
        request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
    )
    post = Mock(return_value=chat_response)
    monkeypatch.setattr(httpx, "get", Mock(return_value=tags_response))
    monkeypatch.setattr(httpx, "post", post)

    answer = chatbot.ask_ollama("What data does SecurePulse use?")

    assert answer == "SecurePulse uses synthetic data."
    request_body = post.call_args.kwargs["json"]
    assert request_body["messages"][-1] == {
        "role": "user",
        "content": "What data does SecurePulse use?",
    }
    assert request_body["messages"][0]["role"] == "system"
    assert "never diagnose" in request_body["messages"][0]["content"].lower()
    assert post.call_args.kwargs["trust_env"] is False


def test_ask_ollama_rejects_empty_or_oversized_query():
    with pytest.raises(ValueError, match="Enter a question"):
        chatbot.ask_ollama(" ")

    with pytest.raises(ValueError, match="2000 characters"):
        chatbot.ask_ollama("x" * 2001)
