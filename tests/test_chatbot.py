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
    assert request_body["messages"][-1]["role"] == "user"
    assert "Question: What data does SecurePulse use?" in request_body["messages"][-1]["content"]
    assert "synthetic records only" in request_body["messages"][-1]["content"]
    assert request_body["messages"][0]["role"] == "system"
    assert "never diagnose" in request_body["messages"][0]["content"].lower()
    assert "untrusted reference text" in request_body["messages"][0]["content"]
    assert request_body["options"]["num_ctx"] == 2048
    assert request_body["options"]["num_predict"] == 256
    assert post.call_args.kwargs["trust_env"] is False


def test_retrieve_project_context_finds_relevant_document_section(monkeypatch, tmp_path):
    docs = tmp_path / "README.md"
    docs.write_text(
        "# Setup\n\nRun the local FastAPI backend with `python -m modules.api`.\n\n"
        "# Privacy\n\nThe synthetic audit stores a fixed event type only.",
        encoding="utf-8",
    )
    monkeypatch.setattr(chatbot, "PROJECT_DOCS_PATH", docs)
    monkeypatch.setattr(chatbot, "PROJECT_SOURCE_PATHS", ())

    context = chatbot.retrieve_project_context("How do I run the API backend?")

    assert "Section: Setup" in context
    assert "python -m modules.api" in context
    assert "Section: Privacy" not in context


def test_project_readme_retrieval_includes_exact_backend_command():
    context = chatbot.retrieve_project_context(
        "How do I start the SecurePulse backend?"
    )

    assert "Section: Local frontend and backend" in context
    assert "python -m modules.api" in context


def test_project_source_retrieval_includes_training_implementation():
    context = chatbot.retrieve_project_context(
        "Which metrics does train_model.py calculate for model evaluation?"
    )

    assert "Source: train_model.py" in context
    assert "roc_auc" in context
    assert "accuracy" in context


def test_retrieve_project_context_reports_when_docs_have_no_matching_terms(
    monkeypatch, tmp_path
):
    docs = tmp_path / "README.md"
    docs.write_text("# Setup\n\nRun the local API backend.", encoding="utf-8")
    monkeypatch.setattr(chatbot, "PROJECT_DOCS_PATH", docs)
    monkeypatch.setattr(chatbot, "PROJECT_SOURCE_PATHS", ())

    context = chatbot.retrieve_project_context("What is the moon made of?")

    assert context == "No relevant project documentation was found for this question."


def test_ask_ollama_rejects_empty_or_oversized_query():
    with pytest.raises(ValueError, match="Enter a question"):
        chatbot.ask_ollama(" ")

    with pytest.raises(ValueError, match="2000 characters"):
        chatbot.ask_ollama("x" * 2001)
