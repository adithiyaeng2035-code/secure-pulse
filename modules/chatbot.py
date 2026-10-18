"""Loopback-only Ollama chat for SecurePulse project questions."""

from __future__ import annotations

import os
from ipaddress import ip_address
from urllib.parse import urlsplit

import httpx

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
MAX_QUERY_LENGTH = 2000

SYSTEM_PROMPT = """You are the SecurePulse project help assistant.
Answer only questions about SecurePulse software, its synthetic dataset, local
setup, prediction workflow, privacy behavior, and how to interpret the demo UI.
Treat each question as independent; no previous chat messages are provided.
This project uses generated synthetic data; its models are not clinically
validated. Never diagnose, estimate a real person's medical risk, recommend
treatment, or ask for personal health information. If asked for medical advice,
briefly explain that you cannot provide it and recommend consulting a qualified
health professional. Do not claim this project is deployed to Cloud Run unless
the user explicitly provides verifiable deployment details. Be concise and
state when you do not know an answer."""


class OllamaUnavailableError(RuntimeError):
    """Raised when a local Ollama service or configured model is unavailable."""


def get_ollama_config() -> tuple[str, str]:
    base_url = os.environ.get("SECUREPULSE_OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.environ.get("SECUREPULSE_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
    parsed_url = urlsplit(base_url)
    if parsed_url.scheme != "http" or not parsed_url.hostname:
        raise ValueError("Ollama must use a local HTTP URL.")
    if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
        raise ValueError("Ollama URL cannot contain credentials, query, or fragment.")
    try:
        parsed_url.port
    except ValueError as error:
        raise ValueError("Ollama URL contains an invalid port.") from error
    hostname = parsed_url.hostname.lower()
    try:
        loopback = ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname == "localhost"
    if not loopback or parsed_url.path not in {"", "/"}:
        raise ValueError("Ollama must be available on this machine's loopback interface.")
    if not model or len(model) > 100:
        raise ValueError("Configure a valid local Ollama model name.")
    return base_url, model


def check_ollama() -> dict[str, str | bool]:
    try:
        base_url, model = get_ollama_config()
        response = httpx.get(f"{base_url}/api/tags", timeout=2, trust_env=False)
        response.raise_for_status()
        payload = response.json()
        installed_models = {
            str(item.get("name", "")) for item in payload.get("models", [])
        }
    except (ValueError, httpx.HTTPError, TypeError, AttributeError) as error:
        return {"available": False, "model": "", "detail": str(error)}

    if model not in installed_models:
        return {
            "available": False,
            "model": model,
            "detail": f"Model '{model}' is not installed. Run `ollama pull {model}`.",
        }
    return {"available": True, "model": model, "detail": "Local Ollama model is ready."}


def ask_ollama(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValueError("Enter a question for the SecurePulse assistant.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Questions must be {MAX_QUERY_LENGTH} characters or fewer.")

    status = check_ollama()
    if not status["available"]:
        raise OllamaUnavailableError(str(status["detail"]))

    base_url, model = get_ollama_config()
    try:
        response = httpx.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 400},
            },
            timeout=90,
            trust_env=False,
        )
        response.raise_for_status()
        answer = response.json()["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise OllamaUnavailableError(f"Local Ollama request failed: {error}") from error
    if not answer:
        raise OllamaUnavailableError("Local Ollama returned an empty answer.")
    return answer
