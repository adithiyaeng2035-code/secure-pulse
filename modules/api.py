"""Local HTTP API for synthetic SecurePulse predictions."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hmac
import os
import sqlite3
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modules.anomaly_detection import detect_anomaly, load_anomaly_model
from modules.chatbot import (
    MAX_QUERY_LENGTH,
    OllamaUnavailableError,
    ProjectDocsUnavailableError,
    ask_ollama,
    check_ollama,
    get_ollama_config,
)
from modules.data_processor import DataValidationError, prepare_patient_record
from modules.risk_prediction import load_risk_model, predict_risk
from modules.synthetic_audit import (
    AuditChainIntegrityError,
    append_synthetic_prediction_event,
)

app = FastAPI(
    title="SecurePulse Synthetic Prediction API",
    description=(
        "API for synthetic cardiometabolic demo predictions. Keep local unless "
        "deployed behind verified authentication and access controls."
    ),
    version="1.0.0",
)
bearer_auth = HTTPBearer(auto_error=False)


class SyntheticReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=110)
    systolic_bp: int = Field(ge=70, le=250)
    diastolic_bp: int = Field(ge=40, le=150)
    cholesterol: int = Field(ge=80, le=500)
    glucose: int = Field(ge=40, le=500)
    bmi: float = Field(ge=10, le=80)
    smoking: int = Field(ge=0, le=1)
    family_history: int = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_blood_pressure(self) -> SyntheticReading:
        if self.diastolic_bp >= self.systolic_bp:
            raise ValueError("Diastolic blood pressure must be lower than systolic blood pressure.")
        return self


class PredictionResponse(BaseModel):
    submitted_at: datetime
    risk_band: Literal["Low", "Moderate", "High"]
    risk_probability: float = Field(ge=0, le=1)
    is_anomaly: bool
    anomaly_score: float
    audit_sequence: int | None
    audit_error: str | None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)


class ChatResponse(BaseModel):
    answer: str
    model: str


def verify_api_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_auth),
) -> None:
    expected_token = os.environ.get("SECUREPULSE_API_TOKEN")
    token_required = os.environ.get("SECUREPULSE_REQUIRE_API_TOKEN", "").lower() in {
        "1",
        "true",
        "yes",
    }
    if not expected_token:
        if token_required:
            raise HTTPException(
                status_code=503,
                detail="Prediction API authentication is required but not configured.",
            )
        return
    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode("utf-8"), expected_token.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="A valid bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@lru_cache(maxsize=1)
def get_models() -> tuple[Any, Any]:
    return load_risk_model(), load_anomaly_model()


@app.get("/health")
def health() -> dict[str, str]:
    try:
        get_models()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail=f"Model engine unavailable: {error}") from error
    return {"status": "ok", "engine": "synthetic-risk-api"}


@app.get("/chat/health")
def chat_health() -> dict[str, str | bool]:
    return check_ollama()


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_token)],
)
def create_chat_response(request: ChatRequest) -> ChatResponse:
    try:
        answer = ask_ollama(request.query)
        _, model = get_ollama_config()
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ProjectDocsUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return ChatResponse(answer=answer, model=model)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_token)],
)
def create_prediction(reading: SyntheticReading) -> PredictionResponse:
    try:
        risk_model, anomaly_model = get_models()
        features = prepare_patient_record(reading.model_dump())
        risk = predict_risk(features, model=risk_model)
        anomaly = detect_anomaly(features, model=anomaly_model)
    except DataValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (OSError, ValueError, TypeError) as error:
        raise HTTPException(status_code=503, detail=f"Prediction engine failed: {error}") from error

    audit_sequence = None
    audit_error = None
    try:
        audit_sequence = append_synthetic_prediction_event().sequence
    except (AuditChainIntegrityError, OSError, sqlite3.Error) as error:
        audit_error = str(error)

    return PredictionResponse(
        submitted_at=datetime.now(timezone.utc),
        risk_band=str(risk["risk_band"]),
        risk_probability=float(risk["risk_probability"]),
        is_anomaly=bool(anomaly["is_anomaly"]),
        anomaly_score=float(anomaly["anomaly_score"]),
        audit_sequence=audit_sequence,
        audit_error=audit_error,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "modules.api:app",
        host="127.0.0.1",
        port=int(os.environ.get("SECUREPULSE_API_PORT", "8000")),
        reload=False,
    )
