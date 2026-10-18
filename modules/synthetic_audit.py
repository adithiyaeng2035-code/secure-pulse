"""Local hash-chain audit prototype for synthetic demo predictions."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LEDGER_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "securepulse_demo_audit.sqlite3"
)
SYNTHETIC_PREDICTION_EVENT = "synthetic_prediction_completed"
GENESIS_HASH = "0" * 64


class AuditChainIntegrityError(RuntimeError):
    """Raised when an existing local audit chain fails its integrity check."""


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    event_hash: str


@dataclass(frozen=True)
class AuditChainStatus:
    valid: bool
    event_count: int
    last_hash: str


def _calculate_hash(
    sequence: int, timestamp: str, event_type: str, previous_hash: str
) -> str:
    canonical_event = f"{sequence}\n{timestamp}\n{event_type}\n{previous_hash}"
    return hashlib.sha256(canonical_event.encode("utf-8")).hexdigest()


def _verify_connection(connection: sqlite3.Connection) -> AuditChainStatus:
    try:
        rows = connection.execute(
            """
            SELECT sequence, timestamp, event_type, previous_hash, event_hash
            FROM audit_events
            ORDER BY sequence
            """
        )
    except sqlite3.OperationalError as error:
        if "no such table" in str(error).lower():
            return AuditChainStatus(True, 0, GENESIS_HASH)
        raise

    expected_previous_hash = GENESIS_HASH
    expected_sequence = 1
    last_hash = GENESIS_HASH

    for sequence, timestamp, event_type, previous_hash, event_hash in rows:
        if (
            sequence != expected_sequence
            or event_type != SYNTHETIC_PREDICTION_EVENT
            or previous_hash != expected_previous_hash
            or event_hash
            != _calculate_hash(sequence, timestamp, event_type, previous_hash)
        ):
            return AuditChainStatus(False, expected_sequence - 1, last_hash)

        expected_previous_hash = event_hash
        last_hash = event_hash
        expected_sequence += 1

    return AuditChainStatus(True, expected_sequence - 1, last_hash)


def verify_synthetic_audit_chain(
    path: str | Path | None = None,
) -> AuditChainStatus:
    """Check the local hash chain; this does not establish external immutability."""
    database_path = Path(path) if path is not None else DEFAULT_LEDGER_PATH
    if not database_path.is_file():
        return AuditChainStatus(True, 0, GENESIS_HASH)

    with sqlite3.connect(database_path, timeout=10) as connection:
        return _verify_connection(connection)


def append_synthetic_prediction_event(
    path: str | Path | None = None,
) -> AuditEvent:
    """Append only a UTC timestamp and fixed event type; no input or score is accepted."""
    database_path = Path(path) if path is not None else DEFAULT_LEDGER_PATH
    database_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with sqlite3.connect(database_path, timeout=10, isolation_level=None) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                sequence INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL
                    CHECK (event_type = 'synthetic_prediction_completed'),
                previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL
            )
            """
        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            status = _verify_connection(connection)
            if not status.valid:
                raise AuditChainIntegrityError(
                    "Local demo audit chain integrity check failed; no event was appended."
                )

            sequence = status.event_count + 1
            event_hash = _calculate_hash(
                sequence,
                timestamp,
                SYNTHETIC_PREDICTION_EVENT,
                status.last_hash,
            )
            connection.execute(
                """
                INSERT INTO audit_events
                    (sequence, timestamp, event_type, previous_hash, event_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    sequence,
                    timestamp,
                    SYNTHETIC_PREDICTION_EVENT,
                    status.last_hash,
                    event_hash,
                ),
            )
            connection.commit()
        finally:
            if connection.in_transaction:
                connection.rollback()

    return AuditEvent(sequence, event_hash)
