import sqlite3

import pytest

from modules.synthetic_audit import (
    AuditChainIntegrityError,
    SYNTHETIC_PREDICTION_EVENT,
    append_synthetic_prediction_event,
    verify_synthetic_audit_chain,
)


def test_new_chain_is_valid_and_empty(tmp_path):
    status = verify_synthetic_audit_chain(tmp_path / "audit.sqlite3")

    assert status.valid
    assert status.event_count == 0


def test_append_creates_valid_hash_chain_without_reading_or_score_fields(tmp_path):
    path = tmp_path / "audit.sqlite3"

    first = append_synthetic_prediction_event(path)
    second = append_synthetic_prediction_event(path)
    status = verify_synthetic_audit_chain(path)

    assert first.sequence == 1
    assert second.sequence == 2
    assert first.event_hash != second.event_hash
    assert status.valid
    assert status.event_count == 2

    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(audit_events)")
        }
        rows = connection.execute(
            "SELECT timestamp, event_type FROM audit_events ORDER BY sequence"
        ).fetchall()

    assert columns == {
        "sequence",
        "timestamp",
        "event_type",
        "previous_hash",
        "event_hash",
    }
    assert all(row[1] == SYNTHETIC_PREDICTION_EVENT for row in rows)
    assert all(isinstance(row[0], str) and row[0].endswith("+00:00") for row in rows)


def test_chain_detects_tampered_event_and_refuses_append(tmp_path):
    path = tmp_path / "audit.sqlite3"
    append_synthetic_prediction_event(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE audit_events SET timestamp = ? WHERE sequence = 1",
            ("2000-01-01T00:00:00+00:00",),
        )

    status = verify_synthetic_audit_chain(path)

    assert not status.valid
    with pytest.raises(AuditChainIntegrityError, match="integrity check failed"):
        append_synthetic_prediction_event(path)
