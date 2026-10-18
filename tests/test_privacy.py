import pandas as pd
import pytest

from modules.privacy import pseudonymize_identifier, remove_direct_identifiers


def test_pseudonym_is_stable_for_same_identifier_and_salt():
    first = pseudonymize_identifier("patient-123", salt="test-secret-that-is-long")
    second = pseudonymize_identifier("patient-123", salt="test-secret-that-is-long")

    assert first == second
    assert first.startswith("patient-")
    assert "patient-123" not in first


def test_pseudonym_uses_salt():
    first = pseudonymize_identifier("patient-123", salt="salt-one-is-long-enough")
    second = pseudonymize_identifier("patient-123", salt="salt-two-is-long-enough")

    assert first != second


def test_pseudonym_rejects_empty_identifier():
    with pytest.raises(ValueError, match="non-empty"):
        pseudonymize_identifier(" ", salt="test-secret-that-is-long")


def test_pseudonym_rejects_weak_salt():
    with pytest.raises(ValueError, match="at least 16 characters"):
        pseudonymize_identifier("patient-123", salt="weak")


def test_remove_direct_identifiers_returns_safe_copy():
    original = pd.DataFrame(
        {
            "Name": ["Synthetic Person"],
            "email": ["person@example.test"],
            "Patient ID": ["SYN-001"],
            "Date-of-Birth": ["1980-01-01"],
            "age": [45],
        }
    )

    result = remove_direct_identifiers(original)

    assert result.columns.tolist() == ["age"]
    assert "Name" in original.columns


def test_remove_direct_identifiers_normalizes_custom_column_names():
    original = pd.DataFrame(
        {"Contact Number": ["555-0100"], "record-id": ["row-1"], "measurement": [42]}
    )

    result = remove_direct_identifiers(original, extra_columns=["Contact Number"])

    assert result.columns.tolist() == ["measurement"]
