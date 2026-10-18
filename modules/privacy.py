"""Privacy utilities for demo data and safe display."""

from __future__ import annotations

import hashlib
import hmac
import unicodedata
from collections.abc import Iterable

import pandas as pd

DIRECT_IDENTIFIER_COLUMNS = {
    "id",
    "patient_id",
    "record_id",
    "subject_id",
    "mrn",
    "medical_record_number",
    "name",
    "full_name",
    "first_name",
    "last_name",
    "email",
    "phone",
    "address",
    "date_of_birth",
    "dob",
    "postal_code",
    "zip_code",
    "ip_address",
    "ssn",
    "social_security_number",
}
MIN_SALT_LENGTH = 16


def _normalize_column_name(column: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(column)).casefold()
    return "".join(character for character in normalized if character.isalnum())


def pseudonymize_identifier(identifier: str, *, salt: str) -> str:
    """Return a keyed pseudonym; the secret salt must be kept private."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Identifier must be a non-empty string.")
    if not isinstance(salt, str) or len(salt) < MIN_SALT_LENGTH:
        raise ValueError(f"Salt must be at least {MIN_SALT_LENGTH} characters.")
    digest = hmac.new(
        salt.encode("utf-8"),
        identifier.strip().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"patient-{digest[:12]}"


def remove_direct_identifiers(
    data: pd.DataFrame, *, extra_columns: Iterable[str] = ()
) -> pd.DataFrame:
    """Return a copy without known direct-identifier columns."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame.")
    excluded = {_normalize_column_name(column) for column in DIRECT_IDENTIFIER_COLUMNS}
    excluded.update(_normalize_column_name(column) for column in extra_columns)
    return data.drop(
        columns=[
            column
            for column in data.columns
            if _normalize_column_name(column) in excluded
        ],
        errors="ignore",
    ).copy()
