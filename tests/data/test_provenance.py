from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sensorbudget.data.provenance import (
    sha256_file,
    validate_source_checksums,
)

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures"


def test_sha256_file_matches_hashlib() -> None:
    path = FIXTURE_DIR / "sample.bin"
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_checksum_validation_detects_changed_file() -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_source_checksums(
            FIXTURE_DIR,
            FIXTURE_DIR / "invalid_checksums.json",
        )
