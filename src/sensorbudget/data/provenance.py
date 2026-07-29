"""Create reproducibility metadata for immutable source files."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sensorbudget.data.schema import SOURCE_DOI, SOURCE_LICENSE, SOURCE_URL


def sha256_file(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_checksums(
    raw_dir: Path | str,
    checksum_path: Path | str,
) -> dict[str, dict[str, Any]]:
    """Verify source file sizes and hashes against committed expectations."""

    expectations = json.loads(Path(checksum_path).read_text(encoding="utf-8"))
    raw_path = Path(raw_dir)
    errors = []
    for filename, expected in expectations["files"].items():
        source_path = raw_path / filename
        if not source_path.is_file():
            errors.append(f"Missing source file: {source_path}")
            continue

        actual_size = source_path.stat().st_size
        actual_hash = sha256_file(source_path)
        if actual_size != expected["bytes"]:
            errors.append(
                f"{filename}: expected {expected['bytes']} bytes, "
                f"found {actual_size}."
            )
        if actual_hash != expected["sha256"]:
            errors.append(
                f"{filename}: SHA-256 mismatch "
                f"(expected {expected['sha256']}, found {actual_hash})."
            )

    if errors:
        raise ValueError("\n".join(errors))
    return expectations["files"]


def create_manifest(
    raw_dir: Path | str,
    source_files: Mapping[str, str],
    summaries: Mapping[str, Mapping[str, Any]],
    processed_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Build a JSON-serializable provenance manifest."""

    raw_path = Path(raw_dir)
    file_records = {}
    for split, filename in source_files.items():
        path = raw_path / filename
        file_records[split] = {
            "filename": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            **dict(summaries[split]),
        }

    return {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": "UCI Occupancy Detection",
            "url": SOURCE_URL,
            "doi": SOURCE_DOI,
            "license": SOURCE_LICENSE,
        },
        "files": file_records,
        "processed": {
            "rows": int(len(processed_frame)),
            "columns": processed_frame.columns.tolist(),
            "split_counts": {
                str(split): int(count)
                for split, count in processed_frame["source_split"]
                .value_counts()
                .items()
            },
        },
    }
