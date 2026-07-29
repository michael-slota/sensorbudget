"""Validate raw occupancy files and build reproducible processed artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sensorbudget.data.load import combine_source_splits, load_source_splits
from sensorbudget.data.provenance import create_manifest, validate_source_checksums
from sensorbudget.data.schema import (
    DEFAULT_CHECKSUM_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PROCESSED_PATH,
    DEFAULT_RAW_DIR,
    SOURCE_FILES,
)
from sensorbudget.data.validate import validate_source_splits


def build_processed_dataset(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    checksum_path: Path | str = DEFAULT_CHECKSUM_PATH,
    output_path: Path | str = DEFAULT_PROCESSED_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict:
    """Validate sources, write a combined CSV, and write its provenance manifest."""

    raw_path = Path(raw_dir)
    output = Path(output_path)
    manifest_output = Path(manifest_path)

    validate_source_checksums(raw_path, checksum_path)
    frames = load_source_splits(raw_path)
    summaries = validate_source_splits(frames)
    combined = combine_source_splits(frames)
    manifest = create_manifest(raw_path, SOURCE_FILES, summaries, combined)

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    manifest_output.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Build processed artifacts from command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--checksums",
        type=Path,
        default=DEFAULT_CHECKSUM_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED_PATH)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    args = parser.parse_args(argv)

    manifest = build_processed_dataset(
        raw_dir=args.raw_dir,
        checksum_path=args.checksums,
        output_path=args.output,
        manifest_path=args.manifest,
    )
    print(
        f"Built {manifest['processed']['rows']:,} rows at {args.output} "
        f"with manifest {args.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
