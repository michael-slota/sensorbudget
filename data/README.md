# Data handling

## Source

The project uses the
[UCI Occupancy Detection dataset](https://archive.ics.uci.edu/dataset/357/occupancy%2Bdetection),
distributed under CC BY 4.0.

Suggested citation:

> Candanedo, L. (2016). Occupancy Detection [Dataset]. UCI Machine Learning
> Repository. https://doi.org/10.24432/C5X01N

## Directory policy

| Directory | Purpose | Versioned? |
|---|---|---:|
| `raw/` | Exact downloaded source files | No |
| `external/` | Other immutable third-party inputs | No |
| `interim/` | Temporary cleaned or joined data | No |
| `processed/` | Final model-ready tables | No |

Only `.gitkeep` placeholders and documentation are committed. Generated data
must be reproducible from source files and code.

## Expected raw files

Place the three supplied files directly under `data/raw/`:

- `datatraining.txt`
- `datatest.txt`
- `datatest2.txt`

The ingestion pipeline should:

1. Record download date, source URL, file size, and SHA-256 checksum.
2. Verify the expected columns and types.
3. Parse `date` as a timestamp and sort chronologically.
4. Reject duplicate timestamps unless explicitly resolved.
5. Confirm `Occupancy` contains only 0 and 1.
6. Preserve an origin/split column identifying the source file.
7. Never modify the raw files in place.

The committed `source_checksums.json` records the expected file sizes and
SHA-256 hashes. The build command refuses to process files that do not match
those values.

## Pipeline commands

Validate the raw files without writing outputs:

```powershell
sensorbudget-validate-data
```

Build the chronological processed table and generated provenance manifest:

```powershell
sensorbudget-build-data
```

Equivalent module commands are:

```powershell
python -m sensorbudget.data.validate
python -m sensorbudget.data.build
```

Generated outputs:

```text
data/processed/occupancy.csv
data/processed/manifest.json
```

The combined table is sorted chronologically but retains `source_split`.
Modeling code must use that column to recover the supplied development and
held-out periods rather than randomly splitting rows.

## Leakage controls

- Do not create the final evaluation split by randomly sampling rows.
- Fit imputers, scalers, selectors, and models using training data only.
- Lagged features may use past values, never future values.
- Time-of-day features are permitted, but their effect must be reported
  separately because they may encode the room's schedule.
- Target-derived features are prohibited.
