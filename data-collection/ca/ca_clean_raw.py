"""
clean_raw.py — Build the `raw` dataset (text preserved and maximally
decomposed, plus the scalar extractions) for LLM-based methods.

Row-aligned with full.csv: same source, same error-row filter, same dedup and
target filtering, so one set of CV folds applies to both.

    python clean_raw.py \
        --input facility_records_sample.csv \
        --output raw.csv \
        --scaffold columns_scaffold.json \
        --log parse_log_raw.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ca_cleaning_utils import (
    TARGET_COL,
    check_grain_uniqueness,
    dollar_strip_df,
    drop_error_rows,
    engineer_features,
    finalize,
    load_scaffold,
    make_logger,
)

WHICH = "raw"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("ca_data/ca_records_anonymized.csv"))
    p.add_argument("--output", type=Path, default=Path("ca_data/ca_records_cleaned_raw.csv"))
    p.add_argument("--scaffold", type=Path, default=Path("ca_columns.json"))
    p.add_argument("--log", type=Path, default=Path("ca_parse_log_raw.txt"))
    args = p.parse_args()

    log = make_logger(args.log)
    scaffold = load_scaffold(args.scaffold)

    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    print(f"[{WHICH}] loaded {len(df)} rows, {df.shape[1]} columns from {args.input.name}")

    df = drop_error_rows(df, log=log)        # user requirement, before everything
    check_grain_uniqueness(df, "facility_number")
    df = dollar_strip_df(df, log=log)        # TYPE 2 (harmless if no currency)
    df = engineer_features(df, WHICH, log=log)
    out = finalize(df, WHICH, scaffold, log=log)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[{WHICH}] wrote {out.shape[0]} rows x {out.shape[1]} cols → {args.output}")
    print(f"[{WHICH}] target distribution:\n{out[TARGET_COL].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
