"""
clean_full.py — Build the `full` dataset (strictly numeric/boolean + provider_id)
for classical / tabular ML.

    python clean_full.py \
        --input facility_records_sample.csv \
        --output full.csv \
        --scaffold columns_scaffold.json \
        --log parse_log_full.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ca_cleaning_utils import (
    ID_COL,
    TARGET_COL,
    check_grain_uniqueness,
    dollar_strip_df,
    drop_error_rows,
    engineer_features,
    finalize,
    load_scaffold,
    make_logger,
)

WHICH = "full"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("ca_data/ca_records_anonymized.csv"))
    p.add_argument("--output", type=Path, default=Path("ca_data/ca_records_cleaned_full.csv"))
    p.add_argument("--scaffold", type=Path, default=Path("ca_columns.json"))
    p.add_argument("--log", type=Path, default=Path("ca_parse_log_full.txt"))
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

    # Contract check: every column except provider_id is numeric/boolean.
    non_numeric = [
        c for c in out.columns
        if c != ID_COL and not pd.api.types.is_numeric_dtype(out[c])
    ]
    if non_numeric:
        print(f"[{WHICH}] WARNING: non-numeric columns present: {non_numeric}")
    else:
        print(f"[{WHICH}] OK: all features numeric/boolean (besides {ID_COL}).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[{WHICH}] wrote {out.shape[0]} rows x {out.shape[1]} cols → {args.output}")
    print(f"[{WHICH}] target distribution:\n{out[TARGET_COL].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
