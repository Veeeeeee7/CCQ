"""
clean_complete_full.py — Build the `complete_full` dataset: identical to
`full` (strictly numeric/boolean + provider_id, classical/tabular ML), but it
KEEPS rows whose quality rating is invalid (out of 1..5) or missing instead of
dropping them.

Use this when you want every facility that survived the error-row filter and
dedup, including the unrated / garbage-rated ones. The valid `full` set is a
row subset of this `complete_full` set: full == complete_full with the
non-1..5 rows removed.

The target column qr_rating here therefore contains: 1..5 for properly rated
facilities, any out-of-range numeric score preserved as-is (e.g. a stray 7),
and <NA> for the '-' sentinel / non-numeric / missing ratings.

    python clean_complete_full.py \
        --input facility_records_sample.csv \
        --output complete_full.csv \
        --scaffold columns_scaffold.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ca_cleaning_utils import (
    ID_COL,
    TARGET_COL,
    VALID_TARGET_VALUES,
    check_grain_uniqueness,
    dollar_strip_df,
    drop_error_rows,
    engineer_features,
    finalize,
    load_scaffold,
)

# Engineering/scaffold are the `full` view; only the target handling differs.
WHICH = "full"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("ca_data/ca_records_anonymized.csv"))
    p.add_argument("--output", type=Path, default=Path("ca_data/ca_records_cleaned_complete_full.csv"))
    p.add_argument("--scaffold", type=Path, default=Path("ca_columns.json"))
    args = p.parse_args()

    scaffold = load_scaffold(args.scaffold)

    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    print(f"[complete_full] loaded {len(df)} rows, {df.shape[1]} columns from {args.input.name}")

    df = drop_error_rows(df)        # user requirement, before everything
    check_grain_uniqueness(df, "facility_number")
    df = dollar_strip_df(df)        # TYPE 2 (harmless if no currency)
    df = engineer_features(df, WHICH)
    # keep_invalid_target=True is the only difference from clean_full.py
    out = finalize(df, WHICH, scaffold, keep_invalid_target=True)

    # Contract check: every column except provider_id is numeric/boolean.
    # (qr_rating stays nullable-Int64 numeric even with invalid/NA scores.)
    non_numeric = [
        c for c in out.columns
        if c != ID_COL and not pd.api.types.is_numeric_dtype(out[c])
    ]
    if non_numeric:
        print(f"[complete_full] WARNING: non-numeric columns present: {non_numeric}")
    else:
        print(f"[complete_full] OK: all features numeric/boolean (besides {ID_COL}).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[complete_full] wrote {out.shape[0]} rows x {out.shape[1]} cols → {args.output}")

    # Show the full target picture: valid levels, out-of-range, and missing.
    t = out[TARGET_COL]
    n_valid = int(t.isin(VALID_TARGET_VALUES).sum())
    n_invalid_num = int((t.notna() & ~t.isin(VALID_TARGET_VALUES)).sum())
    n_missing = int(t.isna().sum())
    print(f"[complete_full] target: {n_valid} valid (1..5), "
          f"{n_invalid_num} out-of-range numeric, {n_missing} missing/NA")
    print(f"[complete_full] target distribution (incl. invalid):\n"
          f"{t.value_counts(dropna=False).sort_index()}")


if __name__ == "__main__":
    main()
