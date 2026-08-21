"""
clean_complete_raw.py — Build the `complete_raw` dataset: identical to `raw`
(text preserved and maximally decomposed, for LLM-based methods), but it KEEPS
rows whose quality rating is invalid (out of 1..5) or missing instead of
dropping them.

Row-aligned with complete_full.csv: same source, same error-row filter, same
dedup, and the same (no-op) target filter, so one set of CV folds applies to
both. The valid `raw` set is a row subset of this `complete_raw` set:
raw == complete_raw with the non-1..5 rows removed.

The target column qr_rating here contains: 1..5 for properly rated facilities,
any out-of-range numeric score preserved as-is (e.g. a stray 7), and <NA> for
the '-' sentinel / non-numeric / missing ratings.

    python clean_complete_raw.py \
        --input facility_records_sample.csv \
        --output complete_raw.csv \
        --scaffold columns_scaffold.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ca_cleaning_utils import (
    TARGET_COL,
    VALID_TARGET_VALUES,
    check_grain_uniqueness,
    dollar_strip_df,
    drop_error_rows,
    engineer_features,
    finalize,
    load_scaffold,
)

# Engineering/scaffold are the `raw` view; only the target handling differs.
WHICH = "raw"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("ca_data/ca_records_anonymized.csv"))
    p.add_argument("--output", type=Path, default=Path("ca_data/ca_records_cleaned_complete_raw.csv"))
    p.add_argument("--scaffold", type=Path, default=Path("ca_columns.json"))
    args = p.parse_args()

    scaffold = load_scaffold(args.scaffold)

    df = pd.read_csv(args.input, dtype=str, low_memory=False)
    print(f"[complete_raw] loaded {len(df)} rows, {df.shape[1]} columns from {args.input.name}")

    df = drop_error_rows(df)        # user requirement, before everything
    check_grain_uniqueness(df, "facility_number")
    df = dollar_strip_df(df)        # TYPE 2 (harmless if no currency)
    df = engineer_features(df, WHICH)
    # keep_invalid_target=True is the only difference from clean_raw.py
    out = finalize(df, WHICH, scaffold, keep_invalid_target=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[complete_raw] wrote {out.shape[0]} rows x {out.shape[1]} cols → {args.output}")

    # Show the full target picture: valid levels, out-of-range, and missing.
    t = out[TARGET_COL]
    n_valid = int(t.isin(VALID_TARGET_VALUES).sum())
    n_invalid_num = int((t.notna() & ~t.isin(VALID_TARGET_VALUES)).sum())
    n_missing = int(t.isna().sum())
    print(f"[complete_raw] target: {n_valid} valid (1..5), "
          f"{n_invalid_num} out-of-range numeric, {n_missing} missing/NA")
    print(f"[complete_raw] target distribution (incl. invalid):\n"
          f"{t.value_counts(dropna=False).sort_index()}")


if __name__ == "__main__":
    main()
