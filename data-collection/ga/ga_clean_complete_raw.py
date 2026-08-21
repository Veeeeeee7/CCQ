"""
ga_clean_complete_raw.py — Build the `complete_raw` dataset: IDENTICAL
text-preserving treatment to `raw`, but WITHOUT dropping rows whose qr_rating is
invalid/unrated.

This is the raw (LLM-oriented) half of the `complete` split. Its full-engineered
sibling is ga_clean_complete_full.py (→ ga_clean_complete_full.csv).

`raw` and `complete_raw` share the exact same early steps, the same (no-op in
raw mode) builders and the same `raw` scaffold; they differ only in target
handling in the finalize tail:

  raw          : qr_rating coerced to {1,2,3}; out-of-range/unrated rows DROPPED.
  complete_raw : qr_rating coerced numeric and KEPT as-is; out-of-range/unrated
                 rows are retained (non-numeric ratings become <NA> in
                 qr_rating, but the row and its original text survive).

Because complete_raw keeps rows that raw drops, it is a row-superset of `raw`
and is row-aligned with complete_full (same provider set, same row order), so a
single fold file applies across the complete_* pair.

Run:
    python ga_clean_complete_raw.py --input ga_data/ga_records.csv \
                                    --output ga_data/ga_clean_complete_raw.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import ga_clean_utils as U


class _NullLog:
    def warn(self, msg: str) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("ga_data/ga_records_anonymized.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("ga_data/ga_records_cleaned_complete_raw.csv"))
    parser.add_argument("--scaffold", type=Path,
                        default=Path(__file__).resolve().parent / "ga_columns.json")
    args = parser.parse_args()

    log = _NullLog()
    scaffold = json.loads(Path(args.scaffold).read_text())

    print(f"[complete_raw] loading {args.input}")
    df = pd.read_csv(args.input, low_memory=False)

    # --- shared early steps (identical to raw, keeps complete_* aligned) -----
    # NOTE: like raw, complete_raw does NOT run clean_currency; strip_dollars
    # handles the generic plain-$ columns (NC-format behaviour).
    df = U.drop_error_rows(df, log)
    df = U.strip_dollars(df)
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns],
                 errors="ignore")
    df = U.coerce_booleans(df, U.BOOLEAN_COLS)

    # --- per-field builders (text-preserving: all no-ops in "raw" mode) ------
    df = U.clean_licensed_capacity(df, "raw")
    df = U.clean_pre_k_slots(df, "raw")
    df = U.clean_has_liability(df, "raw")
    df = U.clean_languages(df, "raw")
    df = U.clean_curriculum(df, "raw")
    df = U.clean_rates_table(df, "raw")
    df = U.clean_compliance_table(df, "raw")
    df = U.clean_operating_hours(df, "raw")
    df = U.clean_provider_type(df, "raw")
    df = U.split_one_hot(df, U.SPLIT_ONEHOT_FIELDS, "raw")
    df = U.clean_operating_months(df, "raw")
    df = U.clean_operating_days(df, "raw")

    # which="raw" → raw scaffold + nan-as-level constant drop; text preserved.
    # keep_invalid_target=True → retain unrated / out-of-range rows.
    out = U.finalize(df, "raw", scaffold, log, keep_invalid_target=True)

    U.write_output(out, args.output)


if __name__ == "__main__":
    main()
