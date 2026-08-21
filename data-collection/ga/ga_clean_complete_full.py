"""
ga_clean_complete_full.py — Build the `complete_full` dataset: IDENTICAL feature
engineering to `full` (strictly numeric/boolean + provider_id) but WITHOUT
dropping rows whose qr_rating is invalid/unrated.

This is the full-engineered half of the `complete` split. Its text-preserving
sibling is ga_clean_complete_raw.py (→ ga_clean_complete_raw.csv).

`full` and `complete_full` share the exact same pipeline and column set; they
differ in one place only — target handling in the finalize tail:

  full          : qr_rating coerced to {1,2,3}; out-of-range/unrated rows DROPPED.
  complete_full : qr_rating coerced numeric and KEPT as-is; out-of-range/unrated
                  rows are retained (non-numeric ratings become <NA> in
                  qr_rating, but the row and all its features survive).

Because complete_full keeps rows that full drops, it is a row-superset of `full`
(and row-aligned with complete_raw). Use it when you need every scraped provider
(e.g. semi-supervised setups, or scoring currently-unrated facilities).

Run:
    python ga_clean_complete_full.py --input ga_data/ga_records.csv \
                                     --output ga_data/ga_clean_complete_full.csv
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
                        default=Path("ga_data/ga_records_cleaned_complete_full.csv"))
    parser.add_argument("--scaffold", type=Path,
                        default=Path(__file__).resolve().parent / "ga_columns.json")
    args = parser.parse_args()

    log = _NullLog()
    scaffold = json.loads(Path(args.scaffold).read_text())

    print(f"[complete_full] loading {args.input}")
    df = pd.read_csv(args.input, low_memory=False)

    # --- shared early steps (identical to full, keeps engineering aligned) ---
    df = U.drop_error_rows(df, log)
    df = U.clean_currency(df)
    df = U.strip_dollars(df)
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns],
                 errors="ignore")
    df = U.coerce_booleans(df, U.BOOLEAN_COLS)

    # --- per-field builders (numeric / boolean only — identical to full) -----
    df = U.clean_licensed_capacity(df, "full")
    df = U.clean_pre_k_slots(df, "full")
    df = U.clean_has_liability(df, "full")
    df = U.clean_languages(df, "full")
    df = U.clean_curriculum(df, "full")
    df = U.clean_rates_table(df, "full")
    df = U.clean_compliance_table(df, "full")
    df = U.clean_operating_hours(df, "full")
    df = U.clean_provider_type(df, "full")
    df = U.split_one_hot(df, U.SPLIT_ONEHOT_FIELDS, "full")
    df = U.clean_operating_months(df, "full")
    df = U.clean_operating_days(df, "full")

    # which="full" → reuse full's scaffold / discovered prefixes / exclusions /
    # numeric cast; keep_invalid_target=True → retain unrated / out-of-range rows.
    out = U.finalize(df, "full", scaffold, log, keep_invalid_target=True)

    # sanity: every non-id column must be numeric/boolean (qr_rating is Int64,
    # nullable so unrated rows carry <NA> without breaking the numeric guarantee)
    bad = [c for c in out.columns
           if c != "provider_id" and not pd.api.types.is_numeric_dtype(out[c])
           and not pd.api.types.is_bool_dtype(out[c])
           and out[c].dtype != "boolean"]
    if bad:
        log.warn(f"[complete_full] NON-NUMERIC columns survived (investigate): {bad}")
        print(f"  WARNING: non-numeric columns in complete_full: {bad}")

    U.write_output(out, args.output)


if __name__ == "__main__":
    main()
