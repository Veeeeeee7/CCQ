"""
ga_clean_full.py — Build the `full` dataset (numeric/boolean + provider_id,
valid ratings only) from the GA DECAL child care records.

  load → drop source-error rows → currency-strip → grain check →
  drop NON_FEATURE_COLS → coerce booleans → per-field builders → finalize.

finalize reindexes to the "full" scaffold, drops unrated rows (qr_rating not in
{1,2,3}), removes all-NaN/constant columns and casts booleans to Int64.

Run:
    python ga_clean_full.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import ga_clean_utils as U


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("ga_data/ga_records_anonymized.csv"))
    parser.add_argument("--output", type=Path, default=Path("ga_data/ga_records_cleaned_full.csv"))
    parser.add_argument("--scaffold", type=Path,
                        default=Path(__file__).resolve().parent / "ga_columns.json")
    parser.add_argument("--log", type=Path, default=Path("ga_clean_full.log"))
    args = parser.parse_args()

    log = U.ParseLog(args.log)
    scaffold = json.loads(Path(args.scaffold).read_text())

    print(f"[full] loading {args.input}")
    # GA provider_id carries a letter prefix ("FR-000000288"), so there are no
    # leading-zero ids to protect and dtypes can be inferred.
    df = pd.read_csv(args.input, low_memory=False)

    df = U.drop_error_rows(df, log)
    df = U.clean_currency(df)
    df = U.strip_dollars(df)
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns],
                 errors="ignore")
    df = U.coerce_booleans(df, U.BOOLEAN_COLS)

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

    out = U.finalize(df, "full", scaffold, log)

    # sanity: every non-id column must be numeric/boolean
    bad = [c for c in out.columns
           if c != "provider_id" and not pd.api.types.is_numeric_dtype(out[c])
           and not pd.api.types.is_bool_dtype(out[c])
           and out[c].dtype != "boolean"]
    if bad:
        log.warn(f"[full] NON-NUMERIC columns survived (investigate): {bad}")
        print(f"  WARNING: non-numeric columns in full: {bad}")

    U.write_output(out, args.output)
    log.save()


if __name__ == "__main__":
    main()