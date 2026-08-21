"""
ga_clean_full.py — Build the `full` dataset (strictly numeric/boolean +
provider_id) for classical/tabular ML from the GA DECAL childcare scrape.

Pipeline:
  load → drop source-error rows → currency-strip → grain check →
  drop NON_FEATURE_COLS → coerce checkmark booleans →
  per-field builders (numeric/boolean only) → finalize.

The per-field builders run in "full" mode: every multi-value / prose field is
engineered into numeric or boolean feature columns, exactly mirroring the order
of the original `cleaning_classification_full.py`:

  licensed_capacity → pre_k_slots → has_liability → languages → curriculum →
  rates_table → compliance_table → operating_hours → provider_type →
  split_one_hot(SPLIT_ONEHOT_FIELDS) → operating_months → operating_days.

`finalize` then reindexes to the "full" scaffold (columns_classification_full),
drops the unrated rows (qr_rating not in {1,2,3}), removes all-NaN/constant
columns and casts surviving booleans to nullable Int64 so the CSV round-trips
as a purely numeric table.

Run:
    python ga_clean_full.py --input ga_data/ga_records.csv \
                            --output ga_data/ga_cleaned_full.csv
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
    # leading-zero ids to protect; reading without dtype=str lets the numeric /
    # boolean crawl columns auto-infer (matches the original GA loader).
    df = pd.read_csv(args.input, low_memory=False)

    # --- shared early steps (identical to raw, keeps both row-aligned) -------
    df = U.drop_error_rows(df, log)
    df = U.clean_currency(df)              # registration_fee / activity_fee → float
    df = U.strip_dollars(df)               # any other plain-$ column → float
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns],
                 errors="ignore")
    df = U.coerce_booleans(df, U.BOOLEAN_COLS)

    # --- per-field builders (numeric / boolean only) -------------------------
    #     order mirrors cleaning_classification_full.py exactly.
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