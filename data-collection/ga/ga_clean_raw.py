"""
ga_clean_raw.py — Build the `raw` dataset (text preserved) for LLM-based methods
from the GA DECAL childcare scrape.

Same early steps and the same row filtering as ga_clean_full.py (so the two
outputs are row-aligned and a single fold file applies to both), but NO feature
engineering: the original multi-value / prose columns are kept verbatim. This
mirrors the original `cleaning_classification_raw.py`, which only:

  strips a leading '$' (and thousands commas) from currency-formatted strings →
  selects the `raw` scaffold columns → dedups on provider_id →
  drops unrated rows → removes all-NaN/constant columns.

The per-field builders are still invoked for structural parity with the full
pipeline, but every one of them is a documented no-op in "raw" mode (it returns
the frame untouched), so the original text survives into the output.

Run:
    python ga_clean_raw.py --input ga_data/ga_records.csv \
                           --output ga_data/ga_cleaned_raw.csv
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
    parser.add_argument("--output", type=Path, default=Path("ga_data/ga_records_cleaned_raw.csv"))
    parser.add_argument("--scaffold", type=Path,
                        default=Path(__file__).resolve().parent / "ga_columns.json")
    parser.add_argument("--log", type=Path, default=Path("ga_clean_raw.log"))
    args = parser.parse_args()

    log = U.ParseLog(args.log)
    scaffold = json.loads(Path(args.scaffold).read_text())

    print(f"[raw] loading {args.input}")
    df = pd.read_csv(args.input, low_memory=False)

    # --- shared early steps (identical to full, keeps both row-aligned) ------
    # NOTE: raw deliberately does NOT run clean_currency; the original raw kept
    # registration_fee / activity_fee as $-stripped values. strip_dollars handles
    # the generic plain-$ columns (NC-format behaviour).
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

    out = U.finalize(df, "raw", scaffold, log)

    U.write_output(out, args.output)
    log.save()


if __name__ == "__main__":
    main()