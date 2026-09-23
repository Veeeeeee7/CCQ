"""
ga_clean_complete_raw.py — Build the `complete_raw` dataset: same
text-preserving treatment as `raw`, but rows whose qr_rating is invalid/unrated
are kept (non-numeric ratings become <NA>). Row-aligned with complete_full.

Run:
    python ga_clean_complete_raw.py
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

    # raw deliberately does NOT run clean_currency; strip_dollars handles the
    # plain-$ columns.
    df = U.drop_error_rows(df, log)
    df = U.strip_dollars(df)
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns],
                 errors="ignore")
    df = U.coerce_booleans(df, U.BOOLEAN_COLS)

    # per-field builders: all no-ops in "raw" mode
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

    out = U.finalize(df, "raw", scaffold, log, keep_invalid_target=True)

    U.write_output(out, args.output)


if __name__ == "__main__":
    main()
