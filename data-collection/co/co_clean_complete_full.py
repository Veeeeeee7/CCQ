"""
co_clean_complete_full.py — Build the `complete_full` dataset: IDENTICAL
feature engineering to `full` (strictly numeric/boolean + provider_id), but
WITHOUT dropping rows whose qr_rating is invalid/unrated (the `complete`
target policy).

The four outputs, on two axes (preprocessing style x target filtering):

                       drop invalid ratings        keep invalid (complete)
  full  (numeric)      co_cleaned_full.csv         co_clean_complete_full.csv
  raw   (text)         co_cleaned_raw.csv          co_clean_complete_raw.csv

complete_full and complete_raw share the same early steps and the same (no)
target filtering, so they are ROW-ALIGNED with each other -- a single fold
file covers both. They are NOT row-aligned with raw/full, which restrict to
valid 1-5 ratings. This includes providers whose type is structurally never
rated by Colorado Shines (School-Age Child Care Center, Resident Camp,
Neighborhood Youth Organization) alongside any
genuinely unrated/pending ratable-type providers.

Uses a no-op logger rather than co_clean_full.py's ParseLog: this runs the
identical engineering over the identical input, so every parse warning it
would produce is already captured in co_clean_full.log by the sibling script.

Run:
    python co_clean_complete_full.py --input co_data/co_records.csv --output co_data/co_clean_complete_full.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import co_cleaning_utils as U


class _NullLog:
    def warn(self, msg: str) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("co_data/co_records_anonymized.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("co_data/co_records_cleaned_complete_full.csv"))
    parser.add_argument("--scaffold", type=Path,
                        default=Path(__file__).resolve().parent / "co_columns.json")
    args = parser.parse_args()

    log = _NullLog()
    scaffold = json.loads(Path(args.scaffold).read_text())

    print(f"[complete_full] loading {args.input}")
    df = pd.read_csv(args.input, low_memory=False, dtype=str)  # preserve leading zeros

    # --- shared early steps (identical to raw/full, keeps engineering aligned)
    df = U.null_out_enrichment_on_mismatch(df, log)
    df = df.drop(columns=["errors"], errors="ignore")
    df = U.strip_dollars(df)
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns], errors="ignore")

    # --- base: id, target, numeric passthroughs (identical to co_clean_full.py)
    base = pd.DataFrame(index=df.index)
    base[U.ID_COL] = df[U.ID_COL]
    base[U.TARGET_COL] = df[U.TARGET_COL]

    numeric_passthrough = [
        "total_licensed_capacity", "licensed_home_capacity", "licensed_infant_capacity",
        "licensed_toddler_capacity", "licensed_preschool_capacity",
        "licensed_school_age_capacity", "licensed_preschool_and_school_age_capacity",
        "licensed_resident_camp_capacity", "licensed_nyo_capacity", "capacity_on_site",
        "openings_infant", "openings_toddler", "openings_preschool", "openings_school_age",
    ]
    for c in numeric_passthrough:
        if c in df.columns:
            base[c] = pd.to_numeric(df[c], errors="coerce")

    yes_no = ["head_start", "accepts_cccap_on_site", "accepting_new_children"]
    true_false = ["school_district_operated_program", "cccap_fa_status_d1",
                  "cccap_authorization_status", "upk_participation_2025_2026",
                  "upk_participation_2026_2027"]
    for c in yes_no:
        if c in df.columns:
            base[c] = U.to_boolean(df[c], ("yes",), ("no",))
    for c in true_false:
        if c in df.columns:
            base[c] = U.to_boolean(df[c], ("true",), ("false",))

    # --- per-field builders (numeric / boolean only -- identical to full) ----
    parts = [base, U.derive_date_features(df, log)]
    if "hours_of_operation" in df.columns:
        parts.append(U.derive_operating_hours(df["hours_of_operation"], log))
    if "provider_service_type" in df.columns:
        parts.append(U.build_categorical_onehot(df["provider_service_type"], "type"))
    if "county" in df.columns:
        parts.append(U.build_categorical_onehot(df["county"], "county"))
    if "license_type" in df.columns:
        parts.append(U.build_categorical_onehot(df["license_type"], "licensetype"))
    if "special_needs" in df.columns:
        parts.append(U.build_multivalue(df["special_needs"], ";", "need", "full"))
    if "languages_spoken" in df.columns:
        parts.append(U.build_keyterm(df["languages_spoken"], U.LANGUAGE_KEYTERMS,
                                     "language", "full", log))
    parts.append(U.build_licensing_history(df, "full"))

    engineered = pd.concat(parts, axis=1)

    # which="full" -> reuse full's scaffold / discovered prefixes / numeric
    # cast; keep_invalid_target=True -> retain unrated/out-of-range rows.
    out = U.finalize(engineered, "full", scaffold, log, keep_invalid_target=True)

    # sanity: every non-id column must be numeric/boolean (qr_rating is
    # nullable Int64 so unrated rows carry <NA> without breaking the guarantee)
    bad = [c for c in out.columns
           if c != U.ID_COL and not pd.api.types.is_numeric_dtype(out[c])
           and not pd.api.types.is_bool_dtype(out[c]) and out[c].dtype != "boolean"]
    if bad:
        log.warn(f"[complete_full] NON-NUMERIC columns survived (investigate): {bad}")
        print(f"  WARNING: non-numeric columns in complete_full: {bad}")

    U.write_output(out, args.output)


if __name__ == "__main__":
    main()