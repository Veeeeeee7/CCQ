"""
co_clean_full.py — Build the `full` dataset (numeric/boolean + provider_id,
valid ratings only) from the Colorado Shines records.

Pipeline:
  load (as strings) -> null out page-derived fields on id_mismatch (row kept)
  -> dollar-strip (no-op today) -> grain check -> drop NON_FEATURE_COLS ->
  per-field builders (numeric/boolean only) -> finalize (valid ratings only).

Same early steps and row filtering as co_clean_raw.py, so the two outputs are
row-aligned.

Run:
    python co_clean_full.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import co_cleaning_utils as U


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("co_data/co_records_anonymized.csv"))
    parser.add_argument("--output", type=Path, default=Path("co_data/co_records_cleaned_full.csv"))
    parser.add_argument("--scaffold", type=Path,
                        default=Path(__file__).resolve().parent / "co_columns.json")
    parser.add_argument("--log", type=Path, default=Path("co_clean_full.log"))
    args = parser.parse_args()

    log = U.ParseLog(args.log)
    scaffold = json.loads(Path(args.scaffold).read_text())

    print(f"[full] loading {args.input}")
    df = pd.read_csv(args.input, low_memory=False, dtype=str)  # preserve leading zeros

    df = U.null_out_enrichment_on_mismatch(df, log)
    df = df.drop(columns=["errors"], errors="ignore")
    df = U.strip_dollars(df)
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns], errors="ignore")

    # base: id, target, numeric passthroughs
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
        # one column per recognised language; the k>=5 sweep folds rare ones
        # into language_other
        parts.append(U.build_language_features(df["languages_spoken"], "full", log))
    parts.append(U.build_licensing_history(df, "full", log))

    engineered = pd.concat(parts, axis=1)

    out = U.finalize(engineered, "full", scaffold, log)

    # sanity: every non-id column must be numeric/boolean
    bad = [c for c in out.columns
           if c != U.ID_COL and not pd.api.types.is_numeric_dtype(out[c])
           and not pd.api.types.is_bool_dtype(out[c]) and out[c].dtype != "boolean"]
    if bad:
        log.warn(f"[full] NON-NUMERIC columns survived (investigate): {bad}")
        print(f"  WARNING: non-numeric columns in full: {bad}")

    U.write_output(out, args.output)
    log.save()


if __name__ == "__main__":
    main()