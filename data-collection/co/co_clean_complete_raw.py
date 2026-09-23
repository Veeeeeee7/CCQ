"""
co_clean_complete_raw.py — Build the `complete_raw` dataset: same
text-preserving feature engineering as `raw`, but rows whose qr_rating is
invalid/unrated are kept. That includes provider types Colorado Shines never
rates (School-Age Child Care Center, Resident Camp, Neighborhood Youth
Organization). Row-aligned with complete_full, not with raw/full.

Uses a no-op logger: every parse warning is already in co_clean_raw.log.

Run:
    python co_clean_complete_raw.py
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
                        default=Path("co_data/co_records_cleaned_complete_raw.csv"))
    parser.add_argument("--scaffold", type=Path,
                        default=Path(__file__).resolve().parent / "co_columns.json")
    args = parser.parse_args()

    log = _NullLog()
    scaffold = json.loads(Path(args.scaffold).read_text())

    print(f"[complete_raw] loading {args.input}")
    df = pd.read_csv(args.input, low_memory=False, dtype=str)  # preserve leading zeros

    df = U.null_out_enrichment_on_mismatch(df, log)
    df = df.drop(columns=["errors"], errors="ignore")
    df = U.strip_dollars(df)
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns], errors="ignore")

    # base: id, target, numeric passthroughs, original text
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

    text_passthrough = [
        "provider_service_type", "county", "license_type",
        "license_issue_date",
        "hours_of_operation", "licensed_to_serve",
        "license_number_on_site",
    ]
    for c in text_passthrough:
        if c not in df.columns:
            continue
        if c == "licensed_to_serve":
            # One age SET, one spelling: canonicalise BEFORE finalize runs the
            # privacy sweep, so word order does not split a set under k.
            base[c] = U.build_licensed_to_serve(df[c], log)
        else:
            base[c] = df[c]

    # Yes/No and True/False text -> nullable boolean (in raw too)
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
    if "special_needs" in df.columns:
        parts.append(U.build_multivalue(df["special_needs"], ";", "need", "raw"))
    if "languages_spoken" in df.columns:
        # one column per recognised language; the k>=5 sweep folds rare ones
        # into language_other
        parts.append(U.build_language_features(df["languages_spoken"], "raw", log))
    parts.append(U.build_licensing_history(df, "raw", log))

    engineered = pd.concat(parts, axis=1)

    out = U.finalize(engineered, "raw", scaffold, log, keep_invalid_target=True)

    U.write_output(out, args.output)


if __name__ == "__main__":
    main()