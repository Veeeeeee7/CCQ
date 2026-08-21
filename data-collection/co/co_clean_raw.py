"""
co_clean_raw.py — Build the `raw` dataset (text preserved + maximally
decomposed) for LLM-based methods from the Colorado Shines scrape.

Pipeline:
  load (as strings) -> null out site-scraped fields on id_mismatch (row kept;
  see co_cleaning_utils docstring) -> dollar-strip (no-op today, kept for
  cross-state parity) -> grain check -> drop NON_FEATURE_COLS -> per-field
  builders (text-preserving) -> finalize (valid ratings only).

Same early steps and same row filtering as co_clean_full.py (so the two
outputs are row-aligned), differing only in which mode each builder runs in
and which columns `base` keeps as original text vs. numeric passthrough.

Run:
    python co_clean_raw.py --input co_data/co_records.csv --output co_data/co_cleaned_raw.csv
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
    parser.add_argument("--output", type=Path, default=Path("co_data/co_records_cleaned_raw.csv"))
    parser.add_argument("--scaffold", type=Path,
                        default=Path(__file__).resolve().parent / "co_columns.json")
    parser.add_argument("--log", type=Path, default=Path("co_clean_raw.log"))
    args = parser.parse_args()

    log = U.ParseLog(args.log)
    scaffold = json.loads(Path(args.scaffold).read_text())

    print(f"[raw] loading {args.input}")
    df = pd.read_csv(args.input, low_memory=False, dtype=str)  # preserve leading zeros

    # --- shared early steps (identical to full, keeps both row-aligned) ------
    df = U.null_out_enrichment_on_mismatch(df, log)
    df = df.drop(columns=["errors"], errors="ignore")
    df = U.strip_dollars(df)
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns], errors="ignore")

    # --- base: id, target, numeric passthroughs, preserved original text -----
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

    # original text columns kept verbatim (raw preserves text) -- includes the
    # on-site ID cross-check field (license_number_on_site; see the utils
    # docstring for why it isn't just dropped).
    #
    # award_date / expiration_date / rating_on_site used to live here and were
    # removed as target leakage -- see U.LEAKAGE_COLS. license_issue_date stays:
    # licensing lifecycle, not rating lifecycle.
    text_passthrough = [
        "provider_service_type", "county", "license_type",
        "license_issue_date",
        "hours_of_operation", "licensed_to_serve",
        "license_number_on_site",
    ]
    for c in text_passthrough:
        if c in df.columns:
            base[c] = df[c]

    # Yes/No and True/False text -> real boolean, kept in raw too (these are
    # derived signal columns, not original free text, so a clean boolean dtype
    # is more useful here than leaving "Yes"/"True" as opaque strings)
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

    # --- per-field builders (text-preserving) --------------------------------
    parts = [base, U.derive_date_features(df, log)]
    if "hours_of_operation" in df.columns:
        parts.append(U.derive_operating_hours(df["hours_of_operation"], log))
    if "special_needs" in df.columns:
        parts.append(U.build_multivalue(df["special_needs"], ";", "need", "raw"))
    if "languages_spoken" in df.columns:
        parts.append(U.build_keyterm(df["languages_spoken"], U.LANGUAGE_KEYTERMS,
                                     "language", "raw", log))
    parts.append(U.build_licensing_history(df, "raw"))

    engineered = pd.concat(parts, axis=1)

    out = U.finalize(engineered, "raw", scaffold, log)

    U.write_output(out, args.output)
    log.save()


if __name__ == "__main__":
    main()