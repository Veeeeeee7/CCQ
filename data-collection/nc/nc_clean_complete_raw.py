"""
clean_complete_raw.py — Build the `complete_raw` dataset: IDENTICAL text-preserving
feature engineering to `raw`, but WITHOUT dropping rows whose star rating is
invalid/unrated (the `complete` target policy).

The five outputs, on two axes (preprocessing style × target filtering):

                       drop invalid ratings        keep invalid (complete)
  full  (numeric)      nc_cleaned_full.csv         nc_clean_complete_full.csv
  raw   (text)         nc_cleaned_raw.csv          nc_clean_complete_raw.csv

complete_full and complete_raw share the same early steps and the same (no)
target filtering, so they are ROW-ALIGNED with each other — a single fold file
covers both. They are NOT row-aligned with full/raw, which restrict to valid
1–5 ratings.

The original text is preserved (raw mode); qr_rating is still coerced to a
nullable Int64, with non-numeric placeholders such as 'GS 110-106' or blanks
becoming <NA> while the row and all its text columns survive.

Run:
    python clean_complete_raw.py --input nc_records_sample.csv --output data/complete_raw.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import nc_clean_utils as U


class _NullLog:
    def warn(self, msg: str) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("nc_data/nc_records_anonymized.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("nc_data/nc_records_cleaned_complete_raw.csv"))
    parser.add_argument("--scaffold", type=Path,
                        default=Path(__file__).resolve().parent / "nc_columns.json")
    args = parser.parse_args()

    log = _NullLog()
    scaffold = json.loads(Path(args.scaffold).read_text())

    print(f"[complete_raw] loading {args.input}")
    df = pd.read_csv(args.input, low_memory=False, dtype=str)  # preserve leading zeros

    # --- shared early steps (identical to full/raw, keeps engineering aligned)
    df = U.drop_error_rows(df, log)
    df = U.strip_dollars(df)
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns],
                 errors="ignore")

    # --- base: id, target, passthrough numerics + preserved original text ----
    base = pd.DataFrame(index=df.index)
    base[U.ID_COL] = df[U.ID_COL]
    base[U.TARGET_COL] = df[U.TARGET_COL]
    for c in ("licensed_capacity", "num_visits"):
        if c in df.columns:
            base[c] = pd.to_numeric(df[c], errors="coerce")
    # original text columns kept verbatim (raw preserves text)
    for c in ("facility_type", "ages_served", "license_issue_date",
              "license_restrictions"):
        if c in df.columns:
            base[c] = df[c]

    # --- per-field builders (text-preserving — identical to raw) -------------
    parts = [base]
    if "ages_served" in df.columns:
        parts.append(U.parse_age_range(df["ages_served"], log))
    if "license_issue_date" in df.columns:
        parts.append(U.derive_license_age_days(df["license_issue_date"], log))
    if "license_restrictions" in df.columns:
        parts.append(U.build_keyterm(df["license_restrictions"],
                                     U.RESTRICTION_KEYTERMS, "restriction", "raw"))
    if "special_features" in df.columns:
        # emits amenity keyterm text + ratio_* ints + cleaned `special_features`
        parts.append(U.build_special_features(df["special_features"], "raw", log))
    if "visits_json" in df.columns:
        parts.append(U.build_json_list(df["visits_json"], "visits", "raw", log,
                                       raw_keys=["date", "announced"]))
    if "violations_json" in df.columns:
        # visit_id joined too, but TRAINING_EXCLUDE['raw'] drops it as not-useful
        parts.append(U.build_json_list(df["violations_json"], "violations", "raw", log,
                                       raw_keys=["text", "visit_id"]))

    engineered = pd.concat(parts, axis=1)

    # which="raw" → reuse raw's scaffold / discovered prefixes / exclusions /
    # text preservation; keep_invalid_target=True → retain unrated/invalid rows.
    out = U.finalize(engineered, "raw", scaffold, log, keep_invalid_target=True)

    U.write_output(out, args.output)


if __name__ == "__main__":
    main()
