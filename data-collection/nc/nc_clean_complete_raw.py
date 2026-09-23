"""
nc_clean_complete_raw.py — Build the `complete_raw` dataset: the same
text-preserving feature engineering as nc_clean_raw.py, but keeping rows whose
star rating is invalid/unrated. qr_rating is nullable Int64; placeholders such
as 'GS 110-106' become <NA> while the row survives.

complete_raw and complete_full are row-aligned with each other, not with
raw/full.

    python nc_clean_complete_raw.py
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

    # --- shared early steps (identical in all four scripts) ------------------
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

    # --- per-field builders (text-preserving) --------------------------------
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

    out = U.finalize(engineered, "raw", scaffold, log, keep_invalid_target=True)

    U.write_output(out, args.output)


if __name__ == "__main__":
    main()
