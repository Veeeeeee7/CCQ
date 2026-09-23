"""
nc_clean_full.py — Build the `full` dataset (strictly numeric/boolean +
provider_id) from the NC records, valid 1–5 ratings only.

  load (as strings) → drop source-error rows → dollar-strip → grain check →
  drop NON_FEATURE_COLS → per-field builders (numeric/boolean only) → finalize.

    python nc_clean_full.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import nc_clean_utils as U


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("nc_data/nc_records_anonymized.csv"))
    parser.add_argument("--output", type=Path, default=Path("nc_data/nc_records_cleaned_full.csv"))
    parser.add_argument("--scaffold", type=Path,
                        default=Path(__file__).resolve().parent / "nc_columns.json")
    parser.add_argument("--log", type=Path, default=Path("nc_clean_full.log"))
    args = parser.parse_args()

    log = U.ParseLog(args.log)
    scaffold = json.loads(Path(args.scaffold).read_text())

    print(f"[full] loading {args.input}")
    df = pd.read_csv(args.input, low_memory=False, dtype=str)  # preserve leading zeros

    # --- shared early steps (identical in all four scripts) ------------------
    df = U.drop_error_rows(df, log)
    df = U.strip_dollars(df)
    U.check_grain_unique(df, U.ID_COL, log)
    df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns],
                 errors="ignore")

    # --- base: id, target, clean passthrough numerics ------------------------
    base = pd.DataFrame(index=df.index)
    base[U.ID_COL] = df[U.ID_COL]
    base[U.TARGET_COL] = df[U.TARGET_COL]
    for c in ("licensed_capacity", "num_visits"):
        if c in df.columns:
            base[c] = pd.to_numeric(df[c], errors="coerce")

    # --- per-field builders (numeric / boolean only) -------------------------
    parts = [base]
    if "facility_type" in df.columns:
        parts.append(U.build_categorical_onehot(df["facility_type"], "facility_type"))
    if "ages_served" in df.columns:
        parts.append(U.parse_age_range(df["ages_served"], log))
    if "license_issue_date" in df.columns:
        parts.append(U.derive_license_age_days(df["license_issue_date"], log))
    if "license_restrictions" in df.columns:
        parts.append(U.build_keyterm(df["license_restrictions"],
                                     U.RESTRICTION_KEYTERMS, "restriction", "full"))
    if "special_features" in df.columns:
        parts.append(U.build_special_features(df["special_features"], "full", log))
    if "visits_json" in df.columns:
        parts.append(U.build_json_list(df["visits_json"], "visits", "full", log,
                                       categorical_value_keys=["announced"]))
    if "violations_json" in df.columns:
        parts.append(U.build_json_list(df["violations_json"], "violations", "full", log,
                                       fetched=df["visits_json"].apply(U._has_value)))

    engineered = pd.concat(parts, axis=1)

    out = U.finalize(engineered, "full", scaffold, log)

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
