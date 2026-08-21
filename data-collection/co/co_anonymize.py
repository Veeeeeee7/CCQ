from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "co_data" / "co_records.csv"
DEFAULT_OUTPUT = HERE / "co_data" / "co_records_anonymized.csv"
LOG_FILE = HERE / "co_privacy_log.txt"
MAP_PATH = HERE.parent / "private" / "provider_id_map_co.csv"

GRAIN_COL = "provider_id"
STATE_CODE = "co"

PROTECTED_COLS = ("provider_id", "quality_rating")

LICENSING_HISTORY_COLS = [
    "inspection_report_text", "complaints_text", "stage_ii_text",
    "injury_investigations_text", "adverse_actions_text",
]
_HISTORY_FLAG_NAMES = {
    "inspection_report_text": "has_documented_inspection_history",
    "complaints_text": "has_documented_complaint",
    "stage_ii_text": "has_documented_stage_ii",
    "injury_investigations_text": "has_documented_injury_investigation",
    "adverse_actions_text": "has_documented_adverse_action",
}

_NO_HISTORY_RE = re.compile(
    r"information is not currently available on the system.*?"
    r"public file review[^.]*\.?",
    flags=re.IGNORECASE | re.DOTALL,
)

PRIVATE_COLS = [
    "provider_name",
    "street_address",
    "phone",
    "website",
    "license_number_on_site",
    "description",
    "detail_url",
    "inspection_report_text",
    "complaints_text",
    "stage_ii_text",
    "injury_investigations_text",
    "adverse_actions_text",
]

LEAKAGE_COLS: list[str] = [
    "rating_on_site",
    "award_date",
    "expiration_date",
]
LEAKAGE_PREFIXES = (
    "rating_status", "rating_pending", "rating_age", "rating_award",
    "rating_expir", "rating_renew", "rating_effective",
    "award_date", "expiration_date", "days_until_rating",
)


def log(message: str, path: Path = LOG_FILE) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(message + "\n")
    print(message)


def _has_value(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    s = str(v).strip()
    return s != "" and s.lower() not in ("nan", "na")


def derive_history_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "errors" in df.columns:
        mismatch = df["errors"].apply(
            lambda v: _has_value(v) and "id_mismatch" in str(v))
    else:
        mismatch = pd.Series(False, index=df.index)

    made = 0
    for col in LICENSING_HISTORY_COLS:
        flag_col = _HISTORY_FLAG_NAMES[col]
        if col not in df.columns:
            df[flag_col] = ""
            continue
        values = df[col].where(~mismatch, "")
        stripped = values.apply(
            lambda v: _NO_HISTORY_RE.sub("", str(v)).strip() if _has_value(v) else "")
        flag = values.apply(_has_value) & stripped.apply(_has_value)
        df[flag_col] = ["True" if f else "False" for f in flag]
        made += 1
    log(f"[derive] {made} narrative(s) -> has_documented_* flag(s) before "
        f"dropping the prose")
    return df


def surrogate_ids(df: pd.DataFrame, state: str, seed=None) -> pd.DataFrame:
    values = df[GRAIN_COL].astype(str)
    distinct = list(dict.fromkeys(values))
    rng = np.random.default_rng(seed)
    labels = rng.permutation(len(distinct))
    lookup = {v: str(labels[i]) for i, v in enumerate(distinct)}

    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"surrogate_provider_id": [lookup[v] for v in distinct],
                   "source_provider_id": distinct}).to_csv(MAP_PATH, index=False)

    df = df.copy()
    df[GRAIN_COL] = [lookup[v] for v in values]
    log(f"[id] surrogated {len(distinct)} provider id(s) in {GRAIN_COL}; "
        f"map -> {MAP_PATH.name}")
    return df


def drop_targets(columns) -> list[tuple[str, str]]:
    out = []
    for col in columns:
        if col in PROTECTED_COLS:
            continue
        if col in PRIVATE_COLS:
            out.append((col, "P"))
        elif col in LEAKAGE_COLS or col.startswith(LEAKAGE_PREFIXES):
            out.append((col, "L"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the drop list and exit without writing")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False, dtype=str,
                     keep_default_na=False)
    before = df.shape

    df = derive_history_flags(df)

    df = surrogate_ids(df, STATE_CODE)

    targets = drop_targets(df.columns)
    for col, cls in targets:
        log(f"[{cls}] dropping {col}")
    missing = [c for c in PRIVATE_COLS + LEAKAGE_COLS if c not in df.columns]
    if missing:
        log(f"[note] {len(missing)} listed column(s) absent from this input: "
            f"{missing}")

    if args.dry_run:
        print(f"\ndry run: would drop {len(targets)} of {before[1]} columns")
        return

    out = df.drop(columns=[c for c, _ in targets], errors="ignore")

    assert len(out) == len(df), "anonymization must not change the row count"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    log(f"[co] {before[0]} rows x {before[1]} cols -> {out.shape[1]} cols "
        f"({len(targets)} dropped) -> {args.output}")


if __name__ == "__main__":
    main()
