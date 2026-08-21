from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "ok_data" / "ok_records.csv"
DEFAULT_OUTPUT = HERE / "ok_data" / "ok_records_anonymized.csv"
LOG_FILE = HERE / "ok_privacy_log.txt"
MAP_PATH = HERE.parent / "private" / "provider_id_map_ok.csv"

GRAIN_COL = "provider_id"
STATE_CODE = "ok"

PROTECTED_COLS = ("provider_id", "qr_rating_raw")

PRIVATE_COLS = [
    "provider_name",
    "source_url",
    "licensing_specialist_name",
    "licensing_specialist_phone",
    "contact_name",
    "contact_title",
    "contact_phone",
    "contact_email",
    "address",
    "monitoring_section_text",
    "complaints_section_text",
    "contact_section_text",
    "monitoring_visits_json",
    "complaint_findings_json",
]

JSON_COLS = [
    ("monitoring_visits_json", "monitoring"),
    ("complaint_findings_json", "complaint"),
]

LEAKAGE_COLS: list[str] = [

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


def slug(text):
    text = str(text).strip().lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def records(value):
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def expand_json(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column, prefix in JSON_COLS:
        if column not in df.columns:
            continue
        rows = [records(v) for v in df[column]]
        keys = []
        for row in rows:
            for rec in row:
                if isinstance(rec, dict):
                    for key in rec:
                        if key not in keys:
                            keys.append(key)
        for key in keys:
            values = []
            for row in rows:
                found = [str(rec[key]).strip() for rec in row
                         if isinstance(rec, dict) and rec.get(key) not in (None, '')]
                values.append(' | '.join(found))
            df[f'{prefix}_{slug(key)}'] = values
        log(f"[derive] {column} -> {len(keys)} {prefix}_* column(s)")
    return df


def summarize_visits(df: pd.DataFrame) -> pd.DataFrame:
    column = 'monitoring_visits_json'
    if column not in df.columns:
        return df
    df = df.copy()
    counts, averages = [], []
    for value in df[column]:
        visits = records(value)
        if not visits:
            counts.append('')
            averages.append('')
            continue
        noncompliant = sum(1 for rec in visits
                           if isinstance(rec, dict) and rec.get('non_compliances'))
        ratios = []
        for rec in visits:
            if not isinstance(rec, dict):
                continue
            try:
                compliant = float(rec.get('areas_compliant'))
                total = float(rec.get('areas_total'))
            except (TypeError, ValueError):
                continue
            if total > 0:
                ratios.append(compliant / total * 100)
        counts.append(str(noncompliant))
        averages.append(str(round(sum(ratios) / len(ratios), 2)) if ratios else '')
    df['n_visits_with_noncompliance'] = counts
    df['avg_compliance_pct'] = averages
    log(f"[derive] {column} -> n_visits_with_noncompliance, avg_compliance_pct")
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

    df = expand_json(df)
    df = summarize_visits(df)
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
    log(f"[ok] {before[0]} rows x {before[1]} cols -> {out.shape[1]} cols "
        f"({len(targets)} dropped) -> {args.output}")


if __name__ == "__main__":
    main()
