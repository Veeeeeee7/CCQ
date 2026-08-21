from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

DEFAULT_INPUT = HERE / "sc_data" / "sc_records.csv"
DEFAULT_OUTPUT = HERE / "sc_data" / "sc_records_anonymized.csv"
LOG_FILE = HERE / "sc_privacy_log.txt"
MAP_PATH = HERE.parent / "private" / "provider_id_map_sc.csv"

GRAIN_COL = "Permit Number"
STATE_CODE = "sc"

PROTECTED_COLS = ("Permit Number", "ABC Level")

NAME_COL = "Provider Name"
PERMIT_COL = "Permit Number"
ZIP_COL = "Zip"

PRIVATE_COLS = [
    "Provider Name",
    "Operator",
    "Street",
    "Phone",
]

LEAKAGE_COLS: list[str] = []
LEAKAGE_PREFIXES = (
    "rating_status", "rating_pending", "rating_age", "rating_award",
    "rating_expir", "rating_renew", "rating_effective",
    "award_date", "expiration_date", "days_until_rating",
)


def _clean_str(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    s = str(value).strip()
    return '' if s.lower() in ('nan', 'none', 'n/a', '-') else s


def normalize_zip(value):
    s = _clean_str(value)
    m = re.match(r'(\d{5})', s)
    return m.group(1) if m else np.nan


def slug(text):
    text = str(text).strip().lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def log(message: str, path: Path = LOG_FILE) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(message + "\n")
    print(message)


def mint_exempt_ids(df: pd.DataFrame) -> pd.DataFrame:
    permits = df[PERMIT_COL].map(_clean_str)
    blank = permits == ""
    if not blank.any():
        log("[id] no blank permit numbers; nothing to mint")
        return df

    zips = df[ZIP_COL].map(normalize_zip)

    def _mint(i):
        z = zips.iat[i] if isinstance(zips.iat[i], str) and zips.iat[i] else "nozip"
        return f"EXEMPT-{z}-{slug(df[NAME_COL].iat[i])}"

    minted = {i: _mint(i) for i in range(len(df)) if blank.iat[i]}
    values = pd.Series(minted)
    if values.duplicated().any():
        dupes = sorted(values[values.duplicated(keep=False)].unique())
        raise ValueError(
            f"Synthesized exempt provider_id is not unique: {dupes[:5]}. "
            f"Extend the key (e.g. add street) before proceeding.")

    df = df.copy()
    df[PERMIT_COL] = [minted.get(i, permits.iat[i]) for i in range(len(df))]
    log(f"[id] minted {len(minted)} EXEMPT-* id(s) from the provider name "
        f"before dropping it")
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False, dtype=str,
                     keep_default_na=False)
    before = df.shape

    df = mint_exempt_ids(df)

    df = surrogate_ids(df, STATE_CODE)

    targets = drop_targets(df.columns)
    for col, cls in targets:
        log(f"[{cls}] dropping {col}")

    if args.dry_run:
        print(f"\ndry run: would drop {len(targets)} of {before[1]} columns")
        return

    out = df.drop(columns=[c for c, _ in targets], errors="ignore")
    assert len(out) == len(df), "anonymization must not change the row count"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    log(f"[sc] {before[0]} rows x {before[1]} cols -> {out.shape[1]} cols "
        f"({len(targets)} dropped) -> {args.output}")


if __name__ == "__main__":
    main()
