"""
ky_anonymize.py — drop identifying columns from ky_records.csv and replace
ProviderCLRNumber with a random surrogate id.

Usage:
    python ky_anonymize.py --dry-run
    python ky_anonymize.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "ky_data" / "ky_records.csv"
DEFAULT_OUTPUT = HERE / "ky_data" / "ky_records_anonymized.csv"
LOG_FILE = HERE / "ky_privacy_log.txt"
MAP_PATH = HERE.parent / "data-private" / "provider_id_map_ky.csv"

GRAIN_COL = "ProviderCLRNumber"
STATE_CODE = "ky"

# Never dropped, whatever else matches. NumberOfStars is the target.
PROTECTED_COLS = ("NumberOfStars",)

PRIVATE_COLS = [
    "ProviderId",
    "ProviderName",
    "PhoneNumber",
    "LocationAddressLine1",
    "LocationAddressLine2",
    "AddressLatitude",
    "AddressLongitude",
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


def surrogate_ids(df: pd.DataFrame, state: str, seed=None,
                  remint: bool = False) -> pd.DataFrame:
    """Mint a fresh surrogate for every provider and write the map.

    The permutation is unseeded, so a second run gives every provider a
    different provider_id and loses the old map, the only link back to the
    real licence numbers; overwriting an existing map needs --remint.
    """
    if MAP_PATH.exists() and not remint:
        raise SystemExit(
            f"{MAP_PATH} already exists. Re-running would mint a NEW random "
            f"permutation and overwrite it, changing every provider_id in the "
            f"release. "
            f"Pass --remint if that is genuinely what you want.")

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
    ap.add_argument("--remint", action="store_true",
                    help="allow overwriting an existing provider id map, "
                         "changing every provider_id in the release")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False, dtype=str,
                     keep_default_na=False)
    before = df.shape

    targets = drop_targets(df.columns)
    for col, cls in targets:
        log(f"[{cls}] dropping {col}")
    missing = [c for c in PRIVATE_COLS + LEAKAGE_COLS if c not in df.columns]
    if missing:
        log(f"[note] {len(missing)} listed column(s) absent from this input: "
            f"{missing}")

    # Return before surrogate_ids(): a dry run must not re-mint the id map.
    if args.dry_run:
        print(f"\ndry run: would drop {len(targets)} of {before[1]} columns; "
              f"no id map written")
        return

    df = surrogate_ids(df, STATE_CODE, remint=args.remint)

    out = df.drop(columns=[c for c, _ in targets], errors="ignore")

    assert len(out) == len(df), "anonymization must not change the row count"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    log(f"[ky] {before[0]} rows x {before[1]} cols -> {out.shape[1]} cols "
        f"({len(targets)} dropped) -> {args.output}")


if __name__ == "__main__":
    main()
