from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "nc_data" / "nc_records.csv"
DEFAULT_OUTPUT = HERE / "nc_data" / "nc_records_anonymized.csv"
LOG_FILE = HERE / "nc_privacy_log.txt"
MAP_PATH = HERE.parent / "data-private" / "provider_id_map_nc.csv"

GRAIN_COL = "facility_id"
STATE_CODE = "nc"

PROTECTED_COLS = ("facility_id", "star_rating")

PRIVATE_COLS = [
    "facility_name",
    "operator_name",
    "address",
    "phone",
    "email",
    "facility_url",
    # Whole-page text dumps: each restates the facility name, street address,
    # phone, owner's name and licence number. `operating_hours` can hold the
    # same page text.
    "star_section_text",
    "visits_section_text",
    "details_section_text",
    "operating_hours",
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
    """Replace the grain with a random surrogate and write the id map.

    np.random.default_rng(None) draws a FRESH permutation on every call, so
    overwriting an existing map re-mints every provider_id and must be asked
    for with --remint.
    """
    if MAP_PATH.exists() and not remint:
        raise SystemExit(
            f"{MAP_PATH} already exists. Re-running would draw a new "
            f"permutation and change every provider_id in the release. "
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
                    help="allow an existing provider id map to be overwritten "
                         "with a fresh permutation (changes every id)")
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

    # --dry-run must return above surrogate_ids(), which rewrites the id map.
    if args.dry_run:
        print(f"\ndry run: would drop {len(targets)} of {before[1]} columns")
        return

    df = surrogate_ids(df, STATE_CODE, remint=args.remint)

    out = df.drop(columns=[c for c, _ in targets], errors="ignore")

    assert len(out) == len(df), "anonymization must not change the row count"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    log(f"[nc] {before[0]} rows x {before[1]} cols -> {out.shape[1]} cols "
        f"({len(targets)} dropped) -> {args.output}")


if __name__ == "__main__":
    main()
