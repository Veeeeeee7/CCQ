"""md_anonymize.py — Maryland EXCELS stage 2: surrogate the grain column and
drop the private columns.

    python md_anonymize.py --dry-run     # report only; writes NOTHING
    python md_anonymize.py               # write the anonymized file
    python md_anonymize.py --remint      # ALSO redraw provider_id_map_md.csv

A fresh permutation renames every provider, so --dry-run writes nothing, and
a real run reuses an existing provider_id_map_md.csv unless --remint is
passed. The map is the only link back to the real providers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "md_data" / "md_records.csv"
DEFAULT_OUTPUT = HERE / "md_data" / "md_records_anonymized.csv"
LOG_FILE = HERE / "md_privacy_log.txt"
MAP_PATH = HERE.parent / "data-private" / "provider_id_map_md.csv"

GRAIN_COL = "Program ID"
STATE_CODE = "md"

PROTECTED_COLS = ("Program ID", "Quality Rating")

PRIVATE_COLS = [
    "Program Name",
    "Doing Business As",
    "Phone",
    "Alternate Phone",
    "Address",
    "Website",
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
    """Replace the grain values with their surrogates.

    If provider_id_map_md.csv already exists it is REUSED, so the ids stay
    stable across runs; pass remint=True to draw a new permutation and
    overwrite it, which renames every provider downstream.
    """
    values = df[GRAIN_COL].astype(str)
    distinct = list(dict.fromkeys(values))

    if MAP_PATH.exists() and not remint:
        table = pd.read_csv(MAP_PATH, dtype=str)
        lookup = dict(zip(table["source_provider_id"],
                          table["surrogate_provider_id"]))
        unmapped = [v for v in distinct if v not in lookup]
        if unmapped:
            raise SystemExit(
                f"{MAP_PATH.name} has no surrogate for {len(unmapped)} grain "
                f"value(s) (e.g. {unmapped[:3]}). The input has grown since the "
                f"map was minted. Re-run with --remint if you really intend to "
                f"redraw every provider_id."
            )
        log(f"[id] reused {MAP_PATH.name} for {len(distinct)} provider id(s) "
            f"(pass --remint to redraw)")
    else:
        rng = np.random.default_rng(seed)
        labels = rng.permutation(len(distinct))
        lookup = {v: str(labels[i]) for i, v in enumerate(distinct)}
        MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"surrogate_provider_id": [lookup[v] for v in distinct],
                      "source_provider_id": distinct}).to_csv(MAP_PATH,
                                                              index=False)
        log(f"[id] MINTED {len(distinct)} fresh provider id(s) in {GRAIN_COL}; "
            f"map -> {MAP_PATH.name}")

    df = df.copy()
    df[GRAIN_COL] = [lookup[v] for v in values]
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
                    help="redraw provider_id_map_md.csv from scratch; this "
                         "renames every provider")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False, dtype=str,
                     keep_default_na=False)
    before = df.shape

    # The dry run must return before surrogate_ids(), which may mint the map.
    targets = drop_targets(df.columns)
    missing = [c for c in PRIVATE_COLS + LEAKAGE_COLS if c not in df.columns]
    if args.dry_run:
        for col, cls in targets:
            print(f"[{cls}] would drop {col}")
        if missing:
            print(f"[note] {len(missing)} listed column(s) absent from this "
                  f"input: {missing}")
        print(f"\ndry run: would drop {len(targets)} of {before[1]} columns; "
              f"nothing written, {MAP_PATH.name} untouched")
        return

    df = surrogate_ids(df, STATE_CODE, remint=args.remint)

    for col, cls in targets:
        log(f"[{cls}] dropping {col}")
    if missing:
        log(f"[note] {len(missing)} listed column(s) absent from this input: "
            f"{missing}")

    out = df.drop(columns=[c for c, _ in targets], errors="ignore")

    assert len(out) == len(df), "anonymization must not change the row count"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    log(f"[md] {before[0]} rows x {before[1]} cols -> {out.shape[1]} cols "
        f"({len(targets)} dropped) -> {args.output}")


if __name__ == "__main__":
    main()
