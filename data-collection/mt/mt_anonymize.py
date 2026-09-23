from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

DEFAULT_INPUT = HERE / "mt_data" / "mt_records.csv"
DEFAULT_OUTPUT = HERE / "mt_data" / "mt_records_anonymized.csv"
LOG_FILE = HERE / "mt_privacy_log.txt"
MAP_PATH = HERE.parent / "data-private" / "provider_id_map_mt.csv"

GRAIN_COL = "provider_number"
STATE_CODE = "mt"

PROTECTED_COLS = ("provider_number", "star_level")

NAME_COL = "program_name"
KEYTERM_PREFIX = "name"

NAME_KEYTERMS = [
    "Head Start",
    "Early Head Start",
    "Montessori",
    "YMCA",
    "Preschool",
    "Academy",
    "Learning Center",
    "Child Development Center",
    "Christian",
    "Lutheran",
    "Cooperative",
]

# Abbreviations a program name may use instead of the keyterm, matched as a
# whole slug token ("(EHS)" sets name_early_head_start, "Ehsan" does not).
NAME_KEYTERM_ALIASES = {
    "Early Head Start": ["EHS"],
}

PRIVATE_COLS = [
    "program_name",
    "provider_name",
    "street",
    "phone",
    "latitude",
    "longitude",
    "sf_id",
]

LEAKAGE_COLS: list[str] = []
LEAKAGE_PREFIXES = (
    "rating_status", "rating_pending", "rating_age", "rating_award",
    "rating_expir", "rating_renew", "rating_effective",
    "award_date", "expiration_date", "days_until_rating",
)


def slug(text):
    text = str(text).strip().lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def log(message: str, path: Path = LOG_FILE) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(message + "\n")
    print(message)


def decompose_program_name(df: pd.DataFrame) -> pd.DataFrame:
    if NAME_COL not in df.columns:
        log(f"[note] {NAME_COL} absent; skipping the keyterm decomposition")
        return df

    cell_slugs = ["_" + slug(v) + "_" if isinstance(v, str) and v else ""
                  for v in df[NAME_COL]]
    made = []
    for keyterm in NAME_KEYTERMS:
        tokens = ["_" + slug(t) + "_"
                  for t in [keyterm] + list(NAME_KEYTERM_ALIASES.get(keyterm, ()))]
        column = f"{KEYTERM_PREFIX}_{slug(keyterm)}"
        df[column] = [keyterm if any(t in cs for t in tokens) else ""
                      for cs in cell_slugs]
        made.append(column)
    log(f"[derive] {NAME_COL} -> {len(made)} name_* keyterm column(s) "
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
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be dropped; write nothing at all")
    ap.add_argument("--remint", action="store_true",
                    help="allow a fresh provider_id permutation to overwrite an "
                         "existing id map (see the refusal message below)")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False, dtype=str,
                     keep_default_na=False)
    before = df.shape

    targets = drop_targets(df.columns)

    # --dry-run must return above surrogate_ids(), which rewrites the id map.
    if args.dry_run:
        for col, cls in targets:
            print(f"[{cls}] would drop {col}")
        print(f"\ndry run: would drop {len(targets)} of {before[1]} columns; "
              f"nothing written, {MAP_PATH.name} untouched")
        return

    if MAP_PATH.exists() and not args.remint:
        raise SystemExit(
            f"refusing to overwrite {MAP_PATH}.\n"
            f"surrogate_ids() draws a fresh random permutation, so re-running "
            f"this script re-mints EVERY provider_id in the release and breaks "
            f"the link to the ids already published. Montana's corrections are "
            f"applied in place with\n"
            f"    python mt_data_correction.py --sync-anonymized\n"
            f"instead. Pass --remint only if you really intend a new id space."
        )

    df = decompose_program_name(df)

    df = surrogate_ids(df, STATE_CODE)

    for col, cls in targets:
        log(f"[{cls}] dropping {col}")

    out = df.drop(columns=[c for c, _ in targets], errors="ignore")
    assert len(out) == len(df), "anonymization must not change the row count"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    log(f"[mt] {before[0]} rows x {before[1]} cols -> {out.shape[1]} cols "
        f"({len(targets)} dropped) -> {args.output}")


if __name__ == "__main__":
    main()
