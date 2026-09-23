from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "ne_data" / "ne_records.csv"
DEFAULT_OUTPUT = HERE / "ne_data" / "ne_records_anonymized.csv"
LOG_FILE = HERE / "ne_privacy_log.txt"
MAP_PATH = HERE.parent / "data-private" / "provider_id_map_ne.csv"

GRAIN_COL = "provider_key"
STATE_CODE = "ne"

# step_rating is the target and must survive whatever else is dropped.
PROTECTED_COLS = ("step_rating",)

NONNULL_HELPER = "_source_nonnull_dropped"

# Direct identifiers: the Step Up finder page fields, the DHHS roster columns,
# and slug/facility_id/license_number, which resolve to the provider's public
# finder page or licence record.
PRIVATE_COLS = [
    "facility_name",
    "director",
    "phone",
    "address_raw",
    "street",
    "facility_url",
    "slug",
    "license_number",
    "facility_id",
    "dhhs_objectid",
    "dhhs_full_name",
    "dhhs_owner_manager",
    "dhhs_license_number",
    "dhhs_address",
    "dhhs_address_2",
    "dhhs_phone",
    "dhhs_zip4",
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


def build_provider_key(df: pd.DataFrame) -> pd.DataFrame:
    lic = df["license_number"].astype(str).str.strip()
    fid = df["facility_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    blank = lic.isin(["", "nan", "None"])
    df = df.copy()
    df[GRAIN_COL] = [f"STQ{f}" if b else l
                     for l, f, b in zip(lic, fid, blank)]
    if df[GRAIN_COL].duplicated().any():
        n = int(df[GRAIN_COL].duplicated().sum())
        log(f"[id] note: {n} row(s) share a grain value; they will share a surrogate")
    return df


def guard_id_map(remint: bool) -> None:
    """Refuse to re-mint an existing surrogate map unless asked explicitly."""
    if MAP_PATH.exists() and not remint:
        raise SystemExit(
            f"refusing to overwrite {MAP_PATH}: re-minting draws a fresh "
            f"permutation, so every provider_id already published in "
            f"ne_data/ne_records_cleaned_*.csv would silently point at a "
            f"different provider. Pass --remint to do it deliberately."
        )


def surrogate_ids(df: pd.DataFrame, state: str, seed=None,
                  remint: bool = False) -> pd.DataFrame:
    """Replace the grain value with a surrogate and write the map.

    The labels are a fresh `rng.permutation`, so a second run pairs the same
    set of surrogates with different providers. The map and
    ne_records_anonymized.csv must be written together or not at all.
    """
    guard_id_map(remint)
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
                    help="allow the run to overwrite an existing "
                         "data-private/provider_id_map_ne.csv; without it an "
                         "existing map is a hard error")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False, dtype=str,
                     keep_default_na=False)
    before = df.shape

    targets = drop_targets(df.columns)
    doomed = [c for c, _ in targets]
    missing = [c for c in PRIVATE_COLS + LEAKAGE_COLS if c not in df.columns]

    # The dry run returns before surrogate_ids() rewrites the id map, and
    # prints rather than appending to ne_privacy_log.txt.
    if args.dry_run:
        for col, cls in targets:
            print(f"[{cls}] dropping {col}")
        if missing:
            print(f"[note] {len(missing)} listed column(s) absent from this "
                  f"input: {missing}")
        print(f"\ndry run: would drop {len(targets)} of {before[1]} columns")
        return

    # Fail before the first log line rather than half way through.
    guard_id_map(args.remint)

    populated = sum((df[c].notna() & (df[c].astype(str) != "")).astype(int)
                    for c in doomed)
    df = df.copy()
    df[NONNULL_HELPER] = populated.astype(int).astype(str)
    log(f"[order] recorded {NONNULL_HELPER} over {len(doomed)} removed "
        f"column(s) so the populated-field count stays recoverable")

    df = build_provider_key(df)
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
    log(f"[ne] {before[0]} rows x {before[1]} cols -> {out.shape[1]} cols "
        f"({len(targets)} dropped) -> {args.output}")


if __name__ == "__main__":
    main()
