"""co_anonymize.py -- anonymization for Colorado.

Derives the has_documented_* flags from the five licensing-history narratives,
surrogates provider_id, and drops the private / target-leakage columns.

Two in-place modes exist so that a repair to co_records.csv can be propagated
WITHOUT re-minting provider ids (surrogate_ids draws a fresh permutation on
every run, which would renumber every provider):

    python co_anonymize.py --replay flags   # recompute the 5 flags only
    python co_anonymize.py --replay all     # + copy every shared column across

Both patch the existing co_records_anonymized.csv positionally -- same rows,
same order, same provider_id (co_records.csv row i is
co_records_anonymized.csv row i).
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "co_data" / "co_records.csv"
DEFAULT_OUTPUT = HERE / "co_data" / "co_records_anonymized.csv"
LOG_FILE = HERE / "co_privacy_log.txt"
MAP_PATH = HERE.parent / "data-private" / "provider_id_map_co.csv"

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

# Each history accordion has its OWN "nothing to report" sentinel. The generic
# disclaimer that prefixes some sections is NOT a sentinel.
_NO_HISTORY_RES: dict[str, re.Pattern] = {
    "inspection_report_text":
        re.compile(r"No Inspections? reported in the last 3 years", re.IGNORECASE),
    "complaints_text":
        re.compile(r"No Complaints? reported in the last 3 years", re.IGNORECASE),
    "stage_ii_text":
        re.compile(r"No Stage II Investigations? reported in the last 3 years",
                   re.IGNORECASE),
    "injury_investigations_text":
        re.compile(r"No Injur(?:y|ies) reported in the last 3 years", re.IGNORECASE),
    "adverse_actions_text":
        re.compile(r"No Actions Reported", re.IGNORECASE),
}

# A dated entry line, "m/d/yyyy <report id> <Outcome> Link to ROI". Used only
# for the drift tripwire below, never to set a flag: each section is truncated
# at 3,000 characters, which cuts the dated rows off some adverse_actions_text
# cells that really do have entries.
_HISTORY_ENTRY_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")

# errors values that mean "this provider's page was never read". The flags are
# blank (NA) on these rows: nothing is known about the provider's licensing
# history, which is not the same as "nothing happened".
_NO_PAGE_ERRORS = {"not_found", "exception", "ambiguous_no_match", "no_search_box"}

PRIVATE_COLS = [
    "provider_name",
    # governing_body: the operating organisation's name, unique to a single
    # provider on most of its values -- a per-provider quasi-identifier.
    "governing_body",
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


_LOG_TO_FILE = True


def log(message: str, path: Path = LOG_FILE) -> None:
    if _LOG_TO_FILE:
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
    """Five narratives -> five has_documented_* flags.

    True  -- the section has text and does NOT carry its sentinel.
    False -- the section carries its sentinel ("No Complaints reported ...").
    ""    -- the provider's page was never read (id_mismatch, or no page at
             all), so nothing is known. Blank reads back as NA.
    """
    if "errors" in df.columns:
        err = df["errors"].fillna("").astype(str)
        unknown = err.apply(
            lambda v: "id_mismatch" in v or v.strip() in _NO_PAGE_ERRORS)
    else:
        unknown = pd.Series(False, index=df.index)

    made = 0
    for col in LICENSING_HISTORY_COLS:
        flag_col = _HISTORY_FLAG_NAMES[col]
        if col not in df.columns:
            df[flag_col] = ""
            continue
        sentinel = _NO_HISTORY_RES[col]
        values = df[col].where(~unknown, "")
        known = values.apply(_has_value)
        documented = [
            k and not bool(sentinel.search(str(v))) for k, v in zip(known, values)
        ]
        df[flag_col] = ["" if not k else ("True" if d else "False")
                        for k, d in zip(known, documented)]
        # Drift tripwire: a section that matches neither the sentinel nor a
        # dated entry. Expected only for truncated adverse_actions_text tables;
        # a jump means the wording has changed and the sentinel regex above
        # needs re-checking.
        odd = sum(1 for k, v in zip(known, values)
                  if k and not sentinel.search(str(v))
                  and not _HISTORY_ENTRY_RE.search(str(v)))
        n_true = sum(1 for f in df[flag_col] if f == "True")
        n_false = sum(1 for f in df[flag_col] if f == "False")
        n_na = len(df) - n_true - n_false
        log(f"[derive] {flag_col}: True {n_true} / False {n_false} / NA {n_na}"
            f"{f'  [tripwire] {odd} section(s) with neither sentinel nor date' if odd else ''}")
        made += 1
    log(f"[derive] {made} narrative(s) -> has_documented_* flag(s) before "
        f"dropping the prose")
    return df


def refuse_unless_remint(remint: bool) -> None:
    """A plain run re-mints every provider id. Make that deliberate."""
    if MAP_PATH.exists() and not remint:
        raise SystemExit(
            f"refusing to overwrite {MAP_PATH}.\n"
            f"  surrogate_ids() draws a FRESH permutation on every run, so this "
            f"would renumber every provider and invalidate every existing "
            f"id reference.\n"
            f"  To propagate a repair to co_records.csv instead, patch the "
            f"existing file in place:\n"
            f"      python co_anonymize.py --replay flags   # the 5 has_documented_* columns\n"
            f"      python co_anonymize.py --replay all     # every shared column + the flags\n"
            f"  To re-mint deliberately, pass --remint.")


def surrogate_ids(df: pd.DataFrame, state: str, seed=None,
                  remint: bool = False) -> pd.DataFrame:
    refuse_unless_remint(remint)
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


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    """Never leave a half-written file behind: write beside the target, then
    rename over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def replay(input_path: Path, output_path: Path, scope: str) -> None:
    """Patch an EXISTING anonymized file in place from a refreshed
    co_records.csv.

    scope="flags"  recompute the five has_documented_* columns only.
    scope="all"    also copy every column the two files share (including
                   `errors`, which marks a mismatched row), then recompute
                   the flags.

    provider_id is never touched, rows are matched positionally, and the row
    count must be unchanged.
    """
    src = pd.read_csv(input_path, low_memory=False, dtype=str, keep_default_na=False)
    dst = pd.read_csv(output_path, low_memory=False, dtype=str, keep_default_na=False)
    if len(src) != len(dst):
        raise SystemExit(f"replay must not change the row count: "
                         f"{input_path} has {len(src)} rows, "
                         f"{output_path} has {len(dst)}")
    before_ids = dst[GRAIN_COL].tolist()
    before_cols = dst.shape[1]

    src = derive_history_flags(src)

    copied = []
    if scope == "all":
        skip = {GRAIN_COL} | {c for c, _ in drop_targets(src.columns)}
        for col in dst.columns:
            if col in skip or col not in src.columns:
                continue
            if col in _HISTORY_FLAG_NAMES.values():
                continue
            changed = int((dst[col].values != src[col].values).sum())
            dst[col] = src[col].values          # positional, same row order
            if changed:
                copied.append(f"{col} ({changed})")
        log(f"[replay] copied {len(copied)} shared column(s) with changed "
            f"cells: {copied if copied else 'none'}")

    for flag_col in _HISTORY_FLAG_NAMES.values():
        if flag_col not in src.columns:
            raise SystemExit(f"{flag_col} missing after derivation -- aborting")
        if flag_col in dst.columns:
            moved = int((dst[flag_col].values != src[flag_col].values).sum())
            log(f"[replay] {flag_col}: {moved} cell(s) change")
        dst[flag_col] = src[flag_col].values

    # Re-apply the current drop list so a column added to PRIVATE_COLS after
    # the file was written leaves on the same pass.
    targets = drop_targets(dst.columns)
    for col, cls in targets:
        log(f"[{cls}] dropping {col} (replay)")
    dst = dst.drop(columns=[c for c, _ in targets], errors="ignore")

    assert dst[GRAIN_COL].tolist() == before_ids, \
        "replay must not touch provider_id"
    _write_atomic(dst, output_path)
    log(f"[replay:{scope}] {len(dst)} rows, {before_cols} -> {dst.shape[1]} "
        f"cols, provider_id untouched -> {output_path}")


def main() -> None:
    global _LOG_TO_FILE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the drop list and exit without writing "
                         "anything -- no output file, no provider-id map")
    ap.add_argument("--replay", choices=("flags", "all"),
                    help="patch the EXISTING --output in place from --input "
                         "instead of rebuilding it: 'flags' refreshes the five "
                         "has_documented_* columns, 'all' also copies every "
                         "shared column across. provider_id is never re-minted.")
    ap.add_argument("--remint", action="store_true",
                    help="allow surrogate_ids() to overwrite an existing "
                         "provider-id map. Renumbers every provider.")
    args = ap.parse_args()

    if args.dry_run:
        # Nothing on disk may change on a dry run -- not the output, not the
        # id map, not the privacy log.
        _LOG_TO_FILE = False
        df = pd.read_csv(args.input, low_memory=False, dtype=str,
                         keep_default_na=False)
        targets = drop_targets(df.columns)
        for col, cls in targets:
            print(f"[{cls}] would drop {col}")
        missing = [c for c in PRIVATE_COLS + LEAKAGE_COLS if c not in df.columns]
        if missing:
            print(f"[note] {len(missing)} listed column(s) absent from this "
                  f"input: {missing}")
        print(f"\ndry run: would drop {len(targets)} of {df.shape[1]} columns; "
              f"nothing written")
        return

    if args.replay:
        replay(args.input, args.output, args.replay)
        return

    # Checked again inside surrogate_ids(); checked here first so a refused run
    # reads nothing, derives nothing and writes no log line.
    refuse_unless_remint(args.remint)

    df = pd.read_csv(args.input, low_memory=False, dtype=str,
                     keep_default_na=False)
    before = df.shape

    df = derive_history_flags(df)

    df = surrogate_ids(df, STATE_CODE, remint=args.remint)

    targets = drop_targets(df.columns)
    for col, cls in targets:
        log(f"[{cls}] dropping {col}")
    missing = [c for c in PRIVATE_COLS + LEAKAGE_COLS if c not in df.columns]
    if missing:
        log(f"[note] {len(missing)} listed column(s) absent from this input: "
            f"{missing}")

    out = df.drop(columns=[c for c, _ in targets], errors="ignore")

    assert len(out) == len(df), "anonymization must not change the row count"

    _write_atomic(out, args.output)
    log(f"[co] {before[0]} rows x {before[1]} cols -> {out.shape[1]} cols "
        f"({len(targets)} dropped) -> {args.output}")


if __name__ == "__main__":
    main()
