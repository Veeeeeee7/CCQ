#!/usr/bin/env python3
"""
wi_data_correction.py -- repair the WI crawl output in place.

Three opt-in repairs. None re-runs the crawl or re-mints an identifier; each
rewrites whole files atomically (.tmp then os.replace).

  --add-seed-columns   Insert the four DCF roster fields (application_type,
                       capacity, from_age, to_age) after regulation_type in
                       wi_records.csv and in the file derived from it.

  --strip-stage2       Remove the direct identifiers and per-provider
                       narratives from the derived file (see STAGE2_DROP).

  --merge-retry        Re-sync the derived file from wi_records.csv, column by
                       column, at matching row positions -- run after a
                       `wi_crawler.py --retry-errors` pass so recovered rows
                       reach the derived file without re-minting its ids.

Every mode takes --check for a dry run that writes nothing.

The join is positional, not by key: the derived file holds a surrogate in
provider_location, so row i of one file is row i of the other. Every mode
asserts that alignment on the shared columns before writing.

Usage
    python wi_data_correction.py --add-seed-columns --check
    python wi_data_correction.py --add-seed-columns
    python wi_data_correction.py --strip-stage2 --check
    python wi_data_correction.py --strip-stage2
    python wi_data_correction.py --merge-retry --check
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

# Long JSON / narrative cells blow past the csv module's default field cap.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

HERE = Path(__file__).resolve().parent
DEFAULT_RECORDS = HERE / "wi_data" / "wi_records.csv"
DEFAULT_DERIVED = HERE / "wi_data" / "wi_records_anonymized.csv"
DEFAULT_SEED = HERE / "wi_data" / "wi_seed.csv"

# Stored verbatim as the directory wrote them, like the crawler does.
SEED_CARRY = ["application_type", "capacity", "from_age", "to_age"]
INSERT_AFTER = "regulation_type"

# Must not survive in the derived file: the real DCF registry keys (a
# crosswalk past the surrogate), documents_json (its URLs embed the registry
# ids) and the per-provider narratives, whose prose is itself an identifier.
STAGE2_DROP = [
    "provider_number",
    "location_number",
    "facility_number",
    "documents_json",
    "youngstar_section_text",
    "regulation_section_text",
    "pr_section_text",
]

# Carried unchanged by both files, so any mismatch means they have drifted.
ALIGN_COLS = ["provider_number", "location_number", "facility_number",
              "regulation_type", "errors"]

NEVER_COPY = {"provider_location"}


def flat(value):
    """One record per physical line, matching what the crawler writes."""
    if value is None:
        return ""
    return re.sub(r"[\r\n]+", " ", str(value)).strip()


def read_header(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def iter_rows(path):
    """Stream a CSV as dicts; nothing loads these files whole."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def count_lines(path):
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


class AtomicCsv:
    """Write to <path>.tmp and os.replace() only on a clean exit."""

    def __init__(self, path, columns):
        self.path = str(path)
        self.tmp = self.path + ".tmp"
        self.columns = columns
        self.rows = 0

    def __enter__(self):
        self._fh = open(self.tmp, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(self.columns)
        return self

    def write(self, row):
        self._w.writerow([flat(row.get(c)) for c in self.columns])
        self.rows += 1

    def __exit__(self, exc_type, exc, tb):
        self._fh.close()
        if exc_type is None:
            os.replace(self.tmp, self.path)
        else:
            try:
                os.remove(self.tmp)
            except OSError:
                pass
        return False


def check_alignment(records_csv, derived_csv, cols=None):
    """Assert row i of records_csv is row i of derived_csv. Returns the row count."""
    h1, h2 = read_header(records_csv), read_header(derived_csv)
    cols = [c for c in (cols or ALIGN_COLS) if c in h1 and c in h2]
    if not cols:
        raise SystemExit(
            "the two files share none of the alignment columns "
            f"{ALIGN_COLS} -- cannot prove the positional join; aborting")
    n = 0
    mismatched = []
    for i, (a, b) in enumerate(zip(iter_rows(records_csv), iter_rows(derived_csv))):
        n += 1
        for c in cols:
            if (a.get(c) or "").strip() != (b.get(c) or "").strip():
                mismatched.append((i, c, a.get(c), b.get(c)))
                break
        if len(mismatched) >= 5:
            break
    if mismatched:
        for i, c, x, y in mismatched:
            print(f"  ! row {i}: {c} {x!r} != {y!r}")
        raise SystemExit("the two files are not positionally aligned; aborting")
    n1 = sum(1 for _ in iter_rows(records_csv))
    n2 = sum(1 for _ in iter_rows(derived_csv))
    if n1 != n2:
        raise SystemExit(f"row counts differ: {n1} vs {n2}; aborting")
    print(f"  aligned on {cols}: {n1} rows, 0 mismatches")
    return n1


def seed_map(seed_csv):
    """provider_location -> the SEED_CARRY values, keyed with the crawler's own
    load_seed() so the zero-padded ids match."""
    sys.path.insert(0, str(HERE))
    from wi_crawler import load_seed  # noqa: E402  (same directory, same seed rules)

    seed = load_seed(str(seed_csv))
    missing = [c for c in SEED_CARRY if c not in seed.columns]
    if missing:
        raise SystemExit(f"seed is missing {missing}")
    return {r["provider_location"]: {c: r[c] for c in SEED_CARRY}
            for r in seed.to_dict("records")}


def add_seed_columns(records_csv, derived_csv, seed_csv, check=False):
    print(f"[add-seed-columns] {records_csv}\n                   {derived_csv}")
    n = check_alignment(records_csv, derived_csv)
    lut = seed_map(seed_csv)
    print(f"  seed: {len(lut)} keys")

    h1 = read_header(records_csv)
    if INSERT_AFTER not in h1:
        raise SystemExit(f"{INSERT_AFTER!r} is not a column of {records_csv}")
    already = [c for c in SEED_CARRY if c in h1]
    if already:
        raise SystemExit(f"{records_csv} already has {already}; nothing to do")

    hits = misses = 0
    sample = []
    for row in iter_rows(records_csv):
        key = (row.get("provider_location") or "").strip()
        if key in lut:
            hits += 1
            if len(sample) < 3:
                sample.append((key, lut[key]))
        else:
            misses += 1
            if misses <= 5:
                print(f"  ! no seed row for {key!r}")
    print(f"  joins: {hits} matched, {misses} unmatched, of {n} rows")
    for key, vals in sample:
        print(f"    e.g. {key} -> {vals}")
    if misses:
        raise SystemExit("every row must find its seed row; aborting")

    def new_header(header):
        at = header.index(INSERT_AFTER) + 1
        return header[:at] + SEED_CARRY + header[at:]

    if check:
        print(f"  dry run: {records_csv.name} "
              f"{len(h1)} -> {len(new_header(h1))} columns, "
              f"{derived_csv.name} {len(read_header(derived_csv))} -> "
              f"{len(new_header(read_header(derived_csv)))} columns; "
              f"nothing written")
        return

    # Looked up on the stage-1 key, copied into the same row index of the
    # derived file, whose own key is a surrogate.
    cols1 = new_header(h1)
    with AtomicCsv(records_csv, cols1) as out1:
        for row in iter_rows(records_csv):
            row.update(lut[(row.get("provider_location") or "").strip()])
            out1.write(row)

    cols2 = new_header(read_header(derived_csv))
    with AtomicCsv(derived_csv, cols2) as out2:
        for src, row in zip(iter_rows(records_csv), iter_rows(derived_csv)):
            row.update({c: src.get(c) for c in SEED_CARRY})
            out2.write(row)

    print(f"  wrote {records_csv.name}: {out1.rows} rows x {len(cols1)} cols")
    print(f"  wrote {derived_csv.name}: {out2.rows} rows x {len(cols2)} cols")


def strip_stage2(derived_csv, check=False):
    print(f"[strip-stage2] {derived_csv}")
    header = read_header(derived_csv)
    drop = [c for c in STAGE2_DROP if c in header]
    absent = [c for c in STAGE2_DROP if c not in header]
    keep = [c for c in header if c not in drop]
    for c in drop:
        print(f"  dropping {c}")
    if absent:
        print(f"  already absent: {absent}")
    if not drop:
        print("  nothing to drop")
        return
    if check:
        print(f"  dry run: {len(header)} -> {len(keep)} columns; nothing written")
        return
    with AtomicCsv(derived_csv, keep) as out:
        for row in iter_rows(derived_csv):
            out.write(row)
    print(f"  wrote {derived_csv.name}: {out.rows} rows x {len(keep)} cols")


def merge_retry(records_csv, derived_csv, check=False):
    """Copy every shared column from records_csv into the derived file at the
    same row index. A no-op unless the crawl output has changed."""
    print(f"[merge-retry] {records_csv} -> {derived_csv}")
    h1, h2 = read_header(records_csv), read_header(derived_csv)
    shared = [c for c in h2 if c in h1 and c not in NEVER_COPY]
    print(f"  copying {len(shared)} shared column(s); "
          f"{sorted(set(h2) - set(shared) - NEVER_COPY)} exist only downstream")
    # Not `errors`: a recovered row's errors cell is exactly what changed.
    align = [c for c in ("provider_number", "location_number", "facility_number")
             if c in h1 and c in h2]
    if align:
        check_alignment(records_csv, derived_csv, align)
    else:
        n1 = sum(1 for _ in iter_rows(records_csv))
        n2 = sum(1 for _ in iter_rows(derived_csv))
        if n1 != n2:
            raise SystemExit(f"row counts differ: {n1} vs {n2}; aborting")
        print(f"  ! no shared identifier column left to align on "
              f"(--strip-stage2 has run); relying on the row count: {n1}")

    changed_rows = changed_cells = 0
    per_col = {}
    for src, row in zip(iter_rows(records_csv), iter_rows(derived_csv)):
        diffs = [c for c in shared
                 if flat(src.get(c)) != flat(row.get(c))]
        if diffs:
            changed_rows += 1
            changed_cells += len(diffs)
            for c in diffs:
                per_col[c] = per_col.get(c, 0) + 1
    print(f"  {changed_rows} row(s), {changed_cells} cell(s) differ")
    for c, k in sorted(per_col.items(), key=lambda kv: -kv[1]):
        print(f"    {c}: {k}")
    if check or not changed_rows:
        print("  dry run: nothing written" if check else "  nothing to merge")
        return
    with AtomicCsv(derived_csv, h2) as out:
        for src, row in zip(iter_rows(records_csv), iter_rows(derived_csv)):
            row.update({c: src.get(c) for c in shared})
            out.write(row)
    print(f"  wrote {derived_csv.name}: {out.rows} rows x {len(h2)} cols")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--derived", type=Path, default=DEFAULT_DERIVED)
    ap.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    ap.add_argument("--add-seed-columns", action="store_true",
                    help=f"add {SEED_CARRY} after {INSERT_AFTER}")
    ap.add_argument("--strip-stage2", action="store_true",
                    help="drop the identifier and narrative columns from the "
                         "derived file")
    ap.add_argument("--merge-retry", action="store_true",
                    help="re-sync the derived file from the crawl output, "
                         "positionally")
    ap.add_argument("--check", action="store_true",
                    help="report what would change and write nothing")
    args = ap.parse_args()

    if not any((args.add_seed_columns, args.strip_stage2, args.merge_retry)):
        ap.error("pick at least one of --add-seed-columns / --strip-stage2 / "
                 "--merge-retry")

    before = {p: (len(read_header(p)), count_lines(p))
              for p in (args.records, args.derived)}

    if args.add_seed_columns:
        add_seed_columns(args.records, args.derived, args.seed, args.check)
    if args.strip_stage2:
        strip_stage2(args.derived, args.check)
    if args.merge_retry:
        merge_retry(args.records, args.derived, args.check)

    print("\nsummary")
    for p, (cols, lines) in before.items():
        now_cols, now_lines = len(read_header(p)), count_lines(p)
        print(f"  {p.name}: {cols} -> {now_cols} cols, "
              f"{lines} -> {now_lines} physical lines")
        if lines != now_lines:
            raise SystemExit(f"{p.name} changed its physical line count "
                             f"({lines} -> {now_lines}); a record must stay on "
                             f"one line")


if __name__ == "__main__":
    main()
