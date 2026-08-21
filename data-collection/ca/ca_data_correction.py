#!/usr/bin/env python3
"""
clean_crawl_errors.py -- repair scrape artifacts so the CSV parses cleanly.

The crawler occasionally dropped the newline between two consecutive
records, gluing them onto one physical line. Because every record has a
fixed 32 fields and the boundary always fuses one record's LAST column
(`errors`, which is '' / 'not_found' / 'exception') to the next record's
FIRST column (`facility_number`, a run of digits), such a line shows up
with 63 fields (32 + 32 - 1) instead of 32 -- and pandas' C parser dies
with "Expected 32 fields ... saw 63".

This script reads the file with the csv module (which, unlike a naive
split on '\n', still honours RFC-4180 quoted fields that legitimately
span multiple lines), detects the over-long rows, splits them back into
valid 32-field records, and writes a clean CSV that imports with a plain
pd.read_csv() -- no on_bad_lines tricks needed.

Usage:
    python clean_crawl_errors.py facility_records_saved.csv -o facility_records_clean.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

N_FIELDS = 32
# A fused boundary token is errors_A glued to facility_number_B, e.g.
# '410517738', 'not_found410517738', 'exception410517738'. errors is one
# of {'', 'not_found', 'exception'}; facility_number is a run of digits.
BOUNDARY_RE = re.compile(r"^(not_found|exception|)(\d+)$")

# csv has a default field-size cap that some long about_text cells exceed.
csv.field_size_limit(10_000_000)


def split_merged(fields: list[str]) -> list[list[str]] | None:
    """Split a >32-field row back into 32-field records.

    Returns the list of recovered records, or None if the row doesn't
    match the known merge pattern (caller should then drop + report it).
    Handles k records glued together, not just two.
    """
    records: list[list[str]] = []
    rest = fields
    while len(rest) > N_FIELDS:
        boundary = rest[N_FIELDS - 1]  # fused errors_A + facility_number_B
        m = BOUNDARY_RE.match(boundary)
        if not m:
            return None
        errors_a, facility_b = m.group(1), m.group(2)
        records.append(rest[: N_FIELDS - 1] + [errors_a])
        rest = [facility_b] + rest[N_FIELDS:]
    if len(rest) != N_FIELDS:
        return None
    records.append(rest)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="raw crawled CSV")
    ap.add_argument("-o", "--output", type=Path, required=True, help="repaired CSV to write")
    args = ap.parse_args()

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        print("Empty file.", file=sys.stderr)
        return 1

    header = rows[0]
    if len(header) != N_FIELDS:
        print(f"WARNING: header has {len(header)} fields, expected {N_FIELDS}.", file=sys.stderr)

    clean: list[list[str]] = [header]
    n_ok = n_merged = n_recovered = n_dropped = 0

    # enumerate over LOGICAL csv rows (a multi-line quoted field counts once)
    for rownum, row in enumerate(rows[1:], start=2):
        if len(row) == N_FIELDS:
            clean.append(row)
            n_ok += 1
        elif len(row) > N_FIELDS:
            pieces = split_merged(row)
            if pieces:
                clean.extend(pieces)
                n_merged += 1
                n_recovered += len(pieces)
                print(f"  repaired row {rownum}: {len(row)} fields -> {len(pieces)} records")
            else:
                n_dropped += 1
                print(f"  DROPPED row {rownum}: {len(row)} fields, pattern unrecognized", file=sys.stderr)
        else:  # fewer than 32 -- a different kind of corruption
            n_dropped += 1
            print(f"  DROPPED row {rownum}: only {len(row)} fields", file=sys.stderr)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(clean)

    n_records = len(clean) - 1
    print("\n--- repair summary ---")
    print(f"clean rows (32 fields):       {n_ok}")
    print(f"merged rows repaired:         {n_merged}  -> {n_recovered} records recovered")
    print(f"rows dropped (unrecognized):  {n_dropped}")
    print(f"total records written:        {n_records}")
    print(f"output: {args.output}")

    # Prove the output now imports under the strict C parser.
    try:
        import pandas as pd

        df = pd.read_csv(args.output)
        print(f"\nVERIFIED: pandas read {df.shape[0]} rows x {df.shape[1]} cols, no parser error.")
    except Exception as e:  # noqa: BLE001
        print(f"\nWARNING: pandas still failed to parse output: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())