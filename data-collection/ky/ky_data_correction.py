"""
ky_data_correction.py -- merge a refetch sidecar into the detail columns of
the rows whose detail fetch failed, in place.

Row i of `ky_data/ky_records_anonymized.csv` is row i of
`ky_data/ky_records.csv` (anonymization never sorts or drops a row, and
re-running it would re-mint every provider_id), so the repair is applied to
BOTH files at the same row positions. It is not an upsert: the row set and the
row order are invariants, asserted before and after.

`ky_records.csv` is CRLF (csv.DictWriter) and `ky_records_anonymized.csv` is
LF (pandas); each is written back with its own line terminator.

Usage:
    python ky_data_correction.py --fix-detail --dry-run
    python ky_data_correction.py --fix-detail
    python ky_data_correction.py --fix-detail --sidecar ky_data/ky_records_refetch.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

try:
    from ky_crawler import DETAIL_COLS, KEY_COL
except ImportError as exc:                       # pragma: no cover
    raise SystemExit(
        f'could not import ky_crawler ({exc}). Run this from inside ky/, and '
        f'note that ky_crawler.py imports playwright even though this script '
        f'does not use it: pip install -r ../requirements.txt')

csv.field_size_limit(10_000_000)

RECORDS_PATH = os.path.join('ky_data', 'ky_records.csv')
ANON_PATH = os.path.join('ky_data', 'ky_records_anonymized.csv')
SIDECAR_PATH = os.path.join('ky_data', 'ky_refetch.csv')

ERROR_COL = 'errors'

# Each file keeps the line terminator it was born with.
TERMINATORS = {RECORDS_PATH: '\r\n', ANON_PATH: '\n'}

# showKICCSCapacity is inserted next to Capacity if the file predates it.
ANCHOR_COL = 'Capacity'


def read_table(path):
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames or [], list(reader)


def write_table(path, columns, rows, terminator):
    """Whole-file rewrite through a temp file: the header may gain a column,
    and a half-written records file is unrecoverable."""
    tmp = f'{path}.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns,
                                lineterminator=terminator)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, '') for c in columns})
    os.replace(tmp, path)


def insert_after(columns, new_column, anchor):
    if new_column in columns:
        return list(columns)
    out = list(columns)
    index = out.index(anchor) + 1 if anchor in out else len(out)
    out.insert(index, new_column)
    return out


def is_error(row):
    return bool((row.get(ERROR_COL) or '').strip())


def load_sidecar(path):
    """CLR -> the best row for it. A --resume run appends a retry after the
    failed attempt, so later rows win and a success always beats a failure."""
    _, rows = read_table(path)
    best = {}
    for row in rows:
        clr = (row.get(KEY_COL) or '').strip()
        if not clr:
            continue
        if clr in best and is_error(row) and not is_error(best[clr]):
            continue
        best[clr] = row
    return best, len(rows)


def fix_detail(sidecar_path, dry_run, backup):
    records_cols, records = read_table(RECORDS_PATH)
    anon_cols, anon = read_table(ANON_PATH)
    if len(records) != len(anon):
        raise SystemExit(f'{len(records)} records vs {len(anon)} anonymized '
                         f'rows -- the positional join is void, stop here')
    before_rows = len(records)
    clr_before = [r.get(KEY_COL) for r in records]
    surrogate_before = [r.get(KEY_COL) for r in anon]
    owed_before = sum(1 for r in records if is_error(r))

    fetched, sidecar_rows = load_sidecar(sidecar_path)
    print(f'{sidecar_path}: {sidecar_rows} row(s), '
          f'{len(fetched)} distinct provider(s), '
          f'{sum(1 for r in fetched.values() if not is_error(r))} usable')

    patched = skipped_missing = skipped_failed = 0
    per_column = {c: 0 for c in DETAIL_COLS}
    for i, record in enumerate(records):
        if not is_error(record):
            continue                        # never touch a row that succeeded
        new = fetched.get((record.get(KEY_COL) or '').strip())
        if new is None:
            skipped_missing += 1
            continue
        if is_error(new):
            skipped_failed += 1             # still failed -> leave the error
            continue
        for column in DETAIL_COLS:
            value = new.get(column, '')
            if value is None:
                value = ''
            if value != (record.get(column) or ''):
                per_column[column] += 1
            record[column] = anon[i][column] = value
        record[ERROR_COL] = anon[i][ERROR_COL] = ''
        patched += 1

    records_cols = insert_after(records_cols, 'showKICCSCapacity', ANCHOR_COL)
    anon_cols = insert_after(anon_cols, 'showKICCSCapacity', ANCHOR_COL)

    owed_after = sum(1 for r in records if is_error(r))
    print(f'\npatched   {patched} row(s)')
    print(f'  no sidecar row     {skipped_missing}')
    print(f'  sidecar still bad  {skipped_failed}')
    print(f'errors non-empty   {owed_before} -> {owed_after}')
    print(f'rows               {before_rows} -> {len(records)} '
          f'(must not change)')
    print('cells changed per column:')
    for column in DETAIL_COLS:
        print(f'  {column:32} {per_column[column]}')

    assert len(records) == before_rows == len(anon), 'row count changed'
    assert [r.get(KEY_COL) for r in records] == clr_before, \
        'ProviderCLRNumber sequence changed in ky_records.csv'
    assert [r.get(KEY_COL) for r in anon] == surrogate_before, \
        'the surrogate id sequence changed in ky_records_anonymized.csv'

    if dry_run:
        print('\ndry run: nothing written')
        return 0
    if backup:
        for path in (RECORDS_PATH, ANON_PATH):
            shutil.copy2(path, f'{path}.pre-merge')
        print(f'\nbackups: {RECORDS_PATH}.pre-merge, {ANON_PATH}.pre-merge')
    write_table(RECORDS_PATH, records_cols, records, TERMINATORS[RECORDS_PATH])
    write_table(ANON_PATH, anon_cols, anon, TERMINATORS[ANON_PATH])
    print(f'wrote {RECORDS_PATH} ({len(records_cols)} cols) and '
          f'{ANON_PATH} ({len(anon_cols)} cols)')
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fix-detail', action='store_true',
                    help='merge a refetch sidecar into the failed rows')
    ap.add_argument('--sidecar', default=SIDECAR_PATH,
                    help='ky_refetch.py sidecar, or a ky_crawler.py output '
                         'from the browser fallback (both are accepted)')
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would change and write nothing')
    ap.add_argument('--no-backup', action='store_true',
                    help='skip the .pre-merge copies')
    args = ap.parse_args()

    if not args.fix_detail:
        ap.error('nothing to do: pass --fix-detail')
    for path in (RECORDS_PATH, ANON_PATH):
        if not os.path.exists(path):
            raise SystemExit(f'{path} not found -- run this from inside ky/')

    status = 0
    if args.fix_detail:
        if not os.path.exists(args.sidecar):
            raise SystemExit(f'{args.sidecar} not found -- run ky_refetch.py '
                             f'first, or pass --sidecar')
        status |= fix_detail(args.sidecar, args.dry_run, not args.no_backup)
    return status


if __name__ == '__main__':
    sys.exit(main())
