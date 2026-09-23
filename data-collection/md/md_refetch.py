"""md_refetch.py — Maryland EXCELS: collect licensed age ranges and licensed
capacity, which the CSV export does not carry, and patch them into the
stage-1 files.

The portal's paginated listing, GET /api/fap/search, carries both for every
program in the same county sweep:

    ccApprovedAge1 .. ccApprovedAge7, ccApprovedAge0m23m   licensed ages
    licensedCapacity                                        licensed places

This adds columns; it never adds, removes or reorders a row, and it never
touches an existing cell.

    --fetch  writes a SIDECAR file and nothing else; resumable.
    --merge  reads that sidecar and patches the stage-1 files in place.

    python md_refetch.py --fetch                     # ~107 requests, ~5 min
    python md_refetch.py --fetch --resume            # continue after a stop
    python md_refetch.py --fetch --limit 1 --count 5 --max-programs 5
                                                     # smoke test, 2 requests
    python md_refetch.py --merge --dry-run           # report, write nothing
    python md_refetch.py --merge                     # patch both stage-1 files

PRIVACY
    /api/fap/search items carry program name, phone, street address and
    lat/long. Only the nine fields below plus the program's id are extracted;
    the raw JSON is never written to disk, not even as a cache.

THE MERGE IS POSITIONAL
    md_records.csv is joined to the sidecar on its 'Program ID' column.
    md_records_anonymized.csv holds the same rows in the same order with a
    surrogate in that column, so it receives the identical per-row values by
    position. Both files are length-checked and written via .tmp + rename.

    licensedCapacity is 0 for every program whose page shows no capacity
    block (all Public Prekindergarten rows, plus a few centres): "not
    published", so it is written as blank. Likewise the page renders its
    "Licensed Age Ranges" card only when at least one band is set, so all
    eight flags false is written as eight blanks; a program with some bands
    set shows the card, and the bands it omits are a genuine No.

Deps: pip install requests pandas   (pandas only via md_crawler's import)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import requests

# Same transport and User-Agent as the county sweep.
from md_crawler import UA, _get, _load_counties, fetch_count

csv.field_size_limit(10_000_000)

SEARCH_URL = 'https://findaprogram.marylandexcels.org/api/fap/search'

RECORDS = 'md_data/md_records.csv'
ANONYMIZED = 'md_data/md_records_anonymized.csv'
SIDECAR = 'md_data/md_licensed_ages.csv'
CHECKPOINT = 'md_data/md_refetch_checkpoint.json'
LOG_FILE = 'md_refetch_log.txt'

KEY_COL = 'Program ID'          # the records file's own id; 'license' in the API
STAMP_COL = '_ages_fetched_at'  # crawl metadata

# Labels and order of the portal's "Licensed Age Ranges" tiles.
FLAG_TO_COL = [
    ('ccApprovedAge1',    'Licensed 6 weeks-17 mos'),
    ('ccApprovedAge2',    'Licensed 18 mos-23 mos'),
    ('ccApprovedAge0m23m', 'Licensed 0 mos-23 mos'),
    ('ccApprovedAge3',    'Licensed 2 years'),
    ('ccApprovedAge4',    'Licensed 3 years'),
    ('ccApprovedAge5',    'Licensed 4 years'),
    ('ccApprovedAge6',    'Licensed 5 yrs preschool'),
    ('ccApprovedAge7',    'Licensed 5 yrs-15 yrs'),
]
AGE_COLS = [col for _, col in FLAG_TO_COL]
CAPACITY_COL = 'License Capacity'
NEW_COLS = AGE_COLS + [CAPACITY_COL]
SIDECAR_COLS = [KEY_COL] + NEW_COLS + [STAMP_COL]

MAX_CONSECUTIVE_5XX = 3


def log(message, file=None):
    # Resolved at call time: main() rewrites LOG_FILE from --log.
    with open(file or LOG_FILE, 'a', encoding='utf-8') as fh:
        fh.write(message + '\n')
    print(message)


def extract(item):
    """The nine values kept from one listing item; everything else is dropped
    here. All eight age flags false are blanked (see the module docstring)."""
    flags = {col: bool(item.get(flag)) for flag, col in FLAG_TO_COL}
    published = any(flags.values())
    row = {col: (('Yes' if value else 'No') if published else '')
           for col, value in flags.items()}
    try:
        capacity = int(item.get('licensedCapacity') or 0)
    except (TypeError, ValueError):
        capacity = 0
    row[CAPACITY_COL] = '' if capacity <= 0 else str(capacity)
    return row


def _status(exc):
    response = getattr(exc, 'response', None)
    return getattr(response, 'status_code', None)


def fetch_county(session, county, count, delay_range, budget=None):
    """Page /api/fap/search for one county. Returns (rows, complete);
    complete is False only when `budget` ran out mid-county."""
    rows, offset, pages, seen_5xx = {}, 0, 0, 0
    while True:
        try:
            response = _get(session, SEARCH_URL,
                            {'county': county, 'count': count,
                             'offset': offset, 'sort': 2}, timeout=60)
            seen_5xx = 0
        except Exception as exc:
            status = _status(exc)
            if status and 500 <= status < 600:
                seen_5xx += 1
                log(f'    ! {county} offset={offset}: HTTP {status} '
                    f'({seen_5xx} in a row after retries)')
                if seen_5xx >= MAX_CONSECUTIVE_5XX:
                    raise SystemExit(
                        f'{MAX_CONSECUTIVE_5XX} consecutive 5xx from the '
                        f'portal — stopping. The sidecar and checkpoint are '
                        f'intact; re-run with --resume when it recovers.')
                time.sleep(30)
                continue
            raise
        payload = response.json()
        items = payload.get('data') if isinstance(payload, dict) else payload
        if items is None:
            raise SystemExit(f'unexpected response shape for {county!r}: '
                             f'keys={sorted(payload)[:8]}')
        pages += 1
        stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
        for item in items:
            key = str(item.get('license') or '').strip()
            if not key:
                continue
            if key in rows:
                continue
            row = extract(item)
            row[KEY_COL] = key
            row[STAMP_COL] = stamp
            rows[key] = row
            if budget is not None and len(rows) >= budget:
                return list(rows.values()), False
        if len(items) < count:
            return list(rows.values()), True
        offset += count
        time.sleep(random.uniform(*delay_range))


def load_checkpoint(path):
    if not os.path.exists(path):
        return {'counties': []}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def save_checkpoint(path, state):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def append_sidecar(path, rows):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=SIDECAR_COLS, lineterminator='\n')
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def do_fetch(args):
    counties = _load_counties()
    state = load_checkpoint(args.checkpoint) if args.resume else {'counties': []}
    done = set(state.get('counties', []))

    sidecar_exists = (os.path.exists(args.sidecar)
                      and os.path.getsize(args.sidecar) > 0)
    if sidecar_exists and not args.resume:
        raise SystemExit(
            f'{args.sidecar} already exists. Pass --resume to continue the run '
            f'that wrote it, or delete it to start over. Refusing to append to '
            f'a sidecar this run did not start.')
    if args.resume and done:
        log(f'Resuming: {len(done)} county/ies already in {args.sidecar}.')

    end = len(counties) if args.limit is None else min(
        len(counties), args.start_index + args.limit)
    session = requests.Session()
    session.headers.update({'User-Agent': UA, 'Accept': 'application/json'})

    collected = 0
    for index in range(args.start_index, end):
        county = counties[index]
        if county in done:
            continue
        budget = None
        if args.max_programs is not None:
            budget = args.max_programs - collected
            if budget <= 0:
                log(f'--max-programs {args.max_programs} reached; stopping.')
                break

        expected = None
        try:
            expected = fetch_count(session, county)
        except Exception:
            log(f'[{index}] (count check failed for {county}, fetching anyway)')
        else:
            if not expected:
                raise SystemExit(
                    f'[{index}] FATAL: /count for {county!r} returned '
                    f'{expected!r}. That spelling is not one the portal knows '
                    f'— fix md_seed.csv rather than fetching a hole.')
        time.sleep(random.uniform(*args.delay_range))

        rows, complete = fetch_county(session, county, args.count,
                                      args.delay_range, budget)
        append_sidecar(args.sidecar, rows)
        collected += len(rows)
        tag = 'OK' if complete else 'PARTIAL (budget)'
        note = ''
        if complete and expected is not None and len(rows) != expected:
            note = f' [/count said {expected}]'
        log(f'[{index}] {tag}: {county} — {len(rows)} programs{note}')

        if complete:
            done.add(county)
            state['counties'] = [c for c in counties if c in done]
            state['updated_at'] = datetime.now(timezone.utc).isoformat(
                timespec='seconds')
            save_checkpoint(args.checkpoint, state)
        if index < end - 1:
            time.sleep(random.uniform(*args.delay_range))

    log(f'Fetched {collected} program(s) this run; {len(done)} of '
        f'{len(counties)} counties complete -> {args.sidecar}')
    if len(done) == len(counties):
        log('All seeded counties complete. Next: '
            'python md_refetch.py --merge --dry-run')


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f'{path} is empty')
    return rows[0], rows[1:]


def write_rows(path, header, body):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(body)
    os.replace(tmp, path)


def load_sidecar(path):
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in SIDECAR_COLS if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f'{path} is missing column(s) {missing}')
        keyed, duplicates = {}, 0
        for row in reader:
            key = (row[KEY_COL] or '').strip()
            if not key:
                continue
            if key in keyed:
                duplicates += 1
                continue
            keyed[key] = row
    return keyed, duplicates


def do_merge(args):
    keyed, duplicates = load_sidecar(args.sidecar)
    log(f'sidecar: {len(keyed)} program(s)'
        + (f' ({duplicates} duplicate id(s) ignored)' if duplicates else ''))

    header, body = read_rows(args.records)
    anon_header, anon_body = read_rows(args.anonymized)
    added = NEW_COLS + [STAMP_COL]

    clash = sorted(set(added) & (set(header) | set(anon_header)))
    if clash:
        raise SystemExit(f'already patched: {clash} present. Nothing written.')
    if KEY_COL not in header:
        raise SystemExit(f'{args.records} has no {KEY_COL} column')
    if len(body) != len(anon_body):
        raise SystemExit(f'{len(body)} rows in {args.records} vs '
                         f'{len(anon_body)} in {args.anonymized} — the '
                         f'positional join is void, nothing written.')

    key_at = header.index(KEY_COL)
    blank = ['' for _ in added]
    values, matched = [], 0
    unmatched_ids = []
    for row in body:
        key = (row[key_at] if key_at < len(row) else '').strip()
        found = keyed.get(key)
        if found is None:
            values.append(blank)
            unmatched_ids.append(key)
        else:
            values.append([found[c] for c in added])
            matched += 1

    rate = matched / len(body) if body else 0.0
    extra = sorted(set(keyed) - {(r[key_at] if key_at < len(r) else '').strip()
                                for r in body})
    log(f'matched {matched} of {len(body)} row(s) ({rate:.2%}); '
        f'{len(unmatched_ids)} row(s) with no listing entry; '
        f'{len(extra)} sidecar program(s) not in the crawl')
    if unmatched_ids:
        log(f'  unmatched record ids (closed since the crawl?): '
            f'{unmatched_ids[:20]}{" ..." if len(unmatched_ids) > 20 else ""}')
    if extra:
        log(f'  listing-only ids (new or from a county the crawl missed): '
            f'{extra[:20]}{" ..." if len(extra) > 20 else ""}')
    if rate < args.min_match:
        raise SystemExit(f'match rate {rate:.2%} is below --min-match '
                         f'{args.min_match:.2%}. Nothing written — check the '
                         f'sidecar before forcing this through.')

    if args.dry_run:
        capacity_at = added.index(CAPACITY_COL)
        with_capacity = sum(1 for v in values if v[capacity_at])
        log(f'dry run: would add {len(added)} column(s) to both files '
            f'({len(header)} -> {len(header) + len(added)} and '
            f'{len(anon_header)} -> {len(anon_header) + len(added)}); '
            f'{with_capacity} row(s) would carry a capacity. Nothing written.')
        return

    write_rows(args.records, header + added,
               [row + value for row, value in zip(body, values)])
    write_rows(args.anonymized, anon_header + added,
               [row + value for row, value in zip(anon_body, values)])

    for path, was in ((args.records, len(header)),
                      (args.anonymized, len(anon_header))):
        new_header, new_body = read_rows(path)
        assert len(new_header) == was + len(added), path
        assert len(new_body) == len(body), path
    log(f'patched {args.records} ({len(header)} -> {len(header) + len(added)} '
        f'cols) and {args.anonymized} '
        f'({len(anon_header)} -> {len(anon_header) + len(added)} cols), '
        f'{len(body)} rows each, same order.')


def main():
    global LOG_FILE
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fetch', action='store_true',
                    help='page /api/fap/search per county into the sidecar')
    ap.add_argument('--merge', action='store_true',
                    help='patch the sidecar into the two stage-1 files')
    ap.add_argument('--resume', action='store_true',
                    help='with --fetch: skip counties named in the checkpoint')
    ap.add_argument('--dry-run', action='store_true',
                    help='with --merge: report the join and write nothing')
    ap.add_argument('--limit', type=int, default=None,
                    help='with --fetch: only the first N seeded counties')
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--max-programs', type=int, default=None,
                    help='with --fetch: stop after N programs (smoke test)')
    ap.add_argument('--count', type=int, default=100,
                    help='listing page size (100 is verified against the API)')
    ap.add_argument('--delay-min', type=float, default=1.5)
    ap.add_argument('--delay-max', type=float, default=3.0)
    ap.add_argument('--min-match', type=float, default=0.97,
                    help='with --merge: refuse below this join rate')
    ap.add_argument('--sidecar', default=SIDECAR)
    ap.add_argument('--checkpoint', default=CHECKPOINT)
    ap.add_argument('--records', default=RECORDS)
    ap.add_argument('--anonymized', default=ANONYMIZED)
    ap.add_argument('--log', default=LOG_FILE)
    args = ap.parse_args()

    if args.fetch == args.merge:
        ap.error('pass exactly one of --fetch or --merge')
    args.delay_range = (args.delay_min, args.delay_max)
    LOG_FILE = args.log

    log(f'--- {"fetch" if args.fetch else "merge"} '
        f'{datetime.now(timezone.utc).isoformat(timespec="seconds")} ---')
    if args.fetch:
        do_fetch(args)
    else:
        do_merge(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
