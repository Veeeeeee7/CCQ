"""
md_crawler.py — Maryland EXCELS "Find a Program" crawler.

findaprogram.marylandexcels.org's Vue front end calls a plain JSON/CSV API
(no auth, no cookies):

  GET /api/fap/referencedata                        — accreditation/achievement
                                                        and program-type lookups
  GET /api/fap/count?county=<county>                 — count of matching programs
  GET /api/fap/csv/programs?county=<county>&sort=2   — full CSV export for a county

The CSV export is unpaginated, so the crawler loops over Maryland's 23
counties + Baltimore City, hits the CSV endpoint once each and concatenates.
The export's columns are kept as-is; _source_county (the query) and
_fetched_at are prepended.

County spellings are the portal's ("Saint Mary's", not MSDE's "St. Mary's";
"Baltimore" and "Baltimore City" are separate). An unknown spelling answers 0,
so the crawler calls /count first and stops on a zero/None count, cross-checks
the parsed row count against it, and at the end asserts that every seeded
county produced rows.

    python md_crawler.py --limit 3             # smoke test, 3 counties
    python md_crawler.py                       # full run (~24 requests)
    python md_crawler.py --resume              # continue an interrupted run

Resume is opt-in: --resume skips COUNTIES (not providers) already present in
the output's _source_county column; without it the crawler refuses to append
to a non-empty output.

Deps: pip install requests pandas
"""
from __future__ import annotations

import argparse
import io
import os
import random
import time
import traceback
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_URL = 'https://findaprogram.marylandexcels.org/api/fap'
COUNT_URL = f'{BASE_URL}/count'
CSV_URL = f'{BASE_URL}/csv/programs'

LOG_FILE = 'md_crawler_log.txt'

# Maryland's 23 counties + Baltimore City, in the portal's own spellings. The
# seed sits beside the scripts because md_data/ is gitignored.
SEED_PATH = "md_seed.csv"


def _load_counties(path=SEED_PATH):
    """The counties to sweep, one per row of the provided seed."""
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as fh:
        return [r["county"].strip() for r in _csv.DictReader(fh)
                if r["county"].strip()]


COUNTIES = _load_counties()

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/120.0.0.0 Safari/537.36')


def create_log_file(path=LOG_FILE):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def _get(session, url, params, timeout, retries=2, backoff=2.0):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise last_exc


def fetch_count(session, county, timeout=20):
    resp = _get(session, COUNT_URL, {'county': county}, timeout)
    return resp.json().get('data')


def fetch_csv(session, county, timeout=30):
    resp = _get(session, CSV_URL, {'county': county, 'sort': 2}, timeout)
    resp.encoding = resp.encoding or 'utf-8'
    return resp.text


def load_completed_counties(output_csv):
    if not output_csv or not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv, dtype=str, usecols=['_source_county'])
        return set(df['_source_county'].dropna().str.strip())
    except Exception as e:
        log(f'Could not read existing {output_csv}: {e}')
        return set()


def append_rows(csv_text, county, output_csv):
    """Append every row of one county's CSV export, tagged with crawl
    metadata, to output_csv. The source's column names/order are untouched."""
    df = pd.read_csv(io.StringIO(csv_text), dtype=str)
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed')]  # trailing blank-header column
    if 'County' in df.columns:
        unexpected = set(df['County'].dropna().unique()) - {county}
        if unexpected:
            log(f'  ! {county}: row(s) with unexpected County value(s) {unexpected} '
                f'— possible API fuzzy-matching or boundary overlap, not dropped.')
    df.insert(0, '_source_county', county)
    df.insert(1, '_fetched_at', datetime.now(timezone.utc).isoformat(timespec='seconds'))
    parent = os.path.dirname(output_csv)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    df.to_csv(output_csv, mode='a', header=not exists, index=False)
    return len(df)


def assert_full_coverage(output_csv, counties):
    """Every seeded county must have produced at least one row (catches a
    county whose fetch raised and was logged-and-skipped)."""
    found = load_completed_counties(output_csv)
    missing = [c for c in counties if c not in found]
    if missing:
        log(f'INCOMPLETE: {len(missing)} of {len(counties)} seeded county/ies '
            f'produced no rows: {missing}. {output_csv} is NOT a complete '
            f'Maryland crawl — re-run those counties before using it.')
        raise SystemExit(3)
    log(f'Coverage OK: all {len(counties)} seeded counties are present in '
        f'{output_csv}.')


def crawler(counties, output_csv='md_data/md_records.csv',
            start_index=0, limit=None, delay_range=(1.5, 3.0), resume=False):
    completed = load_completed_counties(output_csv)
    if completed and not resume:
        log(f'REFUSING to append: {output_csv} already holds '
            f'{len(completed)} county/ies. Move the file aside for a fresh '
            f'crawl, or pass --resume to continue an interrupted one.')
        raise SystemExit(4)
    if completed:
        log(f'Resuming: {len(completed)} counties already in {output_csv} — skipping.')
    end = len(counties) if limit is None else min(len(counties), start_index + limit)

    session = requests.Session()
    session.headers.update({'User-Agent': UA, 'Accept': 'application/json, text/csv'})

    index = start_index
    try:
        for index in range(start_index, end):
            county = counties[index]
            if county in completed:
                continue

            expected = None
            try:
                expected = fetch_count(session, county)
            except Exception:
                log(f'[{index}] (count check failed for {county}, trying the CSV anyway)')
            else:
                if not expected:
                    # A zero count is a spelling the portal does not know;
                    # continuing would append nothing and mark the county done.
                    log(f'[{index}] FATAL: /count for {county!r} returned '
                        f'{expected!r}. The portal does not know that spelling '
                        f'— check it against findaprogram.marylandexcels.org '
                        f'and fix {SEED_PATH}. Nothing was appended for this '
                        f'county; the crawl is stopping rather than shipping a '
                        f'silent hole.')
                    raise SystemExit(2)

            try:
                csv_text = fetch_csv(session, county)
                n = append_rows(csv_text, county, output_csv)
                completed.add(county)
                tag = 'OK'
                if expected is not None and n != expected:
                    tag = f'MISMATCH (got {n} rows, /count said {expected})'
                log(f'[{index}] {tag}: {county} — {n} rows')
            except Exception:
                log(f'[{index}] EXCEPTION for {county}:')
                log(traceback.format_exc())

            if index < end - 1:
                delay = random.uniform(*delay_range)
                log(f'  ...sleeping {delay:.1f}s')
                time.sleep(delay)
    except Exception:
        log(f'CRASHED at index {index}')
        log(traceback.format_exc())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Maryland EXCELS Find-a-Program crawler.')
    ap.add_argument('--output', default='md_data/md_records.csv')
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--delay-min', type=float, default=1.5)
    ap.add_argument('--delay-max', type=float, default=3.0)
    ap.add_argument('--resume', action='store_true',
                    help='continue an interrupted crawl by skipping counties '
                         'already present in the output')
    args = ap.parse_args()

    create_log_file()
    log(f'Counties: {len(COUNTIES)}.')
    crawler(COUNTIES, output_csv=args.output, start_index=args.start_index,
            limit=args.limit, delay_range=(args.delay_min, args.delay_max),
            resume=args.resume)
    if args.start_index == 0 and args.limit is None:
        assert_full_coverage(args.output, COUNTIES)
