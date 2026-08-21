"""
md_crawler.py — Maryland EXCELS "Find a Program" crawler.

Unlike the other four states, this doesn't scrape rendered DOM at all.
Phase 1 recon (network_log.json) caught findaprogram.marylandexcels.org's Vue
front end calling a plain JSON/CSV API, and testing it directly confirmed:

  GET /api/fap/referencedata                        — accreditation/achievement
                                                        and program-type lookups
  GET /api/fap/count?county=<county>                 — count of matching programs
  GET /api/fap/csv/programs?county=<county>&sort=2   — full CSV export for a county

No auth/cookies required. The CSV export has no count/offset params (unlike
the paginated JSON /search endpoint) and returned all 62 Allegany rows in one
response, so this crawler is just: loop over Maryland's 23 counties + independent
Baltimore City, hit the CSV endpoint once each, concatenate. No Playwright, no
browser — this is the simplest of the five states' crawlers by a wide margin.

The CSV's own columns are kept as-is (Program Name, Program Type, Program ID,
..., Quality Rating, LIC/STF/ACR/APV/TQF/AVR/DAP/ADM indicator sub-scores,
Achievements, Accreditations, ...) — renaming/typing happens in md_clean_raw.py,
not here. Two crawl-metadata columns are prepended: _source_county (the query
that produced the row) and _fetched_at.

A few counties' exact server-side spelling is unconfirmed (apostrophes in
"Prince George's"/"Queen Anne's", "St. Mary's" vs "Saint Mary's", "Baltimore"
vs "Baltimore City" both existing as separate jurisdictions). Rather than
trust the hardcoded list blindly, this crawler calls /count for each county
first and logs a loud warning (not a crash) for anything that comes back
zero/None, and cross-checks the parsed CSV row count against that /count —
so a misspelled county is easy to spot in md_crawler_log.txt and fix in
COUNTIES below, rather than silently missing data.

    python md_crawler.py --limit 3                      # smoke test, 3 counties
    python md_crawler.py                                # full run (~24 requests)

Resume-safe: re-running skips counties already present in the output CSV's
_source_county column.

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

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

BASE_URL = 'https://findaprogram.marylandexcels.org/api/fap'
COUNT_URL = f'{BASE_URL}/count'
CSV_URL = f'{BASE_URL}/csv/programs'

LOG_FILE = 'md_crawler_log.txt'

# Maryland's 23 counties + independent Baltimore City. "Allegany" is confirmed
# working against the live API (Phase 1). The rest are the spellings MSDE's
# own monthly EXCELS reports use, but are UNCONFIRMED against this specific
# API — that's what the /count pre-check in crawler() is for.
SEED_PATH = "md_data/md_seed.csv"


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


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def create_log_file(path=LOG_FILE):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# resume + output helpers (rectangular CSV, append-per-county)
# ---------------------------------------------------------------------------

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
    """Parse the CSV export's raw text and append every row, tagged with
    crawl metadata, to output_csv. The source's own column names/order are
    left untouched — cleaning/renaming happens in md_clean_raw.py."""
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


# ---------------------------------------------------------------------------
# main crawler
# ---------------------------------------------------------------------------

def crawler(counties, output_csv='md_data/md_records.csv',
            start_index=0, limit=None, delay_range=(1.5, 3.0)):
    completed = load_completed_counties(output_csv)
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
                if not expected:
                    log(f'[{index}] WARNING: /count for {county!r} returned '
                        f'{expected!r} — double-check this county\'s spelling '
                        f'against findaprogram.marylandexcels.org directly.')
            except Exception:
                log(f'[{index}] (count check failed for {county}, trying the CSV anyway)')

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
    args = ap.parse_args()

    create_log_file()
    log(f'Counties: {len(COUNTIES)}.')
    crawler(COUNTIES, output_csv=args.output, start_index=args.start_index,
            limit=args.limit, delay_range=(args.delay_min, args.delay_max))
