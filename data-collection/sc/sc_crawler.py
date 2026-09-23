"""
sc_crawler.py — build the South Carolina ABC Quality per-provider dataset.

abcquality.org/provider-search is fully server-rendered and exposes a plain CSV
export:

    GET https://abcquality.org/provider-search/excel/?county=<County>

Despite the UI calling it "Export to Excel", the response is `text/csv`. It is
not paginated (the HTML view pages at 8 rows/page; the CSV returns the county's
full result set), so the 46 SC counties are the entire state in 46 requests. No
browser, no API keys, no anti-bot friction observed.

CSV COLUMNS
    Provider Name, Permit Type, Permit Number, Operator, Facility Type, Street,
    City, State, Zip, County, Phone, ABC Level, Last ABC Inspection Date,
    Capacity, Head Start, First Steps, Breastfeeding Friendly, Sleep Safe

  `Permit Number` is the state licensing/registration number; `ABC Level` is
  the rating (A+ / A / B+ / B / C; "P" = score pending; empty = unrated).

QUIRKS
  * `Permit Number` is NOT the id used in detail-page URLs. The site keys detail
    pages on an internal id (/provider/4800/union-day-school/) that is absent
    from the CSV. Detail pages are not crawled, so that id never enters here.
  * Exempt providers (`Permit Type` = "Not Licensed (Exempt)") have a blank
    Permit Number but ARE rated. They are kept; see the `exempt` flag.
  * The HTML results table prints "Child Care Center" as the Facility Type of
    every row. The CSV's `Facility Type` code (A / C / EAA / ...) is the real one.
  * `Permit Number` must stay a string, so everything is read with `dtype=str`.

SELF-CHECK
  For each county the crawler also reads the HTML view's "Displaying N
  Providers" headline and compares N with the CSV row count. A mismatch is
  logged loudly (e.g. a silently truncated export).

Usage:
    python sc_crawler.py --counties Calhoun,McCormick     # smoke test (2 counties)
    python sc_crawler.py --limit 3                        # first 3 counties
    python sc_crawler.py                                  # full 46-county sweep
    python sc_crawler.py --merge-only                     # rebuild sc_records.csv from cache
    python sc_crawler.py --counties Richland --force      # re-pull one county

Outputs:
    sc_data/county_raw/<county>.csv   per-county raw CSV exactly as served
    sc_data/sc_records.csv            merged, deduped, one row per provider
    sc_crawler_log.txt                run log

Deps: pip install requests pandas beautifulsoup4
"""

import argparse
import io
import os
import random
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://abcquality.org'
EXCEL_PATH = '/provider-search/excel/'   # returns text/csv despite the name
HTML_PATH = '/provider-search/'          # used only for the row-count self-check

OUT_DIR = 'sc_data'
RAW_DIR = os.path.join(OUT_DIR, 'county_raw')
MERGED_PATH = os.path.join(OUT_DIR, 'sc_records.csv')
LOG_FILE = 'sc_crawler_log.txt'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

REQUEST_TIMEOUT = 45

# The 46 counties, spelled exactly as the site's county <select> lists them
# (e.g. "McCormick").
SEED_PATH = "sc_data/sc_seed.csv"


def _load_counties(path=SEED_PATH):
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as fh:
        return [r["county"].strip() for r in _csv.DictReader(fh)
                if r["county"].strip()]


SC_COUNTIES = _load_counties()
assert len(SC_COUNTIES) == 46, f'expected 46 SC counties, got {len(SC_COUNTIES)}'
assert len(set(SC_COUNTIES)) == 46, 'duplicate county in SC_COUNTIES'

# Checked on every response so an upstream schema change fails loudly.
EXPECTED_COLUMNS = [
    'Provider Name', 'Permit Type', 'Permit Number', 'Operator',
    'Facility Type', 'Street', 'City', 'State', 'Zip', 'County', 'Phone',
    'ABC Level', 'Last ABC Inspection Date', 'Capacity', 'Head Start',
    'First Steps', 'Breastfeeding Friendly', 'Sleep Safe',
]

EXEMPT_PERMIT_TYPE = 'Not Licensed (Exempt)'

_COUNT_RE = re.compile(r'Displaying\s+([\d,]+)\s+Providers?', re.I)


def create_log_file(path=LOG_FILE):
    with open(path, 'w') as f:
        f.write('')


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def _slug(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _norm_id(value):
    """Permit Number as a clean string, leading zeros intact. Blank/'-'/'nan'
    collapse to '' (exempt providers have no permit number)."""
    if value is None:
        return ''
    s = str(value).strip()
    if s.lower() in ('', 'nan', 'none', '-', 'n/a'):
        return ''
    s = re.sub(r'\.0$', '', s)
    return s


def _norm_rating(value):
    """`ABC Level` as the native letter grade, upper-cased and stripped.
    '' means the provider is licensed but unrated."""
    if value is None:
        return ''
    s = str(value).strip().upper()
    return '' if s.lower() in ('nan', 'none', 'n/a') else s


def fetch_county_csv(county, session=None):
    """The county's full export as CSV text, or None on failure (logged, never
    raised, so one bad county cannot kill the sweep)."""
    session = session or requests
    try:
        r = session.get(BASE_URL + EXCEL_PATH, params={'county': county},
                        headers={'User-Agent': UA}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        ctype = r.headers.get('content-type', '')
        if 'csv' not in ctype.lower():
            log(f'  ! {county}: expected text/csv, got {ctype!r} — the export '
                f'endpoint may have changed; skipping.')
            return None
        return r.text
    except Exception as e:
        log(f'  ! {county}: CSV request failed: {e}')
        return None


def fetch_county_stated_count(county, session=None):
    """The HTML view's 'Displaying N Providers' count, or None."""
    session = session or requests
    try:
        r = session.get(BASE_URL + HTML_PATH, params={'county': county},
                        headers={'User-Agent': UA}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        m = _COUNT_RE.search(soup.get_text(' '))
        return int(m.group(1).replace(',', '')) if m else None
    except Exception as e:
        log(f'  . {county}: stated-count check unavailable ({e})')
        return None


def parse_county_csv(text, county):
    """CSV text -> DataFrame of strings, after validating the header."""
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f'{county}: export is missing expected column(s) {missing}. '
            f'Got: {list(df.columns)}')
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        log(f'  . {county}: export gained new column(s) {extra} — kept as bonus.')
    return df


def raw_path(county):
    return os.path.join(RAW_DIR, f'{_slug(county)}.csv')


def sweep(counties, force=False, delay_range=(1.0, 2.5), check_counts=True):
    os.makedirs(RAW_DIR, exist_ok=True)
    session = requests.Session()

    for i, county in enumerate(counties):
        path = raw_path(county)
        if os.path.exists(path) and not force:
            log(f'[{i + 1}/{len(counties)}] {county}: cached, skipping fetch')
            continue

        log(f'[{i + 1}/{len(counties)}] {county}: fetching')
        text = fetch_county_csv(county, session=session)
        if text is None:
            continue

        try:
            df = parse_county_csv(text, county)
        except Exception as e:
            log(f'  ! {county}: {e}')
            continue

        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

        n = len(df)
        if check_counts:
            stated = fetch_county_stated_count(county, session=session)
            if stated is None:
                log(f'  {county}: {n} row(s) (no stated count to cross-check)')
            elif stated != n:
                log(f'  ! MISMATCH {county}: HTML says {stated} providers but '
                    f'the CSV export returned {n} row(s). The export may be '
                    f'truncated or filtered — inspect {path} before trusting it.')
            else:
                log(f'  {county}: {n} row(s) (confirmed against stated count)')
        else:
            log(f'  {county}: {n} row(s)')

        if i < len(counties) - 1:
            time.sleep(random.uniform(*delay_range))


def merge(counties, out_csv=MERGED_PATH):
    """Concatenate every cached county export into one row-per-provider table.

    Dedup is on Permit Number only among rows that have one: exempt providers
    all share a blank permit number and would otherwise collapse into one row."""
    frames, missing = [], []
    for county in counties:
        path = raw_path(county)
        if not os.path.exists(path):
            missing.append(county)
            continue
        with open(path, encoding='utf-8') as f:
            text = f.read()
        try:
            df = parse_county_csv(text, county)
        except Exception as e:
            log(f'  ! {county}: {e} (skipped in merge)')
            continue
        df['source_county'] = county
        frames.append(df)

    if missing:
        log(f'{len(missing)} county file(s) never fetched, skipped in merge: '
            f'{", ".join(missing)}')
    if not frames:
        raise RuntimeError(
            'No cached county exports found. Run the sweep first (the CSV '
            'endpoint may also have changed — check sc_crawler_log.txt).')

    df = pd.concat(frames, ignore_index=True)

    df['Permit Number'] = df['Permit Number'].map(_norm_id)
    df['ABC Level'] = df['ABC Level'].map(_norm_rating)
    df['exempt'] = (df['Permit Type'].str.strip() == EXEMPT_PERMIT_TYPE)

    before = len(df)
    has_id = df['Permit Number'] != ''
    permitted = df[has_id].drop_duplicates(subset='Permit Number', keep='first')
    unpermitted = df[~has_id].drop_duplicates(
        subset=['Provider Name', 'Street', 'Zip'], keep='first')
    df = pd.concat([permitted, unpermitted], ignore_index=True)
    df = df.sort_values(['Permit Number', 'Provider Name']).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_exempt = int(df['exempt'].sum())
    n_blank_id = int((df['Permit Number'] == '').sum())
    rated = df['ABC Level'].replace('', pd.NA).dropna()

    log('')
    log(f'Merged {before} raw row(s) -> {len(df)} unique provider(s) -> {out_csv}')
    log(f'  providers with a Permit Number : {len(df) - n_blank_id}')
    log(f'  providers WITHOUT one (exempt) : {n_blank_id} '
        f'(Permit Type == "{EXEMPT_PERMIT_TYPE}": {n_exempt})')
    log(f'  rows carrying an ABC Level     : {len(rated)}')
    log(f'  rows with NO ABC Level (unrated): {len(df) - len(rated)}')
    if len(rated):
        log('  ABC Level distribution:')
        for level, count in rated.value_counts().items():
            log(f'    {level:>3} : {count}')
    log('  Permit Type distribution:')
    for ptype, count in df['Permit Type'].value_counts().items():
        log(f'    {ptype} : {count}')
    log('  Facility Type distribution:')
    for ftype, count in df['Facility Type'].value_counts().items():
        log(f'    {ftype} : {count}')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Crawl South Carolina's ABC Quality provider directory.")
    ap.add_argument('--counties', default=None,
                    help='Comma-separated county names (default: all 46).')
    ap.add_argument('--limit', type=int, default=None,
                    help='Only sweep the first N counties (smoke test).')
    ap.add_argument('--force', action='store_true',
                    help='Re-fetch counties even if a cached CSV exists.')
    ap.add_argument('--merge-only', action='store_true',
                    help='Skip fetching; rebuild sc_records.csv from county_raw/.')
    ap.add_argument('--no-count-check', action='store_true',
                    help='Skip the HTML "Displaying N Providers" cross-check.')
    ap.add_argument('--output', default=MERGED_PATH)
    ap.add_argument('--delay-min', type=float, default=1.0)
    ap.add_argument('--delay-max', type=float, default=2.5)
    args = ap.parse_args()

    create_log_file()
    counties = ([c.strip() for c in args.counties.split(',')] if args.counties
                else list(SC_COUNTIES))
    for c in counties:
        if c not in SC_COUNTIES:
            raise SystemExit(f'Unknown county: {c!r} (expected one of SC_COUNTIES)')
    if args.limit:
        counties = counties[:args.limit]

    if not args.merge_only:
        sweep(counties, force=args.force,
              delay_range=(args.delay_min, args.delay_max),
              check_counts=not args.no_count_check)
    merge(counties, out_csv=args.output)
