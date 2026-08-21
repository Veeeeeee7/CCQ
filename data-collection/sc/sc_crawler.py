"""
sc_crawler.py — build the South Carolina ABC Quality per-provider dataset.

SITE / APPROACH (Phase 0-1 recon, 2026-07-08)
---------------------------------------------
South Carolina's QRIS is **ABC Quality** (SCDSS, Division of Early Care and
Education). Two public directories exist:

  * scchildcare.org/provider-search — the state licensing directory. Its results
    are loaded by client-side JS (AJAX); a plain GET returns only the empty form
    listing. It was the original Checkpoint 0 pick but is the harder path (and
    was intermittently down during recon).
  * abcquality.org/provider-search   — the ABC Quality directory. **Fully
    server-rendered**, and — the key find — it exposes a plain CSV export:

        GET https://abcquality.org/provider-search/excel/?county=<County>

    despite the UI calling it "Export to Excel", the response is
    `Content-Type: text/csv`. It is **not paginated** (the HTML view pages at 8
    rows/page; the CSV returns the county's full result set in one shot), so the
    46 SC counties = the entire state in 46 requests. No browser, no API keys,
    no anti-bot friction observed. This is the "downloadable dataset" tier the
    project plan says to prefer, so that's what this crawler uses.

CSV COLUMNS (verified against real responses)
    Provider Name, Permit Type, Permit Number, Operator, Facility Type, Street,
    City, State, Zip, County, Phone, ABC Level, Last ABC Inspection Date,
    Capacity, Head Start, First Steps, Breastfeeding Friendly, Sleep Safe

  `Permit Number` -> provider_id   (the state licensing/registration number)
  `ABC Level`     -> qr_rating     (A+ / A / B+ / B / C; "P" = score pending;
                                    EMPTY = licensed but unrated)

QUIRKS THAT DROVE THE DESIGN
  * `Permit Number` is NOT the id used in detail-page URLs. The site keys detail
    pages on an internal id (/provider/4800/union-day-school/) that is absent
    from the CSV entirely. Per Checkpoint 1 we key on Permit Number only and do
    not crawl detail pages, so that internal id never enters this pipeline.
  * Exempt providers (`Permit Type` = "Not Licensed (Exempt)", `Facility Type`
    = "EAA") have a **blank Permit Number** — they carry no state permit at all.
    They ARE rated. They're kept here (the crawler is faithful to the source);
    a later step decides their fate. See `exempt` flag below.
  * The HTML results table's "Facility Type" column prints "Child Care Center"
    for *every* row, including family child care homes. It is wrong/cosmetic.
    The CSV's `Facility Type` code (A / C / EAA / ...) is the authoritative one.
  * `Permit Number` must stay a STRING. It is numeric-looking; letting pandas
    coerce it to int would destroy any leading zeros and turn blanks into NaN
    -> 0. Everything is read with `dtype=str`.
  * Ratings are ordered A+ > A > B+ > B > C, so the HTML view lists rated
    providers first and unrated ones last. The CSV carries the same ordering;
    it is not relied on.

SELF-CHECK
  For each county the crawler also fetches the HTML view and reads the page's
  own "Displaying N Providers" headline, then compares N against the number of
  CSV rows. A mismatch is logged loudly — it would mean the CSV export and the
  HTML search disagree (e.g. a silently truncated export), which must be
  investigated before the data is trusted.

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

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

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

# All 46 South Carolina counties, taken verbatim from the county <select> on
# abcquality.org/provider-search/advanced/ (so the spelling the site expects,
# e.g. "McCormick", is preserved exactly).
SEED_PATH = "sc_data/sc_seed.csv"


def _load_counties(path=SEED_PATH):
    """The counties to sweep, one per row of the provided seed."""
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as fh:
        return [r["county"].strip() for r in _csv.DictReader(fh)
                if r["county"].strip()]


SC_COUNTIES = _load_counties()
assert len(SC_COUNTIES) == 46, f'expected 46 SC counties, got {len(SC_COUNTIES)}'
assert len(set(SC_COUNTIES)) == 46, 'duplicate county in SC_COUNTIES'

# The exact header the export serves. Checked on every county response so a
# silent upstream schema change becomes a loud failure instead of bad data.
EXPECTED_COLUMNS = [
    'Provider Name', 'Permit Type', 'Permit Number', 'Operator',
    'Facility Type', 'Street', 'City', 'State', 'Zip', 'County', 'Phone',
    'ABC Level', 'Last ABC Inspection Date', 'Capacity', 'Head Start',
    'First Steps', 'Breastfeeding Friendly', 'Sleep Safe',
]

EXEMPT_PERMIT_TYPE = 'Not Licensed (Exempt)'

_COUNT_RE = re.compile(r'Displaying\s+([\d,]+)\s+Providers?', re.I)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def create_log_file(path=LOG_FILE):
    with open(path, 'w') as f:
        f.write('')


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def _slug(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


# ---------------------------------------------------------------------------
# id / value normalization
# ---------------------------------------------------------------------------

def _norm_id(value):
    """Permit Number as a clean STRING, leading zeros intact.

    Guards the one failure mode that silently corrupts the whole dataset: a
    numeric-looking id round-tripping through a float ('16941.0') or losing a
    leading zero. Blank/'-'/'nan' collapse to '' (exempt providers legitimately
    have no permit number)."""
    if value is None:
        return ''
    s = str(value).strip()
    if s.lower() in ('', 'nan', 'none', '-', 'n/a'):
        return ''
    s = re.sub(r'\.0$', '', s)          # '16941.0' -> '16941'
    return s


def _norm_rating(value):
    """`ABC Level` verbatim, upper-cased and whitespace-stripped.

    Kept as the native letter grade here — mapping to the numeric 1-5 scale
    (C=1, B=2, B+=3, A=4, A+=5) is not this script's job -- the letter grade
    is recorded exactly as the site prints it.
    '' means the provider is licensed but carries no ABC Quality rating."""
    if value is None:
        return ''
    s = str(value).strip().upper()
    return '' if s.lower() in ('nan', 'none', 'n/a') else s


# ---------------------------------------------------------------------------
# one county
# ---------------------------------------------------------------------------

def fetch_county_csv(county, session=None):
    """The county's full provider export. Returns raw CSV text, or None on
    failure (logged, never raised — one bad county must not kill the sweep)."""
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
    """The HTML view's own 'Displaying N Providers' headline, for cross-checking
    the CSV row count. None if it can't be read (never fatal)."""
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
    """CSV text -> DataFrame of strings. Validates the header before trusting a
    single row, so an upstream schema change fails loudly rather than quietly
    producing a dataset with the wrong column meanings."""
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


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def merge(counties, out_csv=MERGED_PATH):
    """Concatenate every cached county export into one row-per-provider table.

    Dedup is on Permit Number, but ONLY among rows that actually have one:
    exempt providers all share a blank permit number, so deduping naively would
    collapse every exempt provider in the state into a single row."""
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
