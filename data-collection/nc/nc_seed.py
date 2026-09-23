"""
nc_seed.py — build the North Carolina facility seed list.

Scrapes the DCDEE WORKS "Facility Association" page, which lists every licensed
facility statewide as "Facility_ID - NAME", into a seed CSV that drives
nc_crawler.py. The page is on the modern host (valid TLS), so a plain HTTP GET
is all that's needed — no browser.

    python nc_seed.py                          # -> nc_data/nc_facilities_seed.csv
    python nc_seed.py --output some/where.csv

Output columns:
    facility_id    string, leading zeros preserved (NOT coerced to int)
    facility_name  from the list
    facility_type  CCC / FCC guessed from the name (the crawler confirms it)
    county         blank — filled in by the crawler from each facility page
    license_number blank

Deps: pip install requests pandas beautifulsoup4
"""

import argparse
import os
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

WORKS_FACILITY_LIST_URL = ('https://ncchildcare.ncdhhs.gov/'
                           'works_simulator/facilityAssociation.html')

SEED_COLUMNS = ['facility_id', 'facility_name', 'facility_type', 'county',
                'license_number']

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# A "Facility_ID - NAME" entry. IDs are 6-9 chars, often with leading zeros.
_ENTRY_RE = re.compile(r'\b(\d{6,9})\s*[-\u2013]\s*([^\n<|]{2,120})')


def _clean(s):
    if s is None:
        return None
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s or None


def _norm_fid(series):
    """Keep Facility_IDs as strings: strip a stray float '.0', preserve leading
    zeros verbatim (don't zfill)."""
    return (series.astype(str).str.strip()
            .str.replace(r'\.0$', '', regex=True))


def _guess_type(name):
    """CCC (center) vs FCC (family child care home) from the facility name.

    A SEARCH HINT ONLY (often blank or wrong); nc_crawler.py overwrites it with
    the `Facility/Program Type` label the facility's own page states.
    """
    blob = (name or '').lower()
    if 'family' in blob or 'home' in blob:
        return 'FCC'
    if 'center' in blob or 'academy' in blob or 'preschool' in blob:
        return 'CCC'
    return None


def parse_works_list(html):
    """Extract [{facility_id, facility_name, ...}] from the WORKS page. Handles
    both render shapes — a <select> of <option>s and plain text lines — and
    dedupes by id."""
    soup = BeautifulSoup(html, 'html.parser')
    seen, rows = set(), []

    def add(fid, name):
        fid = (fid or '').strip()
        if not fid or fid in seen:
            return
        seen.add(fid)
        name = _clean(name)
        rows.append({'facility_id': fid, 'facility_name': name,
                     'facility_type': _guess_type(name),
                     'county': None, 'license_number': None})

    # 1) structured elements (option/li/a) whose text is "ID - NAME"
    for el in soup.find_all(['option', 'li', 'a']):
        txt = el.get_text(' ', strip=True)
        m = _ENTRY_RE.match(txt) or _ENTRY_RE.search(txt)
        if m and (m.start() == 0 or txt[:m.start()].strip() == ''):
            add(m.group(1), m.group(2))

    # 2) text fallback for lists rendered as plain lines
    if len(rows) < 50:
        for m in _ENTRY_RE.finditer(soup.get_text('\n')):
            add(m.group(1), m.group(2))

    return rows


def build_seed(out_csv='nc_data/nc_facilities_seed.csv'):
    print(f'Fetching {WORKS_FACILITY_LIST_URL}')
    r = requests.get(WORKS_FACILITY_LIST_URL, headers={'User-Agent': UA}, timeout=60)
    r.raise_for_status()
    rows = parse_works_list(r.text)
    if not rows:
        raise RuntimeError(
            'Parsed 0 facilities — the page markup changed or the list loads '
            'from a separate data file. Inspect the page source.')

    df = pd.DataFrame(rows, columns=SEED_COLUMNS)
    df['facility_id'] = _norm_fid(df['facility_id'])
    df = (df[df['facility_id'].notna() & (df['facility_id'] != '')]
          .drop_duplicates(subset='facility_id').reset_index(drop=True))

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False)

    n_ccc = (df.facility_type == 'CCC').sum()
    n_fcc = (df.facility_type == 'FCC').sum()
    print(f'Seed built: {len(df)} facilities '
          f'({n_ccc} CCC, {n_fcc} FCC, {len(df) - n_ccc - n_fcc} unknown) -> {out_csv}')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build the NC facility seed CSV.')
    ap.add_argument('--output', default='nc_data/nc_facilities_seed.csv')
    args = ap.parse_args()
    build_seed(args.output)
