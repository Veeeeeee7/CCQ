"""
co_seed.py — build the Colorado licensed child care facility seed list.

Source: Colorado Information Marketplace (Socrata), dataset "Colorado Licensed
Child Care Facilities Report" (id a9rr-k8mu, data.colorado.gov): an official,
open, monthly-refreshed dataset read with a paginated GET against the public
SODA API. It carries native `provider_id` and `quality_rating` columns plus
address, capacity by age band, ECC/CCRR affiliation, CCCAP/UPK flags,
governing body and school district linkage. co_crawler.py enriches each
provider_id from the coloradoshines.com "program_details" pages.

"Ratable" filtering: Colorado Shines only rates birth-5 center and
family-child-care programs. Rather than hardcode which `provider_service_type`
values those are, any service type with at least one real 1-5 rating is kept;
a type that is 100% NA is dropped.

Three types that are structurally out of Shines' birth-5 scope --
"School-Age Child Care Center", "Resident Camp" and "Neighborhood Youth
Organization" -- each have a handful of rows with a genuine Level 1 rating, so
the type-level filter keeps them in full. KNOWN_NON_RATABLE only feeds the
sanity-check print; it does not change the filter.

    python co_seed.py                    # -> co_data/co_facilities_seed.csv
    python co_seed.py --output some/where.csv
    python co_seed.py --limit 500        # quick test: stop after ~500 rows
    python co_seed.py --app-token XXXX   # optional, raises the SODA rate limit
                                          # (or set SOCRATA_APP_TOKEN env var)

Output columns: every native column from the open dataset, unchanged, with
provider_id moved to the front.

Read this CSV back with dtype={'provider_id': str} (or dtype=str): a plain
pd.read_csv() would reinterpret the digit-only ids as int64.

Deps: pip install requests pandas
"""

import argparse
import os
import time

import pandas as pd
import requests

DATASET_ID = 'a9rr-k8mu'
BASE_URL = f'https://data.colorado.gov/resource/{DATASET_ID}.json'
PAGE_SIZE = 1000
MAX_RETRIES = 4
RETRY_SLEEP = 3  # seconds; multiplied by attempt number for linear backoff

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# Sanity-check list only -- see module docstring. The real filter is computed
# from the live data in build_seed().
KNOWN_NON_RATABLE = {'School-Age Child Care Center', 'Resident Camp',
                      'Neighborhood Youth Organization'}

VALID_RATINGS = {'1', '2', '3', '4', '5'}

# Full known schema. The SODA JSON API omits a null key from a row entirely,
# so a column that is null on EVERY row of a pull would vanish; reindexing onto
# this list keeps a sparse or all-null field in the output.
EXPECTED_COLUMNS = [
    'provider_id', 'provider_name', 'provider_service_type', 'street_address',
    'city', 'state', 'zip', 'county', 'ecc', 'ccrr',
    'school_district_operated_program', 'school_district', 'quality_rating',
    'award_date', 'expiration_date', 'total_licensed_capacity',
    'cccap_fa_status_d1', 'cccap_authorization_status',
    'licensed_home_capacity', 'licensed_infant_capacity',
    'licensed_toddler_capacity', 'licensed_preschool_capacity',
    'licensed_school_age_capacity', 'licensed_preschool_and_school_age_capacity',
    'licensed_resident_camp_capacity', 'licensed_nyo_capacity', 'governing_body',
    'upk_participation_2025_2026', 'upk_participation_2026_2027',
]


def _norm_id(series):
    """Keep provider_id as clean strings: strip a stray float '.0' but never
    zero-pad or coerce to int."""
    return (series.astype(str).str.strip()
            .str.replace(r'\.0$', '', regex=True))


def _is_valid_rating(val):
    """True if val is one of Colorado Shines' five levels ('NA', the source's
    missing-value marker, blanks and NaN are invalid)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip().upper()
    return s in VALID_RATINGS


def fetch_all(limit=None, app_token=None):
    """Page through the SODA API and return every row as a list of dicts."""
    headers = {'User-Agent': UA}
    if app_token:
        headers['X-App-Token'] = app_token

    rows, offset = [], 0
    while True:
        page_size = PAGE_SIZE if limit is None else min(PAGE_SIZE, limit - len(rows))
        if page_size <= 0:
            break
        params = {'$limit': page_size, '$offset': offset}

        r = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(BASE_URL, params=params, headers=headers, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = RETRY_SLEEP * attempt
                print(f'  retry {attempt}/{MAX_RETRIES} after error ({e}); waiting {wait}s')
                time.sleep(wait)

        page = r.json()
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        print(f'  fetched {offset} rows so far...')
        if len(page) < page_size:
            break  # short page -> that was the last one

    return rows


def build_seed(out_csv='co_data/co_facilities_seed.csv', limit=None, app_token=None):
    print(f'Fetching {BASE_URL} (dataset {DATASET_ID})')
    raw_rows = fetch_all(limit=limit, app_token=app_token)
    if not raw_rows:
        raise RuntimeError('Fetched 0 rows -- check the dataset id / API status.')

    df = pd.DataFrame(raw_rows)
    if 'provider_id' not in df.columns or 'quality_rating' not in df.columns:
        raise RuntimeError(
            f'Expected provider_id/quality_rating columns, got: {list(df.columns)}. '
            'The dataset schema may have changed -- inspect before proceeding.')

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f'NOTE: {missing_cols} absent from every row in this pull (the SODA '
              f'API omits null keys entirely) -- adding as all-empty so the output '
              f'schema stays complete and consistent across runs.')
        for c in missing_cols:
            df[c] = pd.NA
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra_cols:
        print(f'NOTE: dataset now has column(s) not in EXPECTED_COLUMNS -- the '
              f'source schema may have grown, update EXPECTED_COLUMNS: {extra_cols}')

    df['provider_id'] = _norm_id(df['provider_id'])
    df = df[df['provider_id'].notna() & (df['provider_id'] != '')]

    before = len(df)
    df = df.drop_duplicates(subset='provider_id', keep='first')
    if len(df) < before:
        print(f'Dropped {before - len(df)} duplicate provider_id rows.')

    n_missing_type = df['provider_service_type'].isna().sum()
    if n_missing_type:
        print(f'NOTE: {n_missing_type} rows have no provider_service_type at all '
              f'-- these will be excluded since ratability can\'t be determined.')

    # Discover which service types are ever rated rather than hardcoding a list
    # (there are several capacity-based family-child-care license classes).
    ratable_mask = df['quality_rating'].apply(_is_valid_rating)
    ratable_types = sorted(df.loc[ratable_mask, 'provider_service_type'].dropna().unique())
    all_types = sorted(df['provider_service_type'].dropna().unique())
    excluded_types = sorted(set(all_types) - set(ratable_types))

    print('\nProvider service types found:')
    for t in all_types:
        n = int((df['provider_service_type'] == t).sum())
        tag = 'ratable' if t in ratable_types else 'EXCLUDED (never rated)'
        print(f'  {t!r}: {n} rows -- {tag}')

    unexpected_excluded = set(excluded_types) - KNOWN_NON_RATABLE
    if unexpected_excluded:
        print(f'\nNOTE: excluding type(s) not on the known non-ratable list -- '
              f'double check these are really out of scope: {sorted(unexpected_excluded)}')
    expected_but_present = KNOWN_NON_RATABLE & set(ratable_types)
    if expected_but_present:
        print(f'\nNOTE: type(s) expected to be non-ratable now show a real rating -- '
              f'double check: {sorted(expected_but_present)}')

    df = df[df['provider_service_type'].isin(ratable_types)].reset_index(drop=True)

    # EXPECTED_COLUMNS first (provider_id leads), then any new columns.
    cols = [c for c in EXPECTED_COLUMNS if c in df.columns] + extra_cols
    df = df[cols]

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f'\nSeed built: {len(df)} ratable-type providers '
          f'(excluded types: {excluded_types}) -> {out_csv}')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Build the Colorado licensed child care facility seed CSV '
                     'from the Colorado Information Marketplace open dataset.')
    ap.add_argument('--output', default='co_data/co_facilities_seed.csv')
    ap.add_argument('--limit', type=int, default=None,
                     help='cap total rows fetched, for a quick test')
    ap.add_argument('--app-token', default=os.environ.get('SOCRATA_APP_TOKEN'),
                     help='optional Socrata app token (raises the rate limit); '
                          'also read from the SOCRATA_APP_TOKEN env var')
    args = ap.parse_args()
    build_seed(args.output, limit=args.limit, app_token=args.app_token)