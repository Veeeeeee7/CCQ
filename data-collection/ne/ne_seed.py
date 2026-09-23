"""
ne_seed.py — build the Nebraska seed lists.

Two public sources, neither needing a browser:

1. FACILITY SEED — every provider page, from the Yoast sitemaps linked from
   /sitemap_index.xml. The `child-care-facility` post type is not in the wp/v2
   REST API and the search only returns providers near an address, so the
   sitemaps are the only complete index.

       -> ne_data/ne_facilities_seed.csv   (slug, facility_url, lastmod)

2. DHHS LICENSING ROSTER — an Esri feature service, refreshed monthly:

       https://gis.ne.gov/Agency/rest/services/DHHS_Licensed_Child_Care/FeatureServer/0

   No rating, but its License_Number matches the finder's verbatim (CCC8794),
   so it joins onto the crawled rows. The host needs a legacy-tolerant TLS
   adapter and truncates large responses, so attributes are fetched by
   ObjectID in small POSTed chunks rather than `resultOffset` paging.

       -> ne_data/ne_licensing.csv

License numbers are always strings (CCC8794, FI11670, FII9561).

    python ne_seed.py                    # writes both CSVs
    python ne_seed.py --skip-licensing   # sitemap only

Deps: pip install requests pandas
"""

import argparse
import os
import re
import ssl
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

BASE_URL = 'https://stepuptoquality.ne.gov'
SITEMAP_INDEX = f'{BASE_URL}/sitemap_index.xml'
FACILITY_SITEMAP_RE = re.compile(r'child-care-facility-sitemap\d*\.xml$')

LICENSING_LAYER = ('https://gis.ne.gov/Agency/rest/services/'
                   'DHHS_Licensed_Child_Care/FeatureServer/0/query')
# The layer advertises maxRecordCount=2000, but gis.ne.gov truncates a big
# response mid-stream; small chunks always finish.
LICENSING_CHUNK = 200

SM_NS = '{http://www.sitemaps.org/schemas/sitemap/0.9}'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

SEED_COLUMNS = ['slug', 'facility_url', 'lastmod']


class _TLSAdapter(HTTPAdapter):
    """gis.ne.gov negotiates TLS in a way OpenSSL 3 refuses by default
    (`SSLEOFError: UNEXPECTED_EOF_WHILE_READING`). Allow legacy server connect
    + a lower cipher security level for that host only."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= getattr(ssl, 'OP_LEGACY_SERVER_CONNECT', 0x4)
        try:
            ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        except ssl.SSLError:
            pass
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


def _session():
    s = requests.Session()
    s.headers.update({'User-Agent': UA})
    s.mount('https://gis.ne.gov', _TLSAdapter())
    return s


_SESSION = None


def _get(url, params=None, tries=4, timeout=60, method='GET'):
    """Request with backoff. POST is used for the Esri licensing chunks."""
    global _SESSION
    if _SESSION is None:
        _SESSION = _session()
    last = None
    for attempt in range(tries):
        try:
            if method == 'POST':
                r = _SESSION.post(url, data=params, timeout=timeout)
            else:
                r = _SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:                                   # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def facility_sitemap_urls():
    root = ET.fromstring(_get(SITEMAP_INDEX).content)
    locs = [el.text.strip() for el in root.iter(f'{SM_NS}loc') if el.text]
    return [u for u in locs if FACILITY_SITEMAP_RE.search(u)]


def parse_facility_sitemap(url):
    root = ET.fromstring(_get(url).content)
    rows = []
    for u in root.iter(f'{SM_NS}url'):
        loc = u.find(f'{SM_NS}loc')
        mod = u.find(f'{SM_NS}lastmod')
        if loc is None or not loc.text:
            continue
        link = loc.text.strip()
        slug = link.rstrip('/').rsplit('/', 1)[-1]
        rows.append({'slug': slug, 'facility_url': link,
                     'lastmod': (mod.text.strip() if mod is not None and mod.text
                                 else None)})
    return rows


def build_facility_seed(output_csv):
    sitemaps = facility_sitemap_urls()
    print(f'Found {len(sitemaps)} child-care-facility sitemap(s).')
    rows = []
    for sm in sitemaps:
        got = parse_facility_sitemap(sm)
        print(f'  {sm.rsplit("/", 1)[-1]}: {len(got)} URLs')
        rows.extend(got)
        time.sleep(0.5)

    df = pd.DataFrame(rows, columns=SEED_COLUMNS)
    df = df.drop_duplicates(subset='slug').reset_index(drop=True)
    _write(df, output_csv)
    print(f'\nWrote {len(df)} facility URLs -> {output_csv}')
    return df


def fetch_object_ids():
    data = _get(LICENSING_LAYER,
                params={'where': '1=1', 'returnIdsOnly': 'true', 'f': 'json'}).json()
    if 'error' in data:
        raise RuntimeError(f'FeatureServer error: {data["error"]}')
    return data.get('objectIds') or []


def fetch_licensing(chunk=LICENSING_CHUNK):
    """Fetch attributes in small ObjectID chunks via POST."""
    oids = fetch_object_ids()
    print(f'  layer reports {len(oids)} features')
    features = []
    for start in range(0, len(oids), chunk):
        batch_ids = oids[start:start + chunk]
        params = {
            'objectIds': ','.join(str(i) for i in batch_ids),
            'outFields': '*',
            'returnGeometry': 'false',
            'f': 'json',
        }
        data = _get(LICENSING_LAYER, params=params, method='POST').json()
        if 'error' in data:
            raise RuntimeError(f'FeatureServer error: {data["error"]}')
        features.extend(f.get('attributes', {}) for f in data.get('features', []))
        print(f'  fetched {len(features)}/{len(oids)} licensing records...')
        time.sleep(0.3)
    return features


def build_licensing(output_csv, chunk=LICENSING_CHUNK):
    df = pd.DataFrame(fetch_licensing(chunk=chunk))
    if 'License_Number' in df.columns:
        # A stray '.0' would break the join to the finder.
        df['License_Number'] = (df['License_Number'].astype(str).str.strip()
                                .str.replace(r'\.0$', '', regex=True)
                                .replace({'nan': None, 'None': None, '': None}))
    for col in ('Issue_Date', 'Roster_Date', 'Geocoded_Date'):
        # Epoch ms -> ISO date.
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce') \
                if pd.api.types.is_numeric_dtype(df[col]) \
                else pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].dt.strftime('%Y-%m-%d')
    _write(df, output_csv)
    print(f'\nWrote {len(df)} licensing records -> {output_csv}')
    return df


def _write(df, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build the NE facility seed + licensing join table.')
    ap.add_argument('--seed-output', default='ne_data/ne_facilities_seed.csv')
    ap.add_argument('--licensing-output', default='ne_data/ne_licensing.csv')
    ap.add_argument('--skip-licensing', action='store_true')
    ap.add_argument('--licensing-only', action='store_true',
                    help='Skip the sitemap crawl; only refetch the DHHS roster.')
    ap.add_argument('--licensing-chunk', type=int, default=LICENSING_CHUNK,
                    help='ObjectIDs per request. Lower it if the host still '
                         'drops the connection.')
    args = ap.parse_args()

    if not args.licensing_only:
        build_facility_seed(args.seed_output)
    if not args.skip_licensing:
        print('\nFetching DHHS licensing roster...')
        build_licensing(args.licensing_output, chunk=args.licensing_chunk)
