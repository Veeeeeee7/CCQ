"""
wa_seed.py — build the Washington provider seed list (+ fetch the Socrata enrich file).

Washington's public finder, Child Care Check (findchildcarewa.org), is a
Salesforce **Visualforce** page that talks to Apex `@RemoteAction` methods over
`POST /apexremote`. That means we never need a browser: the search is a plain
JSON-RPC call, and the only trick is that each remoted method is signed with a
per-page `csrf` token and an `authorization` JWT, both embedded in the
`PSS_Search` page HTML inside

    new $VFRM.RemotingProviderImpl({"vf":{"vid":...},"actions":{...}})

So we GET the search page once (which also plants the session cookies), parse
that config out, and then replay `getSOSLKeys` ourselves.

Search API (signature read straight off the page's own JS):

    PSS_SearchController.getSOSLKeys(
        searchBy,          # free-text term, e.g. a ZIP code
        facilityTypes,     # '' = no filter, else "'A','B'" (quoted, comma-joined)
        providerTypes,     # list, e.g. ["DEL Licensed", "Exempt Care", ...]
        ageClauses,        # []
        geoLatitude,       # null
        geoLongitude,      # null
        distance,          # null
        additionalClauses  # []
    ) -> ["001t0000008dnBhAAI", ...]   # Salesforce Account Ids

Two things follow from that signature:

  * It's **SOSL** (a text search), so it needs a term — there is no "return
    everything" call. The page also ships a `getKeys` method, but nothing ever
    invokes it. Hence we enumerate by sweeping ZIP codes, which is what a human
    would type in anyway.
  * The Ids it returns are 18-char Salesforce Account Ids (`001t...`) — exactly
    the same values as `wacompassid` in the DCYF Socrata open dataset. That's
    what lets us use one `provider_id` across both sources.

ZIP sweep: Washington's ZIPs live in 98001-99403. We sweep that whole numeric
range by default rather than relying on a canned ZIP list, because family child
care homes sit in residential ZIPs that may never appear in the centers dataset.
Non-existent ZIPs simply return zero keys, which is cheap. A provider is often
returned by several ZIPs (SOSL matches any indexed field), so we dedupe on Id.

    python wa_seed.py                     # sweep 98001-99403 -> wa_data/wa_seed.csv
    python wa_seed.py --socrata           # also download the Socrata enrich CSV
    python wa_seed.py --zip-file zips.txt # one ZIP per line instead of the sweep
    python wa_seed.py --limit 25          # smoke test: only the first 25 ZIPs

Resume-safe: ZIPs already swept are recorded in wa_data/wa_seed_zips_done.txt
and skipped on a re-run.

Output (wa_data/wa_seed.csv):
    provider_id   18-char Salesforce Account Id, kept as a string
    found_zip     the first ZIP whose search surfaced this provider

Deps: pip install requests pandas
"""

import argparse
import json
import os
import random
import time

import pandas as pd
import requests

BASE_URL = 'https://www.findchildcarewa.org'
SEARCH_URL = f'{BASE_URL}/PSS_Search'
APEXREMOTE_URL = f'{BASE_URL}/apexremote'

CONTROLLER = 'PSS_SearchController'

# Exact vocabularies read off the search page's filter checkboxes.
FACILITY_TYPES = ['Child Care Center', 'Family Child Care Home',
                  'School-Age Program', 'Outdoor Nature Based Program']
PROVIDER_TYPES = ['DEL Licensed', 'Exempt Care', 'Formerly Licensed',
                  'Unlawful Care']

# Washington ZIP codes occupy 980xx-994xx.
ZIP_START, ZIP_END = 98001, 99403

# DCYF open data for non-FCC providers (county, region, license dates/capacity,
# licensor, SSPS + FamLink ids). Keyed on wacompassid.
SOCRATA_CSV = 'https://data.wa.gov/resource/was8-3ni8.csv?$limit=50000'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

SEED_COLUMNS = ['provider_id', 'found_zip']

LOG_FILE = 'wa_seed_log.txt'

# Python doesn't use the macOS keychain, so a python.org build or a
# TLS-intercepting proxy can fail verification although the site is fine.
# Honour the standard env vars; --ca-bundle / --insecure are the escape hatch.
CA_BUNDLE = (os.environ.get('REQUESTS_CA_BUNDLE')
             or os.environ.get('SSL_CERT_FILE')
             or None)


def make_session(ca_bundle=None, insecure=False, ua=UA):
    s = requests.Session()
    s.headers.update({'User-Agent': ua})
    if insecure:
        s.verify = False
        requests.packages.urllib3.disable_warnings()
    elif ca_bundle or CA_BUNDLE:
        s.verify = ca_bundle or CA_BUNDLE
    return s


def create_log_file(path=LOG_FILE):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def _extract_braced_json(text, start):
    """Return the JSON object beginning at text[start] == '{', matching braces
    while ignoring braces inside string literals. The remoting config is too
    big and too nested for a regex to carve out safely."""
    depth, i, in_str, esc = 0, start, False, False
    while i < len(text):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        i += 1
    raise ValueError('unbalanced braces in remoting config')


class RemotingClient:
    """Thin client for Salesforce Visualforce Remoting (`/apexremote`).

    Each remoted method carries its own `csrf` + `authorization` JWT, so we key
    them by method name. Tokens are tied to the page load / session cookies and
    do expire, so `call()` transparently re-bootstraps once on an auth failure.
    """

    def __init__(self, session=None, ua=UA, log_fn=print, ca_bundle=None,
                 insecure=False):
        self.s = session or make_session(ca_bundle, insecure, ua)
        self.s.headers.update({'User-Agent': ua})
        self.log = log_fn
        self.vid = None
        self.methods = {}
        self._tid = 0

    def bootstrap(self):
        """GET the search page: sets session cookies and yields the tokens."""
        try:
            r = self.s.get(SEARCH_URL, timeout=60)
        except requests.exceptions.SSLError as e:
            raise RuntimeError(
                f'TLS verification failed for {SEARCH_URL}.\n'
                f'  {e}\n'
                'This is a local trust-store problem, not the site. Try, in order:\n'
                '  1) pip install -U certifi\n'
                '  2) macOS + python.org build: run '
                '"/Applications/Python 3.x/Install Certificates.command"\n'
                '  3) behind a TLS-inspecting proxy: point at its root, e.g.\n'
                '     export REQUESTS_CA_BUNDLE=/path/to/corp-ca.pem\n'
                '     (or pass --ca-bundle /path/to/corp-ca.pem)\n'
                '  4) last resort: --insecure (skips verification)') from e
        r.raise_for_status()
        marker = 'RemotingProviderImpl('
        i = r.text.find(marker)
        if i < 0:
            raise RuntimeError(
                'No RemotingProviderImpl config on PSS_Search — the site '
                'changed. Re-run wa_capture.py and inspect the page.')
        cfg = json.loads(_extract_braced_json(r.text, r.text.index('{', i)))
        self.vid = cfg['vf']['vid']
        self.methods = {m['name']: m
                        for m in cfg['actions'][CONTROLLER]['ms']}
        self.log(f'  bootstrapped remoting: vid={self.vid} '
                 f'methods={sorted(self.methods)}')
        return self

    def call(self, method, data, _retry=True):
        if not self.methods:
            self.bootstrap()
        m = self.methods[method]
        self._tid += 1
        payload = {'action': CONTROLLER, 'method': method, 'data': data,
                   'type': 'rpc', 'tid': self._tid,
                   'ctx': {'csrf': m['csrf'], 'vid': self.vid, 'ns': m.get('ns', ''),
                           'ver': int(m['ver']), 'authorization': m['authorization']}}
        r = self.s.post(APEXREMOTE_URL, json=payload, timeout=60, headers={
            'Content-Type': 'application/json',
            'X-User-Agent': 'Visualforce Remoting',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': SEARCH_URL,
            'Accept': '*/*'})

        body = None
        if r.status_code == 200:
            try:
                body = r.json()
            except ValueError:
                body = None

        if isinstance(body, list) and body and body[0].get('statusCode') == 200:
            return body[0].get('result')

        # Expired csrf/JWT looks like a non-200 statusCode or an HTML login
        # bounce. Re-bootstrap once, then give up.
        if _retry:
            self.log(f'  ! {method} failed (http={r.status_code}); '
                     f're-bootstrapping tokens and retrying')
            time.sleep(2)
            self.bootstrap()
            return self.call(method, data, _retry=False)

        msg = body[0].get('message') if isinstance(body, list) and body else r.text[:200]
        raise RuntimeError(f'{method} failed: http={r.status_code} msg={msg!r}')


def sosl_keys(client, term, facility_types=None, provider_types=None):
    """getSOSLKeys(searchBy, facilityTypes, providerTypes, ageClauses,
    geoLat, geoLng, distance, additionalClauses) -> [AccountId, ...]

    facility_types=None means *no* facility filter (all four types)."""
    if facility_types:
        ft = ','.join("'" + t.replace("'", "\\'") + "'" for t in facility_types)
    else:
        ft = ''
    pt = list(provider_types or PROVIDER_TYPES)
    data = [str(term).replace("'", "\\'"), ft, pt, [], None, None, None, []]
    res = client.call('getSOSLKeys', data)
    return list(res or [])


def _load_done_zips(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as f:
        return {ln.strip() for ln in f if ln.strip()}


def _load_existing_seed(path):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str).fillna('')
    return dict(zip(df['provider_id'], df['found_zip']))


def build_seed(out_csv='wa_data/wa_seed.csv', zips=None, limit=None,
               facility_types=None, provider_types=None,
               min_delay=0.3, max_delay=0.8, ca_bundle=None, insecure=False):
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    done_path = os.path.join(os.path.dirname(out_csv) or '.',
                             'wa_seed_zips_done.txt')

    if zips is None:
        zips = [str(z) for z in range(ZIP_START, ZIP_END + 1)]
    done = _load_done_zips(done_path)
    todo = [z for z in zips if z not in done]
    if limit:
        todo = todo[:limit]

    found = _load_existing_seed(out_csv)
    log(f'ZIPs: {len(zips)} total, {len(done)} already done, {len(todo)} to sweep')
    log(f'Starting with {len(found)} providers already seeded')

    client = RemotingClient(log_fn=log, ca_bundle=ca_bundle,
                            insecure=insecure).bootstrap()

    for n, z in enumerate(todo, 1):
        try:
            ids = sosl_keys(client, z, facility_types, provider_types)
        except Exception as e:
            log(f'  [{n}/{len(todo)}] zip={z} ERROR {e!r} — skipping (not marked done)')
            time.sleep(3)
            continue

        new = 0
        for pid in ids:
            if pid not in found:
                found[pid] = z
                new += 1
        if ids:
            log(f'  [{n}/{len(todo)}] zip={z}: {len(ids)} hits, {new} new '
                f'(total {len(found)})')

        with open(done_path, 'a', encoding='utf-8') as f:
            f.write(z + '\n')

        if n % 25 == 0 or n == len(todo):
            _write_seed(out_csv, found)

        time.sleep(random.uniform(min_delay, max_delay))

    df = _write_seed(out_csv, found)
    log(f'Seed built: {len(df)} unique providers -> {out_csv}')
    return df


def _write_seed(out_csv, found):
    df = pd.DataFrame(sorted(found.items()), columns=SEED_COLUMNS)
    df['provider_id'] = df['provider_id'].astype(str).str.strip()
    df.to_csv(out_csv, index=False)
    return df


def fetch_socrata(out_csv='wa_data/wa_socrata.csv', ca_bundle=None, insecure=False):
    """Download the DCYF open dataset (centers / school-age / outdoor), which
    carries licensing attributes the finder doesn't. FCC homes are absent by
    design."""
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    log(f'Fetching {SOCRATA_CSV}')
    s = make_session(ca_bundle, insecure)
    r = s.get(SOCRATA_CSV, timeout=180)
    r.raise_for_status()
    with open(out_csv, 'wb') as f:
        f.write(r.content)
    df = pd.read_csv(out_csv, dtype=str)
    log(f'Socrata: {len(df)} rows, {df["wacompassid"].nunique()} unique '
        f'wacompassid -> {out_csv}')
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Build the WA provider seed CSV.')
    ap.add_argument('--output', default='wa_data/wa_seed.csv')
    ap.add_argument('--zip-file', default=None,
                    help='File with one ZIP per line (default: sweep 98001-99403).')
    ap.add_argument('--limit', type=int, default=None,
                    help='Only sweep the first N un-done ZIPs (smoke test).')
    ap.add_argument('--facility-types', nargs='*', default=None,
                    help=f'Subset of {FACILITY_TYPES}. Default: no filter (all).')
    ap.add_argument('--provider-types', nargs='*', default=None,
                    help=f'Subset of {PROVIDER_TYPES}. Default: all four.')
    ap.add_argument('--socrata', action='store_true',
                    help='Also download the Socrata enrichment CSV.')
    ap.add_argument('--socrata-only', action='store_true',
                    help='Only download the Socrata CSV; skip the ZIP sweep.')
    ap.add_argument('--ca-bundle', default=None,
                    help='PEM bundle for TLS verification (e.g. a corporate '
                         'proxy root). Also read from REQUESTS_CA_BUNDLE.')
    ap.add_argument('--insecure', action='store_true',
                    help='Skip TLS verification entirely. Last resort.')
    args = ap.parse_args()

    create_log_file()

    if args.socrata_only:
        fetch_socrata(ca_bundle=args.ca_bundle, insecure=args.insecure)
    else:
        zips = None
        if args.zip_file:
            with open(args.zip_file, encoding='utf-8') as f:
                zips = [ln.strip() for ln in f if ln.strip()]
        build_seed(args.output, zips=zips, limit=args.limit,
                   facility_types=args.facility_types,
                   provider_types=args.provider_types,
                   ca_bundle=args.ca_bundle, insecure=args.insecure)
        if args.socrata:
            fetch_socrata(ca_bundle=args.ca_bundle, insecure=args.insecure)
