"""
Washington Child Care Check crawler — one row per provider
==========================================================

Source of truth is the public finder, findchildcarewa.org, a Salesforce
Visualforce site. Two very different mechanisms, both plain HTTP — **no browser
is needed anywhere in this state** (see wa_capture.py for the recon that
established that):

  1. SEARCH / BULK FIELDS — Apex Remoting, `POST /apexremote`:
         PSS_SearchController.queryProviders([AccountId, ...]) -> [record, ...]
     Rich JSON: Early Achiever status, facility type, ages, hours, open slots,
     subsidy/food-program flags, HS/EHS/ECEAP funding, geolocation. Salesforce
     **omits null fields**, so the key set varies per provider — hence the
     column-union flush below (same trick as ga_crawler.py).

  2. DETAIL PAGE — ordinary server-rendered Visualforce:
         GET /PSS_Provider?id={AccountId}
     No remoting, so requests + BeautifulSoup is enough. Adds Provider Contacts,
     License History (incl. human-readable License IDs like PL-81384), and the
     Complaints / Inspections tabs, plus licensed capacity and ages.

  3. ENRICHMENT — the DCYF Socrata columns, provided pre-merged into
     wa_data/wa_seed.csv as socrata_*. They contribute
     county, region, license dates, licensed capacity, primary licensor, and the
     SSPS / FamLink identifiers. It covers centers, school-age, and outdoor
     programs only — **family child care homes are absent from it by design**, so
     those columns are legitimately empty for FCC rows.

GRAIN: one row per provider, keyed on the 18-char Salesforce Account Id. That id
is identical to `wacompassid` in the Socrata dataset, which is what makes the
join — and a single `provider_id` across both sources — possible.

RATING: three candidate columns are collected; one of them is the target.
  Early_Achiever_Status_Internal__c   e.g. "Level 3+"
  Early_Achiever_Status_External__c   e.g. "Quality Level 3+"
  socrata_earating                    e.g. "3+"
They express the same 5-level Early Achievers scale.

COVERAGE CAVEAT: the payload carries `Exclude_FH_provider_on_public_search__c`.
Family home providers can be flagged as hidden from public search (their address
is a private residence), so the finder is a floor on the licensed FCC population,
not a census. There is no other public per-provider FCC source.

Resume-safe: existing wa_data/wa_records.csv is read back at startup and any
provider already present is skipped. Per-section try/except means one broken
field can never kill a row — the section is filled with Nones and its name is
recorded in `errors`.

    python wa_crawler.py --limit 20        # smoke test
    python wa_crawler.py                   # full run
    python wa_crawler.py --skip-detail     # API only, no PSS_Provider GETs

Deps: pip install requests pandas beautifulsoup4
"""

import argparse
import json
import os
import random
import re
import time
import traceback

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

BASE_URL = 'https://www.findchildcarewa.org'
SEARCH_URL = f'{BASE_URL}/PSS_Search'
APEXREMOTE_URL = f'{BASE_URL}/apexremote'
PROVIDER_URL = f'{BASE_URL}/PSS_Provider?id={{pid}}'

CONTROLLER = 'PSS_SearchController'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36')

# TLS verification. Python doesn't use the macOS keychain, so a python.org
# interpreter that never ran "Install Certificates.command" — or a TLS-
# intercepting corporate proxy whose root isn't in certifi's bundle — fails with
# "unable to get local issuer certificate" even though the site is fine in a
# browser. Honour the standard env vars, else fall back to certifi, and expose
# an explicit --ca-bundle / --insecure escape hatch.
CA_BUNDLE = (os.environ.get('REQUESTS_CA_BUNDLE')
             or os.environ.get('SSL_CERT_FILE')
             or None)

SEED_CSV = 'wa_data/wa_seed.csv'
OUT_CSV = 'wa_data/wa_records.csv'
LOG_FILE = 'wa_crawler_log.txt'

KEY = 'provider_id'                 # the grain / resume key

QUERY_BATCH = 25                    # ids per queryProviders call
FLUSH_EVERY = 50                    # providers between full CSV rewrites
MIN_DELAY, MAX_DELAY = 0.4, 1.0     # polite pause between detail-page GETs

# Tab panes on PSS_Provider that hold a table each.
DETAIL_TABS = ['complaints', 'inspections', 'license_history']

# Columns pinned to the front of the CSV for readability; everything else is
# appended alphabetically by the column-union flush.
FRONT_COLS = [KEY, 'found_zip', 'Account_Name_Display_Label__c',
              'Latest_License_Facility_Type_Name__c',
              'Early_Achiever_Status_Internal__c',
              'Early_Achiever_Status_External__c',
              'socrata_earating', 'socrata_eaparticipation', 'errors']


def create_log_file(path=LOG_FILE):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def _clean(s):
    if s is None:
        return None
    s = re.sub(r'\s+', ' ', str(s)).replace('\xa0', ' ').strip()
    return s or None


def _norm_id(series):
    """Salesforce Account Ids are 18-char alphanumerics — no zero padding to
    preserve, but never let pandas coerce them to something else."""
    return series.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


# ---------------------------------------------------------------------------
# Visualforce Remoting client
# ---------------------------------------------------------------------------

def make_session(ca_bundle=None, insecure=False, ua=UA):
    s = requests.Session()
    s.headers.update({'User-Agent': ua})
    if insecure:
        s.verify = False
        requests.packages.urllib3.disable_warnings()
    elif ca_bundle or CA_BUNDLE:
        s.verify = ca_bundle or CA_BUNDLE
    return s


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


# ---------------------------------------------------------------------------
# empty section shapes — a failing section fills these rather than killing the row
# ---------------------------------------------------------------------------

# The detail page renders its facts as `div.form-group > label.control-label`
# ("Key:") followed by a sibling holding `p.form-control-static` ("Value"). We
# harvest *every* such pair rather than a hardcoded list, because several useful
# fields appear only here and never in the API payload — notably the
# human-readable License Number and the two Languages fields. These are the ones
# guaranteed to exist so downstream code can rely on them; anything extra the
# page offers is picked up by the column-union flush.
CORE_DETAIL_FIELDS = [
    'detail_license_name', 'detail_license_number', 'detail_facility_type',
    'detail_licensed_capacity', 'detail_ages', 'detail_languages_spoken',
    'detail_languages_of_instruction', 'detail_provider_status',
]


def _snake(s):
    return re.sub(r'[^a-z0-9]+', '_', (s or '').lower()).strip('_')


def empty_header():
    return {k: None for k in CORE_DETAIL_FIELDS}


def empty_early_achievers():
    return {'detail_ea_status': None, 'detail_ea_specialization': None}


def empty_contacts():
    return {'contacts_json': None, 'contacts_count': None}


def empty_license_history():
    return {'license_history_json': None, 'license_history_count': None,
            'license_id_current': None}


def empty_complaints():
    return {'complaints_json': None, 'complaints_count': None}


def empty_inspections():
    return {'inspections_json': None, 'inspections_count': None}


# ---------------------------------------------------------------------------
# detail-page parsing (server-rendered Visualforce)
# ---------------------------------------------------------------------------

def parse_header(soup):
    """Harvest every `label.control-label` -> `p.form-control-static` pair into
    `detail_<snake_case_key>` columns. Labels that are section headings (no
    trailing colon, no value sibling) are skipped."""
    out = empty_header()
    for lab in soup.select('label.control-label'):
        key = _clean(lab.get_text(' ', strip=True)) or ''
        if not key.endswith(':'):
            continue
        key = key[:-1].strip()
        sib = lab.find_next_sibling()
        if key is None or sib is None:
            continue
        val_node = sib.find('p', class_='form-control-static') or sib
        val = _clean(val_node.get_text(' ', strip=True))
        col = 'detail_' + _snake(key)
        if out.get(col) is None:
            out[col] = val
    return out


def _table_rows(node):
    """Parse the first <table> under `node` into a list of dicts using its <th>
    headers. Returns [] when there's no table or no body rows (a provider with
    zero complaints still renders the table shell)."""
    if node is None:
        return []
    table = node if getattr(node, 'name', None) == 'table' else node.find('table')
    if table is None:
        return []
    heads = [_clean(th.get_text(' ', strip=True)) or f'col{i}'
             for i, th in enumerate(table.find_all('th'))]
    rows = []
    for tr in table.find_all('tr'):
        cells = tr.find_all('td')
        if not cells:
            continue
        vals = [_clean(td.get_text(' ', strip=True)) for td in cells]
        if not any(vals):
            continue
        if heads and len(heads) == len(vals):
            rows.append(dict(zip(heads, vals)))
        else:
            rows.append({f'col{i}': v for i, v in enumerate(vals)})
    return rows


def _find_contacts_table(soup):
    """Provider Contacts sits outside the tab panes; identify it by headers."""
    for table in soup.find_all('table'):
        heads = {_clean(th.get_text(' ', strip=True)) or ''
                 for th in table.find_all('th')}
        if 'Full Name' in heads and 'Role' in heads:
            return table
    return None


def parse_early_achievers(soup):
    """The Early Achievers tab shows `<strong>Status: Quality Level N</strong>`
    plus a bolded "Areas of Specialization:" label whose value trails it inside
    the same paragraph. (The long boilerplate describing what the level means is
    identical for every provider at that level, so we don't keep it.)"""
    pane = soup.find(id='early-achievers')
    out = empty_early_achievers()
    if pane is None:
        return out
    for strong in pane.find_all('strong'):
        txt = _clean(strong.get_text(' ', strip=True)) or ''
        low = txt.lower()
        if low.startswith('status:'):
            out['detail_ea_status'] = _clean(txt.split(':', 1)[1])
        elif 'areas of specialization' in low:
            para = _clean(strong.parent.get_text(' ', strip=True)) or ''
            if para.startswith(txt):
                out['detail_ea_specialization'] = _clean(para[len(txt):])
    return out


def parse_contacts(soup):
    rows = _table_rows(_find_contacts_table(soup))
    return {'contacts_json': json.dumps(rows, ensure_ascii=False) if rows else None,
            'contacts_count': len(rows)}


def parse_license_history(soup):
    rows = _table_rows(soup.find(id='license_history'))
    current = None
    for r in rows:
        if (r.get('License Status') or '').lower() == 'open':
            current = r.get('License ID')
            break
    return {'license_history_json': json.dumps(rows, ensure_ascii=False) if rows else None,
            'license_history_count': len(rows),
            'license_id_current': current}


def parse_complaints(soup):
    rows = _table_rows(soup.find(id='complaints'))
    return {'complaints_json': json.dumps(rows, ensure_ascii=False) if rows else None,
            'complaints_count': len(rows)}


def parse_inspections(soup):
    rows = _table_rows(soup.find(id='inspections'))
    return {'inspections_json': json.dumps(rows, ensure_ascii=False) if rows else None,
            'inspections_count': len(rows)}


DETAIL_SECTIONS = [
    ('header', parse_header, empty_header),
    ('early_achievers', parse_early_achievers, empty_early_achievers),
    ('contacts', parse_contacts, empty_contacts),
    ('license_history', parse_license_history, empty_license_history),
    ('complaints', parse_complaints, empty_complaints),
    ('inspections', parse_inspections, empty_inspections),
]


def scrape_detail(session, pid, errors):
    """GET PSS_Provider?id=... and run every section under its own try/except."""
    row = {}
    try:
        r = session.get(PROVIDER_URL.format(pid=pid), timeout=60)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
    except Exception as e:
        errors.append('detail_fetch')
        log(f'    ! detail fetch failed for {pid}: {e!r}')
        for _, _, empty in DETAIL_SECTIONS:
            row.update(empty())
        return row

    for name, parse, empty in DETAIL_SECTIONS:
        try:
            row.update(parse(soup))
        except Exception:
            errors.append(name)
            row.update(empty())
            log(f'    ! section {name} failed for {pid}\n'
                f'{traceback.format_exc(limit=1)}')
    return row


# ---------------------------------------------------------------------------
# API record flattening
# ---------------------------------------------------------------------------

def flatten_record(rec):
    """queryProviders returns one nested child object, Latest_License_Rec_ID__r.
    Flatten it to license_rec_* and drop Salesforce's `attributes` noise."""
    out = {}
    for k, v in (rec or {}).items():
        if k == 'attributes':
            continue
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if k2 == 'attributes':
                    continue
                out[f'license_rec_{k2}'] = v2
        elif isinstance(v, list):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Socrata enrichment
# ---------------------------------------------------------------------------

def load_socrata(path=SEED_CSV):
    """{provider_id: {socrata_*: value}}. Absent for FCC homes by design.

    The DCYF open dataset is supplied already joined onto the seed on
    wacompassid, which is the same 18-char Account Id we crawl on. Rows that got
    no match carry empty socrata_* cells; they must stay out of the lookup, so
    an unmatched provider ends up with the column absent rather than blank."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # wacompassid is the join key, not an attribute — it would only restate the
    # provider_id it was matched on.
    cols = [c for c in df.columns
            if c.startswith('socrata_') and c != 'socrata_wacompassid']
    if not cols:
        log(f'  no socrata_* columns in {path}; skipping enrichment')
        return {}
    df[KEY] = _norm_id(df[KEY])
    df = df.drop_duplicates(subset=KEY, keep='first')
    lookup = {}
    for rec in df[[KEY] + cols].to_dict('records'):
        pid = rec.pop(KEY)
        if any(str(v).strip() for v in rec.values()):
            lookup[pid] = {k: (v if str(v).strip() else None)
                           for k, v in rec.items()}
    log(f'  Socrata enrichment: {len(lookup)} providers')
    return lookup


# ---------------------------------------------------------------------------
# CSV I/O with column union (Salesforce omits null fields => schema grows)
# ---------------------------------------------------------------------------

def load_existing(path):
    if not os.path.exists(path):
        return [], set()
    df = pd.read_csv(path, dtype=str)
    if KEY not in df.columns:
        return [], set()
    df[KEY] = _norm_id(df[KEY])
    rows = df.to_dict('records')
    return rows, set(df[KEY])


def flush(path, rows):
    cols = set()
    for r in rows:
        cols.update(r)
    ordered = [c for c in FRONT_COLS if c in cols]
    ordered += sorted(c for c in cols if c not in FRONT_COLS)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pd.DataFrame(rows, columns=ordered).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# main crawl
# ---------------------------------------------------------------------------

def crawl(seed_csv=SEED_CSV, out_csv=OUT_CSV, limit=None, skip_detail=False,
          batch=QUERY_BATCH, ca_bundle=None, insecure=False):
    seed = pd.read_csv(seed_csv, dtype=str)
    seed['provider_id'] = _norm_id(seed['provider_id'])
    zip_of = dict(zip(seed['provider_id'], seed.get('found_zip', '')))

    rows, done = load_existing(out_csv)
    todo = [p for p in seed['provider_id'] if p not in done]
    if limit:
        todo = todo[:limit]

    log(f'Seed: {len(seed)} providers | already done: {len(done)} | '
        f'to crawl: {len(todo)}')
    if not todo:
        log('Nothing to do.')
        return

    socrata = load_socrata(seed_csv)

    client = RemotingClient(log_fn=log, ca_bundle=ca_bundle,
                            insecure=insecure).bootstrap()
    session = client.s        # reuse cookies/UA/TLS config for the detail GETs

    processed = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        try:
            recs = client.call('queryProviders', [chunk]) or []
        except Exception as e:
            log(f'  ! queryProviders failed for batch {i//batch}: {e!r}')
            time.sleep(5)
            continue

        by_id = {r.get('Id'): r for r in recs if r.get('Id')}
        missing = [p for p in chunk if p not in by_id]
        if missing:
            log(f'  ! {len(missing)} id(s) returned no record (e.g. {missing[:2]})')

        for pid in chunk:
            errors = []
            row = {KEY: pid, 'found_zip': zip_of.get(pid)}

            rec = by_id.get(pid)
            if rec is None:
                errors.append('queryProviders')
            else:
                try:
                    row.update(flatten_record(rec))
                except Exception:
                    errors.append('flatten')
                    log(f'    ! flatten failed for {pid}\n{traceback.format_exc(limit=1)}')

            if not skip_detail:
                row.update(scrape_detail(session, pid, errors))
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            else:
                for _, _, empty in DETAIL_SECTIONS:
                    row.update(empty())

            row.update(socrata.get(pid, {}))
            row['errors'] = ';'.join(errors) if errors else None

            rows.append(row)
            processed += 1

            if processed % FLUSH_EVERY == 0:
                flush(out_csv, rows)
                log(f'  … {processed}/{len(todo)} crawled (flushed {len(rows)} rows)')

    flush(out_csv, rows)
    log(f'Done. {processed} newly crawled; {len(rows)} rows total -> {out_csv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Crawl WA Child Care Check providers.')
    ap.add_argument('--seed', default=SEED_CSV)
    ap.add_argument('--output', default=OUT_CSV)
    ap.add_argument('--limit', type=int, default=None,
                    help='Only crawl the first N un-done providers (smoke test).')
    ap.add_argument('--skip-detail', action='store_true',
                    help='Skip the PSS_Provider detail GETs (API fields only).')
    ap.add_argument('--batch', type=int, default=QUERY_BATCH)
    ap.add_argument('--ca-bundle', default=None,
                    help='PEM bundle for TLS verification (e.g. a corporate '
                         'proxy root). Also read from REQUESTS_CA_BUNDLE.')
    ap.add_argument('--insecure', action='store_true',
                    help='Skip TLS verification entirely. Last resort.')
    args = ap.parse_args()

    create_log_file()
    crawl(args.seed, args.output, limit=args.limit,
          skip_detail=args.skip_detail, batch=args.batch,
          ca_bundle=args.ca_bundle, insecure=args.insecure)
