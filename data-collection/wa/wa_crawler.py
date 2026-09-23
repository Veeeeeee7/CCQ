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
     column-union flush below.

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

RATING: three columns carry the rating.
  Early_Achiever_Status_Internal__c   e.g. "Level 3+"
  Early_Achiever_Status_External__c   e.g. "Quality Level 3+"
  socrata_earating                    e.g. "3+"
They express the same 5-level Early Achievers scale.

COVERAGE CAVEAT: the payload carries `Exclude_FH_provider_on_public_search__c`.
Family home providers can be flagged as hidden from public search (their address
is a private residence), so the finder is a floor on the licensed FCC population,
not a census. There is no other public per-provider FCC source.

Resume-safe: existing wa_data/wa_records.csv is read back at startup and any
provider already present is skipped; rows with a non-empty `errors` cell are
re-crawled and rewritten in place. Per-section try/except means one broken
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
import sys
import time
import traceback

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://www.findchildcarewa.org'
SEARCH_URL = f'{BASE_URL}/PSS_Search'
APEXREMOTE_URL = f'{BASE_URL}/apexremote'
PROVIDER_URL = f'{BASE_URL}/PSS_Provider?id={{pid}}'

CONTROLLER = 'PSS_SearchController'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36')

# Python doesn't use the macOS keychain, so a python.org build or a
# TLS-intercepting proxy can fail verification although the site is fine.
# Honour the standard env vars; --ca-bundle / --insecure are the escape hatch.
CA_BUNDLE = (os.environ.get('REQUESTS_CA_BUNDLE')
             or os.environ.get('SSL_CERT_FILE')
             or None)

SEED_CSV = 'wa_data/wa_seed.csv'
OUT_CSV = 'wa_data/wa_records.csv'
LOG_FILE = 'wa_crawler_log.txt'

KEY = 'provider_id'                 # resume key

QUERY_BATCH = 25                    # ids per queryProviders call
FLUSH_EVERY = 50                    # providers between full CSV rewrites
MIN_DELAY, MAX_DELAY = 0.4, 1.0     # polite pause between detail-page GETs

# The portal has transient server blips, so both calls back off and retry.
RETRIES = 3                         # total attempts per detail GET / API call
BACKOFF_BASE = 2.0                  # seconds; doubled each attempt
MAX_5XX_IN_A_ROW = 10               # consecutive server errors before stopping

# Columns pinned to the front of the CSV for readability; everything else is
# appended alphabetically by the column-union flush.
FRONT_COLS = [KEY, 'found_zip', 'Account_Name_Display_Label__c',
              'Latest_License_Facility_Type_Name__c',
              'Early_Achiever_Status_Internal__c',
              'Early_Achiever_Status_External__c',
              'socrata_earating', 'socrata_eaparticipation', 'errors']


def create_log_file(path=LOG_FILE):
    """Open the log for append with a run header, so evidence of transient
    failures survives later runs."""
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"\n===== run {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                f"=====\n")


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


# The detail page renders its facts as `div.form-group > label.control-label`
# ("Key:") followed by a sibling holding `p.form-control-static` ("Value").
# parse_header() harvests every such pair, because several fields (License
# Number, the two Languages fields) never appear in the API payload. These are
# the labels expected on every page, declared so a provider missing one still
# gets the column; extras are picked up by the column-union flush.
CORE_DETAIL_FIELDS = [
    'detail_license_name', 'detail_license_number', 'detail_facility_type',
    'detail_licensed_capacity', 'detail_ages', 'detail_languages_spoken',
    'detail_languages_of_instruction',
]


def _snake(s):
    return re.sub(r'[^a-z0-9]+', '_', (s or '').lower()).strip('_')


def empty_header():
    return {k: None for k in CORE_DETAIL_FIELDS}


def empty_early_achievers():
    # detail_ea_specialization is not scaffolded; parse_early_achievers() emits
    # it only when the page has one.
    return {'detail_ea_status': None}


def empty_contacts():
    return {'contacts_json': None, 'contacts_count': None}


def empty_license_history():
    return {'license_history_json': None, 'license_history_count': None,
            'license_id_current': None}


def empty_complaints():
    return {'complaints_json': None, 'complaints_count': None}


def empty_inspections():
    return {'inspections_json': None, 'inspections_count': None}


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
            # Take the text after the label wherever in the paragraph it
            # sits, falling back to the text after the first colon.
            para = _clean(strong.parent.get_text(' ', strip=True)) or ''
            tail = para.split(txt, 1)[1] if txt and txt in para else ''
            if not tail.strip(' :') and ':' in para:
                tail = para.split(':', 1)[1]
            value = _clean(tail.strip().lstrip(':').strip())
            if value:
                out['detail_ea_specialization'] = value
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


def _get_with_retry(session, url, retries=RETRIES, base=BACKOFF_BASE,
                    what='GET'):
    """GET with exponential backoff. Returns (response, server_error_flag).

    A 4xx is permanent and returned immediately; connection resets, timeouts
    and 5xx are retried.
    """
    last = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404:
                r.raise_for_status()
            if r.status_code >= 500:
                last = requests.HTTPError(f'{r.status_code} from {url}')
                raise last
            r.raise_for_status()
            return r, False
        except requests.HTTPError as e:
            status = getattr(e.response, 'status_code', None)
            if status is not None and 400 <= status < 500:
                return None, False          # permanent; do not hammer the site
            last = e
        except Exception as e:              # timeout, reset, DNS, TLS
            last = e
        if attempt < retries - 1:
            wait = base * (2 ** attempt)
            log(f'    . {what} attempt {attempt + 1}/{retries} failed '
                f'({last!r}); retrying in {wait:.0f}s')
            time.sleep(wait)
    log(f'    ! {what} failed after {retries} attempt(s): {last!r}')
    return None, True


def scrape_detail(session, pid, errors):
    """GET PSS_Provider?id=... and run every section under its own try/except.

    Returns (row, server_error). `server_error` is True only when the page was
    lost to something that looked transient after every retry, so crawl() can
    stop rather than burn through the rest of the seed against a sick server.
    """
    row = {}
    r, server_error = _get_with_retry(session, PROVIDER_URL.format(pid=pid),
                                      what=f'detail {pid}')
    if r is None:
        errors.append('detail_fetch')
        log(f'    ! detail fetch failed for {pid}')
        for _, _, empty in DETAIL_SECTIONS:
            row.update(empty())
        return row, server_error
    try:
        soup = BeautifulSoup(r.text, 'html.parser')
    except Exception as e:
        errors.append('detail_fetch')
        log(f'    ! detail parse failed for {pid}: {e!r}')
        for _, _, empty in DETAIL_SECTIONS:
            row.update(empty())
        return row, False

    for name, parse, empty in DETAIL_SECTIONS:
        try:
            row.update(parse(soup))
        except Exception:
            errors.append(name)
            row.update(empty())
            log(f'    ! section {name} failed for {pid}\n'
                f'{traceback.format_exc(limit=1)}')
    return row, False


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


def load_socrata(path=SEED_CSV):
    """{provider_id: {socrata_*: value}}. Absent for FCC homes by design.

    Seed rows with no Socrata match stay out of the lookup, so an unmatched
    provider gets the columns absent rather than blank."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # wacompassid is the join key; it only restates provider_id.
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


# Salesforce omits null fields, so the schema grows: flush takes the column union.
def load_existing(path, retry_errors=True):
    """Read the output file back. Returns (rows, done, repairable).

    `done` is the resume set. With retry_errors, a row whose `errors` cell is
    non-empty is left out of it, and `repairable` maps its id to its row index
    so crawl() can rewrite it in place rather than append a duplicate.
    """
    if not os.path.exists(path):
        return [], set(), {}
    df = pd.read_csv(path, dtype=str)
    if KEY not in df.columns:
        return [], set(), {}
    df[KEY] = _norm_id(df[KEY])
    rows = df.to_dict('records')

    def _failed(row):
        value = row.get('errors')
        return bool(value) and str(value).strip() not in ('', 'nan')

    repairable = {}
    if retry_errors and 'errors' in df.columns:
        for i, row in enumerate(rows):
            if _failed(row):
                repairable.setdefault(row[KEY], i)

    done = {pid for pid in df[KEY] if pid not in repairable}
    return rows, done, repairable


def flush(path, rows):
    cols = set()
    for r in rows:
        cols.update(r)
    ordered = [c for c in FRONT_COLS if c in cols]
    ordered += sorted(c for c in cols if c not in FRONT_COLS)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pd.DataFrame(rows, columns=ordered).to_csv(path, index=False)


def _place(rows, repairable, pid, row):
    """Append a new provider, or rewrite a failed one in place.

    Downstream stages join on row position, so a repair must land at the row's
    existing index. The old cells are kept as a base so a section this run
    could not read does not blank a value the first run did get.
    """
    index = repairable.get(pid)
    if index is None:
        rows.append(row)
        return
    merged = dict(rows[index])
    if row.get('errors'):
        # Still broken, or broken in a different section: keep whatever the
        # first run did manage to read rather than blanking it.
        merged.update({k: v for k, v in row.items() if v is not None})
    else:
        # A clean read is authoritative, Nones included -- otherwise a section
        # that has legitimately emptied keeps a stale blob next to its new 0.
        merged.update(row)
    merged['errors'] = row.get('errors')
    merged[KEY] = pid
    rows[index] = merged


def _report_coverage(seed, rows):
    """Log seeded-vs-crawled; returns the number of seeded ids with no row."""
    seeded = set(_norm_id(seed['provider_id']))
    crawled = {r.get(KEY) for r in rows}
    missing = sorted(seeded - crawled)
    failed = [r.get(KEY) for r in rows
              if r.get('errors') and str(r.get('errors')).strip() not in ('', 'nan')]
    log(f'Coverage: {len(crawled)}/{len(seeded)} seeded provider(s) '
        f'({100.0 * len(crawled) / max(len(seeded), 1):.2f}%); '
        f'{len(failed)} row(s) carry an error')
    if missing:
        log(f'  ! {len(missing)} seeded id(s) have no row at all '
            f'(e.g. {missing[:3]}) -- re-run to pick them up')
    if failed:
        log(f'  ! {len(failed)} row(s) still carry an error '
            f'(e.g. {failed[:3]}) -- re-run to repair them in place')
    return len(missing)


def crawl(seed_csv=SEED_CSV, out_csv=OUT_CSV, limit=None, skip_detail=False,
          batch=QUERY_BATCH, ca_bundle=None, insecure=False,
          retry_errors=True):
    seed = pd.read_csv(seed_csv, dtype=str)
    seed['provider_id'] = _norm_id(seed['provider_id'])
    zip_of = dict(zip(seed['provider_id'], seed.get('found_zip', '')))

    rows, done, repairable = load_existing(out_csv, retry_errors)
    todo = [p for p in seed['provider_id'] if p not in done]
    if limit:
        todo = todo[:limit]

    log(f'Seed: {len(seed)} providers | already done: {len(done)} | '
        f'to crawl: {len(todo)}')
    if repairable:
        log(f'  {len(repairable)} previously-failed row(s) will be re-crawled '
            f'and rewritten IN PLACE (--no-retry-errors to skip them)')
    if not todo:
        log('Nothing to do.')
        return _report_coverage(seed, rows)

    socrata = load_socrata(seed_csv)

    client = RemotingClient(log_fn=log, ca_bundle=ca_bundle,
                            insecure=insecure).bootstrap()
    session = client.s

    processed = 0
    consecutive_server_errors = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]

        # A failed call abandons the whole chunk and its detail GETs, so retry.
        recs, failed = None, None
        for attempt in range(RETRIES):
            try:
                recs = client.call('queryProviders', [chunk]) or []
                break
            except Exception as e:
                failed = e
                if attempt < RETRIES - 1:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    log(f'  . queryProviders batch {i//batch} attempt '
                        f'{attempt + 1}/{RETRIES} failed ({e!r}); retrying in '
                        f'{wait:.0f}s')
                    time.sleep(wait)
        if recs is None:
            log(f'  ! queryProviders failed for batch {i//batch} after '
                f'{RETRIES} attempt(s): {failed!r}; {len(chunk)} id(s) '
                f'(e.g. {chunk[:2]}) left uncrawled -- _report_coverage will '
                f'name them at the end and the next run picks them up')
            consecutive_server_errors += 1
            if consecutive_server_errors >= MAX_5XX_IN_A_ROW:
                log(f'  ! {consecutive_server_errors} consecutive API failures '
                    f'-- stopping so the portal is left alone')
                break
            # No stub row: an id with no row is not in `done`, so the next run
            # crawls it normally.
            continue
        consecutive_server_errors = 0

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
                detail, server_error = scrape_detail(session, pid, errors)
                row.update(detail)
                if server_error:
                    consecutive_server_errors += 1
                else:
                    consecutive_server_errors = 0
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            else:
                for _, _, empty in DETAIL_SECTIONS:
                    row.update(empty())

            row.update(socrata.get(pid, {}))
            row['errors'] = ';'.join(errors) if errors else None

            _place(rows, repairable, pid, row)
            processed += 1

            if processed % FLUSH_EVERY == 0:
                flush(out_csv, rows)
                log(f'  … {processed}/{len(todo)} crawled (flushed {len(rows)} rows)')

            if consecutive_server_errors >= MAX_5XX_IN_A_ROW:
                log(f'  ! {consecutive_server_errors} consecutive server '
                    f'failures -- stopping so the portal is left alone')
                break
        if consecutive_server_errors >= MAX_5XX_IN_A_ROW:
            break

    flush(out_csv, rows)
    log(f'Done. {processed} newly crawled; {len(rows)} rows total -> {out_csv}')
    return _report_coverage(seed, rows)


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
    ap.add_argument('--no-retry-errors', dest='retry_errors',
                    action='store_false',
                    help='Treat every row already in the output as done, '
                         'including the ones whose detail page failed. The '
                         'default re-crawls those rows and rewrites them in '
                         'place (it never appends, so the row count and the '
                         'positional join are preserved).')
    args = ap.parse_args()

    create_log_file()
    missing = crawl(args.seed, args.output, limit=args.limit,
                    skip_detail=args.skip_detail, batch=args.batch,
                    ca_bundle=args.ca_bundle, insecure=args.insecure,
                    retry_errors=args.retry_errors)
    # Non-zero exit when the crawl is short of its seed.
    sys.exit(1 if missing else 0)
