"""
Wisconsin Child Care Finder crawler — Round 1 (text), selectors wired
=====================================================================

Adapted from ca_crawler.py. Backbone is unchanged: one row per provider
location, per-section try/except, resume via the output CSV, randomized delay,
fresh context per provider.

Blazor Server specifics (see wi_capture.py for why): render in a real browser,
never wait on networkidle, keep a polite delay + realistic UA (reCAPTCHA v3).

CONFIRMED FROM THE DOM CAPTURES
-------------------------------
Routing: /ProviderDetails?ProviderNumber={pn}&LocationNumber={ln}&Provider={pn}&CCF=Y
  - IDs are used in zero-padded form, exactly as stored in the directory
    (e.g. ProviderNumber=0000555700, LocationNumber=001).
  - Live URLs also carried UserSessionId + SearchId from a search; the page
    loads from the direct URL above without them.
Sections are accordions with stable heading ids; star rating = count of
span.fa-star; regulation = up to 3 desktop tables (skip .Phone duplicates);
provider-reported = label/value under .dcf-red-font.

OPEN ITEM
---------
goto_provider() implements the direct-URL path. If the site requires a search-
established session, fill establish_search_session() with the click/enter flow
and set REQUIRES_SEARCH_SESSION = True.

Round 1 = text (this run). Round 2 = PDF download, coded but only with --download-pdfs.

Deps: pip install playwright pandas beautifulsoup4  (and: playwright install chromium)
"""

import argparse
import json
import os
import random
import re
import time
import traceback

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

BASE_URL = 'https://childcarefinder.wisconsin.gov'

# Direct-URL form. IDs go in zero-padded, as stored in the directory.
PROVIDER_URL_TEMPLATE = (BASE_URL + '/ProviderDetails'
                         '?ProviderNumber={pn}&LocationNumber={ln}&Provider={pn}&CCF=Y')

# Set True if the detail page requires a search-established server session.
# When True, establish_search_session() runs once per provider (or per batch).
REQUIRES_SEARCH_SESSION = False

# Section accordion heading ids (stable).
HEADING_YOUNGSTAR = 'youngstarDetailsHeading'
HEADING_REGULATION = 'regulationDetailsHeading'
HEADING_PROVIDER_REPORTED = 'providerReportedDetailsHeading'

# Signal that the detail data has loaded. The waiting/not-found screen shares
# the page chrome but contains NONE of the detail components — no accordion
# sections (it shows "Active provider information was not found for: Provider
# Number ''"). The regulation section's heading id therefore appears only once
# the provider's data has actually rendered. (Every regulated provider — i.e.
# every row in the seed — has a Regulation Details section.)
SEL_PROVIDER_LOADED = '#regulationDetailsHeading'

# Max time to wait for that signal before skipping to the next provider (ms).
PROVIDER_WAIT_MS = 60000

MAX_DOCS = 60  # safety cap on documents collected per provider (Round 2)

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/120.0.0.0 Safari/537.36')


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def create_log_file(path='wi_crawler_log.txt'):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file='wi_crawler_log.txt'):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


# ---------------------------------------------------------------------------
# seed list: union the licensed + certified directories
# ---------------------------------------------------------------------------

DIR_RENAME = {
    'Provider Number': 'provider_number', 'Location Number': 'location_number',
    'Application Type': 'application_type', 'County': 'county',
    'Facility Name': 'facility_name', 'Facility Number': 'facility_number',
    'Line Address 1': 'address_1', 'Line Address 2': 'address_2', 'City': 'city',
    'Zip Code': 'zip', 'Contact Name': 'contact_name',
    'Contact Phone': 'contact_phone', 'Capacity': 'capacity',
    'From Age': 'from_age', 'To Age': 'to_age', 'Hours': 'hours',
    'Months': 'months', 'Full Time': 'full_time', 'Star Level ': 'star_level',
}


def _norm_id(series, width):
    """Force an ID column to a zero-padded string. Robust to a source CSV that
    stored the column as integers/floats (which drops leading zeros): casting
    to str and zero-padding to the known width restores them — e.g. 555670 ->
    '0000555670', 4 -> '004'. A trailing '.0' from float storage is stripped."""
    return (series.astype(str).str.strip()
            .str.replace(r'\.0$', '', regex=True)
            .str.replace(r'\D', '', regex=True)
            .str.zfill(width))


def _load_one_directory(df, regulation_type):
    df = df.loc[:, [c for c in df.columns if c]]
    date_col = next((c for c in df.columns if c.strip().endswith('Date')), None)
    rename = dict(DIR_RENAME)
    if date_col:
        rename[date_col] = 'regulated_date'
    df = df.rename(columns=rename)
    df = df[df['provider_number'].astype(str).str.strip().ne('')].copy()
    df['regulation_type'] = regulation_type
    for c in df.columns:
        df[c] = df[c].map(lambda v: re.sub(r'\s+', ' ', v).strip()
                          if isinstance(v, str) else v)
    # Keep IDs as strings with leading zeros preserved (provider = 10 digits,
    # location = 3). zfill restores them even if the source dropped the zeros.
    df['provider_number'] = _norm_id(df['provider_number'], 10)
    df['location_number'] = _norm_id(df['location_number'], 3)
    return df


def load_seed(seed_csv):
    # The two regulation directories were stacked into one seed, tagged by
    # seed_source. Each half kept the two-line preamble the download ships
    # with, so the real column names sit in a row rather than in the header --
    # find that row and promote it instead of assuming a fixed offset.
    whole = pd.read_csv(seed_csv, dtype=str, keep_default_na=False)
    frames = []
    for tag, label in (('lcc', 'licensed'), ('ccc', 'certified')):
        block = whole[whole['seed_source'] == tag].drop(columns='seed_source')
        if block.empty:
            continue
        header_rows = block.index[
            block.iloc[:, 0].str.strip().eq('Provider Number')]
        if header_rows.empty:
            raise ValueError(f'No header row in the {tag} half of {seed_csv}')
        at = header_rows[0]
        part = block.loc[at + 1:].copy()
        part.columns = [str(v).strip() for v in block.loc[at]]
        frames.append(_load_one_directory(part.reset_index(drop=True), label))
    if not frames:
        raise FileNotFoundError(f'No provider rows in {seed_csv}.')
    df = pd.concat(frames, ignore_index=True)
    df['provider_location'] = df['provider_number'] + '-' + df['location_number']
    df = df.drop_duplicates(subset='provider_location').reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# resume helpers
# ---------------------------------------------------------------------------

def all_columns():
    cols = ['provider_location', 'provider_number', 'location_number',
            'facility_number', 'regulation_type', 'provider_url']
    for fn in (empty_youngstar, empty_regulation, empty_provider_reported,
               empty_documents):
        cols.extend(fn().keys())
    cols.append('errors')
    return cols


def append_row(row, output_csv):
    cols = all_columns()
    full = {}
    for c in cols:
        v = row.get(c)
        if isinstance(v, str):
            v = re.sub(r'[\r\n]+', ' ', v).strip() or None
        full[c] = v
    parent = os.path.dirname(output_csv)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    pd.DataFrame([full], columns=cols).to_csv(
        output_csv, mode='a', header=not exists, index=False)


def load_completed(output_csv):
    if not output_csv or not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv, dtype=str, usecols=['provider_location'])
        return set(df['provider_location'].dropna().str.strip())
    except Exception as e:
        log(f'Could not read existing {output_csv}: {e}')
        return set()


# ---------------------------------------------------------------------------
# render wait + navigation
# ---------------------------------------------------------------------------

def wait_for_render(page, selector=None, settle_ms=1500, timeout_ms=30000,
                    poll_ms=400):
    """Blazor-safe wait. Never networkidle."""
    if selector:
        page.wait_for_selector(selector, timeout=timeout_ms)
        return
    deadline = time.time() + timeout_ms / 1000.0
    last_len, stable_since = -1, None
    while time.time() < deadline:
        try:
            cur = page.evaluate(
                "() => (document.body && document.body.innerText || '').length")
        except Exception:
            cur = 0
        if cur > 0 and cur == last_len:
            if stable_since is None:
                stable_since = time.time()
            elif (time.time() - stable_since) * 1000 >= settle_ms:
                return
        else:
            stable_since, last_len = None, cur
        time.sleep(poll_ms / 1000.0)


def wait_for_provider(page, timeout_ms=PROVIDER_WAIT_MS):
    """Wait until the provider's detail data has loaded, signalled by the
    Regulation Details section appearing in the DOM. The waiting/not-found
    screen has no such section, so this cleanly distinguishes a loaded provider
    from a page still resolving (or a genuine miss). Returns True when it
    appears, or False after timeout_ms so the caller skips to the next provider."""
    try:
        page.wait_for_selector(SEL_PROVIDER_LOADED, state='attached', timeout=timeout_ms)
        return True
    except PWTimeout:
        return False


def establish_search_session(page, rec):
    """★ ONLY needed if REQUIRES_SEARCH_SESSION. Fill with the search flow you
    described (enter search criteria, submit, click the matching result) and
    return the resulting detail URL, or None. The captured live URLs show the
    detail page carries UserSessionId + SearchId set during search."""
    raise NotImplementedError('Fill in the search flow, then set '
                              'REQUIRES_SEARCH_SESSION = True.')


def goto_provider(page, rec):
    """Navigate to a provider's detail page; return URL or None (skip)."""
    if REQUIRES_SEARCH_SESSION:
        url = establish_search_session(page, rec)
        if url and wait_for_provider(page):
            return page.url
        return None
    # direct-URL path: the route uses the zero-padded IDs as-is, e.g.
    # ?ProviderNumber=0000555700&LocationNumber=001&Provider=0000555700&CCF=Y
    pn = rec['provider_number']   # padded, e.g. '0000555700'
    ln = rec['location_number']   # padded, e.g. '001'
    url = PROVIDER_URL_TEMPLATE.format(pn=pn, ln=ln)
    page.goto(url, wait_until='domcontentloaded')
    # wait up to PROVIDER_WAIT_MS for the detail data to load; skip if it doesn't
    if not wait_for_provider(page):
        return None
    return page.url


# ---------------------------------------------------------------------------
# parsing helpers (operate on a BeautifulSoup of the rendered page)
# ---------------------------------------------------------------------------

def _clean(s):
    if s is None:
        return None
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s or None


def _section_body(soup, heading_id):
    h = soup.find(id=heading_id)
    if not h:
        return None
    item = h.find_parent(class_='accordion-item')
    return item.find(class_='accordion-body') if item else None


def _parse_grid(table):
    """Turn a .Grid table into a list of {header: cell_text} dicts."""
    headers = [th.get_text(' ', strip=True) for th in table.find_all('th')]
    rows = []
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if not tds:
            continue
        cells = [td.get_text(' ', strip=True) for td in tds]
        if headers and len(headers) == len(cells):
            rows.append(dict(zip(headers, cells)))
        else:
            rows.append({f'col{i}': c for i, c in enumerate(cells)})
    return rows


def _desktop_grids(body):
    """Desktop .Grid tables only (skip .Phone mobile duplicates)."""
    out = []
    for t in body.find_all('table'):
        cls = t.get('class') or []
        if 'Grid' in cls and 'Phone' not in cls:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# section extractors (Round 1 = text)
# ---------------------------------------------------------------------------

def empty_youngstar():
    return {'youngstar_star_rating': None,
            'youngstar_unique_services': None,
            'youngstar_section_text': None}


def crawl_youngstar(soup):
    out = empty_youngstar()
    body = _section_body(soup, HEADING_YOUNGSTAR)
    if body is None:
        return out
    out['youngstar_star_rating'] = len(body.select('span.fa-star'))
    svc = body.find(string=lambda s: s and 'Unique Program Services' in s)
    if svc:
        tbl = svc.find_parent('table')
        if tbl:
            lines = [_clean(li) for li in tbl.get_text('\n').split('\n')
                     if _clean(li) and 'Unique Program Services' not in li]
            out['youngstar_unique_services'] = ' | '.join(lines) or None
    out['youngstar_section_text'] = _clean(body.get_text(' '))
    return out


def empty_regulation():
    return {'regulation_enforcement_json': None,
            'regulation_monitoring_json': None,
            'regulation_violations_json': None,
            'regulation_section_text': None}


def crawl_regulation(soup):
    out = empty_regulation()
    body = _section_body(soup, HEADING_REGULATION)
    if body is None:
        return out
    for t in _desktop_grids(body):
        headers = [th.get_text(strip=True) for th in t.find_all('th')]
        hset = set(headers)
        rows = _parse_grid(t)
        if {'Appeal', 'Decision'} & hset:
            out['regulation_enforcement_json'] = json.dumps(rows, ensure_ascii=False)
        elif 'Rule Monitoring' in hset:
            out['regulation_monitoring_json'] = json.dumps(rows, ensure_ascii=False)
        elif 'Rule Number' in hset and 'Description' in hset:
            out['regulation_violations_json'] = json.dumps(rows, ensure_ascii=False)
    out['regulation_section_text'] = _clean(body.get_text(' '))
    return out


def empty_provider_reported():
    return {'pr_special_types_of_care': None,
            'pr_program_philosophy': None,
            'pr_vacancies': None,
            'pr_waitlist': None,
            'pr_section_text': None}


_PR_LABEL_MAP = {
    'Special Types of Care Available': 'pr_special_types_of_care',
    'Program Philosophy': 'pr_program_philosophy',
    'Vacancies': 'pr_vacancies',
    'Waitlist': 'pr_waitlist',
}


def crawl_provider_reported(soup):
    out = empty_provider_reported()
    body = _section_body(soup, HEADING_PROVIDER_REPORTED)
    if body is None:
        return out
    # Each label is an <h3 class="dcf-red-font"> inside its own col-* div; the
    # value is the rest of that column (a table, span, or div).
    for lab in body.select('.dcf-red-font'):
        label = _clean(lab.get_text())
        key = _PR_LABEL_MAP.get(label)
        if not key:
            continue
        col = lab.parent
        full = _clean(col.get_text(' ')) or ''
        val = _clean(full[len(label):]) if label and full.startswith(label) else full
        out[key] = val or None
    out['pr_section_text'] = _clean(body.get_text(' '))
    return out


# ---------------------------------------------------------------------------
# documents (Round 1 records refs; Round 2 downloads them)
# ---------------------------------------------------------------------------

DOC_PATTERNS = ('ViewMonitoringDocument', 'ViewRatingReport', 'ViewDocument')


def empty_documents():
    return {'num_documents': None, 'documents_json': None}


def collect_documents(soup):
    out = empty_documents()
    seen, docs = set(), []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if any(p in href for p in DOC_PATTERNS):
            if href in seen:
                continue
            seen.add(href)
            url = href if href.startswith('http') else BASE_URL + href
            kind = ('rating' if 'Rating' in href
                    else 'monitoring' if 'Monitoring' in href else 'other')
            docs.append({'label': _clean(a.get_text()), 'type': kind, 'url': url})
            if len(docs) >= MAX_DOCS:
                break
    out['num_documents'] = len(docs)
    out['documents_json'] = json.dumps(docs, ensure_ascii=False) if docs else None
    return out


def download_documents(page, rec, downloads_folder, docs):
    """Round 2. Fetch each document URL in a fresh tab using the live session.
    Only runs with --download-pdfs."""
    if not docs:
        return
    folder = os.path.join(downloads_folder,
                          f"{rec['provider_number']}_{rec['location_number']}")
    os.makedirs(folder, exist_ok=True)
    for i, d in enumerate(docs, 1):
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', (d.get('label') or f'doc{i}'))[:60]
        dest = os.path.join(folder, f'{i:02d}_{d["type"]}_{safe}.pdf')
        try:
            with page.expect_download(timeout=30000) as dl:
                page.evaluate("(u) => window.open(u, '_blank')", d['url'])
            dl.value.save_as(dest)
            time.sleep(0.4)
        except Exception:
            log(f'    ! doc download failed: {d["url"]}')


# ---------------------------------------------------------------------------
# main crawler
# ---------------------------------------------------------------------------

def crawler(records, output_csv='wi_data/wi_records.csv',
            downloads_folder='wi_data/downloads', headless=False, start_index=0,
            limit=None, download_pdfs=False, executable_path=None,
            channel='chrome', user_data_dir='wi_data/wi_profile',
            delay_range=(5, 10)):
    rows = []
    completed = load_completed(output_csv)
    if completed:
        log(f'Resuming: {len(completed)} already in {output_csv} — skipping.')
    end = len(records) if limit is None else min(len(records), start_index + limit)

    # Why a persistent profile instead of a fresh context per provider:
    # childcarefinder is a Blazor app gated by reCAPTCHA v3 (invisible scoring).
    # A brand-new context has zero Google cookies / trust, so it scores like a
    # bot and the server never renders the provider data — the page stays on the
    # empty "Provider Number ''" placeholder. A persistent profile lets that
    # trust accumulate across the run. Combined with a real-Chrome channel and
    # the navigator.webdriver patch below, this is what gets the data to render.
    # If a dedicated profile still scores too low, point --user-data-dir at a
    # COPY of your real Chrome profile (macOS:
    # ~/Library/Application Support/Google/Chrome) — quit Chrome first, since it
    # locks the profile.
    os.makedirs(user_data_dir, exist_ok=True)

    with sync_playwright() as p:
        launch_kwargs = {
            'user_data_dir': user_data_dir,
            'headless': headless,
            'viewport': {'width': 1400, 'height': 1000},
            'args': ['--disable-blink-features=AutomationControlled'],
        }
        # Prefer a real Chrome install (channel) and let it present its own,
        # self-consistent user-agent + client hints — spoofing only the UA
        # string is itself a detection tell. Fall back to a spoofed UA only when
        # running an explicit binary or Playwright's bundled Chromium.
        if executable_path:
            launch_kwargs['executable_path'] = executable_path
            launch_kwargs['user_agent'] = UA
        elif channel:
            launch_kwargs['channel'] = channel
        else:
            launch_kwargs['user_agent'] = UA
        context = p.chromium.launch_persistent_context(**launch_kwargs)
        # Hide the automation flag reCAPTCHA keys on (navigator.webdriver=true).
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        # Warm-up: hit the site root once so reCAPTCHA v3 executes and the
        # profile picks up trust before we start requesting detail pages.
        try:
            warm = context.new_page()
            warm.goto(BASE_URL, wait_until='domcontentloaded')
            time.sleep(3)
            warm.close()
        except Exception:
            log('  ! warm-up visit failed (continuing anyway)')

        index = start_index
        try:
            for index in range(start_index, end):
                rec = records[index]
                key = rec['provider_location']
                if key in completed:
                    continue

                row = {'provider_location': key,
                       'provider_number': rec['provider_number'],
                       'location_number': rec['location_number'],
                       'facility_number': rec.get('facility_number'),
                       'regulation_type': rec.get('regulation_type')}
                errors = []
                # New PAGE per provider (not a new context): the persistent
                # context is shared for the whole run so cookies / reCAPTCHA
                # trust carry over between providers.
                page = context.new_page()
                try:
                    url = goto_provider(page, rec)
                    row['provider_url'] = url
                    if not url:
                        row.update(empty_youngstar() | empty_regulation()
                                   | empty_provider_reported() | empty_documents())
                        row['errors'] = 'not_found'
                        log(f'[{index}] NOT FOUND: {key}')
                    else:
                        soup = BeautifulSoup(page.content(), 'html.parser')
                        for name, fn, empty_fn in (
                            ('youngstar', crawl_youngstar, empty_youngstar),
                            ('regulation', crawl_regulation, empty_regulation),
                            ('provider_reported', crawl_provider_reported, empty_provider_reported),
                            ('documents', collect_documents, empty_documents),
                        ):
                            try:
                                row.update(fn(soup))
                            except Exception:
                                errors.append(name)
                                row.update(empty_fn())
                                log(f'  ! {name} failed for {key}: '
                                    f'{traceback.format_exc().splitlines()[-1]}')

                        if download_pdfs and row.get('documents_json'):
                            try:
                                download_documents(page, rec, downloads_folder,
                                                   json.loads(row['documents_json']))
                            except Exception:
                                errors.append('download')

                        row['errors'] = ','.join(errors)
                        log(f"[{index}] {'OK' if not errors else 'PARTIAL'}: {key} "
                            f"({row.get('youngstar_star_rating')}★)")

                    rows.append(row)
                    append_row(row, output_csv)
                    completed.add(key)
                except Exception:
                    log(f'[{index}] EXCEPTION: {key}')
                    log(traceback.format_exc())
                    row.update(empty_youngstar() | empty_regulation()
                               | empty_provider_reported() | empty_documents())
                    row['errors'] = 'exception'
                    rows.append(row)
                    append_row(row, output_csv)
                    completed.add(key)
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass

                if index < end - 1:
                    delay = random.uniform(*delay_range)
                    log(f'  ...sleeping {delay:.1f}s')
                    time.sleep(delay)
        except Exception:
            log(f'CRASHED at index {index}')
            log(traceback.format_exc())
        finally:
            context.close()
    return pd.DataFrame(rows), index


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='WI Child Care Finder crawler (Round 1).')
    ap.add_argument('--seed', default='wi_data/wi_seed.csv')
    ap.add_argument('--output', default='wi_data/wi_records.csv')
    ap.add_argument('--downloads', default='wi_data/downloads')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--download-pdfs', action='store_true')
    ap.add_argument('--executable-path', default=None)
    ap.add_argument('--user-data-dir', default='wi_data/wi_profile',
                    help='Persistent Chrome profile dir (keeps reCAPTCHA trust '
                         'across the run). Point at a COPY of your real Chrome '
                         'profile if a dedicated one still scores too low.')
    ap.add_argument('--channel', default='chrome',
                    help="Browser channel: 'chrome' (recommended), 'chrome-beta', "
                         "'msedge', or '' to use Playwright's bundled Chromium.")
    ap.add_argument('--delay-min', type=float, default=5)
    ap.add_argument('--delay-max', type=float, default=10)
    args = ap.parse_args()

    create_log_file()
    seed = load_seed(args.seed)
    records = seed.to_dict('records')
    log(f'Seed: {len(records)} locations '
        f'({(seed.regulation_type=="licensed").sum()} licensed, '
        f'{(seed.regulation_type=="certified").sum()} certified).')
    if REQUIRES_SEARCH_SESSION:
        log('NOTE: REQUIRES_SEARCH_SESSION is on — establish_search_session() must be filled.')
    crawler(records, output_csv=args.output, downloads_folder=args.downloads,
            headless=args.headless, start_index=args.start_index, limit=args.limit,
            download_pdfs=args.download_pdfs, executable_path=args.executable_path,
            channel=(args.channel or None), user_data_dir=args.user_data_dir,
            delay_range=(args.delay_min, args.delay_max))