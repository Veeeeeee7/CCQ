"""
nc_crawler.py — North Carolina child care crawler (modern portal).

Reads the provided seed CSV and, for each Facility_ID, drives the DCDEE
"Search for Child Care" app (ncchildcare.ncdhhs.gov/childcaresearch) to pull the
facility's star/license info, DCDEE visit history, and contact details into a
rectangular CSV.

The portal is an ASP.NET WebForms app (DNN + Telerik) whose state lives in
VIEWSTATE, so we drive the real UI with Playwright rather than replaying
postbacks: load search -> type the Facility_ID into the "Search by License
Number" box -> click search -> click the matching result row -> the
FacilityDetail panel renders (all sections present in the DOM even when
collapsed) -> parse by control-id suffix. The host is crawl-allowed.

    python nc_crawler.py --limit 5                       # smoke test (visible browser)
    python nc_crawler.py --headless --delay-min 3 --delay-max 7   # full run

Resume-safe: re-running skips Facility_IDs already in the output CSV. Each
section is parsed under its own try/except so one bad field never kills a row;
failures land in the `errors` column (including `not_found` and `id_mismatch`).

Deps: pip install playwright pandas beautifulsoup4 && playwright install chromium
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

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

SEARCH_URL = 'https://ncchildcare.ncdhhs.gov/childcaresearch'

# Selectors, suffix-matched so the volatile DNN module number (ctr1464) doesn't
# matter. Confirmed against a captured detail page.
LICENSE_INPUT_SEL = ('input[id*="txtLicenseNumber" i], input[id*="LicNum" i], '
                     'input[id*="License" i][type="text"]')
SEARCH_BUTTON_SEL = '[id*="btnSearchLicNum"]'
RESULTS_GRID_SEL = '[id*="rgSearchResults"]'
DETAIL_READY_SEL = '[id*="FacilityDetail_rptBasicFacilityInfo"]'

SEED_COLUMNS = ['facility_id', 'facility_name', 'facility_type', 'county',
                'license_number']

LOG_FILE = 'nc_crawler_log.txt'
PROFILE_DIR = 'nc_data/nc_profile'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_WORD_STARS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}


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
# transport: Playwright driving the WebForms search-by-license-number flow
# ---------------------------------------------------------------------------

class PortalFetcher:
    """Drives the search app and returns the rendered FacilityDetail DOM. One
    browser page is reused across facilities; each call starts fresh at the
    search URL (clean VIEWSTATE)."""

    def __init__(self, headless=True, user_data_dir=PROFILE_DIR):
        from playwright.sync_api import sync_playwright
        os.makedirs(user_data_dir, exist_ok=True)
        self._pw = sync_playwright().start()
        self.context = self._pw.chromium.launch_persistent_context(
            user_data_dir=user_data_dir, headless=headless,
            viewport={'width': 1400, 'height': 1000},
            args=['--disable-blink-features=AutomationControlled'])
        self._pg = None

    def _page(self):
        if self._pg is None or self._pg.is_closed():
            self._pg = self.context.new_page()
        return self._pg

    def get_facility(self, rec):
        """Return ({'summary': {'url', 'soup'}}, url) or ({}, url) if no detail."""
        fid = rec['facility_id']
        page = self._page()
        try:
            page.goto(SEARCH_URL, wait_until='domcontentloaded')

            box = page.locator(LICENSE_INPUT_SEL).first
            if box.count() == 0:
                box = page.locator('input[type="text"]:visible').first
            box.fill(str(fid))

            btn = page.locator(SEARCH_BUTTON_SEL).first
            if btn.count() > 0:
                btn.click()
            else:
                box.press('Enter')

            # search is an async (UpdatePanel) postback — let it land
            try:
                page.wait_for_load_state('networkidle', timeout=12000)
            except Exception:
                pass

            if page.locator(DETAIL_READY_SEL).count() == 0:
                self._click_result_row(page, fid)

            try:
                page.wait_for_selector(DETAIL_READY_SEL, timeout=20000)
            except Exception:
                page.wait_for_timeout(2000)

            soup = BeautifulSoup(page.content(), 'html.parser')
            if soup.find(id=re.compile('FacilityDetail_rptBasicFacilityInfo')) is None:
                return {}, page.url   # no detail -> not_found
            return {'summary': {'url': page.url, 'soup': soup}}, page.url
        except Exception:
            log(f'    ! fetch failed for {fid}: '
                f'{traceback.format_exc().splitlines()[-1]}')
            return {}, None

    def _click_result_row(self, page, fid):
        """Click into the facility's detail from the Telerik results grid.
        Targets the DATA row carrying this license number (not a column header /
        sort link / pager), then clicks its link (or the row itself)."""
        grid = RESULTS_GRID_SEL
        try:
            page.wait_for_selector(f'{grid} tr.rgRow, {grid} tr.rgAltRow', timeout=15000)
        except Exception:
            pass

        want = re.sub(r'\D', '', str(fid)).lstrip('0')
        rows = page.locator(f'{grid} tr.rgRow, {grid} tr.rgAltRow')
        n = rows.count()
        if n == 0:
            rows = page.locator(f'{grid} tr:has(a)')   # non-standard skin fallback
            n = rows.count()

        def _try_click(loc):
            try:
                a = loc.locator('a').first
                (a if a.count() > 0 else loc).click()
                try:
                    page.wait_for_load_state('networkidle', timeout=12000)
                except Exception:
                    page.wait_for_timeout(1500)
                return page.locator(DETAIL_READY_SEL).count() > 0
            except Exception:
                return False

        # 1) the data row whose text contains this license number
        for i in range(n):
            try:
                txt = re.sub(r'\D', '', rows.nth(i).inner_text()).lstrip('0')
            except Exception:
                txt = ''
            if want and want in txt:
                if _try_click(rows.nth(i)):
                    return
                break
        # 2) fallback: the first data row
        if n > 0 and page.locator(DETAIL_READY_SEL).count() == 0:
            if _try_click(rows.first):
                return
        # 3) last resort: any data-row link
        if page.locator(DETAIL_READY_SEL).count() == 0:
            links = page.locator(f'{grid} tr.rgRow a, {grid} tr.rgAltRow a')
            if links.count() > 0:
                _try_click(links.first)

    def close(self):
        try:
            self.context.close()
        finally:
            self._pw.stop()


# ---------------------------------------------------------------------------
# seed loading
# ---------------------------------------------------------------------------

def load_seed(seed_csv):
    if not seed_csv or not os.path.exists(seed_csv):
        raise FileNotFoundError(
            f'Seed CSV not found: {seed_csv}.')
    df = pd.read_csv(seed_csv, dtype=str)
    if 'facility_id' not in df.columns:
        raise ValueError(f'Seed needs a facility_id column; got {list(df.columns)}')
    df['facility_id'] = (df['facility_id'].astype(str).str.strip()
                         .str.replace(r'\.0$', '', regex=True))
    for c in SEED_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = (df[df['facility_id'].notna() & (df['facility_id'] != '')]
          .drop_duplicates(subset='facility_id').reset_index(drop=True))
    return df[SEED_COLUMNS]


# ---------------------------------------------------------------------------
# resume helpers (rectangular CSV, append-per-row)
# ---------------------------------------------------------------------------

def all_columns():
    cols = ['facility_id', 'facility_name', 'facility_type', 'county',
            'facility_url']
    for fn in (empty_star, empty_visits, empty_details):
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
        df = pd.read_csv(output_csv, dtype=str, usecols=['facility_id'])
        return set(df['facility_id'].dropna().str.strip())
    except Exception as e:
        log(f'Could not read existing {output_csv}: {e}')
        return set()


# ---------------------------------------------------------------------------
# parsing helpers (ID-based; detail DOM uses stable control-id suffixes)
# ---------------------------------------------------------------------------

def _clean(s):
    if s is None:
        return None
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s or None


def _detail_soup(pages):
    p = pages.get('summary')
    return p['soup'] if p else None


def _m_one(soup, suffix):
    """Cleaned text of the first element whose id contains `suffix`."""
    if soup is None:
        return None
    el = soup.find(id=re.compile(re.escape(suffix)))
    return _clean(el.get_text(' ')) if el else None


def _m_many(soup, suffix):
    """Texts of all elements whose id matches `suffix_<n>`, ordered by n — i.e.
    the items of an ASP.NET repeater (lblLicenseType_0, _1, ...)."""
    if soup is None:
        return []
    pat = re.compile(re.escape(suffix) + r'_(\d+)$')
    found = []
    for el in soup.find_all(id=True):
        m = pat.search(el['id'])
        if m:
            found.append((int(m.group(1)), _clean(el.get_text(' '))))
    return [t for _, t in sorted(found)]


def _modern_id_mismatch(pages, fid):
    """True if the detail page's license number doesn't match the requested
    Facility_ID (compared ignoring leading zeros)."""
    got = _m_one(_detail_soup(pages), 'LicenseNumberLabel')
    if not got:
        return False
    got_d = re.sub(r'\D', '', got).lstrip('0')
    want_d = re.sub(r'\D', '', str(fid)).lstrip('0')
    return bool(got_d and want_d and got_d != want_d)


def _section_text_by_header(soup, header):
    """Best-effort body text of the accordion following a header toggle whose
    text matches `header`. None if not found (details_section_text still keeps
    everything)."""
    if soup is None:
        return None
    try:
        node = soup.find(lambda t: t.name in ('a', 'div', 'h3', 'h4', 'button')
                         and _clean(t.get_text())
                         and header.lower() in _clean(t.get_text()).lower()
                         and len(_clean(t.get_text())) < len(header) + 25)
        if node:
            body = node.find_next(['div', 'ul', 'table'])
            if body:
                txt = _clean(body.get_text(' '))
                return txt[:1500] if txt else None
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# section extractors — three crawl_*/empty_* pairs, rectangular schema
# ---------------------------------------------------------------------------

def empty_star():
    return {'star_rating': None, 'license_type': None,
            'license_issue_date': None, 'license_expiration_date': None,
            'license_restrictions': None, 'licensed_capacity': None,
            'ages_served': None, 'star_components': None,
            'star_section_text': None}


def crawl_star(pages):
    out = empty_star()
    soup = _detail_soup(pages)
    if soup is None:
        return out
    types = _m_many(soup, 'lblLicenseType')
    fromd = _m_many(soup, 'lblFromDate')
    ages = _m_many(soup, 'lblAgeRange')
    cap1 = _m_many(soup, 'lblFirstShiftCapacity')
    cap2 = _m_many(soup, 'lblSecondShiftCapacity')
    cap3 = _m_many(soup, 'lblThirdShiftCapacity')
    restr = _m_many(soup, 'lblRestriction')

    cur_type = types[0] if types else None
    out['license_type'] = cur_type
    out['license_issue_date'] = fromd[0] if fromd else None
    out['ages_served'] = ages[0] if ages else None

    def _num(x):
        if x and re.search(r'\d', x):
            try:
                return int(re.sub(r'[^0-9]', '', x))
            except Exception:
                return None
        return None
    caps = [c for c in (_num(cap1[0]) if cap1 else None,
                        _num(cap2[0]) if cap2 else None,
                        _num(cap3[0]) if cap3 else None) if c]
    out['licensed_capacity'] = str(max(caps)) if caps else (cap1[0] if cap1 else None)
    out['license_restrictions'] = '; '.join([r for r in restr if r]) or None

    # star rating: the word in the current license type ("Five Star ... License")
    if cur_type:
        m = re.search(r'\b(one|two|three|four|five)\s+star\b', cur_type, re.I)
        if m:
            out['star_rating'] = _WORD_STARS[m.group(1).lower()]
        else:
            m2 = re.search(r'\b([1-5])\s*star\b', cur_type, re.I)
            if m2:
                out['star_rating'] = int(m2.group(1))
        # unrated (Notice of Compliance / Temporary / GS-110) stays None

    scores = {k: _m_one(soup, s) for k, s in (
        ('total_score', 'lblTotalScore'),
        ('program_points', 'lblProgramStandardsPoints'),
        ('program_max', 'lblProgramStandardsMaxPoints'),
        ('education_points', 'lblEducationalStandardsPoints'),
        ('education_max', 'lblEducationalStandardsMaxPoints'))}
    history = []
    for i in range(max(len(types), len(fromd), len(ages))):
        history.append({'license_type': types[i] if i < len(types) else None,
                        'from_date': fromd[i] if i < len(fromd) else None,
                        'age_range': ages[i] if i < len(ages) else None})
    out['star_components'] = json.dumps(
        {'scores': scores, 'license_history': history}, ensure_ascii=False)
    out['star_section_text'] = _clean(soup.get_text(' '))[:8000] or None
    return out


def empty_visits():
    return {'num_visits': None, 'visits_json': None,
            'violations_json': None, 'visits_section_text': None}


def crawl_visits(pages):
    out = empty_visits()
    soup = _detail_soup(pages)
    if soup is None:
        return out
    dates = _m_many(soup, 'lbVisitDate')
    types = _m_many(soup, 'lbVisityType')   # site's own spelling
    visits = []
    for i in range(max(len(dates), len(types))):
        visits.append({'date': dates[i] if i < len(dates) else None,
                       'announced': types[i] if i < len(types) else None})
    if visits:
        out['num_visits'] = len(visits)
        out['visits_json'] = json.dumps(visits, ensure_ascii=False)

    viols = []
    for el in soup.find_all(id=re.compile(r'violationsList\d+')):
        vid = re.search(r'violationsList(\d+)', el.get('id', ''))
        txt = _clean(el.get_text(' '))
        if txt:
            viols.append({'visit_id': vid.group(1) if vid else None,
                          'text': txt[:2000]})
    if viols:
        out['violations_json'] = json.dumps(viols, ensure_ascii=False)
    out['visits_section_text'] = _clean(soup.get_text(' '))[:8000] or None
    return out


def empty_details():
    return {'operator_name': None, 'address': None, 'phone': None,
            'email': None, 'special_features': None,
            'details_section_text': None}


def crawl_details(pages):
    out = empty_details()
    soup = _detail_soup(pages)
    if soup is None:
        return out
    street = _m_one(soup, 'FacilityStreetLabel')
    city = _m_one(soup, 'FacilityCityLabel')
    state = _m_one(soup, 'FacilityStateLabel')
    zc = _m_one(soup, 'FacilityZipLabel')
    locality = ' '.join([x for x in (city, state, zc) if x])
    out['address'] = ', '.join([p for p in (street, locality) if p]) or None
    out['phone'] = _m_one(soup, 'PhoneLabel')

    # email: prefer the mailto href (get_text drops the @ in this markup)
    mail = soup.find('a', href=re.compile(r'^mailto:', re.I))
    if mail:
        out['email'] = _clean(mail.get('href', '')[7:]) or _clean(mail.get_text())
    else:
        out['email'] = _m_one(soup, 'EmailLabel')

    out['operator_name'] = _m_one(soup, 'lblOwnerName')
    out['special_features'] = _section_text_by_header(soup, 'Facility Special Features')
    out['details_section_text'] = _clean(soup.get_text(' '))[:8000] or None
    return out


# ---------------------------------------------------------------------------
# main crawler (resume + per-section try/except + delay)
# ---------------------------------------------------------------------------

def crawler(records, output_csv='nc_data/nc_records.csv', headless=True,
            start_index=0, limit=None, delay_range=(3, 7)):
    completed = load_completed(output_csv)
    if completed:
        log(f'Resuming: {len(completed)} already in {output_csv} — skipping.')
    end = len(records) if limit is None else min(len(records), start_index + limit)

    fetcher = PortalFetcher(headless=headless)
    index = start_index
    try:
        for index in range(start_index, end):
            rec = records[index]
            fid = rec['facility_id']
            if fid in completed:
                continue

            row = {'facility_id': fid,
                   'facility_name': rec.get('facility_name'),
                   'facility_type': rec.get('facility_type'),
                   'county': rec.get('county')}
            errors = []
            try:
                pages, url = fetcher.get_facility(rec)
                row['facility_url'] = url
                if not pages:
                    row.update(empty_star() | empty_visits() | empty_details())
                    row['errors'] = 'not_found'
                    log(f'[{index}] NOT FOUND: {fid}')
                else:
                    if _modern_id_mismatch(pages, fid):
                        errors.append('id_mismatch')
                    for name, fn, empty_fn in (
                        ('star', crawl_star, empty_star),
                        ('visits', crawl_visits, empty_visits),
                        ('details', crawl_details, empty_details),
                    ):
                        try:
                            row.update(fn(pages))
                        except Exception:
                            errors.append(name)
                            row.update(empty_fn())
                            log(f'  ! {name} failed for {fid}: '
                                f'{traceback.format_exc().splitlines()[-1]}')
                    row['errors'] = ','.join(errors)
                    star = row.get('star_rating')
                    tag = 'OK' if not errors else 'PARTIAL'
                    log(f'[{index}] {tag}: {fid}'
                        + (f' ({star}\u2605)' if star is not None else ''))

                append_row(row, output_csv)
                completed.add(fid)
            except Exception:
                log(f'[{index}] EXCEPTION: {fid}')
                log(traceback.format_exc())
                row.update(empty_star() | empty_visits() | empty_details())
                row['errors'] = 'exception'
                append_row(row, output_csv)
                completed.add(fid)

            if index < end - 1:
                delay = random.uniform(*delay_range)
                log(f'  ...sleeping {delay:.1f}s')
                time.sleep(delay)
    except Exception:
        log(f'CRASHED at index {index}')
        log(traceback.format_exc())
    finally:
        fetcher.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='North Carolina child care crawler (modern portal).')
    ap.add_argument('--seed', default='nc_data/nc_seed.csv',
                    help='Provider seed CSV.')
    ap.add_argument('--output', default='nc_data/nc_records.csv')
    ap.add_argument('--headless', action='store_true',
                    help='Run the browser headless (use after the smoke test).')
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--delay-min', type=float, default=3)
    ap.add_argument('--delay-max', type=float, default=7)
    args = ap.parse_args()

    create_log_file()
    seed = load_seed(args.seed)
    records = seed.to_dict('records')
    log(f'Seed: {len(records)} facilities.')
    crawler(records, output_csv=args.output, headless=args.headless,
            start_index=args.start_index, limit=args.limit,
            delay_range=(args.delay_min, args.delay_max))