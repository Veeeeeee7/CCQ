"""
co_crawler.py — Colorado Shines search-scraper (Round 2).

Reads the provided co_data/co_seed.csv and, for each provider_id,
drives the "Find a Program" search (coloradoshines.com/search) by program
name, opens the matching program_details?id=<Salesforce id> page, and adds
the fields that only live there: description, hours, license type/issue
date, phone/website, languages spoken, special needs, Head Start, real-time
openings by age band, and the 5-category licensing/violation history. The
open-data columns from the seed are carried through unchanged; this script
only ever *adds* columns.

Confirmed architecture (see co_capture.py's docstring and the project chat
for the underlying captures): Salesforce Visualforce/JSF, not Lightning --
search is a real <form method="post" action="/search"> postback, not a
stateless API, so this drives a real browser rather than replaying requests.
Detail-page fields use a consistent <strong>Label:</strong> value pattern;
ratings (both on the results list and the detail page) are CSS-class-encoded
(span.rating-2) when a numeric Shines level exists, or literal text "Licensed
Program" when it doesn't.

Name search is NOT exact-match: searching "Discovery Link" alone returned 56
results across 5 pages (DPS's naming isn't even internally consistent -- some
are "DPS Discovery Link @ X", others just "Discovery Link @ X"). So this
cannot assume the top (or only) result is correct. Disambiguation strategy,
in order:
  1. If there's exactly one result, use it.
  2. If there are several, scan result cards (across up to --max-pages-scan
     pages) for one whose visible text contains the seed row's zip code --
     zip is a clean, format-stable string, and every seed row has one
     (unlike street_address, which is "NA" for home-based providers).
  3. Failing that, try a city-name match instead.
  4. Failing that, do NOT guess: leave the new fields blank and record
     `errors=ambiguous_no_match` with the candidate count, for manual review.
This has only been validated against one single-result search (a Child Care
Center) and one 56-result search (used to discover the need for step 2-4,
not to resolve a specific provider) -- the zip/city matching logic, and
whether it also works for home-based (FCC) providers whose street address is
redacted on-site, needs confirming at smoke-test time.

    python co_crawler.py --limit 5                        # smoke test (visible browser)
    python co_crawler.py --headless --delay-min 2 --delay-max 5   # full run

Resume-safe: re-running skips provider_ids already in the output CSV. Each
extraction group runs under its own try/except so one bad field never kills
a row; failures land in the `errors` column (including `not_found`,
`ambiguous_no_match`, and `id_mismatch` -- the on-site License Number not
matching the provider_id we searched for).

Deps: pip install playwright pandas beautifulsoup4 && playwright install chromium
"""

import argparse
import os
import random
import re
import time
import traceback
from urllib.parse import unquote

import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

BASE_URL = 'https://www.coloradoshines.com'
SEARCH_URL = f'{BASE_URL}/search'

PROGRAM_NAME_INPUT_SEL = 'input[id$=":programnamefield"]'
SEARCH_BUTTON_SEL = 'button.search-submit'
RESULTS_HEADING_SEL = 'h1'

LOG_FILE = 'co_crawler_log.txt'
PROFILE_DIR = 'co_data/co_profile'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

_INIT_SCRIPT = "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"

# Substring keys for mapping the 5 licensing-history accordion titles to
# output columns -- only "Inspection Report (ROI)" has been seen verbatim;
# the rest are matched loosely since exact wording wasn't confirmed live.
LICENSING_SECTION_MAP = [
    ('inspection report', 'inspection_report_text'),
    ('complaint', 'complaints_text'),
    ('stage ii', 'stage_ii_text'),
    ('injur', 'injury_investigations_text'),
    ('adverse action', 'adverse_actions_text'),
]


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
# transport: Playwright driving the Visualforce search-by-name flow
# ---------------------------------------------------------------------------

class ShinesFetcher:
    """Drives the "Find a Program" search and returns the rendered detail-page
    DOM. One persistent browser context is reused across providers (helps
    with any Salesforce-side bot scoring, and is simply faster than relaunching);
    each search starts fresh at SEARCH_URL for a clean ViewState."""

    def __init__(self, headless=True, user_data_dir=PROFILE_DIR):
        from playwright.sync_api import sync_playwright
        os.makedirs(user_data_dir, exist_ok=True)
        self._pw = sync_playwright().start()
        launch_kwargs = dict(
            user_data_dir=user_data_dir, headless=headless,
            viewport={'width': 1400, 'height': 1000}, user_agent=UA,
            args=['--disable-blink-features=AutomationControlled'])
        try:
            self.context = self._pw.chromium.launch_persistent_context(
                channel='chrome', **launch_kwargs)
        except Exception:
            self.context = self._pw.chromium.launch_persistent_context(**launch_kwargs)
        self.context.add_init_script(_INIT_SCRIPT)
        self._pg = None
        self._warmed_up = False

    def _page(self):
        if self._pg is None or self._pg.is_closed():
            self._pg = self.context.new_page()
        return self._pg

    def _settle(self, page, settle_ms=1200, timeout_ms=20000, poll_ms=300):
        deadline = time.time() + timeout_ms / 1000.0
        last_len, stable_since = -1, None
        while time.time() < deadline:
            try:
                cur = page.evaluate("() => (document.body && document.body.innerText || '').length")
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

    def get_provider(self, rec, max_pages_scan=5):
        """Return (soup_or_None, detail_url_or_None, match_method, n_candidates)."""
        page = self._page()
        try:
            if not self._warmed_up:
                page.goto(BASE_URL, wait_until='domcontentloaded')
                time.sleep(2)
                self._warmed_up = True

            page.goto(SEARCH_URL, wait_until='domcontentloaded')
            self._settle(page)

            name = str(rec.get('provider_name') or '').strip()
            box = page.locator(PROGRAM_NAME_INPUT_SEL).first
            if box.count() == 0:
                return None, None, 'no_search_box', 0
            box.fill(name)
            btn = page.locator(SEARCH_BUTTON_SEL).first
            (btn if btn.count() > 0 else box).click() if btn.count() > 0 else box.press('Enter')
            page.wait_for_load_state('domcontentloaded', timeout=20000)
            self._settle(page)

            candidates, n_total = self._collect_candidates(page, max_pages_scan)
            if n_total == 0:
                return None, None, 'not_found', 0
            chosen = self._pick_candidate(candidates, rec)
            if chosen is None:
                return None, None, 'ambiguous_no_match', n_total
            method, pid = chosen
            detail_url = f'{BASE_URL}/program_details?id={pid}'
            page.goto(detail_url, wait_until='domcontentloaded')
            self._settle(page)
            soup = BeautifulSoup(page.content(), 'html.parser')
            return soup, page.url, method, n_total
        except Exception:
            log(f"    ! fetch failed for {rec.get('provider_id')}: "
                f"{traceback.format_exc().splitlines()[-1]}")
            return None, None, 'exception', 0

    def _collect_candidates(self, page, max_pages_scan):
        """Parse every result card on the current page, then page forward
        (via the RichFaces AJAX page-number links) up to max_pages_scan pages
        total, collecting all candidates. Returns (candidates, total_count)."""
        html = page.content()
        n_total = _parse_results_count(html)
        candidates = list(_parse_result_cards(html))
        pages_seen = 1
        while pages_seen < max_pages_scan:
            next_n = pages_seen + 1
            link = page.locator('.pagination a', has_text=re.compile(rf'^{next_n}$'))
            if link.count() == 0:
                break
            try:
                link.first.click()
                self._settle(page)
            except Exception:
                break
            html = page.content()
            candidates.extend(_parse_result_cards(html))
            pages_seen += 1
        return candidates, n_total

    def _pick_candidate(self, candidates, rec):
        if len(candidates) == 1:
            return ('single_result', candidates[0]['id'])
        zip_code = str(rec.get('zip') or '').strip()
        if zip_code and zip_code.upper() != 'NA':
            for c in candidates:
                if zip_code in c['text_blob']:
                    return ('zip_match', c['id'])
        city = str(rec.get('city') or '').strip()
        if city and city.upper() != 'NA':
            for c in candidates:
                if city.lower() in c['text_blob'].lower():
                    return ('city_match', c['id'])
        return None

    def close(self):
        try:
            self.context.close()
        finally:
            self._pw.stop()


def _parse_results_count(html):
    m = re.search(r'Results?\s*\((\d+)\)', html, re.I)
    return int(m.group(1)) if m else 0


def _parse_result_cards(html):
    """Regex-based parse anchored on the one reliable per-card marker: the
    view-details link's aria-label. Everything between one card's link and
    the previous one is that card's content -- avoids guessing how many
    parent levels up a BeautifulSoup tree-walk would need to go, which isn't
    known without live markup to test against.

    The first card has no "previous card" to bound its start, so it's
    anchored to the "Results (N)" heading instead of a fixed lookback --
    tested against a real 56-result page where the first card alone had
    7,300+ characters of markup before its view-details link (far more than
    a fixed window would safely cover)."""
    cards = []
    pattern = re.compile(
        r'aria-label="Read more about (.*?) program"[^>]*href="([^"]*)"'
        r'|href="([^"]*)"[^>]*aria-label="Read more about (.*?) program"')
    matches = list(pattern.finditer(html))
    heading = re.search(r'Results?\s*\(\d+\)', html, re.I)
    list_start = heading.end() if heading else 0
    for i, m in enumerate(matches):
        name = m.group(1) or m.group(4) or ''
        href = m.group(2) or m.group(3) or ''
        idm = re.search(r'id=([A-Za-z0-9]+)', href)
        if not idm:
            continue
        pid = idm.group(1)
        start = matches[i - 1].end() if i > 0 else list_start
        chunk = html[start:m.start()]
        rm = re.search(r'class="rating rating-(\d)"', chunk)
        if rm:
            rating_raw = rm.group(1)
        elif 'Licensed Program' in chunk:
            rating_raw = 'Licensed Program'
        else:
            rating_raw = None
        text_blob = _clean(re.sub(r'<[^>]+>', ' ', unquote(chunk)))
        cards.append({'id': pid, 'name': _clean(name), 'rating_raw': rating_raw,
                      'text_blob': text_blob or ''})
    return cards


# ---------------------------------------------------------------------------
# seed loading
# ---------------------------------------------------------------------------

def load_seed(seed_csv):
    if not seed_csv or not os.path.exists(seed_csv):
        raise FileNotFoundError(
            f'Seed CSV not found: {seed_csv}.')
    df = pd.read_csv(seed_csv, dtype=str, keep_default_na=False, na_values=[''])
    if 'provider_id' not in df.columns or 'provider_name' not in df.columns:
        raise ValueError(
            f'Seed needs provider_id/provider_name columns; got {list(df.columns)}')
    df['provider_id'] = (df['provider_id'].astype(str).str.strip()
                         .str.replace(r'\.0$', '', regex=True))
    df = (df[df['provider_id'].notna() & (df['provider_id'] != '')]
          .drop_duplicates(subset='provider_id').reset_index(drop=True))
    return df


# ---------------------------------------------------------------------------
# resume helpers (rectangular CSV, append-per-row)
# ---------------------------------------------------------------------------

def append_row(row, output_csv, columns):
    full = {}
    for c in columns:
        v = row.get(c)
        if isinstance(v, str):
            v = re.sub(r'[\r\n]+', ' ', v).strip() or None
        full[c] = v
    parent = os.path.dirname(output_csv)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    pd.DataFrame([full], columns=columns).to_csv(
        output_csv, mode='a', header=not exists, index=False)


def load_completed(output_csv):
    if not output_csv or not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv, dtype=str, usecols=['provider_id'])
        return set(df['provider_id'].dropna().str.strip())
    except Exception as e:
        log(f'Could not read existing {output_csv}: {e}')
        return set()


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------

def _clean(s):
    if s is None:
        return None
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s or None


def _after_label(soup, label, check_sibling=False):
    """Value text following a <strong>Label:</strong>. Two markup patterns
    confirmed on real pages: (1) label and value share the same parent block
    (most fields -- License Number, Capacity, Special Needs, ...), or (2) the
    label sits alone in its own <p class="field-label">, with the value in a
    following sibling element (confirmed for Hours of Operation, which is a
    <table> of day/open/close rows under a sibling .field-items div).

    check_sibling must be requested explicitly per label, NOT applied as a
    blanket fallback: a real crawl surfaced a bug where a provider with a
    genuinely EMPTY Special Needs field (no same-parent text left after
    stripping the label -- a normal, valid state, not pattern (2)) had the
    sibling fallback wander into the NEXT field's <p> and return "License
    Type: Permanent" as its "special needs" value. Only pass True for a label
    actually confirmed to use pattern (2).

    Script/style content is stripped first so the one field that renders via
    embedded JS (License Issue Date) doesn't come back full of JavaScript
    instead of a date -- that field has its own regex fallback anyway
    (_extract_license_issue_date)."""
    if soup is None:
        return None
    for strong in soup.find_all('strong'):
        text = _clean(strong.get_text())
        if not (text and label.lower() in text.lower()):
            continue
        parent = strong.parent
        if parent is None:
            return None
        for tag in parent.find_all(['script', 'style']):
            tag.decompose()
        full = _clean(parent.get_text(' '))
        idx = full.lower().find(label.lower()) if full else -1
        if idx != -1:
            rest = full[idx + len(label):].lstrip(': ').strip()
            if rest:
                return rest
        if not check_sibling:
            return None  # genuinely empty field -- do NOT guess from a sibling
        # pattern (2), explicitly requested: check a following sibling block
        sib = parent.find_next_sibling()
        hops = 0
        while sib is not None and hops < 3:
            for tag in sib.find_all(['script', 'style']):
                tag.decompose()
            sib_text = _clean(sib.get_text(' '))
            if sib_text:
                return sib_text
            sib = sib.find_next_sibling()
            hops += 1
        return None
    return None


def _extract_rating(soup):
    """Numeric level as a string ('1'-'5'), literal 'Licensed Program' for an
    unrated/licensing-equivalent program, or None if the label wasn't found
    at all (unexpected page state, worth flagging)."""
    if soup is None:
        return None
    for strong in soup.find_all('strong'):
        if 'program quality rating' in _clean(strong.get_text() or '').lower():
            container = strong.parent or strong
            span = container.find('span', class_=re.compile(r'^rating rating-\d$'))
            if span:
                m = re.search(r'rating-(\d)', ' '.join(span.get('class', [])))
                if m:
                    return m.group(1)
            text = _clean(container.get_text(' ')) or ''
            if 'licensed program' in text.lower():
                return 'Licensed Program'
    return None


def _extract_license_issue_date(soup):
    """The generic _after_label often can't reach this one (see module
    docstring / co_capture.py notes -- it renders via an embedded Apex/JS
    snippet). Best-effort fallback: look for a date pattern within a short
    window after the label text in the raw page text."""
    if soup is None:
        return None
    text = soup.get_text(' ')
    m = re.search(r'License Issue Date:?\s*[^0-9]{0,40}(\d{1,2}/\d{1,2}/\d{4})', text, re.I)
    return m.group(1) if m else None


def _extract_licensing_sections(soup):
    out = {col: None for _, col in LICENSING_SECTION_MAP}
    if soup is None:
        return out
    for a in soup.select('a.topic'):
        title = _clean(a.get_text(' ')) or ''
        title = re.sub(r'^\d+\s*', '', title)
        body = a.find_next(class_='topic-body')
        if body is None:
            continue
        text = _clean(body.get_text(' '))
        if text:
            text = text[:3000]
        for key, col in LICENSING_SECTION_MAP:
            if key in title.lower():
                out[col] = text
                break
    return out


def _id_mismatch(soup, provider_id):
    got = _after_label(soup, 'License Number')
    if not got:
        return False
    got_d = re.sub(r'\D', '', got)
    want_d = re.sub(r'\D', '', str(provider_id))
    return bool(got_d and want_d and got_d != want_d)


# ---------------------------------------------------------------------------
# section extractors (empty_*/crawl_* pairs, rectangular schema)
# ---------------------------------------------------------------------------

def empty_program_info():
    return {'license_number_on_site': None, 'rating_on_site': None,
            'description': None, 'hours_of_operation': None,
            'license_type': None, 'license_issue_date': None,
            'licensed_to_serve': None, 'capacity_on_site': None,
            'phone': None, 'website': None}


def crawl_program_info(soup):
    out = empty_program_info()
    if soup is None:
        return out
    out['license_number_on_site'] = _after_label(soup, 'License Number')
    out['rating_on_site'] = _extract_rating(soup)
    out['hours_of_operation'] = _after_label(soup, 'Hours of Operation', check_sibling=True)
    out['license_type'] = _after_label(soup, 'License Type')
    out['license_issue_date'] = _extract_license_issue_date(soup)
    out['licensed_to_serve'] = _after_label(soup, 'Licensed to Serve')
    out['capacity_on_site'] = _after_label(soup, 'Capacity')
    field_phone = soup.find(class_='field-phone')
    if field_phone:
        tel = field_phone.find('a', href=re.compile(r'^tel:', re.I))
        out['phone'] = _clean(tel.get_text()) if tel else _clean(field_phone.get_text(' '))
    field_site = soup.find(class_='field-website')
    if field_site:
        link = field_site.find('a')
        out['website'] = _clean(link.get('href') or link.get_text()) if link \
            else _clean(field_site.get_text(' '))
    # description: NOT reliably locatable. field-name-field-info turned out to
    # be a generic wrapper class reused by License Number/Type/Capacity/Special
    # Needs/etc, not a unique "bio" container -- an earlier version of this
    # grabbed whichever of those happened to come first (wrong). Leaving this
    # None rather than return misattributed data; a real freetext bio (seen in
    # indexed search snippets for other providers) needs its own confirmed
    # selector before this is filled in.
    out['description'] = None
    return out


def empty_family_facing():
    return {'languages_spoken': None, 'special_needs': None, 'head_start': None,
            'accepts_cccap_on_site': None, 'accepting_new_children': None,
            'openings_infant': None, 'openings_toddler': None,
            'openings_preschool': None, 'openings_school_age': None}


def crawl_family_facing(soup):
    out = empty_family_facing()
    if soup is None:
        return out
    out['languages_spoken'] = _after_label(soup, 'Languages Spoken')
    out['special_needs'] = _after_label(soup, 'Special Needs')
    out['head_start'] = _after_label(soup, 'Head Start')
    out['accepts_cccap_on_site'] = _after_label(soup, 'Accepts CCCAP')
    out['accepting_new_children'] = _after_label(soup, 'Accepting New Children')
    out['openings_infant'] = _after_label(soup, 'Infant Openings Available')
    out['openings_toddler'] = _after_label(soup, 'Toddler Openings Available')
    out['openings_preschool'] = _after_label(soup, 'Preschool Openings Available')
    out['openings_school_age'] = (_after_label(soup, 'School Age Openings Available')
                                  or _after_label(soup, 'School Aged Openings Available'))
    return out


def empty_licensing_history():
    return {col: None for _, col in LICENSING_SECTION_MAP}


def crawl_licensing_history(soup):
    return _extract_licensing_sections(soup)


# ---------------------------------------------------------------------------
# main crawler (resume + per-section try/except + delay)
# ---------------------------------------------------------------------------

def output_columns(seed_columns):
    cols = list(seed_columns)
    for fn in (empty_program_info, empty_family_facing, empty_licensing_history):
        cols.extend(fn().keys())
    cols.extend(['detail_url', 'match_method', 'n_candidates', 'errors'])
    return cols


def crawler(seed_df, output_csv='co_data/co_records.csv', headless=True,
            start_index=0, limit=None, delay_range=(2, 5), max_pages_scan=5):
    columns = output_columns(seed_df.columns)
    completed = load_completed(output_csv)
    if completed:
        log(f'Resuming: {len(completed)} already in {output_csv} -- skipping.')

    records = seed_df.to_dict('records')
    end = len(records) if limit is None else min(len(records), start_index + limit)

    fetcher = ShinesFetcher(headless=headless)
    index = start_index
    try:
        for index in range(start_index, end):
            rec = records[index]
            pid = rec['provider_id']
            if pid in completed:
                continue

            row = dict(rec)
            errors = []
            try:
                soup, url, method, n_candidates = fetcher.get_provider(
                    rec, max_pages_scan=max_pages_scan)
                row['detail_url'] = url
                row['match_method'] = method
                row['n_candidates'] = n_candidates

                if soup is None:
                    row.update(empty_program_info() | empty_family_facing()
                              | empty_licensing_history())
                    row['errors'] = method  # 'not_found' / 'ambiguous_no_match' / etc.
                    log(f'[{index}] {method.upper()}: {pid} ({rec.get("provider_name")})')
                else:
                    if _id_mismatch(soup, pid):
                        errors.append('id_mismatch')
                    for name, fn, empty_fn in (
                        ('program_info', crawl_program_info, empty_program_info),
                        ('family_facing', crawl_family_facing, empty_family_facing),
                        ('licensing_history', crawl_licensing_history, empty_licensing_history),
                    ):
                        try:
                            row.update(fn(soup))
                        except Exception:
                            errors.append(name)
                            row.update(empty_fn())
                            log(f'  ! {name} failed for {pid}: '
                                f'{traceback.format_exc().splitlines()[-1]}')
                    row['errors'] = ','.join(errors)
                    tag = 'OK' if not errors else 'PARTIAL'
                    log(f'[{index}] {tag}: {pid} ({rec.get("provider_name")}) via {method}')

                append_row(row, output_csv, columns)
                completed.add(pid)
            except Exception:
                log(f'[{index}] EXCEPTION: {pid}')
                log(traceback.format_exc())
                row.update(empty_program_info() | empty_family_facing()
                          | empty_licensing_history())
                row['detail_url'] = None
                row['match_method'] = 'exception'
                row['n_candidates'] = 0
                row['errors'] = 'exception'
                append_row(row, output_csv, columns)
                completed.add(pid)

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
    ap = argparse.ArgumentParser(description='Colorado Shines search-scraper (Round 2).')
    ap.add_argument('--seed', default='co_data/co_seed.csv',
                    help='Provider seed CSV.')
    ap.add_argument('--output', default='co_data/co_records.csv')
    ap.add_argument('--headless', action='store_true',
                    help='Run the browser headless (use after the smoke test).')
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--delay-min', type=float, default=2)
    ap.add_argument('--delay-max', type=float, default=5)
    ap.add_argument('--max-pages-scan', type=int, default=5,
                    help='When a name search returns multiple results, scan up '
                         'to this many result pages looking for a zip/city match.')
    args = ap.parse_args()

    create_log_file()
    seed = load_seed(args.seed)
    log(f'Seed: {len(seed)} providers.')
    crawler(seed, output_csv=args.output, headless=args.headless,
            start_index=args.start_index, limit=args.limit,
            delay_range=(args.delay_min, args.delay_max),
            max_pages_scan=args.max_pages_scan)