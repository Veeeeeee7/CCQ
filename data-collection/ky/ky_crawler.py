"""
ky_crawler.py — enrich every provider in ky_seed.csv with kynect's per-
provider detail data (Capacity, Cost, Inspections, Accreditation, Food
Permit, Directed Plan of Correction agreements), by ProviderCLRNumber.

The seed's county searches overlap, so the same provider turns up in several
of them; the detail fields only live behind a second Apex call fired when one
provider's detail view is opened. This script runs that lookup once per unique
provider, via the search page's "License" tab.

Decoded detail payload:

    {"bIsSuccess": true,
     "mapResponse": {
        "showKICCSCapacity": true,
        "KICCSDataDetails": {
           "Capacity": 55,
           "ServiceCostList": [{"AgeGroup": "...", "FullTimeCost": 0.01}, ...],
           "InspectionHistoryListUpdated": [
               {"InspectionId": ..., "InspectionStartDate": "...",
                "InspectionEndDate": "...", "InspectionType": "...",
                "ReportName": "...", "ReportId": ..., "docId": "<huge blob>"},
               ...
           ],
           "IsAcceditationsAvailable": "N",   # [sic] -- native field name, typo and all
           "IsFoodPermitAvailable": "Y",
           "DPOCAgreementsListUpdated": [],
           "OngoingProcessListUpdated": []
        }}}

Per provider: type the ProviderCLRNumber into the License tab (a digits-only
retry is attempted if the lettered form finds nothing), open the detail view,
capture the KICCSDataDetails response, merge onto the seed row. Resume-safe:
providers already in the output without an error are skipped.

Usage:
    python ky_crawler.py --limit 10                # smoke test, headful
    python ky_crawler.py --headless                # full run, resumable
    python ky_crawler.py --headless --delay-min 1 --delay-max 2

Deps: pip install playwright pandas && playwright install chromium
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import time
import traceback

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE_URL = 'https://kynect.ky.gov/benefits/'
SEARCH_URL = 'https://kynect.ky.gov/benefits/s/child-care-provider?origin=program-page&language=en_US'

SEED_PATH = 'ky_data/ky_seed.csv'
OUT_PATH = 'ky_data/ky_records.csv'
LOG_FILE = 'ky_crawler_log.txt'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

KEY_COL = 'ProviderCLRNumber'

DETAIL_COLS = [
    'Capacity', 'showKICCSCapacity', 'IsAcceditationsAvailable',
    'IsFoodPermitAvailable', 'ServiceCostList',
    'InspectionHistoryListUpdated', 'DPOCAgreementsListUpdated',
    'OngoingProcessListUpdated',
]

# Opaque internal references, not facts about the inspection: docId is a
# handle for the inspection PDF; the others are plan-of-correction row ids.
DROP_INSPECTION_KEYS = ('docId', 'pocId', 'POC_ID', 'PlanOfCorrectionID')


def create_log_file(path=LOG_FILE):
    with open(path, 'w') as f:
        f.write('')


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


# Set from --verbose. Per-provider failure reasons always land in the output
# CSV's 'errors' column regardless.
VERBOSE = False


def vlog(message, file=LOG_FILE):
    if VERBOSE:
        log(message, file=file)


def _slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


# ---------------------------------------------------------------------------
# Aura response capture. Responses are read with page.expect_response() tied
# to the click that triggers them: route.fetch() failed with a DNS error in
# the Playwright driver's Node process, and a long-lived passive listener
# raced Chromium's eviction of response bodies.
# ---------------------------------------------------------------------------

def _looks_like_aura(resp):
    try:
        return 'kynect.ky.gov' in resp.url and '/sfsites/aura' in resp.url
    except Exception:
        return False


def _lds_endpoint(request):
    """The x-sfdc-lds-endpoints request header names the Apex method an Aura
    call invokes; readable without touching the response body."""
    try:
        return request.header_value('x-sfdc-lds-endpoints') or ''
    except Exception:
        return ''


def _mk_response_counter(stats):
    recent = stats.setdefault('recent', collections.deque(maxlen=60))

    def on_response(resp):
        stats['total'] = stats.get('total', 0) + 1
        try:
            recent.append(f'{resp.request.method} {resp.status} '
                           f'{resp.request.resource_type} {resp.url}')
        except Exception:
            pass
        if _looks_like_aura(resp):
            try:
                endpoint = _lds_endpoint(resp.request)
                vlog(f'[aura] {resp.url} lds_endpoint={endpoint!r} status={resp.status}')
            except Exception:
                pass
    return on_response


def _extract_inner(body):
    # The Apex payload is a JSON string inside the JSON response.
    for act in body.get('actions', []):
        try:
            if act.get('state') != 'SUCCESS':
                continue
            rv = act.get('returnValue', {}).get('returnValue')
            if isinstance(rv, str):
                rv = json.loads(rv)
            if isinstance(rv, dict):
                yield rv
        except Exception:
            continue


def _candidate_shapes(inner):
    """Yield `inner` itself, plus one level inside `mapResponse` if present."""
    yield inner
    mr = inner.get('mapResponse') if isinstance(inner, dict) else None
    if isinstance(mr, dict):
        yield mr


def _is_search_payload(body):
    return any('sspChildCareProviderDetails' in shape
               for inner in _extract_inner(body)
               for shape in _candidate_shapes(inner))


def _get_search_payload(body):
    for inner in _extract_inner(body):
        for shape in _candidate_shapes(inner):
            if 'sspChildCareProviderDetails' in shape:
                return shape
    return None


def _is_detail_payload(body):
    for inner in _extract_inner(body):
        mr = inner.get('mapResponse')
        if isinstance(mr, dict) and 'KICCSDataDetails' in mr:
            return True
    return False


def _get_detail_payload(body):
    """The whole mapResponse: showKICCSCapacity is a sibling of
    KICCSDataDetails."""
    for inner in _extract_inner(body):
        mr = inner.get('mapResponse')
        if isinstance(mr, dict) and 'KICCSDataDetails' in mr:
            return mr
    return None


SEARCH_LDS_HINT = 'getchildcareproviderdetails'


def _matches_search_call(resp):
    """lds-endpoint header first, body content-sniffing as the fallback."""
    if not _looks_like_aura(resp):
        return False
    try:
        if SEARCH_LDS_HINT in _lds_endpoint(resp.request).casefold():
            return True
    except Exception:
        pass
    try:
        return _is_search_payload(resp.json())
    except Exception:
        return False


def _matches_detail_call(resp):
    # No known lds-endpoint hint for the detail call; content-sniffing only.
    if not _looks_like_aura(resp):
        return False
    try:
        return _is_detail_payload(resp.json())
    except Exception:
        return False


def trigger_and_capture(page, stats, action, predicate, timeout_ms=30000, label=''):
    """Run `action` inside a page.expect_response() wait and return the
    matching response's JSON body, or None on timeout/failure."""
    try:
        with page.expect_response(predicate, timeout=timeout_ms) as resp_info:
            action()
    except PWTimeout:
        vlog(f'[{label}] timed out after {timeout_ms}ms waiting for a matching response')
        debug_dump_stats(stats, _slug(label))
        return None
    except Exception as e:
        vlog(f'[{label}] action failed: {type(e).__name__}: {e}')
        return None
    resp = resp_info.value
    try:
        return resp.json()
    except Exception as e:
        vlog(f'[{label}] matching response arrived but its body could not be read: '
             f'{type(e).__name__}: {e}')
        return None


# ---------------------------------------------------------------------------
# page interactions
# ---------------------------------------------------------------------------

def dismiss_cookie_banner(page):
    # count() first: the banner appears once per page load, and a bare
    # click(timeout=3000) would wait 3 s per provider for an absent button.
    try:
        loc = page.get_by_role('button', name=re.compile(r'^Accept$', re.I))
        if loc.count() > 0:
            loc.first.click(timeout=3000)
            return True
    except Exception:
        pass
    return False


def modify_search(page):
    # Only exists on the results view; count() avoids a 4 s wait when absent.
    try:
        loc = page.get_by_role('button', name=re.compile('Modify Search', re.I))
        if loc.count() == 0:
            return False
        loc.first.click(timeout=4000)
        page.wait_for_timeout(500)
        return True
    except Exception:
        return False


def click_back_to_search(page):
    """Leave a provider's detail view, which has no "Modify Search" button.
    The control is a plain <a href="javascript:void(0);"> titled "Go back to
    Search for Providers". No-op when not on a detail view."""
    for attempt in (
        lambda: page.get_by_role('link', name=re.compile('Back to Search for Providers', re.I)),
        lambda: page.get_by_title(re.compile('Go back to Search for Providers', re.I)),
        lambda: page.get_by_text(re.compile('Back to Search for Providers', re.I)),
    ):
        try:
            loc = attempt()
            if loc.count() > 0:
                loc.first.click(timeout=5000)
                page.wait_for_timeout(500)
                return True
        except Exception:
            continue
    return False


def click_license_tab(page):
    for attempt in (
        lambda: page.get_by_role('button', name=re.compile(r'^License$', re.I)),
        lambda: page.get_by_role('tab', name=re.compile(r'^License$', re.I)),
        lambda: page.get_by_text('License', exact=True),
    ):
        try:
            loc = attempt()
            if loc.count() > 0:
                loc.first.click(timeout=3000)
                return True
        except Exception:
            continue
    return False


def _find_license_box(page):
    """role=textbox must come before get_by_label: the page's <label for=...>
    points at the <lightning-input> wrapper, not the real <input>, so a label
    match lands on an element that cannot be filled."""
    for attempt in (
        lambda: page.get_by_role('textbox', name=re.compile('license', re.I)),
        lambda: page.get_by_placeholder(re.compile('license', re.I)),
        lambda: page.get_by_label(re.compile('license', re.I)),
        lambda: page.get_by_role('textbox'),
        lambda: page.get_by_role('combobox'),
    ):
        try:
            loc = attempt()
            if loc.count() > 0:
                return loc.first
        except Exception:
            continue
    return None


def fill_license_box(page, text):
    box = _find_license_box(page)
    if box is None:
        return False
    try:
        box.click(timeout=5000)
        box.fill('')
    except Exception:
        pass
    page.keyboard.type(text, delay=35)
    return True


def click_search(page):
    page.get_by_role('button', name=re.compile(r'^Search$', re.I)).first.click(timeout=10000)


DEBUG_DIR = os.path.join('ky_data', '_debug')


def debug_screenshot(page, tag):
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        path = os.path.join(DEBUG_DIR, f'{tag}.png')
        page.screenshot(path=path)
        vlog(f'[debug] screenshot -> {path}')
    except Exception as e:
        log(f'[debug] screenshot failed for {tag}: {e}')


def debug_dump_stats(stats, tag):
    summary = {k: v for k, v in stats.items() if k != 'recent'}
    vlog(f'[debug:{tag}] listener stats so far this run: {summary}')
    recent = list(stats.get('recent') or [])
    vlog(f'[debug:{tag}] last {len(recent)} response(s) seen, ANY origin '
         f'(method status resource_type url):')
    for line in recent:
        vlog(f'[debug:{tag}]   {line}')


def search_by_license(page, stats, clr, timeout_ms=30000):
    """Try the full CLR string (e.g. "L350533"), then digits only. Returns the
    matched sspChildCareProviderDetails list (possibly empty), or None if the
    search box couldn't be found."""
    for variant in _clr_variants(clr):
        if not click_license_tab(page):
            pass  # may already be selected -- not fatal
        if not fill_license_box(page, variant):
            return None

        vlog(f'[{clr}] waiting up to {timeout_ms / 1000:.0f}s for the License search-results payload...')
        body = trigger_and_capture(page, stats, lambda: click_search(page),
                                    _matches_search_call, timeout_ms=timeout_ms,
                                    label=f'{clr} search click')
        if body is None:
            continue
        payload = _get_search_payload(body) or {}
        matches = payload.get('sspChildCareProviderDetails') or []
        if matches:
            if variant != clr:
                vlog(f'[INFO] {clr}: matched using digits-only variant {variant!r}')
            return matches
        modify_search(page)
    return []


def _reset_to_search(page):
    """A selector miss means the page has no text input at all (error shell,
    expired Aura session, blank render); modify_search() and
    click_back_to_search() cannot recover from that, only a reload can."""
    page.goto(SEARCH_URL, wait_until='domcontentloaded')
    page.wait_for_timeout(1500)
    dismiss_cookie_banner(page)


def _clr_variants(clr):
    variants = [clr]
    digits_only = re.sub(r'[^0-9]', '', clr)
    if digits_only and digits_only != clr:
        variants.append(digits_only)
    return variants


def open_detail_for_result(page, stats, timeout_ms=30000):
    """An exact License match may land straight on the detail view, or may
    need a "View More Details" click; the action clicks the button if present
    and is a no-op otherwise, and the wait catches the payload either way."""
    def action():
        for attempt in (
            lambda: page.get_by_role('button', name=re.compile('View More Details', re.I)),
            lambda: page.get_by_role('link', name=re.compile('View More Details', re.I)),
        ):
            try:
                loc = attempt()
                if loc.count() > 0:
                    loc.first.click(timeout=5000)
                    return
            except Exception:
                continue

    vlog(f'waiting up to {timeout_ms / 1000:.0f}s for the detail (KICCSDataDetails) payload...')
    body = trigger_and_capture(page, stats, action, _matches_detail_call,
                                timeout_ms=timeout_ms, label='detail fetch')
    return _get_detail_payload(body) if body else None


# ---------------------------------------------------------------------------
# per-provider
# ---------------------------------------------------------------------------

def flatten_detail(map_response):
    """mapResponse -> the flat DETAIL_COLS cells.

    When kynect does not publish a provider's capacity it omits
    `showKICCSCapacity` (the page shows an openings-by-age table instead), but
    KICCSDataDetails still carries `Capacity`. The flag is coerced to a real
    boolean so True/False means "fetched" and blank means "not fetched".
    """
    detail = (map_response or {}).get('KICCSDataDetails') or {}
    out = {
        'Capacity': detail.get('Capacity'),
        'showKICCSCapacity': bool((map_response or {}).get('showKICCSCapacity')),
        'IsAcceditationsAvailable': detail.get('IsAcceditationsAvailable'),
        'IsFoodPermitAvailable': detail.get('IsFoodPermitAvailable'),
    }

    cost = detail.get('ServiceCostList') or []
    out['ServiceCostList'] = json.dumps(cost, ensure_ascii=False) if cost else None

    inspections = detail.get('InspectionHistoryListUpdated') or []
    cleaned = [{k: v for k, v in item.items()
                if k not in DROP_INSPECTION_KEYS}
               for item in inspections if isinstance(item, dict)]
    out['InspectionHistoryListUpdated'] = json.dumps(cleaned, ensure_ascii=False) if cleaned else None

    dpoc = detail.get('DPOCAgreementsListUpdated') or []
    out['DPOCAgreementsListUpdated'] = json.dumps(dpoc, ensure_ascii=False) if dpoc else None

    ongoing = detail.get('OngoingProcessListUpdated') or []
    out['OngoingProcessListUpdated'] = json.dumps(ongoing, ensure_ascii=False) if ongoing else None

    return out


def process_provider(page, stats, seed_row):
    """Every seed_row field, plus DETAIL_COLS, plus 'errors' (None on
    success, else a short reason). Never raises."""
    row = dict(seed_row)
    row['errors'] = None
    clr = str(seed_row.get(KEY_COL, '')).strip()

    try:
        if not clr:
            raise RuntimeError('empty ProviderCLRNumber in seed row')

        click_back_to_search(page)  # best-effort: leave a lingering detail view, if any
        modify_search(page)  # best-effort return to the form

        matches = None
        for attempt in (1, 2):
            matches = search_by_license(page, stats, clr)
            if matches is not None:
                break
            vlog(f'[{clr}] selector miss (attempt {attempt}) -- reloading '
                 f'{SEARCH_URL}')
            _reset_to_search(page)
        if matches is None:
            raise RuntimeError('could not submit a License search (selector miss)')
        if not matches:
            raise RuntimeError('License search returned zero matches')
        if len(matches) > 1:
            vlog(f'[WARN] {clr}: License search returned {len(matches)} matches, expected 1')

        detail = open_detail_for_result(page, stats)
        if detail is None:
            raise RuntimeError('no detail (KICCSDataDetails) payload arrived')

        row.update(flatten_detail(detail))
    except Exception as e:
        row['errors'] = str(e)
        vlog(f'[FAIL] {clr}: {e}')
        debug_screenshot(page, f'{clr}_process_failed')

    return row


# ---------------------------------------------------------------------------
# resume / output
# ---------------------------------------------------------------------------

def load_done(out_path, key=KEY_COL, error_col='errors'):
    """Done means SUCCEEDED: a row with a non-empty `errors` cell is still
    owed, so a re-run retries it."""
    if not os.path.exists(out_path):
        return set()
    df = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    if key not in df.columns:
        return set()
    if error_col in df.columns:
        df = df[df[error_col].astype(str).str.strip() == '']
    return set(df[key])


def append_row(out_path, row, fieldnames):
    exists = os.path.exists(out_path)
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', default=SEED_PATH)
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--limit', type=int, default=None,
                    help='Only process the first N not-yet-done seed rows (smoke test).')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--delay-min', type=float, default=1.0)
    ap.add_argument('--delay-max', type=float, default=2.0)
    ap.add_argument('--breaker', type=int, default=5,
                    help='consecutive failures that trigger a cooldown and a '
                         'fresh browser context (0 disables)')
    ap.add_argument('--cooldown', type=float, default=600.0,
                    help='seconds to wait when the breaker trips')
    ap.add_argument('--channel', default=None)
    ap.add_argument('--verbose', action='store_true',
                    help='Full per-request diagnostic logging. Failure reasons '
                         'always land in the output CSV\'s errors column.')
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    create_log_file()

    seed = pd.read_csv(args.seed, dtype=str, keep_default_na=False)
    if KEY_COL not in seed.columns:
        raise SystemExit(f'{args.seed} has no {KEY_COL} column')
    fieldnames = list(seed.columns) + DETAIL_COLS + ['errors']

    done = load_done(args.out)
    todo = seed[~seed[KEY_COL].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    log(f'{len(seed)} in seed, {len(done)} already done, {len(todo)} to process this run')

    # Errored rows are retried, but append_row() appends, so a retry would
    # duplicate them in the output.
    if os.path.exists(args.out):
        existing = set(pd.read_csv(args.out, dtype=str, keep_default_na=False,
                                   usecols=[KEY_COL])[KEY_COL])
        retries = set(todo[KEY_COL]) & existing
        if retries:
            log(f'[warn] {len(retries)} of the {len(todo)} row(s) to process '
                f'are ALREADY in {args.out} with a non-empty errors cell. This '
                f'run APPENDS, so the file would end up with duplicate rows. '
                f'Use ky_refetch.py (writes a sidecar) and then '
                f'`ky_data_correction.py --fix-detail` (patches in place), or '
                f'point --out at a fresh file.')

    if len(todo) == 0:
        log('Nothing to do.')
        return

    with sync_playwright() as p:
        launch_kwargs = {'headless': args.headless,
                          'args': ['--disable-blink-features=AutomationControlled']}
        if args.channel:
            launch_kwargs['channel'] = args.channel
        # Fresh, non-persistent context every run: a reused profile let
        # Salesforce Aura's client-side action cache (IndexedDB) serve stale
        # results with no network traffic. No CAPTCHA here, so nothing is lost.
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, user_agent=UA)
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        stats = {}
        context.on('response', _mk_response_counter(stats))
        page = context.new_page()

        page.goto(BASE_URL, wait_until='domcontentloaded')
        page.wait_for_timeout(1500)
        page.goto(SEARCH_URL, wait_until='domcontentloaded')
        page.wait_for_timeout(1500)
        dismiss_cookie_banner(page)

        ok, fail, consecutive = 0, 0, 0
        for i, (_, seed_row) in enumerate(todo.iterrows()):
            clr = seed_row.get(KEY_COL, '?')
            try:
                row = process_provider(page, stats, seed_row.to_dict())
            except Exception:
                row = seed_row.to_dict()
                row['errors'] = 'unhandled exception'
                log(f'[FAIL] {clr}: unhandled exception\n{traceback.format_exc()}')
            append_row(args.out, row, fieldnames)
            success = not row.get('errors')
            ok += int(success)
            fail += int(not success)
            log(f'[{i + 1}/{len(todo)}] {clr}: {"ok" if success else "FAILED"}')

            # Circuit breaker: a mid-run site outage would otherwise write an
            # unbroken block of error rows.
            consecutive = 0 if success else consecutive + 1
            if args.breaker and consecutive >= args.breaker:
                log(f'[breaker] {consecutive} consecutive failures -- cooling '
                    f'down {args.cooldown:.0f}s and relaunching the context')
                time.sleep(args.cooldown)
                try:
                    context.close()
                except Exception:
                    pass
                context = browser.new_context(
                    viewport={'width': 1440, 'height': 1000}, user_agent=UA)
                context.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', "
                    "{get: () => undefined})")
                context.on('response', _mk_response_counter(stats))
                page = context.new_page()
                page.goto(BASE_URL, wait_until='domcontentloaded')
                page.wait_for_timeout(1500)
                _reset_to_search(page)
                consecutive = 0
                continue

            time.sleep(random.uniform(args.delay_min, args.delay_max))

        context.close()
        browser.close()

    log(f"Listener stats for the whole run: {({k: v for k, v in stats.items() if k != 'recent'})}")
    log(f'\nDone this run: {ok} ok, {fail} errored, out of {len(todo)} attempted -> {args.out}')


if __name__ == '__main__':
    main()
