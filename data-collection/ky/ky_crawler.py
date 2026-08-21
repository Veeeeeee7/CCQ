"""
ky_crawler.py — enrich every provider in ky_seed.csv with kynect's per-
provider detail data (Capacity, Cost, Inspections, Accreditation, Food
Permit, Directed Plan of Correction agreements), by ProviderCLRNumber.

WHY THIS IS SEPARATE FROM THE SEED: the seed's county-by-county Location
searches already returned provider_id (ProviderCLRNumber) and qr_rating
(NumberOfStars) plus a good amount of bonus directory data, for free, as part
of enumerating the state. But that enumeration deliberately searches
overlapping/redundant areas (Kentucky's counties are small enough that one
search returns neighboring counties too), so the SAME provider can turn up in
several county searches. The extra detail fields below only live behind a
SECOND, more expensive Apex call fired when you open one specific provider's
detail view -- so this script runs that lookup exactly once per UNIQUE
provider (post-dedup), by License number, instead of once per redundant
county hit.

CONFIRMED from the Phase 1 capture (ky_captures/resp_030_...r_14...txt),
clicking into "Northern Kentucky Head Start - Elsmere Center" (provider_id
"L350533" from the search results) fired an Apex call whose decoded payload
was:

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

`docId` looks like an opaque reference used to fetch the underlying
inspection PDF/report -- deliberately DROPPED here (see flatten_detail()):
we keep every other inspection field (dates, type, report name/id), which is
genuinely useful without ever touching the document itself, per instruction
not to download any PDFs.

FLOW PER PROVIDER (mirrors mi_gsq_crawler.py's search-by-license-number
pattern, the closest reference-state analog): use the search page's
"License" tab, type the provider's own ProviderCLRNumber (expect exactly one
match), open its detail view, capture the KICCSDataDetails Apex response,
merge onto the seed row, move on. A resume-safe skip-list (by
ProviderCLRNumber already present in the output CSV) means a killed run can
just be restarted.

UNVERIFIED / best-guess -- confirm in the smoke
test: the License tab's accessible name and its input's label/placeholder
(guessed as containing "license"), whether typing the FULL CLR string
(e.g. "L350533", letter prefix included) is what the box expects vs. digits
only (a digit-only retry is attempted automatically if the lettered form
finds nothing), and whether a "View More Details"-style button is still
needed after a License search or whether an exact single match jumps
straight to the detail view (both paths are handled).

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

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

BASE_URL = 'https://kynect.ky.gov/benefits/'
SEARCH_URL = 'https://kynect.ky.gov/benefits/s/child-care-provider?origin=program-page&language=en_US'

SEED_PATH = 'ky_data/ky_seed.csv'
OUT_PATH = 'ky_data/ky_records.csv'
LOG_FILE = 'ky_crawler_log.txt'
# NOTE (2026-07-07): dropped the persistent-profile browser context that used
# to live here. The full story:
# (Salesforce Aura's client-side action cache was found to be silently
# serving stale results from a reused profile's IndexedDB store during
# repeated smoke tests, with zero matching network traffic to show for it).
# KY has no confirmed CAPTCHA, so there's no real trust-accumulation benefit
# being given up by using a fresh context every run instead.

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

KEY_COL = 'ProviderCLRNumber'

# Columns this script adds on top of the seed's own. Fixed and
# known up front (unlike GA's per-provider schema growth), so a plain
# csv.DictWriter with this header works.
DETAIL_COLS = [
    'Capacity', 'IsAcceditationsAvailable', 'IsFoodPermitAvailable',
    'ServiceCostList', 'InspectionHistoryListUpdated',
    'DPOCAgreementsListUpdated', 'OngoingProcessListUpdated',
]


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def create_log_file(path=LOG_FILE):
    with open(path, 'w') as f:
        f.write('')


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


# Silent by default -- set from --verbose in main(). Essential output here
# is deliberately narrow: the per-provider "[i/N] {clr}: ok/FAILED" line,
# run-start/run-end bookkeeping, and genuinely unexpected (unhandled
# exception) failures. Everything else -- raw [aura] traffic, per-call
# timeout/diagnostic detail, routine screenshot confirmations, the
# digits-only-variant note, and the multi-match warning -- goes through
# vlog(), a no-op unless --verbose is passed. None of that is lost data:
# the per-provider failure reason always lands in the output CSV's
# 'errors' column regardless of verbosity.
VERBOSE = False


def vlog(message, file=LOG_FILE):
    if VERBOSE:
        log(message, file=file)


def _slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


# ---------------------------------------------------------------------------
# Aura response capture. Three mechanisms were tried (passive listener ->
# broadened passive listener -> route.fetch()+route.fulfill()), each with
# its own failure mode. Short version: route.fetch()
# (this file's previous mechanism) failed UNCONDITIONALLY on at least one
# real machine with "Error: Route.fetch: getaddrinfo ENOTFOUND
# kynect.ky.gov" -- a DNS resolution failure specific to the Playwright
# driver's own Node process, even though the browser itself loaded
# kynect.ky.gov fine throughout (every failed fetch fell back to
# route.continue_(), so the real response reached the page normally while
# our capture list stayed empty forever). trigger_and_capture() below uses
# page.expect_response() tied directly to the specific action that triggers
# the response we want -- no separate request is ever made, so no DNS
# mismatch is possible.
# ---------------------------------------------------------------------------

def _looks_like_aura(resp):
    try:
        return 'kynect.ky.gov' in resp.url and '/sfsites/aura' in resp.url
    except Exception:
        return False


def _lds_endpoint(request):
    """The x-sfdc-lds-endpoints request header names the specific Apex
    controller/method an Aura call invokes -- confirmed via a live capture.
    Needs no network call and no response body access at all."""
    try:
        return request.header_value('x-sfdc-lds-endpoints') or ''
    except Exception:
        return ''


def _mk_response_counter(stats):
    """Coarse traffic visibility for every response on the page (feeds the
    debug dump if a wait times out), plus a compact [aura] log line naming
    which Apex method each kynect Aura call hit -- entirely from the
    x-sfdc-lds-endpoints request header, no body read required (see the
    capture-mechanism history above for why that matters here)."""
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
    """Yield `inner` itself, plus one level inside `mapResponse` if present.
    A live run showed at least one kynect Apex action using that wrapper
    envelope; cheap to also check it here for the License-tab search, which
    uses the same sspChildCareProviderDetails predicate."""
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
    for inner in _extract_inner(body):
        mr = inner.get('mapResponse')
        if isinstance(mr, dict) and 'KICCSDataDetails' in mr:
            return mr['KICCSDataDetails']
    return None


SEARCH_LDS_HINT = 'getchildcareproviderdetails'


def _matches_search_call(resp):
    """Primary signal: x-sfdc-lds-endpoints request header -- confirmed for
    the Location-tab search; UNCONFIRMED whether this file's
    License-tab search carries the same hint, so the content-sniffing
    fallback matters more here. Reading the body in the fallback is safe --
    see trigger_and_capture's docstring."""
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
    """No confirmed lds-endpoint hint for the detail (KICCSDataDetails) call
    -- content-sniffing is the only signal available. Safe for the same
    reason as _matches_search_call: expect_response() only evaluates this
    against the handful of aura-domain responses arriving during one short,
    tightly-scoped wait, not a page-wide, long-lived listener."""
    if not _looks_like_aura(resp):
        return False
    try:
        return _is_detail_payload(resp.json())
    except Exception:
        return False


def trigger_and_capture(page, stats, action, predicate, timeout_ms=30000, label=''):
    """Run `action` (a zero-arg callable) INSIDE a page.expect_response()
    wait, and return the matching response's decoded JSON body, or None on
    timeout/failure. Short version: this reads the body of the response the
    browser itself already received, so no separate request and no DNS
    mismatch is possible (unlike the retired route.fetch()-based capture)."""
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
# page interactions -- best-guess selectors, confirm in the smoke test
# ---------------------------------------------------------------------------

def dismiss_cookie_banner(page):
    try:
        page.get_by_role('button', name=re.compile(r'^Accept$', re.I)).click(timeout=3000)
    except Exception:
        pass


def modify_search(page):
    try:
        page.get_by_role('button', name=re.compile('Modify Search', re.I)).click(timeout=4000)
        page.wait_for_timeout(500)
        return True
    except Exception:
        return False


def click_back_to_search(page):
    """After open_detail_for_result() succeeds, the page is left sitting on
    that provider's DETAIL view (sspChildCareProviderDetails component) --
    which has NO "Modify Search" button at all (that only lives on the
    search form/results), so modify_search() alone silently no-ops there
    and the crawler would stay stuck on the previous provider's detail page
    for the rest of the run. The real control to leave a detail view is
    this anchor, confirmed via a live capture:

        <a href="javascript:void(0);" class="ssp-anchor ssp-color_monoBody"
           title="Go back to Search for Providers" data-id="925">
           &lt;Back to Search for Providers</a>

    Note it's a plain <a href="javascript:void(0);">, not a <button> --
    real (non-empty) href gives it an implicit role of "link". Called
    best-effort, before modify_search(), at the start of every provider's
    turn: harmless no-op if we're not currently on a detail view (e.g. the
    very first provider, or after a failed attempt that never reached one)."""
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
    """FIXED (2026-07-08, from a live capture of the License tab --
    ky_captures/capture_000.html): the real control is

        <input aria-label="Search by license or certificate number"
               id="input-16" type="text" ...>

    nested several levels inside a <lightning-input id="sspInputText-14">
    wrapper. But the page's own visible <label for="sspInputText-14"> points
    at that WRAPPER, not the real <input> -- a broken association, since
    <lightning-input> is a plain custom element with no native "labelable"
    semantics at all. The previous version of this function tried
    get_by_label('license') BEFORE get_by_role('textbox'), so it matched
    that wrapper first and stopped there: box.fill() on a non-form-control
    element silently threw (swallowed by the bare except below), and the
    subsequent blind page.keyboard.type() had nothing real to type into --
    exactly the "isn't inputting anything after clicking License" symptom.

    FIX: try get_by_role('textbox', name=...) FIRST. Confirmed via the same
    capture that this page has exactly ONE real <input type="text"> and NO
    other role=textbox/combobox/searchbox candidates at all, and a bare
    custom element like <lightning-input> has no implicit ARIA role -- so
    role=textbox can only ever resolve to the genuine native input here,
    completely sidestepping the broken label-for association rather than
    continuing to race it."""
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


# ---------------------------------------------------------------------------
# debugging aids -- cheap, and they turn a failed
# run into something we can just look at instead of another guessing round.
# ---------------------------------------------------------------------------

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
    """Try the full CLR string first (e.g. "L350533"); if that yields zero
    matches, retry with digits only, in case the box expects the number
    without its Licensed/Certified prefix letter. Returns the matched
    sspChildCareProviderDetails list (possibly empty) or None if the box
    itself couldn't be found at all.

    NOTE: a click that fails outright (e.g. the Search button selector
    misses) and a click that succeeds but produces no matching response
    within timeout_ms both now fall through to "try the next variant" via
    trigger_and_capture()'s uniform None-on-any-failure return, rather than
    the previous version's immediate hard-fail on a click exception --
    a small, deliberate simplification (see trigger_and_capture): both
    cases end up logged, and the row is still marked failed by the caller
    either way, so the only real difference is a slightly less specific
    error string in the rare case the Search button truly can't be found."""
    for variant in _clr_variants(clr):
        if not click_license_tab(page):
            pass  # may already be selected -- not fatal
        if not fill_license_box(page, variant):
            return None
        dismiss_cookie_banner(page)

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


def _clr_variants(clr):
    variants = [clr]
    digits_only = re.sub(r'[^0-9]', '', clr)
    if digits_only and digits_only != clr:
        variants.append(digits_only)
    return variants


def open_detail_for_result(page, stats, timeout_ms=30000):
    """The single search result may already BE the detail view (an exact
    License match could jump straight there), or may still need a click on
    a "View More Details"-style button. Handle both by putting the
    button-finding logic INSIDE the trigger_and_capture() action: if such a
    button is findable, click it; otherwise the action is a no-op, and the
    expect_response() wait below still catches the KICCSDataDetails payload
    if it's already in flight from the prior search action or arrives
    shortly after (see module docstring for why this dual path is still an
    UNVERIFIED assumption worth confirming in the smoke test)."""
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
        # No button/link found -- nothing to click; see docstring above.

    vlog(f'waiting up to {timeout_ms / 1000:.0f}s for the detail (KICCSDataDetails) payload...')
    body = trigger_and_capture(page, stats, action, _matches_detail_call,
                                timeout_ms=timeout_ms, label='detail fetch')
    return _get_detail_payload(body) if body else None


# ---------------------------------------------------------------------------
# per-provider
# ---------------------------------------------------------------------------

def flatten_detail(detail):
    out = {
        'Capacity': detail.get('Capacity'),
        'IsAcceditationsAvailable': detail.get('IsAcceditationsAvailable'),
        'IsFoodPermitAvailable': detail.get('IsFoodPermitAvailable'),
    }

    cost = detail.get('ServiceCostList') or []
    out['ServiceCostList'] = json.dumps(cost, ensure_ascii=False) if cost else None

    # docId is a reference used to pull the underlying inspection PDF --
    # deliberately dropped. Every OTHER inspection field is genuinely
    # informative on its own and gets kept, per instruction not to download
    # any PDFs (keeping docId around would be an attractive nuisance for a
    # future "just fetch the doc" temptation, so it's stripped here rather
    # than left in and merely unused).
    inspections = detail.get('InspectionHistoryListUpdated') or []
    cleaned = [{k: v for k, v in item.items() if k != 'docId'}
               for item in inspections if isinstance(item, dict)]
    out['InspectionHistoryListUpdated'] = json.dumps(cleaned, ensure_ascii=False) if cleaned else None

    dpoc = detail.get('DPOCAgreementsListUpdated') or []
    out['DPOCAgreementsListUpdated'] = json.dumps(dpoc, ensure_ascii=False) if dpoc else None

    ongoing = detail.get('OngoingProcessListUpdated') or []
    out['OngoingProcessListUpdated'] = json.dumps(ongoing, ensure_ascii=False) if ongoing else None

    return out


def process_provider(page, stats, seed_row):
    """Returns a dict: every seed_row field, plus DETAIL_COLS, plus
    'errors' (None on success, else a short reason). Never raises -- a
    single bad provider must not kill the whole run."""
    row = dict(seed_row)
    row['errors'] = None
    clr = str(seed_row.get(KEY_COL, '')).strip()

    try:
        if not clr:
            raise RuntimeError('empty ProviderCLRNumber in seed row')

        click_back_to_search(page)  # best-effort: leave a lingering detail view, if any
        modify_search(page)  # best-effort return to the form
        matches = search_by_license(page, stats, clr)
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
        # Reason is intentionally vlog-only here -- it always lands in the
        # output CSV's 'errors' column regardless, and the per-provider
        # "[i/N] {clr}: FAILED" line at the call site already reports the
        # status. --verbose surfaces this line too when actively debugging.
        vlog(f'[FAIL] {clr}: {e}')
        debug_screenshot(page, f'{clr}_process_failed')

    return row


# ---------------------------------------------------------------------------
# resume / output
# ---------------------------------------------------------------------------

def load_done(out_path, key=KEY_COL):
    if not os.path.exists(out_path):
        return set()
    df = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    return set(df[key]) if key in df.columns else set()


def append_row(out_path, row, fieldnames):
    exists = os.path.exists(out_path)
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in fieldnames})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', default=SEED_PATH)
    ap.add_argument('--out', default=OUT_PATH)
    ap.add_argument('--limit', type=int, default=None,
                    help='Only process the first N not-yet-done seed rows (smoke test).')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--delay-min', type=float, default=1.5)
    ap.add_argument('--delay-max', type=float, default=3.5)
    ap.add_argument('--channel', default=None)
    ap.add_argument('--verbose', action='store_true',
                    help='Restore full per-request diagnostic logging (raw [aura] traffic, '
                         'timeout/action-failure detail, screenshot confirmations, the '
                         'digits-only-variant note, the multi-match warning, and per-provider '
                         'failure reasons). Off by default -- routine runs only log the '
                         'per-provider "[i/N] {clr}: ok/FAILED" line plus run bookkeeping; '
                         'failure reasons always still land in the output CSV\'s errors column.')
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

    if len(todo) == 0:
        log('Nothing to do.')
        return

    with sync_playwright() as p:
        launch_kwargs = {'headless': args.headless,
                          'args': ['--disable-blink-features=AutomationControlled']}
        if args.channel:
            launch_kwargs['channel'] = args.channel
        # Fresh, non-persistent browser + context every run -- see the
        # config block's NOTE for why (a reused persistent profile let
        # Salesforce Aura's client-side action cache silently serve stale
        # results with zero matching network traffic).
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

        ok, fail = 0, 0
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
            time.sleep(random.uniform(args.delay_min, args.delay_max))

        context.close()
        browser.close()

    log(f"Listener stats for the whole run: {({k: v for k, v in stats.items() if k != 'recent'})}")
    log(f'\nDone this run: {ok} ok, {fail} errored, out of {len(todo)} attempted -> {args.out}')


if __name__ == '__main__':
    main()
