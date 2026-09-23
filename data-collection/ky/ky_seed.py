"""
ky_seed.py — build the Kentucky provider seed from kynect's Public Child Care
Search, county by county.

kynect's search (https://kynect.ky.gov/benefits/s/child-care-provider) is
Salesforce Experience Cloud (Aura). Every meaningful action is a POST to
/benefits/s/sfsites/aura?r=N&aura.ApexAction.execute=1, whose response body is

    {"actions": [{"id": ..., "state": "SUCCESS",
                  "returnValue": {"returnValue": "<JSON-encoded STRING>",
                                  "cacheable": ...}}]}

i.e. the real payload is a JSON string inside the JSON and has to be decoded
twice. The search's inner payload is

    {"sspChildCareProviderDetails": [ {...one dict per provider...} ],
     "requestId": "...", "IsBrightwheelServiceDisabled": false}

The Location tab is a proximity search, not a per-county filter (a Boone
County search returns Boone, Kenton and Campbell providers), so the script
searches once per Kentucky county and dedupes on ProviderCLRNumber. The first
search response carries the full result set; "View More" is still clicked and
merged until it disappears or stops adding rows.

The form only accepts a query once a Google Places autocomplete suggestion
has been clicked. Typing "<County> COUNTY, KY" (a bare "<County>, KY" can
surface a same-named POI first) and clicking the first suggestion reliably
selects the county.

Per-row fields: ProviderCLRNumber (the licensing/certificate number, "L..."
Licensed, "C..." Certified), ProviderId (small internal int), ProviderName,
ProviderType, ProviderStatus, NumberOfStars (0-5; ALL STARS has Levels 1-5,
0 = no rating), age-group and service Y/N flags, phone, address/county/city/
ZIP, lat/long, IsOngoingProcess, and HoursOfOperationList (list of {Day,
ServiceTime}, kept as a JSON string). Distance is relative to the search
origin and is dropped. Capacity, cost, inspections, etc. come from a separate
per-provider call made by ky_crawler.py.

Usage:
    python ky_seed.py --counties "Boone,Fayette,Jefferson" --limit 3   # smoke test, headful
    python ky_seed.py --headless                                       # full 120-county run
    python ky_seed.py --merge-only                                     # rebuild ky_seed.csv from seed_raw/
    python ky_seed.py --counties Boone --force                         # re-pull one county

Deps: pip install playwright pandas && playwright install chromium
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import time
import traceback
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE_URL = 'https://kynect.ky.gov/benefits/'
SEARCH_URL = 'https://kynect.ky.gov/benefits/s/child-care-provider?origin=program-page&language=en_US'

OUT_DIR = 'ky_data'
RAW_DIR = os.path.join(OUT_DIR, 'seed_raw')
MERGED_PATH = os.path.join(OUT_DIR, 'ky_seed.csv')
LOG_FILE = 'ky_seed_log.txt'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

KY_COUNTIES = [
    'Adair', 'Allen', 'Anderson', 'Ballard', 'Barren', 'Bath', 'Bell', 'Boone',
    'Bourbon', 'Boyd', 'Boyle', 'Bracken', 'Breathitt', 'Breckinridge',
    'Bullitt', 'Butler', 'Caldwell', 'Calloway', 'Campbell', 'Carlisle',
    'Carroll', 'Carter', 'Casey', 'Christian', 'Clark', 'Clay', 'Clinton',
    'Crittenden', 'Cumberland', 'Daviess', 'Edmonson', 'Elliott', 'Estill',
    'Fayette', 'Fleming', 'Floyd', 'Franklin', 'Fulton', 'Gallatin',
    'Garrard', 'Grant', 'Graves', 'Grayson', 'Green', 'Greenup', 'Hancock',
    'Hardin', 'Harlan', 'Harrison', 'Hart', 'Henderson', 'Henry', 'Hickman',
    'Hopkins', 'Jackson', 'Jefferson', 'Jessamine', 'Johnson', 'Kenton',
    'Knott', 'Knox', 'Larue', 'Laurel', 'Lawrence', 'Lee', 'Leslie',
    'Letcher', 'Lewis', 'Lincoln', 'Livingston', 'Logan', 'Lyon',
    'McCracken', 'McCreary', 'McLean', 'Madison', 'Magoffin', 'Marion',
    'Marshall', 'Martin', 'Mason', 'Meade', 'Menifee', 'Mercer', 'Metcalfe',
    'Monroe', 'Montgomery', 'Morgan', 'Muhlenberg', 'Nelson', 'Nicholas',
    'Ohio', 'Oldham', 'Owen', 'Owsley', 'Pendleton', 'Perry', 'Pike',
    'Powell', 'Pulaski', 'Robertson', 'Rockcastle', 'Rowan', 'Russell',
    'Scott', 'Shelby', 'Simpson', 'Spencer', 'Taylor', 'Todd', 'Trigg',
    'Trimble', 'Union', 'Warren', 'Washington', 'Wayne', 'Webster',
    'Whitley', 'Wolfe', 'Woodford',
]
assert len(KY_COUNTIES) == 120, f'expected 120 KY counties, got {len(KY_COUNTIES)}'

# Relative to the search origin, not a stable provider attribute.
DROP_SEARCH_FIELDS = ('Distance',)


def create_log_file(path=LOG_FILE):
    with open(path, 'w') as f:
        f.write('')


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


# Set from --verbose: per-request diagnostic detail goes through vlog().
VERBOSE = False


def vlog(message, file=LOG_FILE):
    if VERBOSE:
        log(message, file=file)


def _slug(county):
    return re.sub(r'[^a-z0-9]+', '_', county.lower()).strip('_')


# ---------------------------------------------------------------------------
# Aura response capture. Responses are read with page.expect_response() tied
# to the click that triggers them: route.fetch() failed with a DNS error in
# the Playwright driver's Node process, and a long-lived passive listener
# raced Chromium's eviction of response bodies. kynect batches several Aura
# actions per XHR, so the URL alone doesn't identify the call; the
# x-sfdc-lds-endpoints request header does.
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


ADDRESS_AUTOCOMPLETE_HINT = 'addressautocomplete'
PLACE_DETAIL_HINT = 'getplacedetail'
SEARCH_LDS_HINT = 'getchildcareproviderdetails'


def _mk_response_counter(stats):
    """Traffic log for the timeout debug dump, plus a timestamp of the latest
    autocomplete response for wait_for_autocomplete_quiet()."""
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
                ec = endpoint.casefold()
                if ADDRESS_AUTOCOMPLETE_HINT in ec or PLACE_DETAIL_HINT in ec:
                    stats['last_predictions_ts'] = time.time()
                    stats['predictions_seen'] = stats.get('predictions_seen', 0) + 1
            except Exception:
                pass
    return on_response


NOISE_HOSTS = re.compile(
    r'benefind\.my\.site\.com|benefind\.my\.salesforce-scrt\.com|'
    r'(^|\.)qualtrics\.com', re.I)


def _mk_noise_blocker(stats):
    """Abort an embedded chat widget and Qualtrics session recording: both
    are fire-and-forget embeds that delay the search response."""
    def handle(route):
        stats['blocked'] = stats.get('blocked', 0) + 1
        try:
            route.abort()
        except Exception:
            try:
                route.continue_()
            except Exception:
                pass
    return handle


def _extract_inner(body):
    """Yield each successful action's decoded returnValue dict."""
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
    """Yield `inner` itself, plus one level inside `mapResponse` if present
    (some kynect Apex actions use that envelope)."""
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


def trigger_and_capture(page, stats, action, predicate, timeout_ms=30000, label=''):
    """Run `action` inside a page.expect_response() wait and return the
    matching response's JSON body, or None on timeout/failure."""
    try:
        with page.expect_response(predicate, timeout=timeout_ms) as resp_info:
            action()
    except PWTimeout:
        log(f'[{label}] timed out after {timeout_ms}ms waiting for a matching response')
        debug_dump_stats(stats, _slug(label))
        return None
    except Exception as e:
        log(f'[{label}] action failed: {type(e).__name__}: {e}')
        return None
    resp = resp_info.value
    try:
        return resp.json()
    except Exception as e:
        log(f'[{label}] matching response arrived but its body could not be read: '
            f'{type(e).__name__}: {e}')
        return None


def wait_for_autocomplete_quiet(stats, county, quiet_ms=900, timeout_ms=6000, poll_ms=150):
    """Wait until no autocomplete response has arrived for `quiet_ms`. A late
    one re-opens the suggestions dropdown after it was dismissed, and the
    next interaction then lands inside it."""
    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        last = stats.get('last_predictions_ts')
        if last is None or (time.time() - last) * 1000 >= quiet_ms:
            return True
        time.sleep(poll_ms / 1000.0)
    vlog(f'[{county}] autocomplete traffic never fully settled within '
         f'{timeout_ms}ms -- proceeding anyway')
    return False


# ---------------------------------------------------------------------------
# page interactions
# ---------------------------------------------------------------------------

def dismiss_cookie_banner(page):
    try:
        page.get_by_role('button', name=re.compile(r'^Accept$', re.I)).click(timeout=3000)
    except Exception:
        pass


def modify_search(page):
    """Return to the search form; no-op on the first county."""
    try:
        page.get_by_role('button', name=re.compile('Modify Search', re.I)).click(timeout=4000)
        page.wait_for_timeout(500)
        return True
    except Exception:
        return False


def click_location_tab(page):
    for attempt in (
        lambda: page.get_by_role('button', name=re.compile(r'^Location$', re.I)),
        lambda: page.get_by_role('tab', name=re.compile(r'^Location$', re.I)),
        lambda: page.get_by_text('Location', exact=True),
    ):
        try:
            loc = attempt()
            if loc.count() > 0:
                loc.first.click(timeout=3000)
                return True
        except Exception:
            continue
    return False  # Location is the default tab -- not fatal if unclickable


def _find_location_box(page):
    for attempt in (
        lambda: page.get_by_placeholder(re.compile('address', re.I)),
        lambda: page.get_by_label(re.compile('address', re.I)),
        lambda: page.get_by_role('combobox'),
        lambda: page.get_by_role('textbox'),
    ):
        try:
            loc = attempt()
            if loc.count() > 0:
                return loc.first
        except Exception:
            continue
    return None


def fill_location_box(page, text):
    box = _find_location_box(page)
    if box is None:
        return False
    try:
        box.click(timeout=5000)
        box.fill('')
    except Exception:
        pass
    page.keyboard.type(text, delay=35)
    return True


def pick_first_suggestion(page, county, stats, settle_ms=1200, timeout_ms=8000, poll_ms=200):
    """Click the first autocomplete suggestion. `settle_ms` is waited first
    because polling right after the last keystroke can see an empty listbox."""
    page.wait_for_timeout(settle_ms)

    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        try:
            opts = page.get_by_role('option')
            n = opts.count()
        except Exception:
            n = 0
        if n > 0:
            try:
                first_text = opts.first.inner_text()
            except Exception:
                first_text = '<unreadable>'
            vlog(f'[{county}] {n} suggestion(s) available -- clicking the first: {first_text!r}')
            try:
                opts.first.click(timeout=3000)
            except Exception as e:
                log(f'[{county}] click on first suggestion failed: {e}')
                return False
            wait_for_autocomplete_quiet(stats, county)
            return True
        time.sleep(poll_ms / 1000.0)

    log(f'[FAIL] {county}: no autocomplete suggestion appeared within {timeout_ms}ms')
    return False


def click_search(page):
    """If the option list is still open, dismiss it with Escape plus a click
    in the header bar (outside the search component) before clicking Search."""
    try:
        still_open = page.get_by_role('option').count() > 0
    except Exception:
        still_open = None
    vlog(f'[click_search] option list open right before clicking Search? {still_open}')
    if still_open:
        log('[click_search] still open -- retrying the header-bar dismiss once more')
        try:
            page.keyboard.press('Escape')
            page.mouse.click(700, 243)
        except Exception:
            pass
        page.wait_for_timeout(400)

    btn = page.get_by_role('button', name=re.compile(r'^Search$', re.I)).first
    btn.click(timeout=8000)


def _find_view_more(page):
    for attempt in (
        lambda: page.get_by_role('button', name=re.compile(r'^View More$', re.I)),
        lambda: page.get_by_role('button', name=re.compile('load more', re.I)),
        lambda: page.get_by_text(re.compile(r'^View More$', re.I)),
    ):
        try:
            loc = attempt()
            if loc.count() > 0:
                return loc.first
        except Exception:
            continue
    return None


DEBUG_DIR = os.path.join(RAW_DIR, '_debug')


def debug_screenshot(page, tag):
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        path = os.path.join(DEBUG_DIR, f'{tag}.png')
        page.screenshot(path=path)
        vlog(f'[debug] screenshot -> {path}')
    except Exception as e:
        log(f'[debug] screenshot failed for {tag}: {e}')


def debug_dump_stats(stats, tag):
    """Listener stats plus the last 60 responses of any origin (verbose only)."""
    summary = {k: v for k, v in stats.items() if k != 'recent'}
    vlog(f'[debug:{tag}] listener stats so far this run: {summary}')
    recent = list(stats.get('recent') or [])
    vlog(f'[debug:{tag}] last {len(recent)} response(s) seen, ANY origin '
         f'(method status resource_type url):')
    for line in recent:
        vlog(f'[debug:{tag}]   {line}')


# ---------------------------------------------------------------------------
# per-county
# ---------------------------------------------------------------------------

def collect_search_results(page, stats, county, first_body, click_timeout_ms=10000, max_clicks=60):
    """Merge the first search payload with whatever each View More click
    returns, keyed by ProviderCLRNumber, until the button disappears or two
    clicks in a row add nothing. Returns a list of provider dicts."""
    all_recs = {}
    payload = _get_search_payload(first_body)
    for r in (payload or {}).get('sspChildCareProviderDetails') or []:
        all_recs[r.get('ProviderCLRNumber')] = r
    vlog(f'[{county}] {len(all_recs)} rows after initial search')

    clicks, stable_rounds = 0, 0
    while clicks < max_clicks:
        loc = _find_view_more(page)
        if loc is None:
            vlog(f'[{county}] no more "View More" control -- treating list as complete')
            break
        clicks += 1

        def do_click(loc=loc):
            loc.scroll_into_view_if_needed(timeout=3000)
            loc.click(timeout=5000)

        body = trigger_and_capture(page, stats, do_click, _matches_search_call,
                                    timeout_ms=click_timeout_ms,
                                    label=f'{county} View More #{clicks}')
        if body is None:
            # Expected: the initial response already carries the full result
            # set and View More paginates it client-side with no round-trip.
            vlog(f'[{county}] click #{clicks} on View More produced no matching payload '
                 f'in time -- stopping pagination here (expected)')
            break
        payload = _get_search_payload(body)
        before = len(all_recs)
        for r in (payload or {}).get('sspChildCareProviderDetails') or []:
            all_recs[r.get('ProviderCLRNumber')] = r
        vlog(f'[{county}] {len(all_recs)} rows after View More click #{clicks} '
             f'(+{len(all_recs) - before})')
        if len(all_recs) == before:
            stable_rounds += 1
            if stable_rounds >= 2:
                vlog(f'[{county}] two View More clicks in a row added nothing new -- stopping')
                break
        else:
            stable_rounds = 0

    return list(all_recs.values())


def process_county(page, county, raw_dir, stats, force=False):
    slug = _slug(county)
    existing = os.listdir(raw_dir) if os.path.isdir(raw_dir) else []
    if any(f.startswith(slug + '.') for f in existing) and not force:
        log(f'[skip] {county}: already have a raw capture')
        return True

    query_text = f'{county} COUNTY, KY'
    modify_search(page)  # best-effort; no-op on the first county
    click_location_tab(page)

    if not fill_location_box(page, query_text):
        log(f'[FAIL] {county}: could not find the location search box')
        debug_screenshot(page, f'{slug}_no_location_box')
        return False
    debug_screenshot(page, f'{slug}_1_typed')

    if not pick_first_suggestion(page, county, stats):
        log(f'[FAIL] {county}: no autocomplete suggestion ever appeared -- '
            f'a Search click now would just fail validation')
        debug_screenshot(page, f'{slug}_no_suggestion')
        return False
    debug_screenshot(page, f'{slug}_2_suggestion_picked')

    dismiss_cookie_banner(page)

    first_body = trigger_and_capture(page, stats, lambda: click_search(page),
                                      _matches_search_call, timeout_ms=120000,
                                      label=f'{county} search click')
    debug_screenshot(page, f'{slug}_3_after_search_click')
    if first_body is None:
        log(f'[FAIL] {county}: no search-results payload arrived in time')
        return False

    recs = collect_search_results(page, stats, county, first_body)

    log(f'[{county}] {len(recs)} provider rows total')
    for r in recs:
        r['source_county_query'] = county
        for k in DROP_SEARCH_FIELDS:
            r.pop(k, None)

    raw_path = os.path.join(raw_dir, slug + '.json')
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(recs, f, ensure_ascii=False)
    return True


def process_county_with_retry(page, county, raw_dir, stats, force=False, retries=1):
    success = process_county(page, county, raw_dir, stats, force=force)
    attempt = 0
    while not success and attempt < retries:
        attempt += 1
        log(f'[retry {attempt}] {county}')
        time.sleep(3)
        success = process_county(page, county, raw_dir, stats, force=force)
    return success


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def merge_seed(raw_dir, merged_path):
    files = sorted(Path(raw_dir).glob('*.json')) if os.path.isdir(raw_dir) else []
    if not files:
        log(f'[merge] nothing found in {raw_dir}/')
        return

    frames = []
    for f in files:
        try:
            with open(f, encoding='utf-8') as fh:
                recs = json.load(fh)
        except Exception as e:
            log(f'[merge] could not read {f.name}: {e}')
            continue
        if not recs:
            log(f'[merge] {f.name}: 0 rows')
            continue
        frames.append(pd.DataFrame(recs))
        log(f'[merge] {f.name}: {len(recs)} rows')

    if not frames:
        log('[merge] no readable rows, nothing written')
        return

    full = pd.concat(frames, ignore_index=True, sort=False)

    # Kept as a JSON string so pandas never explodes it into extra rows.
    if 'HoursOfOperationList' in full.columns:
        full['HoursOfOperationList'] = full['HoursOfOperationList'].apply(
            lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)

    # Always a string: letter prefix and any zero-padding preserved.
    full['ProviderCLRNumber'] = full['ProviderCLRNumber'].astype(str).str.strip()
    before = len(full)
    full = full.drop_duplicates(subset='ProviderCLRNumber', keep='first')
    if len(full) != before:
        log(f'[merge] dropped {before - len(full)} duplicate ProviderCLRNumber rows '
            f'(expected -- neighboring counties\' searches overlap)')

    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    full.to_csv(merged_path, index=False)
    log(f'[merge] wrote {len(full)} unique providers x {full.shape[1]} cols -> {merged_path}')
    if 'ProviderType' in full.columns:
        log(f'[merge] ProviderType breakdown:\n{full["ProviderType"].value_counts(dropna=False)}')
    if 'NumberOfStars' in full.columns:
        log(f'[merge] NumberOfStars breakdown:\n{full["NumberOfStars"].value_counts(dropna=False)}')
    if 'LocationCountyDescription' in full.columns:
        n_counties = full['LocationCountyDescription'].nunique(dropna=True)
        log(f'[merge] providers span {n_counties} distinct LocationCountyDescription values')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--counties', default=None,
                    help='Comma-separated subset, e.g. "Boone,Fayette,Jefferson" (default: all 120).')
    ap.add_argument('--limit', type=int, default=None,
                    help='Only process the first N counties in the list.')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='Re-search counties that already have a file in seed_raw/.')
    ap.add_argument('--merge-only', action='store_true',
                    help='Skip scraping; just rebuild ky_seed.csv from seed_raw/.')
    ap.add_argument('--delay-min', type=float, default=2.0)
    ap.add_argument('--delay-max', type=float, default=5.0)
    ap.add_argument('--channel', default=None,
                    help='e.g. "chrome" to use a real installed Chrome instead of bundled Chromium.')
    ap.add_argument('--verbose', action='store_true',
                    help='Full per-request diagnostic logging.')
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    os.makedirs(RAW_DIR, exist_ok=True)
    create_log_file()

    if args.merge_only:
        merge_seed(RAW_DIR, MERGED_PATH)
        return

    counties = KY_COUNTIES
    if args.counties:
        wanted = {c.strip().casefold() for c in args.counties.split(',')}
        counties = [c for c in KY_COUNTIES if c.casefold() in wanted]
        missing = wanted - {c.casefold() for c in counties}
        if missing:
            log(f'[warn] not recognized, skipping: {sorted(missing)}')
    if args.limit:
        counties = counties[:args.limit]

    log(f'Processing {len(counties)} counties -> {RAW_DIR}/')

    with sync_playwright() as p:
        launch_kwargs = {'headless': args.headless,
                          'args': ['--disable-blink-features=AutomationControlled']}
        if args.channel:
            launch_kwargs['channel'] = args.channel
        # Fresh, non-persistent context every run: a reused profile let
        # Salesforce Aura's client-side action cache (IndexedDB) serve stale
        # search results with no network traffic. No CAPTCHA here, so no
        # trust-accumulation benefit is lost.
        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(viewport={'width': 1440, 'height': 1000}, user_agent=UA)
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        stats = {}
        context.on('response', _mk_response_counter(stats))
        context.route(NOISE_HOSTS, _mk_noise_blocker(stats))
        page = context.new_page()

        page.goto(BASE_URL, wait_until='domcontentloaded')
        page.wait_for_timeout(1500)
        page.goto(SEARCH_URL, wait_until='domcontentloaded')
        page.wait_for_timeout(1500)
        dismiss_cookie_banner(page)

        ok, fail = 0, 0
        for county in counties:
            try:
                success = process_county_with_retry(page, county, RAW_DIR, stats,
                                                     force=args.force)
                ok += int(success)
                fail += int(not success)
            except Exception:
                fail += 1
                log(f'[FAIL] {county}: unhandled exception\n{traceback.format_exc()}')
            time.sleep(random.uniform(args.delay_min, args.delay_max))

        context.close()
        browser.close()

    log(f'\nDone: {ok} ok, {fail} failed, out of {len(counties)}.')
    log(f"Listener stats for the whole run: {({k: v for k, v in stats.items() if k != 'recent'})}")
    merge_seed(RAW_DIR, MERGED_PATH)


if __name__ == '__main__':
    main()
