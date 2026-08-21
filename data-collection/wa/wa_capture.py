"""
wa_capture.py — Washington "Child Care Check" rendered-DOM + network capture
============================================================================

Washington's centers/school-age/outdoor programs already come to us for free
via a documented Socrata open-data API (data.wa.gov dataset `was8-3ni8`), which
carries the Early Achievers rating directly in its `earating` column. That path
needs no browser at all.

The one thing the open dataset does NOT contain is **Family Home Child Care
(FCC) homes** — a group-by on `facilitytypegeneric` returns 0 "Home" rows, and
there is no separate FCC dataset on data.wa.gov. FCC ratings are only published
through the public finder:

    Child Care Check  ->  https://www.findchildcarewa.org/

which is a JavaScript single-page app: a plain HTTP fetch of the root returns an
empty shell, so the provider records are pushed in by JS after bootstrap. That
means we can't inspect it with a normal fetch, and we don't yet know:

  * whether the search results arrive as a clean JSON payload from a backend
    API (which we could hit directly — by far the preferred, most robust path),
    or whether we'd have to scrape rendered DOM;
  * the request shape / pagination of that API, and whether it can enumerate
    ALL family-home providers (e.g. by ZIP, county, or a blank/broad search);
  * whether each FCC record exposes the fields we need — the Early Achievers
    rating, a stable `WacompassId` (our chosen provider_id), facility type,
    and address;
  * the detail-page URL pattern, and whether a provider profile is addressable
    directly by an ID or only reachable by searching (the site mentions a
    per-provider "Search ID").

This script answers those questions. It opens a real, human-driven browser on
--start-url (the finder), and every time you press Enter in this terminal it
saves whatever page is frontmost — rendered HTML, a screenshot, and (cumulative
for the whole session) every XHR/fetch response body. **The network_log.json is
the important artifact**: if the finder's search calls a JSON endpoint, we can
target that directly in wa_crawler.py instead of scraping the DOM.

Two modes:

  Interactive (default): a browser window opens on the finder. Drive it by hand
  — run a broad search, filter to Family Home Child Care, open a provider's
  profile — then come back and press Enter to capture. Repeat; 'q' to quit.

  URL list: pass --urls U1 U2 ... to visit + capture each headlessly (only
  useful once URL patterns are already known — not yet, for us).

Usage:
  python wa_capture.py                         # interactive, headful, starts on the finder
  python wa_capture.py --out-dir wa_captures
  python wa_capture.py --urls https://www.findchildcarewa.org/provider/....   # later

What to send back: wa_captures/manifest.csv, network_log.json, the
resp_*.txt bodies (especially any that look like the search results as JSON),
the capture_*.html files, and the screenshots. Please note:
  * the search request URL + method + payload that returns the result list;
  * whether a Family Home / FCC provider carries an Early Achievers rating and
    a WacompassId (or any stable ID) in that payload;
  * how many results a broad search returns and how it paginates;
  * the detail-page URL pattern and whether it's addressable by ID.

Deps: pip install playwright && playwright install chromium
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# Washington's public "Child Care Check" finder (JS SPA). This is our planned
# SEED + RATING source for Family Home Child Care only; centers/school-age/
# outdoor come from the Socrata API and don't touch this script.
FINDER_URL = 'https://www.findchildcarewa.org/'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/120.0.0.0 Safari/537.36')


def wait_for_render(page, selector=None, settle_ms=1500, timeout_ms=30000,
                    poll_ms=400):
    """Wait for client-side rendering to settle.

    SPA components mount asynchronously. Don't use networkidle — SPAs commonly
    keep background polling/telemetry connections open, which would hang.

    If `selector` is given, wait for that element (once we know a stable
    rendered node, e.g. a results-row class). Otherwise fall back to a
    content-settle heuristic: poll document.body.innerText length and return
    once it's unchanged for `settle_ms`.
    """
    if selector:
        page.wait_for_selector(selector, timeout=timeout_ms)
        return

    deadline = time.time() + timeout_ms / 1000.0
    last_len = -1
    stable_since = None
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
            stable_since = None
            last_len = cur
        time.sleep(poll_ms / 1000.0)
    # timed out -- return anyway; caller still gets whatever rendered


def _mk_network_logger(network_log, out_dir, body_index):
    """Returns a context 'response' handler that records XHR/fetch traffic for
    the whole session. This is how we discover the finder's search/detail JSON
    endpoints instead of scraping rendered HTML — by far the more robust option
    if it exists, and the whole point of this capture."""
    def on_response(resp):
        try:
            req = resp.request
            if req.resource_type not in ('xhr', 'fetch'):
                return
            entry = {'url': resp.url, 'method': req.method,
                     'status': resp.status,
                     'content_type': (resp.headers or {}).get('content-type', '')}
            # Record the request payload too — for a POST search API we need to
            # see the body shape (filters, pagination) to replay it.
            try:
                entry['post_data'] = req.post_data
            except Exception:
                entry['post_data'] = None
            if any(t in entry['content_type'] for t in ('json', 'text', 'javascript')):
                try:
                    text = resp.text()
                except Exception:
                    text = ''
                if text:
                    body_index[0] += 1
                    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', resp.url)[-80:]
                    fn = os.path.join(out_dir, f'resp_{body_index[0]:03d}_{safe}.txt')
                    with open(fn, 'w', encoding='utf-8') as f:
                        f.write(text)
                    entry['body_file'] = os.path.basename(fn)
                    entry['body_len'] = len(text)
            network_log.append(entry)
        except Exception:
            pass
    return on_response


def save_capture(page, out_dir, index, screenshot=True):
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f'capture_{index:03d}.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(page.content())
    if screenshot:
        try:
            page.screenshot(path=os.path.join(out_dir, f'capture_{index:03d}.png'),
                            full_page=True)
        except Exception:
            pass
    return html_path


def append_manifest(out_dir, index, url, html_path, n_xhr_total):
    manifest = os.path.join(out_dir, 'manifest.csv')
    exists = os.path.exists(manifest)
    with open(manifest, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['index', 'captured_at', 'url', 'html_file',
                        'n_xhr_fetch_so_far_this_session'])
        w.writerow([index, datetime.now().isoformat(timespec='seconds'),
                    url, os.path.basename(html_path), n_xhr_total])


def run_interactive(start_url, out_dir, executable_path, wait_selector, settle_ms):
    os.makedirs(out_dir, exist_ok=True)
    with sync_playwright() as p:
        kwargs = {'headless': False,
                  'args': ['--disable-blink-features=AutomationControlled']}
        if executable_path:
            kwargs['executable_path'] = executable_path
        browser = p.chromium.launch(**kwargs)
        context = browser.new_context(user_agent=UA,
                                      viewport={'width': 1400, 'height': 1000})
        page = context.new_page()

        # One growing network log for the whole session (rewritten to disk after
        # every capture). Attaching at the context level means it also catches
        # traffic from a second tab if you open one.
        network_log = []
        body_index = [0]
        context.on('response', _mk_network_logger(network_log, out_dir, body_index))

        page.goto(start_url, wait_until='domcontentloaded')

        print('\n' + '=' * 70)
        print('INTERACTIVE CAPTURE -- Washington (Child Care Check finder)')
        print(f'A browser window is open on:\n  {start_url}')
        print()
        print('Goal: capture the FAMILY HOME CHILD CARE (FCC) path. Centers,')
        print('school-age, and outdoor programs already come from the open-data')
        print('API and are NOT needed here.')
        print()
        print('Suggested pass 1 (enumerate): run a BROAD search — e.g. a single')
        print('populous county/ZIP (King County / Seattle 98104), or a blank/')
        print('map-based search if allowed — and if there is a provider-type')
        print('filter, set it to Family Home Child Care. The aim is to learn how')
        print('the result list is fetched and whether we can page through ALL of')
        print('them. Let the results fully load, then press Enter here.')
        print()
        print('Suggested pass 2 (detail): open ONE family-home provider profile.')
        print('Note whether it shows an Early Achievers rating (Level 1-5) and a')
        print('"Search ID" / provider ID, whether a WacompassId-style id appears')
        print('anywhere, and whether the profile URL is addressable directly by')
        print('that id. Let it load, then press Enter here.')
        print()
        print('The network_log.json (+ resp_*.txt bodies) is the key output: we')
        print('are looking for the JSON search endpoint behind the result list.')
        print()
        print("When a page is fully rendered, come back here and press Enter")
        print("to capture it. Type 'q' then Enter to quit.")
        print('=' * 70 + '\n')

        index = 0
        while True:
            cmd = input("[Enter]=capture current page, 'q'=quit > ").strip().lower()
            if cmd == 'q':
                break
            pg = context.pages[-1] if context.pages else page
            try:
                wait_for_render(pg, selector=wait_selector, settle_ms=settle_ms)
            except PWTimeout:
                print('  (render wait timed out -- capturing current state anyway)')
            url = pg.url
            html_path = save_capture(pg, out_dir, index)
            with open(os.path.join(out_dir, 'network_log.json'), 'w', encoding='utf-8') as f:
                json.dump(network_log, f, indent=2, ensure_ascii=False)
            append_manifest(out_dir, index, url, html_path, len(network_log))
            print(f'  saved {os.path.basename(html_path)}  <-  {url}')
            print(f'  (network_log.json now has {len(network_log)} xhr/fetch total this session)')
            index += 1

        browser.close()
        print(f'\nDone. {index} page(s) saved to {out_dir}/ (see manifest.csv).')
        print(f'Full-session network log: {out_dir}/network_log.json')


def run_urls(urls, out_dir, headless, executable_path, wait_selector, settle_ms):
    os.makedirs(out_dir, exist_ok=True)
    with sync_playwright() as p:
        kwargs = {'headless': headless,
                  'args': ['--disable-blink-features=AutomationControlled']}
        if executable_path:
            kwargs['executable_path'] = executable_path
        browser = p.chromium.launch(**kwargs)
        context = browser.new_context(user_agent=UA,
                                      viewport={'width': 1400, 'height': 1000})
        page = context.new_page()

        network_log = []
        body_index = [0]
        context.on('response', _mk_network_logger(network_log, out_dir, body_index))

        for index, url in enumerate(urls):
            page.goto(url, wait_until='domcontentloaded')
            try:
                wait_for_render(page, selector=wait_selector, settle_ms=settle_ms)
            except PWTimeout:
                print(f'  (render wait timed out for {url})')
            html_path = save_capture(page, out_dir, index)
            append_manifest(out_dir, index, page.url, html_path, len(network_log))
            print(f'  saved {os.path.basename(html_path)}  <-  {page.url}')
            time.sleep(1)

        with open(os.path.join(out_dir, 'network_log.json'), 'w', encoding='utf-8') as f:
            json.dump(network_log, f, indent=2, ensure_ascii=False)
        browser.close()
        print(f'\nDone. {len(urls)} page(s) saved to {out_dir}/.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Capture rendered DOM + network traffic from WA Child Care Check.')
    ap.add_argument('--start-url', default=FINDER_URL,
                    help='Initial URL for interactive mode (default: the finder).')
    ap.add_argument('--out-dir', default='wa_captures')
    ap.add_argument('--urls', nargs='*', default=None,
                    help='If given, visit these URLs instead of interactive mode.')
    ap.add_argument('--headless', action='store_true',
                    help='Run headless (only sensible in --urls mode).')
    ap.add_argument('--executable-path', default=None,
                    help='Path to a chromium/chrome binary, if needed.')
    ap.add_argument('--wait-selector', default=None,
                    help='CSS selector to wait for instead of the settle heuristic '
                         '(set this once you know a stable rendered node).')
    ap.add_argument('--settle-ms', type=int, default=1500)
    args = ap.parse_args()

    if args.urls:
        run_urls(args.urls, args.out_dir, args.headless, args.executable_path,
                 args.wait_selector, args.settle_ms)
    else:
        # interactive is always headful regardless of --headless
        run_interactive(args.start_url, args.out_dir, args.executable_path,
                        args.wait_selector, args.settle_ms)
