"""
co_capture.py — rendered-DOM + network capture for coloradoshines.com/search.

CONFIRMED by the first real capture (TINY HEART ACADEMY, provider_id 1696651):
this runs on Salesforce VISUALFORCE, not Lightning/Aura -- the markup carries
com.salesforce.visualforce.ViewState[MAC/Version] hidden fields, classic JSF
auto-generated component ids (page:j_id62:j_id63:j_id65:programnamefield), and
RichFaces/Ajax4jsf calls (A4J.AJAX.Submit(...)) for in-page sort/filter. The
"Find a Program" search itself is a plain <form method="post" action="/search">
full-page postback, not an isolated JSON API call -- confirmed by testing a
stateless GET of the same URL, which came back with empty results (Visualforce
needs the session/ViewState built up by an actual page visit). So this still
needs a real browser; a `requests`-only crawl for the search step is not
viable without much deeper ViewState-replay engineering that isn't worth the
fragility. Detail pages (program_details?id=<18-char Salesforce id>) render
with a very consistent <strong>Label:</strong> value pattern and CSS-class-
encoded ratings (span.rating-2 etc.), so extraction there is straightforward
once we have the id -- this script's job is mainly to confirm the search flow
and any remaining markup questions for co_crawler.py.

The site sits on the same Salesforce backend as the "Provider Hub" login
(decl.my.site.com), so it gets the same precautions as the WI/Blazor case in
the playbook: persistent context, a warm-up hit to the site root, and
navigator.webdriver patched out. (Unlike WI's Blazor Server app, this is NOT
a persistent-websocket page -- it's a normal request/response postback -- but
the content-settle wait is kept anyway since third-party trackers on the page
mean networkidle may never fire.)

Every time you press Enter, this captures TWO things:
  1. The rendered DOM + a screenshot of whatever page is on screen right now
     (capture_NNN.html / .png, logged to manifest.csv) -- same idea as
     wi_capture.py, so you can capture a search-results state, click into a
     result, capture the detail page, then another provider, etc.
  2. Every request/response to coloradoshines.com for the WHOLE session (not
     just at capture time, and not just xhr/fetch -- a first pass that
     filtered to xhr/fetch only completely missed the real search request,
     since it's a full-page 'document' postback, not an XHR). Non-Colorado
     hosts (Google Ads/Analytics/Amplitude/Maps -- confirmed to be 100+
     tracking beacons carrying zero provider data) are skipped entirely so
     the log stays readable. POST bodies get decoded into individual form
     fields, with the giant ViewState blob redacted to a length.

Usage:
  python co_capture.py
      Opens a browser on coloradoshines.com/search. Search for a SPECIFIC
      provider by name -- try one straight from co_data/co_seed.csv, e.g.
      "TINY HEART ACADEMY" or "EAST LAKE MONTESSORI" -- so the resulting
      detail page's license number can be cross-checked against a known
      provider_id. Once results show, come back here and press Enter to
      capture. Click into that provider's detail page, press Enter again.
      Repeat for another provider or two (ideally a different service type,
      e.g. a family child care home) if you have the patience, then 'q'.

  python co_capture.py --out-dir co_data/captures --settle-ms 2000

Outputs (in --out-dir): capture_NNN.html, capture_NNN.png, manifest.csv,
network_log.json, resp_NNN_*.txt (bodies of interesting responses).

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

BASE_URL = 'https://www.coloradoshines.com'
SEARCH_URL = f'{BASE_URL}/search'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# Field labels already confirmed to exist on real program_details pages (from
# indexed search snippets -- see project chat) but never seen in real markup.
# After a capture, we grep for these as a sanity check that we reached real
# content rather than a blocked/loading/error state.
EXPECTED_DETAIL_LABELS = [
    'Hours of Operation', 'Accepts CCCAP', 'Head Start', 'Licensed to Serve',
    'Languages Spoken', 'Special Needs', 'License Type', 'License Issue Date',
    'Program Licensing Information', 'License Number', 'Openings Available',
]

_INIT_SCRIPT = "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"


def wait_for_render(page, selector=None, settle_ms=1500, timeout_ms=30000, poll_ms=400):
    """Content-settle heuristic (copied from wi_capture.py). Do NOT wait on
    networkidle -- a Lightning/Experience Cloud page may never go idle."""
    if selector:
        page.wait_for_selector(selector, timeout=timeout_ms)
        return
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
    # timed out -- return anyway; caller still gets whatever rendered


def save_capture(page, out_dir, index):
    html_path = os.path.join(out_dir, f'capture_{index:03d}.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(page.content())
    try:
        page.screenshot(path=os.path.join(out_dir, f'capture_{index:03d}.png'), full_page=True)
    except Exception:
        pass
    return html_path


def append_manifest(out_dir, index, url, html_path):
    manifest = os.path.join(out_dir, 'manifest.csv')
    exists = os.path.exists(manifest)
    with open(manifest, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['index', 'captured_at', 'url', 'html_file'])
        w.writerow([index, datetime.now().isoformat(timespec='seconds'), url,
                    os.path.basename(html_path)])


def make_network_logger(out_dir, focus_host='coloradoshines.com'):
    """Returns (on_response handler, list that accumulates entries for the
    whole session -- registered at the context level so it sees new tabs
    too, e.g. if clicking a result opens the detail page in a new page).

    Only requests whose host contains `focus_host` are logged in detail; the
    rest of the web is Google Ads/Analytics/Amplitude/Maps noise (confirmed
    by an earlier capture -- 100+ tracking beacons, zero of which carried
    any provider data) and is skipped entirely to keep this readable.

    Logs EVERY resource type for the focus host, not just xhr/fetch. The
    first real capture showed the search is a classic Visualforce <form
    method="post" action="/search"> full-page postback (confirmed by
    com.salesforce.visualforce.ViewState hidden fields and A4J.AJAX.Submit
    calls in the markup) -- Playwright tags that as a 'document' navigation,
    not 'xhr'/'fetch', so the original xhr/fetch-only filter silently missed
    the one request that actually matters.

    POST bodies get their form fields decoded and logged individually, with
    any ViewState-ish field (the encoded server-side component-tree blob,
    often tens of KB) redacted to a length so the log stays readable while
    still showing exactly which real fields (program name, address, etc.)
    were submitted.
    """
    captured = []
    body_index = 0

    def _decode_post_data(req):
        try:
            raw = req.post_data
        except Exception:
            raw = None
        if not raw:
            return None
        try:
            from urllib.parse import parse_qsl, unquote
            pairs = parse_qsl(raw, keep_blank_values=True)
            if not pairs:
                return {'_raw_len': len(raw)}
            decoded = {}
            for k, v in pairs:
                if 'viewstate' in k.lower() or len(v) > 300:
                    decoded[k] = f'<omitted, {len(v)} chars>'
                else:
                    decoded[k] = unquote(v)
            return decoded
        except Exception:
            return {'_raw_len': len(raw)}

    def on_response(resp):
        nonlocal body_index
        try:
            req = resp.request
            if focus_host not in req.url:
                return  # third-party ad/analytics/maps noise -- skip entirely
            entry = {'url': resp.url, 'method': req.method, 'status': resp.status,
                      'resource_type': req.resource_type,
                      'content_type': (resp.headers or {}).get('content-type', '')}
            post_fields = _decode_post_data(req)
            if post_fields:
                entry['post_fields'] = post_fields
            if any(t in entry['content_type'] for t in ('json', 'text', 'html', 'javascript')):
                try:
                    text = resp.text()
                except Exception:
                    text = ''
                if text:
                    body_index += 1
                    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', resp.url)[-80:]
                    fn = os.path.join(out_dir, f'resp_{body_index:03d}_{safe}.txt')
                    with open(fn, 'w', encoding='utf-8') as f:
                        f.write(text)
                    entry['body_file'] = fn
                    entry['body_len'] = len(text)
            captured.append(entry)
        except Exception:
            pass

    return on_response, captured


def report(out_dir):
    print('\nPlatform fingerprint (Visualforce/JSF markers):')
    fingerprints = {
        'Visualforce ViewState': 'com.salesforce.visualforce.ViewState',
        'JSF auto-id pattern': re.compile(r'id="page:j_id\d+'),
        'RichFaces/Ajax4jsf': 'A4J.AJAX.Submit',
        'Program Name field': 'programnamefield',
        'Rating CSS class': re.compile(r'class="rating rating-\d"'),
    }
    combined_text = ''
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith('.html'):
            combined_text += open(os.path.join(out_dir, fn), encoding='utf-8', errors='replace').read()
    for label, pattern in fingerprints.items():
        found = bool(pattern.search(combined_text)) if hasattr(pattern, 'search') else pattern in combined_text
        print(f'  [{"OK" if found else "--"}] {label}')
    print('  (if these start showing "--", the site migrated off this stack -- '
          'selectors in co_crawler.py will need a fresh look)')

    print('\nField-label sanity check across all captured HTML files:')
    found_any = {label: False for label in EXPECTED_DETAIL_LABELS}
    id_hits = set()
    for fn in sorted(os.listdir(out_dir)):
        if not fn.endswith('.html'):
            continue
        text = open(os.path.join(out_dir, fn), encoding='utf-8', errors='replace').read()
        for label in EXPECTED_DETAIL_LABELS:
            if label in text:
                found_any[label] = True
        id_hits.update(re.findall(r'program_details\?id=([A-Za-z0-9]+)', text))
    for label, ok in found_any.items():
        print(f'  [{"OK" if ok else "--"}] {label}')
    if id_hits:
        print(f'\nFound {len(id_hits)} program_details id(s) embedded in captured HTML:')
        for i in list(id_hits)[:10]:
            print(f'  id={i}')
    else:
        print('\nNo program_details ids found in the raw HTML -- if search results '
              'are links, they may be built client-side from a network response '
              'instead. Check network_log.json / resp_*.txt for the id there.')
    print(f'\nAll output is in {out_dir}/: capture_NNN.html/.png, manifest.csv, '
          f'network_log.json, resp_NNN_*.txt')


def run(out_dir, executable_path, wait_selector, settle_ms):
    os.makedirs(out_dir, exist_ok=True)
    on_response, network_log = make_network_logger(out_dir)

    with sync_playwright() as p:
        kwargs = {'headless': False, 'args': ['--disable-blink-features=AutomationControlled']}
        if executable_path:
            kwargs['executable_path'] = executable_path
        else:
            kwargs['channel'] = 'chrome'  # prefer a real Chrome build if available
        try:
            browser = p.chromium.launch(**kwargs)
        except Exception:
            kwargs.pop('channel', None)
            browser = p.chromium.launch(**kwargs)  # fall back to bundled Chromium

        context = browser.new_context(user_agent=UA, viewport={'width': 1400, 'height': 1000})
        context.add_init_script(_INIT_SCRIPT)
        context.on('response', on_response)
        page = context.new_page()

        # Warm-up hit to the site root before the search page -- the plan's
        # recommendation for reCAPTCHA v3-gated Salesforce/Blazor sites.
        print(f'Warming up at {BASE_URL} ...')
        page.goto(BASE_URL, wait_until='domcontentloaded')
        time.sleep(2)

        print(f'Navigating to {SEARCH_URL} ...')
        page.goto(SEARCH_URL, wait_until='domcontentloaded')
        try:
            wait_for_render(page, selector=wait_selector, settle_ms=settle_ms)
        except PWTimeout:
            pass

        print('\n' + '=' * 70)
        print('INTERACTIVE CAPTURE')
        print('A browser window is open on the Colorado Shines search page.')
        print('Search for a SPECIFIC provider by name (try one from')
        print('co_data/co_seed.csv, e.g. "TINY HEART ACADEMY") so we can')
        print('cross-check its license number against our seed. Once results')
        print("show, come back here and press Enter to capture. Then click into")
        print("that provider's detail page and press Enter again. Repeat for")
        print("another provider or two if you can. 'q' then Enter to quit.")
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
            append_manifest(out_dir, index, url, html_path)
            print(f'  saved {os.path.basename(html_path)}  <-  {url}')
            index += 1

        browser.close()

    with open(os.path.join(out_dir, 'network_log.json'), 'w', encoding='utf-8') as f:
        json.dump(network_log, f, indent=2, ensure_ascii=False)

    print(f'\nDone. {index} page(s) captured, {len(network_log)} coloradoshines.com '
          f'request(s) logged (third-party tracking noise excluded).')
    report(out_dir)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Capture rendered DOM + network traffic from coloradoshines.com/search.')
    ap.add_argument('--out-dir', default='co_data/captures')
    ap.add_argument('--executable-path', default=None,
                     help='Path to a chromium/chrome binary, if needed.')
    ap.add_argument('--wait-selector', default=None,
                     help='CSS selector to wait for instead of the settle heuristic '
                          '(set this once you know a stable rendered node).')
    ap.add_argument('--settle-ms', type=int, default=1500)
    args = ap.parse_args()
    run(args.out_dir, args.executable_path, args.wait_selector, args.settle_ms)