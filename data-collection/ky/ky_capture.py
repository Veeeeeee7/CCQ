"""
ky_capture.py — Kentucky rendered-DOM + network capture helper

Kentucky publishes per-provider Kentucky All STARS ratings through kynect's
Public Child Care Search:

    https://kynect.ky.gov/benefits/s/child-care-provider?origin=program-page&language=en_US

A plain HTTP fetch returns only a Salesforce Lightning/Aura bootstrap shell
("Sorry to interrupt / CSS Error"), so the page needs a real browser.

Opens a human-driven browser on --start-url; every time you press Enter in
this terminal it saves the frontmost page (rendered HTML + screenshot) and
the session's cumulative XHR/fetch traffic, which is where the Aura/Apex JSON
payloads behind the search show up.

Two modes:

  Interactive (default): search, open a couple of provider detail pages,
  press Enter to capture the frontmost page; 'q' to quit.

  URL list: --urls U1 U2 ... visits and captures each URL (optionally
  headless).

Usage:
  python ky_capture.py                        # interactive, headful, starts on the search page
  python ky_capture.py --out-dir ky_captures
  python ky_capture.py --urls "https://kynect.ky.gov/benefits/s/child-care-provider-details?..."   # headless

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

SEARCH_URL = 'https://kynect.ky.gov/benefits/s/child-care-provider?origin=program-page&language=en_US'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/120.0.0.0 Safari/537.36')

# Platform fingerprint printed after every capture.
FINGERPRINTS = {
    'Salesforce Lightning/Aura bootstrap': re.compile(r'auraConfig|/auraFW/|lightning/lightning\.out'),
    'Salesforce Experience Cloud site path': re.compile(r's/sfsites|sfdcStatic'),
    'reCAPTCHA': re.compile(r'recaptcha', re.I),
    'Visualforce (older Salesforce -- unlikely here, but worth knowing)':
        re.compile(r'com\.salesforce\.visualforce\.ViewState'),
}


def wait_for_render(page, selector=None, settle_ms=1500, timeout_ms=30000,
                    poll_ms=400):
    """Wait for `selector`, or until body innerText length is unchanged for
    `settle_ms`. Not networkidle: Experience Cloud keeps background
    connections open, so networkidle would hang."""
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


def _mk_network_logger(network_log, out_dir, body_index):
    """'response' handler recording XHR/fetch traffic (and bodies) for the
    whole session."""
    def on_response(resp):
        try:
            req = resp.request
            if req.resource_type not in ('xhr', 'fetch'):
                return
            entry = {'url': resp.url, 'method': req.method,
                     'status': resp.status,
                     'content_type': (resp.headers or {}).get('content-type', '')}
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
    html = page.content()
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    if screenshot:
        try:
            page.screenshot(path=os.path.join(out_dir, f'capture_{index:03d}.png'),
                            full_page=True)
        except Exception:
            pass
    found = [name for name, pat in FINGERPRINTS.items() if pat.search(html)]
    print(f'  platform fingerprint: {", ".join(found) if found else "(none of the known markers matched)"}')
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

        # Context-level, so traffic from a second tab is caught too.
        network_log = []
        body_index = [0]
        context.on('response', _mk_network_logger(network_log, out_dir, body_index))

        page.goto(start_url, wait_until='domcontentloaded')

        print('\n' + '=' * 70)
        print('INTERACTIVE CAPTURE -- Kentucky')
        print(f'A browser window is open on:\n  {start_url}')
        print()
        print('Run a search, then open one or more provider results.')
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
        description="Capture rendered DOM + network traffic from kynect's KY child care provider search.")
    ap.add_argument('--start-url', default=SEARCH_URL,
                    help='Initial URL for interactive mode (default: kynect Public Child Care Search).')
    ap.add_argument('--out-dir', default='ky_captures')
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
