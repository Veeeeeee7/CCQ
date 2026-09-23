"""
Nebraska Step Up to Quality — finder recon / network-capture helper.

Opens the parent-facing finder in a real browser and logs every network
request/response while you drive it by hand, plus rendered-DOM snapshots on
demand, so selectors and URL patterns can be checked against real markup:

    https://stepuptoquality.ne.gov/resources-parents-families/provider-search/

Runs headful by default because the page loads reCAPTCHA v3.

What it writes (to --out-dir, default ne_captures/):
  * network.jsonl  — one JSON object per finished request: method, url,
    request headers/postData, response status/headers, and response body
    (JSON pretty-printed if parseable, else raw text, truncated to --max-body).
  * capture_NNN.html + capture_NNN.png — rendered DOM + screenshot on demand.
  * manifest.csv   — index, timestamp, url, html file for each manual capture.

Usage:
  pip install playwright && playwright install chromium      # one-time
  python ne_capture.py                 # interactive, headful (default)
  python ne_capture.py --out-dir ne_captures --max-body 40000

Interactive flow:
  1. A browser opens on the finder. Choose "Program Name" as the search-by
     mode and type a common term (e.g. "Learning" or "Kids"), OR choose Address
     and enter a city like "Omaha". Press the site's Search button.
  2. When results are displayed, come back to the terminal and press Enter to
     snapshot the DOM + flush a note into network.jsonl. (Network is logged
     continuously regardless.)
  3. Click into one provider's detail view and press Enter again to snapshot
     it — this records the detail URL pattern.
  4. Type 'q' + Enter to quit.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE_URL = 'https://stepuptoquality.ne.gov/resources-parents-families/provider-search/'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/120.0.0.0 Safari/537.36')

# Every response is logged; bodies are kept only for these.
INTERESTING_HINTS = ('admin-ajax', '/wp-json/', 'search', 'provider', 'rating',
                     'step', 'api', 'query', '.json')


def is_interesting(url, method):
    u = url.lower()
    if method.upper() == 'POST':
        return True
    return any(h in u for h in INTERESTING_HINTS)


def install_network_logging(context, out_dir, max_body):
    """Append every finished response (and its request) to network.jsonl."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'network.jsonl')
    log = open(path, 'a', encoding='utf-8')

    def on_response(response):
        try:
            req = response.request
            interesting = is_interesting(req.url, req.method)
            body = None
            ctype = ''
            try:
                ctype = response.headers.get('content-type', '')
            except Exception:
                pass
            if interesting and ('json' in ctype or 'text' in ctype
                                or 'javascript' in ctype or ctype == ''):
                try:
                    raw = response.text()
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            body = json.dumps(parsed, indent=2)[:max_body]
                        except Exception:
                            body = raw[:max_body]
                except Exception:
                    body = None
            entry = {
                'ts': datetime.now().isoformat(timespec='seconds'),
                'interesting': interesting,
                'method': req.method,
                'url': req.url,
                'status': response.status,
                'req_content_type': req.headers.get('content-type', ''),
                'resp_content_type': ctype,
                'post_data': (req.post_data or '')[:max_body] if req.method == 'POST' else None,
                'body': body,
            }
            log.write(json.dumps(entry, ensure_ascii=False) + '\n')
            log.flush()
            if interesting:
                tag = 'POST' if req.method == 'POST' else 'GET '
                print(f'   [net] {tag} {response.status}  {req.url[:120]}')
        except Exception:
            pass

    context.on('response', on_response)
    return log


def wait_settle(page, settle_ms=1200, timeout_ms=15000, poll_ms=300):
    """Content-settle heuristic (do not use networkidle: recaptcha/analytics
    keep polling). Returns once body text length is stable for settle_ms."""
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


def save_capture(page, out_dir, index):
    os.makedirs(out_dir, exist_ok=True)
    html_path = os.path.join(out_dir, f'capture_{index:03d}.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(page.content())
    try:
        page.screenshot(path=os.path.join(out_dir, f'capture_{index:03d}.png'),
                        full_page=True)
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
        w.writerow([index, datetime.now().isoformat(timespec='seconds'),
                    url, os.path.basename(html_path)])


def run_interactive(out_dir, executable_path, max_body):
    with sync_playwright() as p:
        kwargs = {'headless': False}
        if executable_path:
            kwargs['executable_path'] = executable_path
        browser = p.chromium.launch(**kwargs)
        context = browser.new_context(user_agent=UA,
                                      viewport={'width': 1400, 'height': 1000})
        log = install_network_logging(context, out_dir, max_body)
        page = context.new_page()
        page.goto(BASE_URL, wait_until='domcontentloaded')

        print('\n' + '=' * 72)
        print('STEP UP TO QUALITY — FINDER NETWORK CAPTURE')
        print('A browser window is open on the provider search page.')
        print('  1. Run a search (Program Name = a common word, or Address = a city).')
        print('  2. When results show, come here and press Enter to snapshot the page.')
        print('  3. Click a provider detail (if any) and press Enter again.')
        print("  4. Type 'q' then Enter to quit.")
        print('Every network request/response is being logged to network.jsonl')
        print("(interesting ones are printed above with a [net] tag).")
        print('=' * 72 + '\n')

        index = 0
        while True:
            cmd = input("[Enter]=snapshot page, 'q'=quit > ").strip().lower()
            if cmd == 'q':
                break
            pg = context.pages[-1] if context.pages else page
            try:
                wait_settle(pg)
            except PWTimeout:
                pass
            url = pg.url
            html_path = save_capture(pg, out_dir, index)
            append_manifest(out_dir, index, url, html_path)
            print(f'  saved {os.path.basename(html_path)}  <-  {url}')
            index += 1

        log.close()
        browser.close()
        print(f'\nDone. {index} page snapshot(s) + network.jsonl in {out_dir}/.')


def run_urls(urls, out_dir, headless, executable_path, max_body):
    with sync_playwright() as p:
        kwargs = {'headless': headless}
        if executable_path:
            kwargs['executable_path'] = executable_path
        browser = p.chromium.launch(**kwargs)
        context = browser.new_context(user_agent=UA,
                                      viewport={'width': 1400, 'height': 1000})
        log = install_network_logging(context, out_dir, max_body)
        page = context.new_page()
        for index, url in enumerate(urls):
            page.goto(url, wait_until='domcontentloaded')
            try:
                wait_settle(page)
            except PWTimeout:
                pass
            html_path = save_capture(page, out_dir, index)
            append_manifest(out_dir, index, page.url, html_path)
            print(f'  saved {os.path.basename(html_path)}  <-  {page.url}')
            time.sleep(1)
        log.close()
        browser.close()
        print(f'\nDone. {len(urls)} page(s) + network.jsonl in {out_dir}/.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Capture STQ finder search network traffic + rendered DOM.')
    ap.add_argument('--out-dir', default='ne_captures')
    ap.add_argument('--urls', nargs='*', default=None,
                    help='If given, visit these URLs headlessly instead of interactive mode.')
    ap.add_argument('--headless', action='store_true',
                    help='Run headless (only sensible in --urls mode; the finder needs '
                         'real gestures for reCAPTCHA in interactive mode).')
    ap.add_argument('--executable-path', default=None,
                    help='Path to a chromium/chrome binary, if needed.')
    ap.add_argument('--max-body', type=int, default=40000,
                    help='Max chars of each response/POST body to log.')
    args = ap.parse_args()

    if args.urls:
        run_urls(args.urls, args.out_dir, args.headless, args.executable_path, args.max_body)
    else:
        run_interactive(args.out_dir, args.executable_path, args.max_body)
