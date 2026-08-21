"""
Wisconsin Child Care Finder — rendered-DOM capture helper
=========================================================

The finder (childcarefinder.wisconsin.gov) is a Blazor *Server* app: the
initial HTML is just a bootstrap shell, and the real provider / youngstar /
regulation / provider-reported content is pushed into the DOM over a SignalR
websocket *after* blazor.web.js runs. A plain save-as or HTTP fetch therefore
captures nothing useful.

This script opens a real browser, waits for the Blazor render to settle, and
dumps the *rendered* HTML (page.content()) plus a screenshot so the section
selectors can be written from real markup. It also records the URL of each
captured page to a manifest, which is how we learn the detail-page routing.

Two modes:

  Interactive (default): a browser window opens on the finder home page. You
  navigate manually (search, click into a provider, open tabs/sections), then
  press Enter in the terminal to capture whatever is currently on screen. Repeat
  for as many providers as you like; type 'q' to quit. This is the mode to use
  first, since we don't yet know the detail-page URL pattern — you discover it
  by navigating and letting the manifest record the URLs.

  URL list: pass --urls U1 U2 ... and each is visited and captured headlessly.
  Use this only once you know the URL pattern.

Usage:
  python wi_capture.py                       # interactive, headful
  python wi_capture.py --out-dir captures
  python wi_capture.py --urls https://childcarefinder.wisconsin.gov/...   # later
"""

import argparse
import csv
import os
import time
from datetime import datetime

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE_URL = 'https://childcarefinder.wisconsin.gov'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/120.0.0.0 Safari/537.36')


def wait_for_render(page, selector=None, settle_ms=1500, timeout_ms=30000,
                    poll_ms=400):
    """Wait for the Blazor app to finish rendering.

    Do NOT use networkidle here: a Blazor Server app keeps a SignalR websocket
    open, so the network never goes idle and networkidle waits would hang.

    If `selector` is given, wait for that element (preferred once we know a
    stable node in the rendered page). Otherwise fall back to a content-settle
    heuristic: poll document.body.innerText length and return once it has been
    unchanged for `settle_ms`.
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
    # timed out — return anyway; caller still gets whatever rendered


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


def append_manifest(out_dir, index, url, html_path):
    manifest = os.path.join(out_dir, 'manifest.csv')
    exists = os.path.exists(manifest)
    with open(manifest, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['index', 'captured_at', 'url', 'html_file'])
        w.writerow([index, datetime.now().isoformat(timespec='seconds'),
                    url, os.path.basename(html_path)])


def run_interactive(out_dir, executable_path, wait_selector, settle_ms):
    with sync_playwright() as p:
        kwargs = {'headless': False}
        if executable_path:
            kwargs['executable_path'] = executable_path
        browser = p.chromium.launch(**kwargs)
        context = browser.new_context(user_agent=UA,
                                      viewport={'width': 1400, 'height': 1000})
        page = context.new_page()
        page.goto(BASE_URL, wait_until='domcontentloaded')

        print('\n' + '=' * 70)
        print('INTERACTIVE CAPTURE')
        print('A browser window is open on the Child Care Finder.')
        print('Navigate to a provider detail page (search, then click in).')
        print("When the page is fully shown, come back here and press Enter")
        print("to capture it. Type 'q' then Enter to quit.")
        print('=' * 70 + '\n')

        index = 0
        while True:
            cmd = input("[Enter]=capture current page, 'q'=quit > ").strip().lower()
            if cmd == 'q':
                break
            # capture whatever page/tab is frontmost
            pg = context.pages[-1] if context.pages else page
            try:
                wait_for_render(pg, selector=wait_selector, settle_ms=settle_ms)
            except PWTimeout:
                print('  (render wait timed out — capturing current state anyway)')
            url = pg.url
            html_path = save_capture(pg, out_dir, index)
            append_manifest(out_dir, index, url, html_path)
            print(f'  saved {os.path.basename(html_path)}  <-  {url}')
            index += 1

        browser.close()
        print(f'\nDone. {index} page(s) saved to {out_dir}/ (see manifest.csv).')


def run_urls(urls, out_dir, headless, executable_path, wait_selector, settle_ms):
    with sync_playwright() as p:
        kwargs = {'headless': headless}
        if executable_path:
            kwargs['executable_path'] = executable_path
        browser = p.chromium.launch(**kwargs)
        context = browser.new_context(user_agent=UA,
                                      viewport={'width': 1400, 'height': 1000})
        page = context.new_page()
        for index, url in enumerate(urls):
            page.goto(url, wait_until='domcontentloaded')
            try:
                wait_for_render(page, selector=wait_selector, settle_ms=settle_ms)
            except PWTimeout:
                print(f'  (render wait timed out for {url})')
            html_path = save_capture(page, out_dir, index)
            append_manifest(out_dir, index, page.url, html_path)
            print(f'  saved {os.path.basename(html_path)}  <-  {page.url}')
            time.sleep(1)
        browser.close()
        print(f'\nDone. {len(urls)} page(s) saved to {out_dir}/.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Capture rendered DOM from the WI Child Care Finder.')
    ap.add_argument('--out-dir', default='wi_captures')
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
        run_interactive(args.out_dir, args.executable_path,
                        args.wait_selector, args.settle_ms)
