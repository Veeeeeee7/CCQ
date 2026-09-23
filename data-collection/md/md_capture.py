"""
md_capture.py — rendered-DOM + network-log capture for Maryland's two sites.

  findaprogram.marylandexcels.org — the public EXCELS quality-rating finder.
  Results render client-side; the network log shows the JSON/CSV API the Vue
  front end calls, which md_crawler.py hits directly.

  checkccmd.org — MSDE's separate licensing/inspection lookup ("Check Child
  Care Maryland"), an ASP.NET WebForms app whose results render only after a
  postback. The pre-search page (search_page.html) and its form controls are
  saved and listed.

Interactive: a browser window opens on the chosen site. Search manually, open
a result/detail view, then press Enter in the terminal to capture whatever is
on screen; 'q' to quit. --query VALUE types VALUE into what looks like the
first text box and submits before the interactive loop starts.

Usage:
  python md_capture.py --site excels
  python md_capture.py --site checkccmd
  python md_capture.py --site excels --query "Bright Beginnings"

Outputs (to md_captures/<site>/):
  search_page.html          — DOM before any interaction
  capture_NNN.html          — rendered DOM at each manual capture
  capture_NNN.png           — screenshot at each manual capture
  manifest.csv              — index, timestamp, URL, filename per capture
  network_log.json          — every XHR/fetch seen during the whole session
  resp_NNN_<url-tail>.txt   — JSON/text bodies of those XHR/fetch responses

Deps: pip install playwright beautifulsoup4 && playwright install chromium
"""

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

SITES = {
    'excels': {
        'url': 'https://findaprogram.marylandexcels.org/',
        'tips': (
            "Search a provider name or filter by county, then open a result."
        ),
    },
    'checkccmd': {
        'url': 'https://www.checkccmd.org/Default.aspx',
        'tips': (
            "Search a facility name and submit, then open one result to its "
            "inspection-detail view."
        ),
    },
}

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/120.0.0.0 Safari/537.36')


def wait_for_render(page, selector=None, settle_ms=1500, timeout_ms=20000, poll_ms=400):
    """Wait for `selector`, or until body text length is stable for
    `settle_ms`. Not networkidle, which hangs on a page holding a live
    connection open."""
    if selector:
        page.wait_for_selector(selector, timeout=timeout_ms)
        return
    deadline = time.time() + timeout_ms / 1000.0
    last_len = -1
    stable_since = None
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
            stable_since = None
            last_len = cur
        time.sleep(poll_ms / 1000.0)


def save_html(page, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page.content())


def append_manifest(out_dir, index, url, html_path):
    manifest = os.path.join(out_dir, 'manifest.csv')
    exists = os.path.exists(manifest)
    with open(manifest, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['index', 'captured_at', 'url', 'html_file'])
        w.writerow([index, datetime.now().isoformat(timespec='seconds'),
                    url, os.path.basename(html_path)])


def _summarize_form(path):
    """Print name/id/type of each form control in a saved page."""
    if not os.path.exists(path):
        return
    soup = BeautifulSoup(open(path, encoding='utf-8', errors='replace').read(), 'html.parser')
    controls = soup.find_all(['input', 'select', 'textarea'])
    if not controls:
        print(f'  (no <input>/<select>/<textarea> controls found in {os.path.basename(path)} '
              f'— likely means the form is injected by JS after load)')
        return
    print(f'\n  Form controls found in {os.path.basename(path)}:')
    for c in controls:
        name = c.get('name') or c.get('id') or '(unnamed)'
        ctype = c.get('type', c.name)
        print(f'    {ctype:10s} {name}')


def _best_effort_search(page, query):
    for sel in ('input[type="search"]', 'input[type="text"]', 'input'):
        try:
            box = page.locator(sel).first
            if box.count() > 0 and box.is_visible():
                box.fill(str(query))
                box.press('Enter')
                print(f'  searched with {sel!r}')
                page.wait_for_timeout(3000)
                return True
        except Exception:
            continue
    print('  ! could not find a search box automatically — search manually instead')
    return False


def capture(site, query=None, headless=False, settle_ms=1500, wait_selector=None):
    cfg = SITES[site]
    out_dir = os.path.join('md_captures', site)
    os.makedirs(out_dir, exist_ok=True)
    network_log = []
    body_index = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=['--disable-blink-features=AutomationControlled'])
        ctx = browser.new_context(user_agent=UA,
                                   viewport={'width': 1400, 'height': 1000})
        page = ctx.new_page()

        def on_response(resp):
            nonlocal body_index
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
                        body_index += 1
                        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', resp.url)[-80:]
                        fn = os.path.join(out_dir, f'resp_{body_index:03d}_{safe}.txt')
                        with open(fn, 'w', encoding='utf-8') as f:
                            f.write(text)
                        entry['body_file'] = fn
                        entry['body_len'] = len(text)
                network_log.append(entry)
            except Exception:
                pass

        page.on('response', on_response)

        print(f'Opening {cfg["url"]} ...')
        page.goto(cfg['url'], wait_until='domcontentloaded')
        time.sleep(2)
        search_page_path = os.path.join(out_dir, 'search_page.html')
        save_html(page, search_page_path)
        print('  saved search_page.html (pre-search DOM)')
        _summarize_form(search_page_path)

        if query:
            _best_effort_search(page, query)

        print('\n' + '=' * 70)
        print(f'INTERACTIVE CAPTURE — {site}')
        print(cfg['tips'])
        print("\nWhen a result/detail page is fully shown, come back here and "
              "press Enter to capture it. Type 'q' then Enter to quit.")
        print('=' * 70 + '\n')

        n_captures = 0
        while True:
            cmd = input("[Enter]=capture current page, 'q'=quit > ").strip().lower()
            if cmd == 'q':
                break
            pg = ctx.pages[-1] if ctx.pages else page
            try:
                wait_for_render(pg, selector=wait_selector, settle_ms=settle_ms)
            except PWTimeout:
                print('  (render wait timed out — capturing current state anyway)')
            url = pg.url
            html_path = os.path.join(out_dir, f'capture_{n_captures:03d}.html')
            save_html(pg, html_path)
            try:
                pg.screenshot(path=os.path.join(out_dir, f'capture_{n_captures:03d}.png'),
                              full_page=True)
            except Exception:
                pass
            append_manifest(out_dir, n_captures, url, html_path)
            print(f'  saved capture_{n_captures:03d}.html  <-  {url}')
            n_captures += 1

        browser.close()

    with open(os.path.join(out_dir, 'network_log.json'), 'w', encoding='utf-8') as f:
        json.dump(network_log, f, indent=2, ensure_ascii=False)

    _report(out_dir, network_log, n_captures)


def _report(out_dir, network_log, n_captures):
    print(f'\nDone. Saved to {out_dir}/')
    print(f'  {n_captures} manual capture(s), {len(network_log)} XHR/fetch call(s) logged.')
    endpoints = sorted({f"{e['method']} {re.sub(r'^https?://[^/]+', '', e['url'])}"
                        for e in network_log})
    if endpoints:
        print('\n  Distinct XHR/fetch endpoints hit (path only):')
        for e in endpoints[:30]:
            print(f'    {e}')
        json_hits = [e for e in network_log if 'json' in e.get('content_type', '')]
        if json_hits:
            print(f'\n  {len(json_hits)} response(s) looked like JSON — check resp_*.txt first.')
    else:
        print('  No XHR/fetch traffic captured — data may be server-rendered in the main')
        print('  document (check capture_*.html directly), or the session ended before any')
        print('  request fired.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Capture rendered DOM + network traffic for MD sites.')
    ap.add_argument('--site', required=True, choices=list(SITES.keys()))
    ap.add_argument('--query', default=None,
                    help='Best-effort: type this into the first search box before the '
                         'interactive loop starts.')
    ap.add_argument('--headless', action='store_true',
                    help='Not recommended — you need to see the page to navigate it.')
    ap.add_argument('--wait-selector', default=None,
                    help='CSS selector to wait for instead of the settle heuristic '
                         '(set this once a stable rendered node is known).')
    ap.add_argument('--settle-ms', type=int, default=1500)
    args = ap.parse_args()

    capture(args.site, query=args.query, headless=args.headless,
            settle_ms=args.settle_ms, wait_selector=args.wait_selector)
