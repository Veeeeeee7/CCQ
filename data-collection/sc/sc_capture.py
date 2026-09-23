"""
sc_capture.py — recon the SC Child Care provider search.

https://www.scchildcare.org/provider-search/ is an Umbraco CMS page whose
results are loaded by client-side JavaScript (AJAX); a plain `requests.get()`
returns only the empty form. This drives a real browser, records every
XHR/fetch response body to sc_captures/, saves the rendered HTML and a
screenshot, and ranks the captured bodies by how likely they are to be the
provider-search payload. It does not scrape.

USAGE
-----
    # RECOMMENDED — you drive it:
    python sc_capture.py --manual
        Opens the provider-search page. In the browser:
          1) choose "Search All Providers" (or a Zip/County), click Search,
          2) let the result list render,
          3) click ONE provider to open its detail view,
        then come back to the terminal and press Enter. Everything the page
        requested is dumped to sc_captures/.

    # Best-effort automatic (may miss the right button if markup differs):
    python sc_capture.py --zip 29201

OUTPUTS (sc_captures/)
    rendered.html        - final DOM after your interaction
    screenshot.png       - full-page screenshot
    network_log.json     - every xhr/fetch: url, method, status, content-type
    resp_NNN_*.txt       - each captured response body

Deps: pip install playwright beautifulsoup4 && playwright install chromium
"""

import argparse
import json
import os
import re
import time

from playwright.sync_api import sync_playwright

SEARCH_URL = 'https://www.scchildcare.org/provider-search/'
OUT_DIR = 'sc_captures'

# Keywords used only to rank captured bodies in the summary.
INTEREST_HINTS = ('provider', 'rating', 'quality', 'abcq', 'license',
                   'facility', 'county', 'address')


def capture(manual=False, zip_code=None, headless=False, wait=4.0):
    os.makedirs(OUT_DIR, exist_ok=True)
    captured = []
    body_index = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=['--disable-blink-features=AutomationControlled'])
        ctx = browser.new_context(
            viewport={'width': 1400, 'height': 1000},
            user_agent=('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                        'AppleWebKit/537.36 (KHTML, like Gecko) '
                        'Chrome/120.0.0.0 Safari/537.36'))
        page = ctx.new_page()

        def on_response(resp):
            nonlocal body_index
            try:
                req = resp.request
                if req.resource_type not in ('xhr', 'fetch'):
                    return
                ctype = (resp.headers or {}).get('content-type', '')
                entry = {'url': resp.url, 'method': req.method,
                         'status': resp.status, 'content_type': ctype,
                         'post_data': req.post_data}
                if any(t in ctype for t in ('json', 'text', 'javascript', 'xml')):
                    try:
                        text = resp.text()
                    except Exception:
                        text = ''
                    if text:
                        body_index += 1
                        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', resp.url)[-80:]
                        fn = os.path.join(OUT_DIR, f'resp_{body_index:03d}_{safe}.txt')
                        with open(fn, 'w', encoding='utf-8') as f:
                            f.write(text)
                        entry['body_file'] = fn
                        entry['body_len'] = len(text)
                        low = text.lower()
                        entry['interesting'] = sum(h in low for h in INTEREST_HINTS)
                captured.append(entry)
            except Exception:
                pass

        page.on('response', on_response)

        print(f'Opening {SEARCH_URL} ...')
        page.goto(SEARCH_URL, wait_until='domcontentloaded')
        time.sleep(wait)

        if manual:
            print('\n>>> In the browser: run a search (try "Search All '
                  'Providers", or a Zip/County), let results render, then click '
                  'ONE provider to open its detail view.')
            try:
                input('Press Enter here when a provider detail is showing... ')
            except EOFError:
                time.sleep(45)
        elif zip_code:
            _best_effort_zip(page, zip_code)
            time.sleep(wait)

        try:
            with open(os.path.join(OUT_DIR, 'rendered.html'), 'w',
                      encoding='utf-8') as f:
                f.write(page.content())
        except Exception:
            pass
        try:
            page.screenshot(path=os.path.join(OUT_DIR, 'screenshot.png'),
                            full_page=True)
        except Exception:
            pass
        final_url = page.url
        browser.close()

    with open(os.path.join(OUT_DIR, 'network_log.json'), 'w',
              encoding='utf-8') as f:
        json.dump(captured, f, indent=2, ensure_ascii=False)

    _report(final_url, captured)
    return captured


def _best_effort_zip(page, zip_code):
    """Type a zip into the first plausible search box and submit. Deliberately
    loose — prefer --manual."""
    for sel in ('input[placeholder*="Zip" i]', 'input[placeholder*="Name" i]',
                'input[type="text"]', 'input'):
        try:
            box = page.locator(sel).first
            if box.count() > 0 and box.is_visible():
                box.fill(str(zip_code))
                box.press('Enter')
                print(f'  typed {zip_code!r} into {sel!r} and pressed Enter')
                page.wait_for_timeout(3500)
                return True
        except Exception:
            continue
    print('  ! could not find a search box — re-run with --manual')
    return False


def _report(final_url, captured):
    print(f'\nFinal URL: {final_url}')
    print(f'Saved: {OUT_DIR}/rendered.html, screenshot.png, network_log.json')
    bodies = [c for c in captured if c.get('body_file')]
    if not bodies:
        print('\nNo xhr/fetch JSON/text bodies were captured. The search may '
              'not have fired — re-run with --manual and actually click Search.')
        return
    bodies.sort(key=lambda c: c.get('interesting', 0), reverse=True)
    print(f'\n{len(bodies)} response body(ies) captured. Most likely the '
          'provider-search payload (ranked by keyword hits):')
    for c in bodies[:8]:
        print(f'  [{c.get("interesting", 0)} hits] {c["method"]} '
              f'{c["status"]}  {c["url"]}')
        print(f'        -> {c["body_file"]}  ({c.get("body_len", 0)} bytes)')
    print('\nNext: open the top-ranked resp_*.txt and confirm the provider id / '
          'rating fields.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Capture the SC provider-search network traffic for sc_crawler.')
    ap.add_argument('--manual', action='store_true',
                    help='Drive the search yourself while this records (recommended).')
    ap.add_argument('--zip', dest='zip_code', default=None,
                    help='Best-effort: type this zip into the first search box.')
    ap.add_argument('--headless', action='store_true')
    args = ap.parse_args()
    if not args.manual and not args.zip_code:
        ap.error('use --manual (recommended) or --zip <zipcode>')
    capture(manual=args.manual, zip_code=args.zip_code, headless=args.headless)
