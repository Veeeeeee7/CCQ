"""
nc_capture.py — capture a facility's rendered page for nc_crawler.py.

The DCDEE search portal (ncchildcare.ncdhhs.gov/childcaresearch) is an ASP.NET
WebForms app, so the facility data isn't in the served HTML — it renders after a
search. This tool drives the search in a real browser and saves the rendered
detail DOM (+ a screenshot, + a network log) to nc_captures/. Use it to confirm
the control-id selectors nc_crawler.py relies on, or to re-discover them if the
site changes.

    python nc_capture.py --manual
        Opens a browser at the search page. Search ONE facility (by name or by a
        Facility_ID's license number), open its detail, then press Enter here.

    python nc_capture.py --query 01000203
        Best-effort: types the value into the first search box and submits.

Outputs: nc_captures/rendered.html, screenshot.png, network_log.json, and a
resp_*.txt for each XHR/fetch response body.

Deps: pip install playwright beautifulsoup4 && playwright install chromium
"""

import argparse
import json
import os
import re
import time

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

SEARCH_URL = 'https://ncchildcare.ncdhhs.gov/childcaresearch'
OUT_DIR = 'nc_captures'


def capture(manual=False, query=None, headless=False, wait=4.0):
    os.makedirs(OUT_DIR, exist_ok=True)
    captured = []
    body_index = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=['--disable-blink-features=AutomationControlled'])
        ctx = browser.new_context(viewport={'width': 1400, 'height': 1000})
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
                        fn = os.path.join(OUT_DIR, f'resp_{body_index:03d}_{safe}.txt')
                        with open(fn, 'w', encoding='utf-8') as f:
                            f.write(text)
                        entry['body_file'] = fn
                        entry['body_len'] = len(text)
                captured.append(entry)
            except Exception:
                pass

        page.on('response', on_response)

        print(f'Opening {SEARCH_URL} ...')
        page.goto(SEARCH_URL, wait_until='domcontentloaded')
        time.sleep(wait)

        if manual:
            print('\n>>> In the browser: search ONE facility, open its detail page '
                  '(click a result row), then return here.')
            try:
                input('Press Enter when the detail page is showing... ')
            except EOFError:
                time.sleep(30)
        elif query:
            _best_effort_search(page, query)
            time.sleep(wait)

        try:
            with open(os.path.join(OUT_DIR, 'rendered.html'), 'w', encoding='utf-8') as f:
                f.write(page.content())
        except Exception:
            pass
        try:
            page.screenshot(path=os.path.join(OUT_DIR, 'screenshot.png'), full_page=True)
        except Exception:
            pass
        final_url = page.url
        browser.close()

    with open(os.path.join(OUT_DIR, 'network_log.json'), 'w', encoding='utf-8') as f:
        json.dump(captured, f, indent=2, ensure_ascii=False)

    _report(final_url)
    return captured


def _best_effort_search(page, query):
    for sel in ('input[id*="LicNum" i]', 'input[id*="License" i][type="text"]',
                'input[type="text"]', 'input'):
        try:
            box = page.locator(sel).first
            if box.count() > 0 and box.is_visible():
                box.fill(str(query))
                btn = page.locator('[id*="btnSearchLicNum"]').first
                (btn if btn.count() > 0 else box).click() if btn.count() > 0 else box.press('Enter')
                print(f'  searched with {sel!r}')
                page.wait_for_timeout(3000)
                return True
        except Exception:
            continue
    print('  ! could not find a search box — use --manual')
    return False


def _report(final_url):
    """Confirm the control IDs nc_crawler.py relies on are present in the capture."""
    path = os.path.join(OUT_DIR, 'rendered.html')
    print(f'\nFinal URL: {final_url}')
    print(f'Saved: {OUT_DIR}/rendered.html, screenshot.png, network_log.json')
    if not os.path.exists(path):
        return
    soup = BeautifulSoup(open(path, encoding='utf-8', errors='replace').read(), 'html.parser')
    checks = {
        'basic info (rptBasicFacilityInfo)': 'FacilityDetail_rptBasicFacilityInfo',
        'license/star (lblLicenseType)': 'lblLicenseType',
        'visits (rptVisit_lbVisitDate)': 'rptVisit_lbVisitDate',
        'owner (lblOwnerName)': 'lblOwnerName',
    }
    print('\nControl-id presence in this capture:')
    for label, sub in checks.items():
        found = soup.find(id=re.compile(re.escape(sub))) is not None
        print(f'  [{"OK" if found else "--"}] {label}')
    print('\nIf any show "--", the markup changed; update the *_SEL selectors and '
          'the _m_one/_m_many suffixes in nc_crawler.py to match.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Capture a facility page for nc_crawler.')
    ap.add_argument('--manual', action='store_true',
                    help='Drive the search yourself while the tool captures (recommended).')
    ap.add_argument('--query', default=None,
                    help='Best-effort: type this into the first search box.')
    ap.add_argument('--headless', action='store_true')
    args = ap.parse_args()
    if not args.manual and not args.query:
        ap.error('use --manual (recommended) or --query "<value>"')
    capture(manual=args.manual, query=args.query, headless=args.headless)