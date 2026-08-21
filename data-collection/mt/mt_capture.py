"""
mt_capture.py — Montana MAQCS rendered-DOM + network capture helper
====================================================================

Montana's licensing directory -- our source for the real license number
(provider_id) that the STARS rating PDF lacks -- is the "Licensed Provider
Search", which is embedded from a Salesforce Experience Cloud (Aura/LWC) site:

    https://mtdphhs.my.site.com/MAQCSChildCareLicensing/s/provider-search?language=en_US

(The DPHHS wrapper page is dphhs.mt.gov/ecfsd/childcare/childcarelicensing/
providersearch, whose instructions say: "Click the Search button to pull up a
full list of providers" -- i.e. a blank search enumerates the whole directory,
the MI/CCHIRP pattern.) A plain HTTP fetch of the my.site.com URL returns only
the Lightning bootstrap shell; the real content is pushed in by JS after
bootstrap and LWC renders into shadow DOM that string-serialized page.content()
can't see. So this opens a real, human-driven browser and captures rendered
HTML + a screenshot + the full session's XHR/fetch traffic whenever you press
Enter.

Why the network capture matters most here: if the search results come back as a
clean JSON payload from an Aura/Apex endpoint (aura?r=...&other.ApexAction.
execute or similar), we can hit that directly from mt_crawler.py -- far more
robust than scraping shadow-DOM rows, and the playbook's preferred mechanism.
MI's CCHIRP showed exactly this shape (a ~3MB ApexAction response holding the
whole state); watch for the same here.

What we need to learn from this recon (please note in your reply):
  1. Does a blank "Search" return ALL licensed providers, or only a slice? Is
     there a total count, and pagination?
  2. Is there a "Download for export"/"Export" button (MI had one that dumped
     the full result set)? What file type + columns does it produce?
  3. What columns/fields are shown per provider -- especially the LICENSE /
     REGISTRATION NUMBER, the legal + business name, city, and program type
     (Center / Group / Family / etc.). We join these to the STARS seed on
     name+city, so anything that disambiguates same-name programs helps.
  4. The detail-page URL pattern -- is a provider addressable directly by its
     license number, or only reachable via search?
  5. Confirm whether the STARS/QRS star rating appears here at all (expected:
     NO -- licensing only). If it somehow does, flag it: that would collapse
     the two-source join into one source.
  6. Any reCAPTCHA / bot-scoring (check the CSP header / network log for
     google.com/recaptcha), so the crawler can budget for the persistent-
     context + warm-up playbook.

Two modes:

  Interactive (default): a headful browser opens on MAQCS. Do a blank search,
  note the total, open one provider, then (if present) click Export. Press
  Enter here to capture whatever is frontmost. Repeat; type 'q' to quit.

  URL list: --urls U1 U2 ... visits + captures each (only useful once URL
  patterns are known).

Usage:
  python mt_capture.py                       # interactive, headful, starts on MAQCS
  python mt_capture.py --out-dir mt_captures
  python mt_capture.py --channel chrome      # use real Chrome if reCAPTCHA is fussy
  python mt_capture.py --urls <detail-url>   # later, once a URL pattern is known

What to send back: mt_captures/manifest.csv, the capture_*.html files,
network_log.json, the resp_*.txt bodies (especially any large JSON one), and
the screenshots. Call out anything that looks like a license/provider ID and
the export column headers.

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

MAQCS_URL = ('https://mtdphhs.my.site.com/MAQCSChildCareLicensing/s/'
             'provider-search?language=en_US')
# DPHHS wrapper page (server-rendered) that embeds/points to the above.
DPHHS_WRAPPER = ('https://dphhs.mt.gov/ecfsd/childcare/childcarelicensing/'
                 'providersearch')

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/120.0.0.0 Safari/537.36')


def wait_for_render(page, selector=None, settle_ms=1500, timeout_ms=30000,
                    poll_ms=400):
    """Wait for client-side rendering to settle.

    Salesforce Lightning/Aura components mount asynchronously. Don't use
    networkidle -- Experience Cloud keeps background polling/telemetry
    connections open, so networkidle would hang. If `selector` is given, wait
    for it (once a stable results-row node is known); otherwise poll
    document.body.innerText length until it's unchanged for `settle_ms`.
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
    """Record XHR/fetch traffic for the whole session. This is how we'd
    discover a clean Aura/Apex JSON endpoint behind the search instead of
    scraping rendered HTML -- far more robust if it exists."""
    def on_response(resp):
        try:
            req = resp.request
            if req.resource_type not in ('xhr', 'fetch'):
                return
            entry = {'url': resp.url, 'method': req.method,
                     'status': resp.status,
                     'content_type': (resp.headers or {}).get('content-type', '')}
            # Capture the request body too -- Aura POSTs the message/action
            # payload we'd need to replay in the crawler.
            try:
                pd = req.post_data
                if pd:
                    entry['post_data'] = pd[:4000]
            except Exception:
                pass
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


def run_interactive(start_url, out_dir, channel, wait_selector, settle_ms):
    os.makedirs(out_dir, exist_ok=True)
    with sync_playwright() as p:
        kwargs = {'headless': False,
                  'args': ['--disable-blink-features=AutomationControlled']}
        if channel:
            kwargs['channel'] = channel
        browser = p.chromium.launch(**kwargs)
        context = browser.new_context(user_agent=UA,
                                      viewport={'width': 1400, 'height': 1000})
        # Hide the automation flag (reCAPTCHA-v3 playbook, harmless otherwise).
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()

        # One growing network log for the whole session, attached at the
        # context level so a second tab is captured too.
        network_log = []
        body_index = [0]
        context.on('response', _mk_network_logger(network_log, out_dir, body_index))

        page.goto(start_url, wait_until='domcontentloaded')

        print('\n' + '=' * 70)
        print('INTERACTIVE CAPTURE -- Montana MAQCS licensing search')
        print(f'A browser window is open on:\n  {start_url}')
        print(f'(wrapper page, if the embed misbehaves: {DPHHS_WRAPPER})')
        print()
        print('Suggested pass 1: click Search with NO filters. Note the total')
        print('count and whether every licensed provider is listed. Look for an')
        print('Export / "Download for export" button and click it if present.')
        print()
        print('Suggested pass 2: open ONE provider. Note the LICENSE/REGISTRATION')
        print('NUMBER, the legal + business name, city, and program type. Check')
        print('the detail URL -- is it addressable by license number directly?')
        print('Confirm the STARS star rating does NOT appear (licensing only).')
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


def run_urls(urls, out_dir, headless, channel, wait_selector, settle_ms):
    os.makedirs(out_dir, exist_ok=True)
    with sync_playwright() as p:
        kwargs = {'headless': headless,
                  'args': ['--disable-blink-features=AutomationControlled']}
        if channel:
            kwargs['channel'] = channel
        browser = p.chromium.launch(**kwargs)
        context = browser.new_context(user_agent=UA,
                                      viewport={'width': 1400, 'height': 1000})
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
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
        description='Capture rendered DOM + network traffic from MT MAQCS.')
    ap.add_argument('--start-url', default=MAQCS_URL,
                    help='Initial URL for interactive mode (default: MAQCS search).')
    ap.add_argument('--out-dir', default='mt_captures')
    ap.add_argument('--urls', nargs='*', default=None,
                    help='If given, visit these URLs instead of interactive mode.')
    ap.add_argument('--headless', action='store_true',
                    help='Run headless (only sensible in --urls mode).')
    ap.add_argument('--channel', default=None,
                    help='e.g. "chrome" to use a real installed Chrome instead '
                         'of the bundled Chromium, if reCAPTCHA gets fussy.')
    ap.add_argument('--wait-selector', default=None,
                    help='CSS selector to wait for instead of the settle '
                         'heuristic (set once a stable rendered node is known).')
    ap.add_argument('--settle-ms', type=int, default=1500)
    args = ap.parse_args()

    if args.urls:
        run_urls(args.urls, args.out_dir, args.headless, args.channel,
                 args.wait_selector, args.settle_ms)
    else:
        # interactive is always headful regardless of --headless
        run_interactive(args.start_url, args.out_dir, args.channel,
                        args.wait_selector, args.settle_ms)
