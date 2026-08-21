"""
California Child Care Crawler — mychildcareplan.org

Given a list of facility numbers (CCL license numbers), this script searches
each one on mychildcareplan.org, navigates to the provider details page, and
scrapes all available information into a pandas DataFrame.

Adapted from the Georgia DECAL crawler. Same overall structure:
  - one row per provider
  - per-section try/except so a single broken field doesn't kill the row
  - resumable via start_index
  - file-based logging
"""

import numpy as np
import pandas as pd
import json
import time
import os
import re
import random
import traceback
from playwright.sync_api import (sync_playwright, TimeoutError as PWTimeout,
                                 Error as PWError)


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def create_log_file(path='ca_crawler_log.txt'):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as f:
        f.write('')


def log(message, file='ca_crawler_log.txt'):
    with open(file, 'a') as f:
        f.write(message + '\n')
        print(message)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

BASE_URL = 'https://mychildcareplan.org'
HOME_URL = BASE_URL + '/'
SEARCH_URL = BASE_URL + '/provider-search/'


# ---------------------------------------------------------------------------
# block detection + resilient navigation
# ---------------------------------------------------------------------------

# The site runs Wordfence. When it rate-limits your IP it either serves this
# block page or simply stops responding (navigation then just times out). Both
# are treated as a transient "blocked" condition: the row is marked retryable,
# and after enough of them in a row the crawler cools down and relaunches
# rather than burning through the whole queue marking everything an error.
WORDFENCE_BLOCK_TEXT = 'Your access to this site has been limited by the site owner'


class NavBlocked(Exception):
    """Raised when a navigation fails or returns the Wordfence block page."""


def goto_with_retry(page, url, wait_until='domcontentloaded',
                    timeout=45000, attempts=3):
    """Navigate with exponential backoff. Returns True on success, False if
    every attempt failed (timeout or network error)."""
    for i in range(attempts):
        try:
            page.goto(url, wait_until=wait_until, timeout=timeout)
            return True
        except (PWTimeout, PWError) as e:
            wait = 10 * (2 ** i)  # 10s, 20s, 40s
            msg = str(e).splitlines()[0]
            log(f'  goto failed {url} (attempt {i + 1}/{attempts}: {msg}); '
                f'backing off {wait}s')
            time.sleep(wait)
    return False


def looks_blocked(page):
    """True if the current page is the Wordfence 'access limited' screen."""
    try:
        h1 = page.locator('h1').first
        if h1.count() > 0:
            return WORDFENCE_BLOCK_TEXT in (_clean(_text(h1)) or '')
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# incremental save / resume
# ---------------------------------------------------------------------------

def all_columns():
    """The full ordered list of columns the crawler may produce. Used to
    keep the on-disk CSV consistent across appends, even when individual
    sections fail and produce a subset of keys."""
    cols = ['facility_number', 'provider_url']
    for fn in [empty_header, empty_contact, empty_hours, empty_license,
               empty_tags, empty_openings, empty_basics, empty_about,
               empty_address]:
        cols.extend(fn().keys())
    cols.append('errors')
    return cols


def append_row(row, output_csv):
    """Upsert a single row into the output CSV by facility number.

    If the facility already exists, the newest scrape overwrites the
    existing row instead of adding a duplicate. If older duplicate rows are
    already present, they are collapsed so the file stays one row per
    facility_number.
    """
    cols = all_columns()
    full_row = {}
    for col in cols:
        v = row.get(col)
        if isinstance(v, str):
            v = re.sub(r'\s+', ' ', v).strip()
            v = v if v else None
        full_row[col] = v
    parent = os.path.dirname(output_csv)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    new_row = pd.DataFrame([full_row], columns=cols)
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        # The output file may already contain one or more malformed legacy
        # rows, so read it with the Python engine and skip bad lines instead
        # of crashing the entire crawl.
        existing = pd.read_csv(
            output_csv,
            dtype=str,
            engine='python',
            on_bad_lines='skip',
        )
        for col in cols:
            if col not in existing.columns:
                existing[col] = None
        existing = existing[cols]
        existing = existing.dropna(subset=['facility_number'])
        existing['facility_number'] = existing['facility_number'].astype(str).str.strip()
        existing = existing.drop_duplicates(subset=['facility_number'], keep='last')

        facility_number = new_row.at[0, 'facility_number']
        if facility_number is not None:
            facility_number = str(facility_number).strip()
        if facility_number:
            existing = existing[existing['facility_number'] != facility_number]

        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row

    temp_csv = output_csv + '.tmp'
    updated.to_csv(temp_csv, index=False)
    os.replace(temp_csv, output_csv)


def load_retry_facilities(output_csv):
    """Return facility numbers whose latest row should be retried.

    The retry queue is based on the latest row per facility in
    facility_records.csv, limited to rows whose current errors value is
    either not_found or exception.
    """
    if not output_csv or not os.path.exists(output_csv):
        return []
    try:
        df = pd.read_csv(
            output_csv,
            dtype=str,
            usecols=['facility_number', 'errors'],
            engine='python',
            on_bad_lines='skip',
        )
        df = df.dropna(subset=['facility_number'])
        df['facility_number'] = df['facility_number'].astype(str).str.strip()
        df['errors'] = df['errors'].fillna('').astype(str).str.strip().str.lower()
        df = df.drop_duplicates(subset=['facility_number'], keep='last')
        retryable = df[df['errors'].isin({'not_found', 'exception'})]
        return retryable['facility_number'].tolist()
    except Exception as e:
        log(f'Could not read existing {output_csv}: {e}')
        return []


# ---------------------------------------------------------------------------
# main crawler
# ---------------------------------------------------------------------------

def crawler(facility_numbers, output_csv='ca_data/ca_records.csv',
            downloads_folder=None, headless=True, start_index=0,
            executable_path=None, slow_mo=0, delay_range=(8, 15),
            nav_timeout_ms=45000, nav_attempts=3,
            max_consecutive_failures=5, cooldown_seconds=600,
            relaunch_every=1000):
    """
    Args:
        facility_numbers: list of license/facility numbers (str or int).
        output_csv: path to a CSV that records scraped rows as they're
                    captured. The file is created on first write, appended
                    to thereafter, and used to resume — any facility number
                    already present in this file is skipped on subsequent
                    runs. Pass None to disable both behaviors.
        downloads_folder: where to save anything downloadable. If None,
                          downloads are skipped. (Most CA data is on-page,
                          but the CDSS link can be followed for reports.)
        headless:  run browser headlessly.
        start_index:  resume from this index (legacy; the output_csv resume
                      is preferred and works on top of this).
        executable_path:  path to chrome binary (matches your GA script).
        slow_mo: ms delay between Playwright actions (useful for debugging).
        delay_range: (min, max) seconds to sleep between facilities. A
                     uniform random value in this range is chosen for each
                     iteration to avoid a predictable request cadence.

    Returns:
        (DataFrame of rows scraped *this run*, last_index_processed)
    """
    rows = []

    with sync_playwright() as p:
        launch_kwargs = {
            'headless': headless,
            'slow_mo': slow_mo,
            'args': ['--incognito'],
        }
        if executable_path:
            launch_kwargs['executable_path'] = executable_path
        browser = p.chromium.launch(**launch_kwargs)

        # User-agent / viewport reused for every per-facility context below
        ua = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
              'AppleWebKit/537.36 (KHTML, like Gecko) '
              'Chrome/120.0.0.0 Safari/537.36')

        index = start_index
        consecutive_failures = 0
        processed = 0
        try:
            for index in range(start_index, len(facility_numbers)):
                facility_number = str(facility_numbers[index]).strip()

                row = {'facility_number': facility_number}
                errors = []
                failed = False

                # Fresh incognito-isolated context for every facility — no
                # cookies, cache, or storage carry over from the previous one.
                context = browser.new_context(
                    viewport={'width': 1400, 'height': 900},
                    user_agent=ua,
                )
                page = context.new_page()
                try:
                    # cookie banner / first navigation
                    provider_url = find_provider_url(
                        page, facility_number,
                        nav_timeout_ms=nav_timeout_ms, nav_attempts=nav_attempts)
                    if provider_url is None:
                        log(f'[{index}] NOT FOUND: {facility_number}')
                        row.update(create_empty_row())
                        row['provider_url'] = None
                        row['errors'] = 'dne'
                        rows.append(row)
                        if output_csv:
                            append_row(row, output_csv)
                        page.close()
                        continue

                    if not goto_with_retry(page, provider_url,
                                           timeout=nav_timeout_ms,
                                           attempts=nav_attempts):
                        raise NavBlocked('provider-page navigation timed out')
                    try:
                        page.wait_for_load_state('networkidle', timeout=15000)
                    except PWTimeout:
                        pass
                    time.sleep(1)

                    if looks_blocked(page):
                        log(f'[{index}] BLOCKED: {facility_number} '
                            f'(Wordfence limited access)')
                        failed = True
                        row.update(create_empty_row())
                        row['errors'] = 'exception'
                        rows.append(row)
                        if output_csv:
                            append_row(row, output_csv)
                        page.close()
                        continue

                    row['provider_url'] = page.url

                    # extract each section independently
                    for section_name, fn, empty_fn in [
                        ('header',   crawl_header,   empty_header),
                        ('contact',  crawl_contact,  empty_contact),
                        ('hours',    crawl_hours,    empty_hours),
                        ('license',  crawl_license,  empty_license),
                        ('tags',     crawl_tags,     empty_tags),
                        ('openings', crawl_openings, empty_openings),
                        ('basics',   crawl_basics,   empty_basics),
                        ('about',    crawl_about,    empty_about),
                        ('address',  crawl_address,  empty_address),
                    ]:
                        try:
                            row.update(fn(page))
                        except Exception:
                            errors.append(section_name)
                            row.update(empty_fn())
                            log(f'  ! {section_name} failed for {facility_number}: '
                                f'{traceback.format_exc().splitlines()[-1]}')

                    if errors:
                        log(f'[{index}] PARTIAL: {facility_number} '
                            f'(errors: {errors})')
                    else:
                        log(f'[{index}] OK: {facility_number}')

                    row['errors'] = ','.join(errors)
                    rows.append(row)
                    if output_csv:
                        append_row(row, output_csv)

                except NavBlocked as e:
                    log(f'[{index}] BLOCKED: {facility_number} ({e})')
                    failed = True
                    row.update(create_empty_row())
                    row['errors'] = 'exception'
                    rows.append(row)
                    if output_csv:
                        append_row(row, output_csv)
                except Exception:
                    log(f'[{index}] EXCEPTION: {facility_number}')
                    log(traceback.format_exc())
                    failed = True
                    row.update(create_empty_row())
                    row['errors'] = 'exception'
                    rows.append(row)
                    if output_csv:
                        append_row(row, output_csv)
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass
                    try:
                        context.close()
                    except Exception:
                        pass
                    # politeness delay between requests — random in delay_range
                    # to avoid a regular cadence. Skipped on the last facility.
                    if index < len(facility_numbers) - 1:
                        delay = random.uniform(*delay_range)
                        log(f'  ...sleeping {delay:.1f}s before next request')
                        time.sleep(delay)

                    # ---- failure tracking / circuit breaker -------------------
                    # This runs even on the `continue` paths above (a finally
                    # block always executes), so blocked and not-found rows are
                    # counted too. A genuine "not found" (dne) means the site
                    # responded fine, so it resets the streak.
                    processed += 1
                    consecutive_failures = consecutive_failures + 1 if failed else 0
                    is_last = index >= len(facility_numbers) - 1

                    if not is_last and consecutive_failures >= max_consecutive_failures:
                        log(f'  !! {consecutive_failures} failures in a row — '
                            f'likely IP rate-limited. Cooling down '
                            f'{cooldown_seconds}s and relaunching the browser.')
                        try:
                            browser.close()
                        except Exception:
                            pass
                        time.sleep(cooldown_seconds)
                        browser = p.chromium.launch(**launch_kwargs)
                        consecutive_failures = 0
                    elif (not is_last and relaunch_every
                          and processed % relaunch_every == 0):
                        log(f'  ...periodic browser relaunch after {processed} '
                            f'facilities (frees accumulated memory).')
                        try:
                            browser.close()
                        except Exception:
                            pass
                        browser = p.chromium.launch(**launch_kwargs)

        except Exception:
            log(f'CRASHED at index {index} for facility '
                f'{facility_numbers[index] if index < len(facility_numbers) else "?"}')
            log(traceback.format_exc())
            try:
                browser.close()
            except Exception:
                pass
            return pd.DataFrame(rows), index

        browser.close()

    return pd.DataFrame(rows), index


# ---------------------------------------------------------------------------
# search → provider URL
# ---------------------------------------------------------------------------

def find_provider_url(page, facility_number, nav_timeout_ms=45000, nav_attempts=3):
    """
    Use the site's header search bar to look up a facility number and return
    the resulting provider-details URL. Returns None if genuinely not found,
    and raises NavBlocked if navigation fails or the site blocks us (so the
    caller can mark the row retryable instead of as "does not exist").
    """
    if not goto_with_retry(page, HOME_URL, timeout=nav_timeout_ms,
                           attempts=nav_attempts):
        raise NavBlocked('home navigation timed out')
    if looks_blocked(page):
        raise NavBlocked('Wordfence block on home page')
    time.sleep(1)

    # accept cookie banner if it shows up — selector is best-effort
    try:
        accept = page.locator('button:has-text("Accept"), '
                              'button:has-text("Agree"), '
                              'button:has-text("OK")').first
        if accept.is_visible(timeout=1500):
            accept.click()
            time.sleep(0.3)
    except Exception:
        pass

    # The header has a search input. We try multiple selector strategies
    # because the site's markup may vary slightly across pages.
    search_input = None
    candidates = [
        'input[placeholder*="Search" i]',
        'input[type="search"]',
        'input[name*="search" i]',
        'input[aria-label*="search" i]',
        '#search',
        '.search input',
        'header input[type="text"]',
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0 and loc.is_visible(timeout=1500):
                search_input = loc
                break
        except Exception:
            continue

    if search_input is None:
        # fallback: go straight to the provider-search page and try its query string
        if not goto_with_retry(page, SEARCH_URL + f'?search={facility_number}',
                               timeout=nav_timeout_ms, attempts=nav_attempts):
            raise NavBlocked('search navigation timed out')
        time.sleep(2)
    else:
        # Interact with short timeouts so a flaky box fails fast instead of
        # hanging for the full 30s default, and fall back to the query-string
        # search if anything goes wrong.
        try:
            search_input.click(timeout=5000)
            search_input.fill(str(facility_number), timeout=5000)
            time.sleep(0.3)
            # Use keyboard.press (sends the key to the focused element) rather
            # than locator.press: locator.press re-resolves the element and
            # waits for it to be visible/stable, which can hang the full
            # timeout when the typeahead dropdown re-renders the header right
            # after fill. The box is already focused from the click+fill above.
            page.keyboard.press('Enter')
        except (PWTimeout, PWError):
            log(f'  search-box interaction failed for {facility_number}; '
                f'falling back to direct search URL')
            if not goto_with_retry(page, SEARCH_URL + f'?search={facility_number}',
                                   timeout=nav_timeout_ms, attempts=nav_attempts):
                raise NavBlocked('search navigation timed out')
        time.sleep(2)
        try:
            page.wait_for_load_state('networkidle', timeout=10000)
        except PWTimeout:
            pass

    # Some queries land directly on the details page, others on a results list.
    if '/provider-details' in page.url:
        return page.url

    # Look for a "View Details" link or a card link to the provider.
    # Provider result cards have a link/button that goes to /provider-details/?...
    detail_link = None
    link_selectors = [
        'a[href*="/provider-details"]',
        'a:has-text("View Details")',
        'a:has-text("View details")',
    ]
    for sel in link_selectors:
        try:
            loc = page.locator(sel).first
            if loc.count() > 0:
                href = loc.get_attribute('href')
                if href and '/provider-details' in href:
                    detail_link = href
                    break
        except Exception:
            continue

    if detail_link:
        if detail_link.startswith('/'):
            detail_link = BASE_URL + detail_link
        return detail_link

    # Last resort: try clicking the first result card and reading the URL
    try:
        card = page.locator('a:has-text("View Details")').first
        if card.count() > 0:
            with page.expect_navigation(timeout=10000):
                card.click()
            if '/provider-details' in page.url:
                return page.url
    except Exception:
        pass

    # No provider link found. If the page is the Wordfence block, this is a
    # transient block (retryable), not a genuine "does not exist" — surface it
    # as NavBlocked so the caller doesn't mark the row dne.
    if looks_blocked(page):
        raise NavBlocked('Wordfence block on search results')
    return None


# ---------------------------------------------------------------------------
# scrapers — one per section
# ---------------------------------------------------------------------------

# ---- header (name, type, licensed/claimed status, photo) -------------------

def empty_header():
    return {
        'provider_name': None,
        'provider_type': None,
        'licensed_status': None,
        'claimed_status': None,
        'profile_photo_url': None,
    }


def crawl_header(page):
    out = empty_header()

    # name is the main h1 inside the details body
    h1 = page.locator('h1').first
    if h1.count() > 0:
        out['provider_name'] = _clean(_text(h1))

    # provider_type is shown as a short label near the top. The site renders
    # both the short label ("Small Family Child Care Home") AND a long
    # tooltip explanation ("Small family child care homes may care for...")
    # — we want only the short label. The reliable way is to scan all body
    # text for the canonical type strings and take the *shortest* element
    # that matches, since the short label sits in a small badge while the
    # tooltip is in a long paragraph.
    canonical_types = [
        'Small Family Child Care Home',
        'Large Family Child Care Home',
        'Family Child Care Home',
        'Child Care Center',
        'License-Exempt Center',
        'License Exempt Center',
        'Center',
    ]
    type_pattern = '|'.join(re.escape(t) for t in canonical_types)
    body_text = _all_text(page.locator('body'))
    # find the very first occurrence — the badge appears before the tooltip
    m = re.search(rf'\b({type_pattern})\b', body_text)
    if m:
        # double-check by walking through canonical_types longest-first to
        # avoid "Center" matching inside "Child Care Center"
        for t in sorted(canonical_types, key=len, reverse=True):
            if t in body_text:
                out['provider_type'] = t
                break

    # Licensed / Not Licensed — store as clean canonical strings
    if re.search(r'\bNot Licensed\b', body_text):
        out['licensed_status'] = 'Not Licensed'
    elif re.search(r'\bLicensed\b', body_text):
        out['licensed_status'] = 'Licensed'

    # Claimed / Unclaimed
    if re.search(r'\bUnclaimed\b', body_text):
        out['claimed_status'] = 'Unclaimed'
    elif re.search(r'\bClaimed\b', body_text):
        out['claimed_status'] = 'Claimed'

    # profile photo — an <img> sourced from partners.mychildcareplan.org
    img = page.locator('img[src*="partners.mychildcareplan.org/docs/ProviderPhotos"]').first
    if img.count() > 0:
        out['profile_photo_url'] = img.get_attribute('src')

    return out


# ---- contact (phone, website) ---------------------------------------------

def empty_contact():
    return {'phone': None, 'website_url': None}


def crawl_contact(page):
    out = empty_contact()

    # The site has a hotline tel: link at the top of every page (1-800-543-7793
    # / 1-800-KIDS-793). We skip that and find the *provider's* phone, which
    # appears further down the page.
    HOTLINE_PATTERNS = [
        r'1?\s*\(?800\)?\s*543[-\s]?7793',
        r'1?\s*\(?800\)?\s*KIDS[-\s]?793',
    ]
    def is_hotline(num):
        return any(re.search(pat, num, re.I) for pat in HOTLINE_PATTERNS)

    tels = page.locator('a[href^="tel:"]')
    for i in range(tels.count()):
        href = (tels.nth(i).get_attribute('href') or '').replace('tel:', '').strip()
        if href and not is_hotline(href):
            out['phone'] = _clean(href)
            break

    # external website link — skip social/internal/known-non-provider hosts
    skip_hosts = ('mychildcareplan.org', 'addtoany.com', 'facebook.com',
                  'twitter.com', 'instagram.com', 'youtube.com',
                  'rrnetwork.org', 'ccld.dss.ca.gov', 'leginfo.legislature.ca.gov',
                  'childcare.gov', 'googletagmanager.com', 'google.com',
                  'partners.mychildcareplan.org')
    links = page.locator('a[href^="http"]')
    for i in range(links.count()):
        href = links.nth(i).get_attribute('href') or ''
        if not any(h in href for h in skip_hosts):
            out['website_url'] = _clean(href)
            break

    return out


# ---- business hours --------------------------------------------------------

def empty_hours():
    return {'business_hours': None}


def crawl_hours(page):
    out = empty_hours()
    # The Business Hours panel is collapsed by default — its DOM is present
    # but not in the visible accessibility tree, so we use textContent
    # (via _all_text) instead of innerText.
    block = page.locator('text=/Business Hours/i').first
    if block.count() == 0:
        return out
    container = block.locator('xpath=ancestor::*[self::div or self::section or self::li][1]')
    if container.count() == 0:
        return out
    text = _all_text(container)
    lines = re.findall(r'(?:Mo|Tu|We|Th|Fr|Sa|Su)\s+[0-9: APMapm-]+', text)
    if lines:
        out['business_hours'] = ' | '.join(_clean(l) for l in lines if _clean(l))
    return out


# ---- license number + reports link ----------------------------------------

def empty_license():
    return {'license_number': None, 'licensing_reports_url': None}


def crawl_license(page):
    out = empty_license()
    body_text = _all_text(page.locator('body'))
    m = re.search(r'License\s*Number\s*[:\s]\s*([A-Z0-9-]+)', body_text, re.I)
    if m:
        out['license_number'] = _clean(m.group(1))

    reports = page.locator('a[href*="ccld.dss.ca.gov/carefacilitysearch"]').first
    if reports.count() > 0:
        out['licensing_reports_url'] = _clean(reports.get_attribute('href'))
    return out


# ---- tags (program types) -------------------------------------------------

def empty_tags():
    return {'tags_count': None, 'tags': None}


def crawl_tags(page):
    out = empty_tags()
    # Tags section has a heading like "View Tags (N)" and a list of tags
    # below. Each tag has a "Learn More" link of the form
    # /resources/?search=<Tag+Name>. We extract tags from those hrefs.
    heading = page.locator('text=/View Tags\\s*\\(\\d+\\)/').first
    if heading.count() > 0:
        m = re.search(r'\((\d+)\)', _all_text(heading))
        if m:
            out['tags_count'] = int(m.group(1))

    tag_links = page.locator('a[href*="resources/?search="]')
    seen, ordered = set(), []
    from urllib.parse import unquote
    for i in range(tag_links.count()):
        href = tag_links.nth(i).get_attribute('href') or ''
        m = re.search(r'search=([^&]+)', href)
        if not m:
            continue
        tag = unquote(m.group(1).replace('+', ' '))
        tag = _clean(tag)
        if tag and tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    if ordered:
        out['tags'] = '|'.join(ordered)
        if out['tags_count'] is None:
            out['tags_count'] = len(ordered)
    return out


# ---- openings -------------------------------------------------------------

def empty_openings():
    return {
        'openings_last_updated': None,
        'openings_status': None,
        'openings_capacity': None,
    }


def crawl_openings(page):
    """Parse the Openings section.

    The section has the form:
        Openings (last updated MM/DD/YYYY)
        <group name>           e.g. "Preschool (2 to 5 years)"
        <status>               "No Openings" or a list of openings
        Capacity: <N>
        [next section: The Basics ...]

    We need to bound the scope so we don't accidentally slurp The Basics.
    Strategy: get the heading's containing section, then crop the resulting
    text at the first marker of the next section.
    """
    out = empty_openings()
    heading = page.locator('text=/Openings\\s*\\(last updated/i').first
    if heading.count() == 0:
        return out

    heading_text = _all_text(heading)
    m = re.search(r'last updated\s+([0-9/.-]+)', heading_text, re.I)
    if m:
        out['openings_last_updated'] = _clean(m.group(1).rstrip(') '))

    # walk up to the openings section/container
    container = heading.locator(
        'xpath=ancestor::*[self::section or self::div or self::article][1]'
    )
    if container.count() == 0:
        return out
    raw = _all_text(container)

    # Crop everything from "Openings (..." onwards (drop the header above)
    if 'Openings (' in raw:
        raw = 'Openings (' + raw.split('Openings (', 1)[1]

    # Stop at the next section heading
    for stop_marker in ['The Basics', 'About ', 'Next Steps',
                        'Disclaimer', 'Contact ', 'Tags']:
        idx = raw.find(stop_marker)
        if idx > 0:
            raw = raw[:idx]

    # Strip the "Openings (last updated MM/DD/YYYY)" prefix itself
    raw = re.sub(r'^Openings\s*\(last updated[^)]*\)\s*',
                 '', raw, count=1, flags=re.I)

    # Capacity: pull it out then strip from the remaining status text
    cap_match = re.search(r'Capacity\s*:\s*([0-9]+)', raw, re.I)
    if cap_match:
        out['openings_capacity'] = _clean(cap_match.group(1))
        raw = re.sub(r'Capacity\s*:\s*[0-9]+', '', raw, count=1, flags=re.I)

    # what's left is the status (clean and truncate)
    status = _clean(raw)
    if status:
        out['openings_status'] = status[:500]
    return out


# ---- "The Basics" block ---------------------------------------------------

# These are the labels shown in the basics block. We normalize each label
# into a column name. New labels will still get captured (see below).
BASICS_LABELS = [
    'Language', 'Schedule', 'Transportation', 'Meals',
    'Special Needs Experience', 'Accreditation', 'Subsidies Accepted',
    'Quality Improvement Efforts', 'QCC Score', 'Ages',
]


def _normalize_label(label):
    return (label.strip().lower()
            .replace(' ', '_')
            .replace('/', '_')
            .replace('-', '_')
            .replace(':', ''))


def empty_basics():
    return {f'basics_{_normalize_label(l)}': None for l in BASICS_LABELS}


def crawl_basics(page):
    """Extract The Basics block. Uses the actual page markup:
        <ul class="provider-attributes__list">
          <li class="provider-attributes__item">
            <span class="provider-attributes__icon ..."></span>  <!-- icon -->
            <span>Language: </span>                               <!-- label -->
            <strong>English, Korean, Spanish</strong>             <!-- value -->
          </li>
          ...
        </ul>
    The QCC item also contains tooltip spans which we filter out.
    """
    out = empty_basics()

    items = page.locator('ul.provider-attributes__list > li.provider-attributes__item')
    n = items.count()
    if n == 0:
        # fallback to a more permissive selector in case markup varies
        items = page.locator('li.provider-attributes__item')
        n = items.count()
    if n == 0:
        return out

    for i in range(n):
        item = items.nth(i)

        # The label is the span that has no class (icons and tooltips have
        # classes). Walk through the spans and pick the unclassed one.
        label = None
        spans = item.locator('span')
        for j in range(spans.count()):
            sp = spans.nth(j)
            cls = sp.get_attribute('class') or ''
            if not cls:
                label = _clean(_all_text(sp))
                if label:
                    break

        # Fallback: derive label from the full item text by stripping the
        # value (whatever's in <strong>) and the tooltip text.
        if not label:
            full = _all_text(item)
            strong_text = ''
            strong = item.locator('strong').first
            if strong.count() > 0:
                strong_text = _all_text(strong)
            label_guess = full.replace(strong_text, '', 1)
            # cut tooltip text (starts with "Quality Counts California" for QCC)
            label_guess = re.split(r'Quality Counts California', label_guess)[0]
            label = _clean(label_guess)

        if not label:
            continue
        # strip trailing colon
        label = label.rstrip(':').strip()

        # The value is the <strong>
        strong = item.locator('strong').first
        if strong.count() == 0:
            continue
        value = _clean(_all_text(strong))
        # Treat the literal placeholder "-" as an empty value (CDSS uses
        # a hyphen to mean "no data"). Keep it as the literal '-' so it's
        # distinguishable from a true None / "not scraped".
        out[f'basics_{_normalize_label(label)}'] = value

    return out


# ---- about + age range ----------------------------------------------------

def empty_about():
    return {'age_range': None, 'about_text': None}


def crawl_about(page):
    out = empty_about()

    # Age range shows up near the top as "Ages X years - Y years" or
    # "Ages X months - Y years"
    body_text = _all_text(page.locator('body'))
    m = re.search(r'Ages\s+([0-9]+\s*(?:months?|years?)\s*-\s*'
                  r'[0-9]+\s*(?:months?|years?))', body_text, re.I)
    if m:
        out['age_range'] = _clean(m.group(1))

    # About section: heading "About {provider_name}"
    about_heading = page.locator('h2:has-text("About"), h3:has-text("About")').first
    if about_heading.count() > 0:
        container = about_heading.locator(
            'xpath=ancestor::*[self::section or self::div][1]'
        )
        if container.count() > 0:
            txt = _all_text(container)
            # strip the heading itself
            txt = re.sub(r'^About[^\n]*\n', '', txt).strip()
            # cut off at next section
            for cutoff in ['Next Steps', 'Disclaimer', 'Parental Rights',
                           'Share', 'Print', 'Favorite']:
                if cutoff in txt:
                    txt = txt.split(cutoff)[0]
            cleaned = _clean(txt)
            if cleaned:
                out['about_text'] = cleaned[:2000]
    return out


# ---- address (often hidden for FCCH; centers usually have it) -------------

def empty_address():
    return {'address': None, 'google_maps_url': None}


def crawl_address(page):
    out = empty_address()
    gmap = page.locator('a[href*="google.com/maps"], a:has-text("Google maps")').first
    if gmap.count() > 0:
        href = gmap.get_attribute('href')
        if href:
            out['google_maps_url'] = _clean(href)
            # extract embedded address from the maps query string if present
            from urllib.parse import unquote_plus
            m = re.search(r'q=([^&]+)', href)
            if m:
                out['address'] = _clean(unquote_plus(m.group(1)))
            elif '/place/' in href:
                # URL of form .../maps/place/<address>/...
                place = href.split('/place/', 1)[1].split('/')[0]
                out['address'] = _clean(unquote_plus(place))
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _text(locator):
    """Safely get inner text from a locator."""
    try:
        return (locator.inner_text(timeout=2000) or '').strip()
    except Exception:
        try:
            return (locator.text_content(timeout=2000) or '').strip()
        except Exception:
            return ''


def _all_text(locator):
    """Get text including hidden/collapsed content (uses textContent rather
    than innerText). Used for tooltips and accordions that may not be in
    the visible accessibility tree."""
    try:
        return (locator.text_content(timeout=2000) or '').strip()
    except Exception:
        return ''


def _clean(s):
    """Normalize whitespace: collapse all runs of whitespace (including
    newlines, tabs) into single spaces, strip ends. Returns None for empty."""
    if s is None:
        return None
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s if s else None


def create_empty_row():
    row = {'provider_url': None}
    for fn in [empty_header, empty_contact, empty_hours, empty_license,
               empty_tags, empty_openings, empty_basics, empty_about,
               empty_address]:
        row.update(fn())
    return row


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    create_log_file()

    # ----- input -----------------------------------------------------------
    OUTPUT_CSV = 'ca_data/ca_records.csv'

    facility_numbers = load_retry_facilities(OUTPUT_CSV)
    if not facility_numbers:
        log(f'No retryable facilities found in {OUTPUT_CSV}.')
        raise SystemExit(0)

    log(f'Loaded {len(facility_numbers)} retryable facilities from '
        f'{OUTPUT_CSV}.')

    # ----- run -------------------------------------------------------------
    # Re-scrapes only facilities whose latest row in OUTPUT_CSV is marked
    # not_found or exception. Providers that still cannot be found are
    # written back as dne; repeated crawler failures remain exception so
    # another run can pick them up again.
    crawled_df, last_index = crawler(
        facility_numbers,
        output_csv=OUTPUT_CSV,
        headless=True,
        # executable_path='chrome-headless-shell-mac-arm64/chrome-headless-shell',
    )
    log(f'Done. Scraped {len(crawled_df)} new rows this run. '
        f'Last index processed: {last_index}.')