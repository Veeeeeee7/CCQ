"""
California Child Care Crawler — mychildcareplan.org

Given a list of facility numbers (CCL license numbers), this script searches
each one on mychildcareplan.org, navigates to the provider details page, and
scrapes all available information.

Adapted from the Georgia DECAL crawler. Same overall structure:
  - one row per seeded licence
  - per-section try/except so a single broken field doesn't kill the row
  - resume-safe: facility numbers already in the output CSV are skipped
  - file-based logging

    python ca_crawler.py
"""

import pandas as pd
import time
import os
import re
import random
import traceback
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

from ca_data_correction import LANGUAGE_COLUMN, sanitize_language


def create_log_file(path='ca_crawler_log.txt'):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as f:
        f.write('')


def log(message, file='ca_crawler_log.txt'):
    with open(file, 'a') as f:
        f.write(message + '\n')
        print(message)


BASE_URL = 'https://mychildcareplan.org'
HOME_URL = BASE_URL + '/'
SEARCH_URL = BASE_URL + '/provider-search/'


def all_columns():
    """Every column the crawler may produce, in order, so appended rows stay
    consistent even when a section fails."""
    cols = ['facility_number', 'provider_url']
    for fn in [empty_header, empty_contact, empty_hours, empty_license,
               empty_tags, empty_openings, empty_basics, empty_about,
               empty_address]:
        cols.extend(fn().keys())
    cols.append('errors')
    return cols


def append_row(row, output_csv):
    """Append one row to the output CSV. Whitespace runs (incl. newlines) are
    collapsed so the CSV stays one row per line."""
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
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    pd.DataFrame([full_row], columns=cols).to_csv(
        output_csv, mode='a', header=not file_exists, index=False
    )


def load_completed(output_csv):
    """Return the set of facility numbers already present in output_csv."""
    if not output_csv or not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv, dtype=str, usecols=['facility_number'])
        return set(df['facility_number'].dropna().astype(str).str.strip().tolist())
    except Exception as e:
        log(f'Could not read existing {output_csv}: {e}')
        return set()


def crawler(facility_numbers, output_csv='ca_data/ca_records.csv',
            headless=True, start_index=0,
            executable_path=None, slow_mo=0, delay_range=(5, 10)):
    """Scrape each facility number; each row is written to output_csv (None
    disables writing) as it is captured. delay_range is the (min, max) random
    sleep between facilities, to avoid a predictable cadence.

    Returns (DataFrame of rows scraped this run, last_index_processed).
    """
    rows = []
    completed = load_completed(output_csv)
    if completed:
        log(f'Resuming: {len(completed)} facility numbers already in '
            f'{output_csv} — those will be skipped.')

    with sync_playwright() as p:
        launch_kwargs = {
            'headless': headless,
            'slow_mo': slow_mo,
            'args': ['--incognito'],
        }
        if executable_path:
            launch_kwargs['executable_path'] = executable_path
        browser = p.chromium.launch(**launch_kwargs)

        ua = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
              'AppleWebKit/537.36 (KHTML, like Gecko) '
              'Chrome/120.0.0.0 Safari/537.36')

        index = start_index
        try:
            for index in range(start_index, len(facility_numbers)):
                facility_number = str(facility_numbers[index]).strip()

                if facility_number in completed:
                    continue

                row = {'facility_number': facility_number}
                errors = []

                # Fresh context per facility: no cookies, cache or storage
                # carry over from the previous one.
                context = browser.new_context(
                    viewport={'width': 1400, 'height': 900},
                    user_agent=ua,
                )
                page = context.new_page()
                try:
                    provider_url = find_provider_url(page, facility_number)
                    if provider_url is None:
                        log(f'[{index}] NOT FOUND: {facility_number}')
                        row.update(create_empty_row())
                        row['provider_url'] = None
                        row['errors'] = 'not_found'
                        rows.append(row)
                        if output_csv:
                            append_row(row, output_csv)
                            completed.add(facility_number)
                        page.close()
                        continue

                    page.goto(provider_url, wait_until='domcontentloaded')
                    try:
                        page.wait_for_load_state('networkidle', timeout=15000)
                    except PWTimeout:
                        pass
                    time.sleep(1)
                    row['provider_url'] = page.url

                    # The search returns its first result, which is not always
                    # the provider that was searched for.
                    if not linkage_verified(page, facility_number):
                        errors.append('linkage_unverified')
                        log(f'  ! {facility_number} does not appear on the '
                            f'landed page -- linkage unverified')

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
                        completed.add(facility_number)

                except Exception:
                    log(f'[{index}] EXCEPTION: {facility_number}')
                    log(traceback.format_exc())
                    row.update(create_empty_row())
                    row['errors'] = 'exception'
                    rows.append(row)
                    if output_csv:
                        append_row(row, output_csv)
                        completed.add(facility_number)
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass
                    try:
                        context.close()
                    except Exception:
                        pass

                if index < len(facility_numbers) - 1:
                    delay = random.uniform(*delay_range)
                    log(f'  ...sleeping {delay:.1f}s before next request')
                    time.sleep(delay)

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


def find_provider_url(page, facility_number):
    """
    Use the site's header search bar to look up a facility number and return
    the resulting provider-details URL. Returns None if not found.
    """
    page.goto(HOME_URL, wait_until='domcontentloaded')
    time.sleep(1)

    # cookie banner, if any — selector is best-effort
    try:
        accept = page.locator('button:has-text("Accept"), '
                              'button:has-text("Agree"), '
                              'button:has-text("OK")').first
        if accept.is_visible(timeout=1500):
            accept.click()
            time.sleep(0.3)
    except Exception:
        pass

    # Several selectors, because the header markup varies across pages.
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
        # fall back to the provider-search query string
        page.goto(SEARCH_URL + f'?search={facility_number}',
                  wait_until='domcontentloaded')
        time.sleep(2)
    else:
        search_input.click()
        search_input.fill(str(facility_number))
        time.sleep(0.3)
        search_input.press('Enter')
        time.sleep(2)
        try:
            page.wait_for_load_state('networkidle', timeout=10000)
        except PWTimeout:
            pass

    # Some queries land directly on the details page, others on a results list.
    if '/provider-details' in page.url:
        return page.url

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

    try:
        card = page.locator('a:has-text("View Details")').first
        if card.count() > 0:
            with page.expect_navigation(timeout=10000):
                card.click()
            if '/provider-details' in page.url:
                return page.url
    except Exception:
        pass

    return None


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

    h1 = page.locator('h1').first
    if h1.count() > 0:
        out['provider_name'] = _clean(_text(h1))

    # The type appears both as a short badge ("Small Family Child Care Home")
    # and inside a long tooltip paragraph, so match canonical type strings
    # against the body text instead of reading one element.
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
    m = re.search(rf'\b({type_pattern})\b', body_text)
    if m:
        # longest first, so "Center" does not win over "Child Care Center"
        for t in sorted(canonical_types, key=len, reverse=True):
            if t in body_text:
                out['provider_type'] = t
                break

    if re.search(r'\bNot Licensed\b', body_text):
        out['licensed_status'] = 'Not Licensed'
    elif re.search(r'\bLicensed\b', body_text):
        out['licensed_status'] = 'Licensed'

    if re.search(r'\bUnclaimed\b', body_text):
        out['claimed_status'] = 'Unclaimed'
    elif re.search(r'\bClaimed\b', body_text):
        out['claimed_status'] = 'Claimed'

    img = page.locator('img[src*="partners.mychildcareplan.org/docs/ProviderPhotos"]').first
    if img.count() > 0:
        out['profile_photo_url'] = img.get_attribute('src')

    return out


def empty_contact():
    return {'phone': None, 'website_url': None}


def crawl_contact(page):
    out = empty_contact()

    # Every page carries the site's own hotline tel: link (1-800-KIDS-793);
    # skip it to reach the provider's phone.
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

    # first external link that is not a social/internal/known site host
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


def empty_hours():
    return {'business_hours': None}


def crawl_hours(page):
    out = empty_hours()
    # The Business Hours panel is collapsed by default, so read textContent
    # (_all_text) rather than innerText.
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


# The site prints every licence a profile holds after a single "License Number"
# label, comma separated ("License Number: 304270387, 304270386"). Continuation
# items must start with a digit so the run stops at ordinary prose after a comma.
LICENSE_NUMBER_RE = re.compile(
    r'License\s*Number\s*[:\s]\s*([A-Z0-9-]+(?:\s*,\s*[0-9][A-Z0-9-]*)*)', re.I)
_LICENCE_SPLIT_RE = re.compile(r'\s*,\s*')


def licences_on_page(body_text):
    """Every licence number printed on the page, in order, deduplicated."""
    seen, found = set(), []
    for m in LICENSE_NUMBER_RE.finditer(body_text or ''):
        for part in _LICENCE_SPLIT_RE.split(m.group(1)):
            value = _clean(part)
            if value and value not in seen:
                seen.add(value)
                found.append(value)
    return found


def _norm_licence(value):
    """Normalize a licence number for comparison; leading zeros and
    punctuation are cosmetic on the site."""
    return re.sub(r'[^0-9A-Za-z]', '', str(value or '').strip()).lstrip('0').upper()


def linkage_verified(page, facility_number):
    """True if the searched facility number is one of the licences on the page.

    A profile can hold several licences, so the test is membership in the full
    list rather than equality with the first one. A page that prints no
    licence cannot be checked and passes: unverifiable is not wrong.
    """
    wanted = _norm_licence(facility_number)
    if not wanted:
        return True
    body_text = _all_text(page.locator('body'))
    found = {_norm_licence(v) for v in licences_on_page(body_text)}
    found.discard('')
    return not found or wanted in found


def empty_license():
    return {'license_number': None, 'licensing_reports_url': None}


def crawl_license(page):
    out = empty_license()
    body_text = _all_text(page.locator('body'))
    # license_number keeps only the first licence the page prints.
    licences = licences_on_page(body_text)
    if licences:
        out['license_number'] = licences[0]

    reports = page.locator('a[href*="ccld.dss.ca.gov/carefacilitysearch"]').first
    if reports.count() > 0:
        out['licensing_reports_url'] = _clean(reports.get_attribute('href'))
    return out


def empty_tags():
    return {'tags_count': None, 'tags': None}


def crawl_tags(page):
    """Read the tag list.

    Names come from the <li class="program-tags__tag"> items, which exist for
    every tag; the "Learn More" links are a fallback, as not every tag has one.
    """
    out = empty_tags()
    heading = page.locator('text=/View Tags\\s*\\(\\d+\\)/').first
    if heading.count() > 0:
        m = re.search(r'\((\d+)\)', _all_text(heading))
        if m:
            out['tags_count'] = int(m.group(1))

    seen, ordered = set(), []

    def add(name):
        name = _clean(name)
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)

    items = page.locator('li.program-tags__tag')
    for i in range(items.count()):
        add(_all_text(items.nth(i)))

    if not ordered:
        tag_links = page.locator('a[href*="resources/?search="]')
        from urllib.parse import unquote
        for i in range(tag_links.count()):
            href = tag_links.nth(i).get_attribute('href') or ''
            m = re.search(r'search=([^&]+)', href)
            if m:
                add(unquote(m.group(1).replace('+', ' ')))

    if ordered:
        out['tags'] = '|'.join(ordered)
        if out['tags_count'] is None:
            out['tags_count'] = len(ordered)
    return out


def empty_openings():
    return {
        'openings_last_updated': None,
        'openings_status': None,
        'openings_capacity': None,
    }


# The Openings heading comes in two forms. mychildcareplan.org renders
#     Openings (last updated MM/DD/YYYY)
# only while the provider's confirmation is recent and a bare
#     Openings
# otherwise -- the same block underneath, with the same "Capacity: N" lines.
OPENINGS_HEADING_RE = r'^\s*Openings\s*(\(last updated[^)]*\))?\s*$'
OPENINGS_PREFIX_RE = re.compile(r'^\s*Openings\s*(?:\(last updated[^)]*\))?\s*',
                                re.I)
# 'Contact ' is deliberately NOT a stop marker: an age group's availability
# line can be "Contact provider for details", followed by its "Capacity: N".
OPENINGS_STOP_MARKERS = ('The Basics', 'About ', 'Next Steps', 'Disclaimer',
                         'Tags')
CAPACITY_RE = re.compile(r'Capacity\s*:\s*([0-9]+)', re.I)
CAPACITY_STRIP_RE = re.compile(r'\s*Capacity\s*:\s*[0-9]+', re.I)


def crawl_openings(page):
    """Parse the Openings section.

        Openings [(last updated MM/DD/YYYY)]
        <group name>   <status>   Capacity: <N>     (repeated per age group)

    openings_capacity is the SUM across age groups; the "Capacity:" tokens are
    stripped from openings_status. The body is read from
    <div class="provider-openings">, a sibling of the heading, with a
    heading-ancestor text crop as the fallback.
    """
    out = empty_openings()

    heading = page.locator(f'text=/{OPENINGS_HEADING_RE}/i').first
    has_heading = heading.count() > 0
    if has_heading:
        m = re.search(r'last updated\s+([0-9/.-]+)', _all_text(heading), re.I)
        if m:
            out['openings_last_updated'] = _clean(m.group(1).rstrip(') '))

    block = page.locator('div.provider-openings').first
    if block.count() > 0:
        raw = _clean(_all_text(block)) or ''
    elif has_heading:
        container = heading.locator(
            'xpath=ancestor::*[self::section or self::div or self::article][1]'
        )
        if container.count() == 0:
            return out
        raw = _clean(_all_text(container)) or ''
        head = _clean(_all_text(heading)) or ''
        idx = raw.find(head) if head else -1
        if idx > 0:
            raw = raw[idx:]
        for stop_marker in OPENINGS_STOP_MARKERS:
            idx = raw.find(stop_marker)
            if idx > 0:
                raw = raw[:idx]
        raw = OPENINGS_PREFIX_RE.sub('', raw, count=1)
    else:
        return out

    caps = CAPACITY_RE.findall(raw)
    if caps:
        out['openings_capacity'] = str(sum(int(n) for n in caps))
        raw = CAPACITY_STRIP_RE.sub('', raw)

    status = _clean(raw)
    if status:
        out['openings_status'] = status[:500]
    return out


# Labels in The Basics block; each becomes basics_<label>. Unlisted labels are
# still captured.
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
        items = page.locator('li.provider-attributes__item')
        n = items.count()
    if n == 0:
        return out

    for i in range(n):
        item = items.nth(i)

        # The label is the span with no class (icons and tooltips have one).
        label = None
        spans = item.locator('span')
        for j in range(spans.count()):
            sp = spans.nth(j)
            cls = sp.get_attribute('class') or ''
            if not cls:
                label = _clean(_all_text(sp))
                if label:
                    break

        # Fallback: item text minus the <strong> value and the tooltip.
        if not label:
            full = _all_text(item)
            strong_text = ''
            strong = item.locator('strong').first
            if strong.count() > 0:
                strong_text = _all_text(strong)
            label_guess = full.replace(strong_text, '', 1)
            label_guess = re.split(r'Quality Counts California', label_guess)[0]
            label = _clean(label_guess)

        if not label:
            continue
        label = label.rstrip(':').strip()

        strong = item.locator('strong').first
        if strong.count() == 0:
            continue
        value = _clean(_all_text(strong))
        key = f'basics_{_normalize_label(label)}'
        if key == LANGUAGE_COLUMN:
            # Some pages serve option codes ('00, 01') or leak a component
            # label ('..., 2.Labels- English'); neither is decodable from the
            # page, so those tokens are dropped. See sanitize_language().
            value = sanitize_language(value)
        out[key] = value

    return out


def empty_about():
    return {'age_range': None, 'about_text': None}


def crawl_about(page):
    out = empty_about()

    body_text = _all_text(page.locator('body'))
    m = re.search(r'Ages\s+([0-9]+\s*(?:months?|years?)\s*-\s*'
                  r'[0-9]+\s*(?:months?|years?))', body_text, re.I)
    if m:
        out['age_range'] = _clean(m.group(1))

    about_heading = page.locator('h2:has-text("About"), h3:has-text("About")').first
    if about_heading.count() > 0:
        container = about_heading.locator(
            'xpath=ancestor::*[self::section or self::div][1]'
        )
        if container.count() > 0:
            txt = _all_text(container)
            txt = re.sub(r'^About[^\n]*\n', '', txt).strip()
            for cutoff in ['Next Steps', 'Disclaimer', 'Parental Rights',
                           'Share', 'Print', 'Favorite']:
                if cutoff in txt:
                    txt = txt.split(cutoff)[0]
            cleaned = _clean(txt)
            if cleaned:
                out['about_text'] = cleaned[:2000]
    return out


# The address is often hidden for family child care homes.
def empty_address():
    return {'address': None, 'google_maps_url': None}


def crawl_address(page):
    out = empty_address()
    gmap = page.locator('a[href*="google.com/maps"], a:has-text("Google maps")').first
    if gmap.count() > 0:
        href = gmap.get_attribute('href')
        if href:
            out['google_maps_url'] = _clean(href)
            from urllib.parse import unquote_plus
            m = re.search(r'q=([^&]+)', href)
            if m:
                out['address'] = _clean(unquote_plus(m.group(1)))
            elif '/place/' in href:
                # URL of form .../maps/place/<address>/...
                place = href.split('/place/', 1)[1].split('/')[0]
                out['address'] = _clean(unquote_plus(place))
    return out


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
    """textContent rather than innerText, so hidden tooltips and collapsed
    accordions are included."""
    try:
        return (locator.text_content(timeout=2000) or '').strip()
    except Exception:
        return ''


def _clean(s):
    """Collapse whitespace runs to single spaces and strip; None if empty."""
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


if __name__ == '__main__':
    create_log_file()

    INPUT_CSV = 'ca_data/ca_seed.csv'
    FACILITY_COL = 'facility_number'
    OUTPUT_CSV = 'ca_data/ca_records.csv'

    df_in = pd.read_csv(INPUT_CSV, dtype=str)
    facility_numbers = df_in[FACILITY_COL].dropna().astype(str).tolist()

    # Re-running resumes. To retry failures, delete their rows
    # (errors='not_found' / 'exception') from OUTPUT_CSV first.
    crawled_df, last_index = crawler(
        facility_numbers,
        output_csv=OUTPUT_CSV,
        headless=True,
    )
    log(f'Done. Scraped {len(crawled_df)} new rows this run. '
        f'Last index processed: {last_index}.')