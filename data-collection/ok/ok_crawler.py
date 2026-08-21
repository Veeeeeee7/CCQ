"""
ok_crawler.py — Oklahoma Child Care Locator provider-detail crawler.

Reads the provided ok_data/ok_seed.csv and, for each unique provider_id
(the state license number, e.g. "K830023010"), fetches
https://ccl.dhs.ok.gov/providers/{provider_id} and parses the full
per-provider record: Star Level (qr_rating), hours, ages accepted, capacity,
licensing specialist, full licensing/monitoring-visit history, substantiated
complaints, and contact info -- one row per provider.

WHY THERE IS NO ok_capture.py: the Locator (ccl.dhs.ok.gov) is a Next.js app
that server-renders its search AND detail pages -- a plain `requests.get()`
already returns the fully rendered HTML with real content (confirmed in
Phase 0 by fetching a live provider page directly and getting back the Star
Level, hours, monitoring history, contact info, etc., with no JavaScript
execution). Per the project's Phase 1 guidance, a static/SSR site like this
doesn't need a Playwright-based capture script -- "a quick requests.get +
BeautifulSoup inspection is enough." That inspection happened directly
against the live site during Phase 0 rather than via a human-run capture
script.

CAVEAT WORTH FLAGGING: that Phase 0 inspection was done through a tool that
converts HTML to readable text/markdown, not through raw HTML -- so the
selectors below are TEXT-ANCHORED (they search for the exact visible label
strings observed, e.g. "Total Capacity", "Visit Date", "Star Level Program")
rather than CSS-class-anchored. This is actually fairly robust (it doesn't
care about div/class structure, only that the visible labels stay put), but
it has never been checked against the real DOM. Expect the Checkpoint 2
smoke-test sample to surface a few fields that need a regex tweak, especially
in crawl_monitoring() (the most structurally complex section) and
crawl_contact() (the most heuristic one).

FLOW: one request per unique provider (no separate search step needed here --
that is the seed's job). Sections are parsed independently under their own
try/except, exactly like the other reference crawlers, so one bad field never
kills a row -- failures land in the `errors` column and the section's
empty_*() fills in None so the output stays rectangular.

Usage:
    python ok_crawler.py --limit 20                          # smoke test
    python ok_crawler.py                                     # full run, resumable
    python ok_crawler.py --delay-min 0.5 --delay-max 1.5      # faster (plain HTTP, no browser)

Deps: pip install requests pandas beautifulsoup4
"""

import argparse
import json
import os
import random
import re
import time
import traceback

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

BASE_URL = 'https://ccl.dhs.ok.gov'
DETAIL_PATH_TMPL = '/providers/{pid}'

SEED_PATH = 'ok_data/ok_seed.csv'
OUT_PATH = 'ok_data/ok_records.csv'
LOG_FILE = 'ok_crawler_log.txt'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
            'Saturday', 'Sunday']

# Multi-value fields (program_tags, ages_accepted) are stored raw-joined with
# this delimiter -- chosen instead of a comma/semicolon because individual
# items (e.g. age-group labels like "Toddlers (12-23 months; 1yr.)") already
# contain commas and semicolons. The cleaning phase's build_multivalue_columns
# splits on this later.
MULTI_DELIM = ' | '

_TYPE_ID_RE = re.compile(
    r'\b((?:LARGE )?FAMILY CHILD CARE HOME|CHILD CARE CENTER)\s*-\s*(\S+)', re.I)
_STAR_RE = re.compile(r'(\d)\s*Star Level Program', re.I)
_PHONE_RE = re.compile(r'\(\d{3}\)\s*\d{3}-\d{4}')
_EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
_SUBSIDY_CONTRACT_RE = re.compile(r'Subsidy Contract Number:\s*(\S+)', re.I)
_VISIT_BLOCK_RE = re.compile(
    r'Visit Date\s*(?P<date>\d{4}-\d{2}-\d{2})\s*'
    r'Visit Type\s*(?P<type>.*?)\s*'
    r'Purpose of Visit\s*(?P<purpose>.*?)\s*'
    r'(?P<pass>\d+)\s*of\s*(?P<total>\d+)\s*'
    r'Areas in compliance\s*'
    r'Non Compliances Observed:\s*(?P<noncompliance>.*?)\s*'
    r'(?=Visit Date\s*\d{4}-\d{2}-\d{2}|View Full Report|\Z)', re.S)
_REPORT_HREF_RE = re.compile(r'href="([^"]*?/licensing-history/[^"]+)"')


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def create_log_file(path=LOG_FILE):
    with open(path, 'w') as f:
        f.write('')


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

def fetch_detail(pid, session, timeout=30):
    """GET the provider's detail page. Returns raw HTML, or None on failure
    (404 included -- a delisted/bad ID must never kill the run)."""
    url = BASE_URL + DETAIL_PATH_TMPL.format(pid=pid)
    try:
        r = session.get(url, headers={'User-Agent': UA}, timeout=timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.text
    except Exception as e:
        log(f'  ! request failed for {pid}: {e}')
        return None


# ---------------------------------------------------------------------------
# generic text-anchored parsing helpers
# ---------------------------------------------------------------------------
# These search for the exact visible label text observed in Phase 0 rather
# than CSS classes/ids (see module docstring for why). soup.get_text('\n')
# turns block-level elements into line breaks reasonably reliably, so a
# "label line, then value line" DOM pattern (a <dt>/<dd> or two sibling
# <div>s) should survive this even though we've never seen the raw markup.

def _clean(s):
    if s is None:
        return None
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s or None


def _lines(soup):
    return [_clean(l) for l in soup.get_text('\n').split('\n') if _clean(l)]


def _value_after(lines, label, n=1, exact=True):
    """The line(s) immediately following a match of `label`. Returns a single
    string if n==1 else a list. None/[] if not found or it runs off the end
    of the page."""
    for i, l in enumerate(lines):
        hit = (l == label) if exact else (label.lower() in l.lower())
        if hit:
            vals = lines[i + 1:i + 1 + n]
            if not vals:
                return None if n == 1 else []
            return vals[0] if n == 1 else vals
    return None if n == 1 else []


def _list_between(lines, start_idx, end_label):
    """Lines strictly between position start_idx (exclusive) and the next
    line equal to end_label (exclusive). [] if end_label never appears."""
    out = []
    for l in lines[start_idx + 1:]:
        if l == end_label:
            return out
        out.append(l)
    return []


def _index_matching(lines, pattern):
    """Index of the first line matching `pattern` (compiled regex), or None."""
    for i, l in enumerate(lines):
        if pattern.search(l):
            return i
    return None


def _section_span(full_text, start_pat, end_pats):
    """(start, end) char offsets of the section in `full_text` -- which must
    be NEWLINE-preserving (soup.get_text('\\n')), not space-joined, so line
    structure survives for _section_lines() below. None if start_pat doesn't
    match."""
    m = start_pat.search(full_text)
    if not m:
        return None
    start = m.end()
    end = len(full_text)
    for ep in end_pats:
        m2 = ep.search(full_text, start)
        if m2:
            end = min(end, m2.start())
    return start, end


def _section_text(full_text, start_pat, end_pats, max_len=4000):
    """Single-line, whitespace-normalized text of the section -- for the
    *_section_text fallback columns, same spirit as the reference states':
    if a structured-parse assumption elsewhere is wrong, the raw text
    survives so nothing is silently lost."""
    span = _section_span(full_text, start_pat, end_pats)
    if not span:
        return None
    return _clean(full_text[span[0]:span[1]])[:max_len] or None


def _section_lines(full_text, start_pat, end_pats):
    """The section as a list of cleaned, non-empty lines -- for structured
    sub-parsing that needs line boundaries (e.g. crawl_contact's name/title/
    address heuristic). Unlike _section_text, this does NOT collapse
    newlines, since that would destroy exactly the structure being parsed."""
    span = _section_span(full_text, start_pat, end_pats)
    if not span:
        return []
    raw = full_text[span[0]:span[1]]
    return [_clean(l) for l in raw.split('\n') if _clean(l)]


# ---------------------------------------------------------------------------
# section extractors -- empty_*/crawl_* pairs, rectangular schema
# ---------------------------------------------------------------------------

def empty_basic():
    return {'provider_name': None, 'facility_type': None,
            'program_tags': None, 'id_on_page': None}


def crawl_basic(soup, text, lines, pid):
    """NOTE (fixed after the first real smoke test): this used to take
    soup.find('h1') as the provider name, on the assumption the name was
    the page's first/only h1 -- wrong. Every page (search results, FAQ,
    detail) carries a persistent site-header h1 reading exactly "Child Care
    Locator" (a logo/branding element), and BeautifulSoup's find('h1') was
    grabbing THAT, not the provider's name -- confirmed by the smoke test
    returning "Child Care Locator" for 20/20 rows, with the real names
    (e.g. "POOL, ALLISON") turning up misplaced inside program_tags instead.
    Fixed by anchoring on the "<TYPE> - <ID>" label line (reliably matched
    via _TYPE_ID_RE) and taking the line immediately after it as the name --
    confirmed against the smoke-test data that this label-then-name adjacency
    holds (that's exactly how the real names ended up inside the old,
    wrongly-anchored program_tags slice)."""
    out = empty_basic()
    type_idx = None
    for i, l in enumerate(lines):
        m = _TYPE_ID_RE.match(l)
        if m:
            out['facility_type'] = m.group(1).strip().upper()
            out['id_on_page'] = m.group(2).strip()
            type_idx = i
            break
    if type_idx is None:
        # fallback: infer from the license-number prefix pattern observed in
        # Phase 0 (K83.. = center, K82.. = family child care home) -- best
        # effort only, the on-page label above is the trusted source.
        if re.match(r'^K83', pid):
            out['facility_type'] = 'CHILD CARE CENTER'
        elif re.match(r'^K82', pid):
            out['facility_type'] = 'FAMILY CHILD CARE HOME'

    name_idx = None
    if type_idx is not None and type_idx + 1 < len(lines):
        out['provider_name'] = lines[type_idx + 1]
        name_idx = type_idx + 1

    # program_tags: the bullet list between the name and the "N Star Level
    # Program" heading, e.g. "Accepts Subsidy", "Year Round", "Daytime
    # Hours". Left as raw joined text -- the cleaning phase decomposes
    # multi-value fields into booleans, not the crawler.
    star_idx = _index_matching(lines, _STAR_RE)
    if name_idx is not None and star_idx is not None and star_idx > name_idx:
        tags = [l for l in lines[name_idx + 1:star_idx] if l and l != 'Print']
        out['program_tags'] = MULTI_DELIM.join(tags) if tags else None
    return out


def empty_rating():
    return {'qr_rating_raw': None}


def crawl_rating(text):
    out = empty_rating()
    m = _STAR_RE.search(text)
    if m:
        out['qr_rating_raw'] = m.group(1)
    return out


def empty_hours():
    return {f'hours_{d.lower()}': None for d in WEEKDAYS}


def crawl_hours(lines):
    out = empty_hours()
    for day in WEEKDAYS:
        out[f'hours_{day.lower()}'] = _value_after(lines, day, exact=True)
    return out


def empty_ages_capacity():
    return {'ages_accepted': None, 'total_capacity': None}


def crawl_ages_capacity(lines):
    out = empty_ages_capacity()
    try:
        i = lines.index('Ages Accepted')
        ages = _list_between(lines, i, 'Total Capacity')
        out['ages_accepted'] = MULTI_DELIM.join(ages) if ages else None
    except ValueError:
        pass
    cap = _value_after(lines, 'Total Capacity')
    if cap and re.match(r'^\d+$', cap):
        out['total_capacity'] = cap
    return out


def empty_licensing():
    return {'licensing_specialist_name': None, 'licensing_specialist_phone': None}


def crawl_licensing(lines):
    """Name + phone of the assigned licensing specialist. NOTE: Phase 0
    observed a static boilerplate sentence ("Licensing specialists monitor
    child care programs...") sitting between the heading and the actual
    name/phone, so this can't just take "the next N lines" -- it anchors on
    the phone number (a reliable, distinctively-formatted match) and takes
    the short non-boilerplate line immediately before it as the name."""
    out = empty_licensing()
    i = None
    for idx, l in enumerate(lines):
        if 'licensing specialist' in l.lower():
            i = idx
            break
    if i is None:
        return out
    window = lines[i + 1:i + 6]
    phone_idx = None
    for j, l in enumerate(window):
        if _PHONE_RE.match(l or ''):
            out['licensing_specialist_phone'] = l
            phone_idx = j
            break
    if phone_idx is not None and phone_idx > 0:
        candidate = window[phone_idx - 1]
        if candidate and len(candidate) < 40 and not candidate.endswith('.'):
            out['licensing_specialist_name'] = candidate
    return out


def empty_monitoring():
    return {'monitoring_visits_json': None, 'n_monitoring_visits': None,
            'monitoring_section_text': None}


def crawl_monitoring(text, html):
    out = empty_monitoring()
    section = _section_text(
        text, re.compile(r'Licensing History'),
        [re.compile(r'Substantiated Complaint Summary'),
         re.compile(r'Reach out to this provider')],
        max_len=8000)
    out['monitoring_section_text'] = section
    if not section:
        return out

    report_urls = _REPORT_HREF_RE.findall(html)
    visits = []
    for i, m in enumerate(_VISIT_BLOCK_RE.finditer(section)):
        noncompliance = _clean(m.group('noncompliance'))
        if noncompliance and noncompliance.lower() == 'none':
            noncompliance = None
        visits.append({
            'visit_date': m.group('date'),
            'visit_type': _clean(m.group('type')),
            'purpose': _clean(m.group('purpose')),
            'areas_compliant': m.group('pass'),
            'areas_total': m.group('total'),
            'non_compliances': noncompliance,
            'report_url': report_urls[i] if i < len(report_urls) else None,
        })
    if visits:
        out['monitoring_visits_json'] = json.dumps(visits, ensure_ascii=False)
        out['n_monitoring_visits'] = str(len(visits))
    return out


def empty_complaints():
    return {'has_substantiated_complaints': None, 'complaints_since_date': None,
            'complaint_findings_json': None, 'n_complaint_findings': None,
            'complaints_section_text': None}


# A provider WITH complaints (found in the first real smoke test -- Phase 0
# only saw the zero-complaints placeholder) shows one or more findings. Each
# is introduced by one of THREE variant phrases (confirmed against real data
# in the second smoke test: a bare "Complaint:" turned up as a third variant
# alongside the two seen initially), formatted:
#   "<intro> <category-and-description> Plan to Correct <plan text>
#    [(Documented on NTC)] Regulation Description <code> - <reg title>.
#    <reg description>."
# "<category-and-description>" is itself INCONSISTENTLY formatted in the
# source -- sometimes "Category: Description" (colon), sometimes
# "Category-Description" (hyphen, no colon at all). An earlier version tried
# to split category/description with one regex that searched ahead for the
# first colon, which for the hyphen-only findings ran straight through into
# the NEXT finding's regulation code (e.g. "340:110-...") and mangled both.
# Fixed by splitting on the intro phrases FIRST (so each finding is reliably
# bounded no matter its internal formatting), then doing a bounded
# best-effort category/description split (only within a short prefix) on
# each already-isolated chunk -- verified against the real K820010716 record,
# which has both formatting styles.
#
# ROUND 3 (2026-07-08, after the first full ~2,557-provider crawl): the
# "(Documented on NTC)" phrase, required literally by the body regex, turned
# out to be present in only 1 of 432 real substantiated-complaint records --
# it was an artifact of the single record used to build/verify the earlier
# fix, not a stable part of the format. Made it optional. Verified offline
# against every real complaints_section_text already captured in
# ok_data/ok_records.csv (no live re-crawl needed, same as the round-2 fix):
# findings-bearing rows went from 1/432 to 427/432 (1,198 findings total).
# The remaining 5/432 have no findings text at all after "Allegation
# Findings" in the source (genuinely nothing published, not a parsing miss).
# 8/432 lose exactly their LAST finding because complaints_section_text hit
# the max_len=8000 cap mid-sentence on providers with many findings -- a
# real re-crawl with a larger cap would recover these; low priority since
# it's a bonus field and costs at most one finding on 8 rows.
_FINDING_INTRO_RE = re.compile(
    r'Substantiated Complaint:|'
    r'Complaint: Additional Non-Compliance Found During Investigation:|'
    r'Complaint:')
_FINDING_BODY_RE = re.compile(
    r'(?P<body>.+?)\s*Plan to Correct\s*(?P<plan>.*?)\s*(?:\(Documented on NTC\)\s*)?'
    r'Regulation Description\s*(?P<reg_code>\S+)\s*-\s*(?P<reg_rest>.+)', re.S)


def _parse_findings(section):
    chunks = _FINDING_INTRO_RE.split(section)[1:]   # [0] is pre-intro boilerplate
    findings = []
    for chunk in chunks:
        m = _FINDING_BODY_RE.match(chunk)
        if not m:
            continue   # chunk didn't match the expected shape -- skip, don't guess
        finding_text = _clean(m.group('body'))
        reg_rest = _clean(m.group('reg_rest')) or ''
        reg_title, _, reg_desc = reg_rest.partition('. ')

        category, description = None, finding_text
        if finding_text:
            # only trust a colon as the category/description separator if
            # it's near the start -- a colon found deep in the text is more
            # likely to belong to a regulation code that leaked in, not a
            # real separator (see docstring above).
            head = finding_text[:80]
            if ':' in head:
                idx = finding_text.index(':')
                category = _clean(finding_text[:idx])
                description = _clean(finding_text[idx + 1:])

        findings.append({
            'category': category,
            'description': description,
            'plan_to_correct': _clean(m.group('plan')),
            'regulation_code': m.group('reg_code'),
            'regulation_title': _clean(reg_title) or None,
            'regulation_description': _clean(reg_desc) or None,
        })
    return findings


def crawl_complaints(text):
    out = empty_complaints()
    # max_len bumped from an earlier 2000 -- the first real smoke test showed
    # a genuine multi-finding complaint getting cut off mid-record at 2000
    # chars, silently dropping real violation detail. 8000 matched the
    # monitoring section's budget at the time, but the first full crawl
    # showed 8/2,557 providers with enough findings to still hit that cap
    # (their last finding gets dropped by _parse_findings rather than
    # corrupted -- see the round-3 note above). Bumped to 16000 so a re-crawl
    # would clear essentially all of them; not worth blocking Phase 3 on a
    # re-crawl of 8 rows for one dropped finding each.
    section = _section_text(
        text, re.compile(r'Substantiated Complaint Summary'),
        [re.compile(r'Reach out to this provider')], max_len=16000)
    out['complaints_section_text'] = section
    if not section:
        return out
    m = re.search(r'Since\s+([\d_/-]+)', section)
    if not (m and '_' not in m.group(1)):
        # "Since ____-__-__" (template placeholder) -- zero substantiated
        # complaints for this provider.
        out['has_substantiated_complaints'] = 'False'
        return out

    out['has_substantiated_complaints'] = 'True'
    out['complaints_since_date'] = m.group(1)

    findings = _parse_findings(section)
    if findings:
        out['complaint_findings_json'] = json.dumps(findings, ensure_ascii=False)
        out['n_complaint_findings'] = str(len(findings))
    return out


def empty_contact():
    return {'contact_name': None, 'contact_title': None, 'contact_phone': None,
            'contact_email': None, 'address': None, 'subsidy_contract_number': None,
            'contact_section_text': None}


def crawl_contact(text):
    out = empty_contact()
    start_pat = re.compile(r'Reach out to this provider')
    end_pats = [re.compile(r'\bLocation\b')]
    # display column: single clean line, same spirit as the other
    # *_section_text fallbacks
    out['contact_section_text'] = _section_text(text, start_pat, end_pats, max_len=1500)
    # structural parse: needs real line boundaries, which _section_text's
    # whitespace-collapsing destroys -- see _section_lines' docstring
    seg_lines = _section_lines(text, start_pat, end_pats)
    if not seg_lines:
        return out

    m = _PHONE_RE.search(out['contact_section_text'] or '')
    if m:
        out['contact_phone'] = m.group(0)
    m = _EMAIL_RE.search(out['contact_section_text'] or '')
    if m:
        out['contact_email'] = m.group(0)
    m = _SUBSIDY_CONTRACT_RE.search(out['contact_section_text'] or '')
    if m:
        out['subsidy_contract_number'] = m.group(1)

    # Heuristic (unverified against real markup, see module docstring): first
    # line is the contact's name; the second line is a title if it's short
    # and digit-free (e.g. "Director"); everything else that isn't the
    # phone/email/subsidy-number line is the address.
    if seg_lines:
        out['contact_name'] = seg_lines[0]
    if len(seg_lines) > 1 and len(seg_lines[1]) < 40 and not re.search(r'\d', seg_lines[1]):
        out['contact_title'] = seg_lines[1]
        addr_parts = seg_lines[2:]
    else:
        addr_parts = seg_lines[1:]
    addr_parts = [p for p in addr_parts if p != out['contact_name']
                  and not _PHONE_RE.search(p) and not _EMAIL_RE.search(p)
                  and 'Subsidy Contract Number' not in p]
    out['address'] = ', '.join(addr_parts) if addr_parts else None
    return out


# ---------------------------------------------------------------------------
# id-mismatch check (mirrors NC's _modern_id_mismatch)
# ---------------------------------------------------------------------------

def _id_mismatch(basic, pid):
    got = basic.get('id_on_page')
    if not got:
        return False
    return got.strip().upper() != str(pid).strip().upper()


# ---------------------------------------------------------------------------
# seed / resume helpers
# ---------------------------------------------------------------------------

def load_seed(seed_csv):
    if not seed_csv or not os.path.exists(seed_csv):
        raise FileNotFoundError(
            f'Seed CSV not found: {seed_csv}.')
    df = pd.read_csv(seed_csv, dtype=str)
    if 'provider_id' not in df.columns:
        raise ValueError(f'Seed needs a provider_id column; got {list(df.columns)}')
    df['provider_id'] = df['provider_id'].astype(str).str.strip()
    df = (df[df['provider_id'].notna() & (df['provider_id'] != '')]
          .drop_duplicates(subset='provider_id').reset_index(drop=True))
    return df


def all_columns():
    cols = ['provider_id', 'source_url']
    for fn in (empty_basic, empty_rating, empty_hours, empty_ages_capacity,
               empty_licensing, empty_monitoring, empty_complaints, empty_contact):
        cols.extend(fn().keys())
    cols.append('errors')
    return cols


def append_row(row, output_csv):
    cols = all_columns()
    full = {}
    for c in cols:
        v = row.get(c)
        if isinstance(v, str):
            v = re.sub(r'[\r\n]+', ' ', v).strip() or None
        full[c] = v
    parent = os.path.dirname(output_csv)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    pd.DataFrame([full], columns=cols).to_csv(
        output_csv, mode='a', header=not exists, index=False)


def load_completed(output_csv):
    if not output_csv or not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv, dtype=str, usecols=['provider_id'])
        return set(df['provider_id'].dropna().str.strip())
    except Exception as e:
        log(f'Could not read existing {output_csv}: {e}')
        return set()


# ---------------------------------------------------------------------------
# main crawler
# ---------------------------------------------------------------------------

def crawl_one(pid, session):
    html = fetch_detail(pid, session)
    row = {'provider_id': pid,
           'source_url': BASE_URL + DETAIL_PATH_TMPL.format(pid=pid)}
    errors = []

    if html is None:
        row.update(empty_basic() | empty_rating() | empty_hours()
                   | empty_ages_capacity() | empty_licensing()
                   | empty_monitoring() | empty_complaints() | empty_contact())
        row['errors'] = 'not_found'
        return row, None

    soup = BeautifulSoup(html, 'html.parser')
    # Newline-preserving on purpose: every regex below matches across
    # whitespace with \s* / \s+ (newline-agnostic), but _section_lines()
    # needs real line boundaries for its structural parsing, and a
    # space-joined text would destroy those (see _section_text's docstring).
    text = soup.get_text('\n')
    lines = _lines(soup)

    basic = {}
    for name, fn, args, empty_fn in (
        ('basic', crawl_basic, (soup, text, lines, pid), empty_basic),
        ('rating', crawl_rating, (text,), empty_rating),
        ('hours', crawl_hours, (lines,), empty_hours),
        ('ages_capacity', crawl_ages_capacity, (lines,), empty_ages_capacity),
        ('licensing', crawl_licensing, (lines,), empty_licensing),
        ('monitoring', crawl_monitoring, (text, html), empty_monitoring),
        ('complaints', crawl_complaints, (text,), empty_complaints),
        ('contact', crawl_contact, (text,), empty_contact),
    ):
        try:
            result = fn(*args)
            row.update(result)
            if name == 'basic':
                basic = result
        except Exception:
            errors.append(name)
            row.update(empty_fn())
            log(f'  ! {name} failed for {pid}: '
                f'{traceback.format_exc().splitlines()[-1]}')

    if _id_mismatch(basic, pid):
        errors.append('id_mismatch')
    row['errors'] = ','.join(errors)
    return row, row.get('qr_rating_raw')


def crawler(provider_ids, output_csv=OUT_PATH, start_index=0, limit=None,
            delay_range=(1, 3)):
    completed = load_completed(output_csv)
    if completed:
        log(f'Resuming: {len(completed)} already in {output_csv} -- skipping.')
    end = len(provider_ids) if limit is None else min(len(provider_ids), start_index + limit)

    session = requests.Session()
    index = start_index
    for index in range(start_index, end):
        pid = provider_ids[index]
        if pid in completed:
            continue
        try:
            row, star = crawl_one(pid, session)
            append_row(row, output_csv)
            completed.add(pid)
            tag = 'OK' if not row.get('errors') else 'PARTIAL'
            log(f'[{index}] {tag}: {pid}' + (f' ({star} star)' if star else ''))
        except Exception:
            log(f'[{index}] EXCEPTION: {pid}')
            log(traceback.format_exc())
            row = {'provider_id': pid, 'errors': 'exception'}
            append_row(row, output_csv)
            completed.add(pid)

        if index < end - 1:
            delay = random.uniform(*delay_range)
            time.sleep(delay)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Oklahoma Child Care Locator provider-detail crawler.')
    ap.add_argument('--seed', default=SEED_PATH, help='Provider seed CSV.')
    ap.add_argument('--output', default=OUT_PATH)
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--delay-min', type=float, default=1)
    ap.add_argument('--delay-max', type=float, default=3)
    args = ap.parse_args()

    create_log_file()
    seed = load_seed(args.seed)
    ids = seed['provider_id'].tolist()
    log(f'Seed: {len(ids)} unique providers.')
    crawler(ids, output_csv=args.output, start_index=args.start_index,
            limit=args.limit, delay_range=(args.delay_min, args.delay_max))
