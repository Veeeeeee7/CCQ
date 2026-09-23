"""
mt_seed.py — build the Montana STARS rating seed from the state's published PDF.

Montana's "Best Beginnings STARS to Quality" 5-star ratings are published only
in a static DPHHS PDF:

    STARS Program List by City  (Updated 7/1/2023)
    https://dphhs.mt.gov/assets/ecfsd/childcare/STARS/STARSProgramListbyCity.pdf

a 5-column table (City | STAR Level | Program Type | Program Name | CCR&R
Region) with NO license number, so this seed is the RATING half of the dataset;
mt_crawler.py recovers the license number by joining on name + county.

STAR Level: Pre-Star (accepted but not yet rated, an invalid rating), 1-5.
Program Type: Center, Group, Family. CCR&R Region: 1-7.

Parsing: pdfplumber words are clustered into visual rows by y-coordinate, then
each row is parsed with a regex anchored on the STAR-level and Program-Type
tokens, which never appear inside a city or program name. Unparsed lines are
logged to mt_seed_log.txt so a lost data row is visible.

This script writes the STARS half ONLY. The shipped mt_data/mt_seed.csv also
holds the MAQCS licensing roster, so main() refuses to overwrite it unless you
pass --force; use --out to parse to a scratch file instead.

    python mt_seed.py --out mt_data/mt_seed_stars.csv  # download (if not cached) + parse
    python mt_seed.py --pdf some.pdf                   # parse a local copy instead
    python mt_seed.py --limit 20                       # smoke test: first 20 rows
    python mt_seed.py --self-test                      # parse the built-in wrapped-row fixtures

Deps: pip install requests pdfplumber pandas
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter

import pandas as pd

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

PDF_URL = ('https://dphhs.mt.gov/assets/ecfsd/childcare/STARS/'
           'STARSProgramListbyCity.pdf')

OUT_DIR = 'mt_data'
PDF_CACHE = os.path.join(OUT_DIR, 'STARSProgramListbyCity.pdf')
SEED_PATH = os.path.join(OUT_DIR, 'mt_seed.csv')
LOG_FILE = 'mt_seed_log.txt'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# City | STAR | Type | Name | Region(1-7). Non-greedy Name + anchored trailing
# Region handles names that themselves end in a number. `\s*` between star and
# type: some rows render kerned together ("4Center").
#
# The city group is CITY-shaped (no digits), not `.+?`: with `.+?` a glued line
# spanning a wrapped row and the next row still "parses", the city swallowing
# the whole first row and silently losing it.
CITY_CHARS = r"[A-Za-z][A-Za-z .'’-]*?"
ROW_RE = re.compile(
    r'^(?P<city>' + CITY_CHARS + r')\s+'
    r'(?P<star>Pre-?Star|[1-5])\s*'
    r'(?P<ptype>Center|Group|Family)\s+'
    r'(?P<name>.+?)\s+'
    r'(?P<region>[1-7])$'
)

# The same row with its STAR cell missing: when a row wraps, the STAR glyph can
# land on a visual line of its own, just before or after the rest of the row.
ROW_NOSTAR_RE = re.compile(
    r'^(?P<city>' + CITY_CHARS + r')\s+'
    r'(?P<ptype>Center|Group|Family)\s+'
    r'(?P<name>.+?)\s+'
    r'(?P<region>[1-7])$'
)

# A visual line holding nothing but an orphaned STAR cell.
BARE_STAR_RE = re.compile(r'^(Pre-?Star|[1-5])$', re.I)

# A city group that ends in a star token: "Missoula Pre-Star" would otherwise
# satisfy ROW_NOSTAR_RE on a line that is really a complete row.
CITY_ENDS_IN_STAR_RE = re.compile(r'(?:^|\s)Pre-?Star$', re.I)

# The page banner states the true row count; main() checks against it.
TOTAL_RE = re.compile(r'Total\s+Programs:\s*([\d,]+)', re.I)

# Lines we expect to skip silently (banners / header / footers / title).
NOISE_RE = re.compile(
    r'(^Region\s+\d+:|^City\s+STAR|^STARS to Quality|^Updated\b|^\d+\s+of\s+\d+$'
    r'|CCR&R\s+Region)', re.I)


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
# download
# ---------------------------------------------------------------------------

def _download_plain():
    """Vanilla requests; on some networks it dies in the TLS handshake."""
    import requests
    r = requests.get(PDF_URL, headers={'User-Agent': UA}, timeout=60)
    r.raise_for_status()
    return r.content


def _download_relaxed_tls():
    """dphhs.mt.gov's TLS stack fails the handshake against OpenSSL 3's
    defaults (`SSLEOFError: UNEXPECTED_EOF_WHILE_READING`, before any HTTP is
    sent). Drop to SECLEVEL=1 and re-allow legacy server connects, which is
    what a browser effectively tolerates here."""
    import ssl
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.ssl_ import create_urllib3_context

    ctx = create_urllib3_context(ciphers='DEFAULT@SECLEVEL=1')
    ctx.options |= getattr(ssl, 'OP_LEGACY_SERVER_CONNECT', 0x04)
    ctx.check_hostname = True

    class _TLSAdapter(HTTPAdapter):
        def init_poolmanager(self, *a, **kw):
            kw['ssl_context'] = ctx
            return super().init_poolmanager(*a, **kw)

    s = requests.Session()
    s.mount('https://', _TLSAdapter())
    r = s.get(PDF_URL, headers={'User-Agent': UA}, timeout=60)
    r.raise_for_status()
    return r.content


def _download_playwright():
    """Last resort: Playwright's request context uses the browser's own TLS
    stack (BoringSSL) rather than OpenSSL."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        req = p.request.new_context(user_agent=UA)
        resp = req.get(PDF_URL, timeout=60000)
        if not resp.ok:
            raise RuntimeError(f'HTTP {resp.status}')
        body = resp.body()
        req.dispose()
    return body


def ensure_pdf(pdf_path=None):
    """Return a path to the STARS PDF, downloading to the cache if needed
    through progressively more tolerant transports."""
    if pdf_path:
        if not os.path.exists(pdf_path):
            log(f'[error] --pdf given but not found: {pdf_path}')
            sys.exit(1)
        return pdf_path
    if os.path.exists(PDF_CACHE):
        log(f'[pdf] using cached {PDF_CACHE}')
        return PDF_CACHE

    os.makedirs(OUT_DIR, exist_ok=True)
    log(f'[pdf] downloading {PDF_URL}')

    attempts = (
        ('requests (default TLS)', _download_plain),
        ('requests (SECLEVEL=1 + legacy renegotiation)', _download_relaxed_tls),
        ("playwright request context (browser's TLS stack)", _download_playwright),
    )
    content = None
    for label, fn in attempts:
        try:
            content = fn()
            log(f'[pdf] fetched via {label}')
            break
        except Exception as e:
            log(f'[pdf] {label} failed: {type(e).__name__}: {e}')

    if content is None:
        log('[error] every transport failed. Download the PDF by hand and pass '
            'it in:\n'
            f'    curl -o {PDF_CACHE} "{PDF_URL}"\n'
            f'    python mt_seed.py --pdf {PDF_CACHE}')
        sys.exit(1)

    if not content.startswith(b'%PDF'):
        log(f'[error] response is not a PDF (first bytes: {content[:16]!r}) -- '
            f'the URL may now redirect to an HTML error page.')
        sys.exit(1)

    with open(PDF_CACHE, 'wb') as f:
        f.write(content)
    log(f'[pdf] saved {PDF_CACHE} ({len(content):,} bytes)')
    return PDF_CACHE


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

def _cluster_rows(words, y_tol=3.0):
    """Group pdfplumber words into visual rows by their 'top' coordinate,
    then return each row as a single left-to-right string."""
    rows = []
    for w in sorted(words, key=lambda w: (round(w['top'] / y_tol), w['x0'])):
        placed = False
        for row in rows:
            if abs(row['top'] - w['top']) <= y_tol:
                row['words'].append(w)
                placed = True
                break
        if not placed:
            rows.append({'top': w['top'], 'words': [w]})
    lines = []
    for row in sorted(rows, key=lambda r: r['top']):
        ws = sorted(row['words'], key=lambda w: w['x0'])
        text = ' '.join(w['text'] for w in ws).strip()
        text = re.sub(r'\s+', ' ', text)
        lines.append(text)
    return lines


def _norm_key(s):
    """Loose name/city join key; must stay identical to mt_crawler._norm_key."""
    s = (s or '').lower()
    s = s.replace('&', ' and ')
    s = re.sub(r"[’']", '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    s = re.sub(r'\b(llc|inc|incorporated|co)\b', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def _bare_star(line):
    """The STAR value of a line that holds nothing else, or None."""
    m = BARE_STAR_RE.match((line or '').strip())
    return m.group(1) if m else None


def _emit(fields, pageno):
    star = fields['star']
    star = 'Pre-Star' if star.lower().startswith('pre') else star
    name = fields['name'].strip()
    city = fields['city'].strip()
    return {
        'program_name': name,
        'city': city,
        'star_level': star,
        'program_type': fields['ptype'],
        'ccrr_region': fields['region'],
        'name_key': _norm_key(name),
        'city_key': _norm_key(city),
        'source_page': pageno,
    }


def parse_lines(lines, pageno=1):
    """Parse one page's clustered lines into records; returns (records, skipped).

    Some rows wrap, so city / star / rest land on different visual lines.
    Two recoveries, tried in order:

    1. Glue the line to the next one or two and re-match, consuming the glued
       lines so they are not also reported as orphans.
    2. If the line is a complete row MISSING ONLY ITS STAR, take the star from
       the adjacent line that holds nothing but a star token.
    """
    records, skipped = [], []
    pending_star = None     # an orphaned STAR seen on the previous line
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        m = ROW_RE.match(line)
        consumed = 1
        fields = m.groupdict() if m else None

        if fields is None:
            # (1) try gluing forward 1 then 2 lines (wrapped cell)
            for extra in (1, 2):
                if i + extra < len(lines):
                    glued = ' '.join(lines[i:i + extra + 1]).strip()
                    glued = re.sub(r'\s+', ' ', glued)
                    if NOISE_RE.search(glued):
                        continue
                    m2 = ROW_RE.match(glued)
                    if m2:
                        fields, consumed = m2.groupdict(), extra + 1
                        break

        if fields is None:
            # (2) a complete row whose STAR wrapped onto a line of its own
            ns = ROW_NOSTAR_RE.match(line)
            if ns and not CITY_ENDS_IN_STAR_RE.search(ns.group('city')):
                star_after = (_bare_star(lines[i + 1])
                              if i + 1 < len(lines) else None)
                if pending_star is not None:
                    fields, consumed = dict(ns.groupdict(),
                                            star=pending_star), 1
                elif star_after is not None:
                    fields, consumed = dict(ns.groupdict(),
                                            star=star_after), 2

        if fields is None:
            bare = _bare_star(line)
            if bare is not None:
                # Hold it: the row it belongs to may be the next line.
                pending_star = bare
            elif not NOISE_RE.search(line):
                skipped.append((pageno, line))
            i += 1
            continue

        pending_star = None
        records.append(_emit(fields, pageno))
        i += consumed

    return records, skipped


def parse_pdf(pdf_path, limit=None):
    """Cluster each page into visual rows and hand them to parse_lines()."""
    import pdfplumber
    records, skipped = [], []
    total_expected = None

    with pdfplumber.open(pdf_path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(use_text_flow=False,
                                       keep_blank_chars=False)
            lines = _cluster_rows(words)

            if total_expected is None:
                for ln in lines:
                    t = TOTAL_RE.search(ln)
                    if t:
                        total_expected = int(t.group(1).replace(',', ''))
                        break

            page_records, page_skipped = parse_lines(lines, pageno)
            skipped.extend(page_skipped)
            for rec in page_records:
                records.append(rec)
                if limit and len(records) >= limit:
                    return records, skipped, total_expected

    return records, skipped, total_expected


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------

def foreign_seed_halves(path):
    """Which OTHER halves (e.g. the licensing roster) an existing seed file
    already holds; main() refuses to flatten them without --force."""
    if not os.path.exists(path):
        return {}
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        if 'seed_source' not in (reader.fieldnames or []):
            return {}
        counts = Counter(r.get('seed_source', '') for r in reader)
    return {k: v for k, v in counts.items() if k != 'stars_ratings'}


# Three rows the real PDF wraps, as pdfplumber clusters them: each STAR cell
# landed on a visual line of its own.
SELF_TEST_LINES = [
    'City STAR Level Program Type Program Name CCR&R Region',
    'Kalispell 4 Center Discovery Developmental Center 1',
    'Kalispell Center Woodland Montessori School 1',
    '3',
    'Kalispell 2 Group A Place to Grow Preschool 1',
    'Missoula Center Learning and Belonging LAB Preschool 2',
    '4',
    'Missoula 4 Center Montessori Plus International 2',
    'Missoula Group Beautiful Beginnings 2',
    'Pre-Star',
    'Missoula Pre-Star Center Missoula Parent Co-op/Kid Central 2',
    'Missoula 4Center Kerned Star Preschool 2',
    'Great Falls Pre-Star Family Little Dreamers LLC 2 4',
    'Updated 7/1/2023',
    '1 of 7',
]

SELF_TEST_EXPECT = [
    ('Discovery Developmental Center', 'Kalispell', '4', 'Center', '1'),
    ('Woodland Montessori School', 'Kalispell', '3', 'Center', '1'),
    ('A Place to Grow Preschool', 'Kalispell', '2', 'Group', '1'),
    ('Learning and Belonging LAB Preschool', 'Missoula', '4', 'Center', '2'),
    ('Montessori Plus International', 'Missoula', '4', 'Center', '2'),
    ('Beautiful Beginnings', 'Missoula', 'Pre-Star', 'Group', '2'),
    ('Missoula Parent Co-op/Kid Central', 'Missoula', 'Pre-Star', 'Center', '2'),
    ('Kerned Star Preschool', 'Missoula', '4', 'Center', '2'),
    ('Little Dreamers LLC 2', 'Great Falls', 'Pre-Star', 'Family', '4'),
]


def self_test():
    """Parse SELF_TEST_LINES and check every row comes back intact."""
    records, skipped = parse_lines(SELF_TEST_LINES, pageno=1)
    got = [(r['program_name'], r['city'], r['star_level'],
            r['program_type'], r['ccrr_region']) for r in records]
    ok = True
    for want in SELF_TEST_EXPECT:
        if want not in got:
            print(f'  MISSING {want}')
            ok = False
    for have in got:
        if have not in SELF_TEST_EXPECT:
            print(f'  UNEXPECTED {have}')
            ok = False
    for pageno, line in skipped:
        print(f'  [skip p{pageno}] {line}')
    print(f'self-test: {len(got)} parsed, {len(SELF_TEST_EXPECT)} expected, '
          f'{len(skipped)} skipped -> {"PASS" if ok and not skipped else "FAIL"}')
    return 0 if (ok and not skipped) else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', default=None,
                    help='Parse a local PDF copy instead of downloading.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Only keep the first N parsed rows (smoke test).')
    ap.add_argument('--out', default=SEED_PATH,
                    help=f'Where to write the STARS half (default {SEED_PATH}).')
    ap.add_argument('--force', action='store_true',
                    help='Overwrite a seed that also holds the licensing '
                         'roster. Destroys that half.')
    ap.add_argument('--allow-short', action='store_true',
                    help="Ship even if the row count misses the PDF's own "
                         '"Total Programs" banner.')
    ap.add_argument('--self-test', action='store_true',
                    help='Parse the built-in wrapped-row fixtures and exit.')
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    os.makedirs(OUT_DIR, exist_ok=True)
    create_log_file()

    foreign = foreign_seed_halves(args.out)
    if foreign and not args.force:
        log(f'[error] {args.out} already holds {foreign} besides the STARS '
            f'half, and this script writes the STARS half ONLY -- running it '
            f'would destroy the licensing roster that mt_crawler.py --join-only '
            f'depends on.\n'
            f'    python mt_seed.py --out {os.path.join(OUT_DIR, "mt_seed_stars.csv")}'
            f'   # parse to a scratch file and diff\n'
            f'    python mt_seed.py --force                        '
            f'   # really flatten it')
        sys.exit(1)

    pdf_path = ensure_pdf(args.pdf)
    records, skipped, total_expected = parse_pdf(pdf_path, limit=args.limit)

    if not records:
        log('[error] parsed 0 rows -- the PDF layout likely changed; inspect '
            'the skipped lines below and adjust ROW_RE / clustering.')
        for pageno, line in skipped[:40]:
            log(f'  [skip p{pageno}] {line}')
        sys.exit(1)

    df = pd.DataFrame(records)

    # The PDF states its own total; refuse to write a table that misses it.
    if total_expected and not args.limit:
        if len(df) != total_expected:
            log(f'[error] PDF banner says "Total Programs: {total_expected}" but '
                f'we parsed {len(df)}. {abs(total_expected - len(df))} row(s) '
                f'were lost or invented -- inspect the skipped lines below. '
                f'Nothing was written. Re-run with --allow-short to ship anyway.')
            for pageno, line in skipped[:40]:
                log(f'  [skip p{pageno}] {line}')
            if not args.allow_short:
                sys.exit(1)
        else:
            log(f'[ok] row count matches the PDF banner ({total_expected})')

    df.to_csv(args.out, index=False)

    log(f'[done] parsed {len(df)} rows -> {args.out}')
    log(f'[star_level breakdown]\n{df["star_level"].value_counts().to_string()}')
    log(f'[program_type breakdown]\n{df["program_type"].value_counts().to_string()}')
    log(f'[region breakdown]\n{df["ccrr_region"].value_counts().sort_index().to_string()}')
    dupes = df.duplicated(subset=['name_key', 'city_key']).sum()
    if dupes:
        log(f'[warn] {dupes} rows share a name_key+city_key (possible dup or '
            f'same operator at multiple star records) -- inspect before join')
    if skipped:
        log(f'[note] {len(skipped)} non-matching lines skipped (banners/'
            f'headers expected; verify none are real data rows):')
        for pageno, line in skipped[:40]:
            log(f'  [skip p{pageno}] {line}')


if __name__ == '__main__':
    main()
