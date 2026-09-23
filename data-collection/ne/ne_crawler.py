"""
ne_crawler.py — Nebraska Step Up to Quality provider crawler.

Crawls every `child-care-facility` page on the STQ finder and emits one row per
provider page, then left-joins the DHHS licensing roster from the seed.

    https://stepuptoquality.ne.gov/child-care-facility/{slug}/

No browser needed: the site's reCAPTCHA only guards the newsletter form, and
every provider page is a plain WordPress permalink whose content is in the
initial HTML. The seed enumerates the URLs from the Yoast sitemaps.

The hero carries the name (h1), the program type (h2) and, only when the
provider is rated, a `.step-badge` holding the Step (1-5). The body is a grid
of `.plumb-columns .column` label/value or label/list blocks; fields are
omitted when a provider has no value. Labels without a dedicated column are
kept in the `extra_fields` JSON blob so a site change surfaces as data.

Some pages (typically public-school preschools) carry no License Number; they
are still recorded, with `has_license_number=0`.

Grain: one row per facility page (slug). A license number can appear on more
than one page (e.g. `waldecker-connie` / `waldecker-connie-2`).

Usage:
    python ne_crawler.py --limit 25        # smoke test
    python ne_crawler.py                   # full run (resumable)

Deps: pip install requests pandas beautifulsoup4
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://stepuptoquality.ne.gov'
SEED_CSV = 'ne_data/ne_seed.csv'
OUT_CSV = 'ne_data/ne_records.csv'
LOG_FILE = 'ne_crawler_log.txt'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

SEL_STEP_BADGE = '.step-badge'
SEL_COLUMNS = '.plumb-columns .column'
SEL_ACCRED_LIST = 'ul.accredidation-list'
SEL_PHONE = 'li.icon-phone, li.has-icon.icon-phone'
SEL_ADDRESS = 'li.icon-address, li.has-icon.icon-address'

REQUEST_TIMEOUT = 45
RETRIES = 3
DELAY_RANGE = (0.6, 1.4)

# Anything else -> extra_fields.
LABEL_TO_FIELD = {
    'director': 'director',
    'license number': 'license_number',
    'age of children served': 'age_groups',
    'program accreditations': 'accreditations',
    'full time staff': 'full_time_staff',
    'part time staff': 'part_time_staff',
    'capacity': 'capacity',
    'other program information': 'other_program_info',
}

MULTIVALUE_SEP = '; '


def create_log_file(path=LOG_FILE):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


def now_utc():
    """Second-resolution UTC stamp, ISO-8601 with an explicit Z."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def fetch(url, session):
    last = None
    for attempt in range(RETRIES):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except Exception as e:                                   # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


def _clean(s):
    if s is None:
        return None
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s or None


def _norm_license(value):
    """License numbers are strings: letter prefix + digits (CCC8794, FI11670,
    FII9561). Strip a stray float '.0'; never zfill (NE does not zero-pad)."""
    v = _clean(value)
    if not v:
        return None
    v = re.sub(r'\.0$', '', v)
    return v.upper()


def _label_of(column):
    span = column.select_one('p span.label')
    return _clean(span.get_text(' ', strip=True)).lower() if span else None


def _value_of(column):
    """Text of the <p> with the label span removed — that's the scalar value."""
    p = column.find('p')
    if p is None:
        return None
    clone = BeautifulSoup(str(p), 'html.parser')
    span = clone.select_one('span.label')
    if span:
        span.decompose()
    return _clean(clone.get_text(' ', strip=True))


def _list_items(column):
    ul = column.find('ul')
    if ul is None:
        return []
    return [t for t in (_clean(li.get_text(' ', strip=True))
                        for li in ul.find_all('li')) if t]


def _accreditations(column):
    """[{'name': ..., 'dates': ...}] from ul.accredidation-list."""
    ul = column.select_one(SEL_ACCRED_LIST) or column.find('ul')
    if ul is None:
        return []
    out = []
    for li in ul.find_all('li'):
        date_el = li.select_one('span.date')
        dates = _clean(date_el.get_text(' ', strip=True)) if date_el else None
        name_el = li.select_one('span.font-size-large')
        if name_el:
            name = _clean(name_el.get_text(' ', strip=True))
        else:
            clone = BeautifulSoup(str(li), 'html.parser')
            for d in clone.select('span.date'):
                d.decompose()
            name = _clean(clone.get_text(' ', strip=True))
        if name or dates:
            out.append({'name': name, 'dates': dates})
    return out


# Each section has an empty_*() twin so a failure in one never kills the row.

def empty_header():
    return {'facility_name': None, 'program_type': None, 'step_rating': None}


def crawl_header(soup):
    out = empty_header()
    h1 = soup.find('h1')
    if h1 is not None:
        out['facility_name'] = _clean(h1.get_text(' ', strip=True))
        h2 = h1.find_next('h2')
        if h2 is not None:
            out['program_type'] = _clean(h2.get_text(' ', strip=True))
    badge = soup.select_one(SEL_STEP_BADGE)
    if badge is not None:
        # The badge text is a bare digit ("3"); unrated providers have no badge.
        m = re.search(r'\d+', badge.get_text(' ', strip=True) or '')
        if m:
            out['step_rating'] = m.group(0)
    return out


def empty_program():
    return {'license_number': None, 'director': None, 'age_groups': None,
            'full_time_staff': None, 'part_time_staff': None, 'capacity': None,
            'other_program_info': None, 'accreditations': None,
            'extra_fields': None}


def crawl_program(soup):
    out = empty_program()
    extra = {}
    for column in soup.select(SEL_COLUMNS):
        label = _label_of(column)
        if not label:
            continue
        field = LABEL_TO_FIELD.get(label)

        if field == 'accreditations':
            accs = _accreditations(column)
            out['accreditations'] = json.dumps(accs, ensure_ascii=False) if accs else None
        elif field in ('age_groups', 'other_program_info'):
            items = _list_items(column)
            out[field] = MULTIVALUE_SEP.join(items) if items else None
        elif field:
            out[field] = _value_of(column)
        else:
            items = _list_items(column)
            extra[label] = MULTIVALUE_SEP.join(items) if items else _value_of(column)

    out['license_number'] = _norm_license(out['license_number'])
    out['extra_fields'] = json.dumps(extra, ensure_ascii=False) if extra else None
    return out


def empty_contact():
    return {'phone': None, 'address_raw': None, 'street': None, 'city': None,
            'state': None, 'zip_code': None}


_ADDR_TAIL = re.compile(r'^(?P<city>.+?)\s*,\s*(?P<state>[A-Za-z]{2})\s*,?\s*'
                        r'(?P<zip>\d{5}(?:-\d{4})?)\s*$')


def crawl_contact(soup):
    out = empty_contact()
    phone_el = soup.select_one(SEL_PHONE)
    if phone_el is not None:
        out['phone'] = _clean(phone_el.get_text(' ', strip=True))

    addr_el = soup.select_one(SEL_ADDRESS)
    if addr_el is not None:
        # Keep <br> as a line break so street and city/state/zip stay separable.
        raw = addr_el.get_text('\n', strip=True)
        lines = [_clean(l) for l in raw.split('\n') if _clean(l)]
        out['address_raw'] = _clean(' '.join(lines))
        if lines:
            m = _ADDR_TAIL.match(lines[-1])
            if m:
                out['street'] = _clean(' '.join(lines[:-1])) or None
                out['city'] = _clean(m.group('city'))
                out['state'] = m.group('state').upper()
                out['zip_code'] = m.group('zip')
            else:
                # Single-line / unparseable: keep the raw text, don't guess.
                out['street'] = out['address_raw']
    return out


def empty_meta():
    # fetched_at is filled in by crawl_one(); it lives here because
    # all_columns() builds the CSV header from the empty_* constructors.
    return {'facility_id': None, 'fetched_at': None}


_FACILITY_ID = re.compile(r'[?&]facility=(\d+)')


def crawl_meta(soup):
    """The site's internal WP post id. Not the licensing ID, but a stable
    secondary key (and the only key on pages that lack a license number)."""
    out = empty_meta()
    # The "Favorite" control carries the post id; prefer it over a bare
    # [data-id] match, which could pick up an unrelated widget.
    node = soup.select_one('.favorites-save[data-id]')
    if node is not None and (node.get('data-id') or '').strip().isdigit():
        out['facility_id'] = node['data-id'].strip()
    if out['facility_id'] is None:
        # Fallback: the Print Visitation Checklist links carry ?facility=<id>.
        for a in soup.select('a[href*="facility="]'):
            m = _FACILITY_ID.search(a.get('href', ''))
            if m:
                out['facility_id'] = m.group(1)
                break
    return out


def all_columns():
    cols = ['slug', 'facility_url']
    for fn in (empty_header, empty_program, empty_contact, empty_meta):
        cols.extend(fn().keys())
    cols.extend(['has_license_number', 'errors'])
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


def assert_output_schema(output_csv):
    """An existing output file must have exactly the current header.

    append_row() writes the header only when the file is empty, so resuming
    into a file with a different schema would silently shift fields.
    """
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return
    with open(output_csv, encoding='utf-8') as fh:
        header = (fh.readline().rstrip('\r\n')).split(',')
    expected = all_columns()
    if header != expected:
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        raise SystemExit(
            f'{output_csv} was written with a different schema '
            f'({len(header)} columns vs {len(expected)}). Resuming would write '
            f'rows that do not line up with its header. '
            + (f'Columns this run would add: {missing}. ' if missing else '')
            + (f'Columns it no longer writes: {extra}. ' if extra else '')
            + 'Move the file aside and start a fresh full run.')


def load_completed(output_csv):
    if not output_csv or not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv, dtype=str, usecols=['slug'])
        return set(df['slug'].dropna().str.strip())
    except Exception as e:                                       # noqa: BLE001
        log(f'Could not read existing {output_csv}: {e}')
        return set()


def load_seed(seed_csv):
    if not os.path.exists(seed_csv):
        raise FileNotFoundError(
            f'Seed CSV not found: {seed_csv}.')
    df = pd.read_csv(seed_csv, dtype=str)
    for c in ('slug', 'facility_url'):
        if c not in df.columns:
            raise ValueError(f'Seed needs a {c} column; got {list(df.columns)}')
    df = (df[df['facility_url'].notna()]
          .drop_duplicates(subset='slug').reset_index(drop=True))
    return df


def crawl_one(url, session):
    """Fetch + parse one provider page; a failed section is named in `errors`."""
    html = fetch(url, session)
    fetched_at = now_utc()
    if html is None:
        return None, 'http_404'

    soup = BeautifulSoup(html, 'html.parser')
    row, errors = {}, []
    for name, fn, empty in (('header', crawl_header, empty_header),
                            ('program', crawl_program, empty_program),
                            ('contact', crawl_contact, empty_contact),
                            ('meta', crawl_meta, empty_meta)):
        try:
            row.update(fn(soup))
        except Exception as e:                                   # noqa: BLE001
            row.update(empty())
            errors.append(name)
            log(f'    section "{name}" failed: {e}')
    # After the loop: a failing meta section resets the meta keys.
    row['fetched_at'] = fetched_at
    row['has_license_number'] = 1 if row.get('license_number') else 0
    return row, (';'.join(errors) or None)


def crawler(seed_csv=SEED_CSV, output_csv=OUT_CSV, limit=None, resume=True):
    assert_output_schema(output_csv)
    started = now_utc()
    log(f'=== crawl started {started} | seed={seed_csv} out={output_csv} '
        f'limit={limit} resume={resume}')
    seed = load_seed(seed_csv)
    done = load_completed(output_csv) if resume else set()
    todo = seed[~seed['slug'].isin(done)] if done else seed
    if limit:
        todo = todo.head(int(limit))

    log(f'Seed: {len(seed)} facilities | already done: {len(done)} | '
        f'this run: {len(todo)}')

    session = requests.Session()
    session.headers.update({'User-Agent': UA})

    rated = unrated = no_license = failed = 0
    for i, rec in enumerate(todo.itertuples(index=False), 1):
        url = rec.facility_url
        try:
            row, errors = crawl_one(url, session)
            if row is None:
                log(f'[{i}/{len(todo)}] 404 {rec.slug}')
                failed += 1
                continue
            row['slug'] = rec.slug
            row['facility_url'] = url
            row['errors'] = errors
            append_row(row, output_csv)

            if row.get('step_rating'):
                rated += 1
            else:
                unrated += 1
            if not row.get('license_number'):
                no_license += 1

            log(f'[{i}/{len(todo)}] {rec.slug} | '
                f'step={row.get("step_rating") or "-"} '
                f'lic={row.get("license_number") or "-"} '
                f'type={row.get("program_type") or "-"}'
                + (f' | errors={errors}' if errors else ''))
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            log(f'[{i}/{len(todo)}] FAILED {rec.slug}: {e}')

        time.sleep(random.uniform(*DELAY_RANGE))

    log(f'\nDone. rated={rated} unrated={unrated} '
        f'no_license_number={no_license} failed={failed} -> {output_csv}')
    log(f'=== crawl finished {now_utc()} (started {started})')



def attach_licensing(seed_csv=SEED_CSV, output_csv=OUT_CSV):
    """Left-join the DHHS licensing roster (the seed's `dhhs_roster` half) onto
    the crawled rows, in place.

    LEFT, not inner: a licence missing from the monthly roster (a new or
    recently-closed site) keeps its rating with empty licensing columns.
    """
    seed = pd.read_csv(seed_csv, dtype=str, keep_default_na=False, low_memory=False)
    roster = seed[seed['seed_source'] == 'dhhs_roster'].drop(columns=['seed_source'])
    roster = roster.loc[:, [c for c in roster.columns
                            if roster[c].astype(str).str.strip().ne('').any()]]

    # The roster occasionally repeats a number; the first occurrence is current.
    roster['License_Number'] = (roster['License_Number'].astype(str).str.strip()
                                .str.replace(r'\.0$', '', regex=True).str.upper())
    roster = roster.drop_duplicates(subset='License_Number', keep='first')
    roster = roster.rename(columns={c: f'dhhs_{_slug(c)}' for c in roster.columns})

    records = pd.read_csv(output_csv, dtype=str, keep_default_na=False,
                          low_memory=False)
    key = records['license_number'].astype('string').str.strip().str.upper()
    merged = (records.assign(_key=key)
                     .merge(roster, how='left',
                            left_on='_key', right_on='dhhs_license_number')
                     .drop(columns=['_key']))
    if len(merged) != len(records):
        raise ValueError(f'join changed the row count: {len(records)} -> {len(merged)}')

    merged.to_csv(output_csv, index=False)
    matched = int(merged['dhhs_license_type'].notna().sum())
    log(f'licensing join {now_utc()}: {matched}/{len(merged)} rows matched '
        f'the roster; {records.shape[1]} -> {merged.shape[1]} columns')


def _slug(text):
    text = str(text).strip().lower().replace('&', ' and ')
    return re.sub(r'[^a-z0-9]+', '_', text).strip('_')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Crawl NE Step Up to Quality provider pages.')
    ap.add_argument('--seed', default=SEED_CSV)
    ap.add_argument('--output', default=OUT_CSV)
    ap.add_argument('--limit', type=int, default=None,
                    help='Only crawl the first N not-yet-done facilities (smoke test).')
    ap.add_argument('--no-resume', action='store_true',
                    help='Ignore rows already present in the output CSV.')
    ap.add_argument('--fresh-log', action='store_true')
    ap.add_argument('--join-only', action='store_true',
                    help='Skip the crawl; only fold the DHHS roster into an '
                         'existing records CSV. Run once after a full crawl.')
    args = ap.parse_args()

    if args.fresh_log and args.join_only:
        raise SystemExit(
            '--fresh-log with --join-only truncates the log and then writes '
            'only the join summary, losing the crawl lines. Drop one of the '
            'two flags.')
    if args.fresh_log:
        create_log_file()
    if args.join_only:
        attach_licensing(seed_csv=args.seed, output_csv=args.output)
    else:
        crawler(seed_csv=args.seed, output_csv=args.output, limit=args.limit,
                resume=not args.no_resume)
        attach_licensing(seed_csv=args.seed, output_csv=args.output)
