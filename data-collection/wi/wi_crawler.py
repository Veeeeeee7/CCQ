"""
Wisconsin Child Care Finder crawler.

One row per provider-location. The finder is a Blazor Server app gated by
reCAPTCHA v3, so pages are rendered in a real browser (persistent profile,
never networkidle). Detail pages load from a direct URL:
  /ProviderDetails?ProviderNumber={pn}&LocationNumber={ln}&Provider={pn}&CCF=Y
with zero-padded IDs (e.g. ProviderNumber=0000555700, LocationNumber=001).

A resume skips only rows whose `errors` cell is empty; a failed row is
re-requested and overwritten at its own row index (replace_rows), because
downstream files are joined to this one positionally. `--retry-errors` visits
only the failed rows; `--dry-run` lists what would be requested and writes
nothing. `--download-pdfs` also fetches each provider's documents.

Deps: pip install playwright pandas beautifulsoup4  (and: playwright install chromium)
"""

import argparse
import csv
import json
import os
import random
import re
import time
import traceback

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

BASE_URL = 'https://childcarefinder.wisconsin.gov'

PROVIDER_URL_TEMPLATE = (BASE_URL + '/ProviderDetails'
                         '?ProviderNumber={pn}&LocationNumber={ln}&Provider={pn}&CCF=Y')

# Set True if the detail page ever requires a search-established session.
REQUIRES_SEARCH_SESSION = False

HEADING_YOUNGSTAR = 'youngstarDetailsHeading'
HEADING_REGULATION = 'regulationDetailsHeading'
HEADING_PROVIDER_REPORTED = 'providerReportedDetailsHeading'

# Every regulated provider has a Regulation Details section, and the
# waiting/not-found screen has no accordion sections at all, so this heading
# appears only once the provider's data has rendered.
SEL_PROVIDER_LOADED = '#regulationDetailsHeading'

PROVIDER_WAIT_MS = 60000

# Separates a provider that is really gone from a page that never rendered.
NOT_FOUND_TEXT = 'Active provider information was not found'

MAX_DOCS = 60  # safety cap on documents collected per provider

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/120.0.0.0 Safari/537.36')


def create_log_file(path='wi_crawler_log.txt'):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file='wi_crawler_log.txt'):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


DIR_RENAME = {
    'Provider Number': 'provider_number', 'Location Number': 'location_number',
    'Application Type': 'application_type', 'County': 'county',
    'Facility Name': 'facility_name', 'Facility Number': 'facility_number',
    'Line Address 1': 'address_1', 'Line Address 2': 'address_2', 'City': 'city',
    'Zip Code': 'zip', 'Contact Name': 'contact_name',
    'Contact Phone': 'contact_phone', 'Capacity': 'capacity',
    'From Age': 'from_age', 'To Age': 'to_age', 'Hours': 'hours',
    'Months': 'months', 'Full Time': 'full_time', 'Star Level ': 'star_level',
}


def _norm_id(series, width):
    """Force an ID column to a zero-padded string, restoring leading zeros a
    numeric source may have dropped (555670 -> '0000555670', 4 -> '004')."""
    return (series.astype(str).str.strip()
            .str.replace(r'\.0$', '', regex=True)
            .str.replace(r'\D', '', regex=True)
            .str.zfill(width))


def _load_one_directory(df, regulation_type):
    df = df.loc[:, [c for c in df.columns if c]]
    date_col = next((c for c in df.columns if c.strip().endswith('Date')), None)
    rename = dict(DIR_RENAME)
    if date_col:
        rename[date_col] = 'regulated_date'
    df = df.rename(columns=rename)
    df = df[df['provider_number'].astype(str).str.strip().ne('')].copy()
    df['regulation_type'] = regulation_type
    for c in df.columns:
        df[c] = df[c].map(lambda v: re.sub(r'\s+', ' ', v).strip()
                          if isinstance(v, str) else v)
    df['provider_number'] = _norm_id(df['provider_number'], 10)
    df['location_number'] = _norm_id(df['location_number'], 3)
    return df


def load_seed(seed_csv):
    # The licensed and certified directories are stacked into one seed, tagged
    # by seed_source. Each half keeps its download's two-line preamble, so find
    # and promote each half's own header row.
    whole = pd.read_csv(seed_csv, dtype=str, keep_default_na=False)
    frames = []
    for tag, label in (('lcc', 'licensed'), ('ccc', 'certified')):
        block = whole[whole['seed_source'] == tag].drop(columns='seed_source')
        if block.empty:
            continue
        header_rows = block.index[
            block.iloc[:, 0].str.strip().eq('Provider Number')]
        if header_rows.empty:
            raise ValueError(f'No header row in the {tag} half of {seed_csv}')
        at = header_rows[0]
        part = block.loc[at + 1:].copy()
        part.columns = [str(v).strip() for v in block.loc[at]]
        frames.append(_load_one_directory(part.reset_index(drop=True), label))
    if not frames:
        raise FileNotFoundError(f'No provider rows in {seed_csv}.')
    df = pd.concat(frames, ignore_index=True)
    df['provider_location'] = df['provider_number'] + '-' + df['location_number']
    df = df.drop_duplicates(subset='provider_location').reset_index(drop=True)
    return df


# Roster fields carried verbatim from the seed onto every crawled row (DCF's
# own directory values, not page reads).
SEED_CARRY = ('application_type', 'capacity', 'from_age', 'to_age')


def seed_carry(rec):
    return {c: rec.get(c) for c in SEED_CARRY}


def all_columns():
    cols = ['provider_location', 'provider_number', 'location_number',
            'facility_number', 'regulation_type', *SEED_CARRY, 'provider_url']
    for fn in (empty_youngstar, empty_regulation, empty_provider_reported,
               empty_documents):
        cols.extend(fn().keys())
    cols.append('errors')
    return cols


def _flat(value):
    """One record per physical line: collapse embedded newlines in a cell."""
    if isinstance(value, str):
        return re.sub(r'[\r\n]+', ' ', value).strip() or None
    return value


def append_row(row, output_csv):
    cols = all_columns()
    full = {c: _flat(row.get(c)) for c in cols}
    parent = os.path.dirname(output_csv)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    if exists:
        # An append writes no header, so a layout mismatch would silently
        # shift values between columns.
        with open(output_csv, newline='', encoding='utf-8') as fh:
            on_disk = next(csv.reader(fh))
        if on_disk != cols:
            raise ValueError(
                f'{output_csv} has a different column layout than this crawler '
                f'writes; refusing to append.\n  on disk: {on_disk}\n  '
                f'expected: {cols}')
    pd.DataFrame([full], columns=cols).to_csv(
        output_csv, mode='a', header=not exists, index=False)


def _existing(output_csv):
    """(provider_location -> errors) for every row already in the output."""
    if not output_csv or not os.path.exists(output_csv):
        return {}
    try:
        df = pd.read_csv(output_csv, dtype=str,
                         usecols=['provider_location', 'errors'],
                         keep_default_na=False)
    except Exception as e:
        log(f'Could not read existing {output_csv}: {e}')
        return {}
    return {str(k).strip(): str(v).strip()
            for k, v in zip(df['provider_location'], df['errors'])}


def replace_rows(output_csv, replacements):
    """Rewrite output_csv with `replacements` (provider_location -> row dict)
    substituted at the positions those providers already occupy, via a .tmp.
    Row count and order are preserved: downstream files join positionally."""
    if not replacements:
        return 0
    cols = all_columns()
    tmp = output_csv + '.tmp'
    written = 0
    with open(output_csv, newline='', encoding='utf-8') as src, \
            open(tmp, 'w', newline='', encoding='utf-8') as dst:
        reader = csv.reader(src)
        header = next(reader)
        if header != cols:
            raise ValueError(f'{output_csv} has an unexpected column layout; '
                             f'refusing to rewrite.\n  on disk: {header}')
        writer = csv.writer(dst)
        writer.writerow(header)
        key_at = header.index('provider_location')
        for rec in reader:
            key = rec[key_at].strip() if len(rec) > key_at else ''
            row = replacements.get(key)
            if row is None:
                writer.writerow(rec)
                continue
            writer.writerow([_flat(row.get(c)) for c in cols])
            written += 1
    os.replace(tmp, output_csv)
    return written


def wait_for_provider(page, timeout_ms=PROVIDER_WAIT_MS):
    """True once SEL_PROVIDER_LOADED is attached; False after timeout_ms.
    Never networkidle: Blazor Server holds a SignalR websocket open."""
    try:
        page.wait_for_selector(SEL_PROVIDER_LOADED, state='attached', timeout=timeout_ms)
        return True
    except PWTimeout:
        return False


def classify_load_failure(page):
    """'not_found' if the portal says the provider is gone, else 'timeout'
    (reCAPTCHA placeholder or slow Blazor circuit; worth requesting again)."""
    try:
        body = page.evaluate(
            "() => (document.body && document.body.innerText) || ''") or ''
    except Exception:
        return 'timeout'
    return 'not_found' if NOT_FOUND_TEXT.lower() in body.lower() else 'timeout'


def establish_search_session(page, rec):
    """Only needed if REQUIRES_SEARCH_SESSION: run the search flow and return
    the resulting detail URL (carrying UserSessionId + SearchId), or None."""
    raise NotImplementedError('Fill in the search flow, then set '
                              'REQUIRES_SEARCH_SESSION = True.')


def goto_provider(page, rec):
    """Navigate to a provider's detail page; return URL or None (skip)."""
    if REQUIRES_SEARCH_SESSION:
        url = establish_search_session(page, rec)
        if url and wait_for_provider(page):
            return page.url
        return None
    pn = rec['provider_number']
    ln = rec['location_number']
    url = PROVIDER_URL_TEMPLATE.format(pn=pn, ln=ln)
    page.goto(url, wait_until='domcontentloaded')
    if not wait_for_provider(page):
        return None
    return page.url


def _clean(s):
    if s is None:
        return None
    s = re.sub(r'\s+', ' ', str(s)).strip()
    return s or None


def _section_body(soup, heading_id):
    h = soup.find(id=heading_id)
    if not h:
        return None
    item = h.find_parent(class_='accordion-item')
    return item.find(class_='accordion-body') if item else None


def _parse_grid(table):
    """Turn a .Grid table into a list of {header: cell_text} dicts."""
    headers = [th.get_text(' ', strip=True) for th in table.find_all('th')]
    rows = []
    for tr in table.find_all('tr'):
        tds = tr.find_all('td')
        if not tds:
            continue
        cells = [td.get_text(' ', strip=True) for td in tds]
        if headers and len(headers) == len(cells):
            rows.append(dict(zip(headers, cells)))
        else:
            rows.append({f'col{i}': c for i, c in enumerate(cells)})
    return rows


def _desktop_grids(body):
    """Desktop .Grid tables only (skip .Phone mobile duplicates)."""
    out = []
    for t in body.find_all('table'):
        cls = t.get('class') or []
        if 'Grid' in cls and 'Phone' not in cls:
            out.append(t)
    return out


def empty_youngstar():
    return {'youngstar_star_rating': None,
            'youngstar_unique_services': None,
            'youngstar_section_text': None}


def crawl_youngstar(soup):
    out = empty_youngstar()
    body = _section_body(soup, HEADING_YOUNGSTAR)
    if body is None:
        return out
    out['youngstar_star_rating'] = len(body.select('span.fa-star'))
    svc = body.find(string=lambda s: s and 'Unique Program Services' in s)
    if svc:
        tbl = svc.find_parent('table')
        if tbl:
            lines = [_clean(li) for li in tbl.get_text('\n').split('\n')
                     if _clean(li) and 'Unique Program Services' not in li]
            out['youngstar_unique_services'] = ' | '.join(lines) or None
    out['youngstar_section_text'] = _clean(body.get_text(' '))
    return out


def empty_regulation():
    return {'regulation_enforcement_json': None,
            'regulation_monitoring_json': None,
            'regulation_violations_json': None,
            'regulation_section_text': None}


def crawl_regulation(soup):
    out = empty_regulation()
    body = _section_body(soup, HEADING_REGULATION)
    if body is None:
        return out
    for t in _desktop_grids(body):
        headers = [th.get_text(strip=True) for th in t.find_all('th')]
        hset = set(headers)
        rows = _parse_grid(t)
        if {'Appeal', 'Decision'} & hset:
            out['regulation_enforcement_json'] = json.dumps(rows, ensure_ascii=False)
        elif 'Rule Monitoring' in hset:
            out['regulation_monitoring_json'] = json.dumps(rows, ensure_ascii=False)
        elif 'Rule Number' in hset and 'Description' in hset:
            out['regulation_violations_json'] = json.dumps(rows, ensure_ascii=False)
    out['regulation_section_text'] = _clean(body.get_text(' '))
    return out


def empty_provider_reported():
    return {'pr_special_types_of_care': None,
            'pr_program_philosophy': None,
            'pr_vacancies': None,
            'pr_waitlist': None,
            'pr_section_text': None}


_PR_LABEL_MAP = {
    'Special Types of Care Available': 'pr_special_types_of_care',
    'Program Philosophy': 'pr_program_philosophy',
    'Vacancies': 'pr_vacancies',
    'Waitlist': 'pr_waitlist',
}


def crawl_provider_reported(soup):
    out = empty_provider_reported()
    body = _section_body(soup, HEADING_PROVIDER_REPORTED)
    if body is None:
        return out
    # Each label is an <h3 class="dcf-red-font"> inside its own col-* div; the
    # value is the rest of that column (a table, span, or div).
    for lab in body.select('.dcf-red-font'):
        label = _clean(lab.get_text())
        key = _PR_LABEL_MAP.get(label)
        if not key:
            continue
        col = lab.parent
        full = _clean(col.get_text(' ')) or ''
        val = _clean(full[len(label):]) if label and full.startswith(label) else full
        out[key] = val or None
    out['pr_section_text'] = _clean(body.get_text(' '))
    return out


DOC_PATTERNS = ('ViewMonitoringDocument', 'ViewRatingReport', 'ViewDocument')


def empty_documents():
    return {'num_documents': None, 'documents_json': None}


def collect_documents(soup):
    out = empty_documents()
    seen, docs = set(), []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if any(p in href for p in DOC_PATTERNS):
            if href in seen:
                continue
            seen.add(href)
            url = href if href.startswith('http') else BASE_URL + href
            kind = ('rating' if 'Rating' in href
                    else 'monitoring' if 'Monitoring' in href else 'other')
            docs.append({'label': _clean(a.get_text()), 'type': kind, 'url': url})
            if len(docs) >= MAX_DOCS:
                break
    out['num_documents'] = len(docs)
    out['documents_json'] = json.dumps(docs, ensure_ascii=False) if docs else None
    return out


def download_documents(page, rec, downloads_folder, docs):
    """Fetch each document URL in a fresh tab using the live session
    (--download-pdfs only)."""
    if not docs:
        return
    folder = os.path.join(downloads_folder,
                          f"{rec['provider_number']}_{rec['location_number']}")
    os.makedirs(folder, exist_ok=True)
    for i, d in enumerate(docs, 1):
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', (d.get('label') or f'doc{i}'))[:60]
        dest = os.path.join(folder, f'{i:02d}_{d["type"]}_{safe}.pdf')
        try:
            with page.expect_download(timeout=30000) as dl:
                page.evaluate("(u) => window.open(u, '_blank')", d['url'])
            dl.value.save_as(dest)
            time.sleep(0.4)
        except Exception:
            log(f'    ! doc download failed: {d["url"]}')


def crawler(records, output_csv='wi_data/wi_records.csv',
            downloads_folder='wi_data/downloads', headless=False, start_index=0,
            limit=None, download_pdfs=False, executable_path=None,
            channel='chrome', user_data_dir='wi_data/wi_profile',
            delay_range=(5, 10), retry_errors=False, dry_run=False):
    rows = []
    on_disk = _existing(output_csv)
    completed = {k for k, err in on_disk.items() if not err}
    failed = {k for k, err in on_disk.items() if err}
    if on_disk:
        log(f'{len(on_disk)} row(s) already in {output_csv}: '
            f'{len(completed)} complete, {len(failed)} carrying an error.')
    if retry_errors:
        wanted = failed
        records = [r for r in records if r['provider_location'] in wanted]
        log(f'--retry-errors: {len(records)} of {len(wanted)} failed key(s) '
            f'found in the seed.')
        missing = wanted - {r['provider_location'] for r in records}
        if missing:
            log(f'  ! {len(missing)} failed key(s) are not in this seed and '
                f'cannot be retried: {sorted(missing)[:10]}')
        start_index = 0
        if limit is not None:
            records = records[:limit]
            log(f'  --limit {limit}: visiting the first {len(records)}.')
            limit = None
        if dry_run:
            for i, r in enumerate(records):
                log(f'  [{i}] would retry {r["provider_location"]} '
                    f'(errors={on_disk[r["provider_location"]]!r})')
            log(f'dry run: {len(records)} provider(s) would be requested; '
                f'nothing was written.')
            return pd.DataFrame(rows), start_index
    elif dry_run:
        todo = [r for r in records[start_index:]
                if r['provider_location'] not in completed]
        log(f'dry run: {len(todo)} provider(s) would be requested '
            f'({len(failed)} of them a retry of a failed row); nothing was written.')
        return pd.DataFrame(rows), start_index
    end = len(records) if limit is None else min(len(records), start_index + limit)

    # Rows already in the output are replaced at their own position, never
    # appended; rewrite after each one so an interrupted run keeps its progress.
    pending = {}

    def flush():
        if pending:
            n = replace_rows(output_csv, pending)
            log(f'  ...rewrote {n} row(s) in place')
            pending.clear()

    # Persistent profile: a fresh context has no reCAPTCHA v3 trust, scores
    # like a bot, and the provider data never renders. If a dedicated profile
    # still scores too low, point --user-data-dir at a COPY of a real Chrome
    # profile (quit Chrome first; it locks the profile).
    os.makedirs(user_data_dir, exist_ok=True)

    with sync_playwright() as p:
        launch_kwargs = {
            'user_data_dir': user_data_dir,
            'headless': headless,
            'viewport': {'width': 1400, 'height': 1000},
            'args': ['--disable-blink-features=AutomationControlled'],
        }
        # A real Chrome channel presents a self-consistent UA + client hints;
        # spoofing only the UA string is itself a detection tell.
        if executable_path:
            launch_kwargs['executable_path'] = executable_path
            launch_kwargs['user_agent'] = UA
        elif channel:
            launch_kwargs['channel'] = channel
        else:
            launch_kwargs['user_agent'] = UA
        context = p.chromium.launch_persistent_context(**launch_kwargs)
        # Hide the automation flag reCAPTCHA keys on (navigator.webdriver=true).
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        # Warm-up so reCAPTCHA v3 runs before any detail page is requested.
        try:
            warm = context.new_page()
            warm.goto(BASE_URL, wait_until='domcontentloaded')
            time.sleep(3)
            warm.close()
        except Exception:
            log('  ! warm-up visit failed (continuing anyway)')

        index = start_index
        try:
            for index in range(start_index, end):
                rec = records[index]
                key = rec['provider_location']
                if key in completed:
                    continue

                row = {'provider_location': key,
                       'provider_number': rec['provider_number'],
                       'location_number': rec['location_number'],
                       'facility_number': rec.get('facility_number'),
                       'regulation_type': rec.get('regulation_type'),
                       **seed_carry(rec)}
                errors = []
                # New page, not a new context, so reCAPTCHA trust carries over.
                page = context.new_page()
                try:
                    url = goto_provider(page, rec)
                    row['provider_url'] = url
                    if not url:
                        row.update(empty_youngstar() | empty_regulation()
                                   | empty_provider_reported() | empty_documents())
                        row['errors'] = classify_load_failure(page)
                        log(f'[{index}] {row["errors"].upper()}: {key}')
                    else:
                        soup = BeautifulSoup(page.content(), 'html.parser')
                        for name, fn, empty_fn in (
                            ('youngstar', crawl_youngstar, empty_youngstar),
                            ('regulation', crawl_regulation, empty_regulation),
                            ('provider_reported', crawl_provider_reported, empty_provider_reported),
                            ('documents', collect_documents, empty_documents),
                        ):
                            try:
                                row.update(fn(soup))
                            except Exception:
                                errors.append(name)
                                row.update(empty_fn())
                                log(f'  ! {name} failed for {key}: '
                                    f'{traceback.format_exc().splitlines()[-1]}')

                        if download_pdfs and row.get('documents_json'):
                            try:
                                download_documents(page, rec, downloads_folder,
                                                   json.loads(row['documents_json']))
                            except Exception:
                                errors.append('download')

                        row['errors'] = ','.join(errors)
                        log(f"[{index}] {'OK' if not errors else 'PARTIAL'}: {key} "
                            f"({row.get('youngstar_star_rating')}★)")

                    rows.append(row)
                    if key in on_disk:
                        pending[key] = row
                        flush()
                    else:
                        append_row(row, output_csv)
                    on_disk[key] = row.get('errors') or ''
                    if not row.get('errors'):
                        completed.add(key)
                except Exception:
                    log(f'[{index}] EXCEPTION: {key}')
                    log(traceback.format_exc())
                    row.update(empty_youngstar() | empty_regulation()
                               | empty_provider_reported() | empty_documents())
                    row['errors'] = 'exception'
                    rows.append(row)
                    if key in on_disk:
                        pending[key] = row
                        flush()
                    else:
                        append_row(row, output_csv)
                    on_disk[key] = 'exception'
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass

                if index < end - 1:
                    delay = random.uniform(*delay_range)
                    log(f'  ...sleeping {delay:.1f}s')
                    time.sleep(delay)
        except Exception:
            log(f'CRASHED at index {index}')
            log(traceback.format_exc())
        finally:
            flush()
            context.close()
    if retry_errors:
        still = sum(1 for r in rows if r.get('errors'))
        log(f'--retry-errors: retried {len(rows)}, recovered {len(rows) - still}, '
            f'still failing {still}.')
    return pd.DataFrame(rows), index


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='WI Child Care Finder crawler.')
    ap.add_argument('--seed', default='wi_data/wi_seed.csv')
    ap.add_argument('--output', default='wi_data/wi_records.csv')
    ap.add_argument('--downloads', default='wi_data/downloads')
    ap.add_argument('--headless', action='store_true')
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--download-pdfs', action='store_true')
    ap.add_argument('--executable-path', default=None)
    ap.add_argument('--user-data-dir', default='wi_data/wi_profile',
                    help='Persistent Chrome profile dir (keeps reCAPTCHA trust '
                         'across the run). Point at a COPY of your real Chrome '
                         'profile if a dedicated one still scores too low.')
    ap.add_argument('--channel', default='chrome',
                    help="Browser channel: 'chrome' (recommended), 'chrome-beta', "
                         "'msedge', or '' to use Playwright's bundled Chromium.")
    ap.add_argument('--delay-min', type=float, default=5)
    ap.add_argument('--delay-max', type=float, default=10)
    ap.add_argument('--retry-errors', action='store_true',
                    help='Visit only the providers whose existing row carries a '
                         'non-empty errors cell, and overwrite those rows in '
                         'place. Row count and row order are preserved.')
    ap.add_argument('--dry-run', action='store_true',
                    help='List the providers that would be requested and exit '
                         'without opening a browser or writing anything.')
    args = ap.parse_args()

    create_log_file()
    seed = load_seed(args.seed)
    records = seed.to_dict('records')
    log(f'Seed: {len(records)} locations '
        f'({(seed.regulation_type=="licensed").sum()} licensed, '
        f'{(seed.regulation_type=="certified").sum()} certified).')
    if REQUIRES_SEARCH_SESSION:
        log('NOTE: REQUIRES_SEARCH_SESSION is on — establish_search_session() must be filled.')
    crawler(records, output_csv=args.output, downloads_folder=args.downloads,
            headless=args.headless, start_index=args.start_index, limit=args.limit,
            download_pdfs=args.download_pdfs, executable_path=args.executable_path,
            channel=(args.channel or None), user_data_dir=args.user_data_dir,
            delay_range=(args.delay_min, args.delay_max),
            retry_errors=args.retry_errors, dry_run=args.dry_run)