"""
ga_crawler.py — Georgia (DECAL) child care crawler.

Seeds from the provided export (ga_data/ga_seed.csv), and for each provider
drives the DECAL families portal (families.decal.ga.gov/ChildCare/Detail/<n>)
with Playwright to pull the facility detail panel plus the separate compliance
page (Provider/Details/<n>), then merges in the seed's columns (which carry the
qr_rating target). One record per provider.

When the portal redirects away from a direct URL, the provider is looked up via
the search box instead. Each section runs under its own try/except; failures
land in the `errors` column. Resume-safe: re-running skips provider_ids already
in the output CSV.

The rates-table and compliance sections emit provider-specific columns, so rows
are buffered and flushed with a column-union rewrite every --flush-every
providers, letting the schema grow as new columns are discovered.

    python ga_crawler.py --limit 5                      # quick test (visible browser)
    python ga_crawler.py --headless                     # full run

Deps: pip install playwright pandas && playwright install chromium
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import traceback

import numpy as np
import pandas as pd

PROVIDER_BASE_URL = 'https://families.decal.ga.gov/ChildCare/Detail/'
SEARCH_URL = 'https://families.decal.ga.gov/ChildCare/Search'

IDS_FILE = 'ids.json'
LOG_FILE = 'ga_crawler_log.txt'
PROFILE_DIR = 'ga_data/ga_profile'

# the seed also supplies the merge columns (qr_rating, provider_type, region, ...)
DEFAULT_SEED = 'ga_data/ga_seed.csv'


def create_log_file(path=LOG_FILE):
    if os.path.exists(path):
        os.remove(path)
    open(path, 'w').close()


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)


class PortalFetcher:
    """One persistent browser context; returns the live page for the section
    extractors to query."""

    def __init__(self, headless=True, user_data_dir=PROFILE_DIR, nav_pause=1.0):
        from playwright.sync_api import sync_playwright
        os.makedirs(user_data_dir, exist_ok=True)
        self._pw = sync_playwright().start()
        self.context = self._pw.chromium.launch_persistent_context(
            user_data_dir=user_data_dir, headless=headless,
            viewport={'width': 1400, 'height': 1000},
            args=['--disable-blink-features=AutomationControlled'])
        self._pg = None
        self.nav_pause = nav_pause

    def _page(self):
        if self._pg is None or self._pg.is_closed():
            self._pg = self.context.new_page()
        return self._pg

    def open_detail(self, provider_id):
        """Returns (page, url, found); found=False when neither the direct URL
        nor the search box resolved a single matching provider."""
        page = self._page()
        provider_url = PROVIDER_BASE_URL + provider_id.split('-')[1]
        page.goto(provider_url)
        time.sleep(self.nav_pause)
        if page.url != provider_url:
            page.goto(SEARCH_URL)
            time.sleep(self.nav_pause)
            found_url = find_url(page, provider_id)
            if found_url is None:
                return page, None, False
            page.goto(PROVIDER_BASE_URL + found_url)
            time.sleep(self.nav_pause)
        return page, page.url, True

    def open_compliance(self, provider_id):
        """Compliance page (Provider/Details/<n>). Returns (page, url, found)."""
        page = self._page()
        provider_url = PROVIDER_BASE_URL + provider_id.split('-')[1]
        compliance_url = provider_url.replace('ChildCare/Detail', 'Provider/Details')
        page.goto(compliance_url)
        time.sleep(self.nav_pause)
        if page.url != compliance_url:
            page.goto(SEARCH_URL)
            time.sleep(self.nav_pause)
            found_url = find_url(page, provider_id)
            if found_url is None:
                return page, None, False
            page.goto(provider_url.replace('ChildCare/detail', 'Provider/Details') + found_url)
            time.sleep(self.nav_pause)
            if page.url != compliance_url:
                return page, None, False
        return page, page.url, True

    def close(self):
        try:
            self.context.close()
        finally:
            self._pw.stop()


def find_url(page, provider_id):
    """Search by provider_id in the portal's location-name box; return the
    detail href tail iff exactly one provider matches, else None."""
    page.fill('input[id="Content_Main_ProviderSearch_txtLocationName"]', provider_id)
    page.click('input[id="Content_Main_ProviderSearch_btnSearch"]')
    time.sleep(1)
    if page.locator('p[id="lblTotalRecords"]').inner_text() == '':
        return None
    elif page.locator('p[id="lblTotalRecords"]').inner_text().split(' ')[-1] != '1':
        return None
    else:
        view_button = page.locator('a[class="lId button btn green btn-block no-print track-action"]')
        href = view_button.get_attribute('href').strip()
    return href.split('/')[1]


def create_empty_crawled_span_row(html_ids_dict):
    return {k: None for k in html_ids_dict.values()}


def crawl_span(page, html_ids_dict):
    row = create_empty_crawled_span_row(html_ids_dict)
    for html_id, column in html_ids_dict.items():
        element = page.query_selector(f'#{html_id}')
        if element is None:
            continue
        text = element.inner_text().strip().strip(',').replace('\n', '\t')
        row[column] = text
    return row


def create_empty_crawled_checkmark_row(html_ids_dict):
    return {k: None for k in html_ids_dict.values()}


def crawl_checkmarks(page, html_ids_dict):
    row = create_empty_crawled_checkmark_row(html_ids_dict)
    for html_id, column in html_ids_dict.items():
        element = page.query_selector(f'#{html_id}')
        if element is None:
            continue
        row[column] = bool(element.get_attribute('checked'))
    return row


def create_empty_crawled_list_row(html_ids_dict):
    return {k: None for k in html_ids_dict.values()}


def crawl_list(page, html_ids_dict):
    row = create_empty_crawled_list_row(html_ids_dict)
    for html_id, column in html_ids_dict.items():
        element = page.query_selector(f'#{html_id}')
        if element is None:
            continue
        s = ''
        ul = element.query_selector('ul')
        if ul is None:
            continue
        lis = ul.query_selector_all('li')
        for li in lis:
            s += li.inner_text().strip() + '\t'
        s = s.strip('\t')
        row[column] = s
    return row


def create_empty_crawled_program_type_row():
    return {"program_type": None, "program_subtype": None}


def crawl_program_type(page, html_id='Content_Main_lblProgramType'):
    row = create_empty_crawled_program_type_row()
    element = page.query_selector(f'#{html_id}')
    if element is None:
        return row
    row['program_type'] = element.inner_text().strip().replace('\n', '\t')

    subtype_elements = element.query_selector_all('div')
    s = ''
    for subtype_element in subtype_elements:
        i = subtype_element.query_selector('i')
        s += i.inner_text().strip() + '\t'
    s = s.strip('\t').replace('\n', '\t')
    row['program_subtype'] = s
    return row


def create_empty_crawled_rates_table_row():
    return {}


def crawl_rates_table(page, html_id='Content_Main_gvFacilityRates'):
    weekly_rates_data = create_empty_crawled_rates_table_row()

    weekly_rates_container = page.query_selector(f'#{html_id}')
    if weekly_rates_container is None:
        return weekly_rates_data

    trs = weekly_rates_container.query_selector_all("tr")
    for j in range(1, len(trs)):
        tr = trs[j]
        tds = tr.query_selector_all("td")
        row = tds[0].inner_text().strip().lower().replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_').replace('&', 'and')
        for k in range(1, len(tds)):
            col = tds[k].query_selector("span").inner_text().strip().lower().replace(' ', '_').replace('/', '_')[:-1]
            div = tds[k].query_selector("div")
            if div is None:
                weekly_rates_data[col + '_' + row] = None
            else:
                weekly_rates_data[col + '_' + row] = div.inner_text().strip()
    return weekly_rates_data


def create_empty_crawled_downloads_row():
    return {"num_downloadable_files": None, "download_path": None}


def crawl_pdfs(page, provider_id, downloads_folder):
    links = page.locator('a[href^="javascript:__doPostBack"]')
    num_downloadable_files = links.count()
    if num_downloadable_files == 0:
        return {"num_downloadable_files": 0, "download_path": None}

    for i in range(links.count()):
        download_path = downloads_folder + str(provider_id) + '/'
        link = links.nth(i)
        href = (link.get_attribute("href") or "")
        if "Content_Main" not in href:
            continue
        elif "Report" in href:
            tr = link.locator('xpath=ancestor::tr[1]')
            tds = tr.locator('xpath=./td')
            report_date = tds.nth(1).inner_text().replace(" ", "_").replace(",", "")
            report_type = tds.nth(4).inner_text().replace(" ", "_")
            download_path += report_date + '_' + report_type + '.pdf'
        elif "Enforcement" in href:
            tr = link.locator('xpath=ancestor::tr[1]')
            tds = tr.locator('xpath=./td')
            report_date = tds.nth(3).inner_text().replace(" ", "_").replace(",", "")
            report_type = tds.nth(1).inner_text().replace(' ', '_')
            download_path += report_date + '_' + report_type + '.pdf'
        else:
            tr = link.locator('xpath=ancestor::tr[1]')
            tds = tr.locator('xpath=./td')
            for j in range(tds.count()):
                download_path += tds.nth(j).inner_text() + '_'
            download_path += '.pdf'

        with page.expect_download() as dl:
            link.click()
        download = dl.value
        download.save_as(download_path)
        time.sleep(0.2)

    return {"num_downloadable_files": num_downloadable_files,
            "download_path": downloads_folder + str(provider_id) + '/'}


def create_empty_crawled_compliance_row():
    return {}


def crawl_compliance(page):
    compliance_data = create_empty_crawled_compliance_row()

    year = 0
    while True:
        year += 1
        div_id = f'Content_Main_idYear{year}'
        if page.query_selector(f"#{div_id}") is None:
            break
        if div_id == 'Content_Main_idYear1':
            div = page.query_selector(f"#{div_id}").query_selector_all(":scope > div")[1].query_selector(":scope > div")
        else:
            div = page.query_selector(f"#{div_id}").query_selector_all(":scope > div")[1]

        year = int(page.query_selector(f"#{div_id}").query_selector_all(":scope > div")[0].query_selector_all("span")[1].inner_text().strip())
        suffix = f"{year}_compliance_"

        inspection_rules_met_row = div.query_selector_all(":scope > div")[0]
        inspection_rules_met_ratio = inspection_rules_met_row.query_selector_all(":scope > div")[0].inner_text().strip().split('/')
        total_rules_met = int(inspection_rules_met_ratio[0])
        total_rules_total = int(inspection_rules_met_ratio[1])

        rule_violation_rows = div.query_selector_all(":scope > div")[1]
        rule_violations = int(rule_violation_rows.query_selector_all("div")[0].query_selector(":scope > div").inner_text().strip())
        state_avg_rule_violations = rule_violation_rows.query_selector_all(":scope > div")[1].query_selector("span").inner_text().strip()
        state_avg_rule_violations = np.nan if state_avg_rule_violations == '' else int(state_avg_rule_violations)

        compliance_data[suffix + 'total_rule_violations'] = rule_violations
        compliance_data[suffix + 'total_rules_met'] = total_rules_met

        trs = div.query_selector_all(":scope > div")[2].query_selector_all("tr")
        for tr in trs[1:]:
            tds = tr.query_selector_all("td")
            rule_name = tds[0].inner_text().strip().lower().replace(' ', '_').replace('&', 'and').replace('/', '_').split(":_")[1]
            ratio = tds[2].inner_text().strip().split(": ")[1].split(' of ')
            rules_met = int(ratio[0])
            rules_total = int(ratio[1])

            met_column = suffix + rule_name + '_rules_met'
            compliance_data[met_column] = rules_met

            total_column = suffix + rule_name + '_rules_total'
            compliance_data[total_column] = rules_total

    compliance_img = page.query_selector("#Content_Main_imgCompliance")
    compliance = compliance_img.get_attribute('src').split('/')[-1].split('_FINAL.png')[0]
    compliance_data['compliance'] = compliance

    return compliance_data


def load_seed(seed_csv, additional_map):
    """Load the seed export and rename its columns via the ids.json
    `additional_columns` map (Provider_Number -> provider_id, QR_Rating ->
    qr_rating, ...)."""
    if not seed_csv or not os.path.exists(seed_csv):
        raise FileNotFoundError(f'Seed CSV not found: {seed_csv}')
    df = pd.read_csv(seed_csv, low_memory=False)
    df = df.rename(columns=additional_map)
    if 'provider_id' not in df.columns:
        raise ValueError(
            f'Seed needs a provider_id column after rename; got {list(df.columns)}')
    df['provider_id'] = df['provider_id'].astype(str).str.strip()
    df = (df[df['provider_id'].notna() & (df['provider_id'] != '')]
          .drop_duplicates(subset='provider_id').reset_index(drop=True))
    keep = [c for c in additional_map.values() if c in df.columns]
    return df[keep]


def load_completed(output_csv):
    if not output_csv or not os.path.exists(output_csv):
        return set()
    try:
        df = pd.read_csv(output_csv, usecols=['provider_id'], low_memory=False)
        return set(df['provider_id'].dropna().astype(str).str.strip())
    except Exception as e:
        log(f'Could not read existing {output_csv}: {e}')
        return set()


def flush_rows(rows, output_csv):
    """Concat the buffered rows onto the CSV on disk (column union, missing
    cells NaN) and rewrite it."""
    if not rows:
        return
    parent = os.path.dirname(output_csv)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    new = pd.DataFrame(rows)
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        existing = pd.read_csv(output_csv, low_memory=False)
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset='provider_id', keep='first')
    combined.to_csv(output_csv, index=False)


def crawler(seed_df, crawled_columns, output_csv='ga_data/ga_records.csv',
            headless=True, start_index=0, limit=None, delay_range=(2, 4),
            flush_every=25, download_pdfs=False,
            downloads_folder='ga_data/downloads/'):
    span_ids = crawled_columns['spans']
    check_ids = crawled_columns['checkmarks']
    list_ids = crawled_columns['lists']
    additional_records = seed_df.set_index('provider_id', drop=False).to_dict('index')
    provider_ids = seed_df['provider_id'].tolist()

    completed = load_completed(output_csv)
    if completed:
        log(f'Resuming: {len(completed)} already in {output_csv} — skipping.')
    end = len(provider_ids) if limit is None else min(len(provider_ids),
                                                       start_index + limit)

    fetcher = PortalFetcher(headless=headless)
    buffer = []
    index = start_index
    try:
        for index in range(start_index, end):
            provider_id = str(provider_ids[index])
            if provider_id in completed:
                continue

            row = {'provider_id': provider_id}
            errors = []

            try:
                page, url, found = fetcher.open_detail(provider_id)
                row['provider_url'] = url
                if not found:
                    row.update(create_empty_crawled_span_row(span_ids))
                    row.update(create_empty_crawled_checkmark_row(check_ids))
                    row.update(create_empty_crawled_list_row(list_ids))
                    row.update(create_empty_crawled_program_type_row())
                    row.update(create_empty_crawled_rates_table_row())
                    row.update(create_empty_crawled_downloads_row())
                    row.update(create_empty_crawled_compliance_row())
                    errors.append('provider_page')
                else:
                    for name, fn in (
                        ('spans', lambda: crawl_span(page, span_ids)),
                        ('checkmarks', lambda: crawl_checkmarks(page, check_ids)),
                        ('lists', lambda: crawl_list(page, list_ids)),
                        ('program_type', lambda: crawl_program_type(page)),
                        ('rates_table', lambda: crawl_rates_table(page)),
                    ):
                        try:
                            row.update(fn())
                        except Exception:
                            errors.append(name)
                            row.update(_empty_for(name, span_ids, check_ids, list_ids))
                            log(f'  ! {name} failed for {provider_id}: '
                                f'{traceback.format_exc().splitlines()[-1]}')

                    if download_pdfs:
                        try:
                            row.update(crawl_pdfs(page, provider_id, downloads_folder))
                        except Exception:
                            errors.append('downloads')
                            row.update(create_empty_crawled_downloads_row())
                    else:
                        row.update(create_empty_crawled_downloads_row())

                    cpage, curl, cfound = fetcher.open_compliance(provider_id)
                    if not cfound:
                        errors.append('compliance')
                        row.update(create_empty_crawled_compliance_row())
                    else:
                        try:
                            row.update(crawl_compliance(cpage))
                            row['compliance_url'] = curl
                        except Exception:
                            errors.append('compliance')
                            row.update(create_empty_crawled_compliance_row())

                # merge seed columns (carries qr_rating)
                extra = additional_records.get(provider_id, {})
                for k, v in extra.items():
                    if k == 'provider_id':
                        continue
                    row[k] = v

                row['errors'] = ','.join(errors)
                tag = 'OK' if not errors else 'PARTIAL'
                log(f'[{index}] {tag}: {provider_id}')
                buffer.append(row)
                completed.add(provider_id)

            except Exception:
                log(f'[{index}] EXCEPTION: {provider_id}')
                log(traceback.format_exc())
                row['errors'] = 'exception'
                extra = additional_records.get(provider_id, {})
                for k, v in extra.items():
                    if k != 'provider_id':
                        row[k] = v
                buffer.append(row)
                completed.add(provider_id)

            if len(buffer) >= flush_every:
                flush_rows(buffer, output_csv)
                log(f'  ...flushed {len(buffer)} rows → {output_csv}')
                buffer = []

            if index < end - 1:
                delay = random.uniform(*delay_range)
                time.sleep(delay)
    except Exception:
        log(f'CRASHED at index {index}')
        log(traceback.format_exc())
    finally:
        flush_rows(buffer, output_csv)
        fetcher.close()
        log(f'Done. Output: {output_csv}')


def _empty_for(name, span_ids, check_ids, list_ids):
    """Empty dict for a failed detail section."""
    if name == 'spans':
        return create_empty_crawled_span_row(span_ids)
    if name == 'checkmarks':
        return create_empty_crawled_checkmark_row(check_ids)
    if name == 'lists':
        return create_empty_crawled_list_row(list_ids)
    if name == 'program_type':
        return create_empty_crawled_program_type_row()
    if name == 'rates_table':
        return create_empty_crawled_rates_table_row()
    return {}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Georgia (DECAL) child care crawler.')
    ap.add_argument('--seed', default=DEFAULT_SEED,
                    help='Seed CSV (also supplies merge columns).')
    ap.add_argument('--ids', default=IDS_FILE, help='ids.json field map.')
    ap.add_argument('--output', default='ga_data/ga_records.csv')
    ap.add_argument('--headless', action='store_true',
                    help='Run the browser headless.')
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--delay-min', type=float, default=2)
    ap.add_argument('--delay-max', type=float, default=4)
    ap.add_argument('--flush-every', type=int, default=25)
    ap.add_argument('--download-pdfs', action='store_true',
                    help='Also download report PDFs.')
    args = ap.parse_args()

    create_log_file()
    with open(args.ids, 'r') as f:
        ids_dict = json.load(f)
    crawled_columns = ids_dict['crawled_columns']
    additional_map = ids_dict['additional_columns']

    seed_df = load_seed(args.seed, additional_map)
    log(f'Seed: {len(seed_df)} providers.')
    crawler(seed_df, crawled_columns, output_csv=args.output,
            headless=args.headless, start_index=args.start_index,
            limit=args.limit, delay_range=(args.delay_min, args.delay_max),
            flush_every=args.flush_every, download_pdfs=args.download_pdfs)