#!/usr/bin/env python3
"""
nc_data_correction.py -- stage-1 repairs for the North Carolina records.

Every repair re-derives a value from the page text already saved on disk
(`star_section_text` holds the first 8,000 characters of the rendered facility
page); nothing here needs the network.

The page-text parsers at the top are imported by `nc_crawler.py`, so a fresh
crawl reads the same values at the source. The driver at the bottom applies
them to records already collected.

    python nc_data_correction.py --check              # report, write nothing
    python nc_data_correction.py --fix-columns
    python nc_data_correction.py --fix-facility-type
    python nc_data_correction.py --fix-restrictions
    python nc_data_correction.py --fix-scores
    python nc_data_correction.py --fix-labels
    python nc_data_correction.py --strip-stage2-text
    python nc_data_correction.py --all                # all of the above, in order

`nc_data/nc_records.csv` and `nc_data/nc_records_anonymized.csv` are patched in
place, row for row, in lockstep: the two files are joined positionally, so the
row count and order never change. Both are streamed with the `csv` module,
written to a temp file and `os.replace()`d, so an interrupted run cannot leave
a half-written file. The anonymized file is patched rather than rebuilt because
re-running `nc_anonymize.py` draws a fresh provider id permutation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

HERE = Path(__file__).resolve().parent
RECORDS = HERE / "nc_data" / "nc_records.csv"
ANON = HERE / "nc_data" / "nc_records_anonymized.csv"
LOG_FILE = HERE / "nc_data_correction_log.txt"

# Whole-page dumps: the evidence the repairs are derived from. They stay in the
# stage-1 records file only.
PAGE_TEXT_COLS = ("star_section_text", "visits_section_text",
                  "details_section_text", "operating_hours")


# ---------------------------------------------------------------------------
# page-text parsers -- also imported by nc_crawler.py
# ---------------------------------------------------------------------------

# "Facility/Program Type: Child Care Center Phone: (919) ..." An unrecognised
# label yields None, so a label change is counted rather than written wrong.
TYPE_RE = re.compile(
    r'Facility/Program Type:\s*(.*?)\s+(?:Phone:|Email:|Website:|$)')
TYPE_MAP = {'Child Care Center': 'CCC', 'Family Child Care Home': 'FCC'}

# The page lists the current licence first, then every previous one; a
# first-match read falls through to a previous licence whenever the current one
# omits a field. These delimiters bound the current licence's block.
CUR_START = 'Current License Details'
CUR_ENDS = ('Previous License Details',
            'New Centers and Family Child Care Homes must wait',
            'Facility Special Features')

RESTRICTION_RE = re.compile(r'License Restrictions:\s*(.*)$')
STAR_BLOCK_LABEL = 'Star Rating Information'
TOTAL_RE = re.compile(r'Total Points:\s*(\d+)\s+out of\s+(\d+)')
PROGRAM_RE = re.compile(
    r'Program Standards Points Earned:\s*(\d+)\s+out of\s+(\d+)')
EDUCATION_RE = re.compile(
    r'Educational Standards Points Earned:\s*(\d+)\s+out of\s+(\d+)')

NAME_RE = re.compile(r'Facility Name:\s*(.*?)\s+Address:')
COUNTY_RE = re.compile(r'\s([A-Z][A-Za-z.]+(?: [A-Z][A-Za-z.]+)?) County Email:')

# Fields the current-licence block must reproduce, used to prove the block
# boundaries are right before anything is written.
LICENSE_TYPE_RE = re.compile(r'License Type:\s*(.*?)\s+Effective Date:')
EFFECTIVE_RE = re.compile(r'Effective Date:\s*(\S+)')
AGE_RANGE_RE = re.compile(r'Age Range:\s*(.*?)\s+(?:Star Rating Information|'
                          r'Approved Capacity|License Restrictions:|$)')
SHIFT_RE = re.compile(r'(?:1st|2nd|3rd) Shift:\s*(\d+)')


def parse_facility_type(page_text):
    """'CCC' | 'FCC' | None -- the page's own Facility/Program Type label.

    None means either no page text (a not_found row) or a label this mapping
    does not know; the caller must leave the existing value alone in both cases.
    """
    if not page_text:
        return None
    m = TYPE_RE.search(page_text)
    if not m:
        return None
    return TYPE_MAP.get(m.group(1).strip())


def facility_type_label(page_text):
    """The raw Facility/Program Type label, for logging unknown values."""
    if not page_text:
        return None
    m = TYPE_RE.search(page_text)
    return m.group(1).strip() if m else None


def current_licence_block(page_text):
    """The slice of the page text describing the CURRENT licence only."""
    if not page_text:
        return None
    i = page_text.find(CUR_START)
    if i < 0:
        return None
    seg = page_text[i:]
    ends = [seg.find(x) for x in CUR_ENDS]
    ends = [x for x in ends if x > 0]
    return seg[:min(ends)] if ends else seg


def parse_restrictions(page_text):
    """The current licence's restriction text, or None. Read verbatim from the
    current block: the restriction sentences themselves contain '; '."""
    cur = current_licence_block(page_text)
    if cur is None:
        return None
    m = RESTRICTION_RE.search(cur)
    if not m:
        return None
    return m.group(1).strip() or None


def parse_scores(page_text):
    """The current licence's rubric points, or None when it has no Star block."""
    cur = current_licence_block(page_text)
    if cur is None or STAR_BLOCK_LABEL not in cur:
        return None
    total = TOTAL_RE.search(cur)
    program = PROGRAM_RE.search(cur)
    education = EDUCATION_RE.search(cur)
    return {
        'total_score': total.group(1) if total else None,
        'program_points': program.group(1) if program else None,
        'program_max': program.group(2) if program else None,
        'education_points': education.group(1) if education else None,
        'education_max': education.group(2) if education else None,
    }


def empty_scores():
    """The all-null rubric an unrated current licence should carry."""
    return {'total_score': None, 'program_points': None, 'program_max': None,
            'education_points': None, 'education_max': None}


def parse_facility_name(page_text):
    """The facility name as the page showed it at crawl time."""
    if not page_text:
        return None
    m = NAME_RE.search(page_text)
    return (m.group(1).strip() or None) if m else None


def parse_county(page_text):
    """The county name (without the word 'County')."""
    if not page_text:
        return None
    m = COUNTY_RE.search(page_text)
    return (m.group(1).strip() or None) if m else None


def parse_current_licence_fields(page_text):
    """license_type / effective date / age range / capacity from the current
    block. Used only by --check to prove the block boundaries; never written."""
    cur = current_licence_block(page_text)
    if cur is None:
        return None
    lt = LICENSE_TYPE_RE.search(cur)
    eff = EFFECTIVE_RE.search(cur)
    ages = AGE_RANGE_RE.search(cur)
    caps = [int(x) for x in SHIFT_RE.findall(cur)]
    caps = [c for c in caps if c]
    return {
        'license_type': lt.group(1).strip() if lt else None,
        'license_issue_date': eff.group(1).strip() if eff else None,
        'ages_served': ages.group(1).strip() if ages else None,
        'licensed_capacity': str(max(caps)) if caps else None,
    }


# ---------------------------------------------------------------------------
# file shape
# ---------------------------------------------------------------------------
# Rows appended with 25 fields under a 28-name header (`operating_hours`,
# `num_documents`, `documents_json` are not written by the crawler) put the page
# text under `operating_hours` and the error value under `details_section_text`.
# Padding the short rows puts every value back under its own name.


def row_error(row):
    """The `errors` value of a records row; both the 28- and the 25-field rows
    END with it."""
    return row[-1]


def pad_records_row(row, n_header):
    """A 25-field row -> the 28 fields its header promises."""
    if len(row) == n_header:
        return row
    if len(row) != n_header - 3:
        raise SystemExit(f'unexpected field count {len(row)} '
                         f'(expected {n_header} or {n_header - 3})')
    # ... special_features | operating_hours | details_section_text |
    #                        num_documents | documents_json | errors
    return row[:-2] + ['', row[-2], '', '', row[-1]]


def realign_anon_row(arow, idx):
    """Move the twin file's tail back under the right names.

    pandas read the short records rows under the 28-name header, so the page
    text arrived in `operating_hours`, the error value in
    `details_section_text`, and `errors` came through empty.
    """
    out = list(arow)
    out[idx['errors']] = arow[idx['details_section_text']]
    out[idx['details_section_text']] = arow[idx['operating_hours']]
    out[idx['operating_hours']] = ''
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def log_lines(lines):
    with open(LOG_FILE, 'a', encoding='utf-8') as fh:
        for line in lines:
            fh.write(line + '\n')


def header_of(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return next(csv.reader(fh))


def read_pairs(records_path, anon_path):
    """Yield (i, records_row, anon_row) with both files read in lockstep."""
    with open(records_path, newline='', encoding='utf-8') as fr, \
            open(anon_path, newline='', encoding='utf-8') as fa:
        rr, ra = csv.reader(fr), csv.reader(fa)
        rh, ah = next(rr), next(ra)
        i = 0
        for row, arow in zip(rr, ra):
            yield i, row, arow
            i += 1
        if next(rr, None) is not None or next(ra, None) is not None:
            raise SystemExit('the two files do not have the same row count; '
                             'the positional join is void')


def run(check, fixes):
    rh, ah = header_of(RECORDS), header_of(ANON)
    ridx = {c: i for i, c in enumerate(rh)}
    aidx = {c: i for i, c in enumerate(ah)}
    n_header = len(rh)

    short_rows = 0
    counts = {k: 0 for k in (
        'rows', 'found', 'not_found', 'pad', 'realign',
        'type_written', 'type_unknown_label', 'type_no_match',
        'restr_changed', 'restr_no_label',
        'scores_cleared', 'scores_rewritten', 'scores_unchanged',
        'scores_no_block_text', 'name_changed', 'county_written',
        'boundary_ok', 'boundary_bad')}
    unknown_labels = {}

    out_r = RECORDS.with_suffix('.csv.tmp')
    out_a = ANON.with_suffix('.csv.tmp')
    writing = not check

    keep_a = [i for i, c in enumerate(ah)
              if not (fixes.get('strip') and c in PAGE_TEXT_COLS)]

    fr_out = fa_out = wr = wa = None
    if writing:
        fr_out = open(out_r, 'w', newline='', encoding='utf-8')
        fa_out = open(out_a, 'w', newline='', encoding='utf-8')
        wr = csv.writer(fr_out, lineterminator='\n')
        wa = csv.writer(fa_out, lineterminator='\n')
        wr.writerow(rh)
        wa.writerow([ah[i] for i in keep_a])

    try:
        for i, row, arow in read_pairs(RECORDS, ANON):
            counts['rows'] += 1
            was_short = len(row) != n_header
            if was_short:
                short_rows += 1
            errors = row_error(row)

            if fixes.get('columns'):
                if was_short:
                    row = pad_records_row(row, n_header)
                    arow = realign_anon_row(arow, aidx)
                    counts['pad'] += 1
                    counts['realign'] += 1
                else:
                    # guard: the 28-field rows never had operating_hours
                    if arow[aidx['operating_hours']] != '':
                        raise SystemExit(
                            f'row {i}: full-width row has a non-empty '
                            f'operating_hours; refusing to touch it')
            elif was_short and (fixes.get('strip')):
                raise SystemExit(
                    'the records file still has short rows -- run '
                    '--fix-columns before --strip-stage2-text, or the twin '
                    "file's tail is dropped under the wrong names")

            page = row[ridx['star_section_text']]
            found = errors == ''
            counts['found' if found else 'not_found'] += 1

            if found:
                # --- 1. facility type from the page's own label -------------
                label = facility_type_label(page)
                code = parse_facility_type(page)
                if label is None:
                    counts['type_no_match'] += 1
                elif code is None:
                    counts['type_unknown_label'] += 1
                    unknown_labels[label] = unknown_labels.get(label, 0) + 1
                else:
                    if row[ridx['facility_type']] != code:
                        counts['type_written'] += 1
                    if fixes.get('facility_type'):
                        row[ridx['facility_type']] = code
                        arow[aidx['facility_type']] = code

                # --- 2. current licence's restrictions only -----------------
                cur = current_licence_block(page)
                if cur is not None and not RESTRICTION_RE.search(cur):
                    counts['restr_no_label'] += 1
                else:
                    new_restr = parse_restrictions(page) or ''
                    if new_restr != row[ridx['license_restrictions']]:
                        counts['restr_changed'] += 1
                    if fixes.get('restrictions'):
                        row[ridx['license_restrictions']] = new_restr
                        arow[aidx['license_restrictions']] = new_restr

                # --- 3. rubric points of the CURRENT licence ----------------
                blob = row[ridx['star_components']]
                try:
                    parsed = json.loads(blob) if blob else None
                except json.JSONDecodeError:
                    parsed = None
                if parsed is None:
                    counts['scores_no_block_text'] += 1
                else:
                    scores = parse_scores(page)
                    if scores is None:
                        if any(parsed.get('scores', {}).get(k) is not None
                               for k in empty_scores()):
                            counts['scores_cleared'] += 1
                        scores = empty_scores()
                    elif scores != parsed.get('scores'):
                        counts['scores_rewritten'] += 1
                    else:
                        counts['scores_unchanged'] += 1
                    if fixes.get('scores'):
                        blob_new = json.dumps(
                            {'scores': scores,
                             'license_history': parsed.get('license_history')},
                            ensure_ascii=False)
                        row[ridx['star_components']] = blob_new
                        arow[aidx['star_components']] = blob_new

                # --- 4. the page's own facility name and county -------------
                name = parse_facility_name(page)
                if name and name != row[ridx['facility_name']]:
                    counts['name_changed'] += 1
                county = parse_county(page)
                if county and county != row[ridx['county']]:
                    counts['county_written'] += 1
                if fixes.get('labels'):
                    if name:
                        row[ridx['facility_name']] = name
                    if county:
                        row[ridx['county']] = county
                        arow[aidx['county']] = county

                # --- boundary proof (never written) -------------------------
                if check:
                    got = parse_current_licence_fields(page)
                    same = got is not None and all(
                        (got[k] or '') == row[ridx[k]]
                        for k in ('license_type', 'license_issue_date',
                                  'ages_served', 'licensed_capacity'))
                    counts['boundary_ok' if same else 'boundary_bad'] += 1

            if writing:
                wr.writerow(row)
                wa.writerow([arow[j] for j in keep_a])
    except BaseException:
        if writing:
            fr_out.close(); fa_out.close()
            for p in (out_r, out_a):
                if p.exists():
                    p.unlink()
        raise

    if writing:
        fr_out.close()
        fa_out.close()
        os.replace(out_r, RECORDS)
        os.replace(out_a, ANON)

    return counts, unknown_labels, short_rows


ORDER = ('columns', 'facility_type', 'restrictions', 'scores', 'labels', 'strip')

TITLES = {
    'columns': 'realign the 25-field rows under the 28-name header',
    'facility_type': 'facility_type from the page label',
    'restrictions': 'license_restrictions from the current licence only',
    'scores': 'star_components.scores from the current licence only',
    'labels': 'facility_name and county from the page',
    'strip': 'drop the whole-page text columns from the twin file',
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='report what would change and exit without writing')
    ap.add_argument('--fix-columns', action='store_true')
    ap.add_argument('--fix-facility-type', action='store_true')
    ap.add_argument('--fix-restrictions', action='store_true')
    ap.add_argument('--fix-scores', action='store_true')
    ap.add_argument('--fix-labels', action='store_true')
    ap.add_argument('--strip-stage2-text', action='store_true')
    ap.add_argument('--all', action='store_true',
                    help='every repair above, in the only order that works')
    args = ap.parse_args()

    fixes = {
        'columns': args.all or args.fix_columns,
        'facility_type': args.all or args.fix_facility_type,
        'restrictions': args.all or args.fix_restrictions,
        'scores': args.all or args.fix_scores,
        'labels': args.all or args.fix_labels,
        'strip': args.all or args.strip_stage2_text,
    }
    if not args.check and not any(fixes.values()):
        ap.error('nothing to do: pass --check, a --fix-* flag, or --all')

    counts, unknown, short_rows = run(args.check,
                                      {} if args.check else fixes)

    # Every repair is COUNTED on every run, whether or not its flag was
    # passed, so the verb has to be per-repair: 'changed' only where the flag
    # actually wrote, 'would change' everywhere else.
    def verb(name):
        return 'changed' if (not args.check and fixes.get(name)) else 'would change'

    lines = [f'== nc_data_correction {"--check" if args.check else ""} =='.strip()]
    lines.append(f'rows {counts["rows"]} (found {counts["found"]}, '
                 f'not_found {counts["not_found"]}); short rows {short_rows}')
    if args.check or not fixes.get('columns'):
        lines.append(f'[columns]   {short_rows} rows would change shape')
    else:
        lines.append(f'[columns]   {counts["pad"]} records rows padded, '
                     f'{counts["realign"]} twin rows realigned')
    lines.append(f'[type]      {counts["type_written"]} cells '
                 f'{verb("facility_type")}; '
                 f'{counts["type_unknown_label"]} unknown label(s), '
                 f'{counts["type_no_match"]} rows with no Type label')
    if unknown:
        lines.append(f'[type]      unknown labels: {unknown}')
    lines.append(f'[restr]     {counts["restr_changed"]} cells '
                 f'{verb("restrictions")}; '
                 f'{counts["restr_no_label"]} current blocks with no '
                 f'restriction label (kept as-is)')
    lines.append(f'[scores]    {counts["scores_cleared"]} rows '
                 f'{verb("scores")} cleared, '
                 f'{counts["scores_rewritten"]} rewritten, '
                 f'{counts["scores_unchanged"]} already correct')
    lines.append(f'[labels]    {counts["name_changed"]} facility_name cells '
                 f'{verb("labels")}, {counts["county_written"]} county cells '
                 f'{verb("labels")}')
    if args.check:
        lines.append(f'[boundary]  current-licence block reproduces the crawl on '
                     f'{counts["boundary_ok"]} rows, disagrees on '
                     f'{counts["boundary_bad"]}')
    else:
        applied = [TITLES[k] for k in ORDER if fixes[k]]
        lines.append('[applied]   ' + '; '.join(applied))
    for line in lines:
        print(line)
    if not args.check:
        log_lines(lines)
    return 0


if __name__ == '__main__':
    sys.exit(main())
