"""
sc_refetch.py — re-pull South Carolina's ABC Quality export and replace
`sc_data/sc_records.csv` with a fresh point-in-time snapshot.

It imports sc_crawler.py and calls its fetchers, so the endpoint, User-Agent,
timeout, inter-county sleep, header validation and merge/dedup rules are shared.
What it adds is a safe way to re-run against a records file already in use:

  * the fetch NEVER writes `sc_data/sc_records.csv`; the sweep lands in a
    sidecar (`sc_data/sc_records_refresh.csv`) plus a per-county cache;
  * the sweep is checkpointed, so a stopped run continues where it left off;
  * a separate `--merge` step diffs the sidecar against the live records file,
    prints what moved, and only then promotes it, keeping a dated backup.

The export is a full-state listing, not an event feed: providers enter it,
leave it and change rating between snapshots, so a refresh is a replacement,
not a patch. Every downstream step has to be re-run afterwards (see sc.md).

USAGE
-----
    # 1. sweep all 46 counties into the sidecar (a few minutes)
    python sc_refetch.py --fetch

    # ... interrupted? continue where it stopped:
    python sc_refetch.py --fetch --resume

    # smoke test against 3 counties, no commitment
    python sc_refetch.py --fetch --limit 3
    python sc_refetch.py --fetch --counties Calhoun,McCormick

    # 2. see what a refresh would do -- writes nothing
    python sc_refetch.py --merge --dry-run

    # 3. promote the sidecar to sc_data/sc_records.csv (keeps a dated backup)
    python sc_refetch.py --merge

Outputs (all under sc_data/):
    county_raw_refresh/<county>.csv   per-county export exactly as served
    sc_refetch_checkpoint.json        sweep progress: resume reads this
    sc_seed_observed.csv              county <select> as the site lists it today
    sc_records_refresh.csv            merged sidecar -- the candidate snapshot
    sc_records_<YYYYMMDD>.bak.csv     the file --merge replaced
    sc_records_source.json            provenance of the file now in place

Deps: the same as sc_crawler.py (requests, pandas, beautifulsoup4).
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

# sc_crawler.py addresses everything relative to sc/, so anchor there first.
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import sc_crawler as crawler          # noqa: E402  (must follow the chdir)

OUT_DIR = crawler.OUT_DIR
RAW_DIR = os.path.join(OUT_DIR, 'county_raw_refresh')

# crawler.raw_path() reads this global at call time, so a refresh never touches
# the crawler's own sc_data/county_raw/.
crawler.RAW_DIR = RAW_DIR
CHECKPOINT = os.path.join(OUT_DIR, 'sc_refetch_checkpoint.json')
SIDECAR = os.path.join(OUT_DIR, 'sc_records_refresh.csv')
OBSERVED_SEED = os.path.join(OUT_DIR, 'sc_seed_observed.csv')
LIVE_RECORDS = crawler.MERGED_PATH
PROVENANCE = os.path.join(OUT_DIR, 'sc_records_source.json')
LOG_FILE = 'sc_refetch_log.txt'

ADVANCED_PATH = '/provider-search/advanced/'   # carries the county <select>

# Three counties failing in a row is a site problem, not a flaky connection.
MAX_CONSECUTIVE_FAILURES = 3

GRAIN = 'Permit Number'
NAME = 'Provider Name'


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as fh:
        fh.write(message + '\n')
    print(message)


# sc_crawler's helpers log through its module-global `log`; rebinding it keeps
# this run's output in this run's log.
crawler.log = log


def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return None
    with open(CHECKPOINT, encoding='utf-8') as fh:
        return json.load(fh)


def save_checkpoint(state):
    """Atomic, because --resume trusts whatever checkpoint it finds."""
    tmp = CHECKPOINT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
    os.replace(tmp, CHECKPOINT)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def observed_counties(session=None):
    """The county <select> on the advanced-search page, or None (never fatal)."""
    session = session or requests
    try:
        from bs4 import BeautifulSoup
        r = session.get(crawler.BASE_URL + ADVANCED_PATH,
                        headers={'User-Agent': crawler.UA},
                        timeout=crawler.REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        for select in soup.find_all('select'):
            name = (select.get('name') or select.get('id') or '').lower()
            if 'county' not in name:
                continue
            values = [o.get('value', '').strip() for o in select.find_all('option')]
            values = [v for v in values if v and v.lower() not in
                      ('', 'all', 'any', '0', '-1', 'select county')]
            if len(values) >= 20:
                return values
    except Exception as exc:
        log(f'  . county <select> check unavailable ({exc})')
    return None


def check_seed(session):
    """Compare the shipped seed against the site's own county list."""
    observed = observed_counties(session)
    if not observed:
        log('  . seed check skipped (county <select> not readable)')
        return
    seed = set(crawler.SC_COUNTIES)
    missing = sorted(set(observed) - seed)      # site has, seed does not
    extra = sorted(seed - set(observed))        # seed has, site does not
    pd.DataFrame({'county': observed}).to_csv(OBSERVED_SEED, index=False)
    if missing or extra:
        log(f'  ! SEED DRIFT: {len(observed)} counties listed on the site vs '
            f'{len(seed)} in sc_data/sc_seed.csv.')
        if missing:
            log(f'    not swept (add to the seed): {missing}')
        if extra:
            log(f'    in the seed but not listed : {extra}')
        log(f'    observed list written to {OBSERVED_SEED}')
    else:
        log(f'  seed confirmed: all {len(observed)} counties on the site are in '
            f'sc_data/sc_seed.csv')


def fetch(counties, resume, delay_range, check_counts=True, skip_seed=False):
    os.makedirs(RAW_DIR, exist_ok=True)
    state = load_checkpoint() if resume else None
    if state is None:
        state = {'started_at': _now(), 'finished_at': None, 'counties': {}}
    done = state['counties']

    pending = [c for c in counties
               if not (resume and c in done
                       and os.path.exists(crawler.raw_path(c)))]
    if resume:
        log(f'resuming: {len(done)} county file(s) already cached, '
            f'{len(pending)} to go')

    session = requests.Session()
    if not skip_seed:
        check_seed(session)

    consecutive_failures = 0
    for i, county in enumerate(pending):
        log(f'[{i + 1}/{len(pending)}] {county}: fetching')
        text = crawler.fetch_county_csv(county, session=session)
        if text is None:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                save_checkpoint(state)
                raise SystemExit(
                    f'stopping: {consecutive_failures} counties in a row failed '
                    f'to fetch. The portal is refusing or down -- leave it alone '
                    f'for a while, then re-run with --fetch --resume. '
                    f'{len(done)} of {len(counties)} counties are cached.')
            continue

        try:
            df = crawler.parse_county_csv(text, county)
        except Exception as exc:
            log(f'  ! {county}: {exc}')
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                save_checkpoint(state)
                raise SystemExit(
                    f'stopping: {consecutive_failures} counties in a row failed '
                    f'to parse -- the export schema may have changed. Inspect '
                    f'{RAW_DIR}/ before re-running.')
            continue
        consecutive_failures = 0

        path = crawler.raw_path(county)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(tmp, path)

        entry = {'rows': len(df), 'fetched_at': _now(), 'sha256': _sha256(path)}
        if check_counts:
            stated = crawler.fetch_county_stated_count(county, session=session)
            entry['stated'] = stated
            if stated is None:
                log(f'  {county}: {len(df)} row(s) (no stated count to cross-check)')
            elif stated != len(df):
                log(f'  ! MISMATCH {county}: HTML says {stated} providers but the '
                    f'CSV export returned {len(df)} row(s) -- inspect {path} '
                    f'before trusting it.')
            else:
                log(f'  {county}: {len(df)} row(s) (confirmed against stated count)')
        else:
            log(f'  {county}: {len(df)} row(s)')

        done[county] = entry
        save_checkpoint(state)

        if i < len(pending) - 1:
            time.sleep(random.uniform(*delay_range))

    cached = [c for c in counties if os.path.exists(crawler.raw_path(c))]
    if len(cached) == len(crawler.SC_COUNTIES):
        state['finished_at'] = _now()
    save_checkpoint(state)

    log('')
    log(f'{len(cached)}/{len(counties)} requested county file(s) cached under '
        f'{RAW_DIR}/')
    missing = [c for c in counties if c not in cached]
    if missing:
        log(f'still missing: {missing}')
    build_sidecar(cached)


def build_sidecar(counties):
    """Merge the cached county exports into the sidecar via sc_crawler.merge."""
    crawler.merge(counties, out_csv=SIDECAR)
    log(f'\nsidecar written: {SIDECAR}')
    log('sc_data/sc_records.csv is UNCHANGED. Review the diff with:')
    log('    python sc_refetch.py --merge --dry-run')


def _key(df):
    """A provider's identity across snapshots: the permit number, else (name,
    street, zip) for exempt providers -- the crawler's dedup key."""
    permit = df[GRAIN].fillna('').astype(str).str.strip()
    fallback = (df[NAME].fillna('').astype(str).str.strip() + '|' +
                df['Street'].fillna('').astype(str).str.strip() + '|' +
                df['Zip'].fillna('').astype(str).str.strip())
    return permit.where(permit != '', 'EXEMPT:' + fallback)


def diff(live, fresh):
    lk, fk = _key(live), _key(fresh)
    live = live.assign(_k=lk).drop_duplicates('_k').set_index('_k')
    fresh = fresh.assign(_k=fk).drop_duplicates('_k').set_index('_k')

    leaving = [k for k in live.index if k not in fresh.index]
    entering = [k for k in fresh.index if k not in live.index]
    common = [k for k in fresh.index if k in live.index]

    columns = [c for c in fresh.columns if c in live.columns]
    a, b = live.loc[common, columns], fresh.loc[common, columns]
    moved, examples = {}, {}
    for column in columns:
        x = a[column].fillna('').astype(str).str.strip()
        y = b[column].fillna('').astype(str).str.strip()
        changed = x != y
        n = int(changed.sum())
        if n:
            moved[column] = n
            examples[column] = [(k, x[k], y[k]) for k in x.index[changed][:3]]
    return {
        'leaving': leaving, 'entering': entering, 'common': common,
        'moved': moved, 'examples': examples,
        'live': live, 'fresh': fresh,
        'gained_columns': [c for c in fresh.columns if c not in live.columns],
        'lost_columns': [c for c in live.columns if c not in fresh.columns],
    }


def _rated(df):
    level = df['ABC Level'].fillna('').astype(str).str.strip().str.upper()
    return int(level.isin(('A+', 'A', 'B+', 'B', 'C')).sum())


def report(d, live, fresh):
    log('')
    log('=' * 72)
    log(f'{LIVE_RECORDS}  ->  {SIDECAR}')
    log('=' * 72)
    log(f'  rows                : {len(live)} -> {len(fresh)}')
    log(f'  rows carrying a C..A+ rating : {_rated(live)} -> {_rated(fresh)}')
    log(f'  providers leaving   : {len(d["leaving"])}')
    log(f'  providers entering  : {len(d["entering"])}')
    log(f'  providers in both   : {len(d["common"])}')
    if d['gained_columns'] or d['lost_columns']:
        log(f'  ! COLUMN CHANGE: gained {d["gained_columns"]}, '
            f'lost {d["lost_columns"]}')

    for label, keys, frame in (('LEAVING', d['leaving'], d['live']),
                               ('ENTERING', d['entering'], d['fresh'])):
        if not keys:
            continue
        log(f'\n  {label} ({len(keys)}):')
        for k in keys[:25]:
            row = frame.loc[k]
            log(f'    {str(k)[:44]:46} {str(row.get(NAME, ""))[:30]:32} '
                f'ABC={str(row.get("ABC Level", "")).strip()!r:6} '
                f'{row.get("Permit Type", "")}')
        if len(keys) > 25:
            log(f'    ... and {len(keys) - 25} more')

    log(f'\n  CELLS CHANGED among the {len(d["common"])} providers in both '
        f'snapshots:')
    if not d['moved']:
        log('    none -- the two snapshots agree on every shared provider')
    for column, n in sorted(d['moved'].items(), key=lambda kv: -kv[1]):
        log(f'    {column:28} {n:>5}')
        for k, x, y in d['examples'][column]:
            log(f'        {str(k)[:28]:30} {x[:24]!r:26} -> {y[:24]!r}')
    log('')


def merge(dry_run, allow_partial):
    if not os.path.exists(SIDECAR):
        raise SystemExit(
            f'no sidecar at {SIDECAR}. Run `python sc_refetch.py --fetch` first.')
    if not os.path.exists(LIVE_RECORDS):
        raise SystemExit(f'no records file at {LIVE_RECORDS} to compare against.')

    state = load_checkpoint() or {'counties': {}}
    fetched = len(state.get('counties', {}))
    if fetched < len(crawler.SC_COUNTIES) and not allow_partial:
        raise SystemExit(
            f'refusing to merge a partial sweep: {fetched} of '
            f'{len(crawler.SC_COUNTIES)} counties are in the checkpoint, so '
            f'{SIDECAR} is missing whole counties and promoting it would delete '
            f'every provider in them. Finish with `--fetch --resume`, or pass '
            f'--allow-partial if a short sweep is genuinely what you want.')

    read = dict(dtype=str, keep_default_na=False, low_memory=False)
    live = pd.read_csv(LIVE_RECORDS, **read)
    fresh = pd.read_csv(SIDECAR, **read)
    d = diff(live, fresh)
    report(d, live, fresh)

    if dry_run:
        log('dry run: nothing written.')
        return

    stamp = datetime.now().strftime('%Y%m%d')
    backup = os.path.join(OUT_DIR, f'sc_records_{stamp}.bak.csv')
    if os.path.exists(backup):
        raise SystemExit(
            f'{backup} already exists -- a merge has already run today. Move it '
            f'aside if you really want to overwrite the backup.')
    os.replace(LIVE_RECORDS, backup)
    tmp = LIVE_RECORDS + '.tmp'
    fresh.to_csv(tmp, index=False)
    os.replace(tmp, LIVE_RECORDS)

    provenance = {
        'source': crawler.BASE_URL + crawler.EXCEL_PATH + '?county=<County>',
        'counties_swept': fetched,
        'fetch_started_at': state.get('started_at'),
        'fetch_finished_at': state.get('finished_at'),
        'merged_at': _now(),
        'rows': len(fresh),
        'rated_rows': _rated(fresh),
        'sha256': _sha256(LIVE_RECORDS),
        'replaced': os.path.basename(backup),
        'rows_before': len(live),
        'providers_entering': len(d['entering']),
        'providers_leaving': len(d['leaving']),
        'cells_changed': d['moved'],
    }
    tmp = PROVENANCE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
    os.replace(tmp, PROVENANCE)

    log(f'promoted {SIDECAR} -> {LIVE_RECORDS}')
    log(f'  previous file kept as {backup}')
    log(f'  provenance written to {PROVENANCE}')
    log('')
    log('The row set has changed, so every step that reads sc_records.csv is '
        'now stale. sc/sc.md lists what to re-run, in order.')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fetch', action='store_true',
                    help='sweep the county exports into the sidecar')
    ap.add_argument('--merge', action='store_true',
                    help='diff the sidecar against sc_data/sc_records.csv and '
                         'promote it')
    ap.add_argument('--resume', action='store_true',
                    help='--fetch: continue an interrupted sweep, skipping the '
                         'counties already cached')
    ap.add_argument('--restart', action='store_true',
                    help='--fetch: discard the checkpoint and re-fetch every '
                         'county')
    ap.add_argument('--counties', default=None,
                    help='comma-separated county names (default: all 46)')
    ap.add_argument('--limit', type=int, default=None,
                    help='only sweep the first N pending counties (smoke test)')
    ap.add_argument('--no-count-check', action='store_true',
                    help='skip the HTML "Displaying N Providers" cross-check')
    ap.add_argument('--no-seed-check', action='store_true',
                    help='skip the county <select> comparison')
    ap.add_argument('--dry-run', action='store_true',
                    help='--merge: report the diff and write nothing')
    ap.add_argument('--allow-partial', action='store_true',
                    help='--merge: promote a sidecar built from fewer than 46 '
                         'counties (deletes every provider in the rest)')
    ap.add_argument('--delay-min', type=float, default=1.0)
    ap.add_argument('--delay-max', type=float, default=2.5)
    args = ap.parse_args()

    if args.fetch == args.merge:
        raise SystemExit('pick exactly one of --fetch / --merge '
                         '(see --help for the two-step sequence).')

    if args.merge:
        merge(args.dry_run, args.allow_partial)
        return

    counties = ([c.strip() for c in args.counties.split(',')] if args.counties
                else list(crawler.SC_COUNTIES))
    for c in counties:
        if c not in crawler.SC_COUNTIES:
            raise SystemExit(f'Unknown county: {c!r} -- not in '
                             f'sc_data/sc_seed.csv')

    if args.restart:
        for path in (CHECKPOINT, SIDECAR):
            if os.path.exists(path):
                os.remove(path)
        log('checkpoint discarded (--restart)')
    elif not args.resume and os.path.exists(CHECKPOINT):
        state = load_checkpoint()
        raise SystemExit(
            f'a sweep is already in progress: {len(state.get("counties", {}))} '
            f'of {len(crawler.SC_COUNTIES)} counties cached in {CHECKPOINT}. '
            f'Pass --resume to continue it, or --restart to throw it away and '
            f're-fetch everything.')

    if args.limit:
        counties = counties[:args.limit]

    fetch(counties, resume=args.resume,
          delay_range=(args.delay_min, args.delay_max),
          check_counts=not args.no_count_check,
          skip_seed=args.no_seed_check)


if __name__ == '__main__':
    main()
