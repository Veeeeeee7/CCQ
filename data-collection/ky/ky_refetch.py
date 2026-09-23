"""
ky_refetch.py -- re-collect the per-provider detail payload for providers
whose ky_crawler.py row carries an error, without replaying the browser UI.

The detail payload is one Salesforce Aura POST:

    POST https://kynect.ky.gov/benefits/s/sfsites/aura
    classname = SSP_ChildCareProviderSearchController
    method    = fetchBrightwheelDetailsForProvider
    params    = {"providerId": <kynect ProviderId>, "licenseNumber": <CLR>}
    -> returnValue.returnValue.mapResponse.{KICCSDataDetails, showKICCSCapacity}

The seed carries kynect's `ProviderId`, so no License search is needed and
each provider costs a single ~1 s request. The guest `aura.token` is the
literal string "null"; the only per-run bootstrap is scraping `fwuid` and the
community app version out of the search page, as mt_crawler.py does.

Writes a SIDECAR, never the records file:

    ky_data/ky_refetch.csv              one row per provider fetched
    ky_data/ky_refetch_checkpoint.json  resume state, rewritten as it goes

`ky_data_correction.py --fix-detail` merges the sidecar into ky_records.csv
positionally, in place. A sidecar row with a non-empty `errors` cell is
retried on the next --resume run.

If the Aura path breaks (renamed Apex method, guest token rejected), the
browser flow is the fallback:

    python ky_refetch.py --build-ui-seed          # writes ky_data/ky_refetch_seed.csv
    python ky_crawler.py --headless --channel chrome \
           --seed ky_data/ky_refetch_seed.csv --out ky_data/ky_records_refetch.csv
    python ky_data_correction.py --fix-detail --sidecar ky_data/ky_records_refetch.csv

Usage:
    python ky_refetch.py --limit 3 --verbose     # smoke test
    python ky_refetch.py                         # full run, ~1 h
    python ky_refetch.py --resume                # continue a stopped run
    python ky_refetch.py --build-ui-seed         # browser-fallback seed only

Deps: pip install requests pandas  (playwright only for the fallback path)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

import requests

try:
    from ky_crawler import DETAIL_COLS, KEY_COL, UA, flatten_detail
except ImportError as exc:                       # pragma: no cover
    raise SystemExit(
        f'could not import ky_crawler ({exc}). Run this from inside ky/, and '
        f'note that ky_crawler.py imports playwright even though this script '
        f'does not use it: pip install -r ../requirements.txt')

csv.field_size_limit(10_000_000)

SEARCH_PAGE = ('https://kynect.ky.gov/benefits/s/child-care-provider'
               '?origin=program-page&language=en_US')
AURA_ENDPOINT = 'https://kynect.ky.gov/benefits/s/sfsites/aura'
APEX_CLASS = 'SSP_ChildCareProviderSearchController'
DETAIL_METHOD = 'fetchBrightwheelDetailsForProvider'
SEARCH_METHOD = 'getChildCareProviderDetails'

RECORDS_PATH = os.path.join('ky_data', 'ky_records.csv')
SIDECAR_PATH = os.path.join('ky_data', 'ky_refetch.csv')
CHECKPOINT_PATH = os.path.join('ky_data', 'ky_refetch_checkpoint.json')
UI_SEED_PATH = os.path.join('ky_data', 'ky_refetch_seed.csv')
LOG_FILE = 'ky_refetch_log.txt'

PROVIDER_ID_COL = 'ProviderId'
SIDECAR_COLS = [KEY_COL, PROVIDER_ID_COL] + DETAIL_COLS + ['errors', 'fetched_at']

SEED_PATH = os.path.join('ky_data', 'ky_seed.csv')

REQUEST_TIMEOUT = 60

# Salesforce rejects a POST whose fwuid/app version is stale (the org
# redeployed mid-run). These are what that looks like in an action's error.
_STALE_MARKERS = ('clientoutofsync', 'framework version', 'invalid_session',
                  'aura.clientservice')

FWUID_RE = re.compile(r'"fwuid"\s*:\s*"([^"]+)"')
APPVER_RE = re.compile(
    r'markup://siteforce:communityApp["\\\']*\s*:\s*["\\\']*([A-Za-z0-9_\-]+)')

VERBOSE = False


def log(message, file=LOG_FILE):
    with open(file, 'a', encoding='utf-8') as fh:
        fh.write(message + '\n')
    print(message, flush=True)


def vlog(message):
    if VERBOSE:
        log(message)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


# --------------------------------------------------------------------------
# Aura session
# --------------------------------------------------------------------------

class AuraSession:
    """fwuid + app version are Salesforce build fingerprints that change on
    every org redeploy, so they are scraped per run rather than pinned."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = UA
        self.context = None

    def bootstrap(self):
        response = self.session.get(SEARCH_PAGE, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        html = response.text
        fwuid = FWUID_RE.search(html)
        appver = APPVER_RE.search(html)
        if not fwuid or not appver:
            raise RuntimeError(
                'could not scrape fwuid / app version from the search page -- '
                'the Aura bootstrap markup changed. Fall back to the browser '
                'path: python ky_refetch.py --build-ui-seed')
        self.context = {
            'mode': 'PROD',
            'fwuid': fwuid.group(1),
            'app': 'siteforce:communityApp',
            'loaded': {'APPLICATION@markup://siteforce:communityApp':
                       appver.group(1)},
            'dn': [], 'globals': {}, 'uad': True,
        }
        log(f'[bootstrap] fwuid={fwuid.group(1)[:24]}... '
            f'appver={appver.group(1)}')

    def apex(self, method, params, action_id='1;a'):
        """One ApexActionController call. Returns the decoded returnValue."""
        if self.context is None:
            self.bootstrap()
        message = json.dumps({'actions': [{
            'id': action_id,
            'descriptor': 'aura://ApexActionController/ACTION$execute',
            'callingDescriptor': 'UNKNOWN',
            'params': {'namespace': '', 'classname': APEX_CLASS,
                       'method': method, 'params': params,
                       'cacheable': False, 'isContinuation': False},
        }]})
        response = self.session.post(
            f'{AURA_ENDPOINT}?r=1&aura.ApexAction.execute=1',
            data={'message': message,
                  'aura.context': json.dumps(self.context),
                  'aura.pageURI': '/benefits/s/child-care-provider'
                                  '?origin=program-page&language=en_US',
                  'aura.token': 'null'},
            timeout=REQUEST_TIMEOUT,
            headers={'Content-Type': 'application/x-www-form-urlencoded',
                     'Referer': SEARCH_PAGE})
        if response.status_code >= 500:
            raise RuntimeError(f'HTTP {response.status_code} from kynect')
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError:
            raise RuntimeError(
                f'non-JSON response ({len(response.text)} bytes): '
                f'{response.text[:200]!r}')
        actions = body.get('actions') or []
        if not actions:
            snippet = str(body)[:300].casefold()
            if any(marker in snippet for marker in _STALE_MARKERS):
                raise StaleContext(snippet)
            raise RuntimeError(f'no actions in response: {str(body)[:300]}')
        action = actions[0]
        if action.get('state') != 'SUCCESS':
            detail = str(action.get('error'))[:300]
            if any(marker in detail.casefold() for marker in _STALE_MARKERS):
                raise StaleContext(detail)
            raise RuntimeError(f'apex state={action.get("state")}: {detail}')
        value = (action.get('returnValue') or {}).get('returnValue')
        if isinstance(value, str):
            value = json.loads(value)
        return value


class StaleContext(RuntimeError):
    """fwuid / app version no longer accepted -- re-bootstrap and retry once."""


def fetch_detail(aura, provider_id, clr):
    """mapResponse for one provider, or None if kynect has nothing for it."""
    value = aura.apex(DETAIL_METHOD,
                      {'providerId': provider_id, 'licenseNumber': clr})
    if not isinstance(value, dict):
        raise RuntimeError(f'unexpected detail payload: {str(value)[:200]}')
    if value.get('bIsSuccess') is False:
        raise RuntimeError(f'kynect reported bIsSuccess=false: '
                           f'{str(value)[:200]}')
    map_response = value.get('mapResponse')
    if not isinstance(map_response, dict) or 'KICCSDataDetails' not in map_response:
        raise RuntimeError('no KICCSDataDetails in the response')
    return map_response


def lookup_provider_id(aura, clr):
    """Safety net for a row whose seed ProviderId is blank: the License search
    returns it."""
    value = aura.apex(SEARCH_METHOD, {'queryData': {
        'latitude': '', 'longitude': '', 'providerName': None,
        'licenseNumber': clr, 'zipCode5': None, 'providerIDValues': None,
        'isFavoriteSearch': False, 'SearchCriteria': 'License'}})
    matches = (value or {}).get('sspChildCareProviderDetails') or []
    if not matches:
        raise RuntimeError('License search returned zero matches')
    return int(float(matches[0]['ProviderId']))


def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def owed_rows(records_path):
    """Providers whose detail payload never arrived, in records order.
    Keyed on the errors column: a blank detail cell is ambiguous
    (flatten_detail writes one for a genuinely empty list too)."""
    rows = read_csv_rows(records_path)
    owed = [r for r in rows if (r.get('errors') or '').strip()]
    return rows, owed


def load_sidecar_done(sidecar_path):
    """CLRs already recovered; a row with a non-empty errors cell is retried."""
    if not os.path.exists(sidecar_path):
        return set(), 0
    rows = read_csv_rows(sidecar_path)
    done = {r[KEY_COL] for r in rows if not (r.get('errors') or '').strip()}
    return done, len(rows)


def append_sidecar(path, row):
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=SIDECAR_COLS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in SIDECAR_COLS})
        fh.flush()
        os.fsync(fh.fileno())


def write_checkpoint(path, payload):
    """Written through a temp file so a kill mid-write cannot corrupt it."""
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def build_ui_seed(records_path, seed_path, out_path):
    """The browser fallback's input: the owed rows, projected onto the seed's
    columns so ky_crawler.py can read it unchanged."""
    _, owed = owed_rows(records_path)
    with open(seed_path, newline='', encoding='utf-8') as fh:
        seed_columns = next(csv.reader(fh))
    tmp = f'{out_path}.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=seed_columns)
        writer.writeheader()
        for row in owed:
            writer.writerow({c: row.get(c, '') for c in seed_columns})
    os.replace(tmp, out_path)
    return len(owed), seed_columns


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--records', default=RECORDS_PATH,
                    help='stage-1 records file, read to find the owed rows')
    ap.add_argument('--out', default=SIDECAR_PATH, help='sidecar to write')
    ap.add_argument('--checkpoint', default=CHECKPOINT_PATH)
    ap.add_argument('--limit', type=int, default=None,
                    help='only process the first N owed providers')
    ap.add_argument('--resume', action='store_true',
                    help='continue into an existing sidecar, skipping the '
                         'providers it already recovered')
    ap.add_argument('--restart', action='store_true',
                    help='ignore and overwrite an existing sidecar')
    ap.add_argument('--delay-min', type=float, default=1.0)
    ap.add_argument('--delay-max', type=float, default=2.0)
    ap.add_argument('--breaker', type=int, default=5,
                    help='consecutive failures that trigger a cooldown '
                         '(0 disables)')
    ap.add_argument('--cooldown', type=float, default=600.0)
    ap.add_argument('--max-cooldowns', type=int, default=3,
                    help='give up after this many cooldowns in one run')
    ap.add_argument('--build-ui-seed', action='store_true',
                    help='write the browser-fallback seed and exit')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    if not os.path.exists(args.records):
        raise SystemExit(f'{args.records} not found -- run this from inside ky/')

    if args.build_ui_seed:
        count, columns = build_ui_seed(args.records, SEED_PATH, UI_SEED_PATH)
        print(f'wrote {UI_SEED_PATH}: {count} owed provider(s), '
              f'{len(columns)} seed column(s)')
        print('then:\n'
              f'  python ky_crawler.py --headless --channel chrome \\\n'
              f'         --seed {UI_SEED_PATH} '
              f'--out ky_data/ky_records_refetch.csv\n'
              f'  python ky_data_correction.py --fix-detail '
              f'--sidecar ky_data/ky_records_refetch.csv')
        return 0

    rows, owed = owed_rows(args.records)
    log(f'[{_now()}] {len(rows)} row(s) in {args.records}, {len(owed)} owed')

    if os.path.exists(args.out):
        if args.restart:
            os.replace(args.out, args.out + '.superseded')
            log(f'[restart] previous sidecar moved to {args.out}.superseded')
        elif not args.resume:
            raise SystemExit(
                f'{args.out} already exists. Pass --resume to continue it '
                f'(errored rows are retried) or --restart to set it aside.')

    done, sidecar_rows = load_sidecar_done(args.out)
    todo = [r for r in owed if r[KEY_COL] not in done]
    if args.limit:
        todo = todo[:args.limit]
    log(f'{len(owed)} owed, {len(done)} already recovered '
        f'({sidecar_rows} sidecar row(s)), {len(todo)} to fetch this run')
    if not todo:
        log('Nothing to do.')
        return 0

    aura = AuraSession()
    aura.bootstrap()

    ok = fail = consecutive = cooldowns = 0
    started = time.time()
    for i, record in enumerate(todo, start=1):
        clr = (record.get(KEY_COL) or '').strip()
        raw_pid = (record.get(PROVIDER_ID_COL) or '').strip()
        out = {KEY_COL: clr, PROVIDER_ID_COL: raw_pid, 'errors': None,
               'fetched_at': _now()}
        try:
            if not clr:
                raise RuntimeError('empty ProviderCLRNumber')
            try:
                provider_id = int(float(raw_pid))
            except (TypeError, ValueError):
                vlog(f'[{clr}] no usable ProviderId ({raw_pid!r}) -- '
                     f'resolving it through a License search')
                provider_id = lookup_provider_id(aura, clr)
                out[PROVIDER_ID_COL] = str(provider_id)
            try:
                map_response = fetch_detail(aura, provider_id, clr)
            except StaleContext as stale:
                log(f'[bootstrap] context rejected ({stale}) -- re-scraping '
                    f'and retrying {clr}')
                aura.bootstrap()
                map_response = fetch_detail(aura, provider_id, clr)
            out.update(flatten_detail(map_response))
        except Exception as exc:                 # one bad provider != a dead run
            out['errors'] = f'{type(exc).__name__}: {exc}'
            vlog(f'[FAIL] {clr}: {out["errors"]}')

        append_sidecar(args.out, out)
        success = not out['errors']
        ok += int(success)
        fail += int(not success)
        consecutive = 0 if success else consecutive + 1

        if success:
            vlog(f'[{clr}] capacity={out.get("Capacity")!r} '
                 f'showKICCSCapacity={out.get("showKICCSCapacity")!r} '
                 f'inspections='
                 f'{len(json.loads(out.get("InspectionHistoryListUpdated") or "[]"))}')
        if i % 25 == 0 or i == len(todo) or not success:
            rate = (time.time() - started) / i
            write_checkpoint(args.checkpoint, {
                'updated_at': _now(), 'records': args.records,
                'sidecar': args.out, 'owed': len(owed),
                'attempted_this_run': i, 'ok_this_run': ok,
                'failed_this_run': fail, 'last_provider': clr,
                'seconds_per_provider': round(rate, 2),
                'eta_minutes_remaining': round(rate * (len(todo) - i) / 60, 1),
            })
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - started) / i
            log(f'[{i}/{len(todo)}] {clr}: {"ok" if success else "FAILED"} '
                f'({ok} ok / {fail} failed, {rate:.2f}s per provider, '
                f'~{rate * (len(todo) - i) / 60:.0f} min left)')
        else:
            vlog(f'[{i}/{len(todo)}] {clr}: {"ok" if success else "FAILED"}')

        if args.breaker and consecutive >= args.breaker:
            cooldowns += 1
            if cooldowns > args.max_cooldowns:
                log(f'[breaker] {cooldowns} cooldowns in one run -- stopping. '
                    f'Re-run with --resume once the site is healthy; the '
                    f'{ok} provider(s) recovered so far are already in '
                    f'{args.out}.')
                break
            log(f'[breaker] {consecutive} consecutive failures -- cooling down '
                f'{args.cooldown:.0f}s ({cooldowns}/{args.max_cooldowns}) and '
                f're-bootstrapping')
            time.sleep(args.cooldown)
            try:
                aura = AuraSession()
                aura.bootstrap()
            except Exception as exc:
                log(f'[breaker] re-bootstrap failed: {exc}')
            consecutive = 0
            continue

        time.sleep(random.uniform(args.delay_min, args.delay_max))

    elapsed = time.time() - started
    log(f'[{_now()}] done this run: {ok} ok, {fail} errored, out of '
        f'{len(todo)} attempted in {elapsed / 60:.1f} min -> {args.out}')
    log(f'next: python ky_data_correction.py --fix-detail --dry-run')
    return 0


if __name__ == '__main__':
    sys.exit(main())
