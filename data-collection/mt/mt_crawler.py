"""
mt_crawler.py — Montana: pull the MAQCS licensing directory, join STARS ratings.

Montana is a TWO-SOURCE state, because its rating source carries no identifier:

  RATINGS  DPHHS's "STARS Program List by City" PDF (Updated 7/1/2023, 216
           programs), transcribed into the stars_ratings half of
           mt_data/mt_seed.csv. Columns: program_name, city, star_level,
           program_type, ccrr_region. There is NO license number in that PDF.

  IDS      The MAQCS licensing directory, which HAS the state's provider
           identifier but NO star rating (confirmed from a full capture of the
           search payload: the response objects expose only Id/pid/
           providerNumber/providerName/providerType/street/city/county/state/
           zipCode/phone/latitude/longitude -- no rating field, no
           "quality"/"QRS" key anywhere). It is supplied as the
           licensing_roster half of the same seed; this script re-pulls it from
           the live site only when asked to.

So provider_id is recovered by joining the 216 rated programs onto the
licensing directory by normalized name + city. See join_ratings().

--- The endpoint -----------------------------------------------------------

The "Licensed Provider Search" at
    https://mtdphhs.my.site.com/MAQCSChildCareLicensing/s/provider-search
is a Salesforce Experience Cloud (Aura/LWC) app. A network capture showed the
search is a plain Apex POST, so we skip the browser entirely and replay it with
`requests` -- the playbook's preferred "API over DOM scraping" path, and the
lightest tool that works:

    POST /MAQCSChildCareLicensing/s/sfsites/aura?aura.ApexAction.execute=1
    message      = {"actions":[{"descriptor":"aura://ApexActionController/ACTION$execute",
                    "params":{"classname":"CC_ProviderSearchController",
                              "method":"callApex",
                              "params":{"wrapperData":"{...filters...}",
                                        "tabName":"provider"}}}]}
    aura.context = {"mode":"PROD","fwuid":<..>,"app":"siteforce:communityApp",
                    "loaded":{"APPLICATION@markup://siteforce:communityApp":<..>}}
    aura.token   = null      # public guest community; no auth needed

`fwuid` and the app version are Salesforce build fingerprints that change every
time the org is redeployed -- hardcoding them is the classic way these scrapers
rot. bootstrap_context() scrapes both out of the search page's own inline
bootstrap instead, so the crawler self-heals across releases.

wrapperData filters: providerName, providerNumber, county, city, zipCode (all
optional, empty string = unfiltered). We iterate Montana's 56 counties rather
than trusting a single blank statewide call not to be silently capped; the
observed Yellowstone County response held 159 records with no pagination
envelope, so a cap, if one exists, is above that and unannounced. --statewide
does the single blank call instead, and the two can be cross-checked by
comparing unique providerNumber counts (this is exactly how MI/CCHIRP confirmed
its statewide export was complete).

providerNumber looks like "PV109509" -- alphanumeric, so pandas won't coerce it,
but it is read/written as a string throughout regardless (project rule: IDs are
strings, always).

providerType is one of: Child Care Center, Group Home Child Care, Family Home
Child Care, Family/Friends & Neighbor (FFN) Providers. The STARS PDF's
program_type is one of Center, Group, Family -- mapped in TYPE_MAP and used only
as a tie-breaker when a name matches more than one provider, never as a filter.
(FFN providers are license-exempt and never appear in STARS; they simply never
match, which is correct.)

--- Expect an imperfect join ------------------------------------------------

The ratings are a frozen 7/1/2023 snapshot; the licensing directory is live
(2026). Programs that closed or renamed in the intervening ~3 years will not
match and will land with an empty provider_number. That is a real, reportable
outcome, not a bug -- the match report at the end of the run quantifies it, and
`match_method`/`match_score` are kept on every row so the unmatched tail can be
inspected (and, if the human wants, corrected by hand) rather than silently
dropped.

    python mt_crawler.py --counties Yellowstone,Missoula   # smoke test
    python mt_crawler.py                                   # full 56-county pull + join
    python mt_crawler.py --statewide                       # one blank call instead
    python mt_crawler.py --merge-only                      # rebuild providers csv from cache
    python mt_crawler.py --join-only                       # re-run just the join

Deps: pip install requests pandas
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.parse
from difflib import SequenceMatcher

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

BASE = 'https://mtdphhs.my.site.com/MAQCSChildCareLicensing'
SEARCH_PAGE = BASE + '/s/provider-search?language=en_US'
AURA_ENDPOINT = BASE + '/s/sfsites/aura'

APEX_CLASS = 'CC_ProviderSearchController'
APEX_METHOD = 'callApex'
APEX_TAB = 'provider'

OUT_DIR = 'mt_data'
RAW_DIR = os.path.join(OUT_DIR, 'providers_raw')
# Tabulated form of the cached directory pull. It lives with the cache rather
# than beside the outputs because a fresh run reads its half of the seed
# instead, and only a re-pull of the live directory rewrites it.
PROVIDERS_PATH = os.path.join(RAW_DIR, 'mt_providers.csv')
SEED_PATH = os.path.join(OUT_DIR, 'mt_seed.csv')
RECORDS_PATH = os.path.join(OUT_DIR, 'mt_records.csv')
# Rows the matcher would not auto-accept, with their top candidates, for a
# human to adjudicate; copy to MANUAL_PATH once the provider_number column is
# filled in and re-run with --join-only.
REVIEW_PATH = 'mt_match_review.csv'
MANUAL_PATH = 'mt_manual_matches.csv'
LOG_FILE = 'mt_crawler_log.txt'

UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

REQUEST_TIMEOUT = 60

# The 56 Montana counties. "Lewis and Clark" is spelled out in the picklist;
# the ampersand variant is tried as a fallback (the classic exact-match trap,
# same one that cost MI its St. Clair / St. Joseph counties on the first run).
MT_COUNTIES = [
    'Beaverhead', 'Big Horn', 'Blaine', 'Broadwater', 'Carbon', 'Carter',
    'Cascade', 'Chouteau', 'Custer', 'Daniels', 'Dawson', 'Deer Lodge',
    'Fallon', 'Fergus', 'Flathead', 'Gallatin', 'Garfield', 'Glacier',
    'Golden Valley', 'Granite', 'Hill', 'Jefferson', 'Judith Basin', 'Lake',
    'Lewis and Clark', 'Liberty', 'Lincoln', 'Madison', 'McCone', 'Meagher',
    'Mineral', 'Missoula', 'Musselshell', 'Park', 'Petroleum', 'Phillips',
    'Pondera', 'Powder River', 'Powell', 'Prairie', 'Ravalli', 'Richland',
    'Roosevelt', 'Rosebud', 'Sanders', 'Sheridan', 'Silver Bow', 'Stillwater',
    'Sweet Grass', 'Teton', 'Toole', 'Treasure', 'Valley', 'Wheatland',
    'Wibaux', 'Yellowstone',
]

COUNTY_VARIANTS = {
    'Lewis and Clark': ['Lewis and Clark', 'Lewis & Clark', 'Lewis And Clark'],
}

# STARS program_type -> MAQCS providerType. Tie-breaker only.
TYPE_MAP = {
    'Center': 'Child Care Center',
    'Group': 'Group Home Child Care',
    'Family': 'Family Home Child Care',
}

FUZZY_THRESHOLD = 0.88

# Fields we keep off each MAQCS record. `pid` is an opaque per-session hash and
# `Id` is the Salesforce row id -- neither is the licensing identifier, so both
# are carried as metadata only.
PROVIDER_FIELDS = ['providerNumber', 'providerName', 'providerType', 'city',
                   'county', 'street', 'zipCode', 'phone', 'latitude',
                   'longitude', 'state', 'Id']

# The columns each half of the seed owns. Both halves live in one file, stacked
# and tagged by seed_source, so each carries the other's columns as blanks.
RATING_FIELDS = ['program_name', 'city', 'star_level', 'program_type',
                 'ccrr_region', 'name_key', 'city_key', 'source_page']
ROSTER_FIELDS = PROVIDER_FIELDS + ['name_key', 'city_key']


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


def _slug(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def _norm_id(v):
    """IDs are strings, always. Guard against a stray float round-trip."""
    s = '' if v is None else str(v).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    return s


def _norm_key(s):
    """Loose join key. Must stay byte-identical to the name_key/city_key the
    seed already carries --
    the two are joined on its output."""
    s = (s or '').lower()
    s = s.replace('&', ' and ')
    s = re.sub(r"[’']", '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    s = re.sub(r'\b(llc|inc|incorporated|co)\b', '', s)
    return re.sub(r'\s+', ' ', s).strip()


# Tokens that carry no identifying signal in this domain -- essentially every
# MT child care program contains some of them. Two names that overlap ONLY on
# these are not the same program. This exists because a naive containment rule
# happily matched STARS's "TLC Daycare" to a provider whose slash-split alias
# was literally the word "Daycare" ("B Skogas/Camp Becky's Preschool/Daycare").
GENERIC_TOKENS = {
    'daycare', 'day', 'childcare', 'child', 'children', 'childrens', 'care',
    'preschool', 'pre', 'school', 'center', 'centre', 'academy', 'learning',
    'early', 'education', 'educational', 'kids', 'kid', 'little', 'home',
    'house', 'llc', 'inc', 'incorporated', 'co', 'the', 'a', 'and', 'of',
    'montana', 'mt', 'program', 'programs', 'site', 'services', 'service',
}


# Abbreviations the licensing registry uses that STARS spells out (or vice
# versa). Without this, "University on Princeton" vs "S. Fischer / Univ on
# Princeton" scores 0.85 and lands in review despite being obviously the same.
ABBREV = {
    'univ': 'university',
    'ctr': 'center',
    'dev': 'development',
    'elc': 'center',
    'assn': 'association',
    'assoc': 'association',
    'intl': 'international',
    'ent': 'enterprises',
}


def _stem(t):
    """Crudest possible plural stripping. "Country Bumpkin" vs "Country
    Bumpkins" is a real pair in this data; nothing subtler is needed."""
    if len(t) > 3 and t.endswith('s') and not t.endswith('ss'):
        return t[:-1]
    return t


def _tokens(s):
    return [_stem(ABBREV.get(t, t)) for t in _norm_key(s).split() if t]


def _distinctive(s):
    return {t for t in _tokens(s) if t not in GENERIC_TOKENS}


def provider_name_variants(name):
    """MAQCS family/group homes are registered as "Person Name / Business
    Name" (e.g. "Alisa Clarke / Lisa's Kids", "Sheryl Hutzenbiler / Munchkin
    Land"), while the STARS PDF lists only the business name. Some carry two
    slashes. So each slash-delimited part is a candidate name in its own right,
    alongside the full string."""
    parts = {name}
    if '/' in name:
        parts |= {p.strip() for p in name.split('/')}
    return {p for p in parts if len(_norm_key(p)) > 2}


def seed_name_variants(name):
    """STARS names often append a site qualifier after a dash -- "Explorers
    Academy - Laurel", "Kootenai Valley Head Start, Libby Center". Try the
    bare brand too, since licensing may register only the brand (or vice
    versa)."""
    out = {name}
    out.add(re.split(r'\s[–—-]\s', name)[0])
    out.add(name.split(',')[0])
    return {p for p in out if len(_norm_key(p)) > 2}


def score_pair(a, b):
    """Similarity of two program names in [0,1], plus a tier label.

    Requires a shared DISTINCTIVE token for any strong score -- overlap on
    generic words alone must never produce a match.
    """
    ta, tb = set(_tokens(a)), set(_tokens(b))
    da, db = _distinctive(a), _distinctive(b)
    if not ta or not tb:
        return 0.0, 'empty'

    if ta == tb:
        return 1.0, 'exact'

    shared = da & db
    if not shared:
        # No distinctive word in common -> not the same program, whatever the
        # raw string similarity says.
        return 0.0, 'no_shared_distinctive'

    # Containment: one name's tokens are a subset of the other's, AND the
    # shorter side contributes at least one distinctive token.
    if (ta <= tb or tb <= ta) and (da and db):
        ratio = min(len(ta), len(tb)) / max(len(ta), len(tb))
        return 0.90 + 0.05 * ratio, 'containment'

    # Every distinctive token of the shorter name is present in the longer.
    if da <= db or db <= da:
        ratio = min(len(da), len(db)) / max(len(da), len(db))
        return 0.88 + 0.04 * ratio, 'distinctive_containment'

    jac = len(ta & tb) / len(ta | tb)
    seq = SequenceMatcher(None, ' '.join(sorted(ta)), ' '.join(sorted(tb))).ratio()
    return max(jac, seq), 'fuzzy'


def best_candidate(seed_name, candidates):
    """Score seed_name against every (provider, name-variant) pair.
    Returns (best_row, score, tier, runner_up_score)."""
    scored = []
    svars = seed_name_variants(seed_name)
    for p in candidates:
        best_p = (0.0, 'none')
        for pv in provider_name_variants(p['providerName']):
            for sv in svars:
                sc, tier = score_pair(sv, pv)
                if sc > best_p[0]:
                    best_p = (sc, tier)
        if best_p[0] > 0:
            scored.append((best_p[0], best_p[1], p))
    if not scored:
        return None, 0.0, 'none', 0.0
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    return top[2], top[0], top[1], runner


# ---------------------------------------------------------------------------
# aura bootstrap
# ---------------------------------------------------------------------------

FWUID_RE = re.compile(r'fwuid%22%3A%22([^%"]+)%22|"fwuid"\s*:\s*"([^"]+)"')
APPVER_RE = re.compile(
    r'APPLICATION%40markup%3A%2F%2Fsiteforce%3AcommunityApp%22%3A%22([^%"]+)%22'
    r'|"APPLICATION@markup://siteforce:communityApp"\s*:\s*"([^"]+)"')


def bootstrap_context(session):
    """Scrape fwuid + the loaded app version out of the search page.

    These are Salesforce build fingerprints; they change on every org
    redeploy, and a stale pair makes the Aura endpoint reject the POST
    (typically with an event that reads like a framework version mismatch).
    Scraping them each run is what keeps this crawler from rotting.
    """
    r = session.get(SEARCH_PAGE, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    html = r.text

    m = FWUID_RE.search(html)
    v = APPVER_RE.search(html)
    if not m or not v:
        raise RuntimeError(
            'could not scrape fwuid / app version from the search page -- the '
            'Aura bootstrap markup changed. Re-run mt_capture.py and update '
            'FWUID_RE / APPVER_RE.')
    fwuid = m.group(1) or m.group(2)
    appver = v.group(1) or v.group(2)
    log(f'[bootstrap] fwuid={fwuid[:24]}... appver={appver}')
    return {
        'mode': 'PROD',
        'fwuid': fwuid,
        'app': 'siteforce:communityApp',
        'loaded': {'APPLICATION@markup://siteforce:communityApp': appver},
        'dn': [],
        'globals': {},
        'uad': True,
    }


def apex_search(session, ctx, provider_name='', provider_number='',
                county='', city='', zip_code=''):
    """One CC_ProviderSearchController.callApex call. Returns accountData."""
    wrapper = json.dumps({
        'providerName': provider_name,
        'providerNumber': provider_number,
        'county': county,
        'city': city,
        'zipCode': zip_code,
    })
    message = json.dumps({'actions': [{
        'id': '123;a',
        'descriptor': 'aura://ApexActionController/ACTION$execute',
        'callingDescriptor': 'UNKNOWN',
        'params': {
            'namespace': '',
            'classname': APEX_CLASS,
            'method': APEX_METHOD,
            'params': {'wrapperData': wrapper, 'tabName': APEX_TAB},
            'cacheable': False,
            'isContinuation': False,
        },
    }]})
    payload = {
        'message': message,
        'aura.context': json.dumps(ctx),
        'aura.pageURI': '/MAQCSChildCareLicensing/s/provider-search?language=en_US',
        'aura.token': 'null',
    }
    r = session.post(AURA_ENDPOINT, data=payload, timeout=REQUEST_TIMEOUT,
                     headers={'Content-Type': 'application/x-www-form-urlencoded',
                              'X-SFDC-Page-Scope-Id': '', 'Referer': SEARCH_PAGE})
    r.raise_for_status()
    body = r.json()

    actions = body.get('actions') or []
    if not actions:
        raise RuntimeError(f'no actions in response: {str(body)[:300]}')
    a = actions[0]
    if a.get('state') != 'SUCCESS':
        raise RuntimeError(f'apex state={a.get("state")}: {str(a.get("error"))[:300]}')

    rv = a.get('returnValue', {}).get('returnValue')
    if isinstance(rv, str):
        rv = json.loads(rv)
    if not isinstance(rv, dict):
        return []
    return rv.get('accountData') or []


# ---------------------------------------------------------------------------
# pull the directory
# ---------------------------------------------------------------------------

def _cache_path(key):
    return os.path.join(RAW_DIR, f'{_slug(key)}.json')


def fetch_county(session, ctx, county, force=False):
    """Fetch one county, cached. Tries spelling variants (Lewis and Clark)."""
    path = _cache_path(county)
    if os.path.exists(path) and not force:
        with open(path, encoding='utf-8') as f:
            rows = json.load(f)
        log(f'[skip] {county}: cached ({len(rows)} rows)')
        return rows

    last_err = None
    for variant in COUNTY_VARIANTS.get(county, [county]):
        try:
            rows = apex_search(session, ctx, county=variant)
            # An unrecognized county string returns an empty list rather than
            # an error, so an empty result is ambiguous: genuinely-childless
            # county, or a label miss. Try the next variant before believing it.
            if rows:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(rows, f, ensure_ascii=False)
                log(f'[{county}] {len(rows)} providers'
                    + ('' if variant == county else f' (matched as "{variant}")'))
                return rows
            last_err = 'empty result'
        except Exception as e:
            last_err = e
            log(f'[warn] {county} variant "{variant}" failed: {e}')

    # Persist the empty result so a genuinely empty county isn't re-fetched
    # every run, but say so loudly.
    with open(path, 'w', encoding='utf-8') as f:
        json.dump([], f)
    log(f'[{county}] 0 providers (last: {last_err}) -- verify this county is '
        f'truly empty and not a picklist-label miss')
    return []


def pull_directory(session, ctx, counties, force=False, delay=(1.0, 2.5)):
    all_rows = []
    for i, county in enumerate(counties, 1):
        try:
            all_rows.extend(fetch_county(session, ctx, county, force=force))
        except Exception as e:
            log(f'[FAIL] {county}: {e}')
        if i < len(counties):
            time.sleep(random.uniform(*delay))
    return all_rows


def pull_statewide(session, ctx, force=False):
    path = _cache_path('_statewide')
    if os.path.exists(path) and not force:
        with open(path, encoding='utf-8') as f:
            rows = json.load(f)
        log(f'[skip] statewide: cached ({len(rows)} rows)')
        return rows
    rows = apex_search(session, ctx)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False)
    log(f'[statewide] {len(rows)} providers, '
        f'{len({r.get("county") for r in rows})} distinct counties')
    return rows


def load_seed(path=SEED_PATH):
    """Split the seed back into its two halves: the STARS ratings and the
    licensing roster. They are stacked in one file and tagged by seed_source,
    so each half carries the other's columns as blanks -- take only the columns
    the half actually owns."""
    whole = pd.read_csv(path, dtype=str, keep_default_na=False)
    ratings = whole[whole['seed_source'] == 'stars_ratings'][RATING_FIELDS]
    roster = whole[whole['seed_source'] == 'licensing_roster'][ROSTER_FIELDS]
    return ratings.reset_index(drop=True), roster.reset_index(drop=True)


def merge_providers():
    """Union every cached raw file, dedupe on providerNumber."""
    files = sorted(f for f in os.listdir(RAW_DIR)) if os.path.isdir(RAW_DIR) else []
    rows = []
    for fn in files:
        if not fn.endswith('.json'):
            continue
        with open(os.path.join(RAW_DIR, fn), encoding='utf-8') as f:
            rows.extend(json.load(f))
    if not rows:
        log('[merge] no cached provider files found')
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for c in PROVIDER_FIELDS:
        if c not in df.columns:
            df[c] = None
    df = df[PROVIDER_FIELDS].copy()
    df['providerNumber'] = df['providerNumber'].map(_norm_id)

    before = len(df)
    df = df[df['providerNumber'] != '']
    df = df.drop_duplicates(subset='providerNumber', keep='first')
    log(f'[merge] {before} rows -> {len(df)} unique providerNumber')

    df['name_key'] = df['providerName'].map(_norm_key)
    df['city_key'] = df['city'].map(_norm_key)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(PROVIDERS_PATH, index=False)
    log(f'[merge] wrote {PROVIDERS_PATH} ({len(df)} x {df.shape[1]})')
    log(f'[merge] providerType breakdown:\n{df["providerType"].value_counts().to_string()}')
    return df


# ---------------------------------------------------------------------------
# join ratings -> ids
# ---------------------------------------------------------------------------

def load_manual_matches():
    """Optional human-curated overrides, highest precedence. Columns:
    program_name, city, provider_number. Anything listed here short-circuits
    the matcher -- this is how the review file gets folded back in."""
    if not os.path.exists(MANUAL_PATH):
        return {}
    df = pd.read_csv(MANUAL_PATH, dtype=str, keep_default_na=False)
    out = {}
    for r in df.to_dict('records'):
        pn = _norm_id(r.get('provider_number', ''))
        if pn:
            out[(_norm_key(r['program_name']), _norm_key(r['city']))] = pn
    log(f'[manual] loaded {len(out)} human-curated matches from {MANUAL_PATH}')
    return out


def join_ratings(seed, providers):
    """One row per STARS program; attach the MAQCS providerNumber.

    Scope: candidates are drawn from the seed city's COUNTY, not its city.
    The STARS PDF's "city" is the program's mailing city, which routinely
    disagrees with the licensing address -- "Explorers Academy - Laurel" and
    "- Lockwood" are both filed under Billings in the PDF but licensed in
    Laurel/Lockwood. County is derived from the licensing directory's own
    city -> county map, with a statewide fallback for cities it doesn't know.

    Acceptance is tiered, and deliberately conservative: a wrong provider_id
    silently corrupts the dataset, a missing one is merely a known gap. Only
    exact / containment matches (which require a shared distinctive token) and
    clearly-unambiguous fuzzy matches are auto-accepted. Everything else is
    written to mt_match_review.csv with its top candidates for a human to
    adjudicate, and folded back in via mt_manual_matches.csv.
    """
    manual = load_manual_matches()
    precs = providers.to_dict('records')

    by_number = {_norm_id(p['providerNumber']): p for p in precs}
    city2county = {}
    for p in precs:
        ck = _norm_key(p['city'])
        if ck and p.get('county'):
            city2county.setdefault(ck, p['county'])
    by_county = {}
    for p in precs:
        by_county.setdefault(p.get('county'), []).append(p)

    out, review = [], []
    for s in seed.to_dict('records'):
        name, ck = s['program_name'], _norm_key(s['city'])
        want_type = TYPE_MAP.get(s.get('program_type'))
        hit, method, score = None, 'unmatched', 0.0

        key = (_norm_key(name), ck)
        if key in manual:
            hit = by_number.get(manual[key])
            method, score = ('manual', 1.0) if hit else ('manual_bad_number', 0.0)

        if hit is None and method == 'unmatched':
            county = city2county.get(ck)
            cands = by_county.get(county, []) if county else precs
            scope = 'county' if county else 'statewide'

            # Prefer same-program-type candidates; fall back to all if that
            # empties the pool (STARS type and licensing type disagree often
            # enough -- e.g. Head Starts filed as Centers).
            typed = [c for c in cands if c['providerType'] == want_type]
            pool = typed if typed else cands

            hit, score, tier, runner = best_candidate(name, pool)
            score = round(score, 4)
            ambiguous = (score - runner) < 0.02

            if hit is not None and tier == 'exact' and not ambiguous:
                method = f'exact_{scope}'
            elif hit is not None and tier in ('containment', 'distinctive_containment') \
                    and score >= 0.90 and not ambiguous:
                method = f'{tier}_{scope}'
            elif hit is not None and score >= FUZZY_THRESHOLD and not ambiguous:
                method = f'fuzzy_{scope}'
            else:
                # Not confident. Record the top few for human adjudication.
                # Only surface candidates that scored at all; a 0.000 row
                # (no shared distinctive token) is noise, not a suggestion.
                topn = sorted(
                    ((sc, c) for sc, c in
                     ((best_candidate(name, [c])[1], c) for c in pool) if sc > 0),
                    key=lambda t: t[0], reverse=True)[:3]
                review.append({
                    'program_name': name, 'city': s['city'],
                    'program_type': s['program_type'],
                    'star_level': s['star_level'],
                    'reason': 'ambiguous' if (hit is not None and ambiguous) else 'low_score',
                    'best_score': score,
                    **{f'cand{i+1}': (f'{c["providerName"]} [{c["providerNumber"]}] '
                                      f'({c["city"]}, {sc:.3f})')
                       for i, (sc, c) in enumerate(topn)},
                    'provider_number': '',   # <- human fills this in
                })
                hit, method = None, 'needs_review'

        row = {
            'provider_number': _norm_id(hit['providerNumber']) if hit else '',
            'star_level': s['star_level'],
            'program_name': s['program_name'],
            'city': s['city'],
            'program_type': s['program_type'],
            'ccrr_region': s['ccrr_region'],
            'provider_name': hit['providerName'] if hit else '',
            'provider_type': hit['providerType'] if hit else '',
            'county': hit['county'] if hit else '',
            'street': hit['street'] if hit else '',
            'zip_code': _norm_id(hit['zipCode']) if hit else '',
            'phone': hit['phone'] if hit else '',
            'latitude': hit['latitude'] if hit else '',
            'longitude': hit['longitude'] if hit else '',
            'sf_id': hit['Id'] if hit else '',
            'match_method': method,
            'match_score': score,
            'errors': '' if hit else 'no_licensing_match',
        }
        out.append(row)

    # --- duplicate-claim guard -------------------------------------------
    # A license belongs to exactly one program. When two STARS rows claim the
    # same providerNumber, at least one is WRONG -- this is how the containment
    # tier betrays itself on multi-site brands ("Pete's Place Child Care
    # Center" and "Pete's Place North" both grabbing PV76597; likewise the two
    # YMCA sites). Keep only the strongest claimant and send the rest to
    # review rather than shipping a knowingly-wrong provider_id.
    claims = {}
    for i, row in enumerate(out):
        if row['provider_number'] and row['match_method'] != 'manual':
            claims.setdefault(row['provider_number'], []).append(i)

    n_demoted = 0
    for pn, idxs in claims.items():
        if len(idxs) < 2:
            continue
        ranked = sorted(
            idxs,
            key=lambda i: (out[i]['match_score'],
                           out[i]['match_method'].startswith('exact')),
            reverse=True)
        winner, losers = ranked[0], ranked[1:]
        log(f'[join] contested license {pn}: keeping '
            f'{out[winner]["program_name"]!r} ({out[winner]["match_method"]} '
            f'{out[winner]["match_score"]}); demoting '
            f'{[out[i]["program_name"] for i in losers]}')
        for i in losers:
            contested = out[i]['provider_name']
            review.append({
                'program_name': out[i]['program_name'],
                'city': out[i]['city'],
                'program_type': out[i]['program_type'],
                'star_level': out[i]['star_level'],
                'reason': 'duplicate_claim',
                'best_score': out[i]['match_score'],
                'cand1': f'{contested} [{pn}] -- ALSO claimed by '
                         f'{out[winner]["program_name"]!r}, which scored higher',
                'provider_number': '',
            })
            for k in ('provider_number', 'provider_name', 'provider_type',
                      'county', 'street', 'zip_code', 'phone', 'latitude',
                      'longitude', 'sf_id'):
                out[i][k] = ''
            out[i]['match_method'] = 'needs_review'
            out[i]['errors'] = 'duplicate_license_claim'
            n_demoted += 1

    if n_demoted:
        log(f'[join] demoted {n_demoted} row(s) that collided on a license')

    # --- synthetic ids ----------------------------------------------------
    # Per the human's call: rather than drop the rows we could not tie to a
    # live license (programs closed or renamed since the 7/1/2023 snapshot),
    # mint a deterministic synthetic provider_id so no STARS rating is lost.
    # `provider_id_source` marks which is which, so downstream analysis can
    # exclude synthetics trivially. Prefixed "MT-" so it can never collide
    # with a real MAQCS "PV#####" number.
    seen = {r['provider_number'] for r in out if r['provider_number']}
    n_synth = 0
    for row in out:
        if row['provider_number']:
            row['provider_id_source'] = 'license'
            continue
        base = f'MT-{_slug(row["city"]).upper()}-{_slug(row["program_name"]).upper()}'
        pid, n = base, 1
        while pid in seen:
            n += 1
            pid = f'{base}-{n}'
        seen.add(pid)
        row['provider_number'] = pid
        row['provider_id_source'] = 'synthetic'
        n_synth += 1
    log(f'[join] minted {n_synth} synthetic provider_id(s); '
        f'{len(out) - n_synth} carry a real MAQCS license number')

    df = pd.DataFrame(out)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(RECORDS_PATH, index=False)

    log(f'\n[join] wrote {RECORDS_PATH} ({len(df)} rows)')
    log(f'[join] match_method breakdown:\n{df["match_method"].value_counts().to_string()}')
    matched = (df['provider_number'] != '').sum()
    log(f'[join] provider_number populated: {matched}/{len(df)} '
        f'({matched / max(len(df), 1):.1%})')

    log(f'[join] provider_id_source:\n'
        f'{df["provider_id_source"].value_counts().to_string()}')

    # Should now be impossible (guard + synthetic minting both enforce it);
    # assert anyway, because a duplicate provider_id silently corrupts dedup.
    dup = df['provider_number']
    ndup = dup.duplicated().sum()
    if ndup:
        log(f'[join] WARNING {ndup} duplicate provider_number survived the '
            f'guard -- this should not happen. Offenders:')
        for pn in dup[dup.duplicated(keep=False)].unique():
            names = df[df['provider_number'] == pn]['program_name'].tolist()
            log(f'   {pn}: {names}')

    log(f'[join] star_level breakdown:\n{df["star_level"].value_counts().to_string()}')

    if review:
        rdf = pd.DataFrame(review)
        rdf.to_csv(REVIEW_PATH, index=False)
        log(f'\n[join] {len(rdf)} rows need human review -> {REVIEW_PATH}')
        log(f'       Fill in its `provider_number` column (blank = genuinely '
            f'absent, e.g. closed since the 7/1/2023 snapshot), save as '
            f'{MANUAL_PATH}, then re-run: python mt_crawler.py --join-only')
        for r in review[:25]:
            log(f'   - {r["program_name"]!r} ({r["city"]}, STAR {r["star_level"]}) '
                f'{r["reason"]} best={r["best_score"]} | {r.get("cand1", "-")}')
    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--counties', default=None,
                    help='Comma-separated subset, e.g. "Yellowstone,Missoula".')
    ap.add_argument('--limit', type=int, default=None,
                    help='Only process the first N counties (smoke test).')
    ap.add_argument('--statewide', action='store_true',
                    help='One blank call instead of looping counties.')
    ap.add_argument('--force', action='store_true',
                    help='Re-fetch counties already cached in providers_raw/.')
    ap.add_argument('--merge-only', action='store_true',
                    help='Skip fetching; re-tabulate the cached directory pull.')
    ap.add_argument('--join-only', action='store_true',
                    help='Skip fetching; just re-run the seed<->providers join.')
    ap.add_argument('--delay-min', type=float, default=1.0)
    ap.add_argument('--delay-max', type=float, default=2.5)
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    create_log_file()

    if not os.path.exists(SEED_PATH):
        log(f'[error] {SEED_PATH} not found.')
        sys.exit(1)
    seed, roster = load_seed()
    log(f'[seed] {len(seed)} STARS programs, {len(roster)} licensed providers')

    if args.join_only:
        join_ratings(seed, roster)
        return

    if args.merge_only:
        providers = merge_providers()
        join_ratings(seed, providers)
        return

    session = requests.Session()
    session.headers.update({'User-Agent': UA})
    ctx = bootstrap_context(session)

    if args.statewide:
        pull_statewide(session, ctx, force=args.force)
    else:
        counties = MT_COUNTIES
        if args.counties:
            wanted = {c.strip().casefold() for c in args.counties.split(',')}
            counties = [c for c in MT_COUNTIES if c.casefold() in wanted]
            missing = wanted - {c.casefold() for c in counties}
            if missing:
                log(f'[warn] not recognized, skipping: {sorted(missing)}')
        if args.limit:
            counties = counties[:args.limit]
        log(f'[pull] {len(counties)} counties')
        pull_directory(session, ctx, counties, force=args.force,
                       delay=(args.delay_min, args.delay_max))

    providers = merge_providers()
    if len(providers):
        join_ratings(seed, providers)


if __name__ == '__main__':
    main()
