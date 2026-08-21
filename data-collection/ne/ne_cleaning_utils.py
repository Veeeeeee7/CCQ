"""Shared cleaning utilities for the NE (Step Up to Quality) provider pipelines.

Two inputs are combined here:

  * ne_data/ne_records.csv   — one row per crawled `child-care-facility` page
                              (the only public source of the Step rating).
  * ne_data/ne_licensing.csv — the DHHS licensed-child-care roster, joined on
                              the license number for bonus attributes.

As in the WI/GA pipelines, multi-value and nested fields have an open-ended
category space, so both the raw and full pipelines DISCOVER the schema at
runtime rather than hardcoding a vocabulary from a sample.

Two Nebraska-specific wrinkles drive most of the code below.

1. IDENTIFIERS. The finder publishes a DHHS license number (CCC8794, FI11670,
   FII9561, PRE7890, SAOC8574) for most providers, and that is the natural
   `provider_id`. But 629 of 3,287 pages carry no license number at all —
   overwhelmingly Head Start and public-school programs, which are not DHHS
   licensed. Crucially those are NOT all unrated: 45 of them have a Step, and
   42 of those are Step 3, because the QIS lets Head Start / public-school
   programs enter automatically at Step 3. Dropping them would delete 38% of
   the Step-3 level and bias the target badly. So `build_provider_key` mints a
   synthetic, collision-proof id for them from the site's own facility post id
   (`STQ<facility_id>`); the `STQ` prefix cannot collide with any real license
   prefix, and the function asserts that. `has_license_number` records which
   namespace each provider_id came from.

2. LABELLED BLOBS. The DHHS roster stores several fields as label-prefixed text
   ("Capacity: 212", "Ages: 6 WKS", "Days of Week Open: MTWTHF",
   "Hours: 0600"), and the day codes are entered inconsistently across 31
   variants — Thursday appears as TH, Th or R, Sunday as SU, Su, SN or a second
   S, Saturday as S or SA. `parse_days_open` tokenizes greedily and resolves the
   repeated-letter ambiguity positionally (first T = Tuesday, second T =
   Thursday; first S = Saturday, second S = Sunday).
"""
import re
import json

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# identifiers & target
# ---------------------------------------------------------------------------

# provider_key is assembled by build_provider_key(): the DHHS license number
# where the provider has one, else a synthetic STQ<facility_id>. It is renamed
# to provider_id to match the downstream experiment loader (utils.ID_COL).
ID_RENAME = {"provider_key": "provider_id"}
ID_COL = "provider_id"

# The Step (1-5) scraped from the finder's `.step-badge`. Absent => unrated.
TARGET_RENAME = {"step_rating": "qr_rating"}

# Prefix for synthesised ids. No real NE license number starts with these
# letters (real prefixes: CCC, FI, FII, PRE, SAOC), which build_provider_key
# verifies at runtime rather than trusting.
SYNTHETIC_ID_PREFIX = "STQ"

# Columns excluded from the modelling feature set: redundant identifiers, raw
# source text, and metadata that should not be fed to a model as features.
# provider_id itself is retained (the experiment loader drops it).
NON_FEATURE_COLS = [
    "_source_nonnull_dropped",
    "slug",
    "facility_url",
    "facility_name",
    "license_number",
    "facility_id",
    "director",
    "phone",
    "address_raw",
    "street",
    "state",
    "errors",
    "extra_fields",
    "age_groups",
    "other_program_info",
    "accreditations",
    "dhhs_objectid",
    "dhhs_full_name",
    "dhhs_owner_manager",
    "dhhs_address",
    "dhhs_address_2",
    "dhhs_city",
    "dhhs_state",
    "dhhs_zip4",
    "dhhs_zip_code1",
    "dhhs_phone",
    "dhhs_license_number",
    "dhhs_roster_date",
    "dhhs_geocoded_date",
    "dhhs_gis_status",
]


def slug(text):
    text = str(text).strip().lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def _attach(df, new_columns):
    """Add a batch of derived columns in one shot.

    The discovery-based builders below can each emit dozens of columns (NE
    one-hots 85 counties), and assigning them one at a time makes pandas
    reallocate the block manager on every insert -- correct, but it emits a
    PerformanceWarning and wastes time. Building them as a single frame and
    concatenating once keeps the result identical and the frame unfragmented.
    """
    if not new_columns:
        return df
    block = pd.DataFrame(new_columns, index=df.index)
    return pd.concat([df.drop(columns=[c for c in block.columns
                                       if c in df.columns]), block], axis=1)


def safe_json_list(value):
    """Parse a JSON-list cell, returning [] for NaN/blank/unparseable cells."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------

def build_provider_key(df, log=None):
    """provider_key = license_number, else a synthetic STQ<facility_id>.

    The facility id is the finder's own WordPress post id; it is present and
    unique on every crawled page, so it yields a stable, deterministic key for
    the un-licensed (Head Start / public-school) providers instead of dropping
    them. We assert the synthetic namespace is disjoint from the real one.
    """
    lic = df['license_number'].astype('string').str.strip()
    lic = lic.replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})

    fid = df['facility_id'].astype('string').str.strip()
    fid = fid.str.replace(r'\.0$', '', regex=True)
    synthetic = SYNTHETIC_ID_PREFIX + fid

    real = set(lic.dropna())
    clash = real & set(synthetic.dropna())
    if clash:
        raise ValueError(f'Synthetic ids collide with real license numbers: '
                         f'{sorted(clash)[:5]}')

    # provider_key may already be present on the input, in which case it is
    # authoritative and must not be recomputed -- recomputing would put the
    # licence number back as the grain. has_license_number is a real feature and
    # is always derived here, from the licence column itself.
    if 'provider_key' not in df.columns:
        df['provider_key'] = lic.fillna(synthetic)
    df['has_license_number'] = lic.notna()

    missing = df['provider_key'].isna().sum()
    if missing and log:
        log(f'WARNING: {missing} rows have neither a license number nor a '
            f'facility id; they will be dropped.')
    df = df[df['provider_key'].notna()].copy()

    if log:
        n_syn = int((~df['has_license_number']).sum())
        rated_syn = int(((~df['has_license_number'])
                         & df['step_rating'].notna()).sum())
        log(f'provider_key: {len(df)} rows | {n_syn} synthetic ids '
            f'({rated_syn} of them rated)')
    return df


def prefer_rated_order(df, target='step_rating'):
    """Sort so that finalize()'s drop_duplicates(keep='first') keeps the best
    row for each provider_key.

    A handful of license numbers appear on more than one facility page (11 in
    the full crawl, e.g. `miracles-on-34th-st-daycare` / `-street`, or four
    `tiny-tot-daycare` pages). Two of those pairs are a rated page plus an
    unrated stub, so a naive keep-first would silently discard the rating.
    Order by: has a rating, then by how many fields are populated.
    """
    has_rating = df[target].notna().astype(int)
    populated = df.notna().sum(axis=1)
    helper = '_source_nonnull_dropped'
    if helper in df.columns:
        populated = (populated - 1
                     + pd.to_numeric(df[helper], errors='coerce').fillna(0).astype(int))
    order = (pd.DataFrame({'_r': has_rating, '_p': populated})
             .sort_values(['_r', '_p'], ascending=[False, False]).index)
    return df.loc[order]


# ---------------------------------------------------------------------------
# DHHS licensing join
# ---------------------------------------------------------------------------

def mark_licensing_matched(df, log=None):
    df = df.copy()
    df['dhhs_matched'] = df['dhhs_license_type'].notna()
    if log:
        n = int(df['dhhs_matched'].sum())
        log(f'licensing attributes present on {n}/{len(df)} rows')
    return df


# ---------------------------------------------------------------------------
# label-prefixed blob parsers (DHHS)
# ---------------------------------------------------------------------------

def _strip_label(series, pattern):
    """Drop a leading 'Some Label:' from a text column, leaving the value."""
    return (series.astype('string')
            .str.replace(pattern, '', regex=True, case=False)
            .str.strip()
            .replace({'': pd.NA}))


_AGE_UNIT_MONTHS = {
    'wk': 7.0 / 30.4375, 'wks': 7.0 / 30.4375, 'week': 7.0 / 30.4375,
    'weeks': 7.0 / 30.4375,
    'mo': 1.0, 'mos': 1.0, 'mon': 1.0, 'mons': 1.0, 'mth': 1.0, 'mths': 1.0,
    'month': 1.0, 'months': 1.0,
    'y': 12.0, 'yr': 12.0, 'yrs': 12.0, 'ys': 12.0, 'year': 12.0, 'years': 12.0,
}


def _age_to_months(value):
    """'6 WKS' / '3 yrs' / '18 MOS' -> age in months (float). Units are entered
    inconsistently (WKS/WEEKS, MOS/MTHS/MON, YRS/YR/Y/YS), so normalise."""
    if not isinstance(value, str):
        return np.nan
    m = re.search(r'(\d+(?:\.\d+)?)\s*([A-Za-z]+)', value.strip())
    if not m:
        return np.nan
    qty, unit = float(m.group(1)), m.group(2).lower()
    factor = _AGE_UNIT_MONTHS.get(unit)
    return round(qty * factor, 3) if factor else np.nan


def parse_ages(df):
    """dhhs_ages_from / dhhs_ages_to -> readable text + numeric months."""
    if 'dhhs_ages_from' in df.columns:
        df['dhhs_ages_from'] = _strip_label(df['dhhs_ages_from'], r'^\s*ages\s*:\s*')
        df['dhhs_age_from_months'] = df['dhhs_ages_from'].map(_age_to_months)
    if 'dhhs_ages_to' in df.columns:
        df['dhhs_ages_to'] = _strip_label(df['dhhs_ages_to'], r'^\s*ages\s*:\s*')
        df['dhhs_age_to_months'] = df['dhhs_ages_to'].map(_age_to_months)
    return df


def parse_capacity(df):
    """'Capacity: 212' -> numeric dhhs_capacity."""
    if 'dhhs_capacity' in df.columns:
        txt = _strip_label(df['dhhs_capacity'], r'^\s*capacity\s*:\s*')
        df['dhhs_capacity'] = pd.to_numeric(txt, errors='coerce')
    return df


DAY_NAMES = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
             'saturday', 'sunday']

# Greedy: longest tokens first so 'TH'/'SU'/'SA'/'SN' win over 'T'/'S'.
_DAY_TOKEN = re.compile(r'(TH|SU|SA|SN|M|T|W|R|F|S)')


def _days_from_code(code):
    """Resolve a DHHS day code to a set of weekday names.

    The roster uses 31 spellings for the same handful of schedules. Thursday is
    TH, Th or R; Sunday is SU, Su, SN, or simply a second S; Saturday is S or
    SA. Bare 'T' is Tuesday the first time and Thursday the second time (MTWTF).
    """
    if not isinstance(code, str):
        return set()
    tokens = _DAY_TOKEN.findall(re.sub(r'[^A-Za-z]', '', code).upper())
    days, seen_t, seen_s = set(), False, False
    for tok in tokens:
        if tok == 'M':
            days.add('monday')
        elif tok == 'W':
            days.add('wednesday')
        elif tok == 'F':
            days.add('friday')
        elif tok in ('TH', 'R'):
            days.add('thursday')
        elif tok in ('SU', 'SN'):
            days.add('sunday')
        elif tok == 'SA':
            days.add('saturday')
        elif tok == 'T':
            # first bare T = Tuesday, a later one = Thursday (MTWTF)
            days.add('thursday' if seen_t else 'tuesday')
            seen_t = True
        elif tok == 'S':
            # first bare S = Saturday, a later one = Sunday (MTWTFSS)
            days.add('sunday' if seen_s else 'saturday')
            seen_s = True
    return days


def parse_days_open(df, as_bool):
    """dhhs_days_open -> readable code, per-day columns and a day count.

    as_bool=True (full)  -> day_<weekday> presence booleans.
    as_bool=False (raw)  -> day_<weekday> holds the weekday name where open.
    Returns (df, new_day_columns).
    """
    new_cols = []
    if 'dhhs_days_open' not in df.columns:
        df['dhhs_days_open_count'] = np.nan
        return df, new_cols

    df['dhhs_days_open'] = _strip_label(
        df['dhhs_days_open'], r'^\s*days\s+of\s+week\s+open\s*:\s*')
    per_row = [_days_from_code(v) for v in df['dhhs_days_open']]

    batch = {}
    for day in DAY_NAMES:
        present = [day in s for s in per_row]
        batch[f'day_{day}'] = (present if as_bool
                               else [day.title() if p else np.nan
                                     for p in present])

    counts = [len(s) if s else np.nan for s in per_row]
    batch['dhhs_days_open_count'] = pd.array(
        [int(c) if c == c else pd.NA for c in counts], dtype='Int64')

    new_cols = [f'day_{d}' for d in DAY_NAMES]
    return _attach(df, batch), new_cols


def _hhmm_to_minutes(value):
    """'0700' / '1700.0' / 700 -> minutes past midnight."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = re.sub(r'\.0$', '', str(value).strip())
    if not s.isdigit():
        return np.nan
    s = s.zfill(4)
    hh, mm = int(s[:-2] or 0), int(s[-2:])
    if hh > 24 or mm > 59:
        return np.nan
    return hh * 60 + mm


def parse_hours(df):
    """dhhs_hours_from / dhhs_hours_to -> readable HHMM + minutes + duration."""
    if 'dhhs_hours_from' in df.columns:
        df['dhhs_hours_from'] = _strip_label(df['dhhs_hours_from'], r'^\s*hours\s*:\s*')
        df['dhhs_hours_from_minutes'] = df['dhhs_hours_from'].map(_hhmm_to_minutes)
    if 'dhhs_hours_to' in df.columns:
        df['dhhs_hours_to'] = (df['dhhs_hours_to'].astype('string')
                               .str.replace(r'\.0$', '', regex=True))
        df['dhhs_hours_to_minutes'] = df['dhhs_hours_to'].map(_hhmm_to_minutes)

    start = df.get('dhhs_hours_from_minutes')
    end = df.get('dhhs_hours_to_minutes')
    if start is not None and end is not None:
        span = end - start
        # A close time at/behind the open time is a data-entry artefact, not an
        # overnight program; leave it missing rather than invent a negative day.
        df['dhhs_hours_open_minutes'] = span.where(span > 0)
    return df


def parse_issue_date(df):
    """dhhs_issue_date -> readable ISO date + numeric year."""
    if 'dhhs_issue_date' in df.columns:
        dt = pd.to_datetime(df['dhhs_issue_date'], errors='coerce')
        df['dhhs_issue_date'] = dt.dt.strftime('%Y-%m-%d')
        df['dhhs_issue_year'] = dt.dt.year.astype('Int64')
    return df


# ---------------------------------------------------------------------------
# accreditations (JSON list of {name, dates})
# ---------------------------------------------------------------------------

def accreditation_counts(df):
    """num_accreditations: 0 for providers with none (a real zero, not NaN)."""
    df['num_accreditations'] = df['accreditations'].apply(
        lambda v: len(safe_json_list(v)))
    return df


def build_accreditation_columns(df, prefix='accred'):
    """Presence boolean per discovered accrediting body (full set)."""
    per_row = [{rec.get('name') for rec in safe_json_list(v)
                if isinstance(rec, dict) and rec.get('name')}
               for v in df['accreditations']]
    schema = {}
    for names in per_row:
        for n in names:
            schema.setdefault(slug(n), n)

    batch = {f'{prefix}_{s}': [schema[s] in names for names in per_row]
             for s in sorted(schema)}
    return _attach(df, batch), list(batch)


def build_json_key_columns(df, column, prefix, sep=' | '):
    """Discover the union of keys across a JSON-list column and emit one text
    column per key, joining that key's values across the list (raw set)."""
    parsed = [safe_json_list(v) for v in df[column]]
    keys = []
    for records in parsed:
        for rec in records:
            if isinstance(rec, dict):
                for k in rec:
                    if k not in keys:
                        keys.append(k)
    batch = {}
    for k in keys:
        values = []
        for records in parsed:
            vals = [str(rec[k]).strip() for rec in records
                    if isinstance(rec, dict) and rec.get(k) not in (None, '')]
            values.append(sep.join(vals) if vals else np.nan)
        batch[f'{prefix}_{slug(k)}'] = values
    return _attach(df, batch), list(batch)


# ---------------------------------------------------------------------------
# generic feature builders (shared with the WI/GA pipelines)
# ---------------------------------------------------------------------------

def _row_items(value, delimiter, strip_prefix=None, strip_suffix=None,
               skip_values=(), skip_regex=None):
    """Split one cell into a list of cleaned item phrases."""
    if not isinstance(value, str):
        return []
    items = []
    for part in value.split(delimiter):
        part = part.strip()
        if skip_regex and re.search(skip_regex, part, flags=re.IGNORECASE):
            continue
        if strip_prefix:
            part = re.sub(strip_prefix, '', part, flags=re.IGNORECASE).strip()
        if strip_suffix:
            part = re.sub(strip_suffix, '', part, flags=re.IGNORECASE).strip()
        part = part.strip(' .')
        if not part:
            continue
        if part.lower() in skip_values:
            continue
        items.append(part)
    return items


def build_multivalue_columns(df, column, delimiter, prefix, as_bool,
                             strip_prefix=None, strip_suffix=None,
                             skip_values=(), skip_regex=None):
    """Discover the unique items in a delimited multi-value text column and emit
    one column per item. as_bool=True -> presence booleans (full set);
    as_bool=False -> the item phrase where present else NaN (raw set).
    Returns (df, new_column_names)."""
    skip_values = tuple(s.lower() for s in skip_values)
    per_row = [
        _row_items(v, delimiter, strip_prefix, strip_suffix, skip_values,
                   skip_regex)
        for v in df[column]
    ]
    schema = {}
    for items in per_row:
        for it in items:
            schema.setdefault(slug(it), it)

    batch = {}
    for s in sorted(schema):
        phrase = schema[s]
        col_name = f'{prefix}_{s}'
        present = [phrase in items for items in per_row]
        batch[col_name] = (present if as_bool
                           else [phrase if p else np.nan for p in present])
    return _attach(df, batch), list(batch)


def build_categorical_onehot(df, column, prefix, skip_values=()):
    """One-hot a single-value categorical column over its discovered values."""
    skip_values = tuple(slug(s) for s in skip_values)
    if column not in df.columns:
        return df, []
    batch = {}
    for value in df[column].dropna().unique():
        if not (isinstance(value, str) and value.strip()):
            continue
        s = slug(value)
        if s in skip_values:
            continue
        batch[f'{prefix}_{s}'] = df[column].eq(value)
    return _attach(df, batch), list(batch)


def numeric_columns(df, columns):
    """Coerce self-reported numeric text columns to numbers."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ---------------------------------------------------------------------------
# shared tail
# ---------------------------------------------------------------------------

def finalize(df, columns_file, which, key, target, dynamic_prefixes,
             na_as_level=False, valid_target_values=None):
    """Shared tail: order by the stable scaffold + any discovered columns,
    dedup on the grain, filter the target, then drop empty/constant columns.

    na_as_level: when True (raw set), a column counts as informative if its
    presence/absence varies, i.e. NaN is treated as its own value. This keeps
    decomposed text columns whose non-null value is constant but which are only
    populated for some rows. When False (full set) the rule matches the GA/WI
    pipelines exactly: drop all-NaN columns and columns with one non-null value.

    valid_target_values: controls which rows survive on the target.
      - None (complete set): keep every row, including rows whose Step is
        invalid or missing (the 2,208 unrated providers).
      - an iterable of allowed values (e.g. (1, 2, 3, 4, 5)): keep only rows
        whose target, coerced to numeric, is one of those values.

    Callers are expected to have passed df through prefer_rated_order() first,
    so the keep='first' dedup below retains the rated page when one license
    number spans several facility pages.
    """
    df = df.rename(columns={**ID_RENAME, **TARGET_RENAME})
    df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

    # P1 / P2: direct identifiers, per-provider narratives and coordinates.
    # Dropped BEFORE the scaffold reindex so they cannot survive even if a
    # stale ne_columns.json still lists them.
    df = drop_identifier_columns(df)

    stable = json.load(open(columns_file, 'r'))[which]
    extras = [c for c in df.columns
              if c.startswith(dynamic_prefixes) and c not in stable]
    expected = list(dict.fromkeys(stable + sorted(extras)))
    df = df.reindex(columns=expected)

    df = df.drop_duplicates(subset=[key], keep='first')
    if valid_target_values is not None:
        numeric_target = pd.to_numeric(df[target], errors='coerce')
        df = df[numeric_target.isin(list(valid_target_values))]

    # P2 / P3: geography (collapse -> relabel -> one-hot) and the
    # k-anonymity sweep. These run HERE, after the dedup and the target filter,
    # because k is a property of the rows that actually ship -- computing it
    # before the filter would let a category that is thin in the release pass
    # on the strength of rows that were about to be dropped.
    df = apply_privacy_remediation(df, which)

    cols_to_drop = []
    for col in df.columns:
        if df[col].isna().all():
            cols_to_drop.append(col)
        elif df[col].nunique(dropna=not na_as_level) == 1:
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)


# >>> BEGIN GENERATED PRIVACY BLOCK -- do not edit here >>>
# Generated from docs/2026-08-06/privacy_block.py by
# docs/2026-08-06/sync_privacy_block.py. Edit the canonical file, not
# this copy -- `sync_privacy_block.py --check` fails on drift.

import json as _json
import random as _random
import re as _re
from pathlib import Path as _Path

import numpy as _np
import pandas as _pd

# --- per-state (rewritten by sync_privacy_block.py) --------------------------
STATE = "ne"

# Small-range integer counts to top-code. These are exempt from the categorical
# rules under the "continuous measure" scope decision, but a 7-level count is
# not really continuous: wa:num_contacts ran {2: 2643, 3: 285, 1: 111, 4: 82,
# 0: 28, 5: 18, 6: 1} and the single provider with six contacts was uniquely
# identified. Capping folds it into a group of 19 at no modelling cost, since a
# tree splitting on ">= 5" sees the same thing either way.
TOPCODE_COLS = {}

# Column families whose members are LEVELS of one attribute (built by the
# multi-value / one-hot / keyterm builders), so a rare member can be merged
# into a shared "<prefix>other" bucket without inventing a nonsense feature.
#
# Deliberately NOT included: prefixes that merely share a first token but whose
# members are distinct ATTRIBUTES -- has_*, num_*, and the JSON-derived
# violations_* / monitoring_* / complaints_* / licensehist_* families. OR-ing
# `violations_description` into `violations_other` would be exactly as wrong as
# OR-ing `has_phone` into `has_other`. Rare members of those families fall
# through to the free-text rule instead.
LEVEL_FAMILY_PREFIXES = ('accred_', 'age_', 'day_', 'info_', 'lictype_', 'ptype_')

# Never touched by any rule below.
PROTECTED_COLS = ("provider_id", "qr_rating")
# -----------------------------------------------------------------------------

K_ANON = 5            # minimum providers sharing any surviving category / flag
MAX_LEVELS = 50       # cap on columns emitted when decomposing a list column
OTHER_LABEL = "other"
MULTIVALUE_DELIM = "|"

# Geography collapses at a HIGHER floor than everything else (decision
# 2026-08-06). k=5 is the bare privacy threshold and a county holding exactly 5
# providers is near-isolating once combined with anything else -- but the
# binding argument is modelling, not privacy: ~100 near-empty geographic
# one-hots per state raise the SHAP shadow-feature noise floor and diffuse
# attribution (GA's top 10 carried 31% of attribution across 412 features
# versus MD's 93% across 26). The precedent is GA's region_* buckets, which hold
# 417-527 providers each and are the only geographic features that cleared the
# noise floor.
GEO_K_ANON = 20

# Values that pandas' read_csv turns back into NaN. A cell holding one of these
# exactly is written to CSV verbatim and reloads as missing, producing a column
# that looks 100% null on read-back -- which is how co/raw:need_none shipped as
# a phantom empty column while holding a real value for 15 providers.
_NA_TOKENS = {
    "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
    "n/a", "nan", "null", "none_",
}

# A cell counts as date-like on a loose pattern rather than a strict parse:
# the states use m/d/Y, Y-m-d and d-m-Y interchangeably and we only need to
# know "this column is a clock", not what the clock says.
_DATE_RE = _re.compile(r"\d{1,4}[-/]\d{1,2}[-/]\d{2,4}")

# Private store for the two mappings that must NOT ship with the release: the
# provider_id surrogate map and the geographic label map. Gitignored.
_PRIVATE_DIR = _Path(__file__).resolve().parent.parent / "private"

# --- P1: direct and near-direct identifiers ----------------------------------
# Dropped from EVERY state, not only the one that produces them, so a column
# reintroduced by a future crawler change cannot silently ship. This mirrors
# the PII_COLS tripwire that guards load_data() in the modelling repo.
#
# The narrative columns are here for the same reason as the obvious ones: at a
# uniqueness ratio of 0.93-1.00 the prose IS the identifier, and regex-scrubbing
# names and phone numbers out of a 22,000-character field is not reliable.
PII_COLS = (
    # CO -- inspection / complaint / injury narratives (uniq 0.97, phone hits
    # x500, embedded street addresses) and the on-site licence number.
    # adverse_actions_text was NOT in the audit's P1 list; the gate-2 pattern
    # scan caught 3,338 phone-number hits in it after the first remediation
    # pass, so it is the same class of column and is dropped with its siblings.
    "complaints_text", "injury_investigations_text", "inspection_report_text",
    "stage_ii_text", "adverse_actions_text", "license_number_on_site",
    # MT -- the program's trading name.
    "program_name",
    # NC -- 22,299-character violation narratives (person names, phones,
    # street addresses).
    "violations_text",
    # OK -- monitoring narrative, the per-provider report URL (itself a
    # registry key), and the per-visit compliance vector.
    "monitoring_areas_compliant", "monitoring_report_url",
    "monitoring_section_text",
    # WA -- the contact block: 3,060 distinct real people's names, plus
    # emails, phones, per-provider websites and the licence-history key.
    "contact_email", "contact_phone", "contact_website",
    "contacts_email", "contacts_full_name", "contacts_phone",
    "licensehist_license_id",
)

# --- P2: coordinates ---------------------------------------------------------
# Removed outright. No rounding, no centroid substitute: at provider density
# these locate a building, and a tree can memorise with them (they ranked
# #2-#4 in the MT and KY SHAP results, which is the problem, not the defence).
COORD_COLS = ("latitude", "longitude")

# --- P2: geography routed through collapse -> relabel -> one-hot -------------
# Matched by exact name so that look-alikes are not caught by accident:
# CO's `school_district_operated_program` is a boolean about governance, not a
# district identifier, and must NOT be pseudonymised.
GEO_LABEL_COLS = (
    "zip", "zip_code", "zipcode", "location_zip", "socrata_physicalzip",
    "county", "dhhs_county",
    "city", "location_city", "socrata_physicalcity",
    "school_district", "district", "region", "ccrr_region",
)

# Geographic families that the clean_full scripts one-hot BEFORE finalize runs
# (CO/KY/MT/NE/SC ship county_* this way). These are gathered back into a
# single series, put through the same pipeline, and re-emitted.
GEO_ONEHOT_PREFIXES = ("county_", "city_", "region_", "district_")

# Which geographic fields are one-hotted into `full`. COARSE fields only:
# county- and region-level. City, ZIP and school district stay as label columns
# in `raw` and are dropped from `full`, because one-hotting them would add
# hundreds of near-empty booleans (KY alone has 268 surviving zip and city
# labels) for granularity no tree can use. This rule was implicit in the first
# pass -- it happened to hold because the clean_full scripts only ever carried
# county forward -- and is now enforced rather than incidental.
GEO_FULL_FIELDS = ("county", "dhhs_county", "region", "ccrr_region", "district")

# Source columns that mean the same thing under different names, normalised so
# a field lands as `geo_county` / `geo_region` in every state and both views.
# Without this MT emits `geo_ccrr_region` in raw against `geo_region_*` in full,
# and NE emits `geo_dhhs_county` against everyone else's `geo_county`.
# WA's location_city/socrata_physicalcity (and the matching zip pair) are NOT
# aliased: they are two genuinely different source columns and collapsing them
# to one name would silently drop one.
GEO_FIELD_ALIASES = {
    "ccrr_region": "region",
    "dhhs_county": "county",
    "zip_code": "zip",
    "zipcode": "zip",
}

# Geographic fields NOT pseudonymised. `region` is coarse by construction --
# GA's six regions hold 417-527 providers each and MT's seven are CCR&R service
# areas -- so it identifies nobody on its own, and keeping it readable means the
# paper can report "region_SE cleared the noise floor" without reaching for the
# private map. Everything else is relabelled.
GEO_READABLE = ("region", "ccrr_region")


# =============================================================================
# small helpers
# =============================================================================
def _privacy_log(message, state=None):
    """Append to {state}_privacy_log.txt next to the state's utils and print.

    Deliberately independent of each state's own logger: the four reference
    states use three different logging signatures (plain callable, ParseLog
    with .warn, and CA's three-argument make_logger) and this block has to drop
    into all of them unchanged.
    """
    state = state or STATE
    path = _Path(__file__).resolve().parent / f"{state}_privacy_log.txt"
    try:
        with open(path, "a") as fh:
            fh.write(message + "\n")
    except Exception:
        pass
    print(message)


def _privacy_slug(text):
    text = str(text).strip().lower().replace("&", " and ")
    text = _re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "blank"


def _is_boolish(series):
    """True for a column whose non-null values are only true/false-like.

    Covers the four shapes a boolean survives a CSV round-trip as: real bools,
    nullable booleans, 0/1 ints (finalize casts full's booleans to Int64) and
    the 'True'/'False' strings you get back from a plain read_csv.
    """
    values = series.dropna()
    if values.empty:
        return False
    return set(_pd.unique(values)) <= {True, False, 0, 1, 0.0, 1.0,
                                       "True", "False"}


def _positive_mask(series):
    """Row mask for 'this flag is set', for either a boolean or a raw-style
    text level column (which holds the item phrase where present, else NaN)."""
    if _is_boolish(series):
        return series.map({True: True, False: False, 1: True, 0: False,
                           1.0: True, 0.0: False,
                           "True": True, "False": False}).fillna(False).astype(bool)
    return series.notna()


def _positives(series):
    return int(_positive_mask(series).sum())


def _is_level_column(series):
    """The structural signature of a one-hot / presence LEVEL column: boolean
    (the `full` shape), or exactly one distinct non-null value with NaN
    elsewhere (what build_multivalue_columns(as_bool=False) produces in `raw`).

    Sharing a prefix is NOT sufficient. GA's `operating_` prefix covers both
    operating_months_jan (a level) and operating_hours (a free categorical);
    only the first can be folded into operating_other.
    """
    return _is_boolish(series) or (
        series.nunique(dropna=True) == 1 and series.isna().any())


def _label_or_nan(mask, label=OTHER_LABEL):
    """Object-dtype array holding `label` where mask is set, NaN elsewhere.

    np.where(mask, "other", np.nan) raises DTypePromotionError on numpy 2.x --
    there is no common dtype for str and float -- so build the array explicitly.
    """
    out = _np.empty(len(mask), dtype=object)
    out[:] = _np.nan
    out[_np.asarray(mask, dtype=bool)] = label
    return out


def _private_path(name):
    _PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    return _PRIVATE_DIR / name


# =============================================================================
# hygiene -- not privacy, but it runs here because this is the last hook before
# finalize's all-NaN / constant drop, so a column emptied by a bad round-trip
# still gets caught.
# =============================================================================
def neutralize_na_tokens(df, state=None):
    """Rewrite cells that pandas' read_csv would turn back into NaN.

    A cell holding exactly "None" (or NA / N/A / NULL / nan / null) is written
    to CSV verbatim and reloads as missing, so the column reads as 100% null
    even though it carries real values. That is how `co/raw:need_none` shipped
    as a phantom empty column while actually flagging 15 providers.

    Lowercasing escapes the token set for the cases that occur here ("None" ->
    "none", which pandas does not treat as NA); anything still colliding after
    that gets a trailing underscore.
    """
    changed = []
    for column in df.columns:
        series = df[column]
        if series.dtype != object:
            continue
        hits = series.isin(_NA_TOKENS)
        if not hits.any():
            continue

        def _safe(value):
            if value not in _NA_TOKENS:
                return value
            lowered = str(value).lower()
            return lowered if lowered not in _NA_TOKENS else f"{lowered}_"

        df[column] = series.map(lambda v: _safe(v) if isinstance(v, str) else v)
        changed.append(f"{column} (x{int(hits.sum())})")
    if changed:
        _privacy_log(f"[hygiene] rewrote NA-colliding value(s) in "
                     f"{len(changed)} column(s): {changed}", state)
    return df


def topcode_counts(df, state=None, columns=None):
    """Cap the named small-range counts, so a lone provider at the top of the
    range joins the bucket below instead of standing alone."""
    columns = TOPCODE_COLS if columns is None else columns
    for column, cap in columns.items():
        if column not in df.columns:
            continue
        values = _pd.to_numeric(df[column], errors="coerce")
        above = int((values > cap).sum())
        if not above:
            continue
        df[column] = values.clip(upper=cap)
        _privacy_log(f"[P4] {column}: top-coded at {cap} ({above} row(s) capped)",
                     state)
    return df


# =============================================================================
# P1 / P2 -- outright drops
# =============================================================================
def drop_identifier_columns(df, state=None, log=None):
    """Drop the direct identifiers, the per-provider narratives and the
    coordinates. Called EARLY -- before the scaffold reindex -- so that these
    can never reach the output even if a scaffold still lists them."""
    targets = [c for c in df.columns
               if c in PII_COLS or c.lower() in COORD_COLS]
    if targets:
        _privacy_log(f"[P1/P2] dropped {len(targets)} identifier/coordinate "
                     f"column(s): {sorted(targets)}", state)
    return df.drop(columns=targets)


# =============================================================================
# P3 -- rare levels inside a one-hot / presence family
# =============================================================================
def collapse_rare_level_family(df, prefix, min_n=K_ANON, state=None):
    """Levels held by fewer than min_n providers are OR-ed into <prefix>other
    and their own columns removed.

    Works for both views: in `full` a level is a boolean, in `raw` it is the
    item phrase where present and NaN where not, so the merged bucket is
    emitted in whichever of those two shapes the family already uses.
    """
    other_col = prefix + OTHER_LABEL
    members = [c for c in df.columns
               if c.startswith(prefix) and c != other_col]
    if not members:
        return df

    rare = [c for c in members if 0 < _positives(df[c]) < min_n]
    if not rare:
        return df

    as_bool = all(_is_boolish(df[c]) for c in members)
    mask = _np.zeros(len(df), dtype=bool)
    for col in rare:
        mask |= _positive_mask(df[col]).to_numpy()
    if other_col in df.columns:
        mask |= _positive_mask(df[other_col]).to_numpy()

    df = df.drop(columns=rare)
    df[other_col] = mask if as_bool else _label_or_nan(mask)
    _privacy_log(f"[P3] {prefix}*: merged {len(rare)} level(s) with <{min_n} "
                 f"providers into {other_col} (n={int(mask.sum())})", state)

    # A bucket that is itself thin cannot stand: it would re-isolate exactly
    # the providers the merge was meant to hide.
    if 0 < _positives(df[other_col]) < min_n:
        _privacy_log(f"[P3] {other_col}: still <{min_n} after merging, dropped",
                     state)
        df = df.drop(columns=[other_col])
    return df


# =============================================================================
# P3 -- rare values inside a single-valued categorical
# =============================================================================
def collapse_rare_values(series, min_n=K_ANON, other=OTHER_LABEL):
    """Values held by fewer than min_n rows become `other`. Recomputed after
    merging: if `other` is itself thin it is folded into the smallest surviving
    group rather than left as a thin bucket of its own (spec 2.2 step 1)."""
    values = series.dropna().astype(str)
    counts = values.value_counts()
    rare = set(counts[counts < min_n].index)
    if not rare:
        return series
    out = series.astype(object).where(~series.astype(str).isin(rare), other)

    recounted = out.dropna().astype(str).value_counts()
    if other in recounted.index and recounted[other] < min_n:
        survivors = recounted.drop(index=[other])
        if len(survivors):
            smallest = survivors.idxmin()
            out = out.where(out != other, smallest)
        else:
            out = _pd.Series([_np.nan] * len(series), index=series.index)
    return out


# =============================================================================
# P3 -- free-text columns that cannot reach k>=5 as they stand
# =============================================================================
def classify_text_column(series, min_n=K_ANON):
    """Decide how a text column that fails k>=min_n should be treated.

    Four outcomes, because these columns are three different problems wearing
    the same dtype:

      "date"   -- an administrative clock. Section 5 keeps clocks, but the raw
                  text copy is near-unique and the full view already carries
                  the numeric derivative (license_age_days, days_since_
                  inspection, n_monitoring_visits), so the text column goes.
      "list"   -- a pipe-joined event list. It is unique only as a COMBINATION;
                  the underlying vocabulary is small (ky:inspection_
                  inspectiontype has 417 distinct cells over 13 distinct
                  items). Decomposing to one column per item drops uniqueness
                  to the vocabulary and keeps the signal.
      "single" -- an ordinary categorical with a long tail. Collapse the tail.
      "prose"  -- per-provider narrative. The vocabulary is as large as the
                  data, so no decomposition helps and the text itself is the
                  fingerprint. Drop, same reasoning the spec applies to the P1
                  narratives.

    Returns (kind, kept_items). kept_items is the vocabulary that survives at
    k>=min_n, most frequent first.
    """
    values = series.dropna().astype(str)
    if values.empty:
        return "prose", []

    is_list = values.str.contains(_re.escape(MULTIVALUE_DELIM), regex=True).mean() >= 0.20

    if is_list:
        frequency = {}
        for cell in values:
            for item in {p.strip() for p in cell.split(MULTIVALUE_DELIM) if p.strip()}:
                frequency[item] = frequency.get(item, 0) + 1
        vocabulary = list(frequency)
        if vocabulary and sum(bool(_DATE_RE.search(v)) for v in vocabulary) / len(vocabulary) > 0.7:
            return "date", []
        kept = {v for v, n in frequency.items() if n >= min_n}
        if not kept:
            return "prose", []
        covered = sum(
            1 for cell in values
            if {p.strip() for p in cell.split(MULTIVALUE_DELIM)} & kept
        ) / len(values)
        if covered < 0.5:
            # The common vocabulary explains less than half the rows, so what
            # is left after collapsing is mostly "other" -- it is prose.
            return "prose", []
        ordered = sorted(kept, key=lambda v: (-frequency[v], v))
        return "list", ordered

    counts = values.value_counts()
    if len(counts) and sum(bool(_DATE_RE.search(v)) for v in counts.index) / len(counts) > 0.7:
        return "date", []
    kept = counts[counts >= min_n]
    if kept.sum() / len(values) < 0.5:
        return "prose", []
    return "single", list(kept.index)


def remediate_text_column(df, column, min_n=K_ANON, max_levels=MAX_LEVELS,
                          state=None):
    """Apply the classify_text_column verdict to one column."""
    kind, kept = classify_text_column(df[column], min_n)

    if kind in ("date", "prose"):
        _privacy_log(f"[P3] {column}: dropped ({kind}, cannot reach k>={min_n})",
                     state)
        return df.drop(columns=[column])

    if kind == "single":
        df[column] = collapse_rare_values(df[column], min_n)
        _privacy_log(f"[P3] {column}: collapsed tail into '{OTHER_LABEL}' "
                     f"({len(kept)} value(s) kept)", state)
        return df

    # kind == "list": decompose into one text column per surviving item. The
    # vocabulary is capped so that a long-tailed field (wi:violations_rule_
    # number has 512 items at k>=5) cannot add hundreds of columns to the raw
    # view -- the tail goes to <column>_other along with the sub-threshold
    # items, which is the same bounded treatment geography gets.
    emit = kept[:max_levels]
    tail = set(kept[max_levels:])
    cells = df[column].map(
        lambda v: {p.strip() for p in str(v).split(MULTIVALUE_DELIM) if p.strip()}
        if isinstance(v, str) else set()
    )
    used = set()
    for item in emit:
        name = f"{column}_{_privacy_slug(item)}"
        if name in used or name in df.columns:
            continue
        used.add(name)
        df[name] = [item if item in items else _np.nan for items in cells]

    other_col = f"{column}_{OTHER_LABEL}"
    emitted = set(emit)
    has_other = [
        bool(items - emitted) if items else False for items in cells
    ]
    # The tail bucket is subject to the same floor as everything else: if only
    # a handful of providers land in it, it re-isolates exactly the providers
    # the fold was meant to hide.
    kept_other = sum(has_other) >= min_n
    if kept_other:
        df[other_col] = _label_or_nan(has_other)
    _privacy_log(f"[P3] {column}: decomposed into {len(used)} item column(s)"
                 f"{f' + {other_col}' if kept_other else ''} "
                 f"({len(tail)} tail item(s) folded"
                 f"{f', {sum(has_other)} row(s) below the floor dropped' if any(has_other) and not kept_other else ''})",
                 state)
    return df.drop(columns=[column])


# =============================================================================
# P2 -- geography: collapse -> relabel -> one-hot
# =============================================================================
def _gather_onehot_family(df, prefix):
    """Reverse a pre-built one-hot family back into a single label series so it
    can go through the same collapse/relabel pipeline as a text column."""
    members = [c for c in df.columns if c.startswith(prefix)]
    if len(members) < 2:
        return None, []
    out = _pd.Series([_np.nan] * len(df), index=df.index, dtype=object)
    for col in members:
        mask = _positive_mask(df[col])
        out = out.where(~(mask & out.isna()), col[len(prefix):])
    return out, members


def _load_geo_map(state):
    path = _private_path(f"geo_label_map_{state}.json")
    if path.exists():
        return _json.loads(path.read_text())
    return {}


def _save_geo_map(state, mapping):
    _private_path(f"geo_label_map_{state}.json").write_text(
        _json.dumps(mapping, indent=2, sort_keys=True))


def build_geo_features(df, which, state=None, min_n=GEO_K_ANON, log=None):
    """Collapse -> relabel -> one-hot, in that order.

    Order matters and is not interchangeable: relabeling defeats LOOKUP (a
    reader cannot turn geo_county_37 into a town) while collapsing defeats
    ISOLATION (no surviving group is smaller than min_n). Relabeling alone
    leaves every k=1 group intact, so the collapse has to come first --
    otherwise the "other" bucket ends up defined over shuffled labels and is
    much harder to verify.

    Output shape differs by view, on purpose:
      raw  -> ONE pseudonymous label column per geographic field, keeping the
              text-preserving shape section 3 requires of raw. One-hotting here
              would add ~1,255 boolean columns across the raw files and bury
              the text serialization in false flags.
      full -> one boolean per surviving label, matching how full encodes every
              other categorical.
    Both views collapse and relabel identically, so k>=min_n and the anti-lookup
    property hold either way.

    The real-value -> label mapping is written to private/ and never ships.
    Without it the release is pseudonymised, not anonymised: an adversary
    holding the public licensing registry can match providers on capacity,
    facility type and violation counts and read the mapping off the matches.
    """
    state = state or STATE
    mapping = _load_geo_map(state)
    rng = _random.Random()

    sources, seen = [], set()
    for col in GEO_LABEL_COLS:
        if col in df.columns:
            name = GEO_FIELD_ALIASES.get(col, col)
            # Only alias when the canonical name is still free, so WA's two
            # city columns keep their own identities instead of overwriting.
            if name in seen:
                name = col
            seen.add(name)
            sources.append((name, df[col].astype(object), [col]))
    for prefix in GEO_ONEHOT_PREFIXES:
        series, members = _gather_onehot_family(df, prefix)
        if series is not None:
            name = prefix.rstrip("_")
            if name in seen:
                continue
            seen.add(name)
            sources.append((name, series, members))

    for name, series, drop_cols in sources:
        values = series.dropna().astype(str)
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        if values.empty:
            continue

        # Fine-grained fields (city, ZIP, school district) are raw-only.
        if which == "full" and name not in GEO_FULL_FIELDS:
            _privacy_log(f"[P2] {name}: fine-grained, not one-hotted into full",
                         state)
            continue

        # 1. COLLAPSE
        collapsed = collapse_rare_values(series.astype(object), min_n)
        surviving = sorted(collapsed.dropna().astype(str).unique())

        # 2. RELABEL -- integer labels drawn from a random permutation seeded
        #    independently of the data. Reused across runs where the category
        #    set is unchanged, so labels stay stable between regenerations.
        #    Skipped for the coarse fields in GEO_READABLE.
        readable = name in GEO_READABLE
        if readable:
            # Strip a redundant field-name prefix so WA's literal "Region 1"
            # yields geo_region_1 rather than geo_region_region_1, matching MT.
            def _label(value):
                slug = _privacy_slug(value)
                prefix = f"{name}_"
                return slug[len(prefix):] if slug.startswith(prefix) else slug
            known = {v: _label(v) for v in surviving}
        else:
            known = mapping.get(name, {})
            unlabelled = [v for v in surviving if v not in known]
            if unlabelled:
                free = [i for i in range(len(surviving) + len(known))
                        if i not in set(known.values())]
                rng.shuffle(free)
                for value, label in zip(unlabelled, free):
                    known[value] = label
                mapping[name] = known

        labels = collapsed.map(lambda v: known.get(str(v)) if _pd.notna(v) else None)

        # 3. ONE-HOT (full) or single label column (raw)
        if which == "full":
            for value in surviving:
                label = known[value]
                df[f"geo_{name}_{label}"] = (labels == label).fillna(False).to_numpy()
        else:
            df[f"geo_{name}"] = labels.map(
                lambda v: f"{name}_{v}" if v is not None and _pd.notna(v) else _np.nan)

        _privacy_log(f"[P2] {name}: {len(values.unique())} value(s) -> "
                     f"{len(surviving)} label(s) at k>={min_n} "
                     f"({'one-hot' if which == 'full' else 'label column'}"
                     f"{', readable' if readable else ''})", state)

    if mapping:
        _save_geo_map(state, mapping)
    return df


# =============================================================================
# the sweep
# =============================================================================
def enforce_k_anonymity(df, which, state=None, level_prefixes=None,
                        protect=PROTECTED_COLS, min_n=K_ANON,
                        max_levels=MAX_LEVELS):
    """Every categorical value and every flag ends up shared by >= min_n
    providers. Applied over the RELEASED rows, which is why finalize calls this
    after the dedup and the target filter rather than before: k is a property
    of the rows that ship, not of the rows that were scraped."""
    state = state or STATE
    prefixes = LEVEL_FAMILY_PREFIXES if level_prefixes is None else level_prefixes

    # 1. rare levels inside a declared family -> <prefix>other
    for prefix in prefixes:
        df = collapse_rare_level_family(df, prefix, min_n, state)

    family_member = lambda c: any(c.startswith(p) for p in prefixes)

    # 2. text columns that still fail k
    for column in [c for c in df.columns if c not in protect]:
        if column not in df.columns:
            continue
        series = df[column]
        if _pd.api.types.is_numeric_dtype(series) or _is_boolish(series):
            continue
        values = series.dropna().astype(str)
        if values.empty:
            continue
        counts = values.value_counts()
        if counts.min() >= min_n:
            continue
        if family_member(column) and _is_level_column(series):
            # Step 1 already folded every thin level of this family, so a
            # surviving level column is above the floor by construction.
            continue
        df = remediate_text_column(df, column, min_n, max_levels, state)

    # 3. flags outside any declared family that are set for < min_n providers.
    #    There is no family bucket to merge them into, so they go; a flag true
    #    for three providers is near-constant for modelling and isolating for
    #    privacy, which is the worst trade in the dataset.
    for column in [c for c in df.columns if c not in protect]:
        if column not in df.columns or not _is_boolish(df[column]):
            continue
        if family_member(column):
            continue
        positives = _positives(df[column])
        if 0 < positives < min_n:
            _privacy_log(f"[P3] {column}: flag set for {positives} provider(s), "
                         f"no family to merge into -- dropped", state)
            df = df.drop(columns=[column])

    # 4. flags whose MINORITY class is below the floor, in either direction.
    #    A boolean is k-anonymous at min(positives, negatives), not at
    #    positives: wa/full:has_email was True for 3,167 of 3,168 providers, so
    #    "the one WA provider with no email" was a unique identifier even though
    #    step 3 saw 3,167 and waved it through. Applies to family members too,
    #    since a level true for all-but-one isolates the exception.
    for column in [c for c in df.columns if c not in protect]:
        if column not in df.columns or not _is_boolish(df[column]):
            continue
        mask = _positive_mask(df[column])
        present = int(df[column].notna().sum())
        positives = int(mask.sum())
        minority = min(positives, present - positives)
        if 0 < minority < min_n:
            _privacy_log(f"[P3] {column}: minority class is {minority} "
                         f"provider(s) of {present} -- dropped", state)
            df = df.drop(columns=[column])
    return df


def apply_privacy_remediation(df, which, state=None, level_prefixes=None,
                              protect=PROTECTED_COLS, min_n=K_ANON,
                              max_levels=MAX_LEVELS, log=None):
    """Geography, then the k-anonymity sweep. Call from finalize() AFTER the
    row filters; call drop_identifier_columns() BEFORE the scaffold reindex."""
    state = state or STATE
    order = list(df.columns)
    df = neutralize_na_tokens(df, state)
    df = topcode_counts(df, state)
    df = build_geo_features(df, which, state, max(min_n, GEO_K_ANON), log)
    df = enforce_k_anonymity(df, which, state, level_prefixes, protect,
                             min_n, max_levels)
    # Both steps append their output, so restore the scaffold's ordering for
    # the survivors and park everything newly derived after it, sorted. This
    # keeps provider_id and qr_rating first, which the release contract
    # requires and the downstream loader assumes.
    kept = [c for c in order if c in df.columns]
    added = sorted(c for c in df.columns if c not in order)
    return df[kept + added]


def assert_k_anonymity(df, protect=PROTECTED_COLS, min_n=K_ANON):
    """Gate 3, as an assertion the pipeline can run on itself."""
    failures = []
    for column in df.columns:
        if column in protect:
            continue
        series = df[column]
        if _is_boolish(series):
            present = int(series.notna().sum())
            positives = _positives(series)
            minority = min(positives, present - positives)
            if 0 < minority < min_n:
                failures.append(f"{column} (minority class {minority})")
        elif not _pd.api.types.is_numeric_dtype(series):
            counts = series.dropna().astype(str).value_counts()
            if len(counts) and counts.min() < min_n:
                failures.append(f"{column} (min group {int(counts.min())})")
    return failures

# <<< END GENERATED PRIVACY BLOCK <<<
