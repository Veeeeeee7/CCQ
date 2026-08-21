"""Shared cleaning utilities for the SC (ABC Quality) provider pipelines.

South Carolina's source table is already flat -- the abcquality.org CSV export
has one row per provider and no multi-value, delimited, or nested-JSON cells --
so there is nothing to decompose the way WI/GA do. What SC needs instead is a
small number of careful, state-specific transforms:

  1. RATING. The native rating (`ABC Level`) is a LETTER grade, not a number:
     C < B < B+ < A < A+. Per the Checkpoint 0 decision it is mapped to an
     ordinal 1-5 in BOTH the raw and full outputs (RATING_MAP). Two non-grades
     also appear and are NOT valid ratings:
         "P"   -> quality score pending (provider met pre-reqs, not yet scored)
         ""    -> licensed but carries no ABC Quality rating at all
     Both become <NA> and are therefore dropped by finalize(valid_target_values=
     (1,2,3,4,5)) in the standard pipelines, while surviving in the *complete*
     ones.

     This file used to also export `rating_status` ('rated'/'pending'/
     'unrated', raw) and `rating_pending` (bool, full) to preserve the
     "pending" vs "never rated" distinction that qr_rating alone cannot
     express. Both were REMOVED as target leakage: they are rating-
     administration status, and in the *complete* outputs they identify with
     certainty which rows carry a missing target. See LEAKAGE_COLS below for
     the project-wide rule.

  2. EXEMPT PROVIDERS AND THEIR IDENTIFIER. 91 providers have
     `Permit Type == "Not Licensed (Exempt)"` and carry NO state permit number
     at all -- the `Permit Number` cell is empty. They are nonetheless rated
     (all 91 have an ABC Level; 71 have a valid C..A+ grade). Since provider_id
     must be populated and unique, synthesize_exempt_ids() mints a stable
     surrogate key of the form  EXEMPT-<zip5>-<name-slug>.

  3. *** TARGET LEAKAGE IN `Facility Type` *** (the reason this file exists)
     For NON-exempt rows `Facility Type` is a genuine facility code:
         A -> Child Care Center            (observed capacity 16-622)
         B -> Group Child Care Home        (observed capacity 8-12)
         C -> Family Child Care Home       (observed capacity 0-6)
     (Verified against capacity bands, and these three x the Permit Type map
     one-to-one onto the five facility types offered in the site's Advanced
     Search.)

     But for EXEMPT rows `Facility Type` is not a facility type at all -- it is
     a verbatim re-encoding of the rating:
         EAA -> A+     EXA -> A      EBB -> B+
         EXB -> B      EXT -> C      PEN -> P (pending)
     The observed crosstab is a perfect diagonal (2/12/7/17/33/20 rows). One-hot
     encoding that column as-is would hand a model the target. recode_facility_
     type() therefore collapses every exempt code to a single "EXEMPT" category
     and asserts the leak is gone. The original sub-codes are discarded on
     purpose: they carry zero facility information and 100% of the rating.

Everything else (capacity, the four program-flag booleans, county/city/zip,
inspection date) is a straight tabulation.
"""
import json
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# rating
# ---------------------------------------------------------------------------

# Native letter grade -> ordinal. C is the entry rating (score 50-70) and A+ the
# top (86-100); B/B+ sit between. See abcquality.org/families/quality-rating-system.
RATING_MAP = {
    'C': 1,
    'B': 2,
    'B+': 3,
    'A': 4,
    'A+': 5,
}
RATING_PENDING = 'P'          # scored pending -- NOT a quality level
VALID_RATINGS = tuple(sorted(RATING_MAP.values()))   # (1, 2, 3, 4, 5)

# ---------------------------------------------------------------------------
# facility / permit
# ---------------------------------------------------------------------------

EXEMPT_PERMIT_TYPE = 'Not Licensed (Exempt)'

# Facility codes seen ONLY on exempt rows, each of which encodes the provider's
# ABC rating rather than its facility type. Collapsed to EXEMPT_FACILITY_CODE.
LEAKY_EXEMPT_FACILITY_CODES = {'EAA', 'EXA', 'EBB', 'EXB', 'EXT', 'PEN'}
EXEMPT_FACILITY_CODE = 'EXEMPT'

# Human-readable labels for the raw pipeline. Inferred from the observed capacity
# bands and cross-checked against the five facility types listed on
# abcquality.org/provider-search/advanced/.
FACILITY_TYPE_LABELS = {
    'A': 'Child Care Center',
    'B': 'Group Child Care Home',
    'C': 'Family Child Care Home',
    EXEMPT_FACILITY_CODE: 'Legally Exempt Program',
}

# ---------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------

# Fixed so `days_since_inspection` is reproducible across re-runs. This is the
# date the source CSVs were crawled; bump it only when the data is re-crawled.
REFERENCE_DATE = pd.Timestamp('2026-07-08')
INSPECTION_DATE_FORMAT = '%m/%d/%Y'      # e.g. "08/08/2024"

# ---------------------------------------------------------------------------
# column naming
# ---------------------------------------------------------------------------

# The export's headers -> snake_case. `County` (the CSV's own, oddly title-cased
# e.g. "Mccormick") is dropped in favour of `source_county`, the canonical
# spelling the crawler swept with; they were verified to agree on every row.
SOURCE_RENAME = {
    'Provider Name': 'provider_name',
    'Permit Type': 'permit_type',
    'Permit Number': 'permit_number',
    'Operator': 'operator',
    'Facility Type': 'facility_type_code',
    'Street': 'street',
    'City': 'city',
    'State': 'state',
    'Zip': 'zip',
    'County': 'county_csv',
    'Phone': 'phone',
    'ABC Level': 'abc_level',
    'Last ABC Inspection Date': 'last_inspection_date',
    'Capacity': 'capacity',
    'Head Start': 'head_start',
    'First Steps': 'first_steps',
    'Breastfeeding Friendly': 'breastfeeding_friendly',
    'Sleep Safe': 'sleep_safe',
    'source_county': 'county',
}

BOOL_COLS = ['head_start', 'first_steps', 'breastfeeding_friendly', 'sleep_safe']

# permit_number is the grain; renamed to provider_id to match the downstream
# experiment loader (utils.ID_COL == "provider_id").
ID_RENAME = {'permit_number': 'provider_id'}
ID_COL = 'provider_id'

# abc_level has already been mapped to the 1-5 ordinal by map_rating() before
# finalize() runs; here it is only renamed to the canonical target name.
TARGET_RENAME = {'abc_level': 'qr_rating'}

# Dropped from every output: free-text identity/contact metadata that should
# never be a feature, the duplicate county column, the constant state column,
# the redundant `exempt` flag (perfectly collinear with facility_type_code ==
# "EXEMPT"), and `facility_type_code` itself once it has been one-hot encoded or
# turned into a label.
NON_FEATURE_COLS = [
    'provider_name',
    'operator',
    'phone',
    'street',
    'state',
    'county_csv',
    'exempt',
    'facility_type_code',
]

# --- TARGET-LEAKAGE COLUMNS (project-wide rule; NEVER exported) --------------
# Rule: no exported feature may be derived from the QRIS rating or from its
# administrative lifecycle -- the rating under any other name or rendering, and
# the paperwork it generates (award / effective / expiration / renewal dates,
# rating age, time-to-expiry, rating status) plus anything computed from them,
# including missingness indicators. Only `qr_rating`, the target, survives.
#
# For SC specifically: rating_status / rating_pending are pure rating-
# administration state. In the complete outputs, where unrated and pending rows
# are retained with a <NA> target, either column identifies the missing-target
# rows exactly. Removed outright rather than recoded -- the information is the
# leak, not the format.
#
# NOTE the deliberate exclusion: last_inspection_date / days_since_inspection /
# has_inspection stay. Licensing inspection is a separate regulatory process
# that applies to rated and unrated providers alike.
LEAKAGE_COLS = [
    'rating_status',
    'rating_pending',
]

# Any engineered column starting with one of these is a derived form of a
# leakage field (one-hot, ordinal, missingness indicator, bin) and is swept up
# by the same pass. Defensive against a future builder reintroducing the leak.
LEAKAGE_PREFIXES = (
    'rating_status', 'rating_pending', 'rating_age', 'rating_award',
    'rating_expir', 'rating_renew', 'rating_effective',
    'award_date', 'expiration_date', 'days_until_rating',
)


def drop_leakage_columns(df, log=None):
    """Remove every rating-derived / rating-lifecycle column. The target
    `qr_rating` is explicitly exempt. Column-only: rows and row order are
    untouched, so raw/full stay positionally aligned."""
    doomed = sorted(c for c in df.columns
                    if c != 'qr_rating'
                    and (c in LEAKAGE_COLS or c.startswith(LEAKAGE_PREFIXES)))
    if doomed:
        msg = f'[leakage] dropping {len(doomed)} rating-derived col(s): {doomed}'
        if log is not None:
            log(msg)
        print(f'  {msg}')
        df = df.drop(columns=doomed)
    return df


def slug(text):
    text = str(text).strip().lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


# ---------------------------------------------------------------------------
# source normalization
# ---------------------------------------------------------------------------

def normalize_source_columns(df):
    """Rename the export's headers to snake_case and coerce simple dtypes."""
    df = df.rename(columns=SOURCE_RENAME)

    # permit_number MUST stay a string: it is numeric-looking, and a float
    # round-trip would turn '16941' into '16941.0' and '' into NaN.
    df['permit_number'] = df['permit_number'].map(_clean_str)
    df['abc_level'] = df['abc_level'].map(lambda v: _clean_str(v).upper())

    for col in BOOL_COLS:
        df[col] = to_bool(df[col])

    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce').astype('Int64')
    df['zip'] = df['zip'].map(normalize_zip)
    return df


def _clean_str(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    s = str(value).strip()
    return '' if s.lower() in ('nan', 'none', 'n/a', '-') else s


def to_bool(series):
    """'True'/'False' text (or real bools) -> a numpy bool column."""
    return series.map(
        lambda v: str(v).strip().lower() in ('true', '1', 'yes', 't')
    ).astype(bool)


def normalize_zip(value):
    """Keep the 5-digit ZIP. One row in the source carries a ZIP+4."""
    s = _clean_str(value)
    m = re.match(r'(\d{5})', s)
    return m.group(1) if m else np.nan


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------

def synthesize_exempt_ids(df, log=None):
    """Give the permit-less exempt providers a stable surrogate provider_id.

    Exempt providers carry no state permit number, so `permit_number` is ''. A
    naive dedup on that column would collapse all 91 of them into a single row.
    The surrogate is deterministic (same input -> same id across re-runs) and is
    prefixed so it can never collide with a real numeric permit number."""
    blank = df['permit_number'] == ''
    if not blank.any():
        return df

    def _mint(row):
        z = row['zip'] if isinstance(row['zip'], str) and row['zip'] else 'nozip'
        return f"EXEMPT-{z}-{slug(row['provider_name'])}"

    df.loc[blank, 'permit_number'] = df.loc[blank].apply(_mint, axis=1)

    minted = df.loc[blank, 'permit_number']
    if minted.duplicated().any():
        dupes = sorted(minted[minted.duplicated(keep=False)].unique())
        raise ValueError(
            f'Synthesized exempt provider_id is not unique: {dupes[:5]}. '
            f'Extend the key (e.g. add street) before proceeding.')
    if log:
        log(f'Synthesized {int(blank.sum())} EXEMPT-* provider_id(s) for '
            f'permit-less exempt providers.')
    return df


# ---------------------------------------------------------------------------
# rating
# ---------------------------------------------------------------------------

def map_rating(series):
    """Letter grade -> ordinal 1-5. 'P' (pending) and '' (unrated) -> <NA>,
    which finalize() then drops in the standard pipelines and keeps in the
    complete ones."""
    return pd.array(
        [RATING_MAP.get(str(v).strip().upper(), pd.NA) for v in series],
        dtype='Int64')


def add_rating_status(df, log=None):
    """REMOVED -- retained as a stub so any stale caller fails loudly instead of
    silently reintroducing the leak.

    This used to add rating_status ('rated'/'pending'/'unrated') and
    rating_pending (bool) to preserve the pending-vs-never-rated distinction
    that qr_rating cannot carry. Both are rating-administration status and are
    banned by LEAKAGE_COLS: in the complete outputs they pinpoint exactly which
    rows have a missing target. If the pending/unrated distinction is ever
    needed for analysis, derive it from the source export at analysis time --
    it must not ship in the model-facing exports."""
    raise NotImplementedError(
        'add_rating_status() was removed as target leakage; see '
        'sc_cleaning_utils.LEAKAGE_COLS. Delete the call.')


# ---------------------------------------------------------------------------
# facility type  (leakage fix)
# ---------------------------------------------------------------------------

def recode_facility_type(df, log=None):
    """Collapse the rating-encoding exempt facility codes into one EXEMPT level.

    See this module's docstring: on exempt rows `Facility Type` is a 1:1 alias
    of the ABC rating (EAA=A+, EXA=A, EBB=B+, EXB=B, EXT=C, PEN=pending). Left
    alone it would leak the target straight into the one-hot features. After the
    collapse we assert that no surviving facility code determines the rating."""
    codes = df['facility_type_code'].map(lambda v: _clean_str(v).upper())
    leaky = codes.isin(LEAKY_EXEMPT_FACILITY_CODES)

    exempt = df['permit_type'].map(_clean_str).eq(EXEMPT_PERMIT_TYPE)
    stray = leaky & ~exempt
    if stray.any():
        raise ValueError(
            f'{int(stray.sum())} NON-exempt row(s) carry a rating-encoding '
            f'facility code {sorted(codes[stray].unique())}. The leakage '
            f'assumption no longer holds -- re-inspect the source.')

    df['facility_type_code'] = codes.where(~leaky, EXEMPT_FACILITY_CODE)
    _assert_no_rating_leak(df)

    df['facility_type'] = df['facility_type_code'].map(
        lambda c: FACILITY_TYPE_LABELS.get(c, c))
    if log and leaky.any():
        log(f'Collapsed {int(leaky.sum())} rating-encoding exempt facility '
            f'code(s) -> "{EXEMPT_FACILITY_CODE}" (target-leakage fix).')
    return df


def _assert_no_rating_leak(df):
    """Fail loudly if any facility code still maps to exactly one rating.

    A code that appears on >1 row yet only ever co-occurs with a single ABC
    level is, for modelling purposes, the label in disguise."""
    sub = df[df['abc_level'].map(_clean_str) != '']
    for code, grp in sub.groupby('facility_type_code'):
        if len(grp) > 1 and grp['abc_level'].nunique() == 1:
            raise ValueError(
                f'facility_type_code {code!r} still determines the rating '
                f'({grp["abc_level"].iloc[0]!r} on all {len(grp)} rated rows) '
                f'-- target leakage remains.')


# ---------------------------------------------------------------------------
# inspection date
# ---------------------------------------------------------------------------

def parse_inspection_date(df, reference_date=REFERENCE_DATE, log=None):
    """last_inspection_date (text) -> days_since_inspection (Int64) +
    has_inspection (bool).

    Blank for ~1,067 providers (overwhelmingly the never-rated ones -- ABC only
    inspects participants), so the boolean carries real signal and the Int64
    keeps <NA> rather than a sentinel."""
    parsed = pd.to_datetime(df['last_inspection_date'],
                            format=INSPECTION_DATE_FORMAT, errors='coerce')
    unparsed = parsed.isna() & df['last_inspection_date'].map(_clean_str).ne('')
    if unparsed.any() and log:
        bad = df.loc[unparsed, 'last_inspection_date'].unique()[:5]
        log(f'{int(unparsed.sum())} unparseable inspection date(s), kept as '
            f'<NA>: {list(bad)}')

    days = (reference_date - parsed).dt.days
    df['days_since_inspection'] = pd.array(days, dtype='Int64')
    df['has_inspection'] = parsed.notna()
    return df


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def build_categorical_onehot(df, column, prefix, skip_values=()):
    """One-hot a single-value categorical column over its discovered values.
    Values are discovered from the data, never hardcoded."""
    skip_values = tuple(slug(s) for s in skip_values)
    new_cols = []
    for value in sorted(v for v in df[column].dropna().unique()
                        if isinstance(v, str) and v.strip()):
        s = slug(value)
        if s in skip_values:
            continue
        col_name = f'{prefix}_{s}'
        new_cols.append(col_name)
        df[col_name] = df[column].eq(value)
    return df, new_cols


# ---------------------------------------------------------------------------
# shared tail
# ---------------------------------------------------------------------------

def finalize(df, columns_file, which, key, target, dynamic_prefixes,
             na_as_level=False, valid_target_values=None):
    """Shared tail: order by the stable scaffold + any discovered columns,
    dedup on the grain, filter the target, then drop empty/constant columns.

    na_as_level: when True (raw set), a column counts as informative if its
    presence/absence varies, i.e. NaN is treated as its own value. When False
    (full set) all-NaN columns and single-non-null-value columns are dropped.

    valid_target_values: controls which rows survive on the target.
      - None (complete set): keep every row, including rows whose rating is
        pending ('P') or absent -- both of which arrive here as <NA>.
      - an iterable of allowed values (e.g. (1, 2, 3, 4, 5)): keep only rows
        whose target, coerced to numeric, is one of those values.
    """
    df = df.rename(columns={**ID_RENAME, **TARGET_RENAME})
    df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

    # P1 / P2: direct identifiers, per-provider narratives and coordinates.
    # Dropped BEFORE the scaffold reindex so they cannot survive even if a
    # stale sc_columns.json still lists them.
    df = drop_identifier_columns(df)
    df = drop_leakage_columns(df)

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
STATE = "sc"

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
LEVEL_FAMILY_PREFIXES = ('facilitytype_', 'permittype_')

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
