"""Shared cleaning utilities for the KY (kynect All STARS) provider pipeline.

The records are one row per ProviderCLRNumber, built from two kynect
payloads: the directory search result (populated for every row) and the
per-provider detail lookup (capacity/cost/inspection/DPOC/ongoing-process
fields). The detail columns are empty wherever the detail lookup failed, and
a blank detail cell is indistinguishable from a genuinely empty list, so the
`errors` column is what decides "not fetched" (see detail_missing()).

KY has no delimited multi-value text. It has native Y/N flags, two plain
categoricals (provider_type, provider_status), county, and JSON-list fields:
HoursOfOperationList and ServiceCostList are keyed by a value inside each
record (a day name, an age-group label) and get bespoke parsers; the
history-style lists go through build_json_key_columns.
"""
import re
import json
import numpy as np
import pandas as pd


# ProviderCLRNumber is KY's licensing/certification id (Type I/II licenses
# prefixed "L", Certified FCC homes "C"). NumberOfStars is 0-5, where 0 means
# not participating (opted out via DCC-433, or not yet rated), so 0 appears
# only in the *_complete_* outputs.
ID_RENAME = {"ProviderCLRNumber": "provider_id"}
TARGET_RENAME = {"NumberOfStars": "qr_rating"}

# Internal ids, contact/address metadata, source_county_query (which county
# search found the row, not a provider attribute), the errors column, two
# flags that are constant in the data, and the JSON blobs once decomposed.
NON_FEATURE_COLS = [
    "ProviderId",
    "ProviderName",
    "PhoneNumber",
    "LocationAddressLine1",
    "LocationAddressLine2",
    "LocationStateDescription",
    "source_county_query",
    "errors",
    "NonTraditionalFlag",
    "InfantToddlerFlag",
    "HoursOfOperationList",
    "ServiceCostList",
    "InspectionHistoryListUpdated",
    "DPOCAgreementsListUpdated",
    "OngoingProcessListUpdated",
]

# Native kynect field names -> snake_case; applied early in every clean script.
FIELD_RENAME = {
    "ProviderType": "provider_type",
    "ProviderStatus": "provider_status",
    "LocationCity": "city",
    "LocationCountyDescription": "county",
    "LocationZipCode5": "zip",
    "Transportation": "transportation",
    "Toddler": "toddler",
    "SchoolAge": "school_age",
    "PreSchool": "preschool",
    "Infant": "infant",
    "PreKPartnershipFlag": "prek_partnership",
    "IsSubsidyAccepted": "subsidy_accepted",
    "IsOngoingProcess": "is_ongoing_process",
    "IsAcceditationsAvailable": "accreditation_available",
    "IsFoodPermitAvailable": "food_permit_available",
    "Capacity": "capacity",
    "AddressLatitude": "latitude",
    "AddressLongitude": "longitude",
}

# Native Y/N (blank = missing) flag columns, post-FIELD_RENAME.
FLAG_COLUMNS = [
    "transportation", "toddler", "school_age", "preschool", "infant",
    "prek_partnership", "subsidy_accepted", "is_ongoing_process",
    "accreditation_available", "food_permit_available",
]

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                "Saturday", "Sunday"]

# ServiceCostList's AgeGroup is free text, but only ever one of these 9
# statewide brackets.
AGE_GROUP_SLUGS = {
    "Children under the age of 1": "under1",
    "Children 1 year of age": "age1",
    "Children 2 years of age": "age2",
    "Children 3 years of age": "age3",
    "Children 4 years of age": "age4",
    "Children 5 years of age": "age5",
    "Children ages 6 but under age 8": "age6to8",
    "Children ages 8 but under age 13": "age8to13",
    "Children ages 13 but under age 18": "age13to18",
}


def slug(text):
    text = str(text).strip().lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def safe_json_list(value):
    """Parse a JSON-list cell, returning [] for NaN/blank/unparseable cells."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def apply_field_renames(df):
    return df.rename(columns=FIELD_RENAME)


def clean_zip(df, column='zip'):
    """Zero-padded 5-digit ZIP string from a float (42728.0) or a stringified
    float ("42728.0")."""
    def _fix(x):
        if pd.isna(x) or (isinstance(x, str) and not x.strip()):
            return np.nan
        try:
            return str(int(float(x))).zfill(5)
        except Exception:
            return x if isinstance(x, str) else np.nan
    if column in df.columns:
        df[column] = df[column].map(_fix)
    return df


def yn_to_bool(series):
    """'Y'/'N' -> nullable booleans; anything else -> NA."""
    return series.map({'Y': True, 'N': False}).astype('boolean')


def convert_flags_to_bool(df, columns=FLAG_COLUMNS):
    for col in columns:
        if col in df.columns:
            df[col] = yn_to_bool(df[col])
    return df


def build_categorical_onehot(df, column, prefix, skip_values=()):
    """One-hot a single-value categorical column over its discovered values."""
    skip_values = tuple(slug(s) for s in skip_values)
    new_cols = []
    for value in df[column].dropna().unique():
        if not (isinstance(value, str) and value.strip()):
            continue
        s = slug(value)
        if s in skip_values:
            continue
        col_name = f'{prefix}_{s}'
        new_cols.append(col_name)
        df[col_name] = df[column].eq(value)
    return df, new_cols


def build_json_key_columns(df, column, prefix, sep=' | ', skip_keys=()):
    """One text column per key discovered across a JSON-list column (minus
    skip_keys), joining that key's values across the list (raw set)."""
    parsed = [safe_json_list(v) for v in df[column]]
    keys = []
    for records in parsed:
        for rec in records:
            if isinstance(rec, dict):
                for k in rec:
                    if k not in keys and k not in skip_keys:
                        keys.append(k)
    new_cols = []
    for k in keys:
        col_name = f'{prefix}_{slug(k)}'
        new_cols.append(col_name)
        values = []
        for records in parsed:
            vals = [str(rec[k]).strip() for rec in records
                    if isinstance(rec, dict) and rec.get(k) not in (None, '')]
            values.append(sep.join(vals) if vals else np.nan)
        df[col_name] = values
    return df, new_cols


def parse_hours_of_operation(df, as_bool):
    """HoursOfOperationList (list of {"Day", "ServiceTime"}, closed days
    absent) -> hours_<day>: the ServiceTime text (raw) or open-that-day
    booleans (as_bool=True, full)."""
    per_row = []
    for v in df['HoursOfOperationList']:
        day_map = {}
        for e in safe_json_list(v):
            if isinstance(e, dict) and e.get('Day') in DAYS_OF_WEEK:
                day_map[e['Day']] = e.get('ServiceTime')
        per_row.append(day_map)

    new_cols = []
    for day in DAYS_OF_WEEK:
        col_name = f'hours_{day.lower()}'
        new_cols.append(col_name)
        if as_bool:
            df[col_name] = [day in dm for dm in per_row]
        else:
            df[col_name] = [dm.get(day) if day in dm else np.nan
                             for dm in per_row]
    return df, new_cols


def parse_service_cost(df):
    """ServiceCostList -> numeric cost_<age>_fulltime / cost_<age>_parttime,
    one pair per AGE_GROUP_SLUGS bracket; the same for raw and full."""
    per_row = []
    for v in df['ServiceCostList']:
        bucket = {}
        for e in safe_json_list(v):
            if isinstance(e, dict) and e.get('AgeGroup') in AGE_GROUP_SLUGS:
                bucket[AGE_GROUP_SLUGS[e['AgeGroup']]] = e
        per_row.append(bucket)

    new_cols = []
    for age_slug in AGE_GROUP_SLUGS.values():
        for cost_key, suffix in (('FullTimeCost', 'fulltime'),
                                  ('PartTimeCost', 'parttime')):
            col_name = f'cost_{age_slug}_{suffix}'
            new_cols.append(col_name)
            df[col_name] = pd.to_numeric(pd.Series([
                bucket[age_slug].get(cost_key) if age_slug in bucket else np.nan
                for bucket in per_row
            ]), errors='coerce')
    return df, new_cols


# Per-row failure reason; empty on exactly the rows whose detail payload was
# parsed.
CRAWL_ERROR_COL = 'errors'


def detail_missing(df):
    """Rows whose provider-detail payload never arrived, so their counts are
    NA rather than a measured zero. Warns if the errors column is absent."""
    if CRAWL_ERROR_COL not in df.columns:
        print(f'[ky][WARNING] no {CRAWL_ERROR_COL!r} column: a never-fetched '
              f'detail payload will be counted as a measured zero')
        return pd.Series(False, index=df.index)
    return (df[CRAWL_ERROR_COL].notna()
            & (df[CRAWL_ERROR_COL].astype(str).str.strip() != ''))


def inspection_counts(df):
    """InspectionHistoryListUpdated -> num_inspections,
    num_distinct_inspection_types (full set); NA where detail_missing()."""
    missing = detail_missing(df)
    parsed = [safe_json_list(v) for v in df['InspectionHistoryListUpdated']]
    df['num_inspections'] = pd.array([len(recs) for recs in parsed],
                                     dtype='Int64')
    df['num_distinct_inspection_types'] = pd.array(
        [len({rec.get('InspectionType') for rec in recs
              if isinstance(rec, dict)})
         for recs in parsed], dtype='Int64')
    df.loc[missing, ['num_inspections',
                     'num_distinct_inspection_types']] = pd.NA
    return df


def dpoc_counts(df):
    """DPOCAgreementsListUpdated -> num_dpoc_agreements (full set); NA where
    detail_missing()."""
    missing = detail_missing(df)
    df['num_dpoc_agreements'] = pd.array(
        [len(safe_json_list(v)) for v in df['DPOCAgreementsListUpdated']],
        dtype='Int64')
    df.loc[missing, 'num_dpoc_agreements'] = pd.NA
    return df


def ongoing_process_counts(df):
    """OngoingProcessListUpdated -> num_ongoing_processes plus presence flags
    for the two ProcessType values (full set).

    IsOngoingProcess is populated for every row and agrees with the detail
    payload wherever both exist, so an 'N' still supports a zero on a row
    whose detail fetch failed; only failed non-'N' rows become NA.

    The flag must still be 'Y'/'N' text here: convert_flags_to_bool() runs
    after this in ky_clean_full.py.
    """
    missing = detail_missing(df)
    parsed = [safe_json_list(v) for v in df['OngoingProcessListUpdated']]
    df['num_ongoing_processes'] = pd.array([len(recs) for recs in parsed],
                                           dtype='Int64')
    df['ongoing_has_adverse_action'] = pd.array([
        any(rec.get('ProcessType') == 'Adverse Action' for rec in recs
            if isinstance(rec, dict))
        for recs in parsed], dtype='boolean')
    df['ongoing_has_dpoc'] = pd.array([
        any(rec.get('ProcessType') == 'Directed Plan of Correction'
            for rec in recs if isinstance(rec, dict))
        for recs in parsed], dtype='boolean')

    flag = df.get('is_ongoing_process')
    if flag is None:
        flag = df.get('IsOngoingProcess')
    if flag is None:
        unknown = missing
    else:
        unknown = missing & (flag.astype(str).str.strip().str.upper() != 'N')
    df.loc[unknown, ['num_ongoing_processes', 'ongoing_has_adverse_action',
                     'ongoing_has_dpoc']] = pd.NA
    return df


# The seven hours_<day> columns are one attribute measured seven times, so
# they need a joint k-anonymity decision rather than a per-column one.
HOURS_COLS = [f'hours_{d.lower()}' for d in DAYS_OF_WEEK]


def collapse_hours_consistently(df, min_n=None, other=None):
    """k >= min_n for the hours family, enforced across days as well as
    within one.

    A per-column sweep can suppress a ServiceTime on Monday but keep it on
    Wednesday, and since most providers keep one pattern all week, the
    suppressed day is then readable off a kept day of the same row. Fixed
    point of two rules:
      (1) a (day, pattern) pair held by fewer than min_n providers is
          suppressed;
      (2) a pattern suppressed on ANY day of a row is suppressed on EVERY day
          of that row.
    Rule 2 shrinks the counts rule 1 measures, so the two are iterated until
    nothing moves.

    Raw view only: in full every hours column is a constant True that
    finalize() drops.
    """
    min_n = K_ANON if min_n is None else min_n
    other = OTHER_LABEL if other is None else other
    cols = [c for c in HOURS_COLS if c in df.columns]
    if not cols:
        return df

    values = {c: df[c].astype(object).where(df[c].notna()) for c in cols}
    suppressed = {c: pd.Series(False, index=df.index) for c in cols}

    for _ in range(50):                       # bounded
        changed = False

        for c in cols:                        # rule 1 -- the per-day floor
            live = values[c].where(~suppressed[c])
            counts = live.dropna().astype(str).value_counts()
            thin = set(counts[counts < min_n].index)
            if not thin:
                continue
            hit = live.notna() & live.astype(str).isin(thin)
            if hit.any():
                suppressed[c] = suppressed[c] | hit
                changed = True

        # rule 2 -- positional, because the frame's index is a filtered
        # subset of the original range.
        cells = {c: values[c].to_numpy(dtype=object) for c in cols}
        flags = {c: suppressed[c].to_numpy(dtype=bool) for c in cols}
        banned = [set() for _ in range(len(df))]
        for c in cols:
            cell, flag = cells[c], flags[c]
            for i in range(len(df)):
                if flag[i] and not pd.isna(cell[i]):
                    banned[i].add(str(cell[i]))
        for c in cols:
            cell, flag = cells[c], flags[c]
            hit = np.array([(not flag[i] and not pd.isna(cell[i])
                             and str(cell[i]) in banned[i])
                            for i in range(len(df))], dtype=bool)
            if hit.any():
                suppressed[c] = suppressed[c] | pd.Series(hit, index=df.index)
                changed = True

        if not changed:
            break

    total, kept_values = 0, set()
    for c in cols:
        out = values[c].where(~suppressed[c], other)
        # An `other` bucket itself under the floor would re-isolate the
        # providers it hides, so it is nulled (as in collapse_rare_values()).
        marked = int((out.astype(str) == other).sum())
        if 0 < marked < min_n:
            out = out.where(out.astype(str) != other, np.nan)
            marked = 0
        total += marked
        kept_values |= {v for v in out.dropna().astype(str).unique()
                        if v != other}
        df[c] = out
    _privacy_log(f"[P3] hours_*: re-collapsed jointly across {len(cols)} day(s) "
                 f"at k>={min_n} -- {len(kept_values)} pattern(s) kept, "
                 f"{total} cell(s) suppressed. This REPLACES the per-column "
                 f"hours_* result logged above, which left a suppressed day "
                 f"readable off a kept day of the same row.")
    return df


def finalize(df, columns_file, which, key, target, dynamic_prefixes,
             na_as_level=False, valid_target_values=None):
    """Shared tail: order by the stable scaffold + any discovered columns,
    dedup on the grain, filter the target, then drop empty/constant columns.

    na_as_level: when True (raw set), a column counts as informative if its
    presence/absence varies, i.e. NaN is treated as its own value. When
    False (full set), drop all-NaN columns and columns with one non-null value.

    valid_target_values: None (complete set) keeps every row; an iterable of
    allowed values keeps only rows whose numeric target is one of them.
    """
    df = df.rename(columns={**ID_RENAME, **TARGET_RENAME})
    df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

    # Direct identifiers and coordinates are dropped before the scaffold
    # reindex, so they cannot survive even if ky_columns.json lists them.
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

    # Geography and the k-anonymity sweep run after the dedup and the target
    # filter, because k is a property of the rows that actually ship.
    #
    # The hours family is re-collapsed jointly (collapse_hours_consistently)
    # from its pre-remediation values, restored afterwards rather than
    # exempted via protect=..., which would also exempt it from the flag rules.
    hours_cols = ([c for c in HOURS_COLS if c in df.columns]
                  if which == 'raw' else [])
    hours_before = df[hours_cols].copy() if hours_cols else None
    order = list(df.columns)

    df = apply_privacy_remediation(df, which)

    if hours_cols:
        for column in hours_cols:
            df[column] = hours_before[column]
        df = collapse_hours_consistently(df)
        # A column dropped by the sweep and restored here goes back in place.
        kept = [c for c in order if c in df.columns]
        added = sorted(c for c in df.columns if c not in order)
        df = df[kept + added]

    cols_to_drop = []
    for col in df.columns:
        if df[col].isna().all():
            cols_to_drop.append(col)
        elif df[col].nunique(dropna=not na_as_level) == 1:
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)


# >>> BEGIN PRIVACY BLOCK (shared verbatim by all 12 states) >>>
import json as _json
import random as _random
import re as _re
from pathlib import Path as _Path

import numpy as _np
import pandas as _pd

# --- per-state ----------------------------------------------------------------
STATE = "ky"

# Small-range integer counts capped so a lone provider at the top of the range
# joins the bucket below; lossless for a tree splitting on ">= cap".
TOPCODE_COLS = {}

# Families whose members are LEVELS of one attribute, so a rare member can be
# merged into "<prefix>other". Prefixes whose members are distinct attributes
# (has_*, num_*, violations_*, monitoring_*, ...) are deliberately excluded:
# OR-ing them together would invent a meaningless feature.
LEVEL_FAMILY_PREFIXES = ()

PROTECTED_COLS = ("provider_id", "qr_rating")
# -----------------------------------------------------------------------------

K_ANON = 5            # minimum providers sharing any surviving category / flag
MAX_LEVELS = 50       # cap on columns emitted when decomposing a list column
OTHER_LABEL = "other"
MULTIVALUE_DELIM = "|"

GEO_K_ANON = K_ANON

# Released geography is region-scale only: `region` always ships where a state
# has one; `county` only where there is no region; city, ZIP and school
# district never ship.
GEO_ALWAYS_KEEP = ("region",)
GEO_KEEP_IF_NO_REGION = ("county",)

# Values that read_csv turns back into NaN; a cell holding one of these exactly
# would reload as missing.
_NA_TOKENS = {
    "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
    "n/a", "nan", "null", "none_",
}

# Loose on purpose: states mix m/d/Y, Y-m-d and d-m-Y, and we only need to
# know that a column holds dates.
_DATE_RE = _re.compile(r"\d{1,4}[-/]\d{1,2}[-/]\d{2,4}")

# Private store for the geographic label map, which must not ship. Gitignored.
_PRIVATE_DIR = _Path(__file__).resolve().parent.parent / "data-private"

# Direct identifiers, dropped from every state so a column reintroduced
# upstream cannot silently ship. The narratives are near-unique per provider
# and embed names, phones and addresses, so the prose itself identifies.
PII_COLS = (
    # CO -- inspection / complaint / injury / adverse-action narratives and the
    # on-site licence number.
    "complaints_text", "injury_investigations_text", "inspection_report_text",
    "stage_ii_text", "adverse_actions_text", "license_number_on_site",
    # MT -- the program's trading name.
    "program_name",
    # NC -- violation narratives.
    "violations_text",
    # OK -- monitoring narrative, the per-provider report URL (itself a
    # registry key), and the per-visit compliance vector.
    "monitoring_areas_compliant", "monitoring_report_url",
    "monitoring_section_text",
    # WA -- contact names, emails, phones, websites and the licence-history key.
    "contact_email", "contact_phone", "contact_website",
    "contacts_email", "contacts_full_name", "contacts_phone",
    "licensehist_license_id",
)

# Removed outright: at provider density coordinates locate a building.
COORD_COLS = ("latitude", "longitude")

# Matched by exact name so look-alikes are not caught: CO's
# `school_district_operated_program` is a governance flag, not a district.
GEO_LABEL_COLS = (
    "zip", "zip_code", "zipcode", "location_zip", "socrata_physicalzip",
    "county", "dhhs_county",
    "city", "location_city", "socrata_physicalcity",
    "school_district", "district", "region", "ccrr_region",
)

# Geographic families one-hot encoded before finalize runs; gathered back into
# a single series, put through the same pipeline, and re-emitted.
GEO_ONEHOT_PREFIXES = ("county_", "city_", "region_", "district_")

# Coarse fields only are one-hot encoded into `full`.
GEO_FULL_FIELDS = ("county", "dhhs_county", "region", "ccrr_region", "district")

# Same field under different names, so it lands as geo_county / geo_region in
# every state and both views. WA's location_*/socrata_* pairs are genuinely
# different sources and are not aliased.
GEO_FIELD_ALIASES = {
    "ccrr_region": "region",
    "dhhs_county": "county",
    "zip_code": "zip",
    "zipcode": "zip",
}

# Coarse enough to identify nobody, so kept readable rather than relabelled.
GEO_READABLE = ("region", "ccrr_region")


def _privacy_log(message, state=None):
    """Append to {state}_privacy_log.txt next to the utils module and print."""
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
    """True if the non-null values are only bools, 0/1 or 'True'/'False'
    (the shapes a boolean takes after a CSV round-trip)."""
    values = series.dropna()
    if values.empty:
        return False
    return set(_pd.unique(values)) <= {True, False, 0, 1, 0.0, 1.0,
                                       "True", "False"}


def _positive_mask(series):
    """Row mask for 'this flag is set', for a boolean or a raw-style level
    column (item text where present, else NaN)."""
    if _is_boolish(series):
        return series.map({True: True, False: False, 1: True, 0: False,
                           1.0: True, 0.0: False,
                           "True": True, "False": False}).fillna(False).astype(bool)
    return series.notna()


def _positives(series):
    return int(_positive_mask(series).sum())


def _is_level_column(series):
    """Boolean, or exactly one distinct value with NaN elsewhere. A shared
    prefix alone is not enough (GA's operating_hours vs operating_months_*)."""
    return _is_boolish(series) or (
        series.nunique(dropna=True) == 1 and series.isna().any())


def _label_or_nan(mask, label=OTHER_LABEL):
    """Object array holding `label` where mask is set, NaN elsewhere.

    np.where(mask, "other", np.nan) raises DTypePromotionError on numpy 2.x.
    """
    out = _np.empty(len(mask), dtype=object)
    out[:] = _np.nan
    out[_np.asarray(mask, dtype=bool)] = label
    return out


def _private_path(name):
    _PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    return _PRIVATE_DIR / name


def neutralize_na_tokens(df, state=None):
    """Rewrite cells that read_csv would turn back into NaN ("None" -> "none",
    or a trailing underscore if still colliding)."""
    changed = []
    for column in df.columns:
        series = df[column]
        # Not `dtype != object`: pandas 3 gives text columns the `str` dtype,
        # which an object-only guard would silently skip.
        if not (_pd.api.types.is_object_dtype(series)
                or _pd.api.types.is_string_dtype(series)):
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
    columns = TOPCODE_COLS if columns is None else columns
    for column, cap in columns.items():
        if column not in df.columns:
            continue
        values = _pd.to_numeric(df[column], errors="coerce")
        above = int((values > cap).sum())
        if not above:
            continue
        df[column] = values.clip(upper=cap)
        _privacy_log(f"[topcode] {column}: top-coded at {cap} ({above} row(s) capped)",
                     state)
    return df


def drop_identifier_columns(df, state=None, log=None):
    """Drop identifiers, narratives and coordinates. Called before the scaffold
    reindex so they cannot reach the output even if a scaffold lists them."""
    targets = [c for c in df.columns
               if c in PII_COLS or c.lower() in COORD_COLS]
    if targets:
        _privacy_log(f"[identifiers] dropped {len(targets)} identifier/coordinate "
                     f"column(s): {sorted(targets)}", state)
    return df.drop(columns=targets)


def collapse_rare_level_family(df, prefix, min_n=K_ANON, state=None):
    """OR levels held by fewer than min_n providers into <prefix>other, in the
    shape (boolean or raw text) the family already uses."""
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
    _privacy_log(f"[k-anon] {prefix}*: merged {len(rare)} level(s) with <{min_n} "
                 f"providers into {other_col} (n={int(mask.sum())})", state)

    # A thin bucket would re-isolate the providers the merge was meant to hide.
    if 0 < _positives(df[other_col]) < min_n:
        _privacy_log(f"[k-anon] {other_col}: still <{min_n} after merging, dropped",
                     state)
        df = df.drop(columns=[other_col])
    return df


def collapse_rare_values(series, min_n=K_ANON, other=OTHER_LABEL):
    """Values held by fewer than min_n rows become `other`; if `other` is itself
    thin it is nulled. Folding it into another group would assert a value the
    provider does not hold."""
    values = series.dropna().astype(str)
    counts = values.value_counts()
    rare = set(counts[counts < min_n].index)
    if not rare:
        return series
    out = series.astype(object).where(~series.astype(str).isin(rare), other)

    recounted = out.dropna().astype(str).value_counts()
    if other in recounted.index and recounted[other] < min_n:
        out = out.where(out.astype(str) != other, _np.nan)
    return out


def classify_text_column(series, min_n=K_ANON):
    """Classify a text column that fails k>=min_n as one of:

      "date"   -- dates; the text copy is near-unique and dropped.
      "list"   -- a pipe-joined list, unique only as a combination; decomposed
                  into one column per common item.
      "single" -- a categorical with a long tail; the tail is collapsed.
      "prose"  -- per-provider narrative; dropped.

    Returns (kind, kept_items), kept_items most frequent first.
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
            # Mostly "other" after collapsing, so treat it as prose.
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
    kind, kept = classify_text_column(df[column], min_n)

    if kind in ("date", "prose"):
        _privacy_log(f"[k-anon] {column}: dropped ({kind}, cannot reach k>={min_n})",
                     state)
        return df.drop(columns=[column])

    if kind == "single":
        was_present = df[column].notna()
        df[column] = collapse_rare_values(df[column], min_n)
        suppressed = int((df[column].isna() & was_present).sum())
        blanks = int(df[column].isna().sum())
        if suppressed and blanks < min_n and df[column].dropna().nunique() < 2:
            # One value plus a blank set below the floor distinguishes only
            # which providers were suppressed. Columns that still separate
            # real groups, or whose blanks are mostly genuine missingness, are
            # kept with the tail blanked.
            _privacy_log(f"[k-anon] {column}: one surviving value and a blank set "
                         f"of {blanks} (< {min_n}) after suppressing "
                         f"{suppressed} row(s) -- dropped", state)
            return df.drop(columns=[column])
        _privacy_log(f"[k-anon] {column}: collapsed tail into '{OTHER_LABEL}' "
                     f"({len(kept)} value(s) kept"
                     f"{f', {suppressed} row(s) below the floor nulled' if suppressed else ''})",
                     state)
        return df

    # "list": one text column per surviving item, capped at max_levels; the
    # tail and sub-threshold items go to <column>_other.
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
    # A thin tail bucket would re-isolate the providers it was meant to hide.
    kept_other = sum(has_other) >= min_n
    if kept_other:
        df[other_col] = _label_or_nan(has_other)
    _privacy_log(f"[k-anon] {column}: decomposed into {len(used)} item column(s)"
                 f"{f' + {other_col}' if kept_other else ''} "
                 f"({len(tail)} tail item(s) folded"
                 f"{f', {sum(has_other)} row(s) below the floor dropped' if any(has_other) and not kept_other else ''})",
                 state)
    return df.drop(columns=[column])


def _gather_onehot_family(df, prefix):
    """Reverse a one-hot family back into a single label series."""
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
    """Collapse -> relabel -> one-hot, in that order: collapsing defeats
    isolation (no group smaller than min_n), relabelling defeats lookup.

    raw gets one pseudonymous label column per field; full gets one boolean
    per surviving label. The value -> label map is written to data-private/
    and never ships.
    """
    state = state or STATE
    mapping = _load_geo_map(state)
    rng = _random.Random()

    sources, seen = [], set()
    for col in GEO_LABEL_COLS:
        if col in df.columns:
            name = GEO_FIELD_ALIASES.get(col, col)
            # Alias only when the canonical name is free, so WA's two city
            # columns do not overwrite each other.
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

    # An all-empty region does not count, or the state would ship no geography.
    has_region = any(name in GEO_ALWAYS_KEEP and not series.dropna().empty
                     for name, series, _ in sources)

    for name, series, drop_cols in sources:
        values = series.dropna().astype(str)
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        if values.empty:
            continue

        if name not in GEO_ALWAYS_KEEP and not (
                name in GEO_KEEP_IF_NO_REGION and not has_region):
            reason = ("superseded by region" if name in GEO_KEEP_IF_NO_REGION
                      else "finer than region scale")
            _privacy_log(f"[geo] {name}: dropped from {which} ({reason})", state)
            continue

        if which == "full" and name not in GEO_FULL_FIELDS:
            _privacy_log(f"[geo] {name}: fine-grained, not one-hotted into full",
                         state)
            continue

        # 1. COLLAPSE
        collapsed = collapse_rare_values(series.astype(object), min_n)
        surviving = sorted(collapsed.dropna().astype(str).unique())

        # 2. RELABEL -- integer labels from a random permutation, reused across
        #    runs so labels stay stable. Looked up on the slug, not the literal
        #    value: raw sees "Denver" but full rebuilds "denver" from one-hot
        #    column names, and both must get the same label.
        readable = name in GEO_READABLE
        if readable:
            # Strip a redundant field-name prefix ("Region 1" -> geo_region_1).
            def _label(value):
                slug = _privacy_slug(value)
                prefix = f"{name}_"
                return slug[len(prefix):] if slug.startswith(prefix) else slug
            known = {v: _label(v) for v in surviving}
            by_key = {_privacy_slug(v): lab for v, lab in known.items()}
        else:
            known = mapping.get(name, {})
            by_key = {}
            for value, label in known.items():
                by_key.setdefault(_privacy_slug(value), label)
            unlabelled = [v for v in surviving if _privacy_slug(v) not in by_key]
            if unlabelled:
                free = [i for i in range(len(surviving) + len(known))
                        if i not in set(known.values())]
                rng.shuffle(free)
                for value, label in zip(unlabelled, free):
                    known[value] = label
                    by_key[_privacy_slug(value)] = label
                mapping[name] = known

        # Explicit object Series: .map() with missing values infers float64 and
        # would render label 4 as "county_4.0".
        labels = _pd.Series(
            [by_key.get(_privacy_slug(v)) if _pd.notna(v) else None
             for v in collapsed],
            index=collapsed.index, dtype=object)

        # 3. ONE-HOT (full) or single label column (raw)
        if which == "full":
            for value in surviving:
                label = by_key[_privacy_slug(value)]
                df[f"geo_{name}_{label}"] = (labels == label).fillna(False).to_numpy()
        else:
            df[f"geo_{name}"] = labels.map(
                lambda v: f"{name}_{v}" if v is not None and _pd.notna(v) else _np.nan)

        _privacy_log(f"[geo] {name}: {len(values.unique())} value(s) -> "
                     f"{len(surviving)} label(s) at k>={min_n} "
                     f"({'one-hot' if which == 'full' else 'label column'}"
                     f"{', readable' if readable else ''})", state)

    if mapping:
        _save_geo_map(state, mapping)
    return df


def enforce_k_anonymity(df, which, state=None, level_prefixes=None,
                        protect=PROTECTED_COLS, min_n=K_ANON,
                        max_levels=MAX_LEVELS):
    """Every categorical value and flag ends up shared by >= min_n providers.
    Run on the released rows, i.e. after dedup and the target filter."""
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
            # Already folded in step 1.
            continue
        df = remediate_text_column(df, column, min_n, max_levels, state)

    # 3. thin flags outside any family: nothing to merge into, so dropped.
    for column in [c for c in df.columns if c not in protect]:
        if column not in df.columns or not _is_boolish(df[column]):
            continue
        if family_member(column):
            continue
        positives = _positives(df[column])
        if 0 < positives < min_n:
            _privacy_log(f"[k-anon] {column}: flag set for {positives} provider(s), "
                         f"no family to merge into -- dropped", state)
            df = df.drop(columns=[column])

    # 4. flags whose minority class (True or False) is below the floor: a flag
    #    true for all but one provider isolates that one.
    for column in [c for c in df.columns if c not in protect]:
        if column not in df.columns or not _is_boolish(df[column]):
            continue
        mask = _positive_mask(df[column])
        present = int(df[column].notna().sum())
        positives = int(mask.sum())
        minority = min(positives, present - positives)
        if 0 < minority < min_n:
            _privacy_log(f"[k-anon] {column}: minority class is {minority} "
                         f"provider(s) of {present} -- dropped", state)
            df = df.drop(columns=[column])
    return df


def apply_privacy_remediation(df, which, state=None, level_prefixes=None,
                              protect=PROTECTED_COLS, min_n=K_ANON,
                              max_levels=MAX_LEVELS, log=None):
    """Geography, then the k-anonymity sweep. Call from finalize() after the
    row filters; call drop_identifier_columns() before the scaffold reindex."""
    state = state or STATE
    order = list(df.columns)
    df = neutralize_na_tokens(df, state)
    df = topcode_counts(df, state)
    df = build_geo_features(df, which, state, max(min_n, GEO_K_ANON), log)
    df = enforce_k_anonymity(df, which, state, level_prefixes, protect,
                             min_n, max_levels)
    # Keep the scaffold order (provider_id, qr_rating first) and append newly
    # derived columns after it, sorted.
    kept = [c for c in order if c in df.columns]
    added = sorted(c for c in df.columns if c not in order)
    return df[kept + added]

# <<< END PRIVACY BLOCK <<<
