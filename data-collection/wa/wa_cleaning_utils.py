"""Shared cleaning utilities for the WA provider pipelines.

Washington's records merge three sources: the Child Care Check Apex-Remoting
API, the server-rendered PSS_Provider detail page, and the DCYF Socrata open
dataset. The multi-value / nested fields have an open-ended category space, so
both the raw and full pipelines discover the schema at runtime: they split each
field on its delimiter, collect the unique items across all rows, and emit one
column per discovered item.

TARGET
------
`Early_Achiever_Status_Internal__c` is the rating column. It mixes the five
Early Achievers levels with non-rating statuses:

    Level 2 / Level 3 / Level 3+ / Level 4 / Level 5
    Participating, not yet rated | Not Enrolled | Withdrawn | Rating Expired
    Defaulted | Terminated | (missing)

`normalize_rating()` reduces it to a numeric level, collapsing the streamlined
"Level 3+" pathway into 3 and mapping every non-rating status to NaN. There is
no Level 1 in the data because Level 1 *is* simply holding a license. Of the
three rating encodings it agrees best with the independent Socrata `earating`
and is the most complete.

LEAKAGE
-------
`Early_Achiever_Status_External__c`, `detail_ea_status`, `socrata_earating` and
`socrata_eaparticipation` are alternate encodings of the target and are dropped
via NON_FEATURE_COLS.
"""
import re
import json
import numpy as np
import pandas as pd


# `Id` is the 18-char Salesforce Account Id (identical to Socrata's
# `wacompassid`): the unique grain.
ID_RENAME = {"Id": "provider_id"}

TARGET_RENAME = {"Early_Achiever_Status_Internal__c": "qr_rating"}

# The records also carry a `provider_id` column holding the real Account Id.
# It would collide with ID_RENAME's output, so finalize() drops it first.
CRAWLER_GRAIN_COPY = "provider_id"

# Salesforce __c API names -> snake_case; `detail_` / `socrata_` columns keep
# their source prefix.
FEATURE_RENAME = {
    "Latest_License_Facility_Type_Name__c": "facility_type",
    "Provider_Status__c": "provider_status",
    "Latest_License_Status__c": "license_status",
    "license_rec_License_Type__c": "license_type",
    "Latest_License_Age_Group_Served__c": "age_groups_served",
    "Provider_Slot_Availability__c": "slot_availability",
    "Early_Achievers_Areas_Of_Specialization__c": "ea_specialization",
    "Languages_Spoken__c": "languages_spoken",
    "Languages_of_Instruction__c": "languages_of_instruction",
    "Ages_Served__c": "ages_served",
    "Total_Available_Slots__c": "total_available_slots",
    "Subsidy_Participation__c": "subsidy_participation",
    "Food_Program_Participation__c": "food_program_participation",
    "Is_Funded_HS__c": "funded_head_start",
    "Is_Funded_EHS__c": "funded_early_head_start",
    "Is_Funded_ECEAP__c": "funded_eceap",
    "Is_Formerly_Licensed_For_CCC__c": "formerly_licensed",
    "Is_Unlawful_Care_For_CCC__c": "unlawful_care",
    "Emergency_License__c": "emergency_license",
    "Exclude_FH_provider_on_public_search__c": "excluded_from_public_search",
    "Show_Tribal_EA_On_CCC__c": "tribal_ea_shown",
    "Display_on_Map_View__c": "display_on_map",
    "Opt_In_Map_View__c": "opt_in_map",
    "Physical_Geolocation__Latitude__s": "latitude",
    "Physical_Geolocation__Longitude__s": "longitude",
    "Certifications__c": "certifications",
    "Tribal_Info__c": "tribal_info",
    "license_rec_Update_Approved__c": "license_update_approved",
    "license_rec_Before_approval_reject_license_capacity__c": "license_prior_capacity",
    "detail_licensed_capacity": "licensed_capacity",
    "detail_initial_license_date": "initial_license_date",
    "detail_school_district": "school_district",
    "socrata_physicalcounty": "county",
    "socrata_region": "region",
    "socrata_licensecertificatetypedesc": "license_certificate_type",
    "socrata_latestoperatingstatus": "socrata_operating_status",
}

NON_FEATURE_COLS = [
    # --- target leakage: other encodings of the Early Achievers rating
    "Early_Achiever_Status_External__c",
    "detail_ea_status",
    "detail_ea_specialization",
    "socrata_earating",
    "socrata_eaparticipation",
    # --- redundant / alternate identifiers
    "found_zip",
    "RecordTypeId",
    "Latest_License_Rec_ID__c",
    "license_rec_Id",
    "detail_provider_id",
    "detail_license_number",
    "license_id_current",
    "socrata_famlinkid",
    "socrata_sspsprovidernumber",
    # --- names, addresses, contacts, logos: free text, not features
    "Account_Name_Display_Label__c",
    "Provider_Logo__c",
    "Physical_Street_Display_Label__c",
    "detail_license_name",
    "detail_email",
    "detail_website",
    "detail_primary_contact",
    "socrata_providername",
    "socrata_doingbusinessas",
    "socrata_primarycontactemail",
    "socrata_primarycontactpersonname",
    "socrata_primarycontactphonenumber",
    "socrata_primarylicensor",
    "socrata_physicalstreetaddress",
    "socrata_providershippingaddress",
    "socrata_providershippingcity",
    "socrata_providershippingstate",
    "socrata_shippingzip",
    "socrata_geocodedphysicaladdress",
    # --- duplicated by a better-populated column, or collection metadata
    "Provider_Status_External__c",
    "Emergency_License_Search__c",
    "detail_facility_type",
    "detail_provider_status",
    "detail_total_available_slots",
    "detail_age_groups_of_available_slots",
    "detail_certifications",
    "detail_tribal_information",
    "detail_languages_spoken",
    "detail_languages_of_instruction",
    "detail_license_status",
    "detail_license_type",
    "detail_ages",
    "errors",
    # --- nested blobs and their counts, superseded by the decomposed columns /
    # json_counts(). Dropped explicitly because their names share the raw
    # dynamic prefixes and would otherwise be re-admitted as discovered extras.
    "complaints_json",
    "inspections_json",
    "license_history_json",
    "contacts_json",
    "complaints_count",
    "inspections_count",
    "license_history_count",
    "contacts_count",
]

# Columns whose values are Salesforce 'True'/'False' or the page's 'Yes'/'No'.
BOOLEAN_COLS = [
    "subsidy_participation", "food_program_participation",
    "funded_head_start", "funded_early_head_start", "funded_eceap",
    "formerly_licensed", "unlawful_care", "emergency_license",
    "excluded_from_public_search", "tribal_ea_shown",
    "display_on_map", "opt_in_map", "license_update_approved",
    "PSS_Add_to_my_list_logic__c",
]

_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


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
    if column not in df.columns:
        return df, []
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

    new_cols = []
    for s in sorted(schema):
        phrase = schema[s]
        col_name = f'{prefix}_{s}'
        new_cols.append(col_name)
        present = [phrase in items for items in per_row]
        if as_bool:
            df[col_name] = present
        else:
            df[col_name] = [phrase if p else np.nan for p in present]
    return df, new_cols


def build_categorical_onehot(df, column, prefix, skip_values=()):
    """One-hot a single-value categorical column over its discovered values."""
    if column not in df.columns:
        return df, []
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


def build_json_key_columns(df, column, prefix, sep=' | '):
    """Discover the union of keys across a JSON-list column and emit one text
    column per key, joining that key's values across the list (raw set)."""
    if column not in df.columns:
        return df, []
    parsed = [safe_json_list(v) for v in df[column]]
    keys = []
    for records in parsed:
        for rec in records:
            if isinstance(rec, dict):
                for k in rec:
                    if k not in keys:
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


_LEVEL_RE = re.compile(r'level\s*(\d)(\+?)\s*$', re.IGNORECASE)


def normalize_rating(df, column='Early_Achiever_Status_Internal__c', log=None):
    """Reduce the Early Achievers status to a numeric level, in place.
    "Level 3+" collapses to 3; every non-rating status becomes NaN."""
    if column not in df.columns:
        return df
    levels, dropped = [], {}
    for v in df[column]:
        if not isinstance(v, str):
            levels.append(np.nan)
            continue
        m = _LEVEL_RE.search(v.strip())
        if m:
            levels.append(int(m.group(1)))
        else:
            levels.append(np.nan)
            dropped[v.strip()] = dropped.get(v.strip(), 0) + 1
    df[column] = pd.array(levels, dtype='Int64')
    if log:
        for status, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
            log(f'  non-rating status -> NaN: {status!r} ({n} rows)')
    return df


def _to_months(text):
    """'birth' -> 0; '12 months' -> 12; '6 years 0 months' -> 72; '5 years' -> 60."""
    if not isinstance(text, str):
        return np.nan
    t = text.strip().lower()
    if not t or t == '-':
        return np.nan
    if t.startswith('birth'):
        return 0
    years = re.search(r'(\d+)\s*year', t)
    months = re.search(r'(\d+)\s*month', t)
    if not years and not months:
        return np.nan
    return (int(years.group(1)) * 12 if years else 0) + \
           (int(months.group(1)) if months else 0)


def parse_ages_served(df, column='ages_served', log=None):
    """'12 months - 13 years 0 months' -> age_min_months / age_max_months."""
    lo, hi = [], []
    for x in df.get(column, pd.Series(dtype=object)):
        if not isinstance(x, str) or '-' not in x:
            lo.append(np.nan)
            hi.append(np.nan)
            continue
        try:
            a, b = x.split('-', 1)
            lo.append(_to_months(a))
            hi.append(_to_months(b))
        except Exception:
            if log:
                log(f'Error parsing {column}: {x!r}')
            lo.append(np.nan)
            hi.append(np.nan)
    df['age_min_months'] = pd.array(lo, dtype='Int64')
    df['age_max_months'] = pd.array(hi, dtype='Int64')
    return df


_CONTACT_SPLIT = ' / '
_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
_PHONE_RE = re.compile(r'^[\(\)\d\s\-\.x]+$')


def parse_contact_blob(df, column='PS_Provider_ContactUs_Label__c', log=None):
    """'(206) 625-0842 / lesa@x.org / https://x.org/' -> phone / email / website.
    Parts are identified by shape, not position: the website is often absent."""
    phones, emails, sites = [], [], []
    for x in df.get(column, pd.Series(dtype=object)):
        p = e = w = np.nan
        if isinstance(x, str) and x.strip():
            for part in x.split(_CONTACT_SPLIT):
                part = part.strip()
                if not part:
                    continue
                if _EMAIL_RE.match(part):
                    e = part
                elif _PHONE_RE.match(part) and any(c.isdigit() for c in part):
                    p = part
                else:
                    w = part
        phones.append(p)
        emails.append(e)
        sites.append(w)
    df['contact_phone'] = phones
    df['contact_email'] = emails
    df['contact_website'] = sites
    df['has_phone'] = [isinstance(v, str) for v in phones]
    df['has_email'] = [isinstance(v, str) for v in emails]
    df['has_website'] = [isinstance(v, str) for v in sites]
    return df


def parse_location(df, column='Provider_Location_Label__c', log=None):
    """'2057 Kibler Ave<br>Enumclaw, WA 98022' -> location_city / location_zip.
    Family homes show just 'City, WA ZIP', so the last line is always read."""
    cities, zips = [], []
    for x in df.get(column, pd.Series(dtype=object)):
        c = z = np.nan
        if isinstance(x, str) and x.strip():
            last = x.split('<br>')[-1].strip()
            m = re.match(r'^(.*?),\s*([A-Z]{2})\s*(\d{5})(?:-\d{4})?$', last)
            if m:
                c, z = m.group(1).strip(), m.group(3)
            elif log:
                log(f'Unparsed location: {last!r}')
        cities.append(c)
        zips.append(z)
    df['location_city'] = cities
    df['location_zip'] = zips
    return df


HOURS_COLS = [f'Hours_of_Operation_{d}__c'
              for d in ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')]
HOURS_TEXT = [f'hours_{d.lower()}'
              for d in ('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun')]

_TIME_RE = re.compile(r'(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?', re.IGNORECASE)


def _day_minutes(text):
    """Total open minutes for one day. Handles split shifts ('6:30 AM - 9:20 AM &
    3:50 PM - 6:30 PM') by pairing consecutive timestamps. Returns NaN when the
    free-text hours can't be parsed (~7% of days: '7-8:30 AM', '24', 'to' forms)."""
    if not isinstance(text, str) or not text.strip():
        return np.nan
    if text.strip() == '24':
        return 24 * 60
    stamps = []
    for m in _TIME_RE.finditer(text):
        h = int(m.group(1)) % 12
        mins = int(m.group(2) or 0)
        if m.group(3).lower() == 'p':
            h += 12
        stamps.append(h * 60 + mins)
    if len(stamps) < 2:
        return np.nan
    total = 0
    for i in range(0, len(stamps) - 1, 2):
        span = stamps[i + 1] - stamps[i]
        if span > 0:
            total += span
    return total or np.nan


def parse_hours(df, log=None):
    """Hours_of_Operation_* -> hours_days_open (Int64) and hours_weekly_hours
    (float), plus tidy hours_<day> text columns for the raw set."""
    open_days, weekly = [], []
    for _, row in df.iterrows():
        days = 0
        mins = 0.0
        any_parsed = False
        for c in HOURS_COLS:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                days += 1
                m = _day_minutes(v)
                if not pd.isna(m):
                    mins += m
                    any_parsed = True
        open_days.append(days)
        weekly.append(round(mins / 60.0, 2) if any_parsed else np.nan)
    df['hours_days_open'] = pd.array(open_days, dtype='Int64')
    df['hours_weekly_hours'] = weekly
    for src, dst in zip(HOURS_COLS, HOURS_TEXT):
        if src in df.columns:
            df[dst] = df[src]
    return df


def parse_license_dates(df, column='initial_license_date', log=None):
    """'10/12/2023' -> initial_license_year (Int64)."""
    years = []
    for x in df.get(column, pd.Series(dtype=object)):
        y = np.nan
        if isinstance(x, str):
            m = re.search(r'/(\d{4})\s*$', x.strip())
            if m:
                y = int(m.group(1))
        years.append(y)
    df['initial_license_year'] = pd.array(years, dtype='Int64')
    return df


def presence_flags(df):
    """Free-text fields that are useful only as 'was anything reported?'."""
    for src, dst in (('certifications', 'has_certifications'),
                     ('tribal_info', 'has_tribal_info')):
        if src in df.columns:
            df[dst] = df[src].notna() & df[src].astype(str).str.strip().ne('')
        else:
            df[dst] = False
    return df


def to_bool(df, columns=None):
    """Coerce 'True'/'False' and 'Yes'/'No' into nullable booleans. Nullable,
    because a NaN would otherwise demote the column to object dtype."""
    for c in (columns or BOOLEAN_COLS):
        if c not in df.columns:
            continue
        df[c] = df[c].map(
            lambda v: True if str(v).strip().lower() in _TRUE
            else (False if str(v).strip().lower() in _FALSE else pd.NA)
        ).astype('boolean')
    return df


def _int(v):
    try:
        return int(str(v).strip())
    except Exception:
        return 0


def json_counts(df):
    """JSON-list columns -> record counts plus the severity features that make
    the complaint / inspection / licensing history usable as numbers."""
    comp = df['complaints_json'].apply(safe_json_list)
    insp = df['inspections_json'].apply(safe_json_list)
    lic = df['license_history_json'].apply(safe_json_list)
    con = df['contacts_json'].apply(safe_json_list)

    df['num_complaints'] = comp.apply(len)
    df['num_valid_complaint_issues'] = comp.apply(
        lambda rs: sum(_int(r.get('# Valid Issues')) for r in rs if isinstance(r, dict)))
    df['num_self_reported_complaints'] = comp.apply(
        lambda rs: sum(1 for r in rs if isinstance(r, dict)
                       and str(r.get('Self Reported', '')).strip().lower() == 'yes'))
    # `or ''`, not a .get() default: a blank field is stored as JSON null, so
    # the key exists and str(None) would be the truthy 'None'.
    df['num_serious_injuries'] = comp.apply(
        lambda rs: sum(1 for r in rs if isinstance(r, dict)
                       and str(r.get('Serious Injury Field') or '').strip()))
    df['has_serious_injury'] = df['num_serious_injuries'] > 0

    df['num_inspections'] = insp.apply(len)
    df['num_virtual_inspections'] = insp.apply(
        lambda rs: sum(1 for r in rs if isinstance(r, dict)
                       and str(r.get('Inspection Type', '')).strip().lower() == 'virtual'))

    df['num_licenses'] = lic.apply(len)
    df['num_open_licenses'] = lic.apply(
        lambda rs: sum(1 for r in rs if isinstance(r, dict)
                       and str(r.get('License Status', '')).strip().lower() == 'open'))
    df['num_closed_licenses'] = lic.apply(
        lambda rs: sum(1 for r in rs if isinstance(r, dict)
                       and str(r.get('License Status', '')).strip().lower() == 'closed'))

    df['num_contacts'] = con.apply(len)
    return df


# <section>_count is '0' for an empty section and blank when the detail page
# never loaded, which json_counts() alone cannot tell apart (a blank blob -> 0).
UNKNOWN_COUNT_SOURCES = {
    'complaints_count':      ('num_complaints', 'num_valid_complaint_issues',
                              'num_self_reported_complaints',
                              'num_serious_injuries', 'has_serious_injury'),
    'inspections_count':     ('num_inspections', 'num_virtual_inspections'),
    'license_history_count': ('num_licenses', 'num_open_licenses',
                              'num_closed_licenses'),
    'contacts_count':        ('num_contacts',),
}


def mask_unknown_counts(df, sources=None, log=None):
    """Blank *_count == the detail page never loaded, so the counts are
    unknown, not zero. Run directly after json_counts().

    The Int64 / boolean casts matter: masking a plain int64 column upcasts it
    to float64 and every surviving count would print as `3.0`.
    """
    for src, targets in (sources or UNKNOWN_COUNT_SOURCES).items():
        if src not in df.columns:
            continue
        unknown = (df[src].astype('string').str.strip()
                   .replace('', pd.NA).isna())
        if not unknown.any():
            continue
        for col in targets:
            if col not in df.columns:
                continue
            if col.startswith('has_'):
                df[col] = df[col].astype('boolean').mask(unknown)
            else:
                df[col] = (pd.to_numeric(df[col], errors='coerce')
                           .mask(unknown).astype('Int64'))
        if log:
            log(f'  {src} blank -> {int(unknown.sum())} row(s) unknown: '
                f'{", ".join(targets)}')
    return df


# Whole numbers that pandas upcasts to float64 once the column meets a NaN.
# hours_weekly_hours is not here: it carries real half-hour fractions.
INT_OUTPUT_COLS = ('licensed_capacity', 'license_prior_capacity',
                   'total_available_slots', 'socrata_licensecapacity')


def cast_int_columns(df, cols=INT_OUTPUT_COLS, log=None):
    """Whole-number columns print as 132, not 132.0, even when they hold NaN.
    Call after finalize() (post-FEATURE_RENAME names).
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64')
            if log:
                log(f'  int cast: {col} ({int(df[col].notna().sum())} non-null)')
    return df


def finalize(df, columns_file, which, key, target, dynamic_prefixes,
             na_as_level=False, valid_target_values=None):
    """Shared tail: order by the stable scaffold + any discovered columns,
    dedup on the grain, filter the target, then drop empty/constant columns.

    na_as_level: when True (raw set), NaN counts as its own value, which keeps
    sparse decomposed text columns. When False (full set), all-NaN columns and
    columns with one non-null value are dropped.

    valid_target_values: None (complete set) keeps every row; otherwise keep
    only rows whose numeric target is one of these values.
    """
    if 'Id' in df.columns and CRAWLER_GRAIN_COPY in df.columns:
        df = df.drop(columns=[CRAWLER_GRAIN_COPY])

    df = df.rename(columns={**ID_RENAME, **TARGET_RENAME, **FEATURE_RENAME})
    df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

    # Dropped before the scaffold reindex so a scaffold entry cannot revive them.
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

    # After the dedup and target filter: k is a property of the rows that ship.
    df = apply_privacy_remediation(df, which)

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
STATE = "wa"

# Small-range integer counts capped so a lone provider at the top of the range
# joins the bucket below; lossless for a tree splitting on ">= cap".
TOPCODE_COLS = {'num_contacts': 5}

# Families whose members are LEVELS of one attribute, so a rare member can be
# merged into "<prefix>other". Prefixes whose members are distinct attributes
# (has_*, num_*, violations_*, monitoring_*, ...) are deliberately excluded:
# OR-ing them together would invent a meaningless feature.
LEVEL_FAMILY_PREFIXES = ('agegroup_', 'certtype_', 'factype_', 'funded_', 'langinstr_', 'language_', 'licstatus_', 'slotavail_', 'spec_')

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
