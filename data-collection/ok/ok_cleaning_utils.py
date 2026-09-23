"""Shared cleaning utilities for the OK provider pipelines.

The multi-value fields (program_tags, ages_accepted) have an open-ended
category space, so both pipelines discover their columns from the data.
parse_hours() summarizes the seven hours_<weekday> text columns;
summarize_monitoring() casts the monitoring summaries to numeric dtypes.
"""
import json
import re

import numpy as np
import pandas as pd

# The grain is already named "provider_id"; only the target is renamed.
ID_RENAME = {}

TARGET_RENAME = {'qr_rating_raw': 'qr_rating'}

# Metadata and contact PII, dropped before the schema is assembled. The two
# JSON blobs MUST be dropped here, not left to the reindex: they start with the
# raw pipeline's DYNAMIC_PREFIXES ("monitoring_" / "complaint_"), so finalize()
# would otherwise sweep them back in as extra columns.
NON_FEATURE_COLS = [
    'source_url', 'id_on_page', 'errors',
    'provider_name', 'licensing_specialist_name', 'licensing_specialist_phone',
    'contact_name', 'contact_title', 'contact_phone', 'contact_email',
    'address', 'subsidy_contract_number', 'contact_section_text',
    'monitoring_visits_json', 'complaint_findings_json',
]

WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday']


def slug(text):
    text = str(text).strip().lower().replace('&', ' and ')
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


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


def to_bool(series):
    """Coerce a mix of native True/False, the strings 'True'/'False', and NaN
    into nullable boolean dtype. read_csv may already have parsed the text to
    real bools, which a string-keyed .map() would silently turn into NaN."""
    def _coerce(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.strip().lower() == 'true':
                return True
            if v.strip().lower() == 'false':
                return False
        return pd.NA
    return series.map(_coerce).astype('boolean')


def _time_to_minutes(t):
    """'5:45AM' -> 345 (minutes since midnight). None on parse failure."""
    m = re.match(r'(\d{1,2}):(\d{2})\s*([AP]M)', t.strip(), re.IGNORECASE)
    if not m:
        return None
    h = int(m.group(1)) % 12
    if m.group(3).upper() == 'PM':
        h += 12
    return h * 60 + int(m.group(2))


def parse_hours(df, log=None):
    """hours_<weekday> ('5:45AM - 7:00PM' or NaN, one column per weekday) ->
    days_open (Int64, count of weekdays with a parseable hours range),
    earliest_open / latest_close (Int64, minutes since midnight, min/max
    across the week), weekly_hours (Float64, sum of daily open durations).
    Overnight ranges ("7:30PM - 6:00AM") add 24h to the close time. "CLOSED"
    is skipped silently, like a blank cell."""
    days_open, earliest, latest, weekly = [], [], [], []
    for _, row in df.iterrows():
        opens, closes, total, n = [], [], 0.0, 0
        for day in WEEKDAYS:
            v = row.get(f'hours_{day}')
            if not isinstance(v, str) or not v.strip() or v.strip().upper() == 'CLOSED':
                continue
            # findall, not search: a cell can hold a split shift,
            # '6:00AM - 9:00AM, 3:30PM - 5:00PM'.
            ranges = re.findall(
                r'(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)',
                v, re.IGNORECASE)
            if not ranges:
                if log:
                    log(f'hours_{day}: no time range matched: {v!r}')
                continue
            matched = False
            for start, end in ranges:
                o, c = _time_to_minutes(start), _time_to_minutes(end)
                if o is None or c is None:
                    continue
                if c < o:
                    c += 24 * 60
                opens.append(o)
                closes.append(c)
                total += (c - o) / 60.0
                matched = True
            if matched:
                n += 1
        if n:
            days_open.append(n)
            earliest.append(min(opens))
            latest.append(max(closes))
            weekly.append(round(total, 2))
        else:
            days_open.append(pd.NA)
            earliest.append(pd.NA)
            latest.append(pd.NA)
            weekly.append(pd.NA)
    df['days_open'] = pd.array(days_open, dtype='Int64')
    df['earliest_open'] = pd.array(earliest, dtype='Int64')
    df['latest_close'] = pd.array(latest, dtype='Int64')
    df['weekly_hours'] = pd.array(weekly, dtype='Float64')
    return df


def summarize_monitoring(df):
    """n_visits_with_noncompliance / avg_compliance_pct arrive on the input as
    text; cast them to the nullable numeric dtypes the scaffolds expect."""
    if 'n_visits_with_noncompliance' in df.columns:
        df['n_visits_with_noncompliance'] = pd.to_numeric(
            df['n_visits_with_noncompliance'], errors='coerce').astype('Int64')
    if 'avg_compliance_pct' in df.columns:
        df['avg_compliance_pct'] = pd.to_numeric(
            df['avg_compliance_pct'], errors='coerce').astype('Float64')
    return df


def finalize(df, columns_file, which, key, target, dynamic_prefixes,
             na_as_level=False, valid_target_values=None):
    """Shared tail: order by the stable scaffold + any discovered columns,
    dedup on the grain, filter the target, then drop empty/constant columns.

    na_as_level: when True (raw set), a column counts as informative if its
    presence/absence varies, i.e. NaN is treated as its own value. This keeps
    decomposed text columns whose non-null value is constant but which are
    only populated for some rows. When False (full set), drop all-NaN
    columns and columns with one non-null value.

    valid_target_values: None (complete set) keeps every row; otherwise keep
    only rows whose target, coerced to numeric, is one of those values.
    """
    df = df.rename(columns={**ID_RENAME, **TARGET_RENAME})
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
STATE = "ok"

# Small-range integer counts capped so a lone provider at the top of the range
# joins the bucket below; lossless for a tree splitting on ">= cap".
TOPCODE_COLS = {}

# Families whose members are LEVELS of one attribute, so a rare member can be
# merged into "<prefix>other". Prefixes whose members are distinct attributes
# (has_*, num_*, violations_*, monitoring_*, ...) are deliberately excluded:
# OR-ing them together would invent a meaningless feature.
LEVEL_FAMILY_PREFIXES = ('age_', 'tag_')

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
