"""
nc_clean_utils.py — Shared utilities for cleaning the North Carolina (DCDEE)
provider records into two datasets:

  full : strictly numeric/boolean (+ provider_id) — for classical/tabular ML.
  raw  : text preserved and maximally decomposed — for LLM-based methods.

The four clean scripts import from here. NC-specific choices live in the
EDITABLE CONSTANTS block; everything else discovers its vocabulary from the
data at runtime (the only seeded vocabularies are the keyterm lists for fields
with no usable delimiter). Parse failures are logged, never raised, and
extracted counts use nullable Int64 so a parse failure is distinct from a zero.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# EDITABLE CONSTANTS
# =============================================================================

ID_COL = "facility_id"          # per-facility NC license #
TARGET_COL = "star_rating"

RENAME_MAP = {ID_COL: "provider_id", TARGET_COL: "qr_rating"}

# NC issues 1–5 star ratings. Anything else (religious-exempt "GS 110-106" rows
# whose star is blank, probationary licences) is coerced to missing.
VALID_TARGET_VALUES: set[int] = {1, 2, 3, 4, 5}

# Rows with any of these non-empty carry no data (e.g. not_found) and are
# dropped FIRST, before engineering or dedup.
ERROR_FLAG_COLS: list[str] = ["errors"]

# Metadata and whole-page text dumps, removed up front (both sets). The
# structured content the dumps restate already has dedicated columns.
NON_FEATURE_COLS: list[str] = [
    "facility_url",
    "num_documents",
    "documents_json",
    "star_section_text",
    "visits_section_text",
    "details_section_text",
]

# Applied late, on the engineered column names, keyed by set.
#   leakage : license_type ("Five Star Center License") and star_components
#             (the rubric points that sum to the star rating) restate the target.
#   privacy : names, contact details, county.
#   raw also drops violations_visit_id (internal visit ids, not useful).
#
# `amenity_accredited_by_a_national_organization` is a provider-claimed flag;
# national accreditation can feed NC's program-standards points, so it is a
# candidate leakage flag. It is kept (self-reported, unverified).
TRAINING_EXCLUDE: dict[str, list[str]] = {
    "full": [
        # leakage
        "license_type", "star_components",
        # privacy / identifying
        "facility_name", "operator_name", "address", "phone", "email", "county",
    ],
    "raw": [
        # leakage
        "license_type", "star_components",
        # privacy / identifying (names also carry LLM-memorization risk)
        "facility_name", "operator_name", "address", "phone", "email", "county",
        # not-useful
        "violations_visit_id",
    ],
}

# Keyterm vocabularies for fields with NO usable delimiter, matched
# token-boundary-aware against the slugged field.
AMENITY_KEYTERMS: list[str] = [
    "Will enroll children with special needs",
    "Provides care for head start",
    "Provides transportation",
    "Field Trips",
    "Home Pick Up",
    "Provides NC Pre-K",
    "Accredited by a national organization",
]

# license_restrictions items are free-form sentences (e.g. "Other - Meeting
# reduced ratios.") with no clean category vocabulary.
RESTRICTION_KEYTERMS: list[str] = [
    "reduced ratios",
    "enhanced requirements",
    "daytime care only",
    "no cooking",
    "serves no more than two",   # infant cap phrasings
]

# license_age_days = (REFERENCE_DATE - license_issue_date); the data collection
# date, fixed for reproducibility.
REFERENCE_DATE = pd.Timestamp("2026-06-13")

_SF_DISCLAIMER = re.compile(
    r"^The following information is furnished by the provider.*?"
    r"contacting the provider directly\.\s*",
    flags=re.IGNORECASE | re.DOTALL,
)

# Engineered-column prefixes that the finalize reindex treats as "discovered"
# (appended after the scaffold, sorted). Keyed by set.
DISCOVERED_PREFIXES: dict[str, tuple[str, ...]] = {
    "full": ("facility_type_", "restriction_", "amenity_", "ratio_",
             "visits_", "violations_"),
    "raw":  ("facility_type_", "restriction_", "amenity_", "ratio_",
             "visits_", "violations_"),
}


# =============================================================================
# Parse logger — collects warnings, writes to a file, never raises.
# =============================================================================
class ParseLog:
    """Accumulates non-fatal parse messages and flushes them to a log file."""

    def __init__(self, path: "Path | str"):
        self.path = Path(path)
        self.messages: list[str] = []

    def warn(self, msg: str) -> None:
        self.messages.append(msg)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# parse log ({len(self.messages)} messages)\n"
        self.path.write_text(header + "\n".join(self.messages) + "\n")
        print(f"  parse log → {self.path} ({len(self.messages)} messages)")


# =============================================================================
# Generic helpers
# =============================================================================
def slug(s: object) -> str:
    """lowercase, non-alphanumeric → underscore, collapse repeats, strip ends.

    'Parent Co-Op' and 'parent co_op' both → 'parent_co_op'.
    """
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _has_value(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    s = str(v).strip()
    return s != "" and s.lower() != "nan"


def parse_json_safe(s: object, log: ParseLog, context: str = "") -> list | dict | None:
    """json.loads that never raises. Returns None on failure (logged)."""
    if not _has_value(s):
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError) as e:
        log.warn(f"[json:{context}] failed to parse: {e} :: {str(s)[:80]!r}")
        return None


# =============================================================================
# Shared early steps (run identically for both sets, before engineering)
# =============================================================================
def drop_error_rows(df: pd.DataFrame, log: ParseLog) -> pd.DataFrame:
    """Drop rows flagged by any ERROR_FLAG_COLS (non-null / non-empty)."""
    present = [c for c in ERROR_FLAG_COLS if c in df.columns]
    if not present:
        return df
    flagged = pd.Series(False, index=df.index)
    for c in present:
        flagged |= df[c].apply(_has_value)
    n = int(flagged.sum())
    log.warn(f"[error-rows] dropping {n} rows flagged by {present}")
    print(f"  dropped {n} source-error rows (flagged by {present})")
    return df.loc[~flagged].reset_index(drop=True)


def check_grain_unique(df: pd.DataFrame, col: str, log: ParseLog) -> None:
    """Warn (don't crash) if the chosen grain has duplicate values. Dedup
    (keep-first) happens later in finalize."""
    if col not in df.columns:
        log.warn(f"[grain] column {col!r} missing — cannot verify uniqueness")
        print(f"  WARNING: grain column {col!r} not found")
        return
    dups = int(df[col].duplicated().sum())
    if dups:
        log.warn(f"[grain] {col!r} has {dups} duplicate value(s); keep-first later")
        print(f"  WARNING: grain {col!r} not unique ({dups} dupes) — keep-first applied")
    else:
        print(f"  grain {col!r} is unique ({len(df)} rows)")


def strip_dollars(df: pd.DataFrame) -> pd.DataFrame:
    """Currency-string → float, only for columns whose every non-null value is
    a $-amount. Kept for cross-state parity; NC has no $-fields."""
    money_re = re.compile(r"^\s*\$\s?[\d,]+(?:\.\d+)?\s*$")
    for c in df.columns:
        ser = df[c].dropna().astype(str)
        if len(ser) and ser.str.match(money_re).all():
            df[c] = (
                df[c].astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .replace({"nan": np.nan, "": np.nan})
                .astype(float)
            )
    return df


# =============================================================================
# Per-field builders — each returns a DataFrame of NEW columns (same index).
# `mode` is "full" (numeric/boolean) or "raw" (text-preserving).
# =============================================================================
def build_categorical_onehot(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Low-cardinality single-value categorical → one boolean column per
    discovered value (full). Vocabulary discovered at runtime. NaN rows get
    NaN across all indicator columns (unknown, not False).

    The indicator is a NULLABLE BOOLEAN, not an object column: finalize() casts
    `boolean` columns to Int64 for `full`, and an object column would slip past
    that cast and ship as the text 'True'/'False'.
    """
    vals = sorted({slug(v) for v in series.dropna().unique() if _has_value(v)})
    out = {}
    for v in vals:
        col = f"{prefix}_{v}"
        # None (not np.nan) is what pd.array reads as <NA> for a boolean array
        ind = [(slug(x) == v) if _has_value(x) else None for x in series]
        out[col] = pd.array(ind, dtype="boolean")
    return pd.DataFrame(out, index=series.index)


def parse_age_range(series: pd.Series, log: ParseLog) -> pd.DataFrame:
    """'0 through 5' / '2 through 12' → age_min, age_max (nullable Int64).
    Both sets get these scalars."""
    pat = re.compile(r"(\d+)\s*through\s*(\d+)", re.IGNORECASE)
    mins, maxs = [], []
    for v in series:
        if not _has_value(v):
            mins.append(pd.NA); maxs.append(pd.NA); continue
        m = pat.search(str(v))
        if not m:
            log.warn(f"[ages_served] unparsed: {str(v)[:60]!r}")
            mins.append(pd.NA); maxs.append(pd.NA); continue
        mins.append(int(m.group(1))); maxs.append(int(m.group(2)))
    return pd.DataFrame(
        {"age_min": pd.array(mins, dtype="Int64"),
         "age_max": pd.array(maxs, dtype="Int64")},
        index=series.index,
    )


def derive_license_age_days(series: pd.Series, log: ParseLog) -> pd.DataFrame:
    """license_issue_date → license_age_days (nullable Int64), for BOTH sets."""
    dt = pd.to_datetime(series, errors="coerce")
    bad = series.apply(_has_value) & dt.isna()
    for v in series[bad]:
        log.warn(f"[license_issue_date] unparseable date: {str(v)[:40]!r}")
    days = (REFERENCE_DATE - dt).dt.days
    return pd.DataFrame(
        {"license_age_days": pd.array(days.round().astype("Int64"), dtype="Int64")},
        index=series.index,
    )


def _keyterm_hits(text: str, vocab: list[str]) -> list[str]:
    """Return the slugged keyterms whose tokens appear (token-boundary aware)
    inside the slugged text. Prevents partial-word false positives."""
    s = slug(text)
    hits = []
    for term in vocab:
        kt = slug(term)
        if re.search(r"(?:^|_)" + re.escape(kt) + r"(?:_|$)", s):
            hits.append(kt)
    return hits


def build_keyterm(
    series: pd.Series, vocab: list[str], prefix: str, mode: str,
    present: "pd.Series | None" = None,
) -> pd.DataFrame:
    """Type-5 keyterm field → one column per vocab term.
      full : boolean presence (NaN where the source field is missing).
      raw  : the human-readable keyterm where present, else NaN.

    `present` overrides the "was this field published at all" test, for a
    section that can be published EMPTY (a published "none" must read False,
    not NA).
    """
    term_slugs = [slug(t) for t in vocab]
    readable = {slug(t): t for t in vocab}
    present = series.apply(_has_value) if present is None else present.reset_index(drop=True)
    rows = []
    for v in series:
        hits = set(_keyterm_hits(v, vocab)) if _has_value(v) else set()
        rows.append(hits)
    out = {}
    for ks in term_slugs:
        col = f"{prefix}_{ks}"
        if mode == "full":
            vals = [(ks in hits) if present.iloc[i] else np.nan
                    for i, hits in enumerate(rows)]
            out[col] = pd.array(vals, dtype="boolean")
        else:  # raw
            vals = [readable[ks] if ks in hits else np.nan for hits in rows]
            out[col] = pd.Series(vals, index=series.index, dtype="object")
    return pd.DataFrame(out, index=series.index)


def parse_staff_child_ratios(series: pd.Series, log: ParseLog) -> pd.DataFrame:
    """Extract the per-age-group child counts from the special_features
    'Staff/Child Ratio Policy' section. Age-group labels are discovered.

    Source reads '<group> <A> Adult(s)/<C> Children'. The adult side is almost
    always 0 (a source data error), so only the *children* count is kept:
    ratio_<group>_children (nullable Int64), in both sets.
    """
    pat = re.compile(
        r"(Infants|\d+\s*Year\s*Olds?|5\s*Year\s*Old\s*Pre-?schoolers|"
        r"\d+-\d+\s*Year\s*Olds)\s+(\d+)\s*Adult\(s\)\s*/\s*(\d+)\s*Children",
        re.IGNORECASE,
    )
    per_row: list[dict[str, int]] = []
    all_groups: set[str] = set()
    for v in series:
        d: dict[str, int] = {}
        if _has_value(v):
            for grp, _adults, kids in pat.findall(str(v)):
                col = f"ratio_{slug(grp)}_children"
                d[col] = int(kids)
                all_groups.add(col)
        per_row.append(d)
    cols = sorted(all_groups)
    data = {c: pd.array([r.get(c, pd.NA) for r in per_row], dtype="Int64")
            for c in cols}
    return pd.DataFrame(data, index=series.index)


def build_special_features(series: pd.Series, mode: str, log: ParseLog) -> pd.DataFrame:
    """special_features is mixed (type 5 amenities + type 7 ratios + boilerplate).
      Amenities  → keyterm columns (build_keyterm).
      Ratios     → ratio_<group>_children scalars (parse_staff_child_ratios).
      full : amenity booleans + ratio ints.
      raw  : amenity keyterm text + ratio ints + cleaned original text
             (`special_features`, with the boilerplate disclaimer stripped).
    """
    # Strip boilerplate, then isolate the amenities chunk for keyterm matching.
    cleaned = series.apply(
        lambda v: _SF_DISCLAIMER.sub("", str(v)).strip() if _has_value(v) else np.nan
    )
    # `\s*`, not `\s+`: some providers publish the Amenities header with an
    # EMPTY list and nothing after it, so the text ends at "Amenities". That is
    # a published "no amenities" and must read False, not NA.
    amen_chunk = cleaned.apply(
        lambda v: (re.search(r"Amenities\s*(.*?)(?:\s+Staff/Child Ratio|$)",
                             str(v)).group(1)
                   if _has_value(v) and re.search(r"Amenities\b", str(v)) else np.nan)
    )
    # The amenities section counts as published whenever its header is on the
    # page, even if nothing follows it.
    header_present = cleaned.apply(
        lambda v: bool(_has_value(v) and re.search(r"Amenities\b", str(v))))
    parts = [
        build_keyterm(amen_chunk, AMENITY_KEYTERMS, "amenity", mode,
                      present=header_present),
        parse_staff_child_ratios(series, log),
    ]
    if mode == "raw":
        parts.append(pd.DataFrame({"special_features": cleaned}, index=series.index))
    return pd.concat(parts, axis=1)


def build_json_list(
    series: pd.Series,
    prefix: str,
    mode: str,
    log: ParseLog,
    raw_keys: "list[str] | None" = None,
    categorical_value_keys: "list[str] | None" = None,
    fetched: "pd.Series | None" = None,
) -> pd.DataFrame:
    """Nested JSON list field (taxonomy type 4).
      full : <prefix>_count (list length, Int64) plus, for each key listed in
             `categorical_value_keys`, a per-value count
             <prefix>_<key>_<value>_count.
      raw  : for each key in `raw_keys` (default: all discovered), a single text
             column <prefix>_<key> joining that key's values with ' | '.

    `fetched` decides what an EMPTY source cell means. By default it is "never
    collected" and the count is NA; where `fetched` is True it counts as a real
    0 (violations_json is empty, not missing, when a visits page listed no
    violations).
    """
    categorical_value_keys = categorical_value_keys or []
    parsed = [parse_json_safe(s, log, context=prefix) for s in series]
    present = series.apply(_has_value)

    # discover the union of keys
    all_keys: list[str] = []
    for p in parsed:
        if isinstance(p, list):
            for e in p:
                if isinstance(e, dict):
                    for k in e:
                        if k not in all_keys:
                            all_keys.append(k)
    keys = raw_keys if raw_keys is not None else all_keys

    out: dict[str, object] = {}
    if mode == "full":
        collected = present if fetched is None else (present | fetched.reset_index(drop=True))
        counts = [len(p) if isinstance(p, list) else (pd.NA if not collected.iloc[i] else 0)
                  for i, p in enumerate(parsed)]
        out[f"{prefix}_count"] = pd.array(counts, dtype="Int64")
        for key in categorical_value_keys:
            # discover this key's value vocabulary
            vocab = sorted({slug(e.get(key)) for p in parsed if isinstance(p, list)
                            for e in p if isinstance(e, dict) and _has_value(e.get(key))})
            for val in vocab:
                col = f"{prefix}_{slug(key)}_{val}_count"
                vals = []
                for i, p in enumerate(parsed):
                    if not isinstance(p, list):
                        vals.append(pd.NA if not present.iloc[i] else 0)
                    else:
                        vals.append(sum(1 for e in p if isinstance(e, dict)
                                        and slug(e.get(key)) == val))
                out[col] = pd.array(vals, dtype="Int64")
    else:  # raw
        for key in keys:
            col = f"{prefix}_{slug(key)}"
            joined = []
            for p in parsed:
                if isinstance(p, list):
                    items = [str(e.get(key)) for e in p
                             if isinstance(e, dict) and _has_value(e.get(key))]
                    joined.append(" | ".join(items) if items else np.nan)
                else:
                    joined.append(np.nan)
            out[col] = pd.Series(joined, index=series.index, dtype="object")
    return pd.DataFrame(out, index=series.index)


# =============================================================================
# Shared finalize tail (identical sequence for both sets)
# =============================================================================
def _is_constant(series: pd.Series, treat_nan_as_level: bool) -> bool:
    """True if the column carries no information.
      full (strict)         : <= 1 non-null unique value.
      raw  (nan-as-level)   : <= 1 unique value counting NaN as its own level
                              (so a phrase that's constant where present but
                              only populated for some rows is KEPT, because
                              presence varies)."""
    if treat_nan_as_level:
        return series.nunique(dropna=False) <= 1
    return series.nunique(dropna=True) <= 1


def finalize(
    df: pd.DataFrame,
    which: str,
    scaffold: dict[str, list[str]],
    log: ParseLog,
    *,
    keep_invalid_target: bool = False,
) -> pd.DataFrame:
    """Shared tail; the order of the steps matters.

    keep_invalid_target:
        False (standard sets): rows whose qr_rating is not in
            VALID_TARGET_VALUES are dropped.
        True (complete sets): every row is kept; the numeric rating survives
            where it parses and placeholders such as 'GS 110-106' become <NA>.
    """
    treat_nan_as_level = (which == "raw")

    df = df.rename(columns=RENAME_MAP)

    coerced = pd.to_numeric(df["qr_rating"], errors="coerce")
    valid = coerced.isin(list(VALID_TARGET_VALUES))
    n_invalid = int((df["qr_rating"].apply(_has_value) & ~valid).sum())
    if n_invalid:
        disp = "kept (complete)" if keep_invalid_target else "→ missing"
        log.warn(f"[target] {n_invalid} rows with out-of-range/garbage rating {disp}")
    df["qr_rating"] = coerced if keep_invalid_target else coerced.where(valid, other=np.nan)

    df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns],
                 errors="ignore")

    # Before the scaffold reindex, so identifiers cannot survive even if the
    # scaffold lists them.
    df = drop_identifier_columns(df)

    scaffold_cols = [c for c in scaffold[which] if c in df.columns]
    prefixes = DISCOVERED_PREFIXES[which]
    discovered = sorted(
        c for c in df.columns
        if c not in scaffold_cols and any(c.startswith(p) for p in prefixes)
    )
    keep = scaffold_cols + discovered
    seen, ordered = set(), []
    for c in keep:
        if c not in seen:
            seen.add(c); ordered.append(c)
    df = df.reindex(columns=ordered)

    df = df.drop(columns=[c for c in TRAINING_EXCLUDE[which] if c in df.columns],
                 errors="ignore")

    before = len(df)
    df = df.drop_duplicates(subset=["provider_id"], keep="first").reset_index(drop=True)
    if len(df) != before:
        log.warn(f"[dedup] removed {before - len(df)} duplicate provider_id row(s)")

    if keep_invalid_target:
        n_keep = int(df["qr_rating"].isna().sum())
        if n_keep:
            log.warn(f"[target] keeping {n_keep} rows with missing/invalid qr_rating (complete)")
    else:
        n_missing = int(df["qr_rating"].isna().sum())
        if n_missing:
            log.warn(f"[target] dropping {n_missing} rows with missing qr_rating")
        df = df.dropna(subset=["qr_rating"]).reset_index(drop=True)
    df["qr_rating"] = df["qr_rating"].astype("Int64")

    # Geography and the k-anonymity sweep run after the target filter, because
    # k is a property of the rows that actually ship.
    df = apply_privacy_remediation(df, which)

    protected = {"provider_id", "qr_rating"}
    drop_const = []
    for c in df.columns:
        if c in protected:
            continue
        if df[c].isna().all() or _is_constant(df[c], treat_nan_as_level):
            drop_const.append(c)
    if drop_const:
        log.warn(f"[constant] dropping {len(drop_const)} all-NaN/constant cols: {drop_const}")
    df = df.drop(columns=drop_const)

    # full must be numeric/boolean *through a CSV round-trip*: nullable booleans
    # serialize to 'True'/'False' text and re-read as object, so cast them to
    # nullable Int64 (0/1/<NA>) which reloads cleanly as a numeric column.
    if which == "full":
        for c in df.columns:
            if c == "provider_id":
                continue
            if df[c].dtype == "boolean" or pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].astype("Int64")

    return df


def write_output(df: pd.DataFrame, path: "Path | str") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {len(df)} rows × {df.shape[1]} cols → {path}")


# >>> BEGIN PRIVACY BLOCK (shared verbatim by all 12 states) >>>
import json as _json
import random as _random
import re as _re
from pathlib import Path as _Path

import numpy as _np
import pandas as _pd

# --- per-state ----------------------------------------------------------------
STATE = "nc"

# Small-range integer counts capped so a lone provider at the top of the range
# joins the bucket below; lossless for a tree splitting on ">= cap".
TOPCODE_COLS = {}

# Families whose members are LEVELS of one attribute, so a rare member can be
# merged into "<prefix>other". Prefixes whose members are distinct attributes
# (has_*, num_*, violations_*, monitoring_*, ...) are deliberately excluded:
# OR-ing them together would invent a meaningless feature.
LEVEL_FAMILY_PREFIXES = ('amenity_', 'restriction_')

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
