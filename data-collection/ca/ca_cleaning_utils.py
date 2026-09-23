"""
ca_cleaning_utils.py — Shared utilities for the California cleaning scripts.

Produces two row-aligned views from one export:
  - full : strictly numeric/boolean (+ provider_id) for classical/tabular ML
  - raw  : text preserved and maximally decomposed for LLM methods

Schema is discovered at runtime; the only explicit vocabularies are keyterm
lists for fields with no usable delimiter. Parse failures are logged, never
raised. Extracted counts use nullable Int64 so a parse failure (NA) is
distinct from a true zero. Column drops are data-driven (all-NaN / constant).
Per-field handling lives in engineer_features().
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ID_COL = "provider_id"
TARGET_COL = "qr_rating"

# facility_number is the grain; basics_qcc_score is the Quality Counts
# California rating.
RENAME_MAP: dict[str, str] = {
    "facility_number": ID_COL,
    "basics_qcc_score": TARGET_COL,
}

# Anything else (incl. '-') → NaN → row dropped in the standard views.
VALID_TARGET_VALUES: set[int] = {1, 2, 3, 4, 5}

# Redundant IDs, URLs, contact details and the error flag.
NON_FEATURE_COLS: list[str] = [
    "provider_url",
    "profile_photo_url",
    "phone",
    "license_number",        # == facility_number
    "licensing_reports_url", # encodes facility_number
    "google_maps_url",
    "errors",                # error rows are already filtered out
]

# Keyterm vocabularies for fields with no reliable delimiter.
ACCREDITATION_KEYTERMS: list[str] = [
    "National Association for the Education of Young Children",  # NAEYC
    "NAEYC",
    "National Association for Family Child Care",
    "National Accreditation Commission",
    "Head Start",
]
QUALITY_IMPROVEMENT_KEYTERMS: list[str] = [
    "QRIS",   # Quality Rating and Improvement System
    "QSLA",   # Quality Start Los Angeles
    "QCC",    # Quality Counts California
]

# Engineered column families kept by finalize()'s reindex.
DISCOVERED_PREFIXES: dict[str, list[str]] = {
    "full": ["ptype_", "tag_", "lang_", "sched_", "transport_", "meal_",
             "subsidy_", "accred_", "qi_", "ages_"],
    "raw":  ["tag_", "lang_", "sched_", "transport_", "meal_",
             "subsidy_", "accred_", "qi_", "ages_"],
}

# Engineered columns dropped from each view in finalize():
#   - qi_qris / qi_qsla : label leakage (QRIS / Quality Start Los Angeles are
#     the rating programs that produce qr_rating).
#   - is_claimed / has_website / website_url / tags_count : metadata.
#   - zip / address / street_city / state : identifiers.
#   - openings_last_updated : a timestamp, not a quality signal.
#   - meal_car : parsing artifact (a stray "Car" token in the meals list).
TRAINING_EXCLUDE: dict[str, set[str]] = {
    "full": {
        "is_claimed", "has_website", "tags_count",
        "zip",
        "qi_qris", "qi_qsla",
        "meal_car",
    },
    "raw": {
        "is_claimed", "has_website", "website_url", "tags_count",
        "openings_last_updated",
        "zip", "address", "street_city", "state",
        "qi_qris", "qi_qsla",
        "meal_car",
    },
}

# Boilerplate that about_text repeats on nearly every row.
_ABOUT_BOILERPLATE = (
    "Contact us or your local R&R for more information about our child "
    "care program."
)

# Tokens that mean "missing / none" across this messy export.
_NULL_TOKENS = {"", "-", "nan", "none", "not provided", "n/a", "na", "null"}

# Columns where 'Not provided' is an answer, not a blank: the site prints
# "Meals: Not provided" when a provider serves no meals and omits the line when
# it says nothing. The raw view keeps the literal; the full view's meal_*
# one-hots still go through clean_str() and treat it as null.
KEEP_LITERAL_COLS = {"basics_meals"}
_KEEP_LITERAL_NULL_TOKENS = _NULL_TOKENS - {"not provided"}


def make_logger(log_path: "Path | str") -> Callable[[str, object, object], None]:
    """Return a logger(field, value, reason) that appends to log_path."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"# parse log — {log_path.name}\n")

    def _log(field: str, value: object, reason: object) -> None:
        with open(log_path, "a") as f:
            f.write(f"{field}\t{reason}\t{value!r}\n")

    return _log


def slug(s: object) -> str:
    """lowercase, non-alphanumerics → '_', collapse repeats, strip ends."""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def clean_str(v: object) -> "str | None":
    """Return a trimmed string, or None for null-ish/sentinel values."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    if s.lower() in _NULL_TOKENS:
        return None
    return s


def _raw_str(v: object, column: str) -> "str | None":
    """clean_str(), except KEEP_LITERAL_COLS keep a literal 'Not provided'."""
    if column not in KEEP_LITERAL_COLS:
        return clean_str(v)
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    if s.lower() in _KEEP_LITERAL_NULL_TOKENS:
        return None
    return s


def yn_to_bool(v: object) -> "int | float":
    """Yes/No (and friends) → 1/0; null → pd.NA."""
    s = clean_str(v)
    if s is None:
        return pd.NA
    return 1 if s.lower() in {"yes", "true", "y", "1"} else 0


def _split_sentence_items(s: str) -> list[str]:
    """Split on commas and ' and ' to recover individual items."""
    tmp = re.sub(r"\s+and\s+", ",", s, flags=re.IGNORECASE)
    return [p.strip() for p in tmp.split(",") if p.strip()]


_CURRENCY_RE = re.compile(r"^\s*\$\s*[\d,]+(?:\.\d+)?\s*$")


def dollar_strip_df(df: pd.DataFrame, log=None) -> pd.DataFrame:
    """Convert columns whose non-null cells are mostly '$1,234' strings to
    floats. A no-op when there are no currency columns."""
    df = df.copy()
    for col in df.columns:
        # pandas 3 gives text columns the `str` dtype, not object, so an
        # object-only guard skips every currency column there.
        if not (pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])):
            continue
        nonnull = df[col].dropna().astype(str)
        if nonnull.empty:
            continue
        has_dollar = nonnull.str.contains(r"\$")
        if has_dollar.mean() < 0.5:
            continue  # not a currency column
        def _strip(v):
            s = clean_str(v)
            if s is None:
                return np.nan
            if not _CURRENCY_RE.match(s):
                if log:
                    log(col, v, "currency-strip: non-currency value")
                return np.nan
            try:
                return float(s.replace("$", "").replace(",", "").strip())
            except ValueError as e:
                if log:
                    log(col, v, f"currency-strip failed: {e}")
                return np.nan
        df[col] = df[col].map(_strip)
    return df


def build_multivalue(
    df: pd.DataFrame,
    col: str,
    prefix: str,
    which: str,
    *,
    delimiter: "str | None" = None,
    sentence: bool = False,
    log=None,
) -> dict[str, list]:
    """Discover the vocabulary of a multi-value field and emit one column per
    item: full → 1/0 presence, raw → the item phrase where present, else NaN.

    sentence=True splits on commas AND ' and ' (language fields). For a
    single-value categorical, pass a delimiter that never occurs (e.g. '\\x00').
    """
    if col not in df.columns:
        return {}

    def _items(s: str) -> list[str]:
        if sentence:
            return _split_sentence_items(s)
        return [p.strip() for p in s.split(delimiter) if p.strip()]

    parsed: list[list[str]] = []
    vocab: set[str] = set()
    for v in df[col]:
        s = clean_str(v)
        if not s:
            parsed.append([])
            continue
        try:
            its = _items(s)
        except Exception as e:  # never crash on a bad row
            if log:
                log(col, v, f"multivalue split failed: {e}")
            its = []
        parsed.append(its)
        vocab.update(its)

    out: dict[str, list] = {}
    for item in sorted(vocab):
        cname = f"{prefix}{slug(item)}"
        if which == "full":
            out[cname] = [1 if item in its else 0 for its in parsed]
        else:
            out[cname] = [item if item in its else np.nan for its in parsed]
    return out


def build_keyterm(
    df: pd.DataFrame,
    col: str,
    vocab: list[str],
    prefix: str,
    which: str,
    log=None,
) -> dict[str, list]:
    """Match a keyterm vocabulary against the slugged text on token boundaries.
    full → boolean; raw → keyterm phrase where present."""
    if col not in df.columns:
        return {}
    slugged = df[col].map(lambda v: slug(clean_str(v) or ""))
    out: dict[str, list] = {}
    seen: set[str] = set()
    for term in vocab:
        tslug = slug(term)
        if not tslug or tslug in seen:
            continue
        seen.add(tslug)
        pat = re.compile(r"(^|_)" + re.escape(tslug) + r"($|_)")
        present = slugged.map(lambda s, p=pat: bool(p.search(s)))
        cname = f"{prefix}{tslug}"
        if which == "full":
            out[cname] = present.astype(int).tolist()
        else:
            out[cname] = [term if p else np.nan for p in present]
    return out


def _time_to_minutes(t: str) -> "int | None":
    m = re.match(r"(\d{1,2}):(\d{2})\s*([AP]M)", t.strip(), re.IGNORECASE)
    if not m:
        return None
    h = int(m.group(1)) % 12
    if m.group(3).upper() == "PM":
        h += 12
    return h * 60 + int(m.group(2))


def parse_business_hours(v: object, log=None) -> dict:
    """'Mo 07:00 AM - 05:30 PM | Tu ...' → days_open, earliest_open,
    latest_close (minutes since midnight), weekly_hours (float)."""
    res = {"days_open": pd.NA, "earliest_open": pd.NA,
           "latest_close": pd.NA, "weekly_hours": pd.NA}
    s = clean_str(v)
    if not s:
        return res
    opens, closes, total, days = [], [], 0.0, 0
    for entry in (e.strip() for e in s.split("|") if e.strip()):
        m = re.search(
            r"(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)",
            entry, re.IGNORECASE,
        )
        if not m:
            if log:
                log("business_hours", entry, "no time range matched")
            continue
        o, c = _time_to_minutes(m.group(1)), _time_to_minutes(m.group(2))
        if o is None or c is None:
            continue
        if c < o:           # overnight safety
            c += 24 * 60
        opens.append(o); closes.append(c); total += (c - o) / 60.0; days += 1
    if days:
        res = {"days_open": days, "earliest_open": min(opens),
               "latest_close": max(closes), "weekly_hours": round(total, 2)}
    return res


def parse_age_range(v: object, log=None) -> dict:
    """'2 years - 5 years' / '2 months - 11 years' → min/max age in MONTHS."""
    res = {"min_age_months": pd.NA, "max_age_months": pd.NA}
    s = clean_str(v)
    if not s:
        return res
    parts = re.split(r"\s*-\s*", s)
    if len(parts) != 2:
        if log:
            log("age_range", v, "did not split into two endpoints")
        return res

    def _to_months(p: str) -> "int | float":
        m = re.search(r"(\d+(?:\.\d+)?)\s*(month|year)", p, re.IGNORECASE)
        if not m:
            if log:
                log("age_range", p, "no number/unit")
            return pd.NA
        n = float(m.group(1))
        return int(round(n * (12 if m.group(2).lower() == "year" else 1)))

    res["min_age_months"] = _to_months(parts[0])
    res["max_age_months"] = _to_months(parts[1])
    return res


def parse_openings_status(v: object, log=None) -> dict:
    """'Preschool (2 to 5 years)No Openings' / '... Immediate Openings'
    → has_immediate_openings (1/0/NA) + openings_age_label (raw only)."""
    res = {"has_immediate_openings": pd.NA, "openings_age_label": np.nan}
    s = clean_str(v)
    if not s:
        return res
    if re.search(r"immediate openings", s, re.IGNORECASE):
        res["has_immediate_openings"] = 1
    elif re.search(r"no openings", s, re.IGNORECASE):
        res["has_immediate_openings"] = 0
    else:
        if log:
            log("openings_status", v, "no openings keyword")
    m = re.search(r"\(([^)]*)\)", s)
    if m:
        res["openings_age_label"] = m.group(1).strip()
    return res


def parse_address(v: object, log=None) -> dict:
    """'146 EAST 107TH STREET Los Angeles, CA 90003' → street_city, state, zip.
    Street and city have no reliable delimiter, so the pre-comma chunk is kept
    whole."""
    res = {"street_city": np.nan, "state": np.nan, "zip": pd.NA}
    s = clean_str(v)
    if not s:
        return res
    m = re.search(r"^(.*),\s*([A-Za-z]{2})\s+(\d{5})(?:-\d{4})?$", s)
    if not m:
        if log:
            log("address", v, "did not match '<street city>, ST ZIP'")
        return res
    res["street_city"] = m.group(1).strip()
    res["state"] = m.group(2).upper()
    res["zip"] = int(m.group(3))
    return res


def parse_about(v: object, log=None) -> dict:
    """Strip boilerplate and flag website mentions (raw view only)."""
    res = {"about_clean": np.nan, "mentions_website": pd.NA}
    s = clean_str(v)
    if not s:
        return res
    res["mentions_website"] = 1 if re.search(r"website", s, re.IGNORECASE) else 0
    cleaned = s.replace(_ABOUT_BOILERPLATE, "").strip()
    res["about_clean"] = cleaned or np.nan
    return res


def drop_error_rows(df: pd.DataFrame, log=None) -> pd.DataFrame:
    """Drop every row whose 'errors' column is non-null, before dedup and
    target filtering so all views stay row-aligned."""
    if "errors" not in df.columns:
        return df
    mask = df["errors"].map(lambda v: clean_str(v) is not None)
    n = int(mask.sum())
    if n and log:
        for v in df.loc[mask, "errors"].head(50):
            log("errors", v, "row dropped (errors non-null)")
    print(f"  drop_error_rows: removed {n} rows with non-null 'errors'")
    return df.loc[~mask].reset_index(drop=True)


def check_grain_uniqueness(df: pd.DataFrame, grain_col: str) -> None:
    """Confirm the chosen grain is actually unique; warn loudly if not."""
    if grain_col not in df.columns:
        print(f"  [grain] WARNING: '{grain_col}' not present — cannot verify.")
        return
    dup = int(df[grain_col].duplicated().sum())
    if dup:
        print(
            f"  [grain] WARNING: '{grain_col}' has {dup} duplicate value(s) on "
            f"this data — it may NOT be the true unique grain. finalize() keeps "
            f"the first row per id; revisit the grain choice if this is high."
        )
    else:
        print(f"  [grain] OK: '{grain_col}' is unique ({len(df)} rows).")


def engineer_features(df: pd.DataFrame, which: str, log=None) -> pd.DataFrame:
    """Apply every per-field builder/parser and return df with the engineered
    columns added; finalize() then trims to scaffold + discovered columns."""
    df = df.copy()
    new: dict[str, list] = {}

    if "licensed_status" in df:
        new["is_licensed"] = pd.array(
            [(1 if (clean_str(v) or "").lower().startswith("licensed")
              else (pd.NA if clean_str(v) is None else 0))
             for v in df["licensed_status"]], dtype="Int64")
    if "claimed_status" in df:
        new["is_claimed"] = pd.array(
            [(pd.NA if clean_str(v) is None
              else (1 if "unclaim" not in (clean_str(v) or "").lower() else 0))
             for v in df["claimed_status"]], dtype="Int64")
    if "website_url" in df:
        new["has_website"] = pd.array(
            [1 if clean_str(v) is not None else 0 for v in df["website_url"]],
            dtype="Int64")
    if "basics_special_needs_experience" in df:
        new["special_needs_experience"] = pd.array(
            [yn_to_bool(v) for v in df["basics_special_needs_experience"]],
            dtype="Int64")

    # provider_type: one-hot in full, phrase in raw
    if "provider_type" in df:
        if which == "full":
            new.update(build_multivalue(
                df, "provider_type", "ptype_", which,
                delimiter="\x00", log=log))  # whole cell = one category
        else:
            new["provider_type"] = [clean_str(v) for v in df["provider_type"]]

    for c in ("tags_count", "openings_capacity"):
        if c in df:
            new[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    new.update(build_multivalue(df, "tags", "tag_", which, delimiter="|", log=log))
    new.update(build_multivalue(df, "basics_language", "lang_", which,
                                sentence=True, log=log))
    new.update(build_multivalue(df, "basics_schedule", "sched_", which,
                                delimiter=",", log=log))
    new.update(build_multivalue(df, "basics_transportation", "transport_", which,
                                delimiter=",", log=log))
    new.update(build_multivalue(df, "basics_meals", "meal_", which,
                                delimiter=",", log=log))
    new.update(build_multivalue(df, "basics_subsidies_accepted", "subsidy_", which,
                                delimiter=",", log=log))
    # basics_ages is normally empty; finalize() drops it if all-NaN.
    new.update(build_multivalue(df, "basics_ages", "ages_", which,
                                delimiter=",", log=log))

    new.update(build_keyterm(df, "basics_accreditation",
                             ACCREDITATION_KEYTERMS, "accred_", which, log=log))
    new.update(build_keyterm(df, "basics_quality_improvement_efforts",
                             QUALITY_IMPROVEMENT_KEYTERMS, "qi_", which, log=log))
    if "basics_accreditation" in df:
        new["has_accreditation"] = pd.array(
            [1 if clean_str(v) is not None else 0 for v in df["basics_accreditation"]],
            dtype="Int64")

    if "business_hours" in df:
        bh = [parse_business_hours(v, log) for v in df["business_hours"]]
        new["days_open"] = pd.array([r["days_open"] for r in bh], dtype="Int64")
        new["earliest_open"] = pd.array([r["earliest_open"] for r in bh], dtype="Int64")
        new["latest_close"] = pd.array([r["latest_close"] for r in bh], dtype="Int64")
        new["weekly_hours"] = pd.array([r["weekly_hours"] for r in bh], dtype="Float64")
    if "age_range" in df:
        ar = [parse_age_range(v, log) for v in df["age_range"]]
        new["min_age_months"] = pd.array([r["min_age_months"] for r in ar], dtype="Int64")
        new["max_age_months"] = pd.array([r["max_age_months"] for r in ar], dtype="Int64")
    if "openings_status" in df:
        op = [parse_openings_status(v, log) for v in df["openings_status"]]
        new["has_immediate_openings"] = pd.array(
            [r["has_immediate_openings"] for r in op], dtype="Int64")
        if which == "raw":
            new["openings_age_label"] = [r["openings_age_label"] for r in op]
    if "address" in df:
        ad = [parse_address(v, log) for v in df["address"]]
        new["zip"] = pd.array([r["zip"] for r in ad], dtype="Int64")
        if which == "raw":
            new["street_city"] = [r["street_city"] for r in ad]
            new["state"] = [r["state"] for r in ad]

    if which == "raw" and "about_text" in df:
        ab = [parse_about(v, log) for v in df["about_text"]]
        new["about_clean"] = [r["about_clean"] for r in ab]
        new["mentions_website"] = pd.array(
            [r["mentions_website"] for r in ab], dtype="Int64")

    # Raw-only passthrough originals. `tags` is omitted on purpose: it is
    # already decomposed into tag_*, and apply_privacy_remediation() would
    # one-hot the original text again into a duplicate tags_* family.
    if which == "raw":
        for c in ("provider_name", "website_url", "business_hours",
                  "openings_last_updated", "openings_status", "basics_language",
                  "basics_schedule", "basics_meals", "basics_subsidies_accepted",
                  "age_range", "about_text", "address"):
            if c in df and c not in new:
                new[c] = [_raw_str(v, c) for v in df[c]]

    # engineered columns overwrite originals where names collide
    eng = pd.DataFrame(new, index=df.index)
    df = df.drop(columns=[c for c in eng.columns if c in df.columns])
    return pd.concat([df, eng], axis=1)


def finalize(
    df: pd.DataFrame,
    which: str,
    scaffold: dict[str, list[str]],
    log=None,
    keep_invalid_target: bool = False,
) -> pd.DataFrame:
    """Shared tail: rename → target-filter → drop non-features → reindex to
    scaffold + discovered → dedup → drop missing-target → privacy remediation
    → drop all-NaN / constant columns.

    keep_invalid_target=False (standard views): scores outside 1..5 and
    non-numeric values become NaN and those rows are dropped.
    keep_invalid_target=True (`complete` views): out-of-range numeric scores
    are kept as-is, only non-numeric values become NaN, and no row is dropped
    on the target.
    """
    df = df.rename(columns=RENAME_MAP).copy()

    if TARGET_COL in df.columns:
        t = pd.to_numeric(df[TARGET_COL], errors="coerce")
        if not keep_invalid_target:
            t = t.where(t.isin(VALID_TARGET_VALUES))
        df[TARGET_COL] = t.astype("Int64")

    df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

    # Identifiers are dropped BEFORE the scaffold reindex so they cannot
    # survive even if ca_columns.json lists them.
    df = drop_identifier_columns(df)

    stable = list(scaffold[which])
    prefixes = DISCOVERED_PREFIXES[which]
    discovered = sorted(
        c for c in df.columns
        if c not in stable and any(c.startswith(p) for p in prefixes)
    )
    df = df.reindex(columns=stable + discovered)

    excluded = [c for c in TRAINING_EXCLUDE.get(which, set()) if c in df.columns]
    if excluded and log:
        for c in excluded:
            log("finalize", c, "dropped: training-excluded")
    df = df.drop(columns=excluded)

    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=[ID_COL], keep="first")

    if TARGET_COL in df.columns and not keep_invalid_target:
        df = df[df[TARGET_COL].notna()].reset_index(drop=True)

    # Runs after the dedup and the target-row drop because k-anonymity is a
    # property of the rows that ship; earlier, a thin category could pass on
    # the strength of rows about to be dropped.
    df = apply_privacy_remediation(df, which)

    protect = {ID_COL, TARGET_COL}
    all_nan = [c for c in df.columns if c not in protect and df[c].isna().all()]
    if all_nan and log:
        for c in all_nan:
            log("finalize", c, "dropped: all-NaN")
    df = df.drop(columns=all_nan)

    # Constant columns. In raw, NaN counts as a level, so a phrase that is
    # constant where present but varies in presence is kept.
    const = []
    for c in df.columns:
        if c in protect:
            continue
        if which == "full":
            if df[c].nunique(dropna=True) <= 1:
                const.append(c)
        else:
            if df[c].astype(object).where(df[c].notna(), "__NA__").nunique() <= 1:
                const.append(c)
    if const and log:
        for c in const:
            log("finalize", c, "dropped: constant")
    df = df.drop(columns=const)

    return df.reset_index(drop=True)


def load_scaffold(path: "Path | str") -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


# >>> BEGIN PRIVACY BLOCK (shared verbatim by all 12 states) >>>
import json as _json
import random as _random
import re as _re
from pathlib import Path as _Path

import numpy as _np
import pandas as _pd

# --- per-state ----------------------------------------------------------------
STATE = "ca"

# Small-range integer counts capped so a lone provider at the top of the range
# joins the bucket below; lossless for a tree splitting on ">= cap".
TOPCODE_COLS = {}

# Families whose members are LEVELS of one attribute, so a rare member can be
# merged into "<prefix>other". Prefixes whose members are distinct attributes
# (has_*, num_*, violations_*, monitoring_*, ...) are deliberately excluded:
# OR-ing them together would invent a meaningless feature.
LEVEL_FAMILY_PREFIXES = ('lang_', 'meal_', 'ptype_', 'sched_', 'subsidy_', 'tag_', 'transport_')

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
