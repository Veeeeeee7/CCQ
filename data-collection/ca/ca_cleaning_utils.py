"""
cleaning_utils.py — Shared utilities for the California childcare-provider
preprocessing pipeline.

Produces two row-aligned views from one raw export:
  - full : strictly numeric/boolean (+ provider_id) for classical/tabular ML
  - raw  : text preserved and maximally decomposed for LLM methods

Design (per the methodology prompt):
  - Schema is DISCOVERED at runtime; multi-value vocabularies are never
    hardcoded from a sample. The only explicit vocabularies are keyterm
    lists for fields with no usable delimiter (clearly marked, human-review).
  - Parse failures are logged to a file and never crash (NaN/0 returned).
  - Extracted counts use nullable Int64 so a parse failure (NA) is distinct
    from a true zero.
  - All column drops are data-driven (all-NaN / constant), so a column that
    is constant in a sample but varies on the full data is handled
    automatically — nothing is hardcode-dropped from the sample.

Field handling for THIS schema is centralized in engineer_features(); the two
cleaning scripts stay thin (load → dollar-strip → drop error rows → engineer →
finalize → save).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# =============================================================================
# Editable constants  (the only place to edit if the schema changes)
# =============================================================================

ID_COL = "provider_id"
TARGET_COL = "qr_rating"

# Grain → provider_id, target → qr_rating. Confirmed in conversation:
#   facility_number is the unique grain (uniqueness verified at runtime);
#   basics_qcc_score is the Quality Counts California ordinal target.
RENAME_MAP: dict[str, str] = {
    "facility_number": ID_COL,
    "basics_qcc_score": TARGET_COL,
}

# Valid target levels (QCC tier). Anything else (incl. '-') → NaN → row dropped.
VALID_TARGET_VALUES: set[int] = {1, 2, 3, 4, 5}

# Non-feature columns: redundant IDs, scrape artifacts, URLs, PII, metadata.
# Removed from BOTH outputs entirely (belt-and-suspenders; reindex also drops
# anything not in the scaffold or a discovered prefix).
NON_FEATURE_COLS: list[str] = [
    "provider_url",          # scrape URL (session_id / page_number artifacts)
    "profile_photo_url",     # image URL
    "phone",                 # contact PII
    "license_number",        # == facility_number (redundant ID)
    "licensing_reports_url", # encodes facility_number
    "google_maps_url",       # URL
    "errors",                # scrape-error flag (rows already filtered out)
]

# --- Keyterm vocabularies (TYPE 5: no reliable delimiter) --------------------
# Seeded from observed sample values. *** HUMAN REVIEW REQUIRED *** — extend
# with the full set of accreditors / QI programs once seen on the full data.
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

# Engineered column prefixes per output (used by finalize's reindex to know
# which discovered columns to keep). Edit alongside engineer_features().
DISCOVERED_PREFIXES: dict[str, list[str]] = {
    "full": ["ptype_", "tag_", "lang_", "sched_", "transport_", "meal_",
             "subsidy_", "accred_", "qi_", "ages_"],
    "raw":  ["tag_", "lang_", "sched_", "transport_", "meal_",
             "subsidy_", "accred_", "qi_", "ages_"],
}

# Columns to EXCLUDE FROM TRAINING — single editable place for exclusions that
# are not features: metadata, privacy/identifiers, parsing artifacts, and
# (most important) anything that leaks the qr_rating label. Applied in
# finalize() against the engineered column names, so it catches both stable
# scaffold columns and discovered (prefix) columns.
#
# Reasons:
#   - qi_qris / qi_qsla : LEAKAGE. QRIS = Quality Rating & Improvement System,
#     QSLA = Quality Start Los Angeles — the rating program that produces
#     qr_rating. Among rated-only rows these restate the label's apparatus.
#   - is_claimed / has_website / website_url / tags_count : metadata, not signal.
#   - zip / address / street_city / state : privacy / identifiers (state is also
#     constant CA). Same rationale that removes `zip` from full.
#   - openings_last_updated : scrape timestamp, not a quality signal.
#   - meal_car : parsing artifact (a stray "Car" token in the meals vocabulary).
TRAINING_EXCLUDE: dict[str, set[str]] = {
    "full": {
        "is_claimed", "has_website", "tags_count",   # metadata
        "zip",                                        # privacy / identifier
        "qi_qris", "qi_qsla",                         # target leakage
        "meal_car",                                   # parsing artifact
    },
    "raw": {
        "is_claimed", "has_website", "website_url", "tags_count",  # metadata
        "openings_last_updated",                                   # scrape ts
        "zip", "address", "street_city", "state",                 # privacy / id
        "qi_qris", "qi_qsla",                                      # leakage
        "meal_car",                                               # artifact
    },
}

# Boilerplate that about_text re-states for nearly every row (TYPE 8 noise).
_ABOUT_BOILERPLATE = (
    "Contact us or your local R&R for more information about our child "
    "care program."
)

# Tokens that mean "missing / none" across this messy export.
_NULL_TOKENS = {"", "-", "nan", "none", "not provided", "n/a", "na", "null"}


# =============================================================================
# Parse logging  (never crash; log and move on)
# =============================================================================
def make_logger(log_path: "Path | str") -> Callable[[str, object, object], None]:
    """Return a logger(field, value, reason) that appends to log_path."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # truncate at start of run
    log_path.write_text(f"# parse log — {log_path.name}\n")

    def _log(field: str, value: object, reason: object) -> None:
        with open(log_path, "a") as f:
            f.write(f"{field}\t{reason}\t{value!r}\n")

    return _log


# =============================================================================
# Small helpers
# =============================================================================
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


def yn_to_bool(v: object) -> "int | float":
    """Yes/No (and friends) → 1/0; null → pd.NA."""
    s = clean_str(v)
    if s is None:
        return pd.NA
    return 1 if s.lower() in {"yes", "true", "y", "1"} else 0


def _split_sentence_items(s: str) -> list[str]:
    """TYPE 6: split on commas and ' and ' to recover individual items."""
    tmp = re.sub(r"\s+and\s+", ",", s, flags=re.IGNORECASE)
    return [p.strip() for p in tmp.split(",") if p.strip()]


# =============================================================================
# Dollar-strip pass (TYPE 2) — generic, harmless if no currency columns exist
# =============================================================================
_CURRENCY_RE = re.compile(r"^\s*\$\s*[\d,]+(?:\.\d+)?\s*$")


def dollar_strip_df(df: pd.DataFrame, log=None) -> pd.DataFrame:
    """Convert any column whose non-null cells are predominantly currency
    strings ($1,234) into floats. Conservative: a column is only converted
    when most of its values actually carry a '$', so comma-delimited
    multi-value text columns are never touched."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != object:
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


# =============================================================================
# Discovery builders
# =============================================================================
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
    """TYPE 3/6: discover the vocabulary of a multi-value field at runtime and
    emit one column per discovered item.

      full : 1/0 presence boolean
      raw  : the item phrase where present, else NaN

    delimiter   real delimiter to split on (e.g. '|' or ',').
    sentence    if True, split on commas AND ' and ' (TYPE 6 language fields).

    For a single-value categorical (TYPE 1 one-hot), pass a delimiter that does
    not occur in the data (e.g. '\\x00') so the whole cell is one item.
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
    """TYPE 5: match an editable keyterm vocabulary against the slugged text,
    token-boundary aware (so 'High Scope' matches inside a blob without false
    positives). full → boolean; raw → keyterm phrase where present."""
    if col not in df.columns:
        return {}
    slugged = df[col].map(lambda v: slug(clean_str(v) or ""))
    out: dict[str, list] = {}
    # de-dupe by slug so 'NAEYC' and its long form can share/ separate cleanly
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


# =============================================================================
# Prose scalar parsers (TYPE 7)
# =============================================================================
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
    Street vs city has no reliable delimiter, so we keep the pre-comma chunk
    intact (street_city) rather than fabricate a wrong split. zip is numeric."""
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
    """TYPE 8: strip boilerplate, flag website mentions.
    raw keeps about_clean + mentions_website; full keeps neither (text)."""
    res = {"about_clean": np.nan, "mentions_website": pd.NA}
    s = clean_str(v)
    if not s:
        return res
    res["mentions_website"] = 1 if re.search(r"website", s, re.IGNORECASE) else 0
    cleaned = s.replace(_ABOUT_BOILERPLATE, "").strip()
    res["about_clean"] = cleaned or np.nan
    return res


# =============================================================================
# Error-row filter (user requirement) + grain uniqueness check
# =============================================================================
def drop_error_rows(df: pd.DataFrame, log=None) -> pd.DataFrame:
    """Drop every row whose 'errors' column is non-null. Applied identically
    in both scripts BEFORE dedup/target-filtering so the two outputs stay
    row-aligned."""
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


# =============================================================================
# Per-field engineering for THIS schema
# =============================================================================
def engineer_features(df: pd.DataFrame, which: str, log=None) -> pd.DataFrame:
    """Apply every per-field builder/parser for this California schema and
    return df with the engineered columns added. finalize() then trims to the
    scaffold + discovered columns."""
    df = df.copy()
    new: dict[str, list] = {}

    # --- TYPE 1: simple booleans / categoricals -----------------------------
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

    # provider_type: full → one-hot booleans; raw → keep phrase column
    if "provider_type" in df:
        if which == "full":
            new.update(build_multivalue(
                df, "provider_type", "ptype_", which,
                delimiter="\x00", log=log))  # whole cell = one category
        else:
            new["provider_type"] = [clean_str(v) for v in df["provider_type"]]

    # passthrough numerics
    for c in ("tags_count", "openings_capacity"):
        if c in df:
            new[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # --- TYPE 3/6: delimited / sentence multi-value -------------------------
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
    # basics_ages: empty in sample; routed defensively (dropped if all-NaN).
    new.update(build_multivalue(df, "basics_ages", "ages_", which,
                                delimiter=",", log=log))

    # --- TYPE 5: keyterm vocabularies (no delimiter) ------------------------
    new.update(build_keyterm(df, "basics_accreditation",
                             ACCREDITATION_KEYTERMS, "accred_", which, log=log))
    new.update(build_keyterm(df, "basics_quality_improvement_efforts",
                             QUALITY_IMPROVEMENT_KEYTERMS, "qi_", which, log=log))
    if "basics_accreditation" in df:
        new["has_accreditation"] = pd.array(
            [1 if clean_str(v) is not None else 0 for v in df["basics_accreditation"]],
            dtype="Int64")

    # --- TYPE 7: prose scalar extraction ------------------------------------
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

    # --- TYPE 8: about_text narrative (raw only keeps decomposition) --------
    if which == "raw" and "about_text" in df:
        ab = [parse_about(v, log) for v in df["about_text"]]
        new["about_clean"] = [r["about_clean"] for r in ab]
        new["mentions_website"] = pd.array(
            [r["mentions_website"] for r in ab], dtype="Int64")

    # --- raw-only passthrough originals (preserve + decompose) --------------
    if which == "raw":
        for c in ("provider_name", "website_url", "business_hours", "tags",
                  "openings_last_updated", "openings_status", "basics_language",
                  "basics_schedule", "basics_meals", "basics_subsidies_accepted",
                  "age_range", "about_text", "address"):
            if c in df and c not in new:
                new[c] = [clean_str(v) for v in df[c]]

    # attach engineered columns (overwrite originals where names collide)
    eng = pd.DataFrame(new, index=df.index)
    df = df.drop(columns=[c for c in eng.columns if c in df.columns])
    return pd.concat([df, eng], axis=1)


# =============================================================================
# Shared finalize / tail (identical contract for both sets)
# =============================================================================
def finalize(
    df: pd.DataFrame,
    which: str,
    scaffold: dict[str, list[str]],
    log=None,
    keep_invalid_target: bool = False,
) -> pd.DataFrame:
    """Shared tail: rename → target-filter → drop NON_FEATURE → reindex to
    scaffold + discovered → dedup → drop missing-target → drop all-NaN /
    constant columns (with the raw NaN-as-level nuance).

    keep_invalid_target controls how the qr_rating target is treated:
      - False (default): numeric scores outside 1..5 (and '-' / non-numeric)
        become NaN, and rows with a missing/invalid target are DROPPED. This
        is the modelling set (`full` / `raw`).
      - True: out-of-range numeric scores are PRESERVED as-is (e.g. a stray 7
        stays 7), only genuinely non-numeric values ('-', blanks) become NaN,
        and NO rows are dropped on the basis of the target. This yields the
        `complete` set — a row superset of `full` that still carries the
        invalid scores for inspection.
    """
    df = df.rename(columns=RENAME_MAP).copy()

    # Target → numeric. By default keep only valid 1..5 levels (else NaN);
    # when keep_invalid_target, preserve any numeric value and only blank out
    # the genuinely non-numeric ('-' / empty) cells.
    if TARGET_COL in df.columns:
        t = pd.to_numeric(df[TARGET_COL], errors="coerce")
        if not keep_invalid_target:
            t = t.where(t.isin(VALID_TARGET_VALUES))
        df[TARGET_COL] = t.astype("Int64")

    # Drop explicit non-feature columns.
    df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns])

    # P1 / P2: direct identifiers, per-provider narratives and coordinates.
    # Dropped BEFORE the scaffold reindex so they cannot survive even if a
    # stale ca_columns.json still lists them.
    df = drop_identifier_columns(df)

    # Reindex: scaffold order + sorted discovered (engineered-prefix) columns.
    stable = list(scaffold[which])
    prefixes = DISCOVERED_PREFIXES[which]
    discovered = sorted(
        c for c in df.columns
        if c not in stable and any(c.startswith(p) for p in prefixes)
    )
    df = df.reindex(columns=stable + discovered)

    # Remove training-excluded columns (leakage / privacy / metadata / artifact).
    excluded = [c for c in TRAINING_EXCLUDE.get(which, set()) if c in df.columns]
    if excluded and log:
        for c in excluded:
            log("finalize", c, "dropped: training-excluded")
    df = df.drop(columns=excluded)

    # Dedup on provider_id (keep first).
    if ID_COL in df.columns:
        df = df.drop_duplicates(subset=[ID_COL], keep="first")

    # Drop rows with a missing/invalid target — unless we're explicitly
    # keeping the invalid scores (the `complete` set).
    if TARGET_COL in df.columns and not keep_invalid_target:
        df = df[df[TARGET_COL].notna()].reset_index(drop=True)

    # P2 / P3: geography (collapse -> relabel -> one-hot) and the k-anonymity
    # sweep. These run HERE, after the dedup and the target-row drop, because k
    # is a property of the rows that actually ship -- computing it earlier would
    # let a category that is thin in the release pass on the strength of rows
    # that were about to be dropped.
    df = apply_privacy_remediation(df, which)

    # Drop all-NaN columns (never the id/target).
    protect = {ID_COL, TARGET_COL}
    all_nan = [c for c in df.columns if c not in protect and df[c].isna().all()]
    if all_nan and log:
        for c in all_nan:
            log("finalize", c, "dropped: all-NaN")
    df = df.drop(columns=all_nan)

    # Drop constant columns.
    #   full: strict — drop if <= 1 distinct non-null value.
    #   raw : NaN is its own level — a phrase that is constant where present
    #         but varies in presence (2 levels) is KEPT.
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


# =============================================================================
# Scaffold I/O
# =============================================================================
def load_scaffold(path: "Path | str") -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


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
STATE = "ca"

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
LEVEL_FAMILY_PREFIXES = ('lang_', 'meal_', 'ptype_', 'sched_', 'subsidy_', 'tag_', 'transport_')

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
