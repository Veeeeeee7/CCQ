"""
co_cleaning_utils.py — Shared utilities for cleaning the Colorado Shines
child-care records into two row-aligned datasets:

  full : strictly numeric/boolean (+ provider_id) — for classical/tabular ML.
  raw  : text preserved and maximally decomposed — for LLM-based methods.

Both co_clean_raw.py and co_clean_full.py import from here so the editable
constants, discovery builders, and the shared finalize tail live in exactly
one place. Structure mirrors nc_clean_utils.py (the most-refined reference
pattern; see the project plan's cleaning design contract, section 6).

One deliberate deviation from the NC/WI pattern, flagged here and in the
project chat for review: NC's drop_error_rows() drops an entire row if ANY
error flag is set, because for NC every field comes from the same crawl, so a
failed crawl means no usable data at all. Colorado is two-source: qr_rating
comes from the open-data columns, not the per-provider detail columns,
so a *detail* failure (not_found / ambiguous_no_match /
exception) does not touch the target or any open-data column — those rows
are still perfectly good rated providers, just missing the bonus site-scraped
fields (which are already correctly NaN from the crawler's empty_*()
fallbacks). Blanket-dropping them would silently throw away real data for no
reason. The one case that DOES need special handling is `id_mismatch`: the
crawler found *a* detail page, but it may be for the wrong provider, so its
site-scraped fields are untrustworthy even though they're populated. See
null_out_enrichment_on_mismatch() below.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# EDITABLE CONSTANTS — review these for every schema change.
# =============================================================================

# --- Identity + target -------------------------------------------------------
ID_COL = "provider_id"        # already carries this name on the input
TARGET_COL = "quality_rating"  # native open-data column -> renamed to qr_rating

RENAME_MAP = {TARGET_COL: "qr_rating"}  # ID_COL needs no rename (see above)

# Colorado Shines' five real levels (see project chat: confirmed via the
# source workbook, CDEC's own program pages, and direct inspection of the
# live open dataset). "NA" (the source's own missing-value marker), blank,
# and out-of-range values are coerced to missing.
VALID_TARGET_VALUES: set[int] = {1, 2, 3, 4, 5}

# --- Non-feature columns (dropped from BOTH sets, before engineering) -------
# Crawler QA metadata, redundant identifiers, and identity/contact fields.
# "errors" is deliberately NOT here: null_out_enrichment_on_mismatch() reads
# it first; each clean script drops it explicitly afterward.
# "license_number_on_site" is NOT here: it's kept in the RAW scaffold on
# purpose (cross-validation of provider_id against the live site) and simply
# omitted from the FULL scaffold. See co_columns.json.
NON_FEATURE_COLS: list[str] = [
    "detail_url", "match_method", "n_candidates",
    "provider_name", "street_address", "city", "zip", "state",
    "phone", "website",
    "ecc", "ccrr", "school_district", "governing_body",
]

# --- TARGET-LEAKAGE COLUMNS (project-wide rule; NEVER exported) --------------
# Rule: no exported feature may be derived from the QRIS rating or from its
# administrative lifecycle. That covers (a) the rating itself under any other
# name or rendering, and (b) the paperwork the rating generates -- award /
# effective / expiration / renewal dates and anything computed from them
# (age-of-rating, time-to-expiry, and their missingness indicators). Only
# `qr_rating`, the prediction target, survives.
#
# Empirically measured on CO (3-fold CV single-feature probes, QWK vs
# qr_rating), which is why this is a hard rule rather than a judgement call:
#   rating_on_site             0.976  (near-verbatim copy of the target;
#                                      diagonal crosstab, ~6/3423 mismatches)
#   expiration_date            0.905
#   award_date                 0.841
#   days_until_rating_expires  0.749
#   rating_age_days            0.672
# The date fields are not merely correlated: Colorado awards no rating cycle
# to Level 1 providers, so the *presence* of an award/expiration date is
# itself a class label. That is why these are removed outright rather than
# imputed, bucketed, or otherwise transformed -- the information is the leak,
# not the format.
#
# NOTE the deliberate exclusions: license_issue_date / license_age_days stay.
# Those belong to the *licensing* lifecycle, which exists independently of
# Colorado Shines and applies uniformly to rated and unrated providers alike.
LEAKAGE_COLS: list[str] = [
    "rating_on_site",
    "award_date",
    "expiration_date",
    "rating_age_days",
    "days_until_rating_expires",
]

# Any engineered column whose name begins with one of these is a derived form
# of a LEAKAGE_COLS field (one-hot, ordinal, missingness indicator, binned
# variant, ...) and is dropped by the same sweep. Defensive: nothing currently
# generates these, but it stops a future builder from smuggling the leak back
# in under a new name.
LEAKAGE_PREFIXES: tuple[str, ...] = (
    "rating_on_site", "award_date", "expiration_date",
    "rating_age", "days_until_rating", "rating_expir", "rating_renew",
    "rating_effective",
)

# --- Reference date for "days since / until" features -----------------------
# Editable; set to the scrape date for reproducibility.
REFERENCE_DATE = pd.Timestamp("2026-07-06")

# --- Language vocabulary (no usable delimiter -> seeded + human-reviewed) ---
# languages_spoken is space-joined ("English Spanish"), and "Sign Language" is
# itself two words, so a naive space-split would wrongly produce "Sign" and
# "Language" as separate hits. Vocabulary is the site's own filter dropdown
# (coloradoshines.com/search "Language" picklist -- confirmed via direct
# inspection, not guessed), matched token-boundary-aware like NC's keyterm
# fields. Extend if a value not in this list ever gets logged.
LANGUAGE_KEYTERMS: list[str] = [
    "English", "Spanish", "French", "German", "Mandarin",
    "Sign Language", "Other",
]

# --- Licensing-history boilerplate -------------------------------------------
# Confirmed verbatim (including the source's own "facilities" typo) on a real
# complaints_text field when there's nothing to report; assumed -- not yet
# empirically confirmed on a provider that actually HAS a documented incident
# -- to indicate "no substantive content" the same way across all 5 sections.
# Flag for re-check once real incident data is seen.
_NO_HISTORY_RE = re.compile(
    r"information is not currently available on the system.*?"
    r"public file review[^.]*\.?",
    flags=re.IGNORECASE | re.DOTALL,
)
LICENSING_HISTORY_COLS: list[str] = [
    "inspection_report_text", "complaints_text", "stage_ii_text",
    "injury_investigations_text", "adverse_actions_text",
]
_HISTORY_FLAG_NAMES = {
    "inspection_report_text": "has_documented_inspection_history",
    "complaints_text": "has_documented_complaint",
    "stage_ii_text": "has_documented_stage_ii",
    "injury_investigations_text": "has_documented_injury_investigation",
    "adverse_actions_text": "has_documented_adverse_action",
}

# Columns nulled out (not the whole row) when errors contains 'id_mismatch' --
# every field co_crawler.py could only have gotten from the (possibly wrong)
# detail page. The open-data columns are never touched.
MISMATCH_NULL_COLS: list[str] = [
    "license_number_on_site", "rating_on_site", "description",
    "hours_of_operation", "license_type", "license_issue_date",
    "licensed_to_serve", "capacity_on_site", "phone", "website",
    "languages_spoken", "special_needs", "head_start",
    "accepts_cccap_on_site", "accepting_new_children",
    "openings_infant", "openings_toddler", "openings_preschool",
    "openings_school_age",
] + LICENSING_HISTORY_COLS

# Engineered-column prefixes the finalize reindex treats as "discovered"
# (appended after the scaffold, sorted). Keyed by set. provider_service_type
# and county are kept as plain text columns in raw (see the scaffold), so they
# don't need a prefix there -- only their full-set one-hot forms are dynamic.
DISCOVERED_PREFIXES: dict[str, tuple[str, ...]] = {
    "full": ("type_", "county_", "licensetype_", "language_", "need_"),
    "raw": ("language_", "need_"),
}


# --- Weekly hours parsing ----------------------------------------------------
# hours_of_operation is "Monday 8:00 AM 6:00 PM Tuesday ... " -- a day can be
# entirely absent (a provider open Mon-Thu only won't mention Friday at all)
# or present with zero following times (explicitly closed, e.g. "Saturday
# Sunday" with nothing after). Both mean 0 hours that day, not a parse
# failure. Confirmed against all 10 rows of a real crawl, including a 2-day/
# week home provider and both "8:00 AM" and "08:30 AM" padding styles.
_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_DAY_RE = re.compile(r"\b(" + "|".join(_DAY_NAMES) + r")\b", re.IGNORECASE)
_TIME_RE = re.compile(r"(\d{1,2}:\d{2}\s*[AP]M)", re.IGNORECASE)


class ParseLog:
    def __init__(self, path: "Path | str"):
        self.path = Path(path)
        self.messages: list[str] = []

    def warn(self, msg: str) -> None:
        self.messages.append(msg)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# parse log ({len(self.messages)} messages)\n"
        self.path.write_text(header + "\n".join(self.messages) + "\n")
        print(f"  parse log -> {self.path} ({len(self.messages)} messages)")


# =============================================================================
# Generic helpers
# =============================================================================
def slug(s: object) -> str:
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
    return s != "" and s.lower() not in ("nan", "na")


# =============================================================================
# Shared early steps
# =============================================================================
def null_out_enrichment_on_mismatch(df: pd.DataFrame, log: ParseLog) -> pd.DataFrame:
    """Where errors contains 'id_mismatch', null out MISMATCH_NULL_COLS (the
    site-scraped fields only) rather than dropping the row -- see module
    docstring. Reads 'errors' but does not drop it; the caller drops it
    afterward via NON_FEATURE_COLS."""
    if "errors" not in df.columns:
        return df
    flagged = df["errors"].apply(
        lambda v: _has_value(v) and "id_mismatch" in str(v))
    n = int(flagged.sum())
    if n:
        log.warn(f"[id_mismatch] nulling site-scraped fields for {n} row(s)")
        print(f"  nulling site-scraped fields for {n} id_mismatch row(s) "
              f"(open-data columns untouched)")
        cols = [c for c in MISMATCH_NULL_COLS if c in df.columns]
        df.loc[flagged, cols] = np.nan
    return df


def check_grain_unique(df: pd.DataFrame, col: str, log: ParseLog) -> None:
    if col not in df.columns:
        log.warn(f"[grain] column {col!r} missing -- cannot verify uniqueness")
        print(f"  WARNING: grain column {col!r} not found")
        return
    dups = int(df[col].duplicated().sum())
    if dups:
        log.warn(f"[grain] {col!r} has {dups} duplicate value(s); keep-first later")
        print(f"  WARNING: grain {col!r} not unique ({dups} dupes) -- keep-first applied")
    else:
        print(f"  grain {col!r} is unique ({len(df)} rows)")


def strip_dollars(df: pd.DataFrame) -> pd.DataFrame:
    """Generic currency-string -> float pass, kept for cross-state parity even
    though nothing in the CO schema is a $-field today. A column converts only
    if EVERY non-null value matches a $-amount pattern."""
    money_re = re.compile(r"^\s*\$\s?[\d,]+(?:\.\d+)?\s*$")
    for c in df.columns:
        ser = df[c].dropna().astype(str)
        if len(ser) and ser.str.match(money_re).all():
            df[c] = (df[c].astype(str).str.replace(r"[\$,]", "", regex=True)
                     .replace({"nan": np.nan, "": np.nan}).astype(float))
    return df


# =============================================================================
# Per-field builders — each takes a Series (or df slice) and returns a
# DataFrame of NEW columns aligned to the same index. mode is "full" or "raw".
# =============================================================================
def to_boolean(series: pd.Series, true_values: tuple, false_values: tuple) -> pd.Series:
    """Map a Yes/No- or True/False-style text column to real nullable boolean
    dtype. Unrecognized non-null values become NA (not silently True/False)."""
    tv = {v.lower() for v in true_values}
    fv = {v.lower() for v in false_values}

    def _map(v):
        if not _has_value(v):
            return pd.NA
        s = str(v).strip().lower()
        if s in tv:
            return True
        if s in fv:
            return False
        return pd.NA
    return series.apply(_map).astype("boolean")


def build_categorical_onehot(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Low-cardinality single-value categorical -> one boolean column per
    discovered value. NaN rows get NaN across all indicator columns."""
    vals = sorted({slug(v) for v in series.dropna().unique() if _has_value(v)})
    out = {}
    present = series.apply(_has_value)
    for v in vals:
        col = f"{prefix}_{v}"
        ind = series.apply(lambda x: slug(x) == v if _has_value(x) else np.nan)
        out[col] = ind.where(present, other=np.nan)
    return pd.DataFrame(out, index=series.index)


def build_multivalue(series: pd.Series, delimiter: str, prefix: str, mode: str) -> pd.DataFrame:
    """Delimiter-split multi-value text column (e.g. special_needs, ';'
    separated) -> one column per discovered item.
      full : boolean presence.
      raw  : the item phrase where present, else NaN.
    """
    per_row = []
    for v in series:
        if not _has_value(v):
            per_row.append([])
            continue
        items = [p.strip() for p in str(v).split(delimiter) if p.strip()]
        per_row.append(items)
    schema: dict = {}
    for items in per_row:
        for it in items:
            schema.setdefault(slug(it), it)
    present = series.apply(_has_value)
    out = {}
    for s in sorted(schema):
        phrase = schema[s]
        col = f"{prefix}_{s}"
        hit = [phrase in items for items in per_row]
        if mode == "full":
            out[col] = pd.array(
                [h if present.iloc[i] else pd.NA for i, h in enumerate(hit)],
                dtype="boolean")
        else:
            out[col] = [phrase if h else np.nan for h in hit]
    return pd.DataFrame(out, index=series.index)


def _keyterm_hits(text, vocab: list) -> list:
    s = slug(text)
    hits = []
    for term in vocab:
        kt = slug(term)
        if re.search(r"(?:^|_)" + re.escape(kt) + r"(?:_|$)", s):
            hits.append(kt)
    return hits


def build_keyterm(series: pd.Series, vocab: list, prefix: str, mode: str,
                  log: "ParseLog | None" = None) -> pd.DataFrame:
    """Space-concatenated field with no usable delimiter (languages_spoken)
    -> one column per vocab term, token-boundary matched on slugged text.
      full : boolean presence.
      raw  : the human-readable term where present, else NaN.
    Logs any non-blank cell that produced zero hits (vocabulary gap)."""
    term_slugs = [slug(t) for t in vocab]
    readable = {slug(t): t for t in vocab}
    present = series.apply(_has_value)
    rows = []
    for v in series:
        hits = set(_keyterm_hits(v, vocab)) if _has_value(v) else set()
        if _has_value(v) and not hits and log is not None:
            log.warn(f"[{prefix}] no vocabulary match for: {str(v)[:60]!r}")
        rows.append(hits)
    out = {}
    for ks in term_slugs:
        col = f"{prefix}_{ks}"
        if mode == "full":
            out[col] = pd.array(
                [(ks in hits) if present.iloc[i] else pd.NA
                 for i, hits in enumerate(rows)], dtype="boolean")
        else:
            out[col] = [readable[ks] if ks in hits else np.nan for hits in rows]
    return pd.DataFrame(out, index=series.index)


def derive_date_features(df: pd.DataFrame, log: ParseLog) -> pd.DataFrame:
    """license_issue_date -> license_age_days (nullable Int64), for BOTH sets.

    This used to also emit rating_age_days (from award_date) and
    days_until_rating_expires (from expiration_date). Both were removed as
    target leakage -- see LEAKAGE_COLS above for the measured QWK figures and
    the missingness argument. license_issue_date survives because the licensing
    lifecycle is independent of the rating: every licensed provider has an
    issue date regardless of whether Colorado Shines ever rated them."""
    out = {}
    specs = [("license_issue_date", "license_age_days", REFERENCE_DATE, "sub")]
    for col, new_col, ref, direction in specs:
        if col not in df.columns:
            out[new_col] = pd.array([pd.NA] * len(df), dtype="Int64")
            continue
        dt = pd.to_datetime(df[col], errors="coerce")
        bad = df[col].apply(_has_value) & dt.isna()
        for v in df.loc[bad, col]:
            log.warn(f"[{col}] unparseable date: {str(v)[:40]!r}")
        days = (ref - dt).dt.days if direction == "sub" else (dt - ref).dt.days
        out[new_col] = pd.array(days.astype("Int64"), dtype="Int64")
    return pd.DataFrame(out, index=df.index)


def _parse_weekly_hours(text: str) -> "float | None":
    """One hours_of_operation cell -> total open hours across the week, or
    None if no day structure is found at all (blank/garbage). A day with no
    following time pair contributes 0, not a failure -- see the module-level
    note above _DAY_RE."""
    if not isinstance(text, str) or not text.strip():
        return None
    days = list(_DAY_RE.finditer(text))
    if not days:
        return None
    total = 0.0
    for i, d in enumerate(days):
        start = d.end()
        end = days[i + 1].start() if i + 1 < len(days) else len(text)
        times = _TIME_RE.findall(text[start:end])
        if len(times) >= 2:
            try:
                o = datetime.strptime(times[0].strip().upper().replace(" ", ""), "%I:%M%p")
                c = datetime.strptime(times[1].strip().upper().replace(" ", ""), "%I:%M%p")
                delta = (c - o).total_seconds() / 3600
                total += delta + 24 if delta < 0 else delta  # spans midnight
            except ValueError:
                pass  # malformed time pair for this one day -- contributes 0
    return round(total, 2)


def derive_operating_hours(series: pd.Series, log: ParseLog) -> pd.DataFrame:
    """hours_of_operation free text -> operating_hours_per_week (float), for
    BOTH sets -- a genuinely useful numeric signal (more so than the raw
    day-by-day text) that was missing from `full` entirely until now."""
    vals = []
    for v in series:
        if not _has_value(v):
            vals.append(np.nan)
            continue
        h = _parse_weekly_hours(v)
        if h is None:
            log.warn(f"[hours_of_operation] no day structure found: {str(v)[:60]!r}")
            vals.append(np.nan)
        else:
            vals.append(h)
    return pd.DataFrame({"operating_hours_per_week": vals}, index=series.index)


def build_licensing_history(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """The 5 licensing-history text fields -> has_documented_* booleans (BOTH
    sets) + boilerplate-stripped text (RAW only, NaN where nothing but
    boilerplate remains). See _NO_HISTORY_RE's docstring note above: the
    "no history" phrasing is confirmed on one field, assumed for the other 4
    pending a real example with an actual documented incident."""
    out = {}
    for col in LICENSING_HISTORY_COLS:
        flag_col = _HISTORY_FLAG_NAMES[col]
        if col not in df.columns:
            if flag_col in df.columns:
                out[flag_col] = pd.array(
                    df[flag_col].map({"True": True, "False": False,
                                      True: True, False: False}),
                    dtype="boolean")
            else:
                out[flag_col] = pd.array([pd.NA] * len(df), dtype="boolean")
            if mode == "raw":
                out[col] = [np.nan] * len(df)
            continue
        stripped = df[col].apply(
            lambda v: _NO_HISTORY_RE.sub("", str(v)).strip() if _has_value(v) else np.nan)
        has_doc = df[col].apply(_has_value) & stripped.apply(_has_value)
        out[flag_col] = pd.array(has_doc, dtype="boolean")
        if mode == "raw":
            out[col] = [s if _has_value(s) else np.nan for s in stripped]
    return pd.DataFrame(out, index=df.index)


# =============================================================================
# Shared finalize tail
# =============================================================================
def _is_constant(series: pd.Series, treat_nan_as_level: bool) -> bool:
    if treat_nan_as_level:
        return series.nunique(dropna=False) <= 1
    return series.nunique(dropna=True) <= 1


def drop_leakage_columns(df: pd.DataFrame, log: "ParseLog | None" = None) -> pd.DataFrame:
    """Remove every rating-derived / rating-lifecycle column (LEAKAGE_COLS and
    anything matching LEAKAGE_PREFIXES). qr_rating -- the target -- is
    explicitly exempt. Called from finalize() so it applies to all four
    pipelines; column-only, so rows and row order are untouched."""
    doomed = sorted(
        c for c in df.columns
        if c != "qr_rating" and c != TARGET_COL
        and (c in LEAKAGE_COLS or c.startswith(LEAKAGE_PREFIXES))
    )
    if doomed:
        msg = f"[leakage] dropping {len(doomed)} rating-derived col(s): {doomed}"
        if log is not None:
            log.warn(msg)
        print(f"  {msg}")
        df = df.drop(columns=doomed)
    return df


def finalize(df: pd.DataFrame, which: str, scaffold: dict, log: ParseLog, *,
            keep_invalid_target: bool = False) -> pd.DataFrame:
    """Shared tail -- same sequence as nc_clean_utils.finalize():
    rename -> coerce target -> drop NON_FEATURE_COLS -> drop leakage cols ->
    reindex to scaffold+discovered -> dedup -> drop missing-target rows (unless
    keep_invalid_target) -> drop all-NaN/constant columns -> bool->Int64 cast
    for CSV round-trip safety on the full set.
    """
    treat_nan_as_level = (which == "raw")

    df = df.rename(columns=RENAME_MAP)

    coerced = pd.to_numeric(df["qr_rating"], errors="coerce")
    valid = coerced.isin(list(VALID_TARGET_VALUES))
    n_invalid = int((df["qr_rating"].apply(_has_value) & ~valid).sum())
    if n_invalid:
        disp = "kept (complete)" if keep_invalid_target else "-> missing"
        log.warn(f"[target] {n_invalid} rows with out-of-range/garbage rating {disp}")
    df["qr_rating"] = coerced if keep_invalid_target else coerced.where(valid, other=np.nan)

    df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns], errors="ignore")
    df = drop_leakage_columns(df, log)

    # P1 / P2: direct identifiers, per-provider narratives and coordinates.
    # Dropped BEFORE the scaffold reindex so they cannot survive even if a
    # stale co_columns.json still lists them.
    df = drop_identifier_columns(df)

    scaffold_cols = [c for c in scaffold[which] if c in df.columns]
    prefixes = DISCOVERED_PREFIXES[which]
    discovered = sorted(c for c in df.columns
                        if c not in scaffold_cols and any(c.startswith(p) for p in prefixes))
    keep = scaffold_cols + discovered
    seen, ordered = set(), []
    for c in keep:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    df = df.reindex(columns=ordered)

    before = len(df)
    df = df.drop_duplicates(subset=[ID_COL], keep="first").reset_index(drop=True)
    if len(df) != before:
        log.warn(f"[dedup] removed {before - len(df)} duplicate {ID_COL} row(s)")

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

    # P2 / P3: geography (collapse -> relabel -> one-hot) and the k-anonymity
    # sweep. These run HERE, after the dedup and the target-row drop, because k
    # is a property of the rows that actually ship -- computing it earlier would
    # let a category that is thin in the release pass on the strength of rows
    # that were about to be dropped.
    df = apply_privacy_remediation(df, which)

    protected = {ID_COL, "qr_rating"}
    drop_const = []
    for c in df.columns:
        if c in protected:
            continue
        if df[c].isna().all() or _is_constant(df[c], treat_nan_as_level):
            drop_const.append(c)
    if drop_const:
        log.warn(f"[constant] dropping {len(drop_const)} all-NaN/constant cols: {drop_const}")
    df = df.drop(columns=drop_const)

    if which == "full":
        for c in df.columns:
            if c == ID_COL:
                continue
            if df[c].dtype == "boolean" or pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].astype("Int64")

    return df


def write_output(df: pd.DataFrame, path: "Path | str") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {len(df)} rows x {df.shape[1]} cols -> {path}")


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
STATE = "co"

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
LEVEL_FAMILY_PREFIXES = ('cccap_', 'language_', 'need_', 'type_')

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
