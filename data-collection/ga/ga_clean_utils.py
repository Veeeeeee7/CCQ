"""
ga_clean_utils.py — Shared utilities for cleaning the Georgia (DECAL) childcare-
provider scrape into three datasets that mirror the NC pipeline 1-for-1:

  full     : strictly numeric/boolean (+ provider_id) — classical/tabular ML.
  raw      : text preserved and minimally decomposed — LLM-based methods.
  complete : same engineering as `full` but rows with an invalid/unrated
             qr_rating are KEPT (target coerced, out-of-range/unrated → <NA>).

All three drivers (ga_clean_full.py, ga_clean_raw.py, ga_clean_complete.py)
import from here so the editable constants, per-field builders, and the shared
finalize tail live in exactly one place. Nothing here is GA-specific beyond the
clearly-marked EDITABLE CONSTANTS block; the field builders encode the same
transforms the original cleaning_classification_*.py scripts performed, but are
mode-aware ("full" engineers numeric/boolean columns; "raw" preserves the
original text verbatim).

Design rules carried over from the NC methodology spec:
  * full and raw run the SAME early steps and the SAME row filtering, so the two
    outputs are row-aligned and a single fold file applies to both.
  * Parse failures are logged and never crash (NaN / 0 returned sensibly).
  * Extracted counts use the nullable Int64 dtype so a parse failure is distinct
    from a true zero; booleans use the nullable "boolean" dtype.
  * The GA schema uses column names that intentionally carry punctuation
    (`#_of_rooms_...`, `staff:child_ratios...`, `activities_scouting_(boy_scouts/
    girl_scouts)`, apostrophes, ...). Unlike NC we therefore do NOT slug every
    name to [a-z0-9]; each builder reproduces GA's original, punctuation-
    preserving naming so the engineered columns line up with ga_columns.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# EDITABLE CONSTANTS  — review these for every new dataset / schema change.
# =============================================================================

# --- Identity + target -------------------------------------------------------
# GA records already arrive with these names (the additional-data merge supplies
# provider_id + qr_rating), so RENAME_MAP is an identity safety-net.
ID_COL = "provider_id"
TARGET_COL = "qr_rating"
RENAME_MAP: dict[str, str] = {ID_COL: "provider_id", TARGET_COL: "qr_rating"}

# GA's QRIS issues 1-, 2-, or 3-star ratings. Anything outside this set
# (unrated placeholders, garbage codes, blanks) is coerced to missing and the
# row is dropped for `full`/`raw`; `complete` keeps it (target → <NA>).
VALID_TARGET_VALUES: set[int] = {1, 2, 3}

# --- Source-error rows -------------------------------------------------------
# Rows flagged by any of these columns (non-null / non-empty) failed to scrape.
# Dropped FIRST, before engineering or dedup, so full and raw stay row-aligned.
ERROR_FLAG_COLS: list[str] = ["errors"]

# --- Non-feature columns (dropped up front, BOTH sets) -----------------------
# Pure metadata / URLs / scrape artifacts + identifying/privacy fields. None of
# these appear in either scaffold, so reindex would drop them anyway; removing
# them early keeps the engineered frame small and the intent explicit.
NON_FEATURE_COLS: list[str] = [
    # scrape artifacts / metadata
    "location_name", "provider_url", "compliance_compliance_url",
    "download_path", "num_downloadable_files", "program_subtype",
    "for_profit",                       # superseded by the non_profit flag
    # identifying / privacy (also not in any scaffold)
    "admin_name", "phone", "email", "address", "city", "state", "zip",
    "mailing_address", "mailing_city_state_zip",
    "mailing_city", "mailing_state", "mailing_zip",
]
# NOTE (2026-08-06): `county` and `region` used to be dropped here, which left
# GA as the only state with no geographic feature at all -- and it cost real
# signal, since region_SE and region_CE both cleared the SHAP noise floor. They
# are now kept through the scaffold and handed to the geographic pipeline in
# finalize(), which collapses categories below k=5, relabels them against a
# private map and one-hots the survivors. That is strictly safer than the old
# behaviour was intended to be: the previous release shipped 153 raw county_*
# one-hots, 16 of which named a single provider.

# --- Training-exclusion list (human-reviewed; applied LATE, per set) ----------
# GA reindexes strictly to the scaffold, so privacy/leakage columns are already
# excluded by the scaffold itself. This hook is kept for parity with NC and as a
# documented place to drop any column that DOES match the scaffold but should not
# feed a model. Empty today.
TRAINING_EXCLUDE: dict[str, list[str]] = {"full": [], "raw": []}

# --- Discovered-column prefixes (appended after the scaffold in finalize) -----
# GA's scaffolds enumerate every expected engineered column, so we reindex
# STRICTLY to the scaffold (no discovered append) to reproduce the original
# `df = df[columns]` behaviour. Flip a set's tuple on (e.g. ("activities_",
# "languages_", ...)) if you want novel one-hot categories to survive instead of
# being dropped.
DISCOVERED_PREFIXES: dict[str, tuple[str, ...]] = {"full": (), "raw": ()}

# --- Boolean columns (coerced to nullable boolean in BOTH sets) --------------
# Checkmark fields crawled as bools (serialised True/False/blank → reloaded as
# object) plus the Yes/No flags from the additional-data merge. has_liability is
# handled separately because raw keeps its Yes/No/N/A text.
BOOLEAN_COLS: list[str] = [
    "non_profit",
    "accepts_children_new", "accepts_children_full_time",
    "accepts_children_part_time",
    "ages_served_infant_0_to_12_months", "ages_served_toddler_13mos_to_2yrs",
    "ages_served_preschool_3yrs_to_4yrs", "ages_served_pre_k_served",
    "ages_served_school_age_5yrs_plus", "ages_served_other_than_pre_k",
    "has_transport_to_from_home", "has_transport_to_from_school",
    "has_transport_afterschool_only", "has_transport_georgia_pre_k_only",
    "has_transport_near_public_transport", "has_transport_schoolbus",
    "has_transport_fieldtrips", "has_transport_before_after_school",
    "has_meal_breakfast", "has_meal_lunch", "has_meal_dinner",
    "has_meal_am_snack", "has_meal_pm_snack", "has_special_diets",
    "has_infant_meals", "has_parent_provided_meals",
    "has_summer_camp", "has_summer_camp_before_care", "has_summer_camp_after_care",
    "has_services_caps", "has_services_headstart", "has_services_after_school_only",
    "has_services_religion", "has_services_cacfp",
    "has_services_school_age_summer_care", "has_services_drop_in_care",
    "has_services_sfsp", "has_services_evening_care",
    "environment_has_no_pets", "environment_has_provide_your_own_equipment",
    "environment_has_sports_fields", "environment_has_video_surveillance",
    "environment_has_outdoor_play_area", "environment_has_secure_access",
    "environment_has_tennis_courts", "environment_has_webcam_for_parents",
    "environment_has_pools", "environment_has_smoke_free",
    "qr_participant", "qr_rated", "temporary_closure",
]

# --- Multi-value (tab-joined) fields one-hot expanded in `full` ---------------
SPLIT_ONEHOT_FIELDS: list[str] = [
    "special_hours", "financial_information", "accreditation_status",
    "activities", "other_child_care_types", "family_engagement",
]

# --- Language one-hot target columns -----------------------------------------
LANGUAGE_COLS: list[str] = [
    "languages_arabic_spoken", "languages_arabic_taught",
    "languages_chinese_spoken", "languages_chinese_taught",
    "languages_english_spoken", "languages_english_taught",
    "languages_french_spoken", "languages_french_taught",
    "languages_russian_spoken", "languages_russian_taught",
    "languages_spanish_spoken", "languages_spanish_taught",
    "languages_other_language_spoken", "languages_other_language_taught",
]
_LANGUAGE_FAMILIES = ("arabic", "chinese", "english", "french", "russian", "spanish")

# --- provider_type one-hot codes ---------------------------------------------
PROVIDER_TYPE_CODES: list[str] = [
    "CCLC", "EXMT", "FCCLH", "LSS", "GAHS", "DOD", "GAEHS", "UNIV",
]

# --- Rates-table column detection (name-based; replaces the old positional
#     df.columns[78:134] slice, which was fragile) ----------------------------
RATE_PREFIXES: tuple[str, ...] = (
    "weekly_full_day_", "weekly_before_school_", "weekly_after_school_",
    "vacancies_", "#_of_rooms_", "staff_child_ratio_", "daily_drop_in_care_",
    "day_camp_(min-max)_",
)


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
        print(msg)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# parse log ({len(self.messages)} messages)\n"
        self.path.write_text(header + "\n".join(self.messages) + "\n")
        print(f"  parse log → {self.path} ({len(self.messages)} messages)")


# =============================================================================
# Generic helpers
# =============================================================================
def _has_value(v) -> bool:
    """True for a real, non-empty value (rejects None / NaN / '' / 'nan')."""
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    if v is pd.NA:
        return False
    s = str(v).strip()
    return s != "" and s.lower() != "nan"


def normalize_column_name(name: object) -> object:
    """Collapse curly/back apostrophe variants to a straight quote and trim, so
    one-hot column names line up with the scaffold (which uses straight quotes).
    """
    if not isinstance(name, str):
        return name
    return name.replace("\u2019", "'").replace("`", "'").strip()


def split_onehot_slug(item: str) -> str:
    """GA's exact slug for the tab-joined one-hot fields (special_hours,
    activities, ...). Lowercases and folds dashes/comma-space/space to '_' while
    PRESERVING '/', '(', ')', '.', apostrophes, '#', ':' — matching the column
    names already baked into ga_columns.json."""
    return (str(item).strip().lower()
            .replace(" \u2013 ", "_").replace(" - ", "_").replace("-", "_")
            .replace(", ", "_").replace(" ", "_"))


# =============================================================================
# Shared early steps (run identically for full, raw and complete)
# =============================================================================
def drop_error_rows(df: pd.DataFrame, log: ParseLog) -> pd.DataFrame:
    """Drop rows flagged by any ERROR_FLAG_COLS (non-null / non-empty)."""
    present = [c for c in ERROR_FLAG_COLS if c in df.columns]
    if not present:
        return df.reset_index(drop=True)
    flagged = pd.Series(False, index=df.index)
    for c in present:
        flagged |= df[c].apply(_has_value)
    n = int(flagged.sum())
    log.warn(f"[error-rows] dropping {n} rows flagged by {present}")
    df = df.loc[~flagged].reset_index(drop=True)
    # the flag column has done its job; remove it from both outputs
    return df.drop(columns=present, errors="ignore")


def check_grain_unique(df: pd.DataFrame, col: str, log: ParseLog) -> None:
    """Warn (don't crash) if the chosen grain has duplicate values. Dedup
    (keep-first) happens later in finalize."""
    if col not in df.columns:
        log.warn(f"[grain] column {col!r} missing — cannot verify uniqueness")
        return
    dups = int(df[col].duplicated().sum())
    if dups:
        log.warn(f"[grain] {col!r} has {dups} duplicate value(s); keep-first later")
    else:
        print(f"  grain {col!r} is unique ({len(df)} rows)")


def clean_currency(df: pd.DataFrame) -> pd.DataFrame:
    """registration_fee / activity_fee: '$1,250.00' → 1250.0 (float). Applied to
    BOTH sets (these columns are numeric in both scaffolds). Non-currency / blank
    values coerce to NaN rather than crashing."""
    for c in ("registration_fee", "activity_fee"):
        if c in df.columns:
            s = (df[c].astype("string")
                 .str.replace("$", "", regex=False)
                 .str.replace(",", "", regex=False)
                 .str.strip())
            df[c] = pd.to_numeric(s, errors="coerce").astype(float)
    return df


def strip_dollars(df: pd.DataFrame) -> pd.DataFrame:
    """Generic currency-string → float pass for any *other* column whose every
    non-null value is a plain $-amount (e.g. the weekly rate columns). Columns
    that mix in ratios ('1:8') or ranges ('$100-$200') do NOT all match and are
    left untouched for the rates builder / raw text."""
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


def coerce_booleans(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Map the known checkmark / Yes-No columns to the nullable boolean dtype in
    BOTH sets (True/False where determinable, <NA> otherwise)."""
    def _to_bool(v):
        if v is True or v is False:
            return v
        if not _has_value(v):
            return pd.NA
        s = str(v).strip().lower()
        if s in ("true", "yes", "y", "1", "1.0"):
            return True
        if s in ("false", "no", "n", "0", "0.0"):
            return False
        return pd.NA

    for c in cols:
        if c in df.columns:
            df[c] = pd.array([_to_bool(v) for v in df[c]], dtype="boolean")
    return df


# =============================================================================
# Per-field builders (mutate df → return df).  mode == "raw" keeps the original
# text verbatim; mode == "full" engineers numeric/boolean columns.
# =============================================================================
def clean_licensed_capacity(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """'Licensed Capacity: 50' → 50 (Int64) for `full`; raw keeps the text."""
    if mode == "raw" or "licensed_capacity" not in df.columns:
        return df
    s = df["licensed_capacity"].astype("string").str.split(": ").str[1]
    s = s.replace("Unknown", pd.NA)
    df["licensed_capacity"] = pd.to_numeric(s, errors="coerce").astype("Int64")
    return df


def clean_pre_k_slots(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "raw" or "pre_k_slots_available" not in df.columns:
        return df
    s = df["pre_k_slots_available"].replace("Unknown", pd.NA)
    df["pre_k_slots_available"] = pd.to_numeric(s, errors="coerce").astype("Int64")
    return df


def clean_has_liability(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Yes→True / No→False / N/A→<NA> for `full`; raw keeps the Yes/No/N/A text."""
    if mode == "raw" or "has_liability_insurance" not in df.columns:
        return df

    def _m(v):
        if not _has_value(v):
            return pd.NA
        s = str(v).strip()
        if s == "Yes":
            return True
        if s == "No":
            return False
        return pd.NA

    df["has_liability_insurance"] = pd.array(
        [_m(v) for v in df["has_liability_insurance"]], dtype="boolean")
    return df


def clean_languages(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """languages_offered (tab-joined '<lang> Spoken/Taught') → 14 boolean
    columns for `full`; raw keeps the original languages_offered text."""
    if mode == "raw" or "languages_offered" not in df.columns:
        return df
    rows = []
    for item in df["languages_offered"]:
        row = {c: False for c in LANGUAGE_COLS}
        if _has_value(item) and isinstance(item, str):
            for entry in item.strip().split("\t"):
                e = entry.strip().lower()
                fam = next((f for f in _LANGUAGE_FAMILIES if f in e), "other_language")
                if "spoken" in e:
                    row[f"languages_{fam}_spoken"] = True
                if "taught" in e:
                    row[f"languages_{fam}_taught"] = True
        rows.append(row)
    lang_df = pd.DataFrame(rows, index=df.index)
    for c in LANGUAGE_COLS:
        lang_df[c] = lang_df[c].astype("boolean")
    df = df.drop(columns=["languages_offered"])
    return pd.concat([df, lang_df], axis=1)


def clean_curriculum(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """`full`: curriculum_count = # of distinct non-empty curricula (Int64),
    drop the text. raw keeps the original curriculum text verbatim."""
    if mode == "raw" or "curriculum" not in df.columns:
        return df

    def _count(x):
        if not _has_value(x):
            return 0
        return sum(1 for it in str(x).split("\t") if it.strip())

    df["curriculum_count"] = pd.array(
        [_count(x) for x in df["curriculum"]], dtype="Int64")
    return df.drop(columns=["curriculum"])


def _ratio_value(x):
    """'1:8' → 8/1 = 8.0 (children-per-staff). Non-ratio → NaN."""
    if isinstance(x, str) and ":" in x:
        try:
            a, b = x.split(":")[:2]
            return float(b) / float(a)
        except (ValueError, ZeroDivisionError):
            return np.nan
    return np.nan


def _minmax_value(x, idx):
    """'$100-$200' → 100.0 (idx 0) / 200.0 (idx 1). Non-range → NaN."""
    if isinstance(x, str) and "-" in x:
        try:
            parts = x.replace("$", "").replace(",", "").split("-")
            return float(parts[idx])
        except (ValueError, IndexError):
            return np.nan
    return np.nan


def _to_money_float(series: pd.Series) -> pd.Series:
    """A rate column that is either already numeric (strip_dollars converted it)
    or a leftover '$'-string → float."""
    if series.dtype == object:
        s = (series.astype("string")
             .str.replace("$", "", regex=False)
             .str.replace(",", "", regex=False))
        return pd.to_numeric(s, errors="coerce").astype(float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def clean_rates_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """`full`: every per-age-group rate column → a `rates_table_*` feature:
        staff_child_ratio  '1:8'        → rates_table_..._ratio... = 8.0
        day_camp_(min-max)  '$100-$200' → rates_table_day_camp_min/max_... floats
        $ weekly rates                  → float
        counts / vacancies              → renamed numeric passthrough
    raw keeps the original rate columns (plain $ already stripped to float by
    strip_dollars; ratios and ranges stay as text)."""
    if mode == "raw":
        return df
    rate_cols = [c for c in df.columns if c.startswith(RATE_PREFIXES)]
    for col in rate_cols:
        if "ratio" in col:
            df["rates_table_" + col] = df[col].apply(_ratio_value)
            df = df.drop(columns=[col])
        elif "(min-max)" in col:
            df["rates_table_" + col.replace("(min-max)", "min")] = \
                df[col].apply(lambda x: _minmax_value(x, 0))
            df["rates_table_" + col.replace("(min-max)", "max")] = \
                df[col].apply(lambda x: _minmax_value(x, 1))
            df = df.drop(columns=[col])
        else:
            df["rates_table_" + col] = _to_money_float(df[col])
            df = df.drop(columns=[col])
    return df


def _one_hot_value_column(df: pd.DataFrame, column_name: str, prefix: str) -> pd.DataFrame:
    """Single-value categorical → one boolean column per observed value, named
    f'{prefix}_{value-with-spaces→underscores}' (case/punctuation preserved).
    Mirrors the original one_hot_encode_column()."""
    if column_name not in df.columns:
        return df
    values = [v for v in df[column_name].dropna().unique()
              if isinstance(v, str) and v.strip()]
    for v in values:
        new_col = f"{prefix}_{v.strip().replace(' ', '_')}"
        df[new_col] = df[column_name].eq(v).astype("boolean")
    return df


def clean_compliance_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """`full`: rename '<year>_compliance_<x>' → 'compliance_<year>_<x>' and the
    bare 'compliance' image value → 'compliance_compliance', cast the rule
    met/total/violation counts to Int64, then one-hot the compliance status.
    raw keeps the original '<year>_compliance_*' names and bare 'compliance'."""
    if mode == "raw":
        return df
    renames = {c: "compliance_" + c.replace("_compliance", "")
               for c in df.columns if "compliance" in c}
    df = df.rename(columns=renames)
    for c in df.columns:
        if c.startswith("compliance_") and (
            c.endswith("_rules_met") or c.endswith("_rules_total")
            or c.endswith("_rule_violations")
        ):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return _one_hot_value_column(df, "compliance_compliance", "compliance_compliance")


def _time_to_minutes(t: str) -> int:
    """'6:00 AM' / '11:59 PM' → minutes past midnight."""
    t = t.strip()
    hh_mm, ampm = t.split()
    h, m = (int(x) for x in hh_mm.split(":"))
    ampm = ampm.upper()
    if ampm == "AM":
        if h == 12:
            h = 0
    elif ampm == "PM":
        if h != 12:
            h += 12
    else:
        raise ValueError(f"Unrecognized AM/PM in time: {t}")
    return h * 60 + m


def clean_operating_hours(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """`full`: tab-joined 'Weekday: 6:00 AM - 6:00 PM' rows → numeric span hours
    for weekday/weekend/summer/additional; drop the text. raw keeps the text."""
    if mode == "raw" or "operating_hours" not in df.columns:
        return df
    weekday, weekend, summer, additional = [], [], [], []
    for item in df["operating_hours"]:
        wd = we = su = ad = np.nan
        if _has_value(item):
            try:
                for sub in str(item).split("\t"):
                    span = sub.split(": ", 1)[1]
                    first, second = (x.strip() for x in span.split(" - "))
                    diff = _time_to_minutes(second) - _time_to_minutes(first)
                    if diff < 0:
                        diff += 24 * 60          # crossed midnight
                    hours = diff / 60.0
                    if "Weekday" in sub:
                        wd = hours
                    elif "Weekend" in sub:
                        we = hours
                    elif "Summer" in sub:
                        su = hours
                    elif "Additional" in sub:
                        ad = hours
            except Exception:
                # mirror the original fallback (flag for human review)
                wd = we = ad = np.nan
                su = 6.0
        weekday.append(wd); weekend.append(we); summer.append(su); additional.append(ad)
    df["operating_weekday_hours"] = weekday
    df["operating_weekend_hours"] = weekend
    df["operating_summer_hours"] = summer
    df["operating_additional_hours"] = additional
    return df.drop(columns=["operating_hours"])


_MONTH_MAP = {"jan": "jan", "feb": "feb", "mar": "mar", "apr": "apr", "may": "may",
              "jun": "jun", "jul": "jul", "aug": "aug", "sep": "sep", "oct": "oct",
              "nov": "nov", "dec": "dec"}
_MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                "oct", "nov", "dec"]


def clean_operating_months(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """`full`: comma list → 12 month booleans + operating_months_other; 'year
    round' lights all 12 months. raw keeps the text."""
    if mode == "raw" or "operating_months" not in df.columns:
        return df
    cols = [f"operating_months_{m}" for m in _MONTH_ORDER] + ["operating_months_other"]
    rows = []
    for item in df["operating_months"]:
        row = {c: False for c in cols}
        s = str(item).lower().strip()
        if s not in ("nan", "", "unknown"):
            for month in s.split(", "):
                month = month.strip()
                if month in _MONTH_MAP:
                    row[f"operating_months_{_MONTH_MAP[month]}"] = True
                elif month == "year round":
                    for m in _MONTH_ORDER:
                        row[f"operating_months_{m}"] = True
                elif month == "otherschoolbreak":
                    row["operating_months_other"] = True
        rows.append(row)
    months_df = pd.DataFrame(rows, index=df.index)
    for c in cols:
        months_df[c] = months_df[c].astype("boolean")
    df = df.drop(columns=["operating_months"])
    return pd.concat([df, months_df], axis=1)


_DAY_MAP = {"mo": "mo", "tu": "tu", "we": "we", "th": "th", "fr": "fr",
            "sa": "sa", "su": "su"}
_DAY_ORDER = ["mo", "tu", "we", "th", "fr", "sa", "su"]


def clean_operating_days(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """`full`: comma list → 7 weekday booleans; 'every day' lights all 7,
    'mon - fri' lights Mon–Fri. raw keeps the text."""
    if mode == "raw" or "operating_days" not in df.columns:
        return df
    cols = [f"operating_days_{d}" for d in _DAY_ORDER]
    rows = []
    for item in df["operating_days"]:
        row = {c: False for c in cols}
        s = str(item).lower().strip()
        if s not in ("nan", "", "unknown"):
            for day in s.split(", "):
                day = day.strip()
                if day in _DAY_MAP:
                    row[f"operating_days_{_DAY_MAP[day]}"] = True
                elif day == "every day":
                    for d in _DAY_ORDER:
                        row[f"operating_days_{d}"] = True
                elif day == "mon - fri":
                    for d in _DAY_ORDER[:5]:
                        row[f"operating_days_{d}"] = True
        rows.append(row)
    days_df = pd.DataFrame(rows, index=df.index)
    for c in cols:
        days_df[c] = days_df[c].astype("boolean")
    df = df.drop(columns=["operating_days"])
    return pd.concat([df, days_df], axis=1)


def clean_provider_type(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """`full`: single provider_type code → one boolean column per known code.
    raw keeps the single provider_type text value."""
    if mode == "raw" or "provider_type" not in df.columns:
        return df
    present = [v for v in df["provider_type"].dropna().unique() if v in PROVIDER_TYPE_CODES]
    unknown = [v for v in df["provider_type"].dropna().unique()
               if v not in PROVIDER_TYPE_CODES]
    for v in unknown:
        print(f"  WARNING: unrecognized provider type: {v!r}")
    for code in present:
        df[f"provider_type_{code}"] = df["provider_type"].eq(code).astype("boolean")
    return df.drop(columns=["provider_type"])


def split_one_hot(df: pd.DataFrame, columns_to_convert: list[str], mode: str) -> pd.DataFrame:
    """`full`: each tab-joined multi-value field → one boolean column per
    discovered value, named f'{field}_{split_onehot_slug(value)}'. raw keeps the
    original text columns verbatim."""
    if mode == "raw":
        return df
    for column_name in columns_to_convert:
        if column_name not in df.columns:
            continue
        # discover the value vocabulary at runtime (never hardcode from a sample)
        vocab = set()
        for item in df[column_name].dropna().unique():
            for piece in str(item).split("\t"):
                vocab.add(split_onehot_slug(piece))
        new_cols = sorted(f"{column_name}_{v}" for v in vocab if v)
        block = pd.DataFrame(False, index=df.index, columns=new_cols)
        for i, item in zip(df.index, df[column_name]):
            for piece in str(item).split("\t"):
                if piece == "nan":
                    continue
                key = f"{column_name}_{split_onehot_slug(piece)}"
                if key in block.columns:
                    block.at[i, key] = True
        block = block.astype("boolean")
        df = pd.concat([df, block], axis=1)
        df = df.drop(columns=[column_name])
    return df


# =============================================================================
# Shared finalize tail (identical sequence for all three sets)
# =============================================================================
def _is_constant(series: pd.Series, treat_nan_as_level: bool) -> bool:
    """True if the column carries no information.
      full / complete (strict) : <= 1 non-null unique value.
      raw (nan-as-level)       : <= 1 unique value counting NaN as its own level.
    """
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
    """Shared tail. Order follows the NC methodology spec exactly.

    keep_invalid_target:
        False (full / raw): rows whose qr_rating is not in VALID_TARGET_VALUES
            are coerced to <NA> and dropped.
        True (complete): the coerced numeric rating is kept where it parses and
            non-numeric/out-of-range ratings become <NA>; no target-based row
            drop is applied, so `complete` is a row-superset of `full`.
    """
    treat_nan_as_level = (which == "raw")

    # 1. rename grain + target (identity for GA — safety net)
    df = df.rename(columns=RENAME_MAP)

    # 2. coerce qr_rating; anything not in VALID_TARGET_VALUES → missing (kept
    #    as the coerced numeric for `complete`).
    coerced = pd.to_numeric(df["qr_rating"], errors="coerce")
    valid = coerced.isin(list(VALID_TARGET_VALUES))
    n_invalid = int((df["qr_rating"].apply(_has_value) & ~valid).sum())
    if n_invalid:
        disp = "kept (complete)" if keep_invalid_target else "→ missing"
        log.warn(f"[target] {n_invalid} rows with out-of-range/garbage rating {disp}")
    df["qr_rating"] = coerced if keep_invalid_target else coerced.where(valid, other=np.nan)

    # 3. drop NON_FEATURE_COLS (safety; most already removed pre-engineering)
    df = df.drop(columns=[c for c in NON_FEATURE_COLS if c in df.columns],
                 errors="ignore")

    # 3b. P1 / P2: direct identifiers, per-provider narratives and coordinates.
    #     Dropped BEFORE the scaffold reindex so they cannot survive even if a
    #     stale ga_columns.json still lists them.
    df = drop_identifier_columns(df)

    # 4. normalize apostrophe variants so one-hot names match the scaffold
    df.columns = [normalize_column_name(c) for c in df.columns]
    scaf = [normalize_column_name(c) for c in scaffold[which]]

    # 5. reindex → scaffold[which] (present) + sorted discovered cols (none by
    #    default for GA — strict scaffold, see DISCOVERED_PREFIXES).
    scaffold_cols = [c for c in dict.fromkeys(scaf) if c in df.columns]
    prefixes = DISCOVERED_PREFIXES[which]
    discovered = sorted(
        c for c in df.columns
        if c not in scaffold_cols and prefixes and c.startswith(prefixes)
    )
    ordered = list(dict.fromkeys(scaffold_cols + discovered))
    df = df.reindex(columns=ordered)

    # 6. drop TRAINING_EXCLUDE[which]
    df = df.drop(columns=[c for c in TRAINING_EXCLUDE[which] if c in df.columns],
                 errors="ignore")

    # 7. dedup on provider_id (keep first)
    before = len(df)
    df = df.drop_duplicates(subset=["provider_id"], keep="first").reset_index(drop=True)
    if len(df) != before:
        log.warn(f"[dedup] removed {before - len(df)} duplicate provider_id row(s)")

    # 8. drop rows with missing/invalid target (skipped for `complete`)
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

    # 8b. P2 / P3: geography (collapse -> relabel -> one-hot) and the
    #     k-anonymity sweep. These run HERE, after the dedup and the target-row
    #     drop, because k is a property of the rows that actually ship --
    #     computing it earlier would let a category that is thin in the release
    #     pass on the strength of rows that were about to be dropped.
    df = apply_privacy_remediation(df, which)

    # 9. drop all-NaN and constant columns (never the id/target)
    protected = {"provider_id", "qr_rating"}
    drop_const = []
    for c in df.columns:
        if c in protected:
            continue
        if df[c].isna().all() or _is_constant(df[c], treat_nan_as_level):
            drop_const.append(c)
    if drop_const:
        log.warn(f"[constant] dropping {len(drop_const)} all-NaN/constant cols")
    df = df.drop(columns=drop_const)

    # 10. full/complete must survive a CSV round-trip as numeric: nullable
    #     booleans serialise to 'True'/'False' text, so cast them to Int64.
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
STATE = "ga"

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
LEVEL_FAMILY_PREFIXES = ('accepts_', 'accreditation_', 'activities_', 'ages_', 'environment_', 'family_', 'financial_', 'languages_', 'operating_', 'other_', 'provider_', 'special_', 'temporary_')

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
