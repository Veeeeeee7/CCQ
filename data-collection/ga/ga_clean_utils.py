"""
ga_clean_utils.py — Shared utilities for cleaning the Georgia (DECAL) child care
records into the `full` (numeric/boolean) and `raw` (text-preserving) datasets,
each in a standard (valid ratings only) and `complete` (all rows) variant.

The per-field builders are mode-aware: "full" engineers numeric/boolean columns,
"raw" returns the frame untouched. full and raw run the same early steps and
row filtering, so the two outputs are row-aligned.

  * Parse failures are logged and never crash.
  * Counts use nullable Int64 so a parse failure is distinct from a true zero;
    booleans use the nullable "boolean" dtype.
  * GA column names intentionally carry punctuation (`#_of_rooms_...`,
    `activities_scouting_(boy_scouts/girl_scouts)`, apostrophes, ...), so names
    are NOT slugged to [a-z0-9]; each builder preserves it so the engineered
    columns line up with ga_columns.json.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# GA records already carry these names, so RENAME_MAP is an identity safety-net.
ID_COL = "provider_id"
TARGET_COL = "qr_rating"
RENAME_MAP: dict[str, str] = {ID_COL: "provider_id", TARGET_COL: "qr_rating"}

# Quality Rated issues 1-, 2-, or 3-star ratings.
VALID_TARGET_VALUES: set[int] = {1, 2, 3}

# Rows flagged here failed to collect; dropped first so full and raw stay aligned.
ERROR_FLAG_COLS: list[str] = ["errors"]

# Metadata / URLs + identifying fields, dropped up front in both sets.
NON_FEATURE_COLS: list[str] = [
    "location_name", "provider_url", "compliance_compliance_url",
    "download_path", "num_downloadable_files", "program_subtype",
    "for_profit",                       # superseded by the non_profit flag
    "admin_name", "phone", "email", "address", "city", "state", "zip",
    "mailing_address", "mailing_city_state_zip",
    "mailing_city", "mailing_state", "mailing_zip",
]
# `county` and `region` are deliberately NOT dropped here: finalize() hands them
# to the geographic privacy pipeline (collapse below k=5, relabel, one-hot).

# Scaffold columns that should not feed a model, dropped late per set.
TRAINING_EXCLUDE: dict[str, list[str]] = {"full": [], "raw": []}

# GA's scaffolds enumerate every expected engineered column, so finalize
# reindexes strictly to the scaffold (no discovered columns appended).
DISCOVERED_PREFIXES: dict[str, tuple[str, ...]] = {"full": (), "raw": ()}

# Checkmark fields plus the seed's Yes/No flags, coerced in both sets.
# has_liability is handled separately because raw keeps its Yes/No/N/A text.
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

# Tab-joined multi-value fields one-hot expanded in `full`.
SPLIT_ONEHOT_FIELDS: list[str] = [
    "special_hours", "financial_information", "accreditation_status",
    "activities", "other_child_care_types", "family_engagement",
]

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

PROVIDER_TYPE_CODES: list[str] = [
    "CCLC", "EXMT", "FCCLH", "LSS", "GAHS", "DOD", "GAEHS", "UNIV",
]

RATE_PREFIXES: tuple[str, ...] = (
    "weekly_full_day_", "weekly_before_school_", "weekly_after_school_",
    "vacancies_", "#_of_rooms_", "staff_child_ratio_", "daily_drop_in_care_",
    "day_camp_(min-max)_",
)


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
    """Slug for the tab-joined one-hot fields. Folds dashes/comma-space/space to
    '_' but keeps '/', '(', ')', '.', apostrophes, '#', ':' to match
    ga_columns.json."""
    return (str(item).strip().lower()
            .replace(" \u2013 ", "_").replace(" - ", "_").replace("-", "_")
            .replace(", ", "_").replace(" ", "_"))


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
    return df.drop(columns=present, errors="ignore")


def check_grain_unique(df: pd.DataFrame, col: str, log: ParseLog) -> None:
    """Warn if the grain has duplicates; dedup (keep-first) happens in finalize."""
    if col not in df.columns:
        log.warn(f"[grain] column {col!r} missing — cannot verify uniqueness")
        return
    dups = int(df[col].duplicated().sum())
    if dups:
        log.warn(f"[grain] {col!r} has {dups} duplicate value(s); keep-first later")
    else:
        print(f"  grain {col!r} is unique ({len(df)} rows)")


def clean_currency(df: pd.DataFrame) -> pd.DataFrame:
    """registration_fee / activity_fee: '$1,250.00' → 1250.0 (float)."""
    for c in ("registration_fee", "activity_fee"):
        if c in df.columns:
            s = (df[c].astype("string")
                 .str.replace("$", "", regex=False)
                 .str.replace(",", "", regex=False)
                 .str.strip())
            df[c] = pd.to_numeric(s, errors="coerce").astype(float)
    return df


def strip_dollars(df: pd.DataFrame) -> pd.DataFrame:
    """Currency-string → float for any column whose every non-null value is a
    plain $-amount. Columns mixing in ratios ('1:8') or ranges ('$100-$200') are
    left untouched."""
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
    """Map checkmark / Yes-No columns to nullable boolean (<NA> if undeterminable)."""
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


# Per-field builders: mode == "raw" returns df untouched; "full" engineers
# numeric/boolean columns.
def clean_licensed_capacity(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """'Licensed Capacity: 50' → 50 (Int64)."""
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
    """Yes→True / No→False / N/A→<NA>."""
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
    """languages_offered (tab-joined '<lang> Spoken/Taught') → 14 booleans."""
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
    """curriculum_count = number of non-empty curricula (Int64); drops the text."""
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
    """Already-numeric or '$'-string rate column → float."""
    # pandas 3: text is `str` dtype, not object -- an object-only test sends a
    # "$1,234" column straight to to_numeric and nulls the whole column.
    if (pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)):
        s = (series.astype("string")
             .str.replace("$", "", regex=False)
             .str.replace(",", "", regex=False))
        return pd.to_numeric(s, errors="coerce").astype(float)
    return pd.to_numeric(series, errors="coerce").astype(float)


def clean_rates_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Every per-age-group rate column → a `rates_table_*` feature:
        staff_child_ratio  '1:8'        → rates_table_..._ratio... = 8.0
        day_camp_(min-max)  '$100-$200' → rates_table_day_camp_min/max_... floats
        $ weekly rates                  → float
        counts / vacancies              → renamed numeric passthrough"""
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
    f'{prefix}_{value-with-spaces→underscores}' (case/punctuation preserved)."""
    if column_name not in df.columns:
        return df
    values = [v for v in df[column_name].dropna().unique()
              if isinstance(v, str) and v.strip()]
    for v in values:
        new_col = f"{prefix}_{v.strip().replace(' ', '_')}"
        df[new_col] = df[column_name].eq(v).astype("boolean")
    return df


def clean_compliance_table(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Rename '<year>_compliance_<x>' → 'compliance_<year>_<x>' and the bare
    'compliance' value → 'compliance_compliance', cast the rule counts to Int64,
    then one-hot the compliance status."""
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
    """Tab-joined 'Weekday: 6:00 AM - 6:00 PM' rows → span hours for
    weekday/weekend/summer/additional; drops the text."""
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
    """Comma list → 12 month booleans + operating_months_other; 'year round'
    lights all 12."""
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
    """Comma list → 7 weekday booleans; 'every day' lights all 7, 'mon - fri'
    lights Mon–Fri."""
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
    """provider_type code → one boolean column per known code."""
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
    """Each tab-joined multi-value field → one boolean column per discovered
    value, named f'{field}_{split_onehot_slug(value)}'."""
    if mode == "raw":
        return df
    for column_name in columns_to_convert:
        if column_name not in df.columns:
            continue
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


def _is_constant(series: pd.Series, treat_nan_as_level: bool) -> bool:
    """<= 1 unique value; raw counts NaN as its own level."""
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
    """Shared tail.

    keep_invalid_target:
        False: rows whose qr_rating is not in VALID_TARGET_VALUES are dropped.
        True (complete): the coerced numeric rating is kept (non-numeric → <NA>)
            and no target-based row drop is applied.
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

    # Identifiers and coordinates are dropped BEFORE the scaffold reindex so
    # they cannot survive even if ga_columns.json lists them.
    df = drop_identifier_columns(df)

    df.columns = [normalize_column_name(c) for c in df.columns]
    scaf = [normalize_column_name(c) for c in scaffold[which]]

    scaffold_cols = [c for c in dict.fromkeys(scaf) if c in df.columns]
    prefixes = DISCOVERED_PREFIXES[which]
    discovered = sorted(
        c for c in df.columns
        if c not in scaffold_cols and prefixes and c.startswith(prefixes)
    )
    ordered = list(dict.fromkeys(scaffold_cols + discovered))
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

    # Privacy remediation runs after the dedup and target-row drop because k is
    # a property of the rows that actually ship.
    df = apply_privacy_remediation(df, which)

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

    # nullable booleans serialise to 'True'/'False' text, so cast them to Int64
    # to keep `full` numeric after a CSV round-trip.
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
STATE = "ga"

# Small-range integer counts capped so a lone provider at the top of the range
# joins the bucket below; lossless for a tree splitting on ">= cap".
TOPCODE_COLS = {}

# Families whose members are LEVELS of one attribute, so a rare member can be
# merged into "<prefix>other". Prefixes whose members are distinct attributes
# (has_*, num_*, violations_*, monitoring_*, ...) are deliberately excluded:
# OR-ing them together would invent a meaningless feature.
LEVEL_FAMILY_PREFIXES = ('accepts_', 'accreditation_', 'activities_', 'ages_', 'environment_', 'family_', 'financial_', 'languages_', 'operating_', 'other_', 'provider_', 'special_', 'temporary_')

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
