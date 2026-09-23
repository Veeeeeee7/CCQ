"""
co_cleaning_utils.py — Shared utilities for cleaning the Colorado Shines
child care records into two row-aligned datasets:

  full : strictly numeric/boolean (+ provider_id).
  raw  : text preserved and maximally decomposed.

Colorado is two-source: qr_rating and the open-data columns come from the
seed, the rest from the provider's detail page. Rows flagged in `errors` are
therefore NOT dropped -- a page failure leaves the target and open-data columns
intact and the page columns already NaN. The one special case is
`id_mismatch`: the page may belong to another provider, so its page-derived
fields are nulled (null_out_enrichment_on_mismatch).
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "provider_id"        # already carries this name on the input
TARGET_COL = "quality_rating"  # native open-data column -> renamed to qr_rating

RENAME_MAP = {TARGET_COL: "qr_rating"}

# Colorado Shines' five levels. "NA" (the source's missing-value marker), blank
# and out-of-range values are coerced to missing.
VALID_TARGET_VALUES: set[int] = {1, 2, 3, 4, 5}

# Match metadata, redundant identifiers and identity/contact fields, dropped
# from both sets. "errors" is NOT here: null_out_enrichment_on_mismatch() reads
# it first and each clean script drops it afterwards.
NON_FEATURE_COLS: list[str] = [
    "detail_url", "match_method", "n_candidates",
    "provider_name", "street_address", "city", "zip", "state",
    "phone", "website",
    "ecc", "ccrr", "school_district", "governing_body",
]

# Target leakage: no exported feature may be derived from the rating or from
# its administrative lifecycle (award / expiration / renewal dates and anything
# computed from them). Colorado awards no rating cycle to Level 1 providers, so
# even the *presence* of an award/expiration date is a class label; these are
# removed outright. license_issue_date / license_age_days stay: the licensing
# lifecycle is independent of the rating.
LEAKAGE_COLS: list[str] = [
    "rating_on_site",
    "award_date",
    "expiration_date",
    "rating_age_days",
    "days_until_rating_expires",
]

# Any column starting with one of these is a derived form of a LEAKAGE_COLS
# field and is dropped by the same sweep.
LEAKAGE_PREFIXES: tuple[str, ...] = (
    "rating_on_site", "award_date", "expiration_date",
    "rating_age", "days_until_rating", "rating_expir", "rating_renew",
    "rating_effective",
)

# Reference date for "days since" features: the data's snapshot date.
REFERENCE_DATE = pd.Timestamp("2026-07-06")

# languages_spoken is space-joined ("English Spanish") and "Sign Language" is
# two words, so it is matched token-boundary-aware against a vocabulary rather
# than split. This is the site's own "Language" search picklist.
LANGUAGE_KEYTERMS: list[str] = [
    "English", "Spanish", "French", "German", "Mandarin",
    "Sign Language", "Other",
]

# languages_spoken is the picklist above joined with the site's free-text
# "other languages" box, so a cell reads "English Spanish Arabic Russian" or
# "English ASL".
#
# Two tables, both RECOGNITION vocabularies -- neither decides what ships.
# Every recognised language gets its own `language_*` term and the k>=5 sweep
# folds the ones held by fewer than five providers into `language_other`.
#
#   LANGUAGE_SYNONYMS     alternate spellings of a PICKLIST term.
#   OTHER_LANGUAGE_TERMS  languages outside the picklist, with the spellings
#                         and misspellings providers actually typed.
#
# Both are matched token-boundary-wise on the slugged cell, so accents and
# punctuation ("Espanól", "Q'anjob'al") slug the same way on both sides.
LANGUAGE_SYNONYMS: "dict[str, tuple[str, ...]]" = {
    "Sign Language": ("ASL", "American Sign Language", "Sign Lang", "Baby Sign"),
    "Spanish": ("Espanol", "Español", "Espanól"),
    "English": ("Ingles", "Englih"),
    "French": ("Frenchch",),
}

# Keep sorted by canonical name; add spellings rather than new canonical names
# for a misspelling. The residual-token log in build_language_features says
# when this list has gone stale.
OTHER_LANGUAGE_TERMS: "dict[str, tuple[str, ...]]" = {
    "Afrikaans": ("Afrikaans",),
    "Amharic": ("Amharic", "Amaric", "Ameharic", "Amherick"),
    "Arabic": ("Arabic",),
    "Burmese": ("Burmese",),
    "Chinese": ("Chinese",),          # kept apart from the picklist's Mandarin
    "Cora": ("Cora",),
    "Czech": ("Czech",),
    "Dutch": ("Dutch",),
    "Farsi": ("Farsi", "Persian", "Dari", "Afghni"),   # Dari: Afghan Persian
    "Flemish": ("Flemish",),
    "Frisian": ("Frisian",),
    "Greek": ("Greek",),
    "Gujarati": ("Gujarati",),
    "Hebrew": ("Hebrew",),
    "Hindi": ("Hindi", "Hindu"),
    "Hmong": ("Hmong", "Mung"),
    "Hungarian": ("Hungarian",),
    "Italian": ("Italian",),
    "Japanese": ("Japanese",),
    "Kanjobal": ("Q'anjob'al", "Qanjobal", "Konajobal", "Konajoval"),
    "Kinyarwanda": ("Kinyarwanda",),
    "Korean": ("Korean",),
    "Latin": ("Latin",),
    "Marathi": ("Marathi",),
    "Mixtec": ("Mixtec", "Mixteco"),
    "Mongolian": ("Mongolian",),
    "Nepali": ("Nepali", "Napalese", "Nepalese"),
    "Norwegian": ("Norwegian",),
    "Oromo": ("Oromo",),
    "Pashto": ("Pashto",),
    "Polish": ("Polish",),
    "Portuguese": ("Portuguese", "Portugues", "Portugese"),
    "Punjabi": ("Punjabi", "Pumjabi"),
    "Russian": ("Russian",),
    "Serbian": ("Serbian",),
    "Somali": ("Somali", "Somalian"),
    "Swahili": ("Swahili",),
    "Tagalog": ("Tagalog", "Filipino"),
    "Tamil": ("Tamil",),
    "Thai": ("Thai",),
    "Tigrinya": ("Tigrinya", "Tigrigna", "Tegrigna"),
    "Turkish": ("Turkish",),
    "Twi": ("Twi",),
    "Ukrainian": ("Ukrainian", "Ukranian"),
    "Urdu": ("Urdu", "Urdi", "Urdum"),
    "Vietnamese": ("Vietnamese", "Vietnames"),
}

LANGUAGE_VOCABULARY: list[str] = LANGUAGE_KEYTERMS + sorted(OTHER_LANGUAGE_TERMS)

# Words providers type into the box that are not languages; only used by the
# residual-token tripwire (tokens under 4 characters are ignored there).
# "Guatemalan" and "Indie" name no identifiable language and are deliberately
# NOT guessed at.
LANGUAGE_STOPWORDS: "frozenset[str]" = frozenset("""
    access also american available basic bilingual communicate consistent course
    current deaf depends drop during employees enrollment evenings families
    fluent functioning ghana guatemalan have help indie intermediate ipad lang
    language languages limited lingual little minimal multi multiple needed
    night none nurse offer only others outside owner parent praticed provider
    rely rooms season sign site software some space speak speaking speaks special
    spoken summer system talkers teacher teachers that there translate
    translators twice variety very week well will with words
""".split())

# Colorado Shines renders the age list in its own display order, so one age SET
# arrives as several strings ("Infants, Toddlers, Preschool" vs "Preschool,
# Infants, Toddlers"), each counted separately against k. Canonical order is
# developmental; unknown items sort last, alphabetically.
LICENSED_TO_SERVE_ORDER: tuple = (
    "Home", "Infants", "Toddlers", "Preschool",
    "Mixed Preschool School Age", "School-age",
)
_LTS_RANK = {v.lower(): i for i, v in enumerate(LICENSED_TO_SERVE_ORDER)}

# Each of the five history accordions carries its OWN "nothing to report"
# sentinel; three are also prefixed with a generic disclaimer about the public
# file review, which is NOT a sentinel. Within every section "sentinel present"
# and "contains an m/d/yyyy entry" are complementary, so sentinel-ABSENCE is the
# rule. Date presence is not usable: each section is truncated at 3,000
# characters, which cuts the dated rows off some adverse_actions_text cells.
_NO_HISTORY_RES: "dict[str, re.Pattern]" = {
    "inspection_report_text":
        re.compile(r"No Inspections? reported in the last 3 years", re.IGNORECASE),
    "complaints_text":
        re.compile(r"No Complaints? reported in the last 3 years", re.IGNORECASE),
    "stage_ii_text":
        re.compile(r"No Stage II Investigations? reported in the last 3 years",
                   re.IGNORECASE),
    "injury_investigations_text":
        re.compile(r"No Injur(?:y|ies) reported in the last 3 years", re.IGNORECASE),
    "adverse_actions_text":
        re.compile(r"No Actions Reported", re.IGNORECASE),
}

# The generic disclaimer, stripped from the raw text copy of a section.
_DISCLAIMER_RE = re.compile(
    r"information is not currently available on the system.*?"
    r"public file review[^.]*\.?",
    flags=re.IGNORECASE | re.DOTALL,
)

# A dated entry line, "m/d/yyyy <report id> <Outcome> Link to ROI". Used only
# by the drift tripwire in build_licensing_history, never to set a flag.
_HISTORY_ENTRY_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
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
# every field that comes from the (possibly wrong) detail page.
MISMATCH_NULL_COLS: list[str] = [
    "license_number_on_site", "rating_on_site", "description",
    "hours_of_operation", "license_type", "license_issue_date",
    "licensed_to_serve", "capacity_on_site", "phone", "website",
    "languages_spoken", "special_needs", "head_start",
    "accepts_cccap_on_site", "accepting_new_children",
    "openings_infant", "openings_toddler", "openings_preschool",
    "openings_school_age",
] + LICENSING_HISTORY_COLS

# Engineered-column prefixes appended after the scaffold (sorted) in finalize.
# provider_service_type and county are plain text columns in raw, so only
# their full-set one-hot forms are dynamic.
DISCOVERED_PREFIXES: dict[str, tuple[str, ...]] = {
    "full": ("type_", "county_", "licensetype_", "language_", "need_"),
    "raw": ("language_", "need_"),
}

# hours_of_operation is "Monday 8:00 AM 6:00 PM Tuesday ... " -- a day can be
# entirely absent or present with no following times (closed). Both mean 0
# hours that day, not a parse failure.
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


def null_out_enrichment_on_mismatch(df: pd.DataFrame, log: ParseLog) -> pd.DataFrame:
    """Where errors contains 'id_mismatch', null out MISMATCH_NULL_COLS rather
    than dropping the row. Reads 'errors' but does not drop it."""
    if "errors" not in df.columns:
        return df
    flagged = df["errors"].apply(
        lambda v: _has_value(v) and "id_mismatch" in str(v))
    n = int(flagged.sum())
    if n:
        log.warn(f"[id_mismatch] nulling page-derived fields for {n} row(s)")
        print(f"  nulling page-derived fields for {n} id_mismatch row(s) "
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
    """Currency-string -> float for any column whose EVERY non-null value is a
    $-amount (a no-op on the current CO schema)."""
    money_re = re.compile(r"^\s*\$\s?[\d,]+(?:\.\d+)?\s*$")
    for c in df.columns:
        ser = df[c].dropna().astype(str)
        if len(ser) and ser.str.match(money_re).all():
            df[c] = (df[c].astype(str).str.replace(r"[\$,]", "", regex=True)
                     .replace({"nan": np.nan, "": np.nan}).astype(float))
    return df


# Per-field builders: each returns NEW columns aligned to the input index.
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
        # Nullable boolean, not object: an object column of Python bools is
        # written by to_csv as 'True'/'False' text, and finalize's full-view
        # cast only converts real boolean dtypes.
        out[col] = pd.array(
            [(slug(x) == v) if _has_value(x) else pd.NA for x in series],
            dtype="boolean")
    del present
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


def _token_hit(slugged: str, phrase: str) -> bool:
    """Token-boundary match of one phrase inside an already-slugged cell."""
    kt = slug(phrase)
    return bool(kt) and bool(
        re.search(r"(?:^|_)" + re.escape(kt) + r"(?:_|$)", slugged))


def normalize_languages(value):
    """Rewrite the free-text 'other languages' box onto canonical terms.

    Alternate spellings of a picklist value ("ASL", "Espanol") are folded onto
    that value; every other recognised language is appended under its canonical
    name so build_keyterm can see it. Picklist values already in the cell are
    left exactly as they are, and nothing that is not a recognised language is
    ever turned into a flag -- "Some Spanish" sets Spanish, "None" and "N/A"
    set nothing.

    The return value feeds build_keyterm only; languages_spoken itself is not a
    released column.
    """
    if not _has_value(value):
        return value
    slugged = slug(value)
    extra: list[str] = []
    for term, spellings in LANGUAGE_SYNONYMS.items():
        if any(_token_hit(slugged, sp) for sp in spellings):
            extra.append(term)
    for term, spellings in OTHER_LANGUAGE_TERMS.items():
        if any(_token_hit(slugged, sp) for sp in spellings):
            extra.append(term)
    return " ".join([str(value), *extra]) if extra else value


def language_residual_tokens(value) -> list:
    """Tokens of a languages_spoken cell that no term or stop word explains.

    The vocabulary-gap tripwire. build_keyterm's zero-hit warning cannot do
    this job: nearly every provider with an unrecognised language also lists
    English, so the cell still produces a hit.
    """
    if not _has_value(value):
        return []
    slugged = slug(value)
    for phrases in (LANGUAGE_KEYTERMS, *LANGUAGE_SYNONYMS.values(),
                    *OTHER_LANGUAGE_TERMS.values()):
        for phrase in ([phrases] if isinstance(phrases, str) else phrases):
            slugged = re.sub(r"(?:^|_)" + re.escape(slug(phrase)) + r"(?=_|$)",
                             "_", slugged)
    return [t for t in slugged.split("_")
            if len(t) >= 4 and t not in LANGUAGE_STOPWORDS]


def build_language_features(series: pd.Series, mode: str,
                            log: "ParseLog | None" = None) -> pd.DataFrame:
    """languages_spoken -> language_* columns over the full vocabulary.

    Rare terms are NOT pruned here: the k>=5 sweep folds every language held by
    fewer than five providers into language_other.
    """
    normalized = series.apply(normalize_languages)
    if log is not None:
        gaps: dict = {}
        for v in series:
            for token in set(language_residual_tokens(v)):
                gaps[token] = gaps.get(token, 0) + 1
        if gaps:
            worst = sorted(gaps.items(), key=lambda kv: (-kv[1], kv[0]))[:25]
            log.warn(f"[language] {len(gaps)} unrecognised residual token(s) "
                     f"over {sum(gaps.values())} row-mentions -- extend "
                     f"OTHER_LANGUAGE_TERMS or LANGUAGE_STOPWORDS: {worst}")
    return build_keyterm(normalized, LANGUAGE_VOCABULARY, "language", mode, log)


def canonicalize_licensed_to_serve(value):
    """One age SET, spelled one way.

    De-duplicate the comma-delimited items case-insensitively and sort them
    into developmental order (LICENSED_TO_SERVE_ORDER); anything the site adds
    later sorts last, alphabetically. Matching is on the whole comma-delimited
    item, never on a substring -- "Mixed Preschool School Age" contains the
    words of two other bands.
    """
    if not _has_value(value):
        return value
    seen, items = set(), []
    for part in (p.strip() for p in str(value).split(",")):
        if part and part.lower() not in seen:
            seen.add(part.lower())
            items.append(part)
    items.sort(key=lambda x: (_LTS_RANK.get(x.lower(), len(_LTS_RANK)), x.lower()))
    return ", ".join(items)


def build_licensed_to_serve(series: pd.Series,
                            log: "ParseLog | None" = None) -> pd.Series:
    """canonicalize_licensed_to_serve over a column, logging unknown bands
    (expected only for types Colorado Shines does not rate)."""
    out = series.apply(canonicalize_licensed_to_serve)
    if log is not None:
        unknown: dict = {}
        for v in series:
            if not _has_value(v):
                continue
            for part in (p.strip() for p in str(v).split(",")):
                if part and part.lower() not in _LTS_RANK:
                    unknown[part] = unknown.get(part, 0) + 1
        if unknown:
            log.warn(f"[licensed_to_serve] {len(unknown)} item(s) outside "
                     f"LICENSED_TO_SERVE_ORDER (sorted last): "
                     f"{sorted(unknown.items(), key=lambda kv: (-kv[1], kv[0]))}")
    return out


def derive_date_features(df: pd.DataFrame, log: ParseLog) -> pd.DataFrame:
    """license_issue_date -> license_age_days (nullable Int64), for BOTH sets.
    No rating-lifecycle dates are used (see LEAKAGE_COLS)."""
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
    None if no day structure is found. A day with no time pair contributes 0."""
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
    BOTH sets."""
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


def build_licensing_history(df: pd.DataFrame, mode: str,
                            log: "ParseLog | None" = None) -> pd.DataFrame:
    """The 5 licensing-history text fields -> has_documented_* booleans (BOTH
    sets) + disclaimer-stripped text (RAW only).

    A section is DOCUMENTED when it has text and that text does not carry the
    section's own sentinel (_NO_HISTORY_RES); it is False when the sentinel is
    there; and it is NA when the section has no text at all, which on this
    source means the provider's page was never read -- not that nothing
    happened.

    On co_records_anonymized.csv the five narratives are absent and the flags
    are already present, so they are only re-typed here. The text branch
    applies when the input is co_records.csv.
    """
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
        sentinel = _NO_HISTORY_RES[col]
        known = df[col].apply(_has_value)
        documented = [
            (not bool(sentinel.search(str(v)))) if k else pd.NA
            for k, v in zip(known, df[col])
        ]
        out[flag_col] = pd.array(documented, dtype="boolean")
        if log is not None:
            odd = sum(1 for k, v in zip(known, df[col])
                      if k and not sentinel.search(str(v))
                      and not _HISTORY_ENTRY_RE.search(str(v)))
            if odd:
                log.warn(f"[{col}] {odd} section(s) match neither the "
                         f"sentinel nor a dated entry (expected only for "
                         f"truncated adverse_actions_text tables)")
        if mode == "raw":
            stripped = df[col].apply(
                lambda v: _DISCLAIMER_RE.sub("", str(v)).strip()
                if _has_value(v) else np.nan)
            out[col] = [s if _has_value(s) else np.nan for s in stripped]
    return pd.DataFrame(out, index=df.index)


def _is_constant(series: pd.Series, treat_nan_as_level: bool) -> bool:
    if treat_nan_as_level:
        return series.nunique(dropna=False) <= 1
    return series.nunique(dropna=True) <= 1


def drop_leakage_columns(df: pd.DataFrame, log: "ParseLog | None" = None) -> pd.DataFrame:
    """Remove every LEAKAGE_COLS / LEAKAGE_PREFIXES column; the target is
    exempt."""
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
    """Shared tail:
    rename -> coerce target -> drop NON_FEATURE_COLS -> drop leakage cols ->
    reindex to scaffold+discovered -> dedup -> drop missing-target rows (unless
    keep_invalid_target) -> privacy remediation -> drop all-NaN/constant
    columns -> bool->Int64 cast for CSV round-trip safety on the full set.
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

    # Identifiers and coordinates are dropped BEFORE the scaffold reindex so
    # they cannot survive even if co_columns.json lists them.
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

    # Privacy remediation runs after the dedup and target-row drop because k is
    # a property of the rows that actually ship.
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


# >>> BEGIN PRIVACY BLOCK (shared verbatim by all 12 states) >>>
import json as _json
import random as _random
import re as _re
from pathlib import Path as _Path

import numpy as _np
import pandas as _pd

# --- per-state ----------------------------------------------------------------
STATE = "co"

# Small-range integer counts capped so a lone provider at the top of the range
# joins the bucket below; lossless for a tree splitting on ">= cap".
TOPCODE_COLS = {}

# Families whose members are LEVELS of one attribute, so a rare member can be
# merged into "<prefix>other". Prefixes whose members are distinct attributes
# (has_*, num_*, violations_*, monitoring_*, ...) are deliberately excluded:
# OR-ing them together would invent a meaningless feature.
LEVEL_FAMILY_PREFIXES = ('cccap_', 'language_', 'need_', 'type_')

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
