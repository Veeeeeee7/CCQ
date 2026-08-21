"""Serialization of a provider record into natural-language text.

Produces one string per row, used by every text-based method and by the
embedding-based tabular transfer arm. Follows the textual view of PEDCA-FT
(Lu et al., 2025).

Fields are grouped into bracketed sections. False and missing booleans are
omitted, as are missing numerics; true flags render as phrases rather than
`name: True`. An age-tier block is dropped entirely when every field in it is
missing. Compliance history renders in one of three modes (verbose, summary,
abnormal_only).

The bucketing constants below are schema-specific; the serializer is not.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

# -----------------------------------------------------------------------------
# Bucket rules — edit here if the schema changes
# -----------------------------------------------------------------------------

# Columns NEVER serialized (target + any leakage / index cols).
# provider_id is already dropped by utils.load_data(); listed here for safety.
DROP_COLS: set[str] = {"provider_id", "qr_rating"}

CURRICULUM_COL = "curriculum"

# Boolean columns by prefix. A column matching one of these is treated as
# a Yes/No flag: True/Yes/1 → render under its group section; False/No/NaN → skip.
BOOLEAN_PREFIXES: tuple[str, ...] = (
    "has_",
    "environment_has_",
    "ages_served_",
)

# Boolean columns by exact name (don't match the prefix patterns).
BOOLEAN_EXACT: set[str] = {
    "non_profit",
    "accepts_children_new",
    "accepts_children_full_time",
    "accepts_children_part_time",
    "temporary_closure",
}

# Age-tier blocks: each tier's fields share the same suffix.
# Order matters — sections render in this order.
AGE_TIERS: list[tuple[str, str]] = [
    ("under_1_year",          "Under 1 year"),
    ("1_year",                "1 year"),
    ("2_years",               "2 years"),
    ("3_years",               "3 years"),
    ("4_years",               "4 years"),
    ("5_years_kindergarten",  "5 yr / Kindergarten"),
    ("5_years_and_older",     "5 yr and older"),
]

# Field prefixes that belong to an age tier; each maps to a readable label.
# A column counts as part of a tier iff its name == prefix + tier_suffix.
AGE_FIELD_LABELS: dict[str, str] = {
    "weekly_full_day_":      "weekly full-day rate",
    "weekly_before_school_": "weekly before-school rate",
    "weekly_after_school_":  "weekly after-school rate",
    "vacancies_":            "vacancies",
    "#_of_rooms_":           "rooms",
    "staff_child_ratio_":    "staff:child ratio",
    "daily_drop_in_care_":   "daily drop-in rate",
    "day_camp_(min-max)_":   "day camp (min-max)",
}

# Boolean flag groups: which section a flag appears under, and its order.
# Each entry is (section_label, predicate). Predicates run in order, first
# match wins. Anything boolean that doesn't match falls into "[Other]".
BOOLEAN_GROUPS: list[tuple[str, "callable"]] = [
    ("Accepts",        lambda c: c.startswith("accepts_children_")),
    ("Ages served",    lambda c: c.startswith("ages_served_")),
    ("Meals offered",  lambda c: c.startswith("has_meal_")
                                 or c in {"has_infant_meals",
                                          "has_special_diets",
                                          "has_parent_provided_meals"}),
    ("Transport",      lambda c: c.startswith("has_transport_")),
    ("Summer camp",    lambda c: c.startswith("has_summer_camp")),
    ("Services",       lambda c: c.startswith("has_services_")),
    ("Environment",    lambda c: c.startswith("environment_has_")),
]

ComplianceMode = Literal["verbose", "summary", "abnormal_only"]
COMPLIANCE_YEARS = ("2026", "2025", "2024")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_TRUE_TOKENS = {"true", "yes", "1", "1.0", "t", "y"}

def _is_true(v) -> bool:
    """Robust truthiness for mixed-dtype CSV booleans (True / 'Yes' / 1 / '1.0')."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() in _TRUE_TOKENS


def _has_value(v) -> bool:
    """True iff v is non-null and not the empty string."""
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    s = str(v).strip()
    return s != "" and s.lower() != "nan"


def _is_boolean_col(col: str) -> bool:
    return col in BOOLEAN_EXACT or col.startswith(BOOLEAN_PREFIXES)


def _age_tier_of(col: str) -> tuple[str, str] | None:
    """Return (tier_suffix, tier_name) if col is an age-tier field, else None."""
    for tier_suf, tier_name in AGE_TIERS:
        for pre in AGE_FIELD_LABELS:
            if col == pre + tier_suf:
                return tier_suf, tier_name
    return None


def _strip_flag_prefix(col: str) -> str:
    """Drop the category prefix from a boolean flag column for readable display.

    Order matters: longer / more specific prefixes are checked first so we
    strip 'environment_has_' before falling back to 'has_'.
    """
    for pre in ("environment_has_", "has_services_", "has_transport_",
                "has_meal_", "has_summer_camp_", "has_", "ages_served_",
                "accepts_children_"):
        if col.startswith(pre):
            return col[len(pre):].replace("_", " ")
    return col.replace("_", " ")


def _readable(col: str) -> str:
    """Human-readable column label (underscores → spaces)."""
    return col.replace("_", " ")


# -----------------------------------------------------------------------------
# Section renderers
# -----------------------------------------------------------------------------
def _render_provider_info(row: pd.Series, exclude: set[str]) -> str | None:
    """Everything that isn't a flag, age-tier field, compliance field,
    curriculum, or dropped. Numerics + categoricals + free-text all here."""
    bits = []
    for col in row.index:
        if col in exclude or col in DROP_COLS:
            continue
        if col == CURRICULUM_COL:
            continue
        if _is_boolean_col(col):
            continue
        if _age_tier_of(col) is not None:
            continue
        if "compliance" in col:
            continue
        v = row[col]
        if not _has_value(v):
            continue
        bits.append(f"{_readable(col)}: {v}")
    if not bits:
        return None
    return "[Provider Info] " + ". ".join(bits)


def _render_curriculum(row: pd.Series) -> str | None:
    if CURRICULUM_COL not in row.index:
        return None
    v = row[CURRICULUM_COL]
    if not _has_value(v):
        return None
    return f"[Curriculum] {v}"


def _render_boolean_flags(row: pd.Series) -> list[str]:
    """Group True flags into sections, skip Falses, return one string per
    non-empty group."""
    # First-pass: collect all True boolean columns
    true_cols = [c for c in row.index if _is_boolean_col(c) and _is_true(row[c])]

    # Assign each true col to its group (first matching predicate, else "Other")
    grouped: dict[str, list[str]] = {label: [] for label, _ in BOOLEAN_GROUPS}
    grouped["Other"] = []
    for col in true_cols:
        placed = False
        for label, pred in BOOLEAN_GROUPS:
            if pred(col):
                grouped[label].append(_strip_flag_prefix(col))
                placed = True
                break
        if not placed:
            grouped["Other"].append(_strip_flag_prefix(col))

    # Render non-empty groups in declared order, "Other" last
    out = []
    for label, _ in BOOLEAN_GROUPS:
        items = grouped[label]
        if items:
            out.append(f"[{label}] " + ", ".join(items))
    if grouped["Other"]:
        out.append("[Other] " + ", ".join(grouped["Other"]))
    return out


def _render_age_tiers(row: pd.Series) -> list[str]:
    """One section per age tier with at least one populated field. Tiers
    where every field is NaN drop entirely (implicit 'not served')."""
    sections = []
    for tier_suf, tier_name in AGE_TIERS:
        bits = []
        for pre, lbl in AGE_FIELD_LABELS.items():
            col = pre + tier_suf
            if col not in row.index:
                continue
            v = row[col]
            if not _has_value(v):
                continue
            bits.append(f"{lbl}: {v}")
        if bits:
            sections.append(f"[Age tier — {tier_name}] " + "; ".join(bits))
    return sections


def _render_compliance(row: pd.Series, mode: ComplianceMode) -> str | None:
    """Compliance section in one of three modes."""
    comp_cols = [c for c in row.index if "compliance" in c]
    if not comp_cols:
        return None

    if mode == "verbose":
        bits = []
        for c in comp_cols:
            v = row[c]
            if not _has_value(v):
                continue
            bits.append(f"{_readable(c)}: {v}")
        return "[Compliance] " + ". ".join(bits) if bits else None

    if mode == "summary":
        keys = ["compliance"]
        for y in COMPLIANCE_YEARS:
            keys += [f"{y}_compliance_total_rule_violations",
                     f"{y}_compliance_total_rules_met"]
        bits = [f"{_readable(k)}: {row[k]}"
                for k in keys
                if k in row.index and _has_value(row[k])]
        return "[Compliance] " + ". ".join(bits) if bits else None

    if mode == "abnormal_only":
        bits = []
        if "compliance" in row.index and _has_value(row["compliance"]):
            bits.append(f"overall: {row['compliance']}")
        for y in COMPLIANCE_YEARS:
            tv_col = f"{y}_compliance_total_rule_violations"
            if tv_col in row.index and _has_value(row[tv_col]):
                try:
                    if float(row[tv_col]) > 0:
                        bits.append(f"{y} violations: {row[tv_col]}")
                except (ValueError, TypeError):
                    pass
            # find per-category gaps (rules_met < rules_total)
            met_cols = [c for c in row.index
                        if c.startswith(f"{y}_compliance_")
                        and c.endswith("_rules_met")
                        and c != f"{y}_compliance_total_rules_met"]
            for mc in met_cols:
                tc = mc[:-len("_rules_met")] + "_rules_total"
                if tc not in row.index:
                    continue
                m, t = row[mc], row[tc]
                if not (_has_value(m) and _has_value(t)):
                    continue
                try:
                    mf, tf = float(m), float(t)
                except (ValueError, TypeError):
                    continue
                if mf < tf:
                    cat = mc[len(f"{y}_compliance_"):-len("_rules_met")].replace("_", " ")
                    bits.append(f"{y} {cat}: {int(mf)}/{int(tf)}")
        return "[Compliance] " + "; ".join(bits) if bits else None

    raise ValueError(f"Unknown compliance_mode: {mode}")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def serialize_row(
    row: pd.Series,
    compliance_mode: ComplianceMode = "verbose",
) -> str:
    """Serialize one provider row to a structured natural-language string.

    Section order (any empty section is dropped):
      [Provider Info] → [Curriculum] → boolean groups (Accepts, Ages served,
      Meals offered, Transport, Summer camp, Services, Environment, Other) →
      age-tier blocks → [Compliance]
    """
    sections: list[str] = []
    pi = _render_provider_info(row, exclude=set())
    if pi: sections.append(pi)
    cur = _render_curriculum(row)
    if cur: sections.append(cur)
    sections.extend(_render_boolean_flags(row))
    sections.extend(_render_age_tiers(row))
    comp = _render_compliance(row, compliance_mode)
    if comp: sections.append(comp)
    return " ".join(sections)


def serialize_dataframe(
    df: pd.DataFrame,
    compliance_mode: ComplianceMode = "verbose",
) -> pd.Series:
    """Vectorized: serialize every row of df. Returns a pd.Series of strings."""
    return df.apply(lambda r: serialize_row(r, compliance_mode), axis=1)


# -----------------------------------------------------------------------------
# CLI: preview serializations from the raw CSV for sanity-checking
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from utils import load_data, DEFAULT_DATA_PATH

    parser = argparse.ArgumentParser(description="Preview row serializations.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH,
                        help=f"Raw dataset CSV to preview (default: {DEFAULT_DATA_PATH}).")
    parser.add_argument("--n", type=int, default=3,
                        help="Number of rows to print (default 3)")
    parser.add_argument("--compliance", choices=("verbose", "summary", "abnormal_only"),
                        default="verbose")
    args = parser.parse_args()

    df = load_data(args.input)
    sample = df.head(args.n)
    for i, (_, row) in enumerate(sample.iterrows()):
        s = serialize_row(row, compliance_mode=args.compliance)
        n_words = len(s.split())
        print(f"\n=== row {i}  (qr_rating={row['qr_rating']}, "
              f"~{n_words} words, ~{int(n_words*1.3)} tokens) ===\n")
        print(s)
