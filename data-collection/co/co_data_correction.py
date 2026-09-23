#!/usr/bin/env python3
"""
co_data_correction.py -- provider-matching rules for the Colorado Shines crawl.

Pure helper module: no network, no file IO, no state. Imported by
co_crawler.py and co_refetch.py to turn a search result into the right
provider's detail page.

  * The result card carries NO licence number, so a pick can only be verified
    on the detail page: `licence_matches`.
  * Chains and multi-site operators share a ZIP, so a ZIP/city hit only orders
    the candidates to probe, it never accepts one: `rank_candidates`.
  * Names carrying '&' or '+', doubled spaces or a trailing "DBA ..." clause
    break the site's program search: `sanitize_program_name`.
"""
from __future__ import annotations

import re

__all__ = [
    "licence_digits", "licence_matches", "sanitize_program_name",
    "name_prefix", "rank_candidates",
]

_NON_DIGIT = re.compile(r"\D")
_DBA_SUFFIX = re.compile(r"\s+DBA\b.*$", re.IGNORECASE)


def licence_digits(value) -> str:
    """Digits only, so '1,715,779' / '#1715779' / 1715779.0 all compare equal."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return _NON_DIGIT.sub("", text)


def licence_matches(found, provider_id) -> bool:
    """Does a detail page's License Number belong to the provider we asked for?

    Both sides are reduced to digits; a blank on either side is NOT a match.
    The licence is the only signal on the page that identifies the provider
    unambiguously.
    """
    got, want = licence_digits(found), licence_digits(provider_id)
    return bool(got) and bool(want) and got == want


def sanitize_program_name(name) -> str:
    """A program name the site's search can actually handle.

    '&' and '+' are query operators to the search backend, not literals (a
    name containing them can return thousands of results). A trailing
    "DBA <trading name>" clause is not part of the registered program name on
    the site. Doubled whitespace is collapsed last.
    """
    text = str(name or "")
    text = _DBA_SUFFIX.sub("", text)
    text = text.replace("&", " ").replace("+", " ")
    return re.sub(r"\s+", " ", text).strip()


def name_prefix(name, words: int = 3) -> str:
    """First few words of a sanitized name, for the one retry a `not_found`
    gets. Long registered names often differ from the site's listing after the
    first few words ('... Child Care Center' vs '... Learning Center')."""
    parts = sanitize_program_name(name).split()
    return " ".join(parts[:words])


def rank_candidates(candidates, rec, exclude_id=None) -> list:
    """Search-result cards in the order their detail pages should be probed.

    ZIP hits first, then city hits, then everything else, and the page a
    previous crawl already proved wrong (`exclude_id`) dead last. Ordering is
    stable within each band, so the ranking is reproducible. The ranking only
    decides what to TRY first; `licence_matches` decides what is accepted.
    """
    zip_code = str(rec.get("zip") or "").strip()
    if zip_code.upper() in ("", "NA", "NAN"):
        zip_code = ""
    city = str(rec.get("city") or "").strip()
    if city.upper() in ("", "NA", "NAN"):
        city = ""
    city_lower = city.lower()

    def band(card):
        if exclude_id and card.get("id") == exclude_id:
            return 3
        blob = card.get("text_blob") or ""
        if zip_code and zip_code in blob:
            return 0
        if city_lower and city_lower in blob.lower():
            return 1
        return 2

    return [c for _, c in sorted(enumerate(candidates),
                                 key=lambda item: (band(item[1]), item[0]))]
