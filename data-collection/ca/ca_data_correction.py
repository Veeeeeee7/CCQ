#!/usr/bin/env python3
"""
ca_data_correction.py -- stage-1 repairs to the crawled California CSVs.

Five independent repairs, each opt-in, run in this order:

  (default)                  row-structure repair, described below.
  --fix-language             rewrite basics_language, see sanitize_language().
  --fix-openings-capacity    total openings_capacity across age groups.
  --mark-duplicate-profiles  errors='duplicate_profile' on repeat visits to one
                             provider profile.
  --apply-refetch            merge ca_openings_refetch.csv and audit the
                             seed-to-page licence linkage.

The first two read `input` and write `-o`. The last three patch BOTH
`ca_data/ca_records.csv` and `ca_data/ca_records_anonymized.csv` in place, row
for row, touching only columns that pass through stage 2 unchanged, so the
existing provider_id assignment stays valid. The positional join (anonymized
row i IS records row i) is asserted before any write, against
../data-private/provider_id_map_ca.csv plus a spot check on columns neither
file is allowed to disagree on.

Structural repair: the crawl can drop the newline between two consecutive
records, gluing them onto one physical line. Every record has a fixed 32
fields and the boundary fuses one record's LAST column (`errors`) to the next
record's FIRST column (`facility_number`, a run of digits), so such a line
shows up with 63 fields (32 + 32 - 1) instead of 32. The file is read with the
csv module (which honours RFC-4180 quoted fields that span lines), over-long
rows are split back into 32-field records, and the output imports with a plain
pd.read_csv().

Usage:
    python ca_data_correction.py ca_data/ca_records.csv -o ca_data/ca_records.csv
    python ca_data_correction.py ca_data/ca_records.csv --fix-language -o ca_data/ca_records.csv
    python ca_data_correction.py --fix-openings-capacity
    python ca_data_correction.py --mark-duplicate-profiles
    python ca_data_correction.py --apply-refetch
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent

N_FIELDS = 32
# The `errors` vocabulary, longest first so the empty alternative cannot win a
# prefix match.
ERROR_TOKENS = ("duplicate_profile", "wrong_provider", "not_found",
                "exception", "dne")
# A fused boundary token is errors_A glued to facility_number_B, e.g.
# '410517738', 'not_found410517738', 'exception410517738'.
BOUNDARY_RE = re.compile(r"^(" + "|".join(ERROR_TOKENS) + r"|)(\d+)$")

# csv has a default field-size cap that some long about_text cells exceed.
csv.field_size_limit(10_000_000)


LANGUAGE_COLUMN = "basics_language"

_CODE_TOKEN = re.compile(r"^-?\d+$")
_LABEL_ARTIFACT = re.compile(r"^\d+\.Labels-")


def sanitize_language(value: str | None) -> str | None:
    """Strip page artifacts from a comma-joined basics_language value.

      codes   -- a page variant renders the raw option CODES instead of their
                 labels, so 'English, Spanish' arrives as '00, 01'. The page
                 carries no code table, so a value made only of codes becomes
                 None rather than being guessed at.
      labels  -- a component label bleeds in: 'English, Spanish, 2.Labels-,
                 2.Labels- English'. Only the artifact tokens go.

    Left alone: None, and the literal '-' the site uses to mean "no data".
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return value
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    kept = [t for t in tokens
            if not _CODE_TOKEN.match(t) and not _LABEL_ARTIFACT.match(t)]
    if not kept:
        return None
    return ", ".join(kept)


def split_merged(fields: list[str]) -> list[list[str]] | None:
    """Split a >32-field row back into 32-field records.

    Returns the list of recovered records, or None if the row doesn't
    match the known merge pattern (caller should then drop + report it).
    Handles k records glued together, not just two.
    """
    records: list[list[str]] = []
    rest = fields
    while len(rest) > N_FIELDS:
        boundary = rest[N_FIELDS - 1]  # fused errors_A + facility_number_B
        m = BOUNDARY_RE.match(boundary)
        if not m:
            return None
        errors_a, facility_b = m.group(1), m.group(2)
        records.append(rest[: N_FIELDS - 1] + [errors_a])
        rest = [facility_b] + rest[N_FIELDS:]
    if len(rest) != N_FIELDS:
        return None
    records.append(rest)
    return records


def fix_language(rows: list[list[str]], output: Path) -> int:
    """Apply sanitize_language() to every row and write the result.

    Safe to run against ca_records.csv and ca_records_anonymized.csv alike:
    basics_language passes through stage 2 unchanged.
    """
    if not rows:
        print("Empty file.", file=sys.stderr)
        return 1
    header = rows[0]
    if LANGUAGE_COLUMN not in header:
        print(f"No {LANGUAGE_COLUMN} column in this file.", file=sys.stderr)
        return 1
    idx = header.index(LANGUAGE_COLUMN)

    n_blanked = n_trimmed = 0
    out = [header]
    for row in rows[1:]:
        if len(row) > idx:
            before = row[idx]
            after = sanitize_language(before)
            if after != before:
                if after is None:
                    n_blanked += 1
                else:
                    n_trimmed += 1
                row = list(row)
                row[idx] = "" if after is None else after
        out.append(row)

    with open(output, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out)

    print("\n--- basics_language repair ---")
    print(f"values blanked (codes only):  {n_blanked}")
    print(f"values trimmed (artifact):    {n_trimmed}")
    print(f"rows written:                 {len(out) - 1}")
    print(f"output: {output}")
    return 0


# Columns the anonymized file must agree with the records file on, row for row.
# None of the three repairs writes any of them, so a disagreement means the two
# files have drifted out of positional alignment and nothing may be written.
ALIGNMENT_SPOT_COLS = ("basics_qcc_score", "age_range", "business_hours",
                       "basics_schedule")

VALID_TARGETS = {1.0, 2.0, 3.0, 4.0, 5.0}


def line_terminator(path: Path) -> str:
    """The line ending the file already uses, so a rewrite stays byte-stable."""
    with open(path, "rb") as f:
        chunk = f.read(65536)
    i = chunk.find(b"\n")
    if i > 0 and chunk[i - 1:i] == b"\r":
        return "\r\n"
    return "\n"


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise SystemExit(f"{path}: empty file")
    return rows[0], rows[1:]


def write_table(path: Path, header: list[str], rows: list[list[str]],
                terminator: str) -> None:
    """Write via a temp file + os.replace so an interrupted run never leaves a
    half-written records file behind."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator=terminator)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def col(header: list[str], name: str, path: Path) -> int:
    if name not in header:
        raise SystemExit(f"{path}: no '{name}' column")
    return header.index(name)


def check_alignment(rec_hdr, rec_rows, anon_hdr, anon_rows,
                    rec_path: Path, anon_path: Path, id_map: Path) -> None:
    """Assert `anon_rows[i]` is the anonymized image of `rec_rows[i]`.

    Every repair below is positional, and the anonymized file carries a
    SURROGATE facility_number, so the two files cannot be joined on a key.
    Two independent checks:

      1. ../data-private/provider_id_map_ca.csv maps surrogate -> source; every
         row must resolve back to the records row at the same index.
      2. columns neither file is allowed to disagree on must match row for row,
         which catches drift even if the map is missing or stale.
    """
    if len(rec_rows) != len(anon_rows):
        raise SystemExit(
            f"row-count mismatch: {rec_path.name} has {len(rec_rows)} rows, "
            f"{anon_path.name} has {len(anon_rows)}. The positional join is "
            f"broken; refusing to write.")

    shared = [c for c in ALIGNMENT_SPOT_COLS
              if c in rec_hdr and c in anon_hdr]
    if not shared:
        raise SystemExit("no shared column to verify alignment with")
    ri = [rec_hdr.index(c) for c in shared]
    ai = [anon_hdr.index(c) for c in shared]
    for n, (a, b) in enumerate(zip(rec_rows, anon_rows)):
        for x, y in zip(ri, ai):
            if a[x] != b[y]:
                raise SystemExit(
                    f"row {n}: {shared[ri.index(x)]} differs between "
                    f"{rec_path.name} ({a[x]!r}) and {anon_path.name} "
                    f"({b[y]!r}). Refusing to write.")
    print(f"  alignment: {len(rec_rows)} rows, {len(shared)} spot column(s) "
          f"identical row for row")

    if not id_map.exists():
        print(f"  alignment: {id_map} missing -- surrogate check skipped")
        return
    with open(id_map, newline="", encoding="utf-8") as f:
        lookup = {r["surrogate_provider_id"]: r["source_provider_id"]
                  for r in csv.DictReader(f)}
    rfn = rec_hdr.index("facility_number")
    afn = anon_hdr.index("facility_number")
    bad = 0
    for n, (a, b) in enumerate(zip(rec_rows, anon_rows)):
        if lookup.get(b[afn]) != a[rfn]:
            bad += 1
            if bad <= 3:
                print(f"    row {n}: surrogate {b[afn]!r} -> "
                      f"{lookup.get(b[afn])!r}, expected {a[rfn]!r}",
                      file=sys.stderr)
    if bad:
        raise SystemExit(
            f"{bad} row(s) do not resolve through {id_map.name}. The "
            f"positional join is broken; refusing to write.")
    print(f"  alignment: all {len(rec_rows)} surrogate ids resolve to the "
          f"records row at the same index")


def is_released(qcc: str) -> bool:
    """A released row is error-free (checked by the caller) with a 1-5 QCC
    score."""
    try:
        return float(qcc) in VALID_TARGETS
    except (TypeError, ValueError):
        return False


def provider_uid(url: str) -> str | None:
    """The page identity: the provider_uid query parameter of provider_url.
    Several licence numbers can seed one profile page, and this is the only
    column in the pipeline that says which page a row came from."""
    if not url:
        return None
    return (parse_qs(urlparse(url).query).get("provider_uid") or [None])[0]


def norm_licence(value: str) -> str:
    """Normalize a licence number for comparison; leading zeros and
    punctuation are cosmetic on the site."""
    return re.sub(r"[^0-9A-Za-z]", "", (value or "").strip()).lstrip("0").upper()


CAP_RE = re.compile(r"Capacity\s*:\s*([0-9]+)", re.I)
CAP_STRIP_RE = re.compile(r"\s*Capacity\s*:\s*[0-9]+", re.I)


def sum_openings_capacity(capacity: str, status: str) -> tuple[str, str]:
    """Total capacity across age groups; strip the leftover tokens.

    The site prints one 'Capacity: N' per age group. A row holding only the
    first group's figure, with the rest left inside openings_status, is
    converted to the facility total. Idempotent: after one pass `status` holds
    no 'Capacity:' token.
    """
    extra = CAP_RE.findall(status or "")
    if not extra:
        return capacity, status
    first = int(capacity) if (capacity or "").strip().isdigit() else 0
    total = first + sum(int(n) for n in extra)
    status = CAP_STRIP_RE.sub("", status or "")
    return str(total), re.sub(r"\s{2,}", " ", status).strip()


def fix_openings_capacity(rec_path: Path, anon_path: Path, id_map: Path,
                          dry_run: bool) -> int:
    rec_hdr, rec_rows = read_table(rec_path)
    anon_hdr, anon_rows = read_table(anon_path)
    check_alignment(rec_hdr, rec_rows, anon_hdr, anon_rows,
                    rec_path, anon_path, id_map)

    r_cap, r_st = col(rec_hdr, "openings_capacity", rec_path), col(rec_hdr, "openings_status", rec_path)
    r_err, r_q = col(rec_hdr, "errors", rec_path), col(rec_hdr, "basics_qcc_score", rec_path)
    a_cap, a_st = col(anon_hdr, "openings_capacity", anon_path), col(anon_hdr, "openings_status", anon_path)

    n_all = n_rel = n_comp = 0
    delta_rel = 0
    for i, row in enumerate(rec_rows):
        cap, st = sum_openings_capacity(row[r_cap], row[r_st])
        if cap == row[r_cap] and st == row[r_st]:
            continue
        n_all += 1
        error_free = row[r_err].strip() == ""
        if error_free:
            n_comp += 1
            if is_released(row[r_q]):
                n_rel += 1
                before = int(row[r_cap]) if row[r_cap].strip().isdigit() else 0
                delta_rel += int(cap) - before
        row[r_cap], row[r_st] = cap, st
        anon_rows[i][a_cap], anon_rows[i][a_st] = cap, st

    left = sum(1 for r in rec_rows if CAP_RE.search(r[r_st] or ""))
    assert left == 0, f"{left} row(s) still hold a 'Capacity:' token in openings_status"

    print("\n--- openings_capacity: total across age groups ---")
    print(f"rows changed:                 {n_all}")
    print(f"  of them released (1-5 QCC): {n_rel}")
    print(f"  of them complete (no error):{n_comp}")
    mean = f"{delta_rel / n_rel:.1f}" if n_rel else "n/a"
    print(f"mean increase on released:    {mean}")
    print(f"'Capacity:' left in openings_status: {left}")
    if dry_run:
        print("dry run: nothing written")
        return 0
    write_table(rec_path, rec_hdr, rec_rows, line_terminator(rec_path))
    write_table(anon_path, anon_hdr, anon_rows, line_terminator(anon_path))
    print(f"wrote {rec_path}\nwrote {anon_path}")
    return 0


def _date_key(value: str) -> tuple[int, int, int]:
    m = re.match(r"(\d+)/(\d+)/(\d+)", (value or "").strip())
    return (int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else (0, 0, 0)


def mark_duplicate_profiles(rec_path: Path, anon_path: Path, id_map: Path,
                            dedup_log: Path, sibling_fill: bool,
                            force: bool, dry_run: bool) -> int:
    """Mark every repeat visit to one provider profile errors='duplicate_profile'.

    The seed is a licence (facility) number and several licences can point at
    one provider profile, so some pages are scraped more than once. Page
    identity lives only in provider_url, which stage 2 drops, so this has to
    be a stage-1 mark; every row stays in the file, in its original order.

    Keeper rule, in order:
      (a) a rated row (basics_qcc_score in 1..5) beats an unrated one;
      (b) then the row whose seed facility_number is the licence the page itself
          shows;
      (c) then collection order: the first row wins.
    """
    rec_hdr, rec_rows = read_table(rec_path)
    anon_hdr, anon_rows = read_table(anon_path)
    check_alignment(rec_hdr, rec_rows, anon_hdr, anon_rows,
                    rec_path, anon_path, id_map)

    r_err = col(rec_hdr, "errors", rec_path)
    a_err = col(anon_hdr, "errors", anon_path)
    r_url = col(rec_hdr, "provider_url", rec_path)
    r_fn = col(rec_hdr, "facility_number", rec_path)
    r_lic = col(rec_hdr, "license_number", rec_path)
    r_q = col(rec_hdr, "basics_qcc_score", rec_path)
    r_cap = col(rec_hdr, "openings_capacity", rec_path)
    r_st = col(rec_hdr, "openings_status", rec_path)
    r_upd = col(rec_hdr, "openings_last_updated", rec_path)
    a_cap = col(anon_hdr, "openings_capacity", anon_path)
    a_st = col(anon_hdr, "openings_status", anon_path)
    a_upd = col(anon_hdr, "openings_last_updated", anon_path)

    already = sum(1 for r in rec_rows if r[r_err].strip() == "duplicate_profile")
    if already and not force:
        raise SystemExit(
            f"{already} row(s) already carry errors='duplicate_profile'. "
            "Re-running would regroup only the rows that are still error-free, "
            "and any keeper marked 'wrong_provider' since would be replaced by "
            "a sibling. Pass --force if that is really what you want.")
    if force:
        for i, row in enumerate(rec_rows):
            if row[r_err].strip() == "duplicate_profile":
                row[r_err] = anon_rows[i][a_err] = ""

    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rec_rows):
        if row[r_err].strip() == "":
            groups[provider_uid(row[r_url])].append(i)

    def rank(i: int) -> tuple[int, int, int]:
        row = rec_rows[i]
        own = norm_licence(row[r_lic])
        return (0 if is_released(row[r_q]) else 1,
                0 if own and own == norm_licence(row[r_fn]) else 1,
                i)

    log_rows = [["provider_uid", "group_size", "released_group",
                 "keeper_facility_number", "keeper_reason", "sibling_filled",
                 "dropped_facility_numbers"]]
    n_drop = n_drop_rel = n_fill = n_fill_rel = 0
    for uid, members in sorted(groups.items(), key=lambda kv: min(kv[1])):
        if len(members) < 2:
            continue
        keeper = min(members, key=rank)
        dropped = [i for i in members if i != keeper]
        released_group = any(is_released(rec_rows[i][r_q]) for i in members)
        tier = rank(keeper)
        reason = ("rated+own_licence" if tier[:2] == (0, 0) else
                  "rated" if tier[0] == 0 else
                  "own_licence" if tier[1] == 0 else "first_seen")

        # Sibling fill: a keeper can be blank where a sibling of the same page
        # is not. Copy the three openings fields as a unit -- a capacity
        # without its own status and date would be a fabricated observation.
        filled = ""
        if sibling_fill and not rec_rows[keeper][r_cap].strip():
            donors = [i for i in dropped if rec_rows[i][r_cap].strip()]
            if donors:
                src = max(donors, key=lambda i: (_date_key(rec_rows[i][r_upd]), -i))
                for ri, ai in ((r_cap, a_cap), (r_st, a_st), (r_upd, a_upd)):
                    rec_rows[keeper][ri] = rec_rows[src][ri]
                    anon_rows[keeper][ai] = rec_rows[src][ri]
                filled = rec_rows[src][r_fn]
                n_fill += 1
                if is_released(rec_rows[keeper][r_q]):
                    n_fill_rel += 1

        for i in dropped:
            rec_rows[i][r_err] = anon_rows[i][a_err] = "duplicate_profile"
            n_drop += 1
            if is_released(rec_rows[i][r_q]):
                n_drop_rel += 1
        log_rows.append([uid, len(members), int(released_group),
                         rec_rows[keeper][r_fn], reason, filled,
                         "|".join(rec_rows[i][r_fn] for i in dropped)])

    kept_rel = sum(1 for r in rec_rows
                   if r[r_err].strip() == "" and is_released(r[r_q]))
    kept_comp = sum(1 for r in rec_rows if r[r_err].strip() == "")
    print("\n--- duplicate profiles marked ---")
    print(f"groups with >1 row:           {sum(1 for v in groups.values() if len(v) > 1)}")
    print(f"rows marked duplicate_profile:{n_drop}  (released {n_drop_rel})")
    print(f"sibling openings fills:       {n_fill}  (released {n_fill_rel})")
    print(f"error-free rows remaining:    {kept_comp}  (released {kept_rel})")
    print(f"rows in {rec_path.name}:      {len(rec_rows)} (unchanged)")
    if dry_run:
        print("dry run: nothing written")
        return 0
    write_table(rec_path, rec_hdr, rec_rows, line_terminator(rec_path))
    write_table(anon_path, anon_hdr, anon_rows, line_terminator(anon_path))
    dedup_log.parent.mkdir(parents=True, exist_ok=True)
    write_table(dedup_log, log_rows[0], log_rows[1:], "\r\n")
    print(f"wrote {rec_path}\nwrote {anon_path}\nwrote {dedup_log} "
          f"({len(log_rows) - 1} groups, keeper and dropped seeds recorded)")
    return 0


# Columns the refetch re-observes. All pass through stage 2 unchanged, which is
# what makes patching both files in place equivalent to re-running it.
REFETCH_COLS = ("openings_last_updated", "openings_status",
                "openings_capacity", "tags", "tags_count")


def apply_refetch(rec_path: Path, anon_path: Path, refetch_path: Path,
                  id_map: Path, audit_path: Path, merge_tags: bool,
                  mark_wrong: bool, dry_run: bool) -> int:
    """Merge ca_openings_refetch.csv into both stage files, positionally.

    The sidecar is keyed by facility_number and holds one row per page visited.
    Only fetch_status == 'ok' rows are merged; a blocked or 404 page leaves the
    original values alone.

    Also writes the linkage audit. The refetch records EVERY licence printed
    on the page, which turns "the page's licence differs from the seed" into a
    verdict: the seed is either on the page (correct linkage, possibly a
    multi-licence profile) or it is not (wrong provider, marked and dropped).
    """
    if not refetch_path.exists():
        raise SystemExit(f"{refetch_path} not found -- run ca_refetch_openings.py first")
    with open(refetch_path, newline="", encoding="utf-8") as f:
        side = list(csv.DictReader(f))
    ok = {}
    for r in side:
        if (r.get("fetch_status") or "").strip() != "ok":
            continue
        fn = (r.get("facility_number") or "").strip()
        if fn in ok:
            raise SystemExit(f"{refetch_path}: facility_number {fn} appears twice")
        ok[fn] = r
    print(f"  sidecar: {len(side)} row(s), {len(ok)} usable (fetch_status=ok)")

    rec_hdr, rec_rows = read_table(rec_path)
    anon_hdr, anon_rows = read_table(anon_path)
    check_alignment(rec_hdr, rec_rows, anon_hdr, anon_rows,
                    rec_path, anon_path, id_map)

    r_fn = col(rec_hdr, "facility_number", rec_path)
    r_err = col(rec_hdr, "errors", rec_path)
    r_q = col(rec_hdr, "basics_qcc_score", rec_path)
    a_err = col(anon_hdr, "errors", anon_path)
    cols = [c for c in REFETCH_COLS if merge_tags or not c.startswith("tags")]
    ri = {c: col(rec_hdr, c, rec_path) for c in cols}
    ai = {c: col(anon_hdr, c, anon_path) for c in cols}
    r_st = ri["openings_status"]

    seen = set()
    changed = defaultdict(int)
    n_rows = n_cap_gained = 0
    audit = [["facility_number", "seed_normalized", "page_licences",
              "verdict", "marked"]]
    verdicts = defaultdict(int)
    n_marked = 0
    for i, row in enumerate(rec_rows):
        fn = row[r_fn].strip()
        rf = ok.get(fn)
        if rf is None:
            continue
        if fn in seen:
            raise SystemExit(f"{rec_path.name}: facility_number {fn} appears twice")
        seen.add(fn)
        n_rows += 1
        had_cap = bool(row[ri["openings_capacity"]].strip())
        for c in cols:
            new = (rf.get(c) or "").strip()
            if new != row[ri[c]]:
                changed[c] += 1
            row[ri[c]] = new
            anon_rows[i][ai[c]] = new
        if not had_cap and row[ri["openings_capacity"]].strip():
            n_cap_gained += 1

        page = [p for p in (rf.get("license_numbers_all") or "").split("|") if p.strip()]
        norm_page = [norm_licence(p) for p in page]
        seed = norm_licence(fn)
        if not norm_page:
            verdict = "no_licence_on_page"
        elif seed == norm_page[0]:
            verdict = "own_licence"
        elif seed in norm_page:
            verdict = "multi_licence"
        else:
            verdict = "wrong_provider"
        verdicts[verdict] += 1
        marked = ""
        if (verdict == "wrong_provider" and mark_wrong
                and row[r_err].strip() == ""):
            row[r_err] = anon_rows[i][a_err] = "wrong_provider"
            marked = "wrong_provider"
            n_marked += 1
        audit.append([fn, seed, "|".join(page), verdict, marked])

    missing = sorted(set(ok) - seen)
    if missing:
        raise SystemExit(
            f"{len(missing)} sidecar facility_number(s) are not in "
            f"{rec_path.name} (first: {missing[:3]}). Refusing to write.")

    left = [r[r_fn] for r in rec_rows if CAP_RE.search(r[r_st] or "")]
    if left:
        raise SystemExit(
            f"{len(left)} merged openings_status value(s) still contain a "
            f"'Capacity:' token (first: {left[:3]}). The refetch is supposed to "
            f"write an already-summed capacity with the tokens stripped; "
            f"merging these would double-count if --fix-openings-capacity is "
            f"re-run. Refusing to write.")

    kept_rel = sum(1 for r in rec_rows
                   if r[r_err].strip() == "" and is_released(r[r_q]))
    empty_cap = sum(1 for r in rec_rows
                    if r[r_err].strip() == "" and is_released(r[r_q])
                    and not r[ri["openings_capacity"]].strip())
    print("\n--- refetch merged ---")
    print(f"rows patched:                 {n_rows}")
    for c in cols:
        print(f"  {c:<24} {changed[c]} cell(s) changed")
    print(f"rows that gained a capacity:  {n_cap_gained}")
    print("\n--- linkage audit ---")
    for v in ("own_licence", "multi_licence", "wrong_provider",
              "no_licence_on_page"):
        print(f"  {v:<20} {verdicts[v]}")
    print(f"rows marked wrong_provider:   {n_marked}"
          f"{'' if mark_wrong else '  (marking disabled)'}")
    print(f"\nreleased rows after this merge: {kept_rel}")
    print(f"released rows still with no openings_capacity: {empty_cap}")
    if dry_run:
        print("dry run: nothing written")
        return 0
    write_table(rec_path, rec_hdr, rec_rows, line_terminator(rec_path))
    write_table(anon_path, anon_hdr, anon_rows, line_terminator(anon_path))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(audit_path, audit[0], audit[1:], "\r\n")
    print(f"wrote {rec_path}\nwrote {anon_path}\nwrote {audit_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", type=Path, nargs="?", help="raw crawled CSV "
                    "(structural repair and --fix-language only)")
    ap.add_argument("-o", "--output", type=Path, help="repaired CSV to write "
                    "(structural repair and --fix-language only)")
    ap.add_argument("--fix-language", action="store_true",
                    help=f"rewrite {LANGUAGE_COLUMN} through sanitize_language() "
                         "instead of doing the row-structure repair")
    ap.add_argument("--fix-openings-capacity", action="store_true",
                    help="total openings_capacity across age groups and strip "
                         "the leftover 'Capacity: N' tokens from openings_status")
    ap.add_argument("--mark-duplicate-profiles", action="store_true",
                    help="errors='duplicate_profile' on every repeat visit to "
                         "a provider profile already in the file")
    ap.add_argument("--apply-refetch", action="store_true",
                    help="merge the ca_refetch_openings.py sidecar and write "
                         "the seed-to-page licence audit")
    ap.add_argument("--records", type=Path, default=HERE / "ca_data" / "ca_records.csv")
    ap.add_argument("--anon", type=Path,
                    default=HERE / "ca_data" / "ca_records_anonymized.csv")
    ap.add_argument("--refetch", type=Path,
                    default=HERE / "ca_data" / "ca_openings_refetch.csv")
    ap.add_argument("--id-map", type=Path,
                    default=HERE.parent / "data-private" / "provider_id_map_ca.csv")
    ap.add_argument("--dedup-log", type=Path,
                    default=HERE / "ca_data" / "ca_dedup_log.csv")
    ap.add_argument("--linkage-audit", type=Path,
                    default=HERE / "ca_data" / "ca_linkage_audit.csv")
    ap.add_argument("--no-sibling-fill", action="store_true",
                    help="--mark-duplicate-profiles: do not copy an openings "
                         "observation from a dropped sibling into a blank keeper")
    ap.add_argument("--no-tags", action="store_true",
                    help="--apply-refetch: merge the openings columns only, "
                         "leaving tags and tags_count as crawled")
    ap.add_argument("--no-mark-wrong-provider", action="store_true",
                    help="--apply-refetch: write the linkage audit but do not "
                         "set errors='wrong_provider' on the confirmed cases")
    ap.add_argument("--force", action="store_true",
                    help="--mark-duplicate-profiles: recompute even though the "
                         "file already carries duplicate_profile marks")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what the in-place modes would do, write nothing")
    args = ap.parse_args()

    modes = [args.fix_openings_capacity, args.mark_duplicate_profiles,
             args.apply_refetch]
    if sum(modes) > 1:
        ap.error("pick one in-place mode at a time; they are ordered "
                 "(--fix-openings-capacity, then --mark-duplicate-profiles, "
                 "then --apply-refetch)")
    if any(modes):
        if args.input or args.output:
            ap.error("the in-place modes patch --records and --anon; they do "
                     "not take input/-o")
        if args.fix_openings_capacity:
            return fix_openings_capacity(args.records, args.anon, args.id_map,
                                         args.dry_run)
        if args.mark_duplicate_profiles:
            return mark_duplicate_profiles(
                args.records, args.anon, args.id_map, args.dedup_log,
                not args.no_sibling_fill, args.force, args.dry_run)
        return apply_refetch(args.records, args.anon, args.refetch, args.id_map,
                             args.linkage_audit, not args.no_tags,
                             not args.no_mark_wrong_provider, args.dry_run)

    if args.input is None or args.output is None:
        ap.error("the structural repair and --fix-language need input and -o")

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if args.fix_language:
        return fix_language(rows, args.output)

    if not rows:
        print("Empty file.", file=sys.stderr)
        return 1

    header = rows[0]
    if len(header) != N_FIELDS:
        print(f"WARNING: header has {len(header)} fields, expected {N_FIELDS}.", file=sys.stderr)

    clean: list[list[str]] = [header]
    n_ok = n_merged = n_recovered = n_dropped = 0

    # enumerate over LOGICAL csv rows (a multi-line quoted field counts once)
    for rownum, row in enumerate(rows[1:], start=2):
        if len(row) == N_FIELDS:
            clean.append(row)
            n_ok += 1
        elif len(row) > N_FIELDS:
            pieces = split_merged(row)
            if pieces:
                clean.extend(pieces)
                n_merged += 1
                n_recovered += len(pieces)
                print(f"  repaired row {rownum}: {len(row)} fields -> {len(pieces)} records")
            else:
                n_dropped += 1
                print(f"  DROPPED row {rownum}: {len(row)} fields, pattern unrecognized", file=sys.stderr)
        else:  # fewer than 32 -- a different kind of corruption
            n_dropped += 1
            print(f"  DROPPED row {rownum}: only {len(row)} fields", file=sys.stderr)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(clean)

    n_records = len(clean) - 1
    print("\n--- repair summary ---")
    print(f"clean rows (32 fields):       {n_ok}")
    print(f"merged rows repaired:         {n_merged}  -> {n_recovered} records recovered")
    print(f"rows dropped (unrecognized):  {n_dropped}")
    print(f"total records written:        {n_records}")
    print(f"output: {args.output}")

    # Prove the output now imports under the strict C parser.
    try:
        import pandas as pd

        df = pd.read_csv(args.output)
        print(f"\nVERIFIED: pandas read {df.shape[0]} rows x {df.shape[1]} cols, no parser error.")
    except Exception as e:  # noqa: BLE001
        print(f"\nWARNING: pandas still failed to parse output: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
