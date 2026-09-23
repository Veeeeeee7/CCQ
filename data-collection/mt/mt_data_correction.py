#!/usr/bin/env python3
"""
mt_data_correction.py -- offline repairs to Montana's seed, and the one-way
propagation of a re-run join into the anonymized file derived from it.

mt_anonymize.py draws a FRESH provider_id permutation on every run, so it is
not re-run after the fact; corrections are applied here, in place.

Three modes, each opt-in:

  --fix-seed-city     Repair three mt_data/mt_seed.csv `city` cells that hold a
                      whole swallowed table row from a wrapped PDF row (the
                      real city is the first token). Run it BEFORE
                      `mt_crawler.py --join-only`. Idempotent.

  --sync-anonymized   Propagate a re-run join into mt_records_anonymized.csv and
                      data-private/provider_id_map_mt.csv: copy the pass-through
                      columns, replay the `name_*` keyterm decomposition, and
                      re-point the id map at the licence numbers the join now
                      holds. It NEVER touches `provider_number` in the anonymized
                      file, which is the surrogate id.

  --verify-alignment  Check only: mt_records.csv, the anonymized file and the
                      id map still describe the same rows in the same order.

Run from inside mt/:

    python mt_data_correction.py --fix-seed-city
    python mt_crawler.py --join-only
    python mt_data_correction.py --sync-anonymized
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'mt_data')
SEED_PATH = os.path.join(OUT_DIR, 'mt_seed.csv')
RECORDS_PATH = os.path.join(OUT_DIR, 'mt_records.csv')
ANON_PATH = os.path.join(OUT_DIR, 'mt_records_anonymized.csv')
MAP_PATH = os.path.join(HERE, os.pardir, 'data-private', 'provider_id_map_mt.csv')

# program_name -> the real city (the first token of the garbled string).
SEED_CITY_FIXES = {
    'A Place to Grow Preschool': 'Kalispell',
    'Montessori Plus International': 'Missoula',
    'Missoula Parent Co-op/Kid Central': 'Missoula',
}

# Columns mt_anonymize.py copies through verbatim. provider_number is absent:
# in the anonymized file it is the surrogate, not the licence number.
PASS_THROUGH = ['star_level', 'city', 'program_type', 'ccrr_region',
                'provider_type', 'county', 'zip_code', 'match_method',
                'match_score', 'errors', 'provider_id_source']

# Seed-derived columns the join cannot change: the positional alignment key.
ALIGN_COLS = ['star_level', 'program_type', 'ccrr_region']


# ---------------------------------------------------------------------------
# csv helpers -- csv.writer with these settings reproduces
# pandas.to_csv(index=False) byte for byte.
# ---------------------------------------------------------------------------

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f'[error] {path} is empty')
    return rows, list(rows[0].keys())


def write_csv(path, rows, fields):
    """Write via a temp file + os.replace, so an interrupted run cannot leave
    a half-written file."""
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def _norm_key(s):
    """Copy of mt_seed.py's `_norm_key`."""
    s = (s or '').lower()
    s = s.replace('&', ' and ')
    s = re.sub(r"[’']", '', s)
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    s = re.sub(r'\b(llc|inc|incorporated|co)\b', '', s)
    return re.sub(r'\s+', ' ', s).strip()


# ---------------------------------------------------------------------------
# --fix-seed-city
# ---------------------------------------------------------------------------

def fix_seed_city() -> int:
    rows, fields = read_csv(SEED_PATH)
    stars = [r for r in rows if r.get('seed_source') == 'stars_ratings']
    print(f'[seed] {len(rows)} rows ({len(stars)} stars_ratings, '
          f'{len(rows) - len(stars)} licensing_roster)')

    n_fixed = n_already = 0
    for name, city in SEED_CITY_FIXES.items():
        hits = [r for r in stars if r['program_name'] == name]
        if len(hits) != 1:
            sys.exit(f'[error] {name!r}: expected exactly 1 stars_ratings row, '
                     f'found {len(hits)}')
        row = hits[0]
        if row['city'] == city:
            print(f'  [ok]   {name!r} already reads {city!r}')
            n_already += 1
            continue
        # Only repair a cell that is actually garbled (has digits, starts with
        # the real city); anything else means an unexpected seed.
        if not re.search(r'\d', row['city']) or not row['city'].startswith(city):
            sys.exit(f'[error] {name!r}: city {row["city"]!r} does not look '
                     f'like the known garbled value starting {city!r}')
        print(f'  [fix]  {name!r}\n         {row["city"]!r}\n      -> {city!r}')
        row['city'] = city
        row['city_key'] = _norm_key(city)
        n_fixed += 1

    if n_fixed:
        write_csv(SEED_PATH, rows, fields)
        print(f'[seed] repaired {n_fixed} city cell(s) -> {SEED_PATH}')
    else:
        print(f'[seed] nothing to do ({n_already} already correct)')

    bad = [r['program_name'] for r in stars if re.search(r'\d', r['city'])]
    if bad:
        sys.exit(f'[error] {len(bad)} stars_ratings row(s) still have a digit '
                 f'in city: {bad}')
    print(f'[seed] no stars_ratings city contains a digit; '
          f'{len({r["city"] for r in stars})} distinct cities')
    return 0


# ---------------------------------------------------------------------------
# --sync-anonymized / --verify-alignment
# ---------------------------------------------------------------------------

def _load_three():
    records, rec_fields = read_csv(RECORDS_PATH)
    anon, anon_fields = read_csv(ANON_PATH)
    id_map, map_fields = read_csv(MAP_PATH)
    if len(records) != len(anon):
        sys.exit(f'[error] row counts differ: {RECORDS_PATH} {len(records)} vs '
                 f'{ANON_PATH} {len(anon)}. The positional join is broken; '
                 f'nothing was written.')
    for i, (r, a) in enumerate(zip(records, anon)):
        for c in ALIGN_COLS:
            if r[c] != a[c]:
                sys.exit(f'[error] row {i} disagrees on {c}: '
                         f'{r[c]!r} vs {a[c]!r}. The two files are no longer '
                         f'row-aligned; nothing was written.')
    return (records, rec_fields), (anon, anon_fields), (id_map, map_fields)


def _map_index(id_map):
    by_surrogate = {}
    for row in id_map:
        sid = row['surrogate_provider_id']
        if sid in by_surrogate:
            sys.exit(f'[error] surrogate id {sid} appears twice in the id map')
        by_surrogate[sid] = row
    return by_surrogate


def _keyterm_columns(program_names):
    """Replay mt_anonymize.py's program_name -> name_* decomposition, using
    its vocabulary directly."""
    sys.path.insert(0, HERE)
    import mt_anonymize as anon
    aliases = getattr(anon, 'NAME_KEYTERM_ALIASES', {})
    cell_slugs = ['_' + anon.slug(v) + '_' if v else '' for v in program_names]
    out = {}
    for keyterm in anon.NAME_KEYTERMS:
        tokens = ['_' + anon.slug(t) + '_'
                  for t in [keyterm] + list(aliases.get(keyterm, ()))]
        column = f'{anon.KEYTERM_PREFIX}_{anon.slug(keyterm)}'
        out[column] = [keyterm if any(t in cs for t in tokens) else ''
                       for cs in cell_slugs]
    return out


def _report_relinks(records, anon, by_surrogate):
    """Classify every row whose real provider_number moved since stage 2 ran."""
    moved = []
    for i, (r, a) in enumerate(zip(records, anon)):
        sid = a['provider_number']
        if sid not in by_surrogate:
            sys.exit(f'[error] row {i}: surrogate id {sid!r} is not in '
                     f'{MAP_PATH}. Nothing was written.')
        old = by_surrogate[sid]['source_provider_id']
        new = r['provider_number']
        if old != new:
            kind = ('license -> license' if old.startswith('PV') and new.startswith('PV')
                    else 'synthetic -> license' if new.startswith('PV')
                    else 'license -> synthetic' if old.startswith('PV')
                    else 'synthetic -> synthetic')
            moved.append((i, sid, old, new, kind))
    return moved


def verify_alignment() -> int:
    (records, _), (anon, _), (id_map, _) = _load_three()
    by_surrogate = _map_index(id_map)
    moved = _report_relinks(records, anon, by_surrogate)
    print(f'[align] {len(records)} rows, aligned on {ALIGN_COLS}')
    print(f'[align] id map: {len(id_map)} rows, '
          f'{len({r["source_provider_id"] for r in id_map})} distinct sources')
    if moved:
        print(f'[align] {len(moved)} row(s) whose licence number differs from '
              f'the map (run --sync-anonymized to propagate):')
        for i, sid, old, new, kind in moved:
            print(f'   row {i:>3} surrogate {sid:>3}  {old} -> {new}  ({kind})')
    else:
        print('[align] every row\'s licence number matches the id map')
    n_pass = sum(1 for r, a in zip(records, anon)
                 for c in PASS_THROUGH if r[c] != a[c])
    print(f'[align] {n_pass} pass-through cell(s) differ between stage 1 and '
          f'stage 2')
    return 0


def sync_anonymized() -> int:
    (records, _), (anon, anon_fields), (id_map, map_fields) = _load_three()
    by_surrogate = _map_index(id_map)

    # --- 1. the crosswalk ------------------------------------------------
    moved = _report_relinks(records, anon, by_surrogate)
    print(f'[sync] {len(records)} rows aligned on {ALIGN_COLS}')
    if moved:
        print(f'[sync] re-pointing {len(moved)} crosswalk row(s):')
        for i, sid, old, new, kind in moved:
            print(f'   row {i:>3} surrogate {sid:>3}  {old} -> {new}  ({kind})')
        print('   ' + '; '.join(f'{k}: {v}' for k, v in
                                Counter(m[4] for m in moved).items()))
    for i, sid, old, new, kind in moved:
        by_surrogate[sid]['source_provider_id'] = new

    # --- 2. pass-through columns -----------------------------------------
    per_col = Counter()
    for r, a in zip(records, anon):
        for c in PASS_THROUGH:
            if r[c] != a[c]:
                a[c] = r[c]
                per_col[c] += 1
    print(f'[sync] pass-through cells copied: {sum(per_col.values())} '
          f'{dict(per_col) if per_col else ""}')

    # --- 3. name_* keyterms ----------------------------------------------
    replayed = _keyterm_columns([r['program_name'] for r in records])
    name_col_diffs = Counter()
    for column, values in replayed.items():
        if column not in anon_fields:
            sys.exit(f'[error] {column} is not a column of {ANON_PATH}. A new '
                     f'keyterm cannot be added without re-running stage 2; '
                     f'nothing was written.')
        for a, v in zip(anon, values):
            if a[column] != v:
                a[column] = v
                name_col_diffs[column] += 1
    stale = [c for c in anon_fields
             if c.startswith('name_') and c not in replayed]
    if stale:
        sys.exit(f'[error] {ANON_PATH} has name_* column(s) the current '
                 f'vocabulary no longer produces: {stale}. Nothing was written.')
    print(f'[sync] name_* cells replayed: {sum(name_col_diffs.values())} '
          f'{dict(name_col_diffs) if name_col_diffs else ""}')

    # --- 4. invariants ----------------------------------------------------
    pn = [r['provider_number'] for r in records]
    dup = [k for k, v in Counter(pn).items() if v > 1]
    if dup:
        sys.exit(f'[error] duplicate provider_number in {RECORDS_PATH}: {dup}. '
                 f'Nothing was written.')
    sids = [a['provider_number'] for a in anon]
    dup = [k for k, v in Counter(sids).items() if v > 1]
    if dup:
        sys.exit(f'[error] duplicate surrogate id in {ANON_PATH}: {dup}. '
                 f'Nothing was written.')
    src = [m['source_provider_id'] for m in id_map]
    dup = [k for k, v in Counter(src).items() if v > 1]
    if dup:
        sys.exit(f'[error] the id map is no longer a bijection -- two '
                 f'surrogates now point at {dup}. Nothing was written.')
    if set(src) != set(pn):
        only_map = sorted(set(src) - set(pn))[:5]
        only_rec = sorted(set(pn) - set(src))[:5]
        sys.exit(f'[error] the id map and the records file describe different '
                 f'provider sets. map-only: {only_map}; records-only: '
                 f'{only_rec}. Nothing was written.')

    # --- 5. write ---------------------------------------------------------
    write_csv(ANON_PATH, anon, anon_fields)
    write_csv(MAP_PATH, id_map, map_fields)
    print(f'[sync] wrote {ANON_PATH} ({len(anon)} rows x {len(anon_fields)} cols)')
    print(f'[sync] wrote {MAP_PATH} ({len(id_map)} rows)')
    print('[sync] surrogate ids untouched: every released provider_id is stable')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--fix-seed-city', action='store_true',
                   help='repair the three garbled mt_seed.csv city cells')
    g.add_argument('--sync-anonymized', action='store_true',
                   help='propagate a re-run join into stage 2 and the id map')
    g.add_argument('--verify-alignment', action='store_true',
                   help='check the positional join and the id map; write nothing')
    args = ap.parse_args()

    if args.fix_seed_city:
        return fix_seed_city()
    if args.sync_anonymized:
        return sync_anonymized()
    return verify_alignment()


if __name__ == '__main__':
    sys.exit(main())
