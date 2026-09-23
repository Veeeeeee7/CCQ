#!/usr/bin/env python3
"""
co_refetch.py -- re-request the Colorado Shines pages a crawl missed.

Rows whose `errors` column marks them as having no usable detail page (the
WRONG provider's page, `id_mismatch`, or no page at all: `exception`,
`not_found`, `ambiguous_no_match`, `no_search_box`) cannot be redone by the
crawler, whose resume skips every provider already in co_records.csv.

This is a targeted repair, not a recrawl. It re-requests only the ids in the id
file, accepts a page only when its License Number equals the seed licence
(co_data_correction.licence_matches), and writes everything to a SIDECAR file.
Merging is a separate, explicit step.

    # 1. build the id list (rated rows first, so the run can be cut short)
    python co_refetch.py --build-ids

    # 2. fetch. Stop it whenever you like and re-run with --resume.
    python co_refetch.py --limit 5            # quick test first
    python co_refetch.py --resume             # the rest

    # 3. merge the sidecar into co_data/co_records.csv, in place, positionally
    python co_refetch.py --merge --dry-run    # read this first: writes nothing
    python co_refetch.py --merge

Step 3 patches co_records.csv only; co_records_anonymized.csv is not touched.

Transport is plain HTTP, not the crawler's browser: the detail page is fully
server-rendered and the program search also matches on the licence number, so
one GET finds the page and one more proves it.

Polite by construction: the crawler's own user agent, its 2-5 s delay between
providers, 0.7 s between candidate probes, and a hard stop after 5 consecutive
5xx responses.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import co_crawler as CC

HERE = Path(__file__).resolve().parent
RECORDS = HERE / "co_data" / "co_records.csv"
IDS_FILE = HERE / "co_data" / "co_refetch_ids.txt"
SIDECAR = HERE / "co_data" / "co_refetch_results.csv"
CHECKPOINT = HERE / "co_data" / "co_refetch_checkpoint.json"
LOG_FILE = HERE / "co_refetch_log.txt"

csv.field_size_limit(10_000_000)

# Everything the detail page produces. These are the ONLY columns a merge may
# write, plus the four crawl-metadata columns below: a seed column,
# provider_id and quality_rating are never touched by a refetch.
PAGE_COLS = (list(CC.empty_program_info())
             + list(CC.empty_family_facing())
             + list(CC.empty_licensing_history()))
META_COLS = ["detail_url", "match_method", "n_candidates", "errors"]
SIDECAR_COLS = ["provider_id", "refetched_at", "status"] + META_COLS + PAGE_COLS

# errors values that mark a row as needing a refetch.
REPAIR_ERRORS = {"id_mismatch", "exception", "not_found", "ambiguous_no_match",
                 "no_search_box"}
RATED = {"1", "2", "3", "4", "5"}


def log(message: str) -> None:
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(message + "\n")
    print(message, flush=True)


def build_ids(records: Path, out: Path) -> None:
    """Write the repair list: rated id_mismatch, rated no-page, then the unrated
    rows, so a run cut short has repaired the rated rows first."""
    rated_mismatch, rated_nopage, unrated = [], [], []
    with open(records, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            err = (row.get("errors") or "").strip()
            if err not in REPAIR_ERRORS:
                continue
            pid = (row.get("provider_id") or "").strip()
            if (row.get("quality_rating") or "").strip() not in RATED:
                unrated.append(pid)
            elif "id_mismatch" in err:
                rated_mismatch.append(pid)
            else:
                rated_nopage.append(pid)
    ids = rated_mismatch + rated_nopage + unrated
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + "\n")
    log(f"[ids] {len(ids)} provider(s) -> {out}\n"
        f"      rated id_mismatch {len(rated_mismatch)}, "
        f"rated no-page {len(rated_nopage)}, unrated {len(unrated)}")


def read_ids(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run: python co_refetch.py --build-ids")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_seed_rows(records: Path, wanted: set) -> dict:
    """The requested rows (name, zip, city, ...) keyed by provider_id. Streamed,
    since co_records.csv is large."""
    out = {}
    with open(records, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("provider_id") or "").strip()
            if pid in wanted:
                out[pid] = row
    return out


def sidecar_done(path: Path) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r.get("provider_id") or "").strip() for r in csv.DictReader(fh)}


def append_sidecar(path: Path, record: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SIDECAR_COLS,
                                extrasaction="ignore", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({c: ("" if record.get(c) is None else record.get(c))
                         for c in SIDECAR_COLS})


def write_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, path)


def refetch(ids, seeds, delay_range, max_probe, sidecar, checkpoint,
            max_5xx=5) -> dict:
    counts = {"verified": 0, "not_found": 0, "ambiguous_no_match": 0,
              "exception": 0, "missing_seed": 0}
    consecutive_5xx = 0
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for index, pid in enumerate(ids, start=1):
        rec = seeds.get(pid)
        if rec is None:
            log(f"[{index}/{len(ids)}] {pid}: not in {RECORDS.name} -- skipped")
            counts["missing_seed"] += 1
            continue

        record = {c: "" for c in SIDECAR_COLS}
        record["provider_id"] = pid
        record["refetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            soup, url, method, n_candidates = CC.licence_lookup(
                rec, max_probe=max_probe,
                exclude_id=_previous_card_id(rec))
        except Exception as exc:
            log(f"[{index}/{len(ids)}] {pid}: {type(exc).__name__}: {exc}")
            soup, url, method, n_candidates = None, None, "exception", 0
        # licence_lookup already retried a 5xx twice with a backoff (the
        # session's Retry adapter) before giving up, so 'exception' here means
        # the site is refusing or unreachable, not that this provider is odd.
        # Several in a row is the portal telling us to stop.
        consecutive_5xx = consecutive_5xx + 1 if method == "exception" else 0

        if soup is not None:
            record.update(CC.crawl_program_info(soup))
            record.update(CC.crawl_family_facing(soup))
            record.update(CC.crawl_licensing_history(soup))
            record["detail_url"] = url
            record["match_method"] = method
            record["n_candidates"] = n_candidates
            record["errors"] = ""
            record["status"] = "verified"
            counts["verified"] += 1
            log(f"[{index}/{len(ids)}] {pid}: OK via {method} "
                f"(licensed_to_serve={record.get('licensed_to_serve')!r})")
        else:
            # Not verified: the page fields stay blank on purpose, so a row
            # never keeps a wrong page's values without an id_mismatch tag.
            record["match_method"] = method
            record["n_candidates"] = n_candidates
            record["errors"] = method
            record["status"] = method
            counts[method] = counts.get(method, 0) + 1
            log(f"[{index}/{len(ids)}] {pid}: {method} (n_candidates={n_candidates})")

        append_sidecar(sidecar, record)
        write_checkpoint(checkpoint, {
            "started_at": started,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ids_file": str(IDS_FILE),
            "last_provider_id": pid,
            "position_in_this_run": index,
            "ids_in_this_run": len(ids),
            "counts": counts,
            "sidecar": str(sidecar),
        })

        if consecutive_5xx >= max_5xx:
            log(f"STOPPING: {consecutive_5xx} consecutive transport failures. "
                f"Re-run with --resume once the site recovers.")
            break
        if index < len(ids):
            time.sleep(random.uniform(*delay_range))
    return counts


def _previous_card_id(rec) -> "str | None":
    """The Salesforce id of the page a previous crawl already proved wrong, so
    rank_candidates can put it last."""
    url = str(rec.get("detail_url") or "")
    if "id=" not in url:
        return None
    return url.split("id=", 1)[1].split("&", 1)[0] or None


def merge(records: Path, sidecar: Path, backup: bool = True,
          dry_run: bool = False) -> None:
    """Patch the refetched rows into co_records.csv, in place and positionally.

    Row count, row order, provider_id, every seed column and quality_rating are
    untouched; only the page-derived columns and the four crawl-metadata
    columns move, and only on the rows named in the sidecar. Written to a temp
    file and renamed, so an interrupted merge cannot truncate a 41 MB file.

    dry_run writes NOTHING -- not the records file, not the .bak, not even the
    temp file (the rebuilt rows go to os.devnull). It reports the join rate,
    the sidecar's status mix and the cells that would change, per column.
    """
    if not sidecar.exists():
        raise SystemExit(f"{sidecar} not found -- nothing to merge")
    with open(sidecar, newline="", encoding="utf-8") as fh:
        patches = {(r["provider_id"] or "").strip(): r for r in csv.DictReader(fh)}
    tag = "[merge:dry-run]" if dry_run else "[merge]"
    log(f"{tag} {len(patches)} sidecar row(s) from {sidecar.name}")

    tmp = records.with_suffix(".csv.merge-tmp")
    seen, changed, cells = set(), 0, 0
    per_col, statuses = {}, {}
    with open(records, newline="", encoding="utf-8") as fh, \
            open(os.devnull if dry_run else tmp, "w", newline="",
                 encoding="utf-8") as out:
        reader = csv.reader(fh)
        header = next(reader)
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        pos = {c: i for i, c in enumerate(header)}
        writable = [c for c in PAGE_COLS + META_COLS if c in pos]
        n_rows = 0
        for row in reader:
            n_rows += 1
            pid = row[pos["provider_id"]].strip()
            patch = patches.get(pid)
            if patch is not None and pid not in seen:
                seen.add(pid)
                status = (patch.get("status") or "?").strip() or "?"
                statuses[status] = statuses.get(status, 0) + 1
                before = list(row)
                for col in writable:
                    row[pos[col]] = patch.get(col) or ""
                if row != before:
                    changed += 1
                    cells += sum(1 for a, b in zip(before, row) if a != b)
                    for col in writable:
                        if before[pos[col]] != row[pos[col]]:
                            per_col[col] = per_col.get(col, 0) + 1
            writer.writerow(row)

    missing = sorted(set(patches) - seen)
    rate = 100.0 * len(seen) / len(patches) if patches else 0.0
    log(f"{tag} join {len(seen)}/{len(patches)} sidecar id(s) matched a row in "
        f"{records.name} ({rate:.1f}%), {n_rows} data rows scanned")
    log(f"{tag} sidecar status mix: "
        + (", ".join(f"{k} {v}" for k, v in sorted(statuses.items())) or "none"))
    if missing:
        if dry_run:
            log(f"{tag} WOULD ABORT: {len(missing)} sidecar id(s) are not in "
                f"{records.name}: {missing[:5]}")
            return
        os.remove(tmp)
        raise SystemExit(f"[merge] ABORTED: {len(missing)} sidecar id(s) are not "
                         f"in {records.name}: {missing[:5]}")
    if dry_run:
        log(f"{tag} would patch {changed} row(s), {cells} cell(s); by column: "
            + (", ".join(f"{c} {n}" for c, n in sorted(per_col.items())) or "none"))
        log(f"{tag} nothing written -- {records.name}, its .bak and the temp "
            f"file are all untouched. Re-run without --dry-run to apply.")
        return
    if backup:
        backup_path = records.with_suffix(".csv.bak")
        if not backup_path.exists():
            with open(records, "rb") as src, open(backup_path, "wb") as dst:
                for block in iter(lambda: src.read(1 << 20), b""):
                    dst.write(block)
            log(f"[merge] backup -> {backup_path.name}")
    os.replace(tmp, records)
    log(f"[merge] {n_rows} rows written, {changed} row(s) patched, "
        f"{cells} cell(s) changed, row order and provider_id untouched")
    log("[merge] co_records_anonymized.csv is NOT updated by this step")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument("--ids", type=Path, default=IDS_FILE)
    ap.add_argument("--sidecar", type=Path, default=SIDECAR)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--build-ids", action="store_true",
                    help="write the repair id list and exit")
    ap.add_argument("--merge", action="store_true",
                    help="patch the sidecar into --records and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --merge: report the join rate and the cells "
                         "that would change, and write nothing")
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch at most this many providers this run")
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already in the sidecar (required once a "
                         "sidecar exists)")
    ap.add_argument("--delay-min", type=float, default=2)
    ap.add_argument("--delay-max", type=float, default=5)
    ap.add_argument("--max-probe", type=int, default=5,
                    help="candidate detail pages to probe per provider")
    args = ap.parse_args()

    if args.build_ids:
        build_ids(args.records, args.ids)
        return
    if args.merge:
        merge(args.records, args.sidecar, dry_run=args.dry_run)
        return

    ids = read_ids(args.ids)
    done = sidecar_done(args.sidecar)
    if done and not args.resume:
        raise SystemExit(
            f"{args.sidecar} already holds {len(done)} row(s).\n"
            f"  Re-run with --resume to continue, or move the file aside to "
            f"start over.")
    todo = [pid for pid in ids if pid not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    if not todo:
        log("nothing to do: every id in the list is already in the sidecar")
        return

    seeds = load_seed_rows(args.records, set(todo))
    log(f"[refetch] {len(todo)} provider(s) this run "
        f"({len(done)} already done, {len(ids)} in the list)")
    counts = refetch(todo, seeds, (args.delay_min, args.delay_max),
                     args.max_probe, args.sidecar, args.checkpoint)
    log(f"[refetch] done: {counts}")
    log(f"[refetch] sidecar -> {args.sidecar}\n"
        f"          when the list is finished: python co_refetch.py --merge")


if __name__ == "__main__":
    main()
