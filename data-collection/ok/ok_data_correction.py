"""ok_data_correction.py -- targeted repairs to ok_data/ok_records.csv.

Reads and writes the crawler's own output, using the crawler's own parser.
Three independent, opt-in modes; none of them rewrites a row it was not asked
to touch.

    --backfill-crawled-at   offline, ~2 s.   Add the crawled_at provenance
                            column to a records file written before the crawler
                            stamped one.

    --refetch-failed        NETWORK, ~2 min. Re-request the providers whose row
                            records a failure, and write a SIDECAR. Never edits
                            ok_records.csv. Resumable.

    --merge-refetch         offline, ~5 s.   Apply that sidecar to
                            ok_records.csv in place, positionally: same rows,
                            same order, same count.

ok_crawler.py's append_row() appends, so re-running the crawler against an
existing records file would add duplicate rows. Row i of the records file has
to stay row i, because the files derived from it are joined to it by position.
So a repair patches in place, here, instead.

    # 1. provenance first, so a mixed as-of date is recorded rather than implied
    python ok_data_correction.py --backfill-crawled-at

    # 2. the network pass: writes only the sidecar, safe to run at any time
    python ok_data_correction.py --refetch-failed
    python ok_data_correction.py --refetch-failed --limit 5      # smoke test
    python ok_data_correction.py --refetch-failed --resume       # continue

    # 3. the in-place patch
    python ok_data_correction.py --merge-refetch --dry-run
    python ok_data_correction.py --merge-refetch

Every read and write goes through the csv module, not pandas: the pages
return the literal string 'N/A' for subsidy_contract_number, which a pandas
round-trip with default NA handling would blank, along with every other
NA-looking cell in the file.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import ok_crawler

HERE = Path(__file__).resolve().parent
RECORDS = HERE / "ok_data" / "ok_records.csv"
CHECKPOINT = HERE / "ok_data" / "ok_refetch_checkpoint.json"
REFETCH_LOG = HERE / "ok_data" / "ok_refetch_log.txt"
MERGED_ROWS = HERE / "ok_data" / "ok_refetch_merged_rows.json"
SIDECAR_DIR = HERE.parent / "data-private"

GRAIN_COL = "provider_id"
ERRORS_COL = "errors"
CRAWLED_AT_COL = "crawled_at"

# The date the original crawl ran. Date only, on purpose: the file is in
# licence-number order, so a per-row time would reveal the collection order.
ORIGINAL_CRAWL_DATE = "2026-07-08"

# Rows worth re-requesting: the page was never successfully retrieved.
RETRY_ERRORS = ("not_found", "fetch_error", "exception")

# Sidecar bookkeeping columns, ahead of the full crawler column set.
SIDECAR_META = ["row_index", "fetched_at", "http_outcome", "merge_action"]

csv.field_size_limit(10 ** 9)


def read_header(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def read_rows(path: Path):
    """Yield (index, dict) for each data row, index counted from 0."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for index, row in enumerate(reader):
            yield index, row


def write_rows(path: Path, header: list[str], rows) -> int:
    """Write via a temp file and os.replace, so a crash never truncates."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(c, "") or "" for c in header])
            count += 1
    os.replace(tmp, path)
    return count


def log(message: str) -> None:
    REFETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(REFETCH_LOG, "a", encoding="utf-8") as fh:
        fh.write(message + "\n")
    print(message, flush=True)


def backfill_crawled_at(records: Path, value: str, dry_run: bool) -> int:
    header = read_header(records)
    if GRAIN_COL not in header:
        print(f"{records.name}: no {GRAIN_COL} column", file=sys.stderr)
        return 1

    if CRAWLED_AT_COL in header:
        blank = sum(1 for _, row in read_rows(records)
                    if not (row.get(CRAWLED_AT_COL) or "").strip())
        if not blank:
            print(f"{records.name}: {CRAWLED_AT_COL} already present and "
                  f"populated on every row -- nothing to do")
            return 0
        new_header = header
        print(f"{records.name}: {CRAWLED_AT_COL} present, {blank} blank row(s) "
              f"-> '{value}'")
    else:
        # Immediately before `errors`, matching ok_crawler.all_columns().
        new_header = list(header)
        at = new_header.index(ERRORS_COL) if ERRORS_COL in new_header else len(new_header)
        new_header.insert(at, CRAWLED_AT_COL)
        print(f"{records.name}: adding {CRAWLED_AT_COL} at position {at} "
              f"-> '{value}' on every row ({len(header)} -> {len(new_header)} columns)")

    if dry_run:
        print("dry run: nothing written")
        return 0

    def patched():
        for _, row in read_rows(records):
            if not (row.get(CRAWLED_AT_COL) or "").strip():
                row[CRAWLED_AT_COL] = value
            yield row

    rows = write_rows(records, new_header, patched())
    print(f"  {rows} rows x {len(new_header)} columns written in place")
    return 0


# Fields that can only have come from the provider's own page. Not
# facility_type: crawl_basic() infers it from the licence prefix, so it is
# populated even on a contentless "Loading Provider Profile" shell.
PAGE_ONLY_FIELDS = ("id_on_page", "provider_name", "qr_rating_raw")


def classify(row: dict) -> tuple[str, str]:
    """(http_outcome, merge_action) for one freshly crawled row.

    parsed       the page carried real provider content      -> replace the row
    not_found    HTTP 404, authoritative                     -> errors only
    empty_page   HTTP 200, but no provider content on it     -> errors only
    fetch_error  retries spent without a response            -> errors only,
                 and crawled_at is left alone, because nothing was learned

    Only 'parsed' rows are merged; the others change nothing but the error
    token, so a contentless page never overwrites a record.
    """
    errors = (row.get(ERRORS_COL) or "").strip()
    if errors == "not_found":
        return "not_found", "errors_only"
    if errors == "fetch_error":
        return "fetch_error", "errors_only"
    if not any((row.get(c) or "").strip() for c in PAGE_ONLY_FIELDS):
        row[ERRORS_COL] = "empty_page"
        return "empty_page", "errors_only"
    return "parsed", "replace"


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def refetch(records: Path, sidecar: Path, limit, resume: bool,
            delay_range, attempts: int, backoff: float,
            max_consecutive_failures: int) -> int:
    if not records.exists():
        print(f"missing {records}", file=sys.stderr)
        return 1

    header = read_header(records)
    targets = []
    for index, row in read_rows(records):
        if (row.get(ERRORS_COL) or "").strip() in RETRY_ERRORS:
            targets.append((index, row[GRAIN_COL]))
    if not targets:
        print(f"{records.name}: no row carries {RETRY_ERRORS} -- nothing to refetch")
        return 0

    done = set()
    if resume and sidecar.exists():
        done = {r[GRAIN_COL] for _, r in read_rows(sidecar)}
        print(f"resuming: {len(done)} provider(s) already in {sidecar.name}")
    elif sidecar.exists():
        print(f"{sidecar} already exists. Pass --resume to continue it, or "
              f"--sidecar to write somewhere else.", file=sys.stderr)
        return 1

    pending = [t for t in targets if t[1] not in done]
    if limit is not None:            # --limit 0 means "request nothing"
        pending = pending[:limit]
    if not pending:
        print(f"nothing to request: {len(targets)} eligible, {len(done)} already "
              f"in {sidecar.name}, limit {limit}")
        return 0

    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar_header = SIDECAR_META + [c for c in ok_crawler.all_columns()]
    fresh = not sidecar.exists()

    # The refetch gets its own log; --resume appends to the one it continues.
    ok_crawler.LOG_FILE = str(REFETCH_LOG)
    REFETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    if fresh:
        REFETCH_LOG.write_text("")
    log(f"# ok_data_correction.py --refetch-failed")
    log(f"# started_utc: {datetime.now(timezone.utc).strftime(ok_crawler.ISO_FMT)}")
    log(f"# records: {records}  ({len(targets)} row(s) eligible)")
    log(f"# sidecar: {sidecar}  ({len(pending)} to request this run)")
    log(f"# delay_range_s: {delay_range[0]}-{delay_range[1]}  attempts: "
        f"{attempts}  retry_backoff: {backoff}")

    session = requests.Session()
    counts = {"parsed": 0, "not_found": 0, "empty_page": 0, "fetch_error": 0}
    consecutive = 0

    with open(sidecar, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        if fresh:
            writer.writerow(sidecar_header)
            fh.flush()
        for position, (index, pid) in enumerate(pending):
            fetched_at = datetime.now(timezone.utc).strftime(ok_crawler.ISO_FMT)
            row, star = ok_crawler.crawl_one(pid, session, attempts=attempts,
                                             backoff=backoff)
            outcome, action = classify(row)
            counts[outcome] += 1
            row["row_index"] = index
            row["fetched_at"] = fetched_at
            row["http_outcome"] = outcome
            row["merge_action"] = action
            writer.writerow([str(row.get(c, "") or "") for c in sidecar_header])
            fh.flush()
            os.fsync(fh.fileno())
            log(f"[{position + 1}/{len(pending)}] row {index} {pid}: {outcome}"
                + (f" ({star} star)" if star else ""))

            CHECKPOINT.write_text(json.dumps({
                "sidecar": str(sidecar),
                "records": str(records),
                "eligible": len(targets),
                "fetched": len(done) + position + 1,
                "counts": counts,
                "updated_utc": datetime.now(timezone.utc).strftime(ok_crawler.ISO_FMT),
            }, indent=1) + "\n")

            if outcome == "fetch_error":
                consecutive += 1
                if consecutive >= max_consecutive_failures:
                    log(f"! {consecutive} consecutive transport failures -- "
                        f"stopping so the portal is left alone. Re-run with "
                        f"--resume later; {len(pending) - position - 1} "
                        f"provider(s) still pending.")
                    break
            else:
                consecutive = 0

            if position < len(pending) - 1:
                time.sleep(random.uniform(*delay_range))

    log(f"sidecar: {sidecar}")
    log(f"  parsed {counts['parsed']}, not_found {counts['not_found']}, "
        f"empty_page {counts['empty_page']}, fetch_error {counts['fetch_error']}")
    log(f"  next: python ok_data_correction.py --merge-refetch --dry-run")
    return 0


def merge(records: Path, sidecar: Path, dry_run: bool, no_backup: bool) -> int:
    for path in (records, sidecar):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    header = read_header(records)
    if CRAWLED_AT_COL not in header:
        print(f"{records.name}: no {CRAWLED_AT_COL} column. Run "
              f"--backfill-crawled-at first, so the rows this merge re-dates "
              f"are distinguishable from the ones it does not.", file=sys.stderr)
        return 1

    patch = {}
    for _, row in read_rows(sidecar):
        index = int(row["row_index"])
        if index in patch:
            print(f"{sidecar.name}: row_index {index} appears twice -- "
                  f"de-duplicate the sidecar first", file=sys.stderr)
            return 1
        patch[index] = row

    def apply(row: dict, new: dict) -> str:
        """Patch one row in place; return the action taken."""
        action = new["merge_action"]
        if action == "replace":
            for column in header:
                if column == GRAIN_COL:
                    continue
                if column == CRAWLED_AT_COL:
                    row[column] = new["fetched_at"]
                else:
                    row[column] = new.get(column, "") or ""
        elif action == "errors_only":
            row[ERRORS_COL] = new.get(ERRORS_COL, "") or ""
            if new["http_outcome"] != "fetch_error":
                # A 404 or a contentless 200 IS an observation about the
                # provider; a request that never landed is not.
                row[CRAWLED_AT_COL] = new["fetched_at"]
        return action

    # Pass 1: validate and count. The records file is streamed twice rather
    # than held in memory.
    replaced, errors_only, unchanged, problems, total = 0, 0, 0, [], 0
    summary = {"replace": [], "errors_only": []}
    ratings = {}
    for index, row in read_rows(records):
        total += 1
        new = patch.get(index)
        if new is None:
            continue
        if row[GRAIN_COL] != new[GRAIN_COL]:
            problems.append(f"row {index}: records has {GRAIN_COL} "
                            f"{row[GRAIN_COL]!r}, sidecar has {new[GRAIN_COL]!r}")
            continue
        action = apply(row, new)
        if action == "replace":
            replaced += 1
            summary["replace"].append(index)
            rating = (row.get("qr_rating_raw") or "").strip()
            ratings[rating] = ratings.get(rating, 0) + 1
        elif action == "errors_only":
            errors_only += 1
            summary["errors_only"].append(index)
        else:
            unchanged += 1

    unseen = sorted(set(patch) - set(summary["replace"]) - set(summary["errors_only"]))
    if unseen and not problems:
        problems.append(f"sidecar row_index {unseen[:5]} not present in "
                        f"{records.name} ({total} rows)")
    if problems:
        print("ABORTED, nothing written:", file=sys.stderr)
        for problem in problems:
            print("  " + problem, file=sys.stderr)
        return 1

    print(f"{records.name}: {total} rows, {len(patch)} in the sidecar")
    print(f"  replace      {replaced:4d} row(s) -- every column but {GRAIN_COL}")
    print(f"  errors_only  {errors_only:4d} row(s)")
    if unchanged:
        print(f"  skipped      {unchanged:4d} row(s)")
    if ratings:
        print("  recovered qr_rating_raw: "
              + ", ".join(f"{k or '(blank)'}*{v}" for k, v in sorted(ratings.items())))
    if dry_run:
        print("dry run: nothing written")
        return 0

    if not no_backup:
        backup = records.with_suffix(".csv.bak")
        shutil.copy2(records, backup)
        print(f"  backup -> {backup.name}")

    def patched():
        for index, row in read_rows(records):
            new = patch.get(index)
            if new is not None and row[GRAIN_COL] == new[GRAIN_COL]:
                apply(row, new)
            yield row

    written = write_rows(records, header, patched())
    if written != total:
        print(f"ABORTED: wrote {written} of {total} rows -- restore "
              f"{records.name} from the .bak", file=sys.stderr)
        return 1
    MERGED_ROWS.write_text(json.dumps({
        "records": str(records),
        "sidecar": str(sidecar),
        "merged_utc": datetime.now(timezone.utc).strftime(ok_crawler.ISO_FMT),
        "replaced_rows": summary["replace"],
        "errors_only_rows": summary["errors_only"],
    }, indent=1) + "\n")
    print(f"  {written} rows x {len(header)} columns written in place; "
          f"{GRAIN_COL} untouched on every row")
    print(f"  merged row indices -> {MERGED_ROWS.name}")
    print(f"  re-run the downstream steps for those rows")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill-crawled-at", action="store_true")
    mode.add_argument("--refetch-failed", action="store_true")
    mode.add_argument("--merge-refetch", action="store_true")

    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument("--sidecar", type=Path, default=None,
                    help="default: data-private/ok_refetch_<today>.csv")
    ap.add_argument("--crawl-date", default=ORIGINAL_CRAWL_DATE,
                    help=f"value written by --backfill-crawled-at "
                         f"(default {ORIGINAL_CRAWL_DATE})")
    ap.add_argument("--limit", type=int, default=None,
                    help="request at most N providers this run")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing sidecar, skipping providers "
                         "already in it")
    ap.add_argument("--delay-min", type=float, default=1)
    ap.add_argument("--delay-max", type=float, default=3)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--retry-backoff", type=float, default=2.0)
    ap.add_argument("--max-consecutive-failures", type=int, default=3,
                    help="stop the run after this many providers in a row fail "
                         "every attempt")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    # The default sidecar name is date-stamped, so --resume and --merge-refetch
    # fall back to the path the checkpoint recorded.
    sidecar = args.sidecar
    if sidecar is None and (args.merge_refetch or args.resume):
        recorded = load_checkpoint(CHECKPOINT).get("sidecar")
        if recorded:
            sidecar = Path(recorded)
            print(f"using the sidecar named in {CHECKPOINT.name}: {sidecar}")
    if sidecar is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sidecar = SIDECAR_DIR / f"ok_refetch_{stamp}.csv"

    if args.backfill_crawled_at:
        return backfill_crawled_at(args.records, args.crawl_date, args.dry_run)
    if args.refetch_failed:
        return refetch(args.records, sidecar, args.limit, args.resume,
                       (args.delay_min, args.delay_max), args.attempts,
                       args.retry_backoff, args.max_consecutive_failures)
    return merge(args.records, sidecar, args.dry_run, args.no_backup)


if __name__ == "__main__":
    sys.exit(main())
