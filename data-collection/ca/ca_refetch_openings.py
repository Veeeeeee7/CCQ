"""
ca_refetch_openings.py -- re-observe the openings block on the released
California provider pages.

Re-opens the pages the crawl already resolved, parses them with the parsers in
ca_crawler_errors.py, and writes a SIDE FILE. It never touches
ca_data/ca_records.csv: merging is a separate step
(`python ca_data_correction.py --apply-refetch`), so a crash mid-run cannot
leave a half-written records file.

Scope: the distinct provider pages behind the released rows (error-free,
QCC 1-5), one request each. Every page in scope is refetched, so the released
openings columns share one observation date.

Records per page: openings_last_updated, openings_status, openings_capacity,
license_number_first, license_numbers_all, tags, tags_count.
license_numbers_all feeds the linkage audit: it tells a multi-licence profile
apart from a page that belongs to a different provider.

    python ca_refetch_openings.py --dry-run      # scope, no requests
    python ca_refetch_openings.py --limit 20     # pilot
    python ca_refetch_openings.py --resume       # the rest; repeat if interrupted
    python ca_data_correction.py --apply-refetch # merge, separate step

The host runs Wordfence and rate-limits, so politeness matches
ca_crawler_errors.py: a fresh context per page, 8-15 s between pages, three
navigation attempts with backoff, a 600 s cooldown plus relaunch after five
consecutive failures, and a stop after three cooldowns with no page recovered.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ca_crawler_errors as cce  # noqa: E402  (path set above)
from ca_crawler_errors import (  # noqa: E402
    NavBlocked, _all_text, crawl_openings, crawl_tags,
    find_provider_url, goto_with_retry, licences_on_page, looks_blocked,
)

csv.field_size_limit(10_000_000)

DEFAULT_RECORDS = HERE / "ca_data" / "ca_records.csv"
DEFAULT_OUT = HERE / "ca_data" / "ca_openings_refetch.csv"
DEFAULT_CHECKPOINT = HERE / "ca_data" / "ca_refetch_checkpoint.json"
LOG_FILE = HERE / "ca_refetch_log.txt"

BASE_URL = "https://mychildcareplan.org"
DETAILS_URL = BASE_URL + "/provider-details/"

SIDECAR_COLS = [
    "facility_number", "fetch_status", "refetched_at", "page_url",
    "url_source", "openings_last_updated", "openings_status",
    "openings_capacity", "license_number_first", "license_numbers_all",
    "tags", "tags_count",
]

VALID_TARGETS = {1.0, 2.0, 3.0, 4.0, 5.0}
CAPACITY_RE = re.compile(r"Capacity\s*:\s*[0-9]+", re.I)


def log(message, file=LOG_FILE):
    with open(file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message, flush=True)


# The imported helpers log through ca_crawler_errors.log; route that here so
# one run leaves one log.
cce.log = log


def is_released(qcc):
    try:
        return float(qcc) in VALID_TARGETS
    except (TypeError, ValueError):
        return False


def provider_uid(url):
    if not url:
        return None
    return (parse_qs(urlparse(url).query).get("provider_uid") or [None])[0]


def select_targets(records_path):
    """One row per released provider PAGE, in file order.

    Released rows have empty errors and basics_qcc_score 1-5. They are
    deduplicated on the provider_uid of provider_url, so a profile seeded by
    two licence numbers is visited once.
    """
    with open(records_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader
                if (r.get("errors") or "").strip() == ""
                and is_released(r.get("basics_qcc_score"))]
    targets, seen = [], set()
    for r in rows:
        uid = provider_uid(r.get("provider_url"))
        if uid in seen:
            continue
        seen.add(uid)
        targets.append({"facility_number": (r.get("facility_number") or "").strip(),
                        "provider_url": (r.get("provider_url") or "").strip()})
    return len(rows), targets


def url_candidates(saved_url):
    """The saved URL, then the same page without the session parameters.

    provider_url carries session_id and page_number from the crawl; a stale
    session id is the most likely reason a saved URL stops resolving, and the
    agency_id + provider_id + provider_uid triple is what actually identifies
    the page.
    """
    out = []
    if saved_url:
        out.append(("saved", saved_url))
        q = parse_qs(urlparse(saved_url).query)
        stripped = {k: q[k][0] for k in ("agency_id", "provider_id", "provider_uid")
                    if q.get(k)}
        if stripped:
            candidate = DETAILS_URL + "?" + urlencode(stripped)
            if candidate != saved_url:
                out.append(("stripped", candidate))
    return out


def page_is_provider(page):
    """A provider-details page, not a 404, a search page or a block screen."""
    try:
        if page.locator("div.provider-openings").count() > 0:
            return True
        if page.locator("ul.provider-attributes__list").count() > 0:
            return True
        body = _all_text(page.locator("body"))
        return "License Number" in body and "The Basics" in body
    except Exception:
        return False


def crawl_license_all(page):
    """Every licence number printed on the page, in order, deduplicated."""
    return licences_on_page(_all_text(page.locator("body")))


def scrape(page):
    row = {}
    row.update(crawl_openings(page))
    row.update(crawl_tags(page))
    licences = crawl_license_all(page)
    row["license_number_first"] = licences[0] if licences else None
    row["license_numbers_all"] = "|".join(licences) if licences else None
    status = row.get("openings_status") or ""
    if CAPACITY_RE.search(status):
        # crawl_openings() sums every group's capacity and strips the tokens;
        # a leftover would double-count when --fix-openings-capacity is re-run.
        raise AssertionError(f"openings_status still holds a Capacity token: "
                             f"{status[:120]!r}")
    return row


def load_done(out_path):
    if not out_path.exists() or out_path.stat().st_size == 0:
        return {}
    with open(out_path, newline="", encoding="utf-8") as f:
        return {(r.get("facility_number") or "").strip(): r
                for r in csv.DictReader(f)}


def append_sidecar(out_path, row):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SIDECAR_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        clean = {}
        for c in SIDECAR_COLS:
            v = row.get(c)
            if isinstance(v, str):
                v = re.sub(r"\s+", " ", v).strip()
            clean[c] = "" if v is None else v
        w.writerow(clean)


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def refetch(targets, out_path, checkpoint_path, headless=True,
            executable_path=None, delay_range=(8, 15), nav_timeout_ms=45000,
            nav_attempts=3, max_consecutive_failures=5, cooldown_seconds=600,
            max_cooldowns=3, relaunch_every=200):
    done = load_done(out_path)
    todo = [t for t in targets if t["facility_number"] not in done]
    log(f"refetch: {len(targets)} page(s) in scope, {len(done)} already in "
        f"{out_path.name}, {len(todo)} to go")
    if not todo:
        return 0

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = {"started_at": started, "scope": len(targets),
             "already_done": len(done), "attempted": 0, "ok": 0,
             "not_found": 0, "blocked": 0, "error": 0, "cooldowns": 0,
             "last_facility_number": None, "updated_at": started}

    with sync_playwright() as p:
        launch_kwargs = {"headless": headless, "args": ["--incognito"]}
        if executable_path:
            launch_kwargs["executable_path"] = executable_path
        browser = p.chromium.launch(**launch_kwargs)
        ua = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")
        consecutive_failures = 0
        cooldowns_without_progress = 0
        try:
            for i, target in enumerate(todo):
                facility_number = target["facility_number"]
                row = {"facility_number": facility_number,
                       "refetched_at": datetime.now(timezone.utc)
                                       .isoformat(timespec="seconds")}
                failed = False
                context = page = None
                try:
                    # fresh context per page: nothing carries over
                    context = browser.new_context(
                        viewport={"width": 1400, "height": 900}, user_agent=ua)
                    page = context.new_page()
                    landed = None
                    for source, url in url_candidates(target["provider_url"]):
                        if not goto_with_retry(page, url, timeout=nav_timeout_ms,
                                               attempts=nav_attempts):
                            raise NavBlocked("navigation timed out")
                        if looks_blocked(page):
                            raise NavBlocked("Wordfence limited access")
                        try:
                            page.wait_for_load_state("networkidle", timeout=15000)
                        except Exception:
                            pass
                        if page_is_provider(page):
                            landed = source
                            break
                    if landed is None:
                        # Both saved URLs failed: fall back to the site search
                        # (two more requests, so last resort).
                        found = find_provider_url(
                            page, facility_number, nav_timeout_ms=nav_timeout_ms,
                            nav_attempts=nav_attempts)
                        if found and goto_with_retry(
                                page, found, timeout=nav_timeout_ms,
                                attempts=nav_attempts):
                            if looks_blocked(page):
                                raise NavBlocked("Wordfence limited access")
                            time.sleep(1)
                            if page_is_provider(page):
                                landed = "search"
                    if landed is None:
                        row["fetch_status"] = "not_found"
                        row["page_url"] = page.url if page else ""
                        log(f"[{i + 1}/{len(todo)}] NOT FOUND: {facility_number}")
                        state["not_found"] += 1
                    else:
                        time.sleep(1)
                        row.update(scrape(page))
                        row["fetch_status"] = "ok"
                        row["url_source"] = landed
                        row["page_url"] = page.url
                        state["ok"] += 1
                        log(f"[{i + 1}/{len(todo)}] OK: {facility_number} "
                            f"cap={row.get('openings_capacity')} "
                            f"upd={row.get('openings_last_updated')} "
                            f"lic={row.get('license_numbers_all')} "
                            f"tags={row.get('tags_count')} ({landed})")
                except NavBlocked as e:
                    failed = True
                    row["fetch_status"] = "blocked"
                    log(f"[{i + 1}/{len(todo)}] BLOCKED: {facility_number} ({e})")
                    state["blocked"] += 1
                except Exception:
                    failed = True
                    row["fetch_status"] = "error"
                    log(f"[{i + 1}/{len(todo)}] ERROR: {facility_number}")
                    log(traceback.format_exc())
                    state["error"] += 1
                finally:
                    for closeable in (page, context):
                        try:
                            if closeable is not None:
                                closeable.close()
                        except Exception:
                            pass
                    append_sidecar(out_path, row)
                    state["attempted"] += 1
                    state["last_facility_number"] = facility_number
                    state["updated_at"] = (datetime.now(timezone.utc)
                                           .isoformat(timespec="seconds"))
                    save_checkpoint(checkpoint_path, state)

                if not failed:
                    cooldowns_without_progress = 0
                consecutive_failures = consecutive_failures + 1 if failed else 0
                is_last = i >= len(todo) - 1
                if not is_last:
                    delay = random.uniform(*delay_range)
                    time.sleep(delay)
                if not is_last and consecutive_failures >= max_consecutive_failures:
                    cooldowns_without_progress += 1
                    state["cooldowns"] += 1
                    if cooldowns_without_progress >= max_cooldowns:
                        log(f"  !! {cooldowns_without_progress} cooldowns with "
                            f"no page recovered -- stopping. Re-run with "
                            f"--resume later; nothing is lost.")
                        break
                    log(f"  !! {consecutive_failures} failures in a row -- "
                        f"likely rate-limited. Cooling down {cooldown_seconds}s "
                        f"and relaunching the browser.")
                    try:
                        browser.close()
                    except Exception:
                        pass
                    time.sleep(cooldown_seconds)
                    browser = p.chromium.launch(**launch_kwargs)
                    consecutive_failures = 0
                elif (not is_last and relaunch_every
                      and state["attempted"] % relaunch_every == 0):
                    log(f"  ...periodic browser relaunch after "
                        f"{state['attempted']} page(s)")
                    try:
                        browser.close()
                    except Exception:
                        pass
                    browser = p.chromium.launch(**launch_kwargs)
        finally:
            try:
                browser.close()
            except Exception:
                pass

    log(f"\n--- refetch summary ---")
    for k in ("attempted", "ok", "not_found", "blocked", "error", "cooldowns"):
        log(f"{k:<16} {state[k]}")
    log(f"sidecar:   {out_path}")
    log(f"checkpoint:{checkpoint_path}")
    remaining = len(todo) - state["attempted"]
    if remaining:
        log(f"{remaining} page(s) left -- re-run with --resume")
    else:
        log("scope complete. Merge with:\n"
            "    python ca_data_correction.py --apply-refetch")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="sidecar CSV, appended to as pages are fetched")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch at most this many pages this run (pilot)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing sidecar, skipping pages already in it")
    ap.add_argument("--restart", action="store_true",
                    help="delete the sidecar and checkpoint and start over")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the scope and the first few URLs, fetch nothing")
    ap.add_argument("--headful", action="store_true", help="show the browser")
    ap.add_argument("--executable-path", default=None)
    ap.add_argument("--delay-min", type=float, default=8.0)
    ap.add_argument("--delay-max", type=float, default=15.0)
    ap.add_argument("--facility", nargs="*", default=None,
                    help="fetch only these facility numbers")
    args = ap.parse_args()

    n_rows, targets = select_targets(args.records)
    if args.facility:
        wanted = set(args.facility)
        targets = [t for t in targets if t["facility_number"] in wanted]
    print(f"released rows: {n_rows}   distinct provider pages: {len(targets)}")

    if args.restart:
        for path in (args.out, args.checkpoint):
            if path.exists():
                path.unlink()
                print(f"removed {path}")

    if args.dry_run:
        for t in targets[:5]:
            print(f"  {t['facility_number']}  {url_candidates(t['provider_url'])[0][1]}")
        print(f"  ... {max(0, len(targets) - 5)} more")
        est = len(targets) * 18 / 3600
        print(f"estimated run time at ~18 s/page: {est:.1f} h")
        return 0

    if args.out.exists() and args.out.stat().st_size and not args.resume:
        raise SystemExit(
            f"{args.out} already exists. Pass --resume to continue it (pages "
            f"already in it are skipped) or --restart to throw it away.")

    if args.limit is not None:
        done = load_done(args.out)
        todo = [t for t in targets if t["facility_number"] not in done]
        targets = [t for t in targets if t["facility_number"] in done] + todo[:args.limit]

    return refetch(targets, args.out, args.checkpoint,
                   headless=not args.headful,
                   executable_path=args.executable_path,
                   delay_range=(args.delay_min, args.delay_max))


if __name__ == "__main__":
    raise SystemExit(main())
