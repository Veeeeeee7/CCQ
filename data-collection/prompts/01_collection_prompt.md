# Prompt 1 — Collection

Your job is to collect one U.S. state's child care providers into a single table `{st}_data/{st}_records.csv` from the provided seed `{st}_data/{st}_seed.csv`.:

That table is the whole deliverable. What is built from it afterwards is not your concern and is not described here.

---

## 0. Your task

You are adding one U.S. state to a multi-state dataset of per-provider child care quality ratings. The human will name the state (e.g. _"do Ohio next"_). Your job is to produce a crawler that writes **one row per provider** to `{st}_data/{st}_records.csv`, where `{st}` is the state's two-letter lowercase postal code.

**Georgia is the reference implementation, and it is the only one.** These files are attached to this prompt:

| attachment      | role                                                                                                                              |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `ga_crawler.py` | the reference crawler — structure, logging, resume, per-section `try/except`, column-union flush                                  |
| `ga_ids.json`   | the field map: `crawled_columns` holds a per-section HTML-id → column-name dict; `additional_columns` holds the seed's rename map |

Read both in full before writing any code. Adapt Georgia's structure to your state; do not invent a different architecture. Where your state's site differs from Georgia's — a JSON API instead of a portal, a static directory instead of a JavaScript app — keep Georgia's general design (config block, section extractors with matching empty-row helpers, resume set, buffered flush).

**You will not see Georgia's output.** `ga_records.csv` is not attached, so you cannot check your work against a finished table. Instead, you have the code that produces it, and nothing else. Everything you need is attached or is on the public web.

---

## 1. The goal, precisely

Every state's final output — no exceptions — must contain these two columns, and they must be the first two:

| column        | meaning                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------- |
| `provider_id` | the licensing / registration identifier used by the state (kept as a string with leading zeros preserved)        |
| `qr_rating`   | the quality rating score, on whatever scale the state uses (1–5 stars, 1–4, points, Level 1–5, letter grades, …) |

Everything else is bonus: any additional provider attribute worth collecting (capacity, ages served, violations, vacancies, languages, hours, accreditation, fees, …). Collect what is cheaply available; do not block on it.

You will rarely find these two fields under those names. Scrape them under their native names. Renaming happens later and not by you. Georgia is the exception, not the rule: its `ga_ids.json` `additional_columns` map renames them at seed time. If your state's native names differ, leave them alone and record them in your report.

**Collect generously.** Over-collecting here is cheap and under-collecting is expensive — a missed field means a re-crawl. Capture every informative field the source exposes, including ones you suspect will later be removed as private or as rating-related.

---

## 2. Deliverables

- `{st}_capture.py` _(optional)_ — a reconnaissance helper that opens a real browser and dumps the **rendered** DOM plus a screenshot, so selectors can be written from real markup. Needed for JavaScript-heavy sites.
- `{st}_ids.json` _(optional)_ — the selector/field map, when the state has enough fields that inlining them would bloat the crawler. Mirror `ga_ids.json`.
- `{st}_crawler.py` — the main scraper. **Required.**
- `{st}_data/{st}_records.csv` — the output. **Required.**
- `{st}_crawler_log.txt` — written by the crawler's `log()`.

---

## 3. Workflow — phases and checkpoints

Work through these in order. **Do not skip ahead.** The 🛑 marks are hard stops where you hand control back to the human to run code, or ask questions.

### Phase 0 — Research the state

1. If the human attaches a source workbook of state rating systems, start there: it gives the quality-rating system's name, the rating scale and number of levels, the public website, and document links. Treat it as the **source of truth** for the system's name and scale.
2. Open the website and standards links. **If a link is dead, do your own web search** for the current public search tool.
3. Determine and write down:
    - the rating system's name and the rating scale — every value `qr_rating` can take, and which of them count as _valid_.
    - Is there a public, per-provider search that shows the rating? If ratings are not published per provider (e.g. gated behind a provider login), flag this to the human immediately as the state may not be scrapeable.
    - Rating coverage: does every licensed program get a rating (near universal, e.g. Level 1 = licensed), or only voluntary participants (a subset)? This drives seed scope and how many rows to expect.
    - Delivery mechanism, easiest → hardest:
      (a) a downloadable dataset or documented JSON API → prefer this;
      (b) a server-rendered searchable directory (plain HTTP + BeautifulSoup);
      (c) a JavaScript app needing a real browser;
      (d) PDF-only or login-gated → hardest, raise with the human.
    - Any anti-bot measures (reCAPTCHA, bot-scoring, rate limits).

🛑 **Checkpoint 0 — report findings and ask.** Summarize the above back to the human. Raise the questions in §6, especially program-type scope, whether to prefer an open-data source over the finder site, and how to treat unrated providers. **Wait for answers before writing code.**

### Phase 1 — Recon / capture

Only needed for a JavaScript app or an API.

- API exists → inspect the XHR/fetch endpoints the search page calls, capture the JSON shape, and plan to hit those directly. Fastest and most robust; always prefer this over DOM scraping.
- JavaScript app → write `{st}_capture.py`, and have the human run it to dump the rendered DOM of a couple of provider pages. Write your selectors from that real markup, not from guesses. Record the detail-page URL pattern.
- Static site → a quick `requests.get` + BeautifulSoup inspection is enough; usually no capture script needed.

🛑 **Checkpoint 1** — the human runs `{st}_capture.py` (if written) and returns the captured HTML/URLs. Ask if the markup is ambiguous.

### Phase 2 — Write the crawler

Think out loud first about: the seed source, the URL/endpoint pattern, which sections to extract, and the resume key. Then write `{st}_crawler.py` to the contract in §4.

🛑 **Checkpoint 2 — the crawler is run for you.** Do not run it yourself against the live site. Ask for a small `--limit` smoke test with a visible browser first, then a fuller run. What comes back is whether it ran: exit status, row count, anything in `{st}_crawler_log.txt`, and the `errors` column's value counts. You will not see the collected data itself.

### Phase 3 — Wait for any bugs to fix

You never see the collected table, and inspecting it is not your job. If a problem with it comes back to you, it will arrive as a description rather than as data. Work from that description and from your own knowledge of the code you wrote.

---

## 4. Crawler design contract

Mirror `ga_crawler.py`. Requirements:

- **One row per provider** — or per provider-location, when the state has multiple locations per provider. In that case use a compound `{provider}-{location}` grain and say so in the module docstring.
- **Module docstring** explaining the site's quirks and the approach taken: what the seed is, which pages are visited per provider, why the tool choice, and any render or anti-bot workaround. `ga_crawler.py`'s docstring is the model: it names the portal, the two pages per provider, the reason the output cannot use a fixed header, and the resume guarantee.
- **Config block up top** — base URL, URL/endpoint templates, section id and selector constants, wait timeouts, UA string, log path, default seed path.
- **File-based logging** — a `create_log_file()` / `log()` pair that appends to `{st}_crawler_log.txt` and prints. Copy Georgia's verbatim.
- **Seed loading** — normalize IDs to strings with leading zeros preserved: cast to `str`, strip a stray trailing `.0`, and `zfill` to the known width only if the state actually zero-pads. Never let pandas coerce an ID to int. Drop null/empty IDs and dedup the seed, as `load_seed()` does.
- **Resume-safe** — read the existing output CSV's key column into a set at startup and skip completed providers (`load_completed()`). Append one row at a time, or buffer and flush with a column union the way `flush_rows()` does when the schema grows per provider. Georgia needs the union because its rates-table and compliance sections emit provider-specific columns, so a fixed header is impossible; if your state has a stable header, a plain append is fine and simpler.
- **Per-section `try/except`** — each section gets a `create_empty_*_row()` that returns the section's keys mapped to `None`. On failure, fill from the empty dict, append the section name to an `errors` list, log the exception's last line, and continue. A single broken field must never kill a row.
- **Politeness / stealth** — a randomized delay between providers (`random.uniform(*delay_range)`) and a realistic UA. For a JavaScript app behind bot-scoring: use a persistent browser context so trust accumulates across runs, do a warm-up hit to the site root, disable the automation blink feature as Georgia's launch args do, and prefer a real Chrome channel. Never wait on `networkidle` for an app that holds a websocket open, it never fires. Poll a content-settle heuristic or wait for a known post-render selector.
- **Fallback navigation** — Georgia navigates by direct URL and falls back to the search box when the portal redirects (`open_detail` / `find_url`), and `find_url` returns a result only when exactly one provider matches. Reproduce that discipline: an ambiguous match must produce a not-found row, never a guess.
- **Lightest tool that works** — plain `requests` + BeautifulSoup for static sites and JSON APIs; a real browser only when the page needs one to render.
- **CLI flags** — at minimum `--seed`, `--output`, `--limit`, `--start-index`, `--headless`, `--delay-min` / `--delay-max`. Add `--flush-every` if you buffer. Default to a **visible** browser so the smoke test is inspectable, and make `--headless` opt-in, as Georgia does.
- **Round 2 (optional)** — if per-provider PDFs (rating reports, monitoring documents) are useful, record their URLs in round 1 and gate the downloading behind a `--download-pdfs` flag, as `crawl_pdfs` is gated.

---

## 5. Style rules

- Comments explain why, not what — especially every anti-bot choice, every render-wait, and every fallback path. Georgia's comment on the direct-URL-then-search-box fallback is the register to aim for.
- IDs are strings, always.
- Small, focused functions. One extractor per page section, each with its matching empty-row helper.
- Discover schema from the data where Georgia does. Do not hardcode a value list you could enumerate at runtime.

---

## 6. Pause-and-ask protocol

Ask the human whenever uncertainty arises. Never guess on anything that affects `provider_id`, `qr_rating`, row counts, or which providers are in scope. At minimum, ask for:

- **Program-type scope** — centers only, or centers and family child care homes (and school-age / license-exempt)? Affects the seed and the dedup grain.
- **Rating coverage** — for a voluntary rating system where only some providers are rated, does the human want only rated providers, or all licensed providers (with unrated ones flowing to stage 3's `complete_*` outputs)?
- **Ambiguous identity fields** — which native column is the true licensing ID versus a facility/location number; which column is the official rating versus a component score; what the valid rating values are.
- **Territories / local systems** — some entries are territories or county-level systems rather than statewide. Confirm whether they are in scope.

Mirror this mid-thought: if while reasoning you hit a fork that changes the output, stop and ask rather than assuming.

---

## 7. What to report when you finish

1. the output path (`{st}_data/{st}_records.csv`) and its row count
2. a field inventory: every column your crawler can emit, which page section and selector it comes from, the shape its values take, and whether it is populated on every provider or only some. You wrote the extractors, so this costs you nothing and is the one thing about the table that reading it would not reveal
3. the native column holding the provider ID and the native column holding the rating, named from your own field map
4. section by section against the rendered page: the claim that every informative field visible on the source is in that inventory, and what you chose to skip and why.

Then stop.
