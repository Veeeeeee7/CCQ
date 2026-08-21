# Wisconsin — running the pipeline

YoungStar, rated 1–5. Everything below runs from inside `wi/`.

## 0. What you need first

**The seed is already provided.** `wi_data/wi_seed.csv` ships with this repository — there is no collection step to run and no directory to re-scrape before you start.

The two regulation rosters — licensed (`lcc`) and certified (`ccc`) — stacked into one file and tagged by `seed_source`. Each half keeps the two-line preamble its download shipped with; the crawler finds and promotes each half's own header row.

## 1. Collection

If the site's markup has changed, `python wi_capture.py` dumps a rendered page so selectors can be checked against real HTML. Not needed for a normal run.

```
python wi_crawler.py --headless
```

Blazor SPA. Uses a persistent browser profile so bot-scoring trust accumulates across runs — do not delete it between runs. `--channel chrome` prefers a real Chrome install.

Writes `wi_data/wi_records.csv`, one row per provider. Resume-safe: re-running skips providers already in the output, so an interrupted crawl can be continued rather than restarted.

## 2. Anonymization

```
python wi_anonymize.py --dry-run
python wi_anonymize.py
```

`--dry-run` prints the columns it would remove and writes nothing. The real run writes `wi_data/wi_records_anonymized.csv` — same rows, same order, fewer columns — and a provider-id map to `../private/provider_id_map_wi.csv`.

**That map is the only link back to the real providers. Keep it out of any release.**

## 3. Cleaning

```
python wi_clean_raw.py
python wi_clean_full.py
python wi_clean_complete_raw.py
python wi_clean_complete_full.py
```

Four tables in `wi_data/`:

| file                                   | contents                                          |
| -------------------------------------- | ------------------------------------------------- |
| `wi_records_cleaned_raw.csv`           | rated providers, text preserved (8 columns)       |
| `wi_records_cleaned_full.csv`          | rated providers, numeric/boolean only (9 columns) |
| `wi_records_cleaned_complete_raw.csv`  | every provider, text preserved                    |
| `wi_records_cleaned_complete_full.csv` | every provider, numeric/boolean only              |

The two `complete_*` files keep providers with no valid rating; the two released files drop them. Raw and full are row-aligned, so one fold file applies to both.

## Notes

The grain is a compound `{provider}-{location}`: one row per provider-location, not per provider.
