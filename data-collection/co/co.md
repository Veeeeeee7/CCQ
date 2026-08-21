# Colorado — running the pipeline

Colorado Shines, rated 1–5. Everything below runs from inside `co/`.

## 0. What you need first

**The seed is already provided.** `co_data/co_seed.csv` ships with this repository — there is no collection step to run and no directory to re-scrape before you start.

The licensed-facility list, keyed on `provider_id`, with the name and address columns the crawler carries through.

This state also needs `../private/geo_label_map_co.json`. The cleaning step reads it to label geography consistently and **writes it back if it is missing** — so without the original file the geographic columns come out under different names than the published dataset.

## 1. Collection

If the site's markup has changed, `python co_capture.py` dumps a rendered page so selectors can be checked against real HTML. Not needed for a normal run.

```
python co_crawler.py --headless
```

Playwright. Run once without `--headless` first to confirm the portal renders. `--limit N` for a smoke test.

Writes `co_data/co_records.csv`, one row per provider. Resume-safe: re-running skips providers already in the output, so an interrupted crawl can be continued rather than restarted.

## 2. Anonymization

```
python co_anonymize.py --dry-run
python co_anonymize.py
```

`--dry-run` prints the columns it would remove and writes nothing. The real run writes `co_data/co_records_anonymized.csv` — same rows, same order, fewer columns — and a provider-id map to `../private/provider_id_map_co.csv`.

**That map is the only link back to the real providers. Keep it out of any release.**

## 3. Cleaning

```
python co_clean_raw.py
python co_clean_full.py
python co_clean_complete_raw.py
python co_clean_complete_full.py
```

Four tables in `co_data/`:

| file                                   | contents                                           |
| -------------------------------------- | -------------------------------------------------- |
| `co_records_cleaned_raw.csv`           | rated providers, text preserved (43 columns)       |
| `co_records_cleaned_full.csv`          | rated providers, numeric/boolean only (31 columns) |
| `co_records_cleaned_complete_raw.csv`  | every provider, text preserved                     |
| `co_records_cleaned_complete_full.csv` | every provider, numeric/boolean only               |

The two `complete_*` files keep providers with no valid rating; the two released files drop them. Raw and full are row-aligned, so one fold file applies to both.

## Notes

Five per-provider narrative columns are removed in phase 2, but the `has_documented_*` flags derived from them are kept — that derivation happens in `co_anonymize.py`, before the drop.
