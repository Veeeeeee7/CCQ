# Maryland — running the pipeline

Maryland EXCELS, rated 1–5. Everything below runs from inside `md/`.

## 0. What you need first

**The seed is already provided.** `md_data/md_seed.csv` ships with this repository — there is no collection step to run and no directory to re-scrape before you start.

A single `county` column: the 23 counties plus Baltimore City. The search returns every provider in a county, so the county list is the whole seed.

## 1. Collection

If the site's markup has changed, `python md_capture.py` dumps a rendered page so selectors can be checked against real HTML. Not needed for a normal run.

```
python md_crawler.py
```

Plain HTTP. `--limit N` crawls only the first N counties. Resume-safe: re-running skips counties already in the output.

Writes `md_data/md_records.csv`, one row per provider. Resume-safe: re-running skips providers already in the output, so an interrupted crawl can be continued rather than restarted.

## 2. Anonymization

```
python md_anonymize.py --dry-run
python md_anonymize.py
```

`--dry-run` prints the columns it would remove and writes nothing. The real run writes `md_data/md_records_anonymized.csv` — same rows, same order, fewer columns — and a provider-id map to `../private/provider_id_map_md.csv`.

**That map is the only link back to the real providers. Keep it out of any release.**

## 3. Cleaning

```
python md_clean_raw.py
python md_clean_full.py
python md_clean_complete_raw.py
python md_clean_complete_full.py
```

Four tables in `md_data/`:

| file                                   | contents                                           |
| -------------------------------------- | -------------------------------------------------- |
| `md_records_cleaned_raw.csv`           | rated providers, text preserved (14 columns)       |
| `md_records_cleaned_full.csv`          | rated providers, numeric/boolean only (13 columns) |
| `md_records_cleaned_complete_raw.csv`  | every provider, text preserved                     |
| `md_records_cleaned_complete_full.csv` | every provider, numeric/boolean only               |

The two `complete_*` files keep providers with no valid rating; the two released files drop them. Raw and full are row-aligned, so one fold file applies to both.
