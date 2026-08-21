# North Carolina — running the pipeline

NC Star Rated License, rated 1–5. Everything below runs from inside `nc/`.

## 0. What you need first

**The seed is already provided.** `nc_data/nc_seed.csv` ships with this repository — there is no collection step to run and no directory to re-scrape before you start.

Every DCDEE facility id, with the name, type, county and licence number alongside it.

## 1. Collection

If the site's markup has changed, `python nc_capture.py` dumps a rendered page so selectors can be checked against real HTML. Not needed for a normal run.

```
python nc_crawler.py --headless
```

Playwright. The output is large (~100 MB); allow time.

Writes `nc_data/nc_records.csv`, one row per provider. Resume-safe: re-running skips providers already in the output, so an interrupted crawl can be continued rather than restarted.

## 2. Anonymization

```
python nc_anonymize.py --dry-run
python nc_anonymize.py
```

`--dry-run` prints the columns it would remove and writes nothing. The real run writes `nc_data/nc_records_anonymized.csv` — same rows, same order, fewer columns — and a provider-id map to `../private/provider_id_map_nc.csv`.

**That map is the only link back to the real providers. Keep it out of any release.**

## 3. Cleaning

```
python nc_clean_raw.py
python nc_clean_full.py
python nc_clean_complete_raw.py
python nc_clean_complete_full.py
```

Four tables in `nc_data/`:

| file                                   | contents                                          |
| -------------------------------------- | ------------------------------------------------- |
| `nc_records_cleaned_raw.csv`           | rated providers, text preserved (15 columns)      |
| `nc_records_cleaned_full.csv`          | rated providers, numeric/boolean only (9 columns) |
| `nc_records_cleaned_complete_raw.csv`  | every provider, text preserved                    |
| `nc_records_cleaned_complete_full.csv` | every provider, numeric/boolean only              |

The two `complete_*` files keep providers with no valid rating; the two released files drop them. Raw and full are row-aligned, so one fold file applies to both.
