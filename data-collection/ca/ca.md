# California — running the pipeline

Quality Counts California, rated 1–5. Everything below runs from inside `ca/`.

## 0. What you need first

**The seed is already provided.** `ca_data/ca_seed.csv` ships with this repository — there is no collection step to run and no directory to re-scrapebefore you start.

One column, `facility_number`: every CCLD facility licence number in the state.

## 1. Collection

```
python ca_crawler.py
```

No CLI flags. Paths and headless mode are constants at the bottom of the file; edit there to change them.

Writes `ca_data/ca_records.csv`, one row per provider. Resume-safe: re-running skips providers already in the output, so an interrupted crawl can be continued rather than restarted.

## 2. Anonymization

```
python ca_anonymize.py --dry-run
python ca_anonymize.py
```

`--dry-run` prints the columns it would remove and writes nothing. The real run writes `ca_data/ca_records_anonymized.csv` — same rows, same order, fewer columns — and a provider-id map to `../private/provider_id_map_ca.csv`.

**That map is the only link back to the real providers. Keep it out of any release.**

## 3. Cleaning

```
python ca_clean_raw.py
python ca_clean_full.py
python ca_clean_complete_raw.py
python ca_clean_complete_full.py
```

Four tables in `ca_data/`:

| file                                   | contents                                           |
| -------------------------------------- | -------------------------------------------------- |
| `ca_records_cleaned_raw.csv`           | rated providers, text preserved (21 columns)       |
| `ca_records_cleaned_full.csv`          | rated providers, numeric/boolean only (12 columns) |
| `ca_records_cleaned_complete_raw.csv`  | every provider, text preserved                     |
| `ca_records_cleaned_complete_full.csv` | every provider, numeric/boolean only               |

The two `complete_*` files keep providers with no valid rating; the two released files drop them. Raw and full are row-aligned, so one fold file applies to both.

## Notes

`ca_crawler_errors.py` re-visits rows whose `errors` column is set. `ca_data_analysis.py` and `ca_data_correction.py` are one-off inspection helpers and are not part of the pipeline.
