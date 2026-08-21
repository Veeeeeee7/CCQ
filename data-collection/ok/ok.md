# Oklahoma — running the pipeline

Reaching for the Stars, rated 1–5. Everything below runs from inside `ok/`.

## 0. What you need first

**The seed is already provided.** `ok_data/ok_seed.csv` ships with this repository — there is no collection step to run and no directory to re-scrape before you start.

The OKDHS Child Care Locator's provider list, keyed on `provider_id`.

## 1. Collection

```
python ok_crawler.py
```

Plain HTTP, server-rendered. `--limit N` for a smoke test.

Writes `ok_data/ok_records.csv`, one row per provider. Resume-safe: re-running skips providers already in the output, so an interrupted crawl can be continued rather than restarted.

## 2. Anonymization

```
python ok_anonymize.py --dry-run
python ok_anonymize.py
```

`--dry-run` prints the columns it would remove and writes nothing. The real run writes `ok_data/ok_records_anonymized.csv` — same rows, same order, fewer columns — and a provider-id map to `../private/provider_id_map_ok.csv`.

**That map is the only link back to the real providers. Keep it out of any release.**

## 3. Cleaning

```
python ok_clean_raw.py
python ok_clean_full.py
python ok_clean_complete_raw.py
python ok_clean_complete_full.py
```

Four tables in `ok_data/`:

| file                                   | contents                                           |
| -------------------------------------- | -------------------------------------------------- |
| `ok_records_cleaned_raw.csv`           | rated providers, text preserved (23 columns)       |
| `ok_records_cleaned_full.csv`          | rated providers, numeric/boolean only (12 columns) |
| `ok_records_cleaned_complete_raw.csv`  | every provider, text preserved                     |
| `ok_records_cleaned_complete_full.csv` | every provider, numeric/boolean only               |

The two `complete_*` files keep providers with no valid rating; the two released files drop them. Raw and full are row-aligned, so one fold file applies to both.

## Notes

The two monitoring/complaint JSON blobs are removed in phase 2, but the `monitoring_*`, `complaint_*`, `n_visits_with_noncompliance` and `avg_compliance_pct` columns derived from them are kept — that derivation happens in `ok_anonymize.py`, before the drop.
