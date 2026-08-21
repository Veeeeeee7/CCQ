# Washington — running the pipeline

Early Achievers, rated 2–5. Everything below runs from inside `wa/`.

## 0. What you need first

**The seed is already provided.** `wa_data/wa_seed.csv` ships with this repository — there is no collection step to run and no directory to re-scrape before you start.

One row per provider, keyed on the 18-char Salesforce Account Id, with the DCYF Socrata open dataset already joined on as `socrata_*` columns. Family child care homes are absent from that dataset by design, so their `socrata_*` cells are blank.

This state also needs `../private/geo_label_map_wa.json`. The cleaning step reads it to label geography consistently and **writes it back if it is missing** — so without the original file the geographic columns come out under different names than the published dataset.

## 1. Collection

If the site's markup has changed, `python wa_capture.py` dumps a rendered page so selectors can be checked against real HTML. Not needed for a normal run.

```
python wa_crawler.py
```

Salesforce endpoints, plain HTTP — no browser anywhere. `--ca-bundle` / `--insecure` exist because the certificate chain is awkward on some machines. `--skip-detail` fetches the API fields only.

Writes `wa_data/wa_records.csv`, one row per provider. Resume-safe: re-running skips providers already in the output, so an interrupted crawl can be continued rather than restarted.

## 2. Anonymization

```
python wa_anonymize.py --dry-run
python wa_anonymize.py
```

`--dry-run` prints the columns it would remove and writes nothing. The real run writes `wa_data/wa_records_anonymized.csv` — same rows, same order, fewer columns — and a provider-id map to `../private/provider_id_map_wa.csv`.

**That map is the only link back to the real providers. Keep it out of any release.**

## 3. Cleaning

```
python wa_clean_raw.py
python wa_clean_full.py
python wa_clean_complete_raw.py
python wa_clean_complete_full.py
```

Four tables in `wa_data/`:

| file                                   | contents                                           |
| -------------------------------------- | -------------------------------------------------- |
| `wa_records_cleaned_raw.csv`           | rated providers, text preserved (64 columns)       |
| `wa_records_cleaned_full.csv`          | rated providers, numeric/boolean only (43 columns) |
| `wa_records_cleaned_complete_raw.csv`  | every provider, text preserved                     |
| `wa_records_cleaned_complete_full.csv` | every provider, numeric/boolean only               |

The two `complete_*` files keep providers with no valid rating; the two released files drop them. Raw and full are row-aligned, so one fold file applies to both.

## Notes

The grain is the Salesforce `Id`. The `provider_id` column in the records file is the crawler's resume key and is discarded.
