# Georgia — running the pipeline

Quality Rated, rated 1–3. Everything below runs from inside `ga/`.

## 0. What you need first

**The seed is already provided.** `ga_data/ga_seed.csv` ships with this repository — there is no collection step to run and no directory to re-scrape before you start.

DECAL's provider export. It is more than a list of ids — it also supplies `qr_rating` and the county/region columns, which the crawler renames via `ga_ids.json` and merges into every row.

This state also needs `../private/geo_label_map_ga.json`. The cleaning step reads it to label geography consistently and **writes it back if it is missing** — so without the original file the geographic columns come out under different names than the published dataset.

## 1. Collection

```
python ga_crawler.py --headless
```

Playwright against the DECAL families portal, two pages per provider (detail + compliance). `--download-pdfs` also pulls the report PDFs; the pipeline does not need them.

Writes `ga_data/ga_records.csv`, one row per provider. Resume-safe: re-running skips providers already in the output, so an interrupted crawl can be continued rather than restarted.

## 2. Anonymization

```
python ga_anonymize.py --dry-run
python ga_anonymize.py
```

`--dry-run` prints the columns it would remove and writes nothing. The real run writes `ga_data/ga_records_anonymized.csv` — same rows, same order, fewer columns — and a provider-id map to `../private/provider_id_map_ga.csv`.

**That map is the only link back to the real providers. Keep it out of any release.**

## 3. Cleaning

```
python ga_clean_raw.py
python ga_clean_full.py
python ga_clean_complete_raw.py
python ga_clean_complete_full.py
```

Four tables in `ga_data/`:

| file                                   | contents                                            |
| -------------------------------------- | --------------------------------------------------- |
| `ga_records_cleaned_raw.csv`           | rated providers, text preserved (226 columns)       |
| `ga_records_cleaned_full.csv`          | rated providers, numeric/boolean only (333 columns) |
| `ga_records_cleaned_complete_raw.csv`  | every provider, text preserved                      |
| `ga_records_cleaned_complete_full.csv` | every provider, numeric/boolean only                |

The two `complete_*` files keep providers with no valid rating; the two released files drop them. Raw and full are row-aligned, so one fold file applies to both.

## Notes

`ga_ids.json` maps every HTML id to its column name and holds the seed rename map. This is the reference state the three prompts are written against.
