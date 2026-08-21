# South Carolina — running the pipeline

ABC Quality, rated letter grades → 1–5. Everything below runs from inside `sc/`.

## 0. What you need first

**The seed is already provided.** `sc_data/sc_seed.csv` ships with this repository — there is no collection step to run and no directory to re-scrape before you start.

A single `county` column: all 46 counties. The search returns every provider in a county, so the county list is the whole seed.

This state also needs `../private/geo_label_map_sc.json`. The cleaning step reads it to label geography consistently and **writes it back if it is missing** — so without the original file the geographic columns come out under different names than the published dataset.

## 1. Collection

If the site's markup has changed, `python sc_capture.py` dumps a rendered page so selectors can be checked against real HTML. Not needed for a normal run.

```
python sc_crawler.py
```

`--counties A B` limits the sweep. `--merge-only` rebuilds the records file from per-county dumps without re-fetching. Run `python sc_tls_probe.py` first if the handshake fails — the site is fussy about TLS.

Writes `sc_data/sc_records.csv`, one row per provider. Resume-safe: re-running skips providers already in the output, so an interrupted crawl can be continued rather than restarted.

## 2. Anonymization

```
python sc_anonymize.py --dry-run
python sc_anonymize.py
```

`--dry-run` prints the columns it would remove and writes nothing. The real run
writes `sc_data/sc_records_anonymized.csv` — same rows, same order, fewer
columns — and a provider-id map to `../private/provider_id_map_sc.csv`.

**That map is the only link back to the real providers. Keep it out of any
release.**

## 3. Cleaning

```
python sc_clean_raw.py
python sc_clean_full.py
python sc_clean_complete_raw.py
python sc_clean_complete_full.py
```

Four tables in `sc_data/`:

| file                                   | contents                                          |
| -------------------------------------- | ------------------------------------------------- |
| `sc_records_cleaned_raw.csv`           | rated providers, text preserved (13 columns)      |
| `sc_records_cleaned_full.csv`          | rated providers, numeric/boolean only (9 columns) |
| `sc_records_cleaned_complete_raw.csv`  | every provider, text preserved                    |
| `sc_records_cleaned_complete_full.csv` | every provider, numeric/boolean only              |

The two `complete_*` files keep providers with no valid rating; the two released
files drop them. Raw and full are row-aligned, so one fold file applies to both.

## Checks worth running

- `provider_id` and `qr_rating` are the first two columns of all four files.
- `provider_id` is unique and has not lost a leading zero.
- The released files contain only valid ratings (letter grades → 1–5); the `complete_*`
  files contain every row.
- `_full` is entirely numeric or boolean apart from `provider_id`.

## Notes

Exempt providers carry no permit number, so `sc_anonymize.py` mints a deterministic id for them **before** dropping the provider name it is built from.
