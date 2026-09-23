# Georgia — Quality Rated (1–3)

Source: DECAL families portal. Georgia is the hand-built reference
implementation the other states follow.

## Seed

Download DECAL's provider export from <https://families.decal.ga.gov/provider/data>
(run the export on that page) and save it as `ga_data/ga_seed.csv`. It also
supplies `qr_rating` and county/region.

## Files

| File | Purpose |
|---|---|
| `ga_crawler.py` | Scrapes each provider's detail and compliance pages (Playwright); `--download-pdfs` also saves report PDFs |
| `ga_ids.json` | Maps page element ids to column names, and the seed's column renames |
| `ga_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `ga_columns.json` | Output column list and order |
| `ga_clean_utils.py` | Feature builders for the clean scripts |
| `ga_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `ga/`:

```bash
python ga_crawler.py --headless

python ga_anonymize.py
python ga_clean_raw.py                # raw, rated providers
python ga_clean_full.py               # preprocessed, rated providers
python ga_clean_complete_raw.py       # raw, all providers
python ga_clean_complete_full.py      # preprocessed, all providers
```
