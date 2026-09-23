# Nebraska — Step Up to Quality (1–5)

Built from two sources joined on the licence number: the Step Up to Quality
finder pages (collected 2026-08-18 to 08-21; the only source of `qr_rating`) and
the DHHS licensing roster (dated 2025-10-15; every `dhhs_*` column).

## Seed

Run `python ne_seed.py`. It writes the finder sitemap (`ne_data/ne_facilities_seed.csv`)
and the DHHS roster (`ne_data/ne_licensing.csv`, from
<https://gis.ne.gov/Agency/rest/services/DHHS_Licensed_Child_Care/FeatureServer/0>).
Stack the two into `ne_data/ne_seed.csv` with a `seed_source` column (`finder` /
`dhhs_roster`).

## Files

| File | Purpose |
|---|---|
| `ne_seed.py` | Builds the finder URL list and the DHHS roster |
| `ne_capture.py` | Optional: logs network traffic while browsing the finder |
| `ne_crawler.py` | Scrapes every finder page and joins the DHHS roster |
| `ne_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `ne_columns.json` | Output column list and order |
| `ne_cleaning_utils.py` | Feature builders for the clean scripts |
| `ne_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `ne/`:

```bash
python ne_seed.py
python ne_crawler.py

python ne_anonymize.py
python ne_clean_raw.py                # raw, rated providers
python ne_clean_full.py               # preprocessed, rated providers
python ne_clean_complete_raw.py       # raw, all providers
python ne_clean_complete_full.py      # preprocessed, all providers
```

## Notes

- `dhhs_*` values are the licensed figures on file in October 2025 and can differ from the provider's page (e.g. `capacity` vs `dhhs_capacity`).
