# Colorado — Colorado Shines (1–5)

Source: `coloradoshines.com`. Pages collected 2026-07-06 to 07-07.

## Seed

Run `python co_seed.py --output co_data/co_seed.csv`. It pulls the *Colorado
Licensed Child Care Facilities Report* (dataset `a9rr-k8mu`) from
<https://data.colorado.gov>, which also supplies `qr_rating`. The release used the
2026-07-06 extract.

## Files

| File | Purpose |
|---|---|
| `co_seed.py` | Pulls the licensed-facility list from the Colorado open-data API |
| `co_capture.py` | Optional: saves a rendered page to check selectors |
| `co_crawler.py` | Finds each provider's Colorado Shines page and scrapes it (Playwright) |
| `co_data_correction.py` | Provider-matching rules imported by the crawler and refetcher (not run directly) |
| `co_refetch.py` | Re-requests pages the crawl missed and merges them into the records |
| `co_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `co_columns.json` | Output column list and order |
| `co_cleaning_utils.py` | Feature builders for the clean scripts |
| `co_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `co/`:

```bash
python co_seed.py --output co_data/co_seed.csv
python co_crawler.py --headless
python co_refetch.py --build-ids
python co_refetch.py --resume
python co_refetch.py --merge

python co_anonymize.py
python co_clean_raw.py                # raw, rated providers
python co_clean_full.py               # preprocessed, rated providers
python co_clean_complete_raw.py       # raw, all providers
python co_clean_complete_full.py      # preprocessed, all providers
```

## Notes

- `co_anonymize.py` derives the `has_documented_*` flags from the licensing-history narratives before removing them. To carry a later stage-1 repair into the anonymized file without re-drawing ids, use `co_anonymize.py --replay all`.
