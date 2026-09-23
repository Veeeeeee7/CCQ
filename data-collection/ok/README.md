# Oklahoma — Reaching for the Stars (1–5)

Source: OKDHS Child Care Locator (`ccl.dhs.ok.gov`).

## Seed

Run `python ok_seed.py`. It builds the provider list from the locator search
(<https://ccl.dhs.ok.gov/>) and writes `ok_data/ok_seed.csv`, keyed on the state
licence number.

## Files

| File | Purpose |
|---|---|
| `ok_seed.py` | Builds the provider list from the locator search |
| `ok_crawler.py` | Fetches and parses each provider's page |
| `ok_data_correction.py` | Refetches failed rows and merges them into the records |
| `ok_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `ok_columns.json` | Output column list and order |
| `ok_cleaning_utils.py` | Feature builders for the clean scripts |
| `ok_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `ok/`:

```bash
python ok_seed.py
python ok_crawler.py
python ok_data_correction.py --backfill-crawled-at
python ok_data_correction.py --refetch-failed
python ok_data_correction.py --merge-refetch

python ok_anonymize.py
python ok_clean_raw.py                # raw, rated providers
python ok_clean_full.py               # preprocessed, rated providers
python ok_clean_complete_raw.py       # raw, all providers
python ok_clean_complete_full.py      # preprocessed, all providers
```

## Notes

- Monitoring and complaint counts cover the portal's roughly three-year window, not a provider's full history.
- `hours_<day> = 'other'` marks a rare value suppressed for privacy; the numeric hours columns are still exact.
