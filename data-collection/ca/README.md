# California — Quality Counts California (1–5)

Source: `mychildcareplan.org`. Collected March–June 2026.

## Seed

`ca_data/ca_seed.csv` has one column, `facility_number`. Download the
*Community Care Licensing Facilities — Child Care Centers* CSV from the CHHS open
data portal (<https://data.chhs.ca.gov/dataset/ccl-facilities>) and keep its
`facility_number` column.

## Files

| File | Purpose |
|---|---|
| `ca_crawler.py` | Searches each licence number and scrapes the provider page (Playwright) |
| `ca_crawler_errors.py` | Re-visits rows the crawl marked as failed |
| `ca_data_correction.py` | Repairs the collected records; marks duplicate provider pages and wrong-provider matches, which cleaning drops |
| `ca_refetch_openings.py` | Re-reads the openings block for the released providers into a side file |
| `ca_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `ca_columns.json` | Output column list and order |
| `ca_cleaning_utils.py` | Feature builders for the clean scripts |
| `ca_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `ca/`:

```bash
python ca_crawler.py
python ca_crawler_errors.py
python ca_data_correction.py ca_data/ca_records.csv -o ca_data/ca_records.csv
python ca_data_correction.py ca_data/ca_records.csv --fix-language -o ca_data/ca_records.csv

python ca_anonymize.py
python ca_data_correction.py --fix-openings-capacity
python ca_data_correction.py --mark-duplicate-profiles
python ca_refetch_openings.py --resume
python ca_data_correction.py --apply-refetch

python ca_clean_raw.py                # raw, rated providers
python ca_clean_full.py               # preprocessed, rated providers
python ca_clean_complete_raw.py       # raw, all providers
python ca_clean_complete_full.py      # preprocessed, all providers
```

## Notes

- Several licences can point to one provider page; only one row per page is kept.
- In the released files the openings columns were observed in September 2026; all other columns are from March–June 2026.
