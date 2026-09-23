# North Carolina — NC Star Rated License (1–5)

Source: DCDEE "Search for Child Care" portal.

## Seed

Run `python nc_seed.py --output nc_data/nc_seed.csv`. It scrapes the statewide
facility list from DCDEE WORKS
(<https://ncchildcare.ncdhhs.gov/works_simulator/facilityAssociation.html>):
6,559 facility ids, about half of which return no detail page and are dropped.

## Files

| File | Purpose |
|---|---|
| `nc_seed.py` | Scrapes the statewide facility list |
| `nc_capture.py` | Optional: saves a rendered page |
| `nc_crawler.py` | Scrapes each facility's star rating, licence and visit history (Playwright) |
| `nc_data_correction.py` | Re-derives values from the saved page text and patches the records in place |
| `nc_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `nc_columns.json` | Output column list and order |
| `nc_clean_utils.py` | Feature builders for the clean scripts |
| `nc_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `nc/`:

```bash
python nc_seed.py --output nc_data/nc_seed.csv
python nc_crawler.py --headless

python nc_anonymize.py
python nc_data_correction.py --check
python nc_data_correction.py --all

python nc_clean_raw.py                # raw, rated providers
python nc_clean_full.py               # preprocessed, rated providers
python nc_clean_complete_raw.py       # raw, all providers
python nc_clean_complete_full.py      # preprocessed, all providers
```
