# Kentucky — Kentucky All STARS (1–5)

Source: kynect Public Child Care Search.

## Seed

Run `python ky_seed.py`. It searches kynect
(<https://kynect.ky.gov/benefits/s/child-care-provider>) county by county and
writes `ky_data/ky_seed.csv`, one row per provider, keyed on the licence number.

## Files

| File | Purpose |
|---|---|
| `ky_seed.py` | Builds the seed from kynect's search, county by county |
| `ky_capture.py` | Optional: saves a rendered page and network log |
| `ky_crawler.py` | Adds each provider's detail data: capacity, cost, inspections, accreditation (Playwright) |
| `ky_refetch.py` | Re-collects the detail payload for rows whose crawl failed |
| `ky_data_correction.py` | Merges the refetched detail into the collected and anonymized records |
| `ky_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `ky_columns.json` | Output column list and order |
| `ky_cleaning_utils.py` | Feature builders for the clean scripts |
| `ky_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `ky/`:

```bash
python ky_seed.py
python ky_crawler.py --headless
python ky_refetch.py

python ky_anonymize.py
python ky_data_correction.py --fix-detail

python ky_clean_raw.py                # raw, rated providers
python ky_clean_full.py               # preprocessed, rated providers
python ky_clean_complete_raw.py       # raw, all providers
python ky_clean_complete_full.py      # preprocessed, all providers
```

## Notes

- A blank `capacity`, `num_inspections` or `num_dpoc_agreements` means the detail was not collected, not zero.
