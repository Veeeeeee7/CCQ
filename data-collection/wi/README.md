# Wisconsin — YoungStar (1–5)

Source: Wisconsin Child Care Finder. Rows are per provider-location
(`{provider}-{location}`). Collected 2026-06-05 to 06-08.

## Seed

Download DCF's statewide licensed
(<https://dcf.wisconsin.gov/files/ccdir/lic/excel/LCC%20Directory.xlsx>) and
certified (<https://dcf.wisconsin.gov/files/ccdir/cert/excel/CCC%20Directory.xlsx>)
directories and stack them as `wi_data/wi_seed.csv` with a `seed_source` column
(`lcc` / `ccc`), each half keeping its own header rows. The release used the
6/5/26 refresh. The directories also supply `regulation_type`, `capacity` and the
age range.

## Files

| File | Purpose |
|---|---|
| `wi_capture.py` | Optional: saves a rendered page |
| `wi_crawler.py` | Scrapes each provider-location page (Playwright); `--retry-errors` re-visits failed rows |
| `wi_data_correction.py` | Adds roster fields and strips identifiers from the anonymized file; `--merge-retry` syncs rows recovered by `--retry-errors` |
| `wi_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `wi_columns.json` | Output column list and order |
| `wi_cleaning_utils.py` | Feature builders for the clean scripts |
| `wi_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `wi/`:

```bash
python wi_crawler.py --headless

python wi_anonymize.py
python wi_data_correction.py --add-seed-columns
python wi_data_correction.py --strip-stage2

python wi_clean_raw.py                # raw, rated providers
python wi_clean_full.py               # preprocessed, rated providers
python wi_clean_complete_raw.py       # raw, all providers
python wi_clean_complete_full.py      # preprocessed, all providers
```

## Notes

- `capacity` is licensed capacity for licensed types and group size for certified family homes, so compare it within a `regulation_type`.
