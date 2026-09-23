# South Carolina — ABC Quality (1–5)

Source: the `abcquality.org` provider-search CSV export, one request per county.

## Seed

`sc_data/sc_seed.csv` is a one-column CSV, `county`, listing South Carolina's 46
counties as the site's county menu spells them (e.g. `McCormick`). Create it by
hand.

## Files

| File | Purpose |
|---|---|
| `sc_capture.py` | Optional: saves a rendered page and network log |
| `sc_crawler.py` | Downloads every county's export |
| `sc_refetch.py` | Re-pulls the export as a fresh snapshot (`--fetch`, then `--merge`) |
| `sc_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `sc_columns.json` | Output column list and order |
| `sc_cleaning_utils.py` | Feature builders for the clean scripts |
| `sc_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `sc/`:

```bash
python sc_crawler.py

python sc_anonymize.py
python sc_clean_raw.py                # raw, rated providers
python sc_clean_full.py               # preprocessed, rated providers
python sc_clean_complete_raw.py       # raw, all providers
python sc_clean_complete_full.py      # preprocessed, all providers
```

## Notes

- Exempt providers have no permit number; `sc_anonymize.py` gives them a deterministic id before dropping the name.
