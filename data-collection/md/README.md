# Maryland — Maryland EXCELS (1–5)

Source: the EXCELS "Find a Program" portal's public JSON/CSV API; each county
search returns every provider in it. Collected 2026-07-06.

## Seed

`md_seed.csv` (beside the scripts, not in `md_data/`) is a one-column CSV,
`county`, listing Maryland's 24 jurisdictions as the portal spells them (e.g.
`Saint Mary's`). Create it by hand.

## Files

| File | Purpose |
|---|---|
| `md_capture.py` | Optional: saves a rendered page and network log |
| `md_crawler.py` | Downloads every county's provider export |
| `md_refetch.py` | Adds licensed age ranges and capacity, which the export lacks, to the collected and anonymized records |
| `md_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `md_columns.json` | Output column list and order |
| `md_cleaning_utils.py` | Feature builders for the clean scripts |
| `md_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `md/`:

```bash
python md_crawler.py
python md_refetch.py --fetch

python md_anonymize.py
python md_refetch.py --merge

python md_clean_raw.py                # raw, rated providers
python md_clean_full.py               # preprocessed, rated providers
python md_clean_complete_raw.py       # raw, all providers
python md_clean_complete_full.py      # preprocessed, all providers
```

## Notes

- St. Mary's County is missing from the released data.
- No geography columns are released for Maryland.
