# Montana — Best Beginnings STARS to Quality (1–5)

Montana's ratings exist only in a DPHHS PDF (*STARS Program List by City*,
updated 7/1/2023) that carries no licence number; the crawler joins them to the
MAQCS licensed-provider directory by program name and county.

## Seed

`mt_data/mt_seed.csv` stacks two halves, tagged by a `seed_source` column:

- `stars_ratings`: run `python mt_seed.py --out mt_data/mt_seed_stars.csv`, which
  transcribes the PDF
  (<https://dphhs.mt.gov/assets/ecfsd/childcare/STARS/STARSProgramListbyCity.pdf>).
- `licensing_roster`: the MAQCS directory
  (<https://mtdphhs.my.site.com/MAQCSChildCareLicensing/s/provider-search>), pulled
  by `python mt_crawler.py` (without `--join-only`). The release used the
  2026-07-09 pull.

## Files

| File | Purpose |
|---|---|
| `mt_seed.py` | Transcribes the STARS PDF (needs the PDF; refuses to overwrite the full seed without `--force`) |
| `mt_capture.py` | Optional: saves a rendered page and network log |
| `mt_crawler.py` | Joins ratings to the licensing directory; `--join-only` runs offline from the seed |
| `mt_data_correction.py` | Repairs seed cities; `--sync-anonymized` carries a re-run join into the anonymized file |
| `mt_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `mt_columns.json` | Output column list and order |
| `mt_cleaning_utils.py` | Feature builders for the clean scripts |
| `mt_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `mt/`:

```bash
python mt_data_correction.py --fix-seed-city
python mt_crawler.py --join-only

python mt_anonymize.py
python mt_clean_raw.py                # raw, rated providers
python mt_clean_full.py               # preprocessed, rated providers
python mt_clean_complete_raw.py       # raw, all providers
python mt_clean_complete_full.py      # preprocessed, all providers
```

## Notes

- The ratings are a 2023 snapshot. 156 of 213 rated programs matched a licence; the rest keep their rating under a synthetic id, and `provider_id_source` (raw) / `has_license` (preprocessed) say which.
