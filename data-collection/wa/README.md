# Washington — Early Achievers (1–5)

Source: Child Care Check (`findchildcarewa.org`), plain HTTP. Collected 2026-07-09.

## Seed

Run `python wa_seed.py --socrata`. It sweeps `findchildcarewa.org` by ZIP code into
`wa_data/wa_seed.csv` and downloads the DCYF open dataset
(<https://data.wa.gov/resource/was8-3ni8>) to `wa_data/wa_socrata.csv`. Join the
latter onto the seed (provider id = `wacompassid`), prefixing its columns with
`socrata_`.

## Files

| File | Purpose |
|---|---|
| `wa_seed.py` | Builds the provider list and fetches the DCYF open dataset |
| `wa_capture.py` | Optional: saves a rendered page and network log |
| `wa_crawler.py` | Collects each provider's record |
| `wa_anonymize.py` | Removes private and rating-leakage columns, assigns surrogate ids |
| `wa_columns.json` | Output column list and order |
| `wa_cleaning_utils.py` | Feature builders for the clean scripts |
| `wa_clean_*.py` | The four cleaned files (raw / preprocessed × rated / all) |

## Run

From inside `wa/`:

```bash
python wa_seed.py --socrata
python wa_crawler.py

python wa_anonymize.py
python wa_clean_raw.py                # raw, rated providers
python wa_clean_full.py               # preprocessed, rated providers
python wa_clean_complete_raw.py       # raw, all providers
python wa_clean_complete_full.py      # preprocessed, all providers
```

## Notes

- `qr_rating` is the provider's last recorded Early Achievers level; no released provider is at level 1.
- 109 released rows belong to closed providers. To keep operating providers only, drop rows where `provider_status` starts with `Closed` (raw) or `status_closed` is true (preprocessed).
