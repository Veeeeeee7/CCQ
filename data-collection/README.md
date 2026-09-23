# Data collection

Code for the CCQ construction pipeline (Section 3.2): **collection**,
**anonymization** and **cleaning**, one self-contained folder per state. Georgia
is the hand-built reference implementation; the other eleven states follow its
structure.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium     # only for the browser-driven states: CA, CO, GA, KY, NC, WI
```

Versions in `requirements.txt` are pinned; a different pandas version can change
cell formatting in the output CSVs.

## Per-state files

| File | Stage | Purpose |
|---|---|---|
| `README.md` | | How to run the state, and state-specific notes |
| `{st}_seed.py` | collection | Builds the seed (states whose seed is scraped; see below) |
| `{st}_crawler.py` | collection | Fetches each provider's public record → `{st}_records.csv` |
| `{st}_capture.py` | collection | Optional: saves a rendered page to check selectors |
| `{st}_refetch*.py`, `{st}_crawler_errors.py` | collection | Re-requests rows the crawl missed or failed |
| `{st}_data_correction.py` | collection | Repairs collected records in place |
| `{st}_anonymize.py` | anonymization | Drops private and rating-leakage columns and replaces `provider_id` with a surrogate → `{st}_records_anonymized.csv` |
| `{st}_columns.json` | cleaning | Output schema: column list and order |
| `{st}_cleaning_utils.py` | cleaning | Feature builders shared by the four clean scripts (`{st}_clean_utils.py` in GA and NC) |
| `{st}_clean_raw.py`, `{st}_clean_full.py` | cleaning | Released **raw** and **preprocessed** files (rated providers only) |
| `{st}_clean_complete_raw.py`, `{st}_clean_complete_full.py` | cleaning | Same two formats, keeping unrated providers |

Not every state has every optional script; see each state's `README.md`.

## Seeds

Collection starts from a seed, `{st}_data/{st}_seed.csv` (MD: `md/md_seed.csv`):
a list of providers, or of counties to search. Seeds are not included; build or
download them as below (details in each state's `README.md`).

| State | How to get the seed |
|---|---|
| CA | Download the *Community Care Licensing Facilities — Child Care Centers* CSV from <https://data.chhs.ca.gov/dataset/ccl-facilities>; keep `facility_number` |
| CO | `python co_seed.py --output co_data/co_seed.csv` (Colorado open data, dataset `a9rr-k8mu`) |
| GA | Download DECAL's provider export from <https://families.decal.ga.gov/provider/data> |
| KY | `python ky_seed.py` |
| MD | Hand-made list of Maryland's 24 counties (one `county` column) |
| MT | `python mt_seed.py` (STARS PDF) stacked with the MAQCS directory pulled by `python mt_crawler.py` |
| NC | `python nc_seed.py --output nc_data/nc_seed.csv` |
| NE | `python ne_seed.py` (finder sitemap and DHHS roster, stacked) |
| OK | `python ok_seed.py` |
| SC | Hand-made list of South Carolina's 46 counties (one `county` column) |
| WA | `python wa_seed.py --socrata` |
| WI | Download DCF's licensed (<https://dcf.wisconsin.gov/files/ccdir/lic/excel/LCC%20Directory.xlsx>) and certified (<https://dcf.wisconsin.gov/files/ccdir/cert/excel/CCC%20Directory.xlsx>) directories and stack them |

## Running a state

Scripts run from inside the state folder and read and write `{st}/{st}_data/`
(not tracked in this repository). The general order is:

```bash
cd {st}
python {st}_seed.py                  # if the seed is scraped
python {st}_crawler.py               # collection
python {st}_anonymize.py --dry-run   # list the columns that will be removed
python {st}_anonymize.py             # anonymization
python {st}_clean_raw.py             # cleaning
python {st}_clean_full.py
python {st}_clean_complete_raw.py
python {st}_clean_complete_full.py
```

To rebuild the cleaned files for every state at once:

```bash
python regenerate_release.py                  # all states
python regenerate_release.py --states wi sc   # a subset
```

## Outputs

```
{st}_data/{st}_records_cleaned_raw.csv            raw, rated providers        (released)
{st}_data/{st}_records_cleaned_full.csv           preprocessed, rated         (released)
{st}_data/{st}_records_cleaned_complete_raw.csv   raw, all providers
{st}_data/{st}_records_cleaned_complete_full.csv  preprocessed, all providers
```

`{st}_anonymize.py` also writes the surrogate-to-real id map to
`data-private/provider_id_map_{st}.csv`, and cleaning in CO, GA, KY, MT, NE, SC
and WA writes a geography pseudonymization map to
`data-private/geo_label_map_{st}.json`. Neither is released. Without the
original maps, a rebuild produces different surrogate ids and geography labels
than the published files.

## Prompts

The prompts for the three stages are in [`prompts/`](prompts/):
`01_collection_prompt.md`, `02_anonymization_prompt.md` and
`03_cleaning_prompt.md`.
