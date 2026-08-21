# Montana — running the pipeline

Best Beginnings STARS, rated 1–5. Everything below runs from inside `mt/`.

## 0. What you need first

**The seed is already provided.** `mt_data/mt_seed.csv` ships with this repository — there is no collection step to run and no directory to re-scrape before you start.

Two halves stacked in one file, tagged by `seed_source`:

- `stars_ratings` — the 213 rated programs transcribed from DPHHS's _STARS Program List by City_ PDF. **The ratings exist nowhere else**, and the PDF carries no licence number.
- `licensing_roster` — the MAQCS licensing directory, which has the provider numbers but no ratings.

This state also needs `../private/geo_label_map_mt.json`. The cleaning step reads it to label geography consistently and **writes it back if it is missing** — so without the original file the geographic columns come out under different names than the published dataset.

## 1. Collection

If the site's markup has changed, `python mt_capture.py` dumps a rendered page so selectors can be checked against real HTML. Not needed for a normal run.

```
python mt_crawler.py --join-only
```

`--join-only` matches the two halves by normalized name + county and writes the records straight from the seed — no network access at all. Drop the flag only to re-pull the live licensing directory; `--merge-only` rebuilds from that pull's cache without re-fetching.

Writes `mt_data/mt_records.csv`, one row per provider. The join is deterministic and rewrites the file whole, so it is safe to re-run at any time.

## 2. Anonymization

```
python mt_anonymize.py --dry-run
python mt_anonymize.py
```

`--dry-run` prints the columns it would remove and writes nothing. The real run writes `mt_data/mt_records_anonymized.csv` — same rows, same order, fewer columns — and a provider-id map to `../private/provider_id_map_mt.csv`.

**That map is the only link back to the real providers. Keep it out of any release.**

## 3. Cleaning

```
python mt_clean_raw.py
python mt_clean_full.py
python mt_clean_complete_raw.py
python mt_clean_complete_full.py
```

Four tables in `mt_data/`:

| file                                   | contents                                          |
| -------------------------------------- | ------------------------------------------------- |
| `mt_records_cleaned_raw.csv`           | rated providers, text preserved (12 columns)      |
| `mt_records_cleaned_full.csv`          | rated providers, numeric/boolean only (5 columns) |
| `mt_records_cleaned_complete_raw.csv`  | every provider, text preserved                    |
| `mt_records_cleaned_complete_full.csv` | every provider, numeric/boolean only              |

The two `complete_*` files keep providers with no valid rating; the two released files drop them. Raw and full are row-aligned, so one fold file applies to both.

## Notes

Rows the matcher will not auto-accept are written to `mt_match_review.csv` with their top candidates. Fill in `provider_number`, save as `mt_manual_matches.csv`, and re-run `--join-only` to fold them in.

The program name is the only self-description Montana publishes, so `mt_anonymize.py` matches a keyterm vocabulary against it and emits `name_*` columns before dropping the name.
