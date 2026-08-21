# Prompt 3 — Cleaning

Your job is to turn one U.S. state's provider table into four modelling-ready tables:

```
{st}_data/{st}_records_anonymized.csv             ← your input
{st}_data/{st}_records_cleaned_raw.csv            ← produced
{st}_data/{st}_records_cleaned_full.csv           ← produced
{st}_data/{st}_records_cleaned_complete_raw.csv   ← produced
{st}_data/{st}_records_cleaned_complete_full.csv  ← produced
```

Treat the input as given. Its columns are the ones you have to work with, and its shape is not something to second-guess.

---

## 0. Your task

Write the state's complete cleaning pipeline in a single pass. There is no live site to probe and no recon loop, so there are no checkpoints: you have everything you need to finish.

You work from two things:

1. **`{st}_records_anonymized.csv`'s column names and their unique values.** This is what tells you a cell is tab-joined, or a ratio, or a `$`-prefixed string, or a JSON blob, and therefore which treatment it needs.
2. **The Georgia reference implementation**, attached, which you adapt.

**Georgia is the reference, and it is the only one.** These files are attached:

| attachment                                              | role                                                                 |
| ------------------------------------------------------- | -------------------------------------------------------------------- |
| `ga_clean_utils.py`                                     | constants, mode-aware feature builders, the shared `finalize()` tail |
| `ga_columns.json`                                       | the `{"full": [...], "raw": [...]}` scaffold                         |
| `ga_clean_raw.py`, `ga_clean_full.py`                   | the two standard drivers                                             |
| `ga_clean_complete_raw.py`, `ga_clean_complete_full.py` | the two complete drivers                                             |
| `ga_records_anonymized.csv`                             | Georgia's stage-2 output, i.e. what these scripts read               |

Read them in full before writing anything. You have no access to a wider repository. Everything you need is attached.

---

## 1. Deliverables

- `{st}_columns.json` — the stable output scaffold: a
  `{"raw": [...], "full": [...]}` object listing the canonical columns for each variant, in order. `provider_id` and `qr_rating` are always the **first two entries of both lists**.
- `{st}_clean_utils.py` — constants, rename maps, feature builders, and `finalize()`.
- `{st}_clean_raw.py` — light preprocessing, valid ratings only.
- `{st}_clean_full.py` — full preprocessing, valid ratings only.
- `{st}_clean_complete_raw.py` — light preprocessing, all rows kept.
- `{st}_clean_complete_full.py` — full preprocessing, all rows kept.

Outputs:

```
{st}_data/{st}_records_cleaned_raw.csv
{st}_data/{st}_records_cleaned_complete_raw.csv
{st}_data/{st}_records_cleaned_full.csv
{st}_data/{st}_records_cleaned_complete_full.csv
```

---

## 2. The two axes

Four scripts, from two orthogonal choices.

### Preprocessing depth

**raw** — minimal, _text-preserving_ tabulation, for inspection and LLM-based methods. Strip prefixes and suffixes (`$`, a leading `"This program provides "`, a trailing `" provided"`); split a multi-value cell into one text column per discovered item; explode nested JSON into per-key text columns; parse a structured blob into its parts — but **keep human-readable text in place**.

Georgia's builders are all documented no-ops in `"raw"` mode, because Georgia's raw scaffold keeps the tab-joined originals verbatim. That is the simplest instance of the rule, not a different rule. A state whose cells pack pipe-joined event lists or JSON objects should decompose them in raw mode, into text columns. Decompose when a cell holds several values; leave it alone when it already holds one readable value.

**full** — strictly **numeric / boolean** (plus `provider_id`), ready for classical tabular ML. Multi-value fields become **presence booleans**; categoricals become **one-hot**; JSON becomes **counts**; ratios and time strings become numbers. Text byproducts are dropped simply by leaving them out of the `full` list in `{st}_columns.json`.

### Row filtering

**standard** (`{st}_clean_raw.py` / `{st}_clean_full.py`) — keep only rows whose `qr_rating` is a valid score. Out-of-range, non-numeric and missing ratings are coerced to `None` and the row is dropped.

**complete** (`{st}_clean_complete_raw.py` / `{st}_clean_complete_full.py`) — keep **every** row. The rating is still coerced to numeric, but an invalid one becomes `None` and the row survives, so `complete_*` is a row-superset of its sibling and the two `complete_*` files are row-aligned with each other.

Keep each `*_complete_*` script identical to its sibling except for the one `finalize()` argument (`keep_invalid_target=True`) and the cosmetic differences that follow from it: the docstring, the default `--output` path, and the `[complete_raw]` / `[complete_full]` print tag. Georgia also swaps `ParseLog` for a `_NullLog` stub in the complete pair, since only the released pair writes a parse log; mirror that.

Raw and full run the **same early steps** and the **same row filtering**, so their outputs are row-aligned and one fold file applies to both.

---

## 3. Driver structure

Each of the four scripts is short. Georgia's shape, in order:

```python
log = U.ParseLog(args.log)                      # or _NullLog() for complete_*
scaffold = json.loads(Path(args.scaffold).read_text())
df = pd.read_csv(args.input, low_memory=False)  # dtype=str for the id if the
                                                # state zero-pads — see §4

# --- shared early steps (identical across all four) ---
df = U.drop_error_rows(df, log)
df = U.clean_currency(df)          # full only, if the state has currency cols
df = U.strip_dollars(df)
U.check_grain_unique(df, U.ID_COL, log)
df = df.drop(columns=[c for c in U.NON_FEATURE_COLS if c in df.columns],
             errors="ignore")
df = U.coerce_booleans(df, U.BOOLEAN_COLS)

# --- per-field builders, in a fixed documented order ---
df = U.clean_<field>(df, mode)     # mode is "raw" or "full"
...

out = U.finalize(df, which, scaffold, log)                            # standard
out = U.finalize(df, which, scaffold, log, keep_invalid_target=True)  # complete

U.write_output(out, args.output)
log.save()
```

`--input`, `--output`, `--scaffold` and `--log` are argparse flags with sensible defaults; `--input` defaults to `{st}_data/{st}_records_anonymized.csv` in all four scripts, and `--output` to that script's file from §1.

The `full` drivers end with a sanity check: every column except `provider_id` must be numeric or boolean, and anything that survives non-numeric is logged as a warning naming the offending columns. Copy Georgia's.

---

## 4. Constants block

At the top of `{st}_clean_utils.py`, under a clearly-marked **EDITABLE CONSTANTS** header, with a comment on each explaining the state-specific choice:

- **`ID_COL` / `TARGET_COL` / `RENAME_MAP`** — the native column names and the map to `provider_id` / `qr_rating`. Georgia's is an identity safety-net because its crawler already renamed them; most states need a real map.
- **`VALID_TARGET_VALUES`** — the set of scores that count as valid. Georgia's system issues 1-, 2- or 3-star ratings, so `{1, 2, 3}`. Anything outside the set is coerced to missing. Confirm this set against the state's published rating scale, not against the observed values — an absent level is still a valid level.
- **`ERROR_FLAG_COLS`** — the crawler's `errors` column (and any sibling). Rows flagged here failed to scrape and are dropped **first**, before engineering or dedup, so raw and full stay row-aligned.
- **`NON_FEATURE_COLS`** — source artifacts and metadata: detail URLs, match diagnostics, download paths, raw section text superseded by a parsed column, a redundant complement flag (Georgia drops `for_profit` because `non_profit` says the same thing).
- **`BOOLEAN_COLS`** — checkmark fields and Yes/No flags to coerce to the nullable `boolean` dtype in both variants.
- **The multi-value / one-hot / language field lists** your builders consume.
- **`DISCOVERED_PREFIXES`** — see §6.
- **`TRAINING_EXCLUDE = {"full": [], "raw": []}`** — a documented late hook for a column that matches the scaffold but should not feed a model. Usually empty; keep it so the hook has an obvious home.

**IDs are strings, always.** If the state zero-pads its identifier, read the input with `dtype={ID_COL: str}` and say so in a comment. Georgia can skip this because its ID carries a letter prefix (`"FR-000000288"`), so there are no leading zeros to lose — that comment is in `ga_clean_full.py` and is exactly the kind of note to reproduce for your state's own situation.

---

## 5. Feature builders

Each builder takes `(df, mode)` and branches on `mode in {"raw", "full"}`, so one function documents both variants of a field and the two can never drift. Each is small, focused, and never raises: a parse failure is logged and returns `NaN` or a sensible zero.

Georgia's builders, and what to generalise from each:

| Georgia builder                                | raw                                                   | full                                                         |
| ---------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| `split_one_hot` (over `SPLIT_ONEHOT_FIELDS`)   | the joined original kept, or one text column per item | presence booleans, one column per discovered item            |
| `clean_provider_type`                          | the label text                                        | one-hot over the declared code list                          |
| `clean_languages`                              | the text                                              | spoken/taught booleans per language family                   |
| `clean_curriculum`                             | the text                                              | booleans                                                     |
| `clean_rates_table`                            | the parts, as text                                    | money as floats, ratios and min–max spans split into numbers |
| `clean_compliance_table`                       | the text                                              | per-year counts and one-hot of the compliance grade          |
| `clean_licensed_capacity`, `clean_pre_k_slots` | the text                                              | nullable `Int64`                                             |
| `clean_operating_hours` / `_months` / `_days`  | the text                                              | minutes since midnight, span lengths, per-day booleans       |
| `clean_has_liability`                          | the Yes/No/N-A text                                   | boolean                                                      |

Write new state-specific parsers in the same style — a JSON blob becomes per-key text columns in raw and per-key counts in full; a structured vacancy or waitlist string becomes its parts in raw and numbers in full.

Two dtype rules, both load-bearing:

- extracted **counts** use the nullable `Int64` dtype, so a parse failure stays distinguishable from a true zero;
- **booleans** use the nullable `"boolean"` dtype during engineering, then `finalize()` casts them to `Int64` for `full`, because nullable booleans serialise to `'True'`/`'False'` text and would break the CSV round-trip's numeric guarantee.

**Naming.** Reproduce the source's own naming rather than imposing a slug. Georgia deliberately does _not_ normalise column names to `[a-z0-9]` — its schema carries `#_of_rooms_…`, `staff:child_ratios…`, `activities_scouting_(boy_scouts/girl_scouts)` and apostrophes, and each builder reproduces that punctuation so the engineered names line up with `ga_columns.json`. Slugging everything is equally defensible. Pick one, apply it consistently, and say which in the module docstring.

---

## 6. `finalize()` — the shared tail

One function, identical sequence for all four scripts:

```python
def finalize(df, which, scaffold, log, *, keep_invalid_target=False):
```

1. **rename** the grain → `provider_id` and the target → `qr_rating` via `RENAME_MAP`.
2. **coerce the target.** `pd.to_numeric(errors="coerce")`, then test membership in `VALID_TARGET_VALUES`. Log the count of out-of-range or garbage ratings. For `complete`, keep the coerced value; for standard, blank the invalid ones.
3. **drop `NON_FEATURE_COLS`** as a safety net (most were removed pre-engineering).
4. **normalize** any name variants that would stop an engineered column from matching the scaffold (Georgia normalizes apostrophe characters).
5. **reindex** to `scaffold[which]` — the scaffold columns that are present, in scaffold order — **plus** any discovered dynamic columns whose names start with `DISCOVERED_PREFIXES[which]`, sorted, appended after. Georgia reindexes strictly (`DISCOVERED_PREFIXES = {"full": (), "raw": ()}`) because its scaffold enumerates every expected engineered column. A state whose builders discover categories at runtime should list those prefixes instead, choosing per variant to match the raw-vs-full intent.
6. **drop `TRAINING_EXCLUDE[which]`.**
7. **dedup** on the grain, `keep="first"`, logging how many rows went.
8. **row filter.** Standard: drop rows whose `qr_rating` is missing. Complete: keep them, log how many carry `None`. Then cast `qr_rating` to `Int64`. This step is the only difference between a standard script and its complete sibling.
9. **drop all-NaN and constant columns**, never `provider_id` or `qr_rating`. The constant test differs by variant: `full` counts a column as constant at `nunique(dropna=True) <= 1`; `raw` uses `nunique(dropna=False) <= 1`, so a column whose present/absent pattern varies counts as informative — that is what keeps sparse decomposed text columns alive.
10. **for `full` only**, cast every surviving boolean column to `Int64`.

---

## 7. Building `{st}_columns.json`

Derive it from the column-name + unique-values digest, then freeze it:

1. work out what each builder will produce and list every resulting column
2. sort into the two lists — `raw` gets the text-preserving columns, `full` gets the numeric/boolean ones. A field usually appears in both, under different names
3. `provider_id`, `qr_rating` first in both, then a stable, readable order (Georgia groups by source section)
4. leave out anything you want dropped — the reindex in `finalize()` step 5 is what actually enforces the schema, so omission is the drop mechanism for text byproducts in `full`
5. anything discovered at runtime does **not** go in the scaffold; it comes in through `DISCOVERED_PREFIXES`

---

## 8. Style rules

- Module docstring on every file explaining the state's quirks and the approach: what each variant is for, the builder order, and any naming decision. Georgia's four driver docstrings each state explicitly how that script differs from its siblings.
- Comments explain why, not what.
- Discover schema **from the data**, not from a hardcoded value list, wherever Georgia does. Only the stable scaffold lives in `{st}_columns.json`.
- Keep the four scripts' differences minimal and intentional.
- Never raise on a parse failure. `ParseLog` collects, prints, and writes at the end.

---

## 9. Validation

The four scripts are run for you. Confirm from the result that:

- `provider_id` and `qr_rating` exist, are the first two columns, and are populated.
- `provider_id` lost no leading zeros and is unique after dedup.
- Standard outputs contain **only** valid ratings; complete outputs contain every row and are a strict row-superset of their siblings.
- `raw` keeps readable text; `full` is entirely numeric/boolean (the driver's own sanity check should have reported nothing).
- Row counts are sane against the expected provider population from stage 1, and the raw/full pair is row-aligned.
- Every column in each output traces to a column in `{st}_records_anonymized.csv`, and every informative input column either appears in an output or is named in `NON_FEATURE_COLS` with a reason.

---

## 10. Pause-and-ask protocol

You write this in one pass, but that does not mean guessing. Before you finalize, ask about any **ambiguous fields**:

- which native column is the true licensing ID versus a facility or location number
- which column is the official rating versus a component score
- what the valid rating values are, per the published scale
- how to interpret a coded or loosely-formatted field
- whether any discovered field should be dropped rather than engineered

If while reasoning you hit a fork that changes the output, **stop and ask** rather than assuming. Never guess on anything affecting `provider_id`, `qr_rating`, row counts, or which providers are in scope.
