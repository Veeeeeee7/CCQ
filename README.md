# CCQ — Child Care Quality Dataset

Paper Link: [link]()

A child care quality dataset constructed for and provider quality-rating prediction across twelve US states: **CA, CO, GA, KY, MD, MT, NC, NE, OK, SC, WA, WI**.

| state | rating system             | scale      |  providers |      rated | raw cols | full cols |
| ----- | ------------------------- | ---------- | ---------: | ---------: | -------: | --------: |
| CA    | Quality Counts California | 1–5        |     12,530 |        975 |      170 |        89 |
| CO    | Colorado Shines           | 1–5        |      4,508 |      3,423 |       51 |        76 |
| GA    | Quality Rated             | 1–3        |      8,192 |      2,897 |      203 |       280 |
| KY    | Kentucky All STARS        | 1–5        |      1,929 |      1,896 |       61 |        59 |
| MD    | Maryland EXCELS           | 1–5        |      7,106 |      5,005 |       28 |        27 |
| MT    | Best Beginnings STARS     | 1–5        |        213 |        181 |       15 |        22 |
| NC    | NC Star Rated License     | 1–5        |      6,550 |      3,050 |       30 |        32 |
| NE    | Step Up to Quality        | 1–5        |      3,274 |      1,079 |       40 |        55 |
| OK    | Reaching for the Stars    | 1–5        |      2,557 |      2,507 |      178 |        28 |
| SC    | ABC Quality               | C–A+ → 1–5 |      2,406 |      1,221 |       12 |        38 |
| WA    | Early Achievers           | 2–5        |     10,399 |      3,168 |      177 |       120 |
| WI    | YoungStar                 | 1–5        |      4,815 |      3,750 |      161 |        32 |
|       |                           |            | **64,479** | **29,152** |          |           |

---

## Repository Layout

```
README.md
.gitignore
data-collection/  # Phase 1: data collection and cleaning, one subdirectory per state
experiments/  # Phase 2: experiments, detailed in paper
```

Each phase has its own dependencies and its own entry points. Nothing in `experiments/` imports anything from `data-collection/`. The only thing that crosses the boundary is a set of CSV files. We recommend creating a virtual environment for each phase.

---

## Phase 1 — Data Collection

`data-collection/` holds one self-contained directory per state, plus the prompts
each stage was specified from.

```
data-collection/
  {st}/
    {st}.md                      how to run this state, start to finish
    {st}_capture.py              optional: dump rendered DOM for selector work
    {st}_crawler.py              stage 1
    {st}_anonymize.py            stage 2
    {st}_columns.json            output scaffold: canonical column list, in order
    {st}_cleaning_utils.py       feature builders + shared finalize()
    {st}_clean_raw.py            stage 3 ─┐
    {st}_clean_full.py                    ├ four outputs
    {st}_clean_complete_raw.py            │
    {st}_clean_complete_full.py          ─┘
    {st}_data/                   seed + records + anonymized + four cleaned
  requirements.txt               python dependencies shared across states

  prompts/
    01_collection_prompt.md
    02_anonymization_prompt.md
    03_cleaning_prompt.md

  private/                       NOT FOR RELEASE
    provider_id_map_{st}.csv     surrogate -> real id
    geo_label_map_{st}.json      geography pseudonymization (7 states)
```

### The three stages

```
1. COLLECTION     {st}_seed.csv               ── {st}_crawler.py ──► {st}_records.csv
                                              │  ← human verification
2. ANONYMIZATION  {st}_records.csv            ── {st}_anonymize.py ──► {st}_records_anonymized.csv
                                              │  ← human verification
3. CLEANING       {st}_records_anonymized.csv ── {st}_clean_*.py ──► four cleaned tables
```

The stages know nothing about each other: a crawler never mentions cleaning, a cleaning script never mentions crawling, and only the CSVs cross a boundary. Stage 2 is the only one that ever sees identifying information. By stage 3, names, addresses, coordinates, phone numbers and contact details are gone, and `provider_id` has been replaced by a surrogate. The mapping in `private/provider_id_map_{st}.csv` is the only link back to real providers and isn't released.

Seeds are shipped with the repository as `{st}_data/{st}_seed.csv`, so there is nothing to enumerate before you start. Some are a bare list of ids while others carry more attributes.

### Running a state

`{st}.md` is the guide for running each state's data collection phase and differs across states. The general shape of each state's suite is as follows:

```bash
cd data-collection/{st}
python {st}_crawler.py                 # stage 1
python {st}_anonymize.py --dry-run     # review the drop list
python {st}_anonymize.py               # stage 2
python {st}_clean_raw.py               # stage 3
python {st}_clean_full.py
python {st}_clean_complete_raw.py
python {st}_clean_complete_full.py
```

Environment Requirements: Python 3.10+ and install via `pip install -r requirements.txt`

### Four outputs per state

Two orthogonal axes, giving four files:

|                       | rated providers only            | every provider                           |
| --------------------- | ------------------------------- | ---------------------------------------- |
| **text preserved**    | `{st}_records_cleaned_raw.csv`  | `{st}_records_cleaned_complete_raw.csv`  |
| **numeric / boolean** | `{st}_records_cleaned_full.csv` | `{st}_records_cleaned_complete_full.csv` |

**raw** keeps human-readable text, only removing simple prefixes stripped. **full** is strictly numeric and boolean: multi-value and categorical fields become one-hot encodings.

**standard** holds only providers with a valid rating; **complete** holds every provider, including unrated ones.

Raw and full are row-aligned within a filter, so one fold assignment applies to both. The `complete_*` scripts differ from their siblings by a single `finalize()` argument.

The experiments use only the **standard** outputs from `experiments/data/` and can be copied over via commands:

```bash
mkdir -p experiments/data
cp data-collection/*/*_data/*_records_cleaned_full.csv experiments/data/
cp data-collection/*/*_data/*_records_cleaned_raw.csv  experiments/data/
```

---

## Phase 2 — Experiments

`experiments/` includes all the Python scripts for running the four experiments. Every module is either a method, shared plumbing, or a tool. The instructions for running these experiments can be found in `run_experiments.md`. The data directory is `experiments/data/` and contains the 24 cleaned CSVs from Phase 1.

```
experiments/
  run_experiments.md    ← every command, environment setup onward
  requirements.txt
  data/                 ← 24 cleaned CSVs from Phase 1
  *.py
```

### Getting Started

`run_experiments.md` is the starting point for recreating the results from the paper. It gives the exact commands, in order, from a clean machine to the full result set: environment setup, model checkpoint, fold caches, then one runnable block per experiment.

**Hardware:** the tabular baselines and the SHAP experiment are CPU-only. Anything involving ModernBERT or Qwen3-4B needs a CUDA GPU.

### Codebase

Helper modules, shared across experiments:

| Module                  | Role                                                                   |
| ----------------------- | ---------------------------------------------------------------------- |
| `utils.py`              | Data loading, leakage guard, fold caching, metrics, results logging    |
| `transfer_common.py`    | Rating-scale reconciliation, split protocols, class weights, bootstrap |
| `text_serialization.py` | Row-to-text serialization                                              |
| `ranking_metrics.py`    | Concordance index and companions                                       |
| `llm_common.py`         | Exemplar selection for the RAG variants                                |
| `download_qwen.py`      | Fetch the Qwen3-4B checkpoint to local disk                            |
| `keys.py`               | Load optional API tokens from a gitignored `keys.json`                 |

Methods, grouped by the experiment that uses them:

| Module                 | Experiment                | Method                                                       |
| ---------------------- | ------------------------- | ------------------------------------------------------------ |
| `baselines_ml.py`      | within-state, cross-state | Dummy, logistic regression, random forest, gradient boosting |
| `tabular_dl.py`        | within-state              | TabNet, TabPFN                                               |
| `llm.py`               | within-state              | ModernBERT — also holds the shared HuggingFace plumbing      |
| `transfer_tabular.py`  | cross-state               | TabNet, TabPFN over sentence embeddings                      |
| `transfer_plm.py`      | cross-state               | ModernBERT                                                   |
| `transfer_plm_corn.py` | cross-state               | ModernBERT with a rank-consistent ordinal head               |
| `transfer_llm_cls.py`  | cross-state, within-state | Qwen3-4B classification head (frozen / RAG / LoRA)           |
| `shap_xgb.py`          | attribution               | TreeSHAP over an XGBoost regressor                           |
| `cross_scale.py`       | cross-rubric              | Tabular regressors, scored by rank                           |
| `cross_scale_deep.py`  | cross-rubric              | TabNet, TabPFN, ModernBERT, Qwen3-4B, as regressors          |

### Default Output Directories

Every default resolves inside the repository, so a fresh checkout runs with no configuration. Each is environment-overridable.

| What                | Default              | Override                            |
| ------------------- | -------------------- | ----------------------------------- |
| Data                | `./data/`            | `--input` / `--source` / `--target` |
| Fold Cache          | `./fold_indices/`    | `--folds`                           |
| Results             | `./results/{DATE}/`  | `--results`, `--date`               |
| Logs                | `./logs/{DATE}/`     | `--logs`, `--date`                  |
| Artifacts           | `./artifacts/`       | `CCQ_ARTIFACT_ROOT`                 |
| Qwen3-4B Checkpoint | `./models/qwen3_4b/` | `LLM_CLS_MODEL`, or `--model`       |
