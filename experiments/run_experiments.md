# Running the experiments

Every command needed to go from a clean machine to the full result set. We assume that you are in the `experiments/` sub-directory and that the prerequisites described below are met.

---

## Contents

- [Running the experiments](#running-the-experiments)
    - [Contents](#contents)
    - [0. Prerequisites](#0-prerequisites)
    - [1. Environment Setup](#1-environment-setup)
    - [2. Model Checkpoint and Credentials](#2-model-checkpoint-and-credentials)
        - [Qwen3-4B (required for the LLM experiments)](#qwen3-4b-required-for-the-llm-experiments)
        - [TabPFN token (required only for the cross-state transfer experiment)](#tabpfn-token-required-only-for-the-cross-state-transfer-experiment)
        - [Scratch space (optional, but set it on a cluster)](#scratch-space-optional-but-set-it-on-a-cluster)
    - [3. Build the Fold Caches](#3-build-the-fold-caches)
    - [4. Within-State Experiments](#4-within-state-experiments)
    - [5. Cross-State Transfer Experiments](#5-cross-state-transfer-experiments)
        - [Experiment 1 — the eleven five-level states](#experiment-1--the-eleven-five-level-states)
        - [Experiment 2 — Georgia as the target with reconciled five-level sources](#experiment-2--georgia-as-the-target-with-reconciled-five-level-sources)
    - [6. Feature attribution (SHAP)](#6-feature-attribution-shap)
    - [7. Cross-rubric transfer experiment](#7-cross-rubric-transfer-experiment)
        - [The within-state ceiling](#the-within-state-ceiling)
    - [8. Runtime and running a subset](#8-runtime-and-running-a-subset)

---

## 0. Prerequisites

**Data:** `data/` must contain two views per state, for twelve states (CA CO GA KY MD MT NC NE OK SC WA WI). Each CSV needs a `provider_id` column and an ordinal `qr_rating` column, where every other column is treated as a feature.

```
data/<state>_records_cleaned_full.csv   preprocessed numeric feature table
data/<state>_records_cleaned_raw.csv    raw record, serialized to text at load time
```

**Hardware:** the tabular baselines and the SHAP experiment are CPU-only. Everything involving ModernBERT or Qwen3-4B needs a CUDA GPU. The LLM experiments want ~40 GB of VRAM at the default batch size, and you can lower `CCQ_LLMCLS_BATCH` if you have less.

**A note on scales.** Georgia rates on `{1,2,3}` and the other eleven states rate on `{1..5}`. Every command below therefore passes `--rating-scale 3star` for GA and `5star` for everyone else. Georgia has no 5-star map, so `--rating-scale 5star` on GA is a deliberate hard error.

---

## 1. Environment Setup

One conda environment serves all experiments.

```bash
conda create -n ccq python=3.11 -y
conda activate ccq

# Install vLLM FIRST so it pins the torch version, then keep the CUDA build
# matched to your driver. Installing torch without the index URL pulls a wheel
# built for a different CUDA, and the GPU then silently goes unused.
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
conda install -y -c conda-forge "ffmpeg>=6,<8"   # torchcodec runtime dependency
```

Verify GPU visibility on the machine that will run the jobs (on a cluster, that
means a compute node, not the login node):

```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

Two version pins are intentional rather than incidental:

- `xgboost>=2.0,<3` — xgboost 3.x serialises `base_score` in a form the SHAP tree loader cannot parse, so `TreeExplainer` raises after every fit.
- `shap>=0.44` — needed to parse XGBoost 2.x trees.

---

## 2. Model Checkpoint and Credentials

### Qwen3-4B (required for the LLM experiments)

The LLM experiments load the checkpoint from local disk, because compute nodes usually have no outbound network access during a job. Fetch it once:

```bash
python download_qwen.py                       # -> models/qwen3_4b, the default
export LLM_CLS_MODEL=./models/qwen3_4b
```

On a cluster, put the weights on scratch instead and point the variable at them:

```bash
python download_qwen.py --dest /path/on/scratch/qwen3_4b
export LLM_CLS_MODEL=/path/on/scratch/qwen3_4b
```

Every command below that touches an LLM reads `$LLM_CLS_MODEL`. Resolution order is `--model`, then `$LLM_CLS_MODEL`, then `models/qwen3_4b` in the repository. So if you accept the default download location, you can skip the export entirely.

### TabPFN token (required only for the cross-state transfer experiment)

TabPFN's **regressor** weights are licence-gated and cannot prompt inside a batch job. Accept the licence at <https://ux.priorlabs.ai>, copy the API key, and either export it or place it in a gitignored `keys.json`:

```bash
export TABPFN_TOKEN=<key>
# or:  echo '{"TABPFN_TOKEN": "<key>"}' > keys.json
```

The environment variable wins over the file, a missing file is a no-op, and the value is never printed. The classifier used by the cross-state suite is a different, freely downloadable checkpoint and needs no token.

### Scratch space (optional, but set it on a cluster)

Model checkpoints, embedding caches and the HuggingFace download cache are written outside the results tree, defaulting to `artifacts/` inside the repository. That works out of the box, but the tree grows to tens of GB during a sweep, so on a cluster point it at scratch rather than a quota-limited home volume:

```bash
export CCQ_ARTIFACT_ROOT=/path/with/room/ccq_experiments
```

Everything under it is regenerable and can be deleted between runs.

---

## 3. Build the Fold Caches

Every experiment indexes into a cached 5-fold split, so this must run before anything else.

```bash
mkdir -p fold_indices
for s in ca ga nc wi co ky md mt ne ok sc wa; do
  S=$(echo "$s" | tr a-z A-Z)
  SC=5star; [ "$S" = "GA" ] && SC=3star
  python utils.py \
    --input data/${s}_records_cleaned_full.csv \
    --folds fold_indices/${s}_native_folds.json \
    --remap-state "$S" --rating-scale "$SC"
done
```

**Regenerate these whenever `data/` changes.** Fold indices are positional, and the cache is reused whenever the row count still matches. If a data refresh reorders rows without changing their number, a re-export or a reshuffle will do exactly that, the stale folds are silently reused against a different permutation, and every downstream number is quietly different. Add `--overwrite` to force a rebuild:

```bash
python utils.py --input data/mt_records_cleaned_full.csv \
  --folds fold_indices/mt_native_folds.json \
  --remap-state MT --rating-scale 5star --overwrite
```

---

## 4. Within-State Experiments

Full in-state supervision: 5-fold CV per state at that state's native rating granularity.

```bash
DATE=$(date +%F)
mkdir -p results/$DATE/within_state

for s in ca ga nc wi co ky md mt ne ok sc wa; do
  S=$(echo "$s" | tr a-z A-Z)
  SC=5star; [ "$S" = "GA" ] && SC=3star
  FULL=data/${s}_records_cleaned_full.csv
  RAW=data/${s}_records_cleaned_raw.csv
  FOLDS=fold_indices/${s}_native_folds.json
  RES=results/$DATE/within_state/experiment_within_state_${s}_results.csv
  COMMON=(--folds "$FOLDS" --output "$RES" --remap-state "$S" --rating-scale "$SC")

  # Classical baselines on the preprocessed feature table
  python baselines_ml.py --models dummy lr rf xgb --input "$FULL" "${COMMON[@]}"

  # TabNet + TabPFN on the preprocessed feature table
  python tabular_dl.py --features numeric --models tabnet tabpfn_balanced \
      --input "$FULL" "${COMMON[@]}"

  # The same two families on MiniLM embeddings of the serialized text ("_raw" tags)
  python tabular_dl.py --features text --method-suffix raw --compliance verbose \
      --models tabnet tabpfn_balanced --input "$RAW" "${COMMON[@]}"
  python baselines_ml.py --features text --method-suffix raw --compliance verbose \
      --models dummy lr rf xgb --input "$RAW" "${COMMON[@]}"

  # ModernBERT on the serialized text
  python llm.py --models bert_textualized_full --compliance verbose \
      --input "$RAW" "${COMMON[@]}"

  # Qwen3-4B classification-head trio (frozen head / + RAG exemplars / LoRA)
  for VARIANT in "--adapt head" "--adapt head --rag" "--adapt lora"; do
    python transfer_llm_cls.py --within-cv 5 $VARIANT --compliance verbose \
        --model "$LLM_CLS_MODEL" \
        --source "$RAW" --target "$RAW" --src-name "$S" --tgt-name "$S" \
        --rating-scale "$SC" --output "$RES"
  done
done
```

**Output:** `results/<date>/within_state/experiment_within_state_<state>_results.csv`,
one per state.

To run only part of it, drop the blocks you do not want, or narrow `--models`.

---

## 5. Cross-State Transfer Experiments

Leave-one-state-out: each state is held out in turn while the rest are pooled as the source, over a target-supervision curve p ∈ {0, 20, 40, 60, 80, 100}.

The split protocol is worth understanding before reading any cell. A stratified 20% slice of the target is carved once and is the scored block at every p, which is what makes a QWK column comparable down the curve. `--target-frac` is then p% of the remaining 80%, so p=20 adapts on 16% of the target and p=100 on 80% of it. p=0 is genuine zero-shot: no target label reaches training or model selection.

### Experiment 1 — the eleven five-level states

```bash
DATE=$(date +%F)
mkdir -p results/$DATE/5_star/loso_few_shot
FIVE="nc wi ca co ky md mt ne ok sc wa"

for TGT in $FIVE; do
  T=$(echo "$TGT" | tr a-z A-Z)
  OUT=results/$DATE/5_star/loso_few_shot/experiment_loso_few_shot_5star_${TGT}_results.csv

  # Pool = every five-level state except the held-out target
  POOL=()
  for s in $FIVE; do
    [ "$s" = "$TGT" ] && continue
    POOL+=(--pool-sources "$(echo "$s" | tr a-z A-Z)=data/${s}_records_cleaned_raw.csv")
  done
  BASE=("${POOL[@]}" --target "data/${TGT}_records_cleaned_raw.csv" --tgt-name "$T"
        --rating-scale 5star --output "$OUT")

  for P in 0 20 40 60 80 100; do
    F=$(awk -v p="$P" 'BEGIN{printf "%.10g", p/100}')
    SPLIT=(--target-frac "$F" --target-test-frac 0.2)

    # One-shot fit on pooled source + the p% target draw
    python transfer_tabular.py "${BASE[@]}" "${SPLIT[@]}" \
        --models dummy lr rf xgb tabpfn_balanced

    # Trainable families. At p=0 there is no second phase, so no sequential flag.
    if [ "$P" = "0" ]; then
      python transfer_tabular.py  "${BASE[@]}" "${SPLIT[@]}" --models tabnet
      python transfer_plm.py      "${BASE[@]}" "${SPLIT[@]}"
      python transfer_plm_corn.py "${BASE[@]}" "${SPLIT[@]}"
    else
      python transfer_tabular.py  "${BASE[@]}" "${SPLIT[@]}" --transfer-mode sequential --models tabnet
      python transfer_plm.py      "${BASE[@]}" "${SPLIT[@]}" --transfer-mode sequential
      python transfer_plm_corn.py "${BASE[@]}" "${SPLIT[@]}" --transfer-mode sequential
    fi

    # Qwen trio, at every curve point
    for VARIANT in "--adapt head" "--adapt head --rag" "--adapt lora"; do
      python transfer_llm_cls.py "${BASE[@]}" "${SPLIT[@]}" \
          $VARIANT --model "$LLM_CLS_MODEL"
    done
  done
done
```

### Experiment 2 — Georgia as the target with reconciled five-level sources

Georgia is natively 3-level, so the pool is collapsed 5→3 to meet it. `--rating-scale 3star` is what performs that collapse.

```bash
mkdir -p results/$DATE/3_star/loso_ga_few_shot
OUT_GA=results/$DATE/3_star/loso_ga_few_shot/experiment_loso_ga_few_shot_3star_results.csv

POOL=()
for s in $FIVE; do
  POOL+=(--pool-sources "$(echo "$s" | tr a-z A-Z)=data/${s}_records_cleaned_raw.csv")
done
BASE=("${POOL[@]}" --target data/ga_records_cleaned_raw.csv --tgt-name GA
      --rating-scale 3star --output "$OUT_GA")

for P in 0 20 40 60 80 100; do
  F=$(awk -v p="$P" 'BEGIN{printf "%.10g", p/100}')
  SPLIT=(--target-frac "$F" --target-test-frac 0.2)
  python transfer_tabular.py "${BASE[@]}" "${SPLIT[@]}" --models dummy lr rf xgb tabpfn_balanced
  if [ "$P" = "0" ]; then
    python transfer_tabular.py  "${BASE[@]}" "${SPLIT[@]}" --models tabnet
    python transfer_plm.py      "${BASE[@]}" "${SPLIT[@]}"
    python transfer_plm_corn.py "${BASE[@]}" "${SPLIT[@]}"
  else
    python transfer_tabular.py  "${BASE[@]}" "${SPLIT[@]}" --transfer-mode sequential --models tabnet
    python transfer_plm.py      "${BASE[@]}" "${SPLIT[@]}" --transfer-mode sequential
    python transfer_plm_corn.py "${BASE[@]}" "${SPLIT[@]}" --transfer-mode sequential
  fi
  for VARIANT in "--adapt head" "--adapt head --rag" "--adapt lora"; do
    python transfer_llm_cls.py "${BASE[@]}" "${SPLIT[@]}" $VARIANT --model "$LLM_CLS_MODEL"
  done
done
```

---

## 6. Feature attribution (SHAP)

CPU-only, do not put this on a GPU. This is the one place in the suite that fits a regressor rather than the K-way classifier, because `multi:softprob` trains K separate ensembles and SHAP would return one attribution table per class per state. `--qwk-check` verifies the regressor is a fair stand-in for the classifier it explains rather than a quietly weaker model.

```bash
DATE=$(date +%F)
mkdir -p results/$DATE/shap_xgb

for s in ca ga nc wi co ky md mt ne ok sc wa; do
  S=$(echo "$s" | tr a-z A-Z)
  SC=5star; [ "$S" = "GA" ] && SC=3star
  CMP=results/$DATE/within_state/experiment_within_state_${s}_results.csv

  CMP_ARG=()
  [ -f "$CMP" ] && CMP_ARG=(--compare-results "$CMP")

  python shap_xgb.py \
      --input data/${s}_records_cleaned_full.csv \
      --output results/$DATE/shap_xgb/shap_xgb_${s}_results.csv \
      --folds fold_indices/${s}_native_folds.json \
      --remap-state "$S" --rating-scale "$SC" \
      --top-k 10 --n-shadow 10 "${CMP_ARG[@]}"
done

# Paper figure: the 3x4 grid, one panel per state. Run after all twelve finish.
python shap_xgb.py --grid-only --output results/$DATE/shap_xgb/grid.csv
```

`--compare-results` is optional and only supplies the QWK delta against the reported `xgb` row, so run §4 first if you want it.

---

## 7. Cross-rubric transfer experiment

Georgia's three levels and the other states' five have no defensible correspondence, so this experiment never constructs one. A regressor trains on the source's native rating, predicts a continuous latent-quality score on the target, and is scored by concordance index against the target's native labels. Rank association is invariant to any monotone reparameterisation of either scale, so GA→WI and WI→GA are directly comparable with no bridge, cutpoints or rounding.

```bash
DATE=$(date +%F)
mkdir -p results/$DATE/cross_scale
FIVE="ca co ky md mt nc ne ok sc wa wi"
MODELS="dummy ridge rf xgb tabnet tabpfn mbert qwen qwen_rag qwen_lora"
CACHE=${CCQ_ARTIFACT_ROOT:-./artifacts}/cross_scale/embeddings

for s in $FIVE; do
  for pair in "ga:$s" "$s:ga"; do
    SRC="${pair%%:*}"; TGT="${pair##*:}"
    for P in 0 20 40 60 80 100; do
      F=$(awk -v p="$P" 'BEGIN{printf "%.10g", p/100}')
      python cross_scale.py \
          --source data/${SRC}_records_cleaned_full.csv --src-name "$(echo $SRC | tr a-z A-Z)" \
          --target data/${TGT}_records_cleaned_full.csv --tgt-name "$(echo $TGT | tr a-z A-Z)" \
          --target-frac "$F" --target-test-frac 0.2 \
          --models $MODELS --cache-dir "$CACHE" --llm-model "$LLM_CLS_MODEL" \
          --n-bootstrap 1000 --seed 42 --dump-predictions \
          --output results/$DATE/cross_scale/experiment_cross_scale_${SRC}2${TGT}_results.csv
    done
  done
done
```

### The within-state ceiling

Every transfer cell above is read against this, so it is not optional. Run it as a separate pass. It is per-state, not per-pair, and folding it into the grid loop would recompute and re-append each state's ceiling once per pair.

```bash
for s in ga $FIVE; do
  python cross_scale.py \
      --within data/${s}_records_cleaned_full.csv --state "$(echo $s | tr a-z A-Z)" \
      --folds fold_indices/${s}_native_folds.json \
      --models $MODELS --cache-dir "$CACHE" --llm-model "$LLM_CLS_MODEL" \
      --n-bootstrap 1000 --seed 42 \
      --output results/$DATE/cross_scale/experiment_cross_scale_within_${s}_results.csv
done
```

---

## 8. Runtime and running a subset

**Results CSVs append and never deduplicate.** Re-running a cell adds a second row rather than replacing the first, so aggregate by method tag and take the latest, and use a fresh `$DATE` for a full re-run.

Rough per-state or per-target costs on one modern GPU:

| Block                           | Cost                                 |
| ------------------------------- | ------------------------------------ |
| Classical baselines, both views | seconds                              |
| TabNet / TabPFN                 | seconds to minutes                   |
| ModernBERT                      | minutes                              |
| Qwen3-4B trio                   | the dominant cost — hours per target |
| SHAP, per state                 | minutes, CPU                         |
| Cross-rubric, tabular models    | seconds per cell                     |
| Cross-rubric, `qwen*`           | hours per cell                       |

The suite is extremely parallel: within-state and SHAP are per-state, cross-state is per-held-out-target, and cross-rubric is per-ordered-pair. Each writes its own CSV, so those units can run concurrently on separate machines with no coordination.

To reproduce the tabular results only — no GPU required anywhere — restrict every `--models` list to `dummy lr rf xgb` (or `dummy ridge rf xgb` for the cross-rubric experiment) and skip the ModernBERT and Qwen blocks entirely.
