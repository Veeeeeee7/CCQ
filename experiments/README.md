# Experiments

Code for the CCQ benchmark: within-state QR prediction (Section 5.2), driving
factor analysis (Section 5.3), cross-state QR prediction (Section 5.4) and the
transfer utility analysis (Section 5.5). Hyperparameters are fixed to Table 3.

## Setup

```bash
conda create -n ccq python=3.11 -y && conda activate ccq
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements.txt
conda install -y -c conda-forge "ffmpeg>=6,<8"
```

- **Data.** Place the 24 released files in `data/`:
  `{st}_records_cleaned_full.csv` (*preprocessed*) and
  `{st}_records_cleaned_raw.csv` (*raw*) for `ca co ga ky md mt nc ne ok sc wa wi`.
- **Qwen3-4B** (LLM-CLS / LLM-RAG / LLM-LoRA): `python download_qwen.py` saves it to
  `models/qwen3_4b`. Use `--dest` and `export LLM_CLS_MODEL=<dir>` for another location.
- **TabPFN** weights are licence-gated: accept the licence at
  <https://ux.priorlabs.ai> and `export TABPFN_TOKEN=<key>`.
- Checkpoints and caches go to `artifacts/`; set `CCQ_ARTIFACT_ROOT` to move them.
- Tabular models and SHAP run on CPU. MBERT, MBERT-CORN and the LLM variants need
  a CUDA GPU (~40 GB for the LLMs at the default batch size).

## Files

| File | Contents |
|---|---|
| `utils.py` | Data loading, fold caching, metrics, results logging; builds the fold cache |
| `transfer_common.py` | Rating-scale maps, the leave-one-state-out split, class weights, bootstrap |
| `text_serialization.py` | Textualizer: serializes a *raw* record to text (PedCA-FT style) |
| `llm_common.py` | Medoid exemplar selection for LLM-RAG |
| `baselines_ml.py` | Within-state Dummy, LR, RF, XGB |
| `tabular_dl.py` | Within-state TabNN, TabPFN |
| `llm.py` | Within-state MBERT |
| `transfer_tabular.py` | Cross-state Dummy, LR, RF, XGB, TabNN, TabPFN on sentence embeddings |
| `transfer_plm.py` | Cross-state MBERT |
| `transfer_plm_corn.py` | Cross-state MBERT-CORN |
| `transfer_llm_cls.py` | LLM-CLS, LLM-RAG and LLM-LoRA, cross-state and within-state (`--within-cv`) |
| `shap_xgb.py` | Driving factor analysis: XGB-reg, TreeSHAP and shadow-feature noise floor |
| `download_qwen.py` | Downloads the Qwen3-4B checkpoint |

`-tab` methods read the *preprocessed* files; `-txt` methods read the *raw* files
(tabular models via all-MiniLM-L6-v2 embeddings). In the results, `lr` is LR-tab
and `lr_raw` is LR-txt (likewise for the other tabular models; `tabnet` is TabNN,
`tabpfn_balanced` is TabPFN). `bert_textualized_full_verbose` / `xfer_bert_textualized_full_verbose*`
is MBERT, `xfer_bert_corn*` is MBERT-CORN, and `qwen_cls*`, `qwen_cls_rag*`,
`qwen_cls_lora*` are LLM-CLS, LLM-RAG and LLM-LoRA.

Every script takes `--results` / `--logs` (default `results/`, `logs/`) and an
optional `--date` subfolder. Georgia uses `--rating-scale 3star`; every other
state uses `5star`.

## 1. Fold cache (run first)

```bash
for s in ca co ga ky md mt nc ne ok sc wa wi; do
  S=$(echo $s | tr a-z A-Z); SC=5star; [ "$S" = GA ] && SC=3star
  python utils.py --input data/${s}_records_cleaned_full.csv \
    --folds fold_indices/${s}_native_folds.json --remap-state $S --rating-scale $SC
done
```

## 2. Within-state QR prediction

```bash
for s in ca co ga ky md mt nc ne ok sc wa wi; do
  S=$(echo $s | tr a-z A-Z); SC=5star; [ "$S" = GA ] && SC=3star
  FULL=data/${s}_records_cleaned_full.csv; RAW=data/${s}_records_cleaned_raw.csv
  C=(--folds fold_indices/${s}_native_folds.json --remap-state $S --rating-scale $SC)

  python baselines_ml.py --models dummy lr rf xgb --input $FULL "${C[@]}"
  python tabular_dl.py --models tabnet tabpfn_balanced --input $FULL "${C[@]}"
  python baselines_ml.py --features text --method-suffix raw --models dummy lr rf xgb --input $RAW "${C[@]}"
  python tabular_dl.py --features text --method-suffix raw --models tabnet tabpfn_balanced --input $RAW "${C[@]}"
  python llm.py --models bert_textualized_full --input $RAW "${C[@]}"
  for V in "--adapt head" "--adapt head --rag" "--adapt lora"; do
    python transfer_llm_cls.py --within-cv 5 $V --source $RAW --target $RAW \
      --src-name $S --tgt-name $S --rating-scale $SC
  done
done
```

## 3. Driving factor analysis

Run after step 2, since the QWK check compares against the within-state XGB-tab row.

```bash
for s in ca co ga ky md mt nc ne ok sc wa wi; do
  S=$(echo $s | tr a-z A-Z); SC=5star; [ "$S" = GA ] && SC=3star
  python shap_xgb.py --input data/${s}_records_cleaned_full.csv \
    --folds fold_indices/${s}_native_folds.json --remap-state $S --rating-scale $SC \
    --compare-results results/within_state/experiment_within_state_${s}_results.csv
done
python shap_xgb.py --grid-only        # Figure 6
```

## 4. Cross-state QR prediction

Leave-one-state-out over the target supervision levels p = 0, 16, 32, 48, 64, 80
(% of the target). `--target-frac` is p as a share of the 80% non-test pool, so
the six levels are `0 0.2 0.4 0.6 0.8 1`.

```bash
FIVE="ca co ky md mt nc ne ok sc wa wi"
run_target () {   # $1 = target state, $2 = rating scale, remaining = pool states
  T=$1; SC=$2; shift 2
  POOL=(); for s in "$@"; do POOL+=(--pool-sources $(echo $s | tr a-z A-Z)=data/${s}_records_cleaned_raw.csv); done
  B=("${POOL[@]}" --target data/${T}_records_cleaned_raw.csv --tgt-name $(echo $T | tr a-z A-Z) --rating-scale $SC)
  for F in 0 0.2 0.4 0.6 0.8 1; do
    SEQ=(); [ "$F" != 0 ] && SEQ=(--transfer-mode sequential)
    python transfer_tabular.py "${B[@]}" --target-frac $F --models dummy lr rf xgb tabpfn_balanced
    python transfer_tabular.py "${B[@]}" --target-frac $F --models tabnet "${SEQ[@]}"
    python transfer_plm.py      "${B[@]}" --target-frac $F "${SEQ[@]}"
    python transfer_plm_corn.py "${B[@]}" --target-frac $F "${SEQ[@]}"
    for V in "--adapt head" "--adapt head --rag" "--adapt lora"; do
      python transfer_llm_cls.py "${B[@]}" --target-frac $F $V
    done
  done
}

for t in $FIVE; do run_target $t 5star $(echo $FIVE | tr ' ' '\n' | grep -vx $t); done
run_target ga 3star $FIVE     # GA target: five-level pool collapsed to three levels
```

The transfer utility analysis (Figure 7b) compares the within-state results
(step 2) with the p = 80 cross-state results; it needs no extra runs.

## Outputs

```
results/within_state/experiment_within_state_{st}_results.csv
results/5_star/loso_few_shot/experiment_loso_few_shot_5star_{tgt}_results.csv
results/3_star/loso_ga_few_shot/experiment_loso_ga_few_shot_3star_results.csv
results/shap_xgb/shap_xgb_{st}_results.csv, shap_xgb_qwk_check.csv, figs/
```

Result files are appended to, so re-running a cell adds a row; use a new `--date`
for a fresh run. Within-state rows report the mean over five folds; cross-state
rows are scored on the fixed 20% target test block with bootstrap standard
deviations.
