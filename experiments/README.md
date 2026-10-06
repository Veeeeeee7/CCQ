# Experiments

Code for the CCQ benchmark: within-state QR prediction (Section 5.2), driving
factor analysis (Section 5.3), cross-state QR prediction (Section 5.4) and the
transfer utility analysis (Section 5.5). Hyperparameters are fixed to Table 4.

## Setup

```bash
conda create -n ccq python=3.11 -y && conda activate ccq
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements.txt
conda install -y -c conda-forge "ffmpeg>=6,<8"
```

- **Data.** Download the 24 released files from <https://huggingface.co/datasets/GAIN-Lab/CCQ> (gated: request access first) and place them in `data/`:
  `{st}_records_cleaned_full.csv` (*preprocessed*) and
  `{st}_records_cleaned_raw.csv` (*raw*) for `ca co ga ky md mt nc ne ok sc wa wi`.
- **Qwen3-4B** (LLM-CLS / LLM-RAG / LLM-LoRA): `python download_qwen.py` saves it to
  `models/qwen3_4b`. Use `--dest` and `export LLM_CLS_MODEL=<dir>` for another location.
- **TabPFN** weights are licence-gated: accept the licence at
  <https://ux.priorlabs.ai> and `export TABPFN_TOKEN=<key>`.
- Checkpoints and caches go to `artifacts/`; set `CCQ_ARTIFACT_ROOT` to move them.
- Tabular models and SHAP run on CPU. MBERT, MBERT-CORN and the LLM variants need
  a CUDA GPU (~40 GB for the LLMs at the default batch size).
- The SHAP grid figure is typeset by LaTeX in the paper's font, which needs the
  `libertine` and `newtx` LaTeX packages (and `dvipng` for the PNG copy).

## Files

| File | Contents |
|---|---|
| `utils.py` | Data loading, fold caching, metrics, results logging; builds the fold cache |
| `transfer_common.py` | Rating-scale maps, the leave-one-state-out split, class weights, bootstrap |
| `text_serialization.py` | Textualizer: serializes a *raw* record to text (PedCA-FT style) |
| `llm_common.py` | Medoid exemplar selection for LLM-RAG |
| `baselines_ml.py` | Within-state Dummy, LR, RF, XGB |
| `tabular_dl.py` | Within-state TabNN, TabPFN |
| `llm.py` | Within-state MBERT and MBERT-CORN |
| `transfer_tabular.py` | Cross-state Dummy, LR, RF, XGB, TabNN, TabPFN on sentence embeddings |
| `transfer_plm.py` | Cross-state MBERT |
| `transfer_plm_corn.py` | Cross-state MBERT-CORN |
| `transfer_llm_cls.py` | LLM-CLS, LLM-RAG and LLM-LoRA, cross-state and within-state (`--within-cv`) |
| `shap_xgb.py` | Driving factor analysis: XGB-reg, TreeSHAP and shadow-feature noise floor |
| `shap_seed_check.py` | Seed stability of the SHAP noise floor |
| `count_tuned_params.py` | Tuned-parameter counts for the within-state Pareto figures |
| `elo.py` | Within-state and cross-state Elo scores with bootstrap confidence intervals |
| `download_qwen.py` | Downloads the Qwen3-4B checkpoint |

`-num` methods read the *preprocessed* files; `-txt` methods read the *raw* files
(tabular models via all-MiniLM-L6-v2 embeddings). In the results, `lr` is LR-num
and `lr_raw` is LR-txt (likewise for the other tabular models; `tabnet` is TabNN,
`tabpfn_balanced` is TabPFN). `bert_textualized_full_verbose` / `xfer_bert_textualized_full_verbose*`
is MBERT, `bert_corn_textualized_full_verbose` / `xfer_bert_corn*` is MBERT-CORN,
and `qwen_cls*`, `qwen_cls_rag*`, `qwen_cls_lora*` are LLM-CLS, LLM-RAG and LLM-LoRA.

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
  python llm.py --models bert_corn_textualized_full --input $RAW "${C[@]}"
  for V in "--adapt head" "--adapt head --rag" "--adapt lora"; do
    python transfer_llm_cls.py --within-cv 5 $V --source $RAW --target $RAW \
      --src-name $S --tgt-name $S --rating-scale $SC
  done
done
```

The tuned-parameter counts of the Pareto figures (Figure 6) come from
`python count_tuned_params.py`.

## 3. Driving factor analysis

Run after step 2, since the QWK check compares against the within-state XGB-num row.

```bash
for s in ca co ga ky md mt nc ne ok sc wa wi; do
  S=$(echo $s | tr a-z A-Z); SC=5star; [ "$S" = GA ] && SC=3star
  python shap_xgb.py --input data/${s}_records_cleaned_full.csv \
    --folds fold_indices/${s}_native_folds.json --remap-state $S --rating-scale $SC \
    --compare-results results/within_state/experiment_within_state_${s}_results.csv
done
python shap_xgb.py --grid-only        # Figure 7

# Seed stability of MT's noise floor, with and without class weights
for W in "" "--unweighted"; do
  python shap_seed_check.py --input data/mt_records_cleaned_full.csv --remap-state MT \
    --rating-scale 5star --n-seeds 30 $W \
    --output results/shap_xgb_seed_check/shap_xgb_seed_check_mt${W:+_unweighted}.csv
done
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

The transfer utility analysis (Figure 8b) compares the within-state results
(step 2) with the p = 80 cross-state results; it needs no extra runs.

## 5. Elo scores

Run after steps 2 and 4. Each state and metric (BA, QWK) is one match between
every pair of models, scored on the values as reported in the paper's tables
(percentages to two decimals). The script reports the mean Elo over 2,000
bootstrap rounds that resample the twelve states, with 95% intervals and RF-txt
anchored at 1,000. Cross-state Elo is computed at each supervision level and
averaged over the six levels.

```bash
python elo.py --results results --output results/elo    # Figure 5
```

With `--date`, point `--results` at `results/<date>`.

## Outputs

```
results/within_state/experiment_within_state_{st}_results.csv
results/5_star/loso_few_shot/experiment_loso_few_shot_5star_{tgt}_results.csv
results/3_star/loso_ga_few_shot/experiment_loso_ga_few_shot_3star_results.csv
results/shap_xgb/shap_xgb_{st}_results.csv, shap_xgb_qwk_check.csv, figs/
results/shap_xgb_seed_check/shap_xgb_seed_check_mt.csv
results/param_counts/param_counts_summary.csv
results/elo/elo_within_state.csv, elo_cross_state.csv
```

Result files are appended to, so re-running a cell adds a row; use a new `--date`
for a fresh run. Within-state rows report the mean over five folds; cross-state
rows are scored on the fixed 20% target test block with bootstrap standard
deviations.
