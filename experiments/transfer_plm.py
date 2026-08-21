"""ModernBERT for cross-state transfer over the serialized text view.

The MBERT arm of the cross-state experiment; the ordinal variant is in
`transfer_plm_corn`.

Two regimes. `pool` fits once on the source pool combined with whatever target
supervision is available. `sequential` pretrains on the pool and then fine-tunes
on the target adaptation rows, and is what the `_seq` tags report. At p=0 the two
coincide and only the pretrain phase runs; target labels are then read only to
score, and early stopping watches a held-out slice of the source.

Data handling is delegated to `transfer_common`; only the fine-tuning loop is
local.

    python transfer_plm.py \\
        --source data/wi_records_cleaned_raw.csv --target data/ga_records_cleaned_raw.csv \\
        --src-name WI --tgt-name GA
    python transfer_plm.py ... --transfer-mode sequential --target-frac 0.4
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from transfer_common import (  # noqa: E402
    add_transfer_args,
    balanced_class_weights,
    bootstrap_target_std,
    configure_verbosity,
    load_and_prepare,
    load_and_prepare_pooled,
    parse_pool_sources,
    log_failed_transfer,
    log_transfer_result,
    scale_infix,
    split_target_fewshot,
    stratified_source_split,
    target_cv_folds,
)

from llm import (  # noqa: E402
    _TextDataset,
    _make_weighted_trainer_class,
    make_training_arguments,
    predict_with_labels,
    trainer_tokenizer_kwarg,
    make_best_model_keeper,
    report_token_lengths,
    DEVICE,
    USE_FP16,
    USE_BF16,
    NUM_WORKERS,
    MODEL_NAME,
    MAX_LEN,
    BATCH_SIZE,
    EVAL_BATCH_SIZE,
    GROUP_BY_LENGTH,
    EPOCHS,
    LR,
    WARMUP_RATIO,
)

from utils import SEED, compute_metrics, run_artifact_dir  # noqa: E402


def _train_source_trainer(
    train_texts, train_labels, val_texts, val_labels,
    tokenizer, n_classes, idx_to_label, classes_sorted, out_dir,
):
    """Fine-tune on (train), select best epoch by QWK on (val), and return the
    trainer with the best weights loaded — WITHOUT predicting. Split out of
    _train_source_eval_target so the zero-shot multi-target
    path can train ONE source model and predict many targets with it."""
    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments,
        DataCollatorWithPadding,
    )

    train_enc = tokenizer(train_texts, truncation=True, padding=False, max_length=MAX_LEN)
    val_enc = tokenizer(val_texts, truncation=True, padding=False, max_length=MAX_LEN)
    train_ds = _TextDataset(train_enc, train_labels)
    val_ds = _TextDataset(val_enc, val_labels)

    class_weights = torch.as_tensor(
        balanced_class_weights(train_labels, n_classes), dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = make_training_arguments(
        output_dir=str(out_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        group_by_length=GROUP_BY_LENGTH,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        eval_strategy="epoch",
        # Write NOTHING to disk during training. Best-epoch selection (by QWK on
        # source-val) and patience-2 early stopping are handled in RAM by
        # make_best_model_keeper, shared with llm.py. Checkpointing per epoch
        # would write roughly 0.6 GB each time, and with many folds and many
        # source/target pairs sharing one scratch subtree that fills the quota.
        save_strategy="no",
        logging_steps=50,
        seed=SEED,
        report_to=[],
        dataloader_num_workers=NUM_WORKERS,
        fp16=USE_FP16,
        bf16=USE_BF16,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        proba = exp / exp.sum(axis=-1, keepdims=True)
        pred_idx = np.argmax(logits, axis=-1)
        y_pred = np.array([idx_to_label[i] for i in pred_idx])
        y_true = np.array([idx_to_label[i] for i in labels])
        return compute_metrics(y_true, y_pred, y_proba=proba, labels=classes_sorted)

    WeightedTrainer = _make_weighted_trainer_class(class_weights)
    keeper = make_best_model_keeper(metric="eval_qwk", greater_is_better=True,
                                    patience=2)
    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        **trainer_tokenizer_kwarg(tokenizer), data_collator=collator,
        compute_metrics=_hf_compute_metrics,
        callbacks=[keeper],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()

    # Restore the best-on-source-val weights (kept in RAM, never on disk).
    if keeper.best_state is not None:
        model.load_state_dict(keeper.best_state)
    return trainer


def _predict_trainer(trainer, tokenizer, test_texts, test_labels, idx_to_label):
    """Predict a test set with an already-trained trainer -> (y_true, y_pred, proba).

    y_true is rebuilt from the label_ids the Trainer gathered, NOT from
    ``test_labels``: ``.predictions`` order is sampler-dependent, so positional
    pairing is unsafe (see llm.predict_with_labels). ``test_labels`` still defines
    the dataset content, and the multiset check below pins label_ids to it."""
    test_ds = _TextDataset(
        tokenizer(test_texts, truncation=True, padding=False, max_length=MAX_LEN),
        test_labels)
    logits, out_labels = predict_with_labels(trainer, test_ds)
    if not np.array_equal(np.sort(out_labels),
                          np.sort(np.asarray(test_labels).astype(np.int64))):
        raise RuntimeError("predict label_ids are not a permutation of the "
                           "test labels — dataset/collator drift, refusing to score.")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    y_proba = exp / exp.sum(axis=-1, keepdims=True)
    y_pred = np.array([idx_to_label[i] for i in np.argmax(logits, axis=-1)])
    y_true = np.array([idx_to_label[i] for i in out_labels])
    return y_true, y_pred, y_proba


def _release_trainer(trainer, out_dir):
    """Free the model + scratch dir once every prediction that needs it is done."""
    del trainer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    # Belt-and-suspenders: with save_strategy="no" this dir holds at most a few
    # KB of run metadata, but drop it so nothing lingers under the scratch tree.
    shutil.rmtree(out_dir, ignore_errors=True)


def _train_source_eval_target(
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
    tokenizer, n_classes, idx_to_label, classes_sorted, out_dir,
):
    """Fine-tune on (train), select best epoch by QWK on (val), predict on
    (test). In transfer: train=source-train, val=source-val (early stopping),
    test=target. val and test are distinct so selection never sees target.
    Thin composition of _train_source_trainer and _predict_trainer;
    behavior is unchanged."""
    trainer = _train_source_trainer(
        train_texts, train_labels, val_texts, val_labels,
        tokenizer, n_classes, idx_to_label, classes_sorted, out_dir)
    y_true, y_pred, y_proba = _predict_trainer(
        trainer, tokenizer, test_texts, test_labels, idx_to_label)
    _release_trainer(trainer, out_dir)
    return y_true, y_pred, y_proba



def _train_sequential_eval_target(
    src_train_texts, src_train_labels, src_val_texts, src_val_labels,
    tgt_train_texts, tgt_train_labels, tgt_val_texts, tgt_val_labels,
    test_texts, test_labels,
    tokenizer, n_classes, idx_to_label, classes_sorted, out_dir, tgt_lr=None,
):
    """SEQUENTIAL pretrain->finetune (Phase E). Two phases on ONE model:
      1. fine-tune on SOURCE-train, early-stop on SOURCE-val (best-by-QWK in RAM);
      2. CONTINUE fine-tuning that model on TARGET adapt-train, early-stop on a
         TARGET-carved val (best-by-QWK).
    Then predict the target test set. Distinct from the pooling path
    (_train_source_eval_target), which unions source+target into one phase.
    Class weights are recomputed PER PHASE (source weights for phase 1, target
    weights for phase 2 -- the per-domain rule)."""
    from transformers import (
        AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        proba = exp / exp.sum(axis=-1, keepdims=True)
        y_pred = np.array([idx_to_label[i] for i in np.argmax(logits, axis=-1)])
        y_true = np.array([idx_to_label[i] for i in labels])
        return compute_metrics(y_true, y_pred, y_proba=proba, labels=classes_sorted)

    def _phase(model, tr_texts, tr_labels, va_texts, va_labels, lr, phase_dir):
        tr_ds = _TextDataset(
            tokenizer(tr_texts, truncation=True, padding=False, max_length=MAX_LEN), tr_labels)
        va_ds = _TextDataset(
            tokenizer(va_texts, truncation=True, padding=False, max_length=MAX_LEN), va_labels)
        cw = torch.as_tensor(balanced_class_weights(tr_labels, n_classes), dtype=torch.float32)
        phase_dir.mkdir(parents=True, exist_ok=True)
        args = make_training_arguments(
            output_dir=str(phase_dir), num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
            group_by_length=GROUP_BY_LENGTH,
            learning_rate=lr, warmup_ratio=WARMUP_RATIO, weight_decay=0.01,
            eval_strategy="epoch", save_strategy="no", logging_steps=50, seed=SEED,
            report_to=[], dataloader_num_workers=NUM_WORKERS, fp16=USE_FP16, bf16=USE_BF16,
        )
        WeightedTrainer = _make_weighted_trainer_class(cw)
        keeper = make_best_model_keeper(metric="eval_qwk", greater_is_better=True, patience=2)
        callbacks = [keeper]
        trainer = WeightedTrainer(
            model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
            **trainer_tokenizer_kwarg(tokenizer), data_collator=collator,
            compute_metrics=_hf_compute_metrics, callbacks=callbacks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.train()
        if keeper.best_state is not None:
            model.load_state_dict(keeper.best_state)
        return trainer

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_classes)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("    [seq] phase 1/2: source pretrain")
    _phase(model, src_train_texts, src_train_labels, src_val_texts, src_val_labels,
           LR, out_dir / "src")
    print("    [seq] phase 2/2: target finetune (continue)")
    trainer = _phase(model, tgt_train_texts, tgt_train_labels, tgt_val_texts, tgt_val_labels,
                     tgt_lr or LR, out_dir / "tgt")

    # Sampler-proof pairing: label_ids travel with predictions.
    test_ds = _TextDataset(
        tokenizer(test_texts, truncation=True, padding=False, max_length=MAX_LEN), test_labels)
    logits, out_labels = predict_with_labels(trainer, test_ds)
    # Permutation guard, matching _predict_trainer.
    if not np.array_equal(np.sort(out_labels),
                          np.sort(np.asarray(test_labels).astype(np.int64))):
        raise RuntimeError("predict label_ids are not a permutation of the "
                           "test labels — dataset/collator drift, refusing to score.")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    y_proba = exp / exp.sum(axis=-1, keepdims=True)
    y_pred = np.array([idx_to_label[i] for i in np.argmax(logits, axis=-1)])
    y_true = np.array([idx_to_label[i] for i in out_labels])

    del model, trainer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)
    return y_true, y_pred, y_proba



def run_transfer(data, models_dir, *, target_frac=0.0, test_frac=0.2,
                 val_frac=0.15, n_boot=1000, seed=SEED, cv_folds=0,
                 no_source=False, mode="pool"):
    """Fit on the source (plus any target supervision) and score the target.

    `cv_folds == 0` uses the single-split protocol: a fixed 20% target test block
    with an optional p% adaptation draw. `cv_folds > 0` runs full-target K-fold CV,
    training on the source plus the other K-1 folds and predicting the held-out
    fold.

    In both modes the early-stopping set is source-val only, so target labels
    never influence model selection. `no_source` trains on target rows alone, as a
    same-representation control.
    """
    from transformers import AutoTokenizer

    PCT = 100 if cv_folds > 0 else int(round(target_frac * 100))
    if data.text_mode == "curriculum_only":
        label_tag = "xfer_bert_curriculum_only"
    else:
        label_tag = f"xfer_bert_textualized_full_{data.compliance_mode}"
    infix = scale_infix(data.scale)
    seq = {"pool": "", "sequential": "_seq"}.get(mode, "")
    if mode == "sequential":
        if no_source:
            raise ValueError(f"--transfer-mode {mode} is a source->target method; "
                             "it is incompatible with --no-source (target-only).")
        if cv_folds <= 0 and target_frac <= 0:
            raise ValueError(f"--transfer-mode {mode} needs target rows: use "
                             "--target-frac > 0 or --target-cv-folds > 0.")
    if no_source:
        method = f"{label_tag}{infix}{seq}_nosrc_tgt{PCT}_{data.tgt_name}"
    else:
        method = f"{label_tag}{infix}{seq}_tgt{PCT}_{data.src_name}2{data.tgt_name}"

    mode_str = (f"full-target {cv_folds}-fold CV (tgt100)" if cv_folds > 0
                else f"target_frac={target_frac} (tgt{PCT})  test_frac={test_frac}")
    print(f"\n{'=' * 70}")
    exp = "EXP9" if cv_folds > 0 else "EXP6/7/8"
    print(f"{exp} PLM: {data.src_name} -> {data.tgt_name}  [{data.text_mode}]")
    print(f"  {mode_str}")
    print(f"  classes={data.labels}  device={DEVICE}  model={MODEL_NAME}")
    print(f"  (target labels used ONLY for final scoring; early-stop val = source only)")
    print(f"{'=' * 70}")

    # Source train/val split is seeded and fold-independent -> compute once.
    tr_pos, va_pos = stratified_source_split(data.y_src, val_frac, seed)
    print(f"  source split: train={len(tr_pos)}  val(early-stop)={len(va_pos)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    report_token_lengths(data.src_texts, tokenizer, f"{data.src_name}/source")
    report_token_lengths(data.tgt_texts, tokenizer, f"{data.tgt_name}/target")

    t0 = time.time()
    if cv_folds > 0:
        # Full-target CV: K independent fine-tunes; concatenate out-of-fold predictions so
        # every target row is scored exactly once by a model that never saw it.
        folds = target_cv_folds(data.y_tgt, cv_folds, seed)
        yt_parts, yp_parts, pr_parts = [], [], []
        for fi, (tr, te) in enumerate(folds, 1):
            print(f"\n  -- fold {fi}/{len(folds)}: target train={len(tr)}  "
                  f"held-out test={len(te)} --")
            if no_source:
                # Target-only: carve the early-stop val from THIS fold's target
                # train rows (no source-val exists). Predict the held-out fold.
                sub_tr, sub_va = stratified_source_split(
                    data.y_tgt[tr], val_frac, seed)
                tr_idx, va_idx = tr[sub_tr], tr[sub_va]
                train_texts = data.tgt_texts.iloc[tr_idx].tolist()
                train_labels = data.y_tgt[tr_idx].astype(int)
                val_texts = data.tgt_texts.iloc[va_idx].tolist()
                val_labels = data.y_tgt[va_idx]
            if mode == "sequential" and not no_source:
                # Phase 1 = source (tr_pos/va_pos); phase 2 = this fold's target
                # train rows, early-stopping on a target-carved val slice.
                sub_tr, sub_va = stratified_source_split(data.y_tgt[tr], val_frac, seed)
                f_tr, f_va = tr[sub_tr], tr[sub_va]
                yt, yp, pr = _train_sequential_eval_target(
                    data.src_texts.iloc[tr_pos].tolist(), data.y_src[tr_pos].astype(int),
                    data.src_texts.iloc[va_pos].tolist(), data.y_src[va_pos],
                    data.tgt_texts.iloc[f_tr].tolist(), data.y_tgt[f_tr].astype(int),
                    data.tgt_texts.iloc[f_va].tolist(), data.y_tgt[f_va],
                    data.tgt_texts.iloc[te].tolist(), data.y_tgt[te],
                    tokenizer, data.n_classes, data.idx_to_label,
                    classes_sorted=data.labels, out_dir=models_dir / f"{method}_fold{fi}",
                )
                yt_parts.append(yt); yp_parts.append(yp); pr_parts.append(pr)
                continue
            if no_source:
                pass  # train_texts/labels already set above
            else:
                train_texts = (data.src_texts.iloc[tr_pos].tolist()
                               + data.tgt_texts.iloc[tr].tolist())
                train_labels = np.concatenate(
                    [data.y_src[tr_pos], data.y_tgt[tr]]).astype(int)
                val_texts = data.src_texts.iloc[va_pos].tolist()
                val_labels = data.y_src[va_pos]
            yt, yp, pr = _train_source_eval_target(
                train_texts, train_labels,
                val_texts, val_labels,
                data.tgt_texts.iloc[te].tolist(), data.y_tgt[te],
                tokenizer, data.n_classes, data.idx_to_label,
                classes_sorted=data.labels,
                out_dir=models_dir / f"{method}_fold{fi}",
            )
            yt_parts.append(yt)
            yp_parts.append(yp)
            pr_parts.append(pr)
        y_true = np.concatenate(yt_parts)
        y_pred = np.concatenate(yp_parts)
        y_proba = np.concatenate(pr_parts, axis=0)
    else:
        # Single-split: fixed 20% test set + (possibly empty) adaptation draw.
        adapt_idx, test_idx = split_target_fewshot(
            data.y_tgt, target_frac, test_frac, seed)
        print(f"  target split: adapt={len(adapt_idx)}  test={len(test_idx)}")
        if no_source:
            # Target-only single-split: train on the adaptation rows alone,
            # early-stop on a stratified slice of them. Needs target labels.
            if len(adapt_idx) == 0:
                raise ValueError(
                    "--no-source with target_frac=0 leaves an empty training set; "
                    "the target-only baseline needs target labels (use CV mode for "
                    "full-target CV, or --target-frac > 0).")
            sub_tr, sub_va = stratified_source_split(
                data.y_tgt[adapt_idx], val_frac, seed)
            tr_idx, va_idx = adapt_idx[sub_tr], adapt_idx[sub_va]
            train_texts = data.tgt_texts.iloc[tr_idx].tolist()
            train_labels = data.y_tgt[tr_idx].astype(int)
            val_texts = data.tgt_texts.iloc[va_idx].tolist()
            val_labels = data.y_tgt[va_idx]
        elif mode == "sequential":
            # SEQUENTIAL: phase 1 = source (tr_pos/va_pos); phase 2 = the target
            # adaptation rows, split into train + a carved early-stop val.
            sub_tr, sub_va = stratified_source_split(data.y_tgt[adapt_idx], val_frac, seed)
            a_tr, a_va = adapt_idx[sub_tr], adapt_idx[sub_va]
            y_true, y_pred, y_proba = _train_sequential_eval_target(
                data.src_texts.iloc[tr_pos].tolist(), data.y_src[tr_pos].astype(int),
                data.src_texts.iloc[va_pos].tolist(), data.y_src[va_pos],
                data.tgt_texts.iloc[a_tr].tolist(), data.y_tgt[a_tr].astype(int),
                data.tgt_texts.iloc[a_va].tolist(), data.y_tgt[a_va],
                data.tgt_texts.iloc[test_idx].tolist(), data.y_tgt[test_idx],
                tokenizer, data.n_classes, data.idx_to_label,
                classes_sorted=data.labels, out_dir=models_dir / method,
            )
            train_texts = None  # sequential produced predictions already
        else:
            # Train = source-train UNION target-adaptation (text + target labels).
            # Class weights recomputed over the COMBINED labels inside
            # _train_source_eval_target via balanced_class_weights(train_labels, ...).
            train_texts = (data.src_texts.iloc[tr_pos].tolist()
                           + data.tgt_texts.iloc[adapt_idx].tolist())
            train_labels = np.concatenate(
                [data.y_src[tr_pos], data.y_tgt[adapt_idx]]).astype(int)
            val_texts = data.src_texts.iloc[va_pos].tolist()
            val_labels = data.y_src[va_pos]
        if not (mode == "sequential" and not no_source):
            y_true, y_pred, y_proba = _train_source_eval_target(
                train_texts, train_labels,
                val_texts, val_labels,
                data.tgt_texts.iloc[test_idx].tolist(), data.y_tgt[test_idx],
                tokenizer, data.n_classes, data.idx_to_label,
                classes_sorted=data.labels, out_dir=models_dir / method,
            )

    point = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=data.labels)
    std = bootstrap_target_std(y_true, y_pred, y_proba, labels=data.labels,
                               n_boot=n_boot, seed=seed)
    elapsed = time.time() - t0

    scope = "TARGET OOF" if cv_folds > 0 else "TARGET TEST"
    print(
        f"\n  {scope} ({data.tgt_name}, n={len(y_true)}, +/-=bootstrap std over {n_boot}):\n"
        f"    acc={point['accuracy']:.3f}+/-{std['accuracy']:.3f}  "
        f"bal_acc={point['balanced_acc']:.3f}+/-{std['balanced_acc']:.3f}  "
        f"macroF1={point['macro_f1']:.3f}+/-{std['macro_f1']:.3f}\n"
        f"    qwk={point['qwk']:.3f}+/-{std['qwk']:.3f}  "
        f"mae={point['mae']:.3f}+/-{std['mae']:.3f}  "
        f"ll={point['log_loss']:.3f}+/-{std['log_loss']:.3f}"
    )
    print(f"  elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    return method, point, std



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 2/6/7/8: PLM (few-shot) cross-state transfer.")
    add_transfer_args(parser)
    parser.add_argument("--no-source", action="store_true",
                        help="target-only baseline: omit the source and train on "
                             "target rows only (CV: the other K-1 folds; single-split: "
                             "the adaptation draw), early-stopping on a target-carved "
                             "slice. Source-independent -> logged once per target as "
                             "..._nosrc_tgt<PCT>_<TGT>. Same-representation control for "
                             "the full-target CV delta.")
    parser.add_argument("--transfer-mode",
                        choices=("pool", "sequential"), default="pool",
                        help="'pool' (default): one phase on source-train UNION target adapt. "
                             "'sequential' (Phase E): source pretrain -> continue on target "
                             "(_seq; source->target, not --no-source; needs target rows).")
    args = parser.parse_args()
    configure_verbosity(args.verbose)

    if args.pool_sources:  # LOSO: pool many sources vs the held-out --target
        data = load_and_prepare_pooled(
            parse_pool_sources(args.pool_sources), args.tgt_name, args.target,
            text_mode=args.text_mode, compliance_mode=args.compliance,
            scale=args.rating_scale, pool_name=args.pool_name)
    else:
        data = load_and_prepare(
            args.source, args.target, args.src_name, args.tgt_name,
            text_mode=args.text_mode, compliance_mode=args.compliance,
            scale=args.rating_scale,
        )
    models_dir = run_artifact_dir(args.output) / "plm_transfer_models"

    method, point, std = run_transfer(
        data, models_dir, target_frac=args.target_frac,
        test_frac=args.target_test_frac, val_frac=args.val_frac,
        n_boot=args.n_bootstrap, seed=args.seed, cv_folds=args.target_cv_folds,
        no_source=args.no_source, mode=args.transfer_mode)
    log_transfer_result(method, point, std, output_path=args.output, notes=method,
                        source=None if args.no_source else data.src_name,
                        target=data.tgt_name, classes=data.labels)

    print(f"\nDone. See {args.output} for transfer scores.")
    print(f"Per-run model artifacts at {models_dir} (safe to delete).")


def _failed_method(a):
    PCT = (100 if getattr(a, "target_cv_folds", 0) > 0
           else int(round(getattr(a, "target_frac", 0.0) * 100)))
    tag = ("xfer_bert_curriculum_only" if a.text_mode == "curriculum_only"
           else f"xfer_bert_textualized_full_{a.compliance}")
    infix = scale_infix(getattr(a, "rating_scale", "3star"))
    seq = {"pool": "", "sequential": "_seq"}.get(getattr(a, "transfer_mode", "pool"), "")
    if getattr(a, "no_source", False):
        return f"{tag}{infix}{seq}_nosrc_tgt{PCT}_{a.tgt_name}"
    # LOSO invocations pass --pool-sources without --src-name, and the success
    # tag uses pool_name ("LOSO"), so the FAILED tag must too. Otherwise a crashed
    # cell is filed under a method name nothing looks for.
    src = a.pool_name if getattr(a, "pool_sources", None) else a.src_name
    return f"{tag}{infix}{seq}_tgt{PCT}_{src}2{a.tgt_name}"


if __name__ == "__main__":
    # Record a FAILED row and a traceback instead of vanishing: a crash that
    # leaves no row at all is invisible to anything reading the results tree.
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        _p = argparse.ArgumentParser(add_help=False)
        add_transfer_args(_p)
        _p.add_argument("--no-source", action="store_true")
        _p.add_argument("--transfer-mode",
                        choices=("pool", "sequential"), default="pool")
        _a, _ = _p.parse_known_args()
        _pool_src = _a.pool_name if getattr(_a, "pool_sources", None) else _a.src_name
        log_transfer_source = None if getattr(_a, "no_source", False) else _pool_src
        log_failed_transfer(_failed_method(_a), output_path=_a.output,
                            source=log_transfer_source, target=_a.tgt_name,
                            error=f"{type(exc).__name__}: {exc}")
        raise