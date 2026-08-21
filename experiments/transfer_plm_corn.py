"""ModernBERT for cross-state transfer, with a rank-consistent ordinal head.

The MBERT-CORN arm. Replaces the K-way softmax with K-1 units, unit k modelling
P(y > rank_k | y > rank_{k-1}) (Shi, Cao and Raschka 2021, arXiv:2111.08851).
Chaining the conditionals makes the implied cumulative probabilities monotone, so
prediction is the count of cumulative probabilities above 0.5 and the per-class
probabilities telescope from the same chain.

Split protocol, scoring and the two transfer regimes are shared with
`transfer_plm`; only the head, the loss and the decode differ.

`corn_loss_weighted` adds per-class weighting, which `coral_pytorch`'s stock loss
does not expose. With uniform weights it reduces to the stock loss, checked at
import by `_assert_corn_equivalence`.

    python transfer_plm_corn.py \
        --source data/nc_records_cleaned_raw.csv --target data/wi_records_cleaned_raw.csv \
        --src-name NC --tgt-name WI --output results.csv
"""
from __future__ import annotations

import argparse
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
    log_failed_transfer,
    log_transfer_result,
    parse_pool_sources,
    scale_infix,
    split_target_fewshot,
    stratified_source_split,
    target_cv_folds,
)
from llm import (  # noqa: E402
    _TextDataset,
    make_best_model_keeper,
    make_training_arguments,
    predict_with_labels,
    trainer_tokenizer_kwarg,
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


def _corn_decode(logits, idx_to_label, n_classes):
    """Convert (N, K-1) CORN logits to (labels, per-class probabilities).

    cum[:, k] = P(y > rank_k) is the cumulative product of the sigmoids. Per-class
    probabilities telescope from it; the label is the canonical CORN rank, the
    count of cumulative probabilities above 0.5.
    """
    from coral_pytorch.dataset import corn_label_from_logits

    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    cum = torch.cumprod(torch.sigmoid(logits_t), dim=1)  # (N, K-1)

    parts = [1.0 - cum[:, :1]]                            # P(y=0)
    if n_classes > 2:
        parts.append(cum[:, :-1] - cum[:, 1:])           # P(y=1..K-2)
    parts.append(cum[:, -1:])                            # P(y=K-1)
    proba = torch.clamp(torch.cat(parts, dim=1), min=0.0)
    proba = (proba / proba.sum(dim=1, keepdim=True)).cpu().numpy()

    pred_idx = corn_label_from_logits(logits_t).cpu().numpy().astype(int)
    y_pred = np.array([idx_to_label[int(i)] for i in pred_idx])
    return y_pred, proba


def corn_loss_weighted(logits, y_train, num_classes, class_weights=None):
    """CORN loss with optional per-class weighting.

    Task k trains only on samples with y > k-1, against the binary target
    1[y > k]. Each sample's contribution is scaled by its class weight and the
    normaliser is the summed weight rather than the sample count, so uniform
    weights reduce to `coral_pytorch.losses.corn_loss`.

    `class_weights` is indexed by class and has length K.
    """
    import torch
    import torch.nn.functional as F

    if class_weights is None:
        from coral_pytorch.losses import corn_loss
        return corn_loss(logits, y_train, num_classes)

    w_all = class_weights.to(logits.device)[y_train]     # (N,) per-sample
    losses = 0.0
    denom = 0.0
    for task_index in range(num_classes - 1):
        label_mask = y_train > task_index - 1
        train_labels = (y_train[label_mask] > task_index).to(torch.int64)
        if len(train_labels) < 1:
            continue
        pred = logits[label_mask, task_index]
        w = w_all[label_mask]
        # Same per-sample term as corn_loss, before its torch.sum.
        per = -(F.logsigmoid(pred) * train_labels
                + (F.logsigmoid(pred) - pred) * (1 - train_labels))
        losses = losses + torch.sum(per * w)
        denom = denom + w.sum()
    return losses / denom


def _assert_corn_equivalence():
    """Check that uniform weights reproduce `coral_pytorch`'s `corn_loss`."""
    import torch
    from coral_pytorch.losses import corn_loss

    g = torch.Generator().manual_seed(0)
    K = 5
    logits = torch.randn(64, K - 1, generator=g)
    y = torch.randint(0, K, (64,), generator=g)
    ref = corn_loss(logits, y, K)
    ours = corn_loss_weighted(logits, y, K, class_weights=torch.ones(K))
    if not torch.allclose(ref, ours, atol=1e-6):
        raise RuntimeError(
            "corn_loss_weighted with uniform weights does not match "
            f"coral_pytorch.corn_loss ({ref.item():.8f} vs {ours.item():.8f}) "
            "— refusing to train a CORN model on a loss that is not CORN.")


def _make_corn_trainer_class(num_classes: int, class_weights=None):
    """Trainer subclass whose loss is corn_loss over K-1 logits. Built as a
    factory so num_classes is captured without touching the HF Trainer API.

    `class_weights`: a length-K tensor of balanced
    per-class weights, or None for the historical unweighted loss.
    """
    from transformers import Trainer

    if class_weights is not None:
        _assert_corn_equivalence()

    class CornTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits  # (N, K-1)
            loss = corn_loss_weighted(logits, labels, num_classes,
                                      class_weights=class_weights)
            return (loss, outputs) if return_outputs else loss

    return CornTrainer


def _train_source_corn_trainer(
    train_texts, train_labels, val_texts, val_labels,
    tokenizer, n_classes, idx_to_label, classes_sorted, out_dir,
):
    """Fine-tune a CORN-headed ModernBERT on (train), select the best epoch by
    QWK on (val), and return the trainer (best weights loaded) WITHOUT
    predicting — split out so the zero-shot multi-target path
    trains once per source. Mirrors transfer_plm._train_source_trainer but with
    a K-1-unit head, corn_loss, and CORN decoding."""
    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments,
        DataCollatorWithPadding,
    )

    train_enc = tokenizer(train_texts, truncation=True, padding=False, max_length=MAX_LEN)
    val_enc = tokenizer(val_texts, truncation=True, padding=False, max_length=MAX_LEN)
    train_ds = _TextDataset(train_enc, train_labels)
    val_ds = _TextDataset(val_enc, val_labels)

    # CORN: K-1 output units (vs K for softmax).
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes - 1)
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
        save_strategy="no",   # best epoch kept in RAM (make_best_model_keeper)
        logging_steps=50,
        seed=SEED,
        report_to=[],
        dataloader_num_workers=NUM_WORKERS,
        fp16=USE_FP16,
        bf16=USE_BF16,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _hf_compute_metrics(eval_pred):
        logits = np.asarray(eval_pred.predictions if hasattr(eval_pred, "predictions")
                            else eval_pred[0])
        labels = np.asarray(eval_pred.label_ids if hasattr(eval_pred, "label_ids")
                            else eval_pred[1])
        y_pred, proba = _corn_decode(logits, idx_to_label, n_classes)
        y_true = np.array([idx_to_label[int(i)] for i in labels])
        return compute_metrics(y_true, y_pred, y_proba=proba, labels=classes_sorted)

    # Balanced per-class weights, recomputed from THIS phase's training
    # labels so CORN follows the same
    # balanced-only policy as every other method in the suite.
    import torch as _torch
    _cw = _torch.as_tensor(
        balanced_class_weights(np.asarray(train_labels), n_classes),
        dtype=_torch.float32)
    CornTrainer = _make_corn_trainer_class(n_classes, class_weights=_cw)
    keeper = make_best_model_keeper(metric="eval_qwk", greater_is_better=True,
                                    patience=2)
    trainer = CornTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        **trainer_tokenizer_kwarg(tokenizer), data_collator=collator,
        compute_metrics=_hf_compute_metrics,
        callbacks=[keeper],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()

    if keeper.best_state is not None:
        model.load_state_dict(keeper.best_state)
    return trainer


def _predict_corn_trainer(trainer, tokenizer, test_texts, test_labels,
                          idx_to_label, n_classes):
    """Predict a test set with a trained CORN trainer -> (y_true, y_pred, proba).

    NB: y_true comes from the label_ids the Trainer gathered, NOT
    from ``test_labels`` — ``.predictions`` order is sampler-dependent
    (group_by_length now permutes eval/test loaders on newer transformers). See
    llm.predict_with_labels."""
    test_ds = _TextDataset(
        tokenizer(test_texts, truncation=True, padding=False, max_length=MAX_LEN),
        test_labels)
    logits, out_labels = predict_with_labels(trainer, test_ds)
    if not np.array_equal(np.sort(out_labels),
                          np.sort(np.asarray(test_labels).astype(np.int64))):
        raise RuntimeError("predict label_ids are not a permutation of the "
                           "test labels — dataset/collator drift, refusing to score.")
    y_pred, y_proba = _corn_decode(logits, idx_to_label, n_classes)
    y_true = np.array([idx_to_label[int(i)] for i in out_labels])
    return y_true, y_pred, y_proba


def _train_source_eval_target_corn(
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels,
    tokenizer, n_classes, idx_to_label, classes_sorted, out_dir,
):
    """Original single-pair path: train + predict (unchanged behavior)."""
    trainer = _train_source_corn_trainer(
        train_texts, train_labels, val_texts, val_labels,
        tokenizer, n_classes, idx_to_label, classes_sorted, out_dir)
    return _predict_corn_trainer(trainer, tokenizer, test_texts, test_labels,
                                 idx_to_label, n_classes)


def _corn_phase(model, tr_texts, tr_labels, va_texts, va_labels, tokenizer,
                n_classes, idx_to_label, classes_sorted, lr, phase_dir):
    """Fine-tune ONE already-constructed CORN-headed model for one phase (source
    OR target), early-stopping on (va) by QWK, best weights loaded back in RAM.
    Returns the trainer. Factored out so the sequential path reuses ONE model
    across two phases (source pretrain -> target continue). Loss is the native
    (unweighted) corn_loss, matching the zero-shot CORN variant."""
    from transformers import DataCollatorWithPadding

    tr_ds = _TextDataset(
        tokenizer(tr_texts, truncation=True, padding=False, max_length=MAX_LEN), tr_labels)
    va_ds = _TextDataset(
        tokenizer(va_texts, truncation=True, padding=False, max_length=MAX_LEN), va_labels)
    phase_dir.mkdir(parents=True, exist_ok=True)
    args = make_training_arguments(
        output_dir=str(phase_dir), num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
        group_by_length=GROUP_BY_LENGTH, learning_rate=lr, warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01, eval_strategy="epoch", save_strategy="no", logging_steps=50,
        seed=SEED, report_to=[], dataloader_num_workers=NUM_WORKERS,
        fp16=USE_FP16, bf16=USE_BF16,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _hf_compute_metrics(eval_pred):
        logits = np.asarray(eval_pred.predictions if hasattr(eval_pred, "predictions")
                            else eval_pred[0])
        labels = np.asarray(eval_pred.label_ids if hasattr(eval_pred, "label_ids")
                            else eval_pred[1])
        y_pred, proba = _corn_decode(logits, idx_to_label, n_classes)
        y_true = np.array([idx_to_label[int(i)] for i in labels])
        return compute_metrics(y_true, y_pred, y_proba=proba, labels=classes_sorted)

    # Balanced per-class weights, recomputed from THIS phase's training
    # labels so CORN follows the same
    # balanced-only policy as every other method in the suite.
    import torch as _torch
    _cw = _torch.as_tensor(
        balanced_class_weights(np.asarray(tr_labels), n_classes),
        dtype=_torch.float32)
    CornTrainer = _make_corn_trainer_class(n_classes, class_weights=_cw)
    keeper = make_best_model_keeper(metric="eval_qwk", greater_is_better=True, patience=2)
    trainer = CornTrainer(
        model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
        **trainer_tokenizer_kwarg(tokenizer), data_collator=collator,
        compute_metrics=_hf_compute_metrics, callbacks=[keeper],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()
    if keeper.best_state is not None:
        model.load_state_dict(keeper.best_state)
    return trainer


def _train_sequential_eval_target_corn(
    src_train_texts, src_train_labels, src_val_texts, src_val_labels,
    tgt_train_texts, tgt_train_labels, tgt_val_texts, tgt_val_labels,
    test_texts, test_labels,
    tokenizer, n_classes, idx_to_label, classes_sorted, out_dir,
):
    """SEQUENTIAL CORN pretrain->finetune (few-shot _seq): two phases on ONE
    K-1-unit CORN model -- phase 1 fine-tunes on SOURCE-train (early-stop on
    SOURCE-val), phase 2 CONTINUES on TARGET adapt-train (early-stop on a
    TARGET-carved val) -- then predicts the target test set. The CORN counterpart
    of transfer_plm._train_sequential_eval_target; distinct from the pooling path
    (_train_source_eval_target_corn), which unions source+target into one phase."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes - 1)   # CORN: K-1 units
    out_dir.mkdir(parents=True, exist_ok=True)
    print("    [corn-seq] phase 1/2: source pretrain")
    _corn_phase(model, src_train_texts, src_train_labels, src_val_texts, src_val_labels,
                tokenizer, n_classes, idx_to_label, classes_sorted, LR, out_dir / "src")
    print("    [corn-seq] phase 2/2: target finetune (continue)")
    trainer = _corn_phase(model, tgt_train_texts, tgt_train_labels,
                          tgt_val_texts, tgt_val_labels, tokenizer, n_classes,
                          idx_to_label, classes_sorted, LR, out_dir / "tgt")
    return _predict_corn_trainer(trainer, tokenizer, test_texts, test_labels,
                                 idx_to_label, n_classes)


def _corn_method_tag(data, target_frac: float, cv_folds: int, mode: str = "pool") -> str:
    """Method tag for CORN, generalized over supervision level + transfer mode.
    tgt0 (zero-shot) / tgt{PCT} (few-shot) / tgt100 (full-target CV); '_seq' for
    sequential. Mirrors transfer_plm's tagging so CORN rows line up 1:1 with the
    softmax MBERT rows at every percentage."""
    PCT = 100 if cv_folds > 0 else int(round(target_frac * 100))
    seq = "_seq" if mode == "sequential" else ""
    return (f"xfer_bert_corn_{data.compliance_mode}{scale_infix(data.scale)}{seq}"
            f"_tgt{PCT}_{data.src_name}2{data.tgt_name}")


def _method_tag(data) -> str:
    return _corn_method_tag(data, 0.0, 0, "pool")


def run_transfer_corn(data, models_dir, *, target_frac=0.0, test_frac=0.2,
                      val_frac=0.15, n_boot=1000, seed=SEED, cv_folds=0,
                      mode="pool", output_path=None):
    """Fit on the source (plus any target supervision) and score the target.

    Mirrors `transfer_plm.run_transfer` -- same splits, same regimes, same scoring
    -- with the CORN head, loss and decode substituted.
    """
    from transformers import AutoTokenizer

    PCT = 100 if cv_folds > 0 else int(round(target_frac * 100))
    method = _corn_method_tag(data, target_frac, cv_folds, mode)
    if mode == "sequential" and cv_folds <= 0 and target_frac <= 0:
        raise ValueError("--transfer-mode sequential needs target rows: use "
                         "--target-frac > 0 or --target-cv-folds > 0.")

    mode_str = (f"full-target {cv_folds}-fold CV (tgt100)" if cv_folds > 0
                else f"target_frac={target_frac} (tgt{PCT})  test_frac={test_frac}")
    print(f"\n{'=' * 70}")
    exp = "EXP9" if cv_folds > 0 else "EXP6/7/8"
    print(f"{exp} CORN PLM [{mode}]: {data.src_name} -> {data.tgt_name}  [{data.text_mode}]")
    print(f"  {mode_str}")
    print(f"  classes={data.labels}  device={DEVICE}  model={MODEL_NAME}  "
          f"(K-1={data.n_classes - 1} CORN units)")
    print(f"{'=' * 70}")

    tr_pos, va_pos = stratified_source_split(data.y_src, val_frac, seed)
    print(f"  source split: train={len(tr_pos)}  val(early-stop)={len(va_pos)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    report_token_lengths(data.src_texts, tokenizer, f"{data.src_name}/source")
    report_token_lengths(data.tgt_texts, tokenizer, f"{data.tgt_name}/target")

    t0 = time.time()
    if cv_folds > 0:
        folds = target_cv_folds(data.y_tgt, cv_folds, seed)
        yt_parts, yp_parts, pr_parts = [], [], []
        for fi, (tr, te) in enumerate(folds, 1):
            print(f"\n  -- fold {fi}/{len(folds)}: target train={len(tr)}  "
                  f"held-out test={len(te)} --")
            if mode == "sequential":
                sub_tr, sub_va = stratified_source_split(data.y_tgt[tr], val_frac, seed)
                f_tr, f_va = tr[sub_tr], tr[sub_va]
                yt, yp, pr = _train_sequential_eval_target_corn(
                    data.src_texts.iloc[tr_pos].tolist(), data.y_src[tr_pos].astype(int),
                    data.src_texts.iloc[va_pos].tolist(), data.y_src[va_pos],
                    data.tgt_texts.iloc[f_tr].tolist(), data.y_tgt[f_tr].astype(int),
                    data.tgt_texts.iloc[f_va].tolist(), data.y_tgt[f_va],
                    data.tgt_texts.iloc[te].tolist(), data.y_tgt[te],
                    tokenizer, data.n_classes, data.idx_to_label,
                    classes_sorted=data.labels, out_dir=models_dir / f"{method}_fold{fi}")
            else:
                train_texts = (data.src_texts.iloc[tr_pos].tolist()
                               + data.tgt_texts.iloc[tr].tolist())
                train_labels = np.concatenate(
                    [data.y_src[tr_pos], data.y_tgt[tr]]).astype(int)
                yt, yp, pr = _train_source_eval_target_corn(
                    train_texts, train_labels,
                    data.src_texts.iloc[va_pos].tolist(), data.y_src[va_pos],
                    data.tgt_texts.iloc[te].tolist(), data.y_tgt[te],
                    tokenizer, data.n_classes, data.idx_to_label,
                    classes_sorted=data.labels, out_dir=models_dir / f"{method}_fold{fi}")
            yt_parts.append(yt); yp_parts.append(yp); pr_parts.append(pr)
        y_true = np.concatenate(yt_parts)
        y_pred = np.concatenate(yp_parts)
        y_proba = np.concatenate(pr_parts, axis=0)
    else:
        adapt_idx, test_idx = split_target_fewshot(
            data.y_tgt, target_frac, test_frac, seed)
        print(f"  target split: adapt={len(adapt_idx)}  test={len(test_idx)}")
        if mode == "sequential":
            sub_tr, sub_va = stratified_source_split(data.y_tgt[adapt_idx], val_frac, seed)
            a_tr, a_va = adapt_idx[sub_tr], adapt_idx[sub_va]
            y_true, y_pred, y_proba = _train_sequential_eval_target_corn(
                data.src_texts.iloc[tr_pos].tolist(), data.y_src[tr_pos].astype(int),
                data.src_texts.iloc[va_pos].tolist(), data.y_src[va_pos],
                data.tgt_texts.iloc[a_tr].tolist(), data.y_tgt[a_tr].astype(int),
                data.tgt_texts.iloc[a_va].tolist(), data.y_tgt[a_va],
                data.tgt_texts.iloc[test_idx].tolist(), data.y_tgt[test_idx],
                tokenizer, data.n_classes, data.idx_to_label,
                classes_sorted=data.labels, out_dir=models_dir / method)
        else:
            train_texts = (data.src_texts.iloc[tr_pos].tolist()
                           + data.tgt_texts.iloc[adapt_idx].tolist())
            train_labels = np.concatenate(
                [data.y_src[tr_pos], data.y_tgt[adapt_idx]]).astype(int)
            y_true, y_pred, y_proba = _train_source_eval_target_corn(
                train_texts, train_labels,
                data.src_texts.iloc[va_pos].tolist(), data.y_src[va_pos],
                data.tgt_texts.iloc[test_idx].tolist(), data.y_tgt[test_idx],
                tokenizer, data.n_classes, data.idx_to_label,
                classes_sorted=data.labels, out_dir=models_dir / method)

    point = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=data.labels)
    std = bootstrap_target_std(y_true, y_pred, y_proba, labels=data.labels,
                               n_boot=n_boot, seed=seed)
    elapsed = time.time() - t0
    print(
        f"\n  TARGET {'OOF' if cv_folds > 0 else 'TEST'} ({data.tgt_name}, "
        f"n={len(y_true)}, +/-=bootstrap std):\n"
        f"    qwk={point['qwk']:.3f}+/-{std['qwk']:.3f}  "
        f"bal_acc={point['balanced_acc']:.3f}+/-{std['balanced_acc']:.3f}  "
        f"mae={point['mae']:.3f}+/-{std['mae']:.3f}\n"
        f"  elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)"
    )
    log_transfer_result(method, point, std, output_path=output_path, notes=method,
                        source=data.src_name, target=data.tgt_name, classes=data.labels)
    return method


def main() -> None:
    parser = argparse.ArgumentParser(
        description="experiment_zero_shot: PLM source-only transfer with a CORN "
                    "ordinal head (zero-shot; scores on the fixed 20% target slice).")
    add_transfer_args(parser)
    parser.add_argument("--transfer-mode", choices=("pool", "sequential"), default="pool",
                        help="Few-shot CORN mode. 'pool' (default): fit once on "
                             "source-train U target-adapt. 'sequential' (_seq tag): "
                             "source pretrain then continue-finetune on the target "
                             "adapt rows. Ignored at zero-shot (target_frac=0).")
    args = parser.parse_args()
    configure_verbosity(args.verbose)

    # Few-shot / full-target-CV CORN (Phase: lifted the old zero-shot-only guard).
    if (args.target_frac and args.target_frac > 0) or \
       (getattr(args, "target_cv_folds", 0) and args.target_cv_folds > 0):
        if args.pool_sources:  # LOSO + target supervision (pooled sources folded in)
            data = load_and_prepare_pooled(
                parse_pool_sources(args.pool_sources), args.tgt_name, args.target,
                text_mode=args.text_mode, compliance_mode=args.compliance,
                scale=args.rating_scale, pool_name=args.pool_name)
        else:
            data = load_and_prepare(
                args.source, args.target, args.src_name, args.tgt_name,
                text_mode=args.text_mode, compliance_mode=args.compliance,
                scale=args.rating_scale)
        run_transfer_corn(
            data, run_artifact_dir(args.output) / "plm_corn_transfer_models",
            target_frac=args.target_frac, test_frac=args.target_test_frac,
            val_frac=args.val_frac, n_boot=args.n_bootstrap, seed=args.seed,
            cv_folds=args.target_cv_folds, mode=args.transfer_mode,
            output_path=args.output)
        print(f"\nDone. See {args.output} for transfer scores.")
        return

    if args.transfer_mode == "sequential":
        parser.error("--transfer-mode sequential needs target rows "
                     "(--target-frac>0 or --target-cv-folds>0); it has no zero-shot form.")


    if args.pool_sources:  # LOSO zero-shot CORN: pooled sources vs held-out target
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
    models_dir = run_artifact_dir(args.output) / "plm_corn_transfer_models"
    method = _method_tag(data)

    print(f"\n{'=' * 70}")
    print(f"ZERO-SHOT CORN PLM: {data.src_name} -> {data.tgt_name}  [{data.text_mode}]")
    print(f"  classes={data.labels}  device={DEVICE}  model={MODEL_NAME}  (K-1={data.n_classes - 1} CORN units)")
    print(f"  target labels used ONLY for final scoring; early-stop val = source only")
    print(f"{'=' * 70}")

    from transformers import AutoTokenizer

    # Source train/val split (early stopping on source-val), and the SAME fixed
    # 20% target test slice used by the softmax MBERT zero-shot (comparability).
    tr_pos, va_pos = stratified_source_split(data.y_src, args.val_frac, args.seed)
    _, test_idx = split_target_fewshot(
        data.y_tgt, 0.0, args.target_test_frac, args.seed)
    print(f"  source split: train={len(tr_pos)}  val(early-stop)={len(va_pos)}  "
          f"target test={len(test_idx)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    report_token_lengths(data.src_texts, tokenizer, f"{data.src_name}/source")
    report_token_lengths(data.tgt_texts, tokenizer, f"{data.tgt_name}/target")

    t0 = time.time()
    y_true, y_pred, y_proba = _train_source_eval_target_corn(
        data.src_texts.iloc[tr_pos].tolist(), data.y_src[tr_pos].astype(int),
        data.src_texts.iloc[va_pos].tolist(), data.y_src[va_pos],
        data.tgt_texts.iloc[test_idx].tolist(), data.y_tgt[test_idx],
        tokenizer, data.n_classes, data.idx_to_label,
        classes_sorted=data.labels, out_dir=models_dir / method,
    )

    point = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=data.labels)
    std = bootstrap_target_std(y_true, y_pred, y_proba, labels=data.labels,
                               n_boot=args.n_bootstrap, seed=args.seed)
    elapsed = time.time() - t0
    print(
        f"\n  TARGET TEST ({data.tgt_name}, n={len(y_true)}, +/-=bootstrap std):\n"
        f"    qwk={point['qwk']:.3f}+/-{std['qwk']:.3f}  "
        f"bal_acc={point['balanced_acc']:.3f}+/-{std['balanced_acc']:.3f}  "
        f"mae={point['mae']:.3f}+/-{std['mae']:.3f}\n"
        f"  elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)"
    )

    log_transfer_result(method, point, std, output_path=args.output, notes=method,
                        source=data.src_name, target=data.tgt_name, classes=data.labels)
    print(f"\nDone. See {args.output} for transfer scores.")
    print(f"Per-run model artifacts at {models_dir} (safe to delete).")



def _failed_method(a) -> str:
    infix = scale_infix(getattr(a, "rating_scale", "3star"))
    cv = getattr(a, "target_cv_folds", 0) or 0
    PCT = 100 if cv > 0 else int(round(getattr(a, "target_frac", 0.0) * 100))
    seq = "_seq" if getattr(a, "transfer_mode", "pool") == "sequential" else ""
    # Pool-aware source token, matching the success tag: LOSO passes
    # --pool-sources without --src-name, so the source reads as pool_name.
    src = a.pool_name if getattr(a, "pool_sources", None) else a.src_name
    return f"xfer_bert_corn_{a.compliance}{infix}{seq}_tgt{PCT}_{src}2{a.tgt_name}"


if __name__ == "__main__":
    # Mirror transfer_plm.py: on any crash, record a FAILED row + traceback so a
    # dead run self-reports instead of leaving a silent gap in the results CSV.
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        _p = argparse.ArgumentParser(add_help=False)
        add_transfer_args(_p)
        _p.add_argument("--transfer-mode", choices=("pool", "sequential"), default="pool")
        _a, _ = _p.parse_known_args()
        _src = _a.pool_name if getattr(_a, "pool_sources", None) else _a.src_name
        log_failed_transfer(_failed_method(_a), output_path=_a.output,
                            source=_src, target=_a.tgt_name,
                            error=f"{type(exc).__name__}: {exc}")
        raise
