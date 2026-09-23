"""ModernBERT for cross-state transfer over the serialized text view.

`--transfer-mode pool` fits once on the source plus any target adaptation rows;
`sequential` (the `_seq` tags) pretrains on the source and then fine-tunes on the
adaptation rows. At p=0 only the source phase runs, early-stopped on a source slice.

    python transfer_plm.py --pool-sources NC=data/nc_records_cleaned_raw.csv ... \\
        --target data/wi_records_cleaned_raw.csv --tgt-name WI --rating-scale 5star \\
        --target-frac 0.4 --transfer-mode sequential
"""
from __future__ import annotations

import argparse
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
    _resolve_transfer_output,
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


def _softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _metrics_fn(idx_to_label, classes_sorted):
    def _hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        y_pred = np.array([idx_to_label[i] for i in np.argmax(logits, axis=-1)])
        y_true = np.array([idx_to_label[i] for i in labels])
        return compute_metrics(y_true, y_pred, y_proba=_softmax(logits),
                               labels=classes_sorted)
    return _hf_compute_metrics


def _phase(model, tokenizer, tr_texts, tr_labels, va_texts, va_labels,
           n_classes, idx_to_label, classes_sorted, phase_dir):
    """Train `model` in place, early-stopping by QWK on (va); returns the trainer."""
    from transformers import DataCollatorWithPadding

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
        learning_rate=LR, warmup_ratio=WARMUP_RATIO, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="no", logging_steps=50, seed=SEED,
        report_to=[], dataloader_num_workers=NUM_WORKERS, fp16=USE_FP16, bf16=USE_BF16,
    )
    WeightedTrainer = _make_weighted_trainer_class(cw)
    keeper = make_best_model_keeper(metric="eval_qwk", greater_is_better=True, patience=2)
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
        **trainer_tokenizer_kwarg(tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_metrics_fn(idx_to_label, classes_sorted), callbacks=[keeper])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()
    if keeper.best_state is not None:
        model.load_state_dict(keeper.best_state)
    return trainer


def _predict(trainer, tokenizer, test_texts, test_labels, idx_to_label):
    """Score a test set -> (y_true, y_pred, proba), with labels paired by the Trainer."""
    test_ds = _TextDataset(
        tokenizer(test_texts, truncation=True, padding=False, max_length=MAX_LEN),
        test_labels)
    logits, out_labels = predict_with_labels(trainer, test_ds)
    if not np.array_equal(np.sort(out_labels),
                          np.sort(np.asarray(test_labels).astype(np.int64))):
        raise RuntimeError("predict label_ids are not a permutation of the "
                           "test labels — dataset/collator drift, refusing to score.")
    y_pred = np.array([idx_to_label[i] for i in np.argmax(logits, axis=-1)])
    y_true = np.array([idx_to_label[i] for i in out_labels])
    return y_true, y_pred, _softmax(logits)


def _fit_and_score(phases, test_texts, test_labels, tokenizer, n_classes,
                   idx_to_label, classes_sorted, out_dir):
    """Train one model through `phases` [(tr_texts, tr_y, va_texts, va_y), ...]
    in order, then score the test set."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_classes)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, (tr_t, tr_y, va_t, va_y) in enumerate(phases, 1):
        if len(phases) > 1:
            print(f"    [seq] phase {k}/{len(phases)}")
        trainer = _phase(model, tokenizer, tr_t, tr_y, va_t, va_y, n_classes,
                         idx_to_label, classes_sorted, out_dir / f"phase{k}")
    result = _predict(trainer, tokenizer, test_texts, test_labels, idx_to_label)

    del model, trainer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)
    return result


def method_tag(compliance, scale, mode, target_frac, src, tgt) -> str:
    PCT = int(round(target_frac * 100))
    seq = "_seq" if mode == "sequential" else ""
    return (f"xfer_bert_textualized_full_{compliance}{scale_infix(scale)}{seq}"
            f"_tgt{PCT}_{src}2{tgt}")


def run_transfer(data, models_dir, *, target_frac=0.0, test_frac=0.2,
                 val_frac=0.15, n_boot=1000, seed=SEED, mode="pool"):
    """Train on the source (plus any target adaptation rows) and score the target
    test block. Target labels never reach model selection at p=0."""
    from transformers import AutoTokenizer

    method = method_tag(data.compliance_mode, data.scale, mode, target_frac,
                        data.src_name, data.tgt_name)
    if mode == "sequential" and target_frac <= 0:
        raise ValueError("--transfer-mode sequential needs --target-frac > 0.")

    print(f"\n{'=' * 70}")
    print(f"PLM: {data.src_name} -> {data.tgt_name}  [{mode}]")
    print(f"  target_frac={target_frac}  test_frac={test_frac}")
    print(f"  classes={data.labels}  device={DEVICE}  model={MODEL_NAME}")
    print(f"{'=' * 70}")

    tr_pos, va_pos = stratified_source_split(data.y_src, val_frac, seed)
    print(f"  source split: train={len(tr_pos)}  val(early-stop)={len(va_pos)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    report_token_lengths(data.src_texts, tokenizer, f"{data.src_name}/source")
    report_token_lengths(data.tgt_texts, tokenizer, f"{data.tgt_name}/target")

    t0 = time.time()
    adapt_idx, test_idx = split_target_fewshot(data.y_tgt, target_frac, test_frac, seed)
    print(f"  target split: adapt={len(adapt_idx)}  test={len(test_idx)}")
    src_va = (data.src_texts.iloc[va_pos].tolist(), data.y_src[va_pos])
    if mode == "sequential":
        sub_tr, sub_va = stratified_source_split(data.y_tgt[adapt_idx], val_frac, seed)
        a_tr, a_va = adapt_idx[sub_tr], adapt_idx[sub_va]
        phases = [
            (data.src_texts.iloc[tr_pos].tolist(), data.y_src[tr_pos].astype(int), *src_va),
            (data.tgt_texts.iloc[a_tr].tolist(), data.y_tgt[a_tr].astype(int),
             data.tgt_texts.iloc[a_va].tolist(), data.y_tgt[a_va]),
        ]
    else:
        phases = [(
            data.src_texts.iloc[tr_pos].tolist() + data.tgt_texts.iloc[adapt_idx].tolist(),
            np.concatenate([data.y_src[tr_pos], data.y_tgt[adapt_idx]]).astype(int),
            *src_va,
        )]
    y_true, y_pred, y_proba = _fit_and_score(
        phases, data.tgt_texts.iloc[test_idx].tolist(), data.y_tgt[test_idx],
        tokenizer, data.n_classes, data.idx_to_label, data.labels, models_dir / method)

    point = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=data.labels)
    std = bootstrap_target_std(y_true, y_pred, y_proba, labels=data.labels,
                               n_boot=n_boot, seed=seed)
    elapsed = time.time() - t0
    print(
        f"\n  TARGET TEST ({data.tgt_name}, n={len(y_true)}, +/-=bootstrap std over {n_boot}):\n"
        f"    acc={point['accuracy']:.3f}+/-{std['accuracy']:.3f}  "
        f"bal_acc={point['balanced_acc']:.3f}+/-{std['balanced_acc']:.3f}  "
        f"macroF1={point['macro_f1']:.3f}+/-{std['macro_f1']:.3f}\n"
        f"    qwk={point['qwk']:.3f}+/-{std['qwk']:.3f}  "
        f"mae={point['mae']:.3f}+/-{std['mae']:.3f}  "
        f"ll={point['log_loss']:.3f}+/-{std['log_loss']:.3f}"
    )
    print(f"  elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    return method, point, std


def load_transfer_data(args):
    if args.pool_sources:
        return load_and_prepare_pooled(
            parse_pool_sources(args.pool_sources), args.tgt_name, args.target,
            compliance_mode=args.compliance, scale=args.rating_scale,
            pool_name=args.pool_name)
    return load_and_prepare(
        args.source, args.target, args.src_name, args.tgt_name,
        compliance_mode=args.compliance, scale=args.rating_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="ModernBERT cross-state transfer.")
    add_transfer_args(parser)
    parser.add_argument("--transfer-mode", choices=("pool", "sequential"), default="pool",
                        help="'pool': one phase on source + adaptation rows. "
                             "'sequential': source pretrain, then target fine-tune.")
    args = parser.parse_args()
    configure_verbosity(args.verbose)
    _resolve_transfer_output(args, "transfer_plm")
    src = args.pool_name if args.pool_sources else args.src_name

    try:
        data = load_transfer_data(args)
        models_dir = run_artifact_dir(args.output) / "plm_transfer_models"
        method, point, std = run_transfer(
            data, models_dir, target_frac=args.target_frac,
            test_frac=args.target_test_frac, val_frac=args.val_frac,
            n_boot=args.n_bootstrap, seed=args.seed, mode=args.transfer_mode)
        log_transfer_result(method, point, std, output_path=args.output, notes=method,
                            source=data.src_name, target=data.tgt_name, classes=data.labels)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        log_failed_transfer(
            method_tag(args.compliance, args.rating_scale, args.transfer_mode,
                       args.target_frac, src, args.tgt_name),
            output_path=args.output, source=src, target=args.tgt_name,
            error=f"{type(exc).__name__}: {exc}")
        raise
    print(f"\nDone. See {args.output} for transfer scores.")


if __name__ == "__main__":
    main()
