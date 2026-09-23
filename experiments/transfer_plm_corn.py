"""ModernBERT for cross-state transfer with a rank-consistent ordinal (CORN) head.

The head has K-1 units, unit k modelling P(y > k | y > k-1) (Shi, Cao and Raschka
2021, arXiv:2111.08851). Splits, regimes and scoring match `transfer_plm`; only
the head, the loss and the decode differ.

    python transfer_plm_corn.py --pool-sources NC=data/nc_records_cleaned_raw.csv ... \\
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
    log_failed_transfer,
    log_transfer_result,
    scale_infix,
    split_target_fewshot,
    stratified_source_split,
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
from transfer_plm import load_transfer_data  # noqa: E402
from utils import SEED, compute_metrics, run_artifact_dir  # noqa: E402


def _corn_decode(logits, idx_to_label, n_classes):
    """(N, K-1) CORN logits -> (labels, per-class probabilities)."""
    from coral_pytorch.dataset import corn_label_from_logits

    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    cum = torch.cumprod(torch.sigmoid(logits_t), dim=1)  # P(y > k)

    parts = [1.0 - cum[:, :1]]
    if n_classes > 2:
        parts.append(cum[:, :-1] - cum[:, 1:])
    parts.append(cum[:, -1:])
    proba = torch.clamp(torch.cat(parts, dim=1), min=0.0)
    proba = (proba / proba.sum(dim=1, keepdim=True)).cpu().numpy()

    pred_idx = corn_label_from_logits(logits_t).cpu().numpy().astype(int)
    y_pred = np.array([idx_to_label[int(i)] for i in pred_idx])
    return y_pred, proba


def corn_loss_weighted(logits, y_train, num_classes, class_weights=None):
    """CORN loss with per-class sample weights, normalised by the summed weight.
    Uniform weights reduce to `coral_pytorch.losses.corn_loss`."""
    import torch.nn.functional as F

    if class_weights is None:
        from coral_pytorch.losses import corn_loss
        return corn_loss(logits, y_train, num_classes)

    w_all = class_weights.to(logits.device)[y_train]
    losses = 0.0
    denom = 0.0
    for task_index in range(num_classes - 1):
        label_mask = y_train > task_index - 1
        train_labels = (y_train[label_mask] > task_index).to(torch.int64)
        if len(train_labels) < 1:
            continue
        pred = logits[label_mask, task_index]
        w = w_all[label_mask]
        per = -(F.logsigmoid(pred) * train_labels
                + (F.logsigmoid(pred) - pred) * (1 - train_labels))
        losses = losses + torch.sum(per * w)
        denom = denom + w.sum()
    return losses / denom


def _assert_corn_equivalence():
    """Check that uniform weights reproduce `coral_pytorch`'s `corn_loss`."""
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
            f"coral_pytorch.corn_loss ({ref.item():.8f} vs {ours.item():.8f}).")


def _make_corn_trainer_class(num_classes: int, class_weights=None):
    from transformers import Trainer

    if class_weights is not None:
        _assert_corn_equivalence()

    class CornTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = corn_loss_weighted(outputs.logits, labels, num_classes,
                                      class_weights=class_weights)
            return (loss, outputs) if return_outputs else loss

    return CornTrainer


def _corn_phase(model, tr_texts, tr_labels, va_texts, va_labels, tokenizer,
                n_classes, idx_to_label, classes_sorted, phase_dir):
    """Train `model` in place, early-stopping by QWK on (va); returns the trainer."""
    from transformers import DataCollatorWithPadding

    tr_ds = _TextDataset(
        tokenizer(tr_texts, truncation=True, padding=False, max_length=MAX_LEN), tr_labels)
    va_ds = _TextDataset(
        tokenizer(va_texts, truncation=True, padding=False, max_length=MAX_LEN), va_labels)
    phase_dir.mkdir(parents=True, exist_ok=True)
    args = make_training_arguments(
        output_dir=str(phase_dir), num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=EVAL_BATCH_SIZE,
        group_by_length=GROUP_BY_LENGTH, learning_rate=LR, warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01, eval_strategy="epoch", save_strategy="no", logging_steps=50,
        seed=SEED, report_to=[], dataloader_num_workers=NUM_WORKERS,
        fp16=USE_FP16, bf16=USE_BF16,
    )

    def _hf_compute_metrics(eval_pred):
        logits = np.asarray(eval_pred.predictions if hasattr(eval_pred, "predictions")
                            else eval_pred[0])
        labels = np.asarray(eval_pred.label_ids if hasattr(eval_pred, "label_ids")
                            else eval_pred[1])
        y_pred, proba = _corn_decode(logits, idx_to_label, n_classes)
        y_true = np.array([idx_to_label[int(i)] for i in labels])
        return compute_metrics(y_true, y_pred, y_proba=proba, labels=classes_sorted)

    cw = torch.as_tensor(balanced_class_weights(np.asarray(tr_labels), n_classes),
                         dtype=torch.float32)
    CornTrainer = _make_corn_trainer_class(n_classes, class_weights=cw)
    keeper = make_best_model_keeper(metric="eval_qwk", greater_is_better=True, patience=2)
    trainer = CornTrainer(
        model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
        **trainer_tokenizer_kwarg(tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_hf_compute_metrics, callbacks=[keeper],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()
    if keeper.best_state is not None:
        model.load_state_dict(keeper.best_state)
    return trainer


def _predict_corn(trainer, tokenizer, test_texts, test_labels, idx_to_label, n_classes):
    """Score a test set -> (y_true, y_pred, proba), with labels paired by the Trainer."""
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


def method_tag(compliance, scale, mode, target_frac, src, tgt) -> str:
    PCT = int(round(target_frac * 100))
    seq = "_seq" if mode == "sequential" else ""
    return f"xfer_bert_corn_{compliance}{scale_infix(scale)}{seq}_tgt{PCT}_{src}2{tgt}"


def run_transfer_corn(data, models_dir, *, target_frac=0.0, test_frac=0.2,
                      val_frac=0.15, n_boot=1000, seed=SEED, mode="pool"):
    """`transfer_plm.run_transfer` with the CORN head, loss and decode."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    method = method_tag(data.compliance_mode, data.scale, mode, target_frac,
                        data.src_name, data.tgt_name)
    if mode == "sequential" and target_frac <= 0:
        raise ValueError("--transfer-mode sequential needs --target-frac > 0.")

    print(f"\n{'=' * 70}")
    print(f"CORN PLM [{mode}]: {data.src_name} -> {data.tgt_name}")
    print(f"  target_frac={target_frac}  test_frac={test_frac}")
    print(f"  classes={data.labels}  device={DEVICE}  model={MODEL_NAME}  "
          f"(K-1={data.n_classes - 1} CORN units)")
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

    out_dir = models_dir / method
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=data.n_classes - 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, (tr_t, tr_y, va_t, va_y) in enumerate(phases, 1):
        if len(phases) > 1:
            print(f"    [corn-seq] phase {k}/{len(phases)}")
        trainer = _corn_phase(model, tr_t, tr_y, va_t, va_y, tokenizer, data.n_classes,
                              data.idx_to_label, data.labels, out_dir / f"phase{k}")
    y_true, y_pred, y_proba = _predict_corn(
        trainer, tokenizer, data.tgt_texts.iloc[test_idx].tolist(), data.y_tgt[test_idx],
        data.idx_to_label, data.n_classes)
    del model, trainer
    shutil.rmtree(out_dir, ignore_errors=True)

    point = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=data.labels)
    std = bootstrap_target_std(y_true, y_pred, y_proba, labels=data.labels,
                               n_boot=n_boot, seed=seed)
    elapsed = time.time() - t0
    print(
        f"\n  TARGET TEST ({data.tgt_name}, n={len(y_true)}, +/-=bootstrap std):\n"
        f"    qwk={point['qwk']:.3f}+/-{std['qwk']:.3f}  "
        f"bal_acc={point['balanced_acc']:.3f}+/-{std['balanced_acc']:.3f}  "
        f"mae={point['mae']:.3f}+/-{std['mae']:.3f}\n"
        f"  elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)"
    )
    return method, point, std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ModernBERT cross-state transfer with a CORN ordinal head.")
    add_transfer_args(parser)
    parser.add_argument("--transfer-mode", choices=("pool", "sequential"), default="pool",
                        help="'pool': one phase on source + adaptation rows. "
                             "'sequential': source pretrain, then target fine-tune.")
    args = parser.parse_args()
    configure_verbosity(args.verbose)
    _resolve_transfer_output(args, "transfer_plm_corn")
    src = args.pool_name if args.pool_sources else args.src_name

    try:
        data = load_transfer_data(args)
        method, point, std = run_transfer_corn(
            data, run_artifact_dir(args.output) / "plm_corn_transfer_models",
            target_frac=args.target_frac, test_frac=args.target_test_frac,
            val_frac=args.val_frac, n_boot=args.n_bootstrap, seed=args.seed,
            mode=args.transfer_mode)
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
