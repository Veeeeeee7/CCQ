"""ModernBERT for within-state prediction, plus the shared HuggingFace plumbing.

Supplies the MBERT arm of the within-state experiment, running 5-fold CV over the
shared fold indices on the serialized text view. Also houses the Trainer plumbing
(`make_training_arguments`, `make_best_model_keeper`, `predict_with_labels`, the
class-weighted Trainer subclass) that the cross-state text and LLM methods import.

    python llm.py --input data/wi_records_cleaned_raw.csv --output results.csv
    python llm.py --models bert_textualized_full --compliance verbose
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from text_serialization import serialize_dataframe  # noqa: E402

from utils import (  # noqa: E402
    run_artifact_dir,
    SEED,
    TARGET_COL,
    add_io_args,
    compute_metrics,
    configure_verbosity,
    get_folds,
    load_data,
    maybe_remap,
    log_results,
)


def _detect_device() -> str:
    """Pick the best available accelerator. Priority: cuda → mps → cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _detect_device()
# bf16 where supported, else fp16 on CUDA, else full precision.
# Override with CCQ_PLM_PRECISION=bf16|fp16|off.
_prec = os.environ.get("CCQ_PLM_PRECISION", "").lower()
if not _prec:
    if DEVICE == "cuda":
        _prec = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    else:
        _prec = "off"
USE_BF16 = (DEVICE == "cuda" and _prec == "bf16")
USE_FP16 = (DEVICE == "cuda" and _prec == "fp16")
# 0 workers on MPS to avoid macOS fork/multiprocessing issues.
NUM_WORKERS = 0 if DEVICE == "mps" else int(os.environ.get("CCQ_PLM_WORKERS", "4"))
MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LEN = 8192      # ModernBERT's full window
# The train batch is an optimization hyperparameter and changes results; the
# eval batch affects speed only.
BATCH_SIZE = int(os.environ.get("CCQ_PLM_BATCH", "8"))
EVAL_BATCH_SIZE = int(os.environ.get("CCQ_PLM_EVAL_BATCH", "16"))
# Sort-by-length batching, to limit padding waste at 8k context.
GROUP_BY_LENGTH = os.environ.get("CCQ_PLM_GROUP_BY_LENGTH", "1") == "1"
EPOCHS = 4
LR = 2e-5
WARMUP_RATIO = 0.1
# Fraction of each training fold held out for early stopping.
VAL_FRAC = 0.15
# TF32 matmuls on Ampere+.
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# -----------------------------------------------------------------------------
# Version-tolerant TrainingArguments
# -----------------------------------------------------------------------------
# Drops kwargs the installed transformers does not accept, but only those on the
# performance-only allow-list below. Anything that would change training
# semantics still raises.
_TA_DROPPABLE = {"group_by_length", "dataloader_num_workers"}  # perf-only knobs


def trainer_tokenizer_kwarg(tokenizer) -> dict:
    """Return the tokenizer kwarg under whichever name the installed Trainer
    accepts (`processing_class` or `tokenizer`)."""
    import inspect
    from transformers import Trainer
    params = inspect.signature(Trainer.__init__).parameters
    key = "processing_class" if "processing_class" in params else "tokenizer"
    return {key: tokenizer}


def make_training_arguments(**kwargs):
    import inspect
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    unsupported = [k for k in kwargs if k not in sig.parameters]
    bad = [k for k in unsupported if k not in _TA_DROPPABLE]
    if bad:
        raise TypeError(
            f"TrainingArguments (installed transformers version) does not accept "
            f"{bad}, and dropping them would change training semantics. Update the "
            f"call site or pin a compatible transformers.")
    for k in unsupported:
        kwargs.pop(k)
        print(f"  [compat] transformers' TrainingArguments lacks '{k}' "
              f"(perf-only) -- dropped")
    return TrainingArguments(**kwargs)


# -----------------------------------------------------------------------------
# Token-length diagnostics.
# -----------------------------------------------------------------------------
def report_token_lengths(texts: pd.Series, tokenizer, label: str) -> None:
    lens = [len(tokenizer.encode(t, add_special_tokens=True, truncation=False))
            for t in texts.head(min(len(texts), 500))]  # sample for speed
    lens = np.array(lens)
    pct_over = float((lens > MAX_LEN).mean()) * 100
    print(
        f"  [{label}] token-length on sample of {len(lens)}: "
        f"mean={lens.mean():.0f}  median={np.median(lens):.0f}  "
        f"p95={np.percentile(lens, 95):.0f}  max={lens.max():.0f}  "
        f">MAX_LEN({MAX_LEN}): {pct_over:.1f}%"
    )


# -----------------------------------------------------------------------------
# Dataset wrapper: tokenize once per fold.
# -----------------------------------------------------------------------------
class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: torch.as_tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.as_tensor(self.labels[i])
        return item


# -----------------------------------------------------------------------------
# Ordered predict, shared with the transfer modules.
# -----------------------------------------------------------------------------
def predict_with_labels(trainer, dataset):
    """Run ``trainer.predict`` and return ``(logits, label_ids)`` as paired by the
    Trainer.

    Prediction order is sampler-dependent, so ``.predictions`` must never be
    zipped against a caller-held label array. ``label_ids`` comes through the same
    dataloader and is therefore correctly aligned under any sampler.
    """
    out = trainer.predict(dataset)
    if out.label_ids is None:
        raise RuntimeError(
            "trainer.predict returned no label_ids; refusing to pair "
            "predictions with caller-held labels (order is sampler-dependent).")
    return np.asarray(out.predictions), np.asarray(out.label_ids)


# -----------------------------------------------------------------------------
# Class-weighted Trainer
# -----------------------------------------------------------------------------
def _make_weighted_trainer_class(class_weights: torch.Tensor):
    """Build a Trainer subclass using class-weighted CE, constructed per fold so
    the weights match the train fold's label distribution."""
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(logits.device)
            )
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


# -----------------------------------------------------------------------------
# Best-model-in-RAM keeper (shared with transfer_plm.py)
# -----------------------------------------------------------------------------
def make_best_model_keeper(metric: str = "eval_qwk",
                           greater_is_better: bool = True, patience: int = 2):
    """TrainerCallback that keeps the best epoch's weights in CPU RAM and stops
    after ``patience`` non-improving evals.

    Lets the Trainer run with ``save_strategy="no"``, at the cost of one model's
    worth of CPU RAM. The caller restores ``keeper.best_state`` after
    ``trainer.train()``. Imported lazily to keep importing this module cheap.
    """
    from transformers import TrainerCallback

    class _BestModelInRAM(TrainerCallback):
        def __init__(self):
            self.metric = metric
            self.greater = greater_is_better
            self.patience = patience
            self.best = None
            self.best_state = None
            self.wait = 0

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
            if metrics is None or model is None:
                return
            val = metrics.get(self.metric)
            if val is None:  # tolerate an un-prefixed key just in case
                val = metrics.get(self.metric.removeprefix("eval_"))
            if val is None:
                return
            improved = (self.best is None
                        or (val > self.best if self.greater else val < self.best))
            if improved:
                self.best = val
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    control.should_training_stop = True

    return _BestModelInRAM()


# -----------------------------------------------------------------------------
# Single fold
# -----------------------------------------------------------------------------
def _train_one_fold(
    fold_idx: int,
    train_texts: list[str],
    train_labels: np.ndarray,
    val_texts: list[str],
    val_labels: np.ndarray,
    tokenizer,
    n_classes: int,
    idx_to_label: dict,
    classes_sorted: list,
    models_dir: Path,
    score_texts: "list[str] | None" = None,
    score_labels: "np.ndarray | None" = None,
) -> dict:
    """Fine-tune ModernBERT for one fold; return metrics on the scored block.

    `val_*` is the early-stopping slice. `score_*`, when given, is the disjoint
    block the metrics are computed on; when omitted the two coincide.
    """
    import shutil

    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments,
        DataCollatorWithPadding,
    )

    # Pre-tokenize once per fold (small dataset; no benefit to per-epoch).
    train_enc = tokenizer(
        train_texts, truncation=True, padding=False, max_length=MAX_LEN,
    )
    val_enc = tokenizer(
        val_texts, truncation=True, padding=False, max_length=MAX_LEN,
    )
    train_ds = _TextDataset(train_enc, train_labels)
    val_ds = _TextDataset(val_enc, val_labels)

    # Per-fold class weights (balanced, computed on train labels only).
    # compute_class_weight errors if any class in `classes` is absent from y,
    # which happens when a fold's train set misses an ultra-rare ordinal class
    # (WI 5-star 1-star: 2 providers total). Weight only the classes actually
    # present and leave absent classes at weight 1.0, so the model still trains.
    train_labels_arr = np.asarray(train_labels)
    present = np.unique(train_labels_arr)
    cw_present = compute_class_weight(
        class_weight="balanced",
        classes=present,
        y=train_labels_arr,
    )
    cw = np.ones(n_classes, dtype=np.float32)
    cw[present.astype(int)] = cw_present
    class_weights = torch.as_tensor(cw, dtype=torch.float32)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes,
    )

    out_dir = models_dir / f"fold_{fold_idx}"
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
        # Write NOTHING to disk during training. Best-epoch selection (by val QWK)
        # and patience-2 early stopping are handled entirely in CPU RAM by the
        # keeper below. Checkpointing per epoch would write roughly 0.6 GB each
        # time, so a 5-fold run would leave several GB on the scratch quota for a
        # model that is reloaded from RAM anyway.
        save_strategy="no",
        logging_steps=50,
        seed=SEED,
        report_to=[],          # no wandb / tensorboard by default
        dataloader_num_workers=NUM_WORKERS,
        fp16=USE_FP16,
        bf16=USE_BF16,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # compute_metrics adapter for HF Trainer: map back to ordinal labels and
    # softmax the logits to get probabilities for log_loss.
    def _hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        y_proba = exp / exp.sum(axis=-1, keepdims=True)
        pred_idx = np.argmax(logits, axis=-1)
        y_pred = np.array([idx_to_label[i] for i in pred_idx])
        y_true = np.array([idx_to_label[i] for i in labels])
        return compute_metrics(
            y_true, y_pred, y_proba=y_proba, labels=classes_sorted,
        )

    WeightedTrainer = _make_weighted_trainer_class(class_weights)
    keeper = make_best_model_keeper(metric="eval_qwk", greater_is_better=True,
                                    patience=2)
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        **trainer_tokenizer_kwarg(tokenizer),
        data_collator=collator,
        compute_metrics=_hf_compute_metrics,
        callbacks=[keeper],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()

    # Restore the best-on-val weights (kept in RAM, never on disk) before scoring.
    if keeper.best_state is not None:
        model.load_state_dict(keeper.best_state)

    # Score on the held-out CV fold, which the early-stopping slice is disjoint
    # from. Falls back to the early-stopping set only when no scoring block was
    # supplied.
    if score_texts is not None:
        score_enc = tokenizer(
            score_texts, truncation=True, padding=False, max_length=MAX_LEN,
        )
        score_ds = _TextDataset(score_enc, score_labels)
    else:
        score_ds = val_ds

    eval_out = trainer.evaluate(eval_dataset=score_ds)
    # HF prefixes keys with 'eval_'; strip and keep only our metric keys.
    metric_keys = {"accuracy", "balanced_acc", "macro_f1", "micro_f1",
                   "qwk", "mae", "log_loss"}
    m = {k.removeprefix("eval_"): v for k, v in eval_out.items()
         if k.removeprefix("eval_") in metric_keys}

    del model, trainer, keeper
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    # With save_strategy="no" this dir holds at most a few KB of run metadata,
    # but drop it so nothing lingers under the scratch tree.
    shutil.rmtree(out_dir, ignore_errors=True)
    return m


# -----------------------------------------------------------------------------
# Runners
# -----------------------------------------------------------------------------
def _run_plm_cv(
    df: pd.DataFrame,
    folds: list[dict],
    texts: pd.Series,
    label: str,
    models_dir: Path,
) -> list[dict]:
    """Generic 5-fold runner. `texts` is one string per row of df; splits are
    positional indices from the shared folds."""
    from transformers import AutoTokenizer

    n_classes = df[TARGET_COL].nunique()
    classes_sorted = sorted(df[TARGET_COL].unique())
    label_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    print(f"\n{'=' * 60}")
    print(f"Training {label.upper()} ({n_classes} classes, device={DEVICE})")
    print(f"  model={MODEL_NAME}, max_len={MAX_LEN}, lr={LR}, "
          f"batch={BATCH_SIZE}, epochs={EPOCHS}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    report_token_lengths(texts, tokenizer, label)

    fold_metrics: list[dict] = []
    t0 = time.time()
    y_all = df[TARGET_COL].map(label_to_idx).to_numpy()

    for fold in folds:
        fold_idx = fold["fold"]
        train_idx, val_idx = fold["train_idx"], fold["val_idx"]

        # INNER EARLY-STOPPING SPLIT. A stratified VAL_FRAC slice is carved out
        # of the TRAINING fold and used for model selection; val_idx is untouched
        # until scoring. Selecting the best epoch on the scored fold would bias
        # its reported metrics optimistically.
        # Imported lazily: transfer_common pulls in the whole transfer CLI
        # surface, and llm.py is imported by within-state-only entry points.
        from transfer_common import stratified_source_split
        train_idx = np.asarray(train_idx)
        inner_tr, inner_va = stratified_source_split(
            y_all[train_idx], VAL_FRAC, SEED)
        fit_idx = train_idx[inner_tr]
        es_idx = train_idx[inner_va]

        train_texts = texts.iloc[fit_idx].tolist()
        es_texts = texts.iloc[es_idx].tolist()
        val_texts = texts.iloc[val_idx].tolist()
        train_labels = y_all[fit_idx]
        es_labels = y_all[es_idx]
        val_labels = y_all[val_idx]

        fold_t0 = time.time()
        m = _train_one_fold(
            fold_idx, train_texts, train_labels, es_texts, es_labels,
            tokenizer, n_classes, idx_to_label, classes_sorted,
            models_dir / label,
            score_texts=val_texts, score_labels=val_labels,
        )
        fold_metrics.append(m)
        print(
            f"  Fold {fold_idx}: "
            f"acc={m['accuracy']:.3f}  bal_acc={m['balanced_acc']:.3f}  "
            f"macroF1={m['macro_f1']:.3f}  qwk={m['qwk']:.3f}  "
            f"mae={m['mae']:.3f}  ll={m['log_loss']:.3f}  "
            f"(elapsed={time.time() - fold_t0:.0f}s)"
        )

    elapsed = time.time() - t0
    agg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    std = {k: np.std([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    print(
        f"  MEAN:   "
        f"acc={agg['accuracy']:.3f}±{std['accuracy']:.3f}  "
        f"bal_acc={agg['balanced_acc']:.3f}±{std['balanced_acc']:.3f}  "
        f"macroF1={agg['macro_f1']:.3f}±{std['macro_f1']:.3f}  "
        f"qwk={agg['qwk']:.3f}±{std['qwk']:.3f}  "
        f"mae={agg['mae']:.3f}±{std['mae']:.3f}  "
        f"ll={agg['log_loss']:.3f}±{std['log_loss']:.3f}"
    )
    print(f"  Total elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    return fold_metrics


def run_curriculum_only(
    df: pd.DataFrame, folds: list[dict], models_dir: Path, **_
) -> list[dict]:
    """Just the raw curriculum cell as input. NaN/empty → empty string."""
    if "curriculum" not in df.columns:
        raise KeyError("Expected a 'curriculum' column in the raw dataframe.")
    texts = df["curriculum"].fillna("").astype(str)
    empty = (texts.str.strip() == "").mean()
    print(f"  curriculum-only: {empty:.1%} of rows have empty curriculum text.")
    return _run_plm_cv(df, folds, texts, "bert_curriculum_only", models_dir)


def run_textualized_full(
    df: pd.DataFrame, folds: list[dict], models_dir: Path,
    compliance_mode: str = "verbose",
) -> list[dict]:
    """Full row serialization (sections + skip-defaults)."""
    print(f"  textualizing rows (compliance_mode={compliance_mode!r})...")
    texts = serialize_dataframe(df, compliance_mode=compliance_mode)
    return _run_plm_cv(df, folds, texts,
                       f"bert_textualized_full_{compliance_mode}", models_dir)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
MODEL_RUNNERS: dict[str, Callable] = {
    "bert_curriculum_only":  run_curriculum_only,
    "bert_textualized_full": run_textualized_full,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_io_args(parser)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["bert_curriculum_only", "bert_textualized_full"],
        choices=list(MODEL_RUNNERS.keys()),
        help="Which run(s) to execute",
    )
    parser.add_argument(
        "--compliance",
        choices=("verbose", "summary", "abnormal_only"),
        default="verbose",
        help="Compliance section rendering for textualized_full (default verbose). "
             "Switch to 'summary' if verbose underperforms.",
    )
    args = parser.parse_args()
    configure_verbosity(args.verbose)

    df = load_data(args.input)
    df = maybe_remap(df, args.remap_state, allow_identity=args.allow_identity, scale=args.rating_scale)
    folds = get_folds(df, folds_path=args.folds)

    # Fine-tuned checkpoints land next to the results file.
    models_dir = run_artifact_dir(args.output) / "plm_models"

    for model_name in args.models:
        runner = MODEL_RUNNERS[model_name]
        if model_name == "bert_textualized_full":
            fold_results = runner(df, folds, models_dir, compliance_mode=args.compliance)
            log_name = f"bert_textualized_full_{args.compliance}"
        else:
            fold_results = runner(df, folds, models_dir)
            log_name = model_name
        log_results(log_name, fold_results, output_path=args.output, notes=log_name)

    print(f"\nDone. See {args.output} for aggregated scores.")


if __name__ == "__main__":
    main()