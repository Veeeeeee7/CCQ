"""Qwen3-4B with a trained classification head, for cross-state and within-state runs.

  --adapt head            xfer_qwen_cls_*        frozen backbone, K-way head only
  --adapt head --rag      xfer_qwen_cls_rag_*    same, inputs prefixed with
                                                 per-class medoid exemplars
  --adapt lora            xfer_qwen_cls_lora_*   LoRA adapters plus head

The input is the serialized row and the output is the head's K-way softmax; there
is no prompting or generation. Cross-state runs pretrain on the source pool and,
at p > 0, fine-tune on the target adaptation rows. `--within-cv K` runs K-fold CV
within one state instead.

    python transfer_llm_cls.py --pool-sources WI=data/wi_records_cleaned_raw.csv ... \\
        --target data/nc_records_cleaned_raw.csv --tgt-name NC --rating-scale 5star \\
        --adapt head --target-frac 0.2
    python transfer_llm_cls.py --within-cv 5 --adapt lora \\
        --source data/nc_records_cleaned_raw.csv --target data/nc_records_cleaned_raw.csv \\
        --src-name NC --tgt-name NC --rating-scale 5star
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

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
    _make_weighted_trainer_class,
    make_best_model_keeper,
    make_training_arguments,
    predict_with_labels,
    trainer_tokenizer_kwarg,
    DEVICE,
    USE_FP16,
    USE_BF16,
    NUM_WORKERS,
    GROUP_BY_LENGTH,
)
from llm_common import class_medoid_exemplar_indices  # noqa: E402
from utils import SEED, compute_metrics, run_artifact_dir  # noqa: E402

# A local directory (fetch with download_qwen.py); --model and $LLM_CLS_MODEL override.
DEFAULT_CLS_LLM_PATH = (
    os.environ.get("LLM_CLS_MODEL")
    or str(Path(__file__).resolve().parent / "models" / "qwen3_4b")
)
MAX_LEN = int(os.environ.get("CCQ_LLMCLS_MAX_LEN", "8192"))
BATCH_SIZE = int(os.environ.get("CCQ_LLMCLS_BATCH", "4"))
EVAL_BATCH_SIZE = int(os.environ.get("CCQ_LLMCLS_EVAL_BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("CCQ_LLMCLS_ACCUM", "4"))
EPOCHS = int(os.environ.get("CCQ_LLMCLS_EPOCHS", "4"))
HEAD_LR = float(os.environ.get("CCQ_LLMCLS_HEAD_LR", "1e-3"))
LORA_LR = float(os.environ.get("CCQ_LLMCLS_LORA_LR", "1e-4"))
WARMUP_RATIO = 0.1

# -----------------------------------------------------------------------------
# Caches (under the run's artifact directory)
# -----------------------------------------------------------------------------
# PRETRAIN: the pooled-source pretrain does not depend on p, so it is computed once.
# FEATURE: with --adapt head the backbone is frozen, so pooled hidden states are
# extracted once and the head trains on them. Exact only without active dropout.
PRETRAIN_CACHE = os.environ.get("CCQ_LLMCLS_PRETRAIN_CACHE", "1") == "1"
FEATURE_CACHE = os.environ.get("CCQ_LLMCLS_FEATURE_CACHE", "1") == "1"
# Tolerance for the check that cached features reproduce the model's logits:
# atol + rtol * |logits|max, since bf16 error scales with the logit magnitude.
FEATURE_CHECK_ATOL = float(os.environ.get("CCQ_LLMCLS_FEATURE_ATOL", "2e-2"))
FEATURE_CHECK_RTOL = float(os.environ.get("CCQ_LLMCLS_FEATURE_RTOL", "5e-2"))
_FEATURE_CACHE_VERSION = 2


def _digest_texts(texts, *scalars) -> str:
    """md5 over the exact strings plus the scalars that affect the result."""
    h = hashlib.md5()
    for s in scalars:
        h.update(repr(s).encode("utf-8"))
        h.update(b"\x1f")
    h.update(b"\x1e")
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


_TMP_SEQ = itertools.count()


def _atomic_save(obj, path: Path, saver) -> None:
    """Write via a unique temp file + os.replace, so a killed job never leaves a
    half-written cache entry. ``saver(obj, tmp)`` must write to exactly ``tmp``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}_{next(_TMP_SEQ)}")
    try:
        saver(obj, tmp)
        if not tmp.exists():
            raise RuntimeError(
                f"saver did not write to the temp path it was given ({tmp}); "
                f"an atomic write is impossible. Savers must not rewrite the "
                f"path (np.save appends '.npy' — pass an open file handle).")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _np_save(arr, path) -> None:
    """np.save to an exact path (a file handle stops it appending '.npy')."""
    with open(path, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)


# -----------------------------------------------------------------------------
# RAG input construction
# -----------------------------------------------------------------------------
def _embed_texts(texts, model_name, device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device=device)
    emb = np.asarray(m.encode(list(texts), batch_size=64,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False))
    del m
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return emb


def _exemplar_block(pool_texts, pool_label_values, rep_idx, max_chars):
    blocks = [f"Example — rating {pool_label_values[j]}:\n{pool_texts[j][:max_chars]}"
              for j in rep_idx]
    return "Labeled reference providers:\n\n" + "\n\n".join(blocks)


def _rag_wrap(query_texts, pool_texts, pool_y_idx, pool_emb, idx_to_label,
              max_chars, self_positions=None):
    """Prefix every query text with per-class medoid exemplars from the pool.

    `self_positions` gives each query's position in the pool when the queries are
    pool rows, so a row is never its own exemplar.
    """
    pool_vals = [idx_to_label[int(i)] for i in pool_y_idx]
    global_rep = class_medoid_exemplar_indices(pool_emb, pool_y_idx)
    global_block = _exemplar_block(pool_texts, pool_vals, global_rep, max_chars)
    if self_positions is None:
        return [f"{global_block}\n\nProvider to rate:\n{t}" for t in query_texts]
    rep_set = set(global_rep)
    out = []
    for q, t in zip(self_positions, query_texts):
        if int(q) in rep_set:
            rep = class_medoid_exemplar_indices(pool_emb, pool_y_idx,
                                                exclude=[int(q)])
            block = _exemplar_block(pool_texts, pool_vals, rep, max_chars)
        else:
            block = global_block
        out.append(f"{block}\n\nProvider to rate:\n{t}")
    return out


# -----------------------------------------------------------------------------
# Model factory + one train phase
# -----------------------------------------------------------------------------
def _load_cls_model(model_name, n_classes, adapt, tokenizer,
                    lora_r=16, lora_alpha=32, lora_dropout=0.05):
    from transformers import AutoModelForSequenceClassification

    # fp32 weights; mixed precision comes from autocast in TrainingArguments.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes)
    # The head pools the last non-pad position, which needs pad_token_id set.
    model.config.pad_token_id = tokenizer.pad_token_id

    # Place the model now: with cached features the Trainer only sees the head,
    # and would otherwise move it away from the backbone.
    model = model.to(DEVICE)

    if adapt == "head":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.score.parameters():
            p.requires_grad = True
    elif adapt == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            bias="none", task_type="SEQ_CLS",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["score"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        if DEVICE == "cuda":
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
    else:
        raise ValueError(f"unknown --adapt {adapt!r}")
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [cls] adapt={adapt}  trainable params={n_train:,}")
    return model


def _phase(model, tokenizer, tr_texts, tr_labels, va_texts, va_labels,
           n_classes, idx_to_label, classes_sorted, lr, phase_dir):
    """Train on (tr), early-stop by QWK on (va), restore the best epoch."""
    import torch
    from transformers import DataCollatorWithPadding

    tr_ds = _TextDataset(tokenizer(list(tr_texts), truncation=True, padding=False,
                                   max_length=MAX_LEN), np.asarray(tr_labels))
    va_ds = _TextDataset(tokenizer(list(va_texts), truncation=True, padding=False,
                                   max_length=MAX_LEN), np.asarray(va_labels))
    cw = torch.as_tensor(balanced_class_weights(np.asarray(tr_labels), n_classes),
                         dtype=torch.float32)
    phase_dir.mkdir(parents=True, exist_ok=True)

    def _hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        proba = exp / exp.sum(axis=-1, keepdims=True)
        y_pred = np.array([idx_to_label[i] for i in np.argmax(logits, axis=-1)])
        y_true = np.array([idx_to_label[i] for i in labels])
        return compute_metrics(y_true, y_pred, y_proba=proba, labels=classes_sorted)

    args = make_training_arguments(
        output_dir=str(phase_dir), num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        group_by_length=GROUP_BY_LENGTH,
        learning_rate=lr, warmup_ratio=WARMUP_RATIO, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="no", logging_steps=50, seed=SEED,
        report_to=[], dataloader_num_workers=NUM_WORKERS,
        fp16=USE_FP16, bf16=USE_BF16,
    )
    WeightedTrainer = _make_weighted_trainer_class(cw)
    keeper = make_best_model_keeper(metric="eval_qwk", greater_is_better=True,
                                    patience=2)
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
        **trainer_tokenizer_kwarg(tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_hf_compute_metrics, callbacks=[keeper])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()
    if keeper.best_state is not None:
        model.load_state_dict(keeper.best_state)
    return trainer


def _predict(trainer, tokenizer, texts, labels, idx_to_label):
    """Score texts -> (y_true, y_pred, proba), with labels paired by the Trainer."""
    ds = _TextDataset(tokenizer(list(texts), truncation=True, padding=False,
                                max_length=MAX_LEN), np.asarray(labels))
    logits, out_labels = predict_with_labels(trainer, ds)
    if not np.array_equal(np.sort(out_labels),
                          np.sort(np.asarray(labels).astype(np.int64))):
        raise RuntimeError("predict label_ids are not a permutation of the "
                           "test labels — dataset/collator drift, refusing to score.")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    proba = exp / exp.sum(axis=-1, keepdims=True)
    y_pred = np.array([idx_to_label[i] for i in np.argmax(logits, axis=-1)])
    y_true = np.array([idx_to_label[i] for i in out_labels])
    return y_true, y_pred, proba


def _trainable_state(model):
    """CPU copy of the trainable parameters, restored with ``strict=False``.

    Only these can change during training, and a full state dict is ~16 GB.
    """
    return {k: v.detach().cpu().clone()
            for k, v in model.named_parameters() if v.requires_grad}


# -----------------------------------------------------------------------------
# Pretrain-state cache
# -----------------------------------------------------------------------------
def _pretrain_cache_path(args, cache_dir: Path, variant: str, tr_texts, tr_labels,
                         va_texts, va_labels, n_classes, lr) -> Path:
    """Cache path keyed on everything that can change the pretrained weights."""
    key = _digest_texts(
        list(tr_texts) + list(va_texts),
        args.model, args.adapt,
        (args.lora_r, args.lora_alpha, args.lora_dropout) if args.adapt == "lora" else None,
        np.asarray(tr_labels).astype(int).tolist(),
        np.asarray(va_labels).astype(int).tolist(),
        n_classes, lr, EPOCHS, BATCH_SIZE, GRAD_ACCUM, EVAL_BATCH_SIZE,
        MAX_LEN, WARMUP_RATIO, SEED, args.val_frac,
        USE_BF16, USE_FP16, GROUP_BY_LENGTH,
    )
    return cache_dir / f"pretrain_{variant}_{key}.pt"


def _load_pretrain_state(path: Path):
    """Cached state dict, or None on a miss."""
    if not (PRETRAIN_CACHE and path.exists()):
        return None
    import torch
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  [pretrain-cache] IGNORING unreadable entry {path.name}: {exc}")
        return None
    print(f"  [pretrain-cache] HIT {path.name} — skipping the pool pretrain phase")
    return state


def _save_pretrain_state(model, path: Path) -> None:
    if not PRETRAIN_CACHE:
        return
    import torch
    try:
        _atomic_save(_trainable_state(model), path, torch.save)
        print(f"  [pretrain-cache] saved -> {path}")
    except Exception as exc:                                    # noqa: BLE001
        print(f"  [pretrain-cache] WARNING: could not save {path.name}: {exc}")


def _predict_only_trainer(model, tokenizer, phase_dir: Path):
    """Predict-only Trainer for a cache-served pretrain, with _phase's arguments."""
    from transformers import DataCollatorWithPadding
    phase_dir.mkdir(parents=True, exist_ok=True)
    args_hf = make_training_arguments(
        output_dir=str(phase_dir), num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        group_by_length=GROUP_BY_LENGTH,
        learning_rate=HEAD_LR, warmup_ratio=WARMUP_RATIO, weight_decay=0.01,
        eval_strategy="no", save_strategy="no", logging_steps=50, seed=SEED,
        report_to=[], dataloader_num_workers=NUM_WORKERS,
        fp16=USE_FP16, bf16=USE_BF16,
    )
    from transformers import Trainer
    return Trainer(model=model, args=args_hf,
                   **trainer_tokenizer_kwarg(tokenizer),
                   data_collator=DataCollatorWithPadding(tokenizer=tokenizer))


# -----------------------------------------------------------------------------
# Frozen-backbone feature cache (--adapt head only)
# -----------------------------------------------------------------------------
def _backbone_is_deterministic(model) -> bool:
    """True iff the model has no active dropout, so cached features equal
    train-mode features."""
    import torch.nn as nn
    live = [f"{name or '<root>'}(p={m.p})"
            for name, m in model.named_modules()
            if isinstance(m, nn.Dropout) and float(m.p) > 0.0]
    if live:
        print(f"  [feature-cache] DISABLED: active dropout in the backbone "
              f"({', '.join(live[:4])}{' ...' if len(live) > 4 else ''}). "
              f"Cached features would not equal train-mode features.")
        return False
    return True


class _FeatureDataset:
    """(features, label) pairs for the head-only Trainer."""

    def __init__(self, feats, labels):
        import torch
        self.feats = torch.as_tensor(np.asarray(feats), dtype=torch.float32)
        self.labels = np.asarray(labels).astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        import torch
        return {"features": self.feats[i],
                "labels": torch.as_tensor(self.labels[i])}


def _make_head_module(score):
    """Wrap the model's own `score` Linear so it trains on precomputed features."""
    import torch.nn as nn
    from transformers.modeling_outputs import SequenceClassifierOutput

    class _HeadOnly(nn.Module):
        def __init__(self, score_module):
            super().__init__()
            self.score = score_module

        def forward(self, features=None, labels=None, **_):
            logits = self.score(features)
            # The plain predict-only Trainer requires a model-side loss.
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.float(), labels.view(-1).long())
            return SequenceClassifierOutput(loss=loss, logits=logits)

    return _HeadOnly(score)


def _extract_features(model, tokenizer, texts):
    """Pooled hidden states the head consumes, one row per text.

    Verifies that `score(pooled)` reproduces the model's own logits, and raises
    rather than caching the wrong vectors.
    """
    import torch
    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    enc = tokenizer(list(texts), truncation=True, padding=False, max_length=MAX_LEN)
    n = len(texts)
    order = np.argsort([len(x) for x in enc["input_ids"]])
    was_training = model.training
    model.eval()
    score = model.score if hasattr(model, "score") else model.base_model.model.score
    if score.weight.device != model.device:
        raise RuntimeError(
            f"backbone is on {model.device} but the classification head is on "
            f"{score.weight.device}. The head-only Trainer moved the shared "
            f"`score` Linear off the backbone's device (see _make_head_module). "
            f"Check that _load_cls_model still does model.to(DEVICE) and that "
            f"TrainingArguments resolves to the same device.")
    out = None
    t0 = time.time()
    amp_dtype = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else None)
    try:
        for start in range(0, n, EVAL_BATCH_SIZE):
            idx = order[start:start + EVAL_BATCH_SIZE]
            batch = collator([{k: enc[k][int(i)] for k in enc} for i in idx])
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                if amp_dtype is not None and model.device.type == "cuda":
                    with torch.autocast("cuda", dtype=amp_dtype):
                        res = model(**batch, output_hidden_states=True)
                else:
                    res = model(**batch, output_hidden_states=True)
            hs = res.hidden_states[-1]
            mask = batch["attention_mask"]
            # The pooled position depends on the padding side: try both.
            right = mask.sum(-1) - 1
            left = torch.full_like(right, hs.shape[1] - 1)
            b = torch.arange(hs.shape[0], device=hs.device)
            scale = res.logits.float().abs().max().item()
            tol = FEATURE_CHECK_ATOL + FEATURE_CHECK_RTOL * scale
            pooled, best_err = None, None
            for cand in (right, left):
                p = hs[b, cand]
                err = (score(p).float() - res.logits.float()).abs().max().item()
                if best_err is None or err < best_err:
                    best_err = err
                if err <= tol:
                    pooled = p
                    break
            if pooled is None:
                raise RuntimeError(
                    f"feature extraction could not reproduce the model's logits "
                    f"from any candidate pooled position (best max|Δ| = "
                    f"{best_err:.4g} > tol {tol:.4g} = atol {FEATURE_CHECK_ATOL:g} "
                    f"+ rtol {FEATURE_CHECK_RTOL:g} x |logits|max {scale:.4g}). "
                    f"That is far too large to be autocast noise, so the head's "
                    f"pooling rule is not what this code assumes. Refusing to "
                    f"cache features — rerun with CCQ_LLMCLS_FEATURE_CACHE=0 to "
                    f"use the (slower) text path, or raise "
                    f"CCQ_LLMCLS_FEATURE_RTOL if you are sure this is numerics.")
            if out is None:
                out = np.zeros((n, pooled.shape[-1]), dtype=np.float32)
            out[idx] = pooled.float().cpu().numpy()
    finally:
        if was_training:
            model.train()
    print(f"    [feature-cache] extracted {n} x {out.shape[1]}-d in "
          f"{time.time() - t0:.0f}s (logit check passed)")
    return out


_FEATURE_MEMO: dict[str, "np.ndarray"] = {}


def _features_for(model, tokenizer, texts, cache_dir: Path, model_name: str):
    """_extract_features with an in-process memo and an on-disk cache.

    DEVICE is in the key: autocast applies only on cuda, so CPU and GPU features differ.
    """
    key = _digest_texts(texts, model_name, MAX_LEN, USE_BF16, USE_FP16,
                        DEVICE, _FEATURE_CACHE_VERSION)
    if key in _FEATURE_MEMO:
        return _FEATURE_MEMO[key]
    path = cache_dir / f"feats_{key}.npy"
    if FEATURE_CACHE and path.exists():
        try:
            feats = np.load(path)
            if len(feats) == len(texts):
                print(f"    [feature-cache] HIT {path.name} {feats.shape}")
                return feats
            print(f"    [feature-cache] size mismatch in {path.name}; recomputing")
        except Exception as exc:                                # noqa: BLE001
            print(f"    [feature-cache] IGNORING unreadable {path.name}: {exc}")
    feats = _extract_features(model, tokenizer, texts)
    if FEATURE_CACHE:
        try:
            _atomic_save(feats, path, _np_save)
            print(f"    [feature-cache] saved -> {path}")
        except Exception as exc:                                # noqa: BLE001
            print(f"    [feature-cache] WARNING: could not save {path.name}: {exc}")
    return feats


def _phase_features(model, tr_feats, tr_labels, va_feats, va_labels,
                    n_classes, idx_to_label, classes_sorted, lr, phase_dir):
    """_phase over precomputed features, training only the head."""
    import torch
    from transformers import default_data_collator

    tr_ds = _FeatureDataset(tr_feats, tr_labels)
    va_ds = _FeatureDataset(va_feats, va_labels)
    cw = torch.as_tensor(balanced_class_weights(np.asarray(tr_labels), n_classes),
                         dtype=torch.float32)
    phase_dir.mkdir(parents=True, exist_ok=True)

    def _hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        proba = exp / exp.sum(axis=-1, keepdims=True)
        y_pred = np.array([idx_to_label[i] for i in np.argmax(logits, axis=-1)])
        y_true = np.array([idx_to_label[i] for i in labels])
        return compute_metrics(y_true, y_pred, y_proba=proba, labels=classes_sorted)

    args_hf = make_training_arguments(
        output_dir=str(phase_dir), num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=lr, warmup_ratio=WARMUP_RATIO, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="no", logging_steps=50, seed=SEED,
        report_to=[], dataloader_num_workers=0,
        fp16=USE_FP16, bf16=USE_BF16,
    )
    head = _make_head_module(model.score if hasattr(model, "score")
                             else model.base_model.model.score)
    WeightedTrainer = _make_weighted_trainer_class(cw)
    keeper = make_best_model_keeper(metric="eval_qwk", greater_is_better=True,
                                    patience=2)
    trainer = WeightedTrainer(
        model=head, args=args_hf, train_dataset=tr_ds, eval_dataset=va_ds,
        data_collator=default_data_collator,
        compute_metrics=_hf_compute_metrics, callbacks=[keeper])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()
    if keeper.best_state is not None:
        head.load_state_dict(keeper.best_state)
    return trainer


def _phase_features_predictor(model, phase_dir: Path):
    """Feature-path analogue of _predict_only_trainer."""
    from transformers import Trainer, default_data_collator
    phase_dir.mkdir(parents=True, exist_ok=True)
    args_hf = make_training_arguments(
        output_dir=str(phase_dir), num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=HEAD_LR, warmup_ratio=WARMUP_RATIO, weight_decay=0.01,
        eval_strategy="no", save_strategy="no", logging_steps=50, seed=SEED,
        report_to=[], dataloader_num_workers=0,
        fp16=USE_FP16, bf16=USE_BF16,
    )
    head = _make_head_module(model.score if hasattr(model, "score")
                             else model.base_model.model.score)
    return Trainer(model=head, args=args_hf, data_collator=default_data_collator)


def _predict_features(trainer, feats, labels, idx_to_label):
    """_predict over cached features."""
    ds = _FeatureDataset(feats, labels)
    logits, out_labels = predict_with_labels(trainer, ds)
    if not np.array_equal(np.sort(out_labels),
                          np.sort(np.asarray(labels).astype(np.int64))):
        raise RuntimeError("predict label_ids are not a permutation of the "
                           "test labels — dataset/collator drift, refusing to score.")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    proba = exp / exp.sum(axis=-1, keepdims=True)
    y_pred = np.array([idx_to_label[i] for i in np.argmax(logits, axis=-1)])
    y_true = np.array([idx_to_label[i] for i in out_labels])
    return y_true, y_pred, proba


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_transfer_cls(data, args, models_dir):
    from transformers import AutoTokenizer

    PCT = int(round(args.target_frac * 100))
    variant = "cls_rag" if args.rag else ("cls_lora" if args.adapt == "lora" else "cls")
    method = (f"xfer_qwen_{variant}{scale_infix(data.scale)}"
              f"_tgt{PCT}_{data.src_name}2{data.tgt_name}")
    lr = LORA_LR if args.adapt == "lora" else HEAD_LR
    seed = args.seed

    print(f"\n{'=' * 70}")
    print(f"LLM-CLS ({variant}, adapt={args.adapt}): {data.src_name} -> "
          f"{data.tgt_name}  tgt{PCT}")
    print(f"  model={args.model}  classes={data.labels}  device={DEVICE}")
    print(f"  batch={BATCH_SIZE}x{GRAD_ACCUM} eval={EVAL_BATCH_SIZE} "
          f"max_len={MAX_LEN} epochs={EPOCHS} lr={lr}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    src_texts = data.src_texts.tolist()
    tgt_texts = data.tgt_texts.tolist()

    src_emb = tgt_emb = None
    if args.rag:
        dev = "cuda" if (DEVICE == "cuda") else "cpu"
        print("  [rag] embedding source pool for medoid exemplars...")
        src_emb = _embed_texts(src_texts, args.embed_model, dev)
        if PCT > 0:
            print("  [rag] embedding target for medoid exemplars...")
            tgt_emb = _embed_texts(tgt_texts, args.embed_model, dev)

    def rag_src(qtexts, qpos=None):
        return _rag_wrap(qtexts, src_texts, data.y_src, src_emb,
                         data.idx_to_label, args.max_exemplar_chars, qpos)

    def rag_tgt_pool(qtexts, pool_pos, qpos_in_pool=None):
        """Exemplars drawn from the target rows at `pool_pos`."""
        pool_t = [tgt_texts[int(i)] for i in pool_pos]
        pool_y = data.y_tgt[pool_pos]
        pool_e = tgt_emb[pool_pos]
        return _rag_wrap(qtexts, pool_t, pool_y, pool_e,
                         data.idx_to_label, args.max_exemplar_chars, qpos_in_pool)

    t0 = time.time()
    model = _load_cls_model(args.model, data.n_classes, args.adapt, tokenizer,
                            args.lora_r, args.lora_alpha, args.lora_dropout)

    cache_dir = run_artifact_dir(args.output) / "llm_cls_cache"
    use_feats = (FEATURE_CACHE and args.adapt == "head"
                 and _backbone_is_deterministic(model))
    if use_feats:
        print("  [feature-cache] ON — backbone frozen and deterministic; "
              "training the K-way head on cached pooled states")

    def feats(texts):
        return _features_for(model, tokenizer, texts, cache_dir, args.model)

    def phase(tr_texts, tr_y, va_texts, va_y, tag):
        print(f"  [phase:{tag}] train={len(tr_texts)} val={len(va_texts)}")
        if use_feats:
            return _phase_features(
                model, feats(tr_texts), tr_y.astype(int), feats(va_texts), va_y,
                data.n_classes, data.idx_to_label, data.labels, lr,
                models_dir / method / tag)
        return _phase(model, tokenizer, tr_texts, tr_y.astype(int), va_texts, va_y,
                      data.n_classes, data.idx_to_label, data.labels, lr,
                      models_dir / method / tag)

    def predict(trainer, texts, labels):
        if use_feats:
            return _predict_features(trainer, feats(texts), labels,
                                     data.idx_to_label)
        return _predict(trainer, tokenizer, texts, labels, data.idx_to_label)

    def pretrain_phase(tr_texts, tr_y, va_texts, va_y):
        """Pool pretrain, served from cache when available; returns None on a hit."""
        path = _pretrain_cache_path(args, cache_dir, variant, tr_texts,
                                    tr_y, va_texts, va_y, data.n_classes, lr)
        state = _load_pretrain_state(path)
        if state is not None:
            model.load_state_dict(state, strict=False)
            return None
        tr = phase(tr_texts, tr_y, va_texts, va_y, "pretrain")
        _save_pretrain_state(model, path)
        return tr

    adapt_idx, test_idx = split_target_fewshot(
        data.y_tgt, args.target_frac, args.target_test_frac, seed)
    print(f"  target split: adapt={len(adapt_idx)}  test={len(test_idx)}")
    if PCT == 0:
        # Zero-shot: pretrain on the pool only; no fine-tune phase.
        p_tr, p_va = stratified_source_split(data.y_src, args.val_frac, seed)
        if args.rag:
            ptr = rag_src([src_texts[i] for i in p_tr], p_tr)
            pva = rag_src([src_texts[i] for i in p_va], p_va)
            te_texts = rag_src([tgt_texts[i] for i in test_idx])
        else:
            ptr = [src_texts[i] for i in p_tr]; pva = [src_texts[i] for i in p_va]
            te_texts = [tgt_texts[i] for i in test_idx]
        trainer = pretrain_phase(ptr, data.y_src[p_tr], pva, data.y_src[p_va])
        if trainer is None:
            trainer = (_phase_features_predictor(model, models_dir / method / "pretrain")
                       if use_feats
                       else _predict_only_trainer(model, tokenizer,
                                                  models_dir / method / "pretrain"))
    else:
        p_tr, p_va = stratified_source_split(data.y_src, args.val_frac, seed)
        s_tr, s_va = stratified_source_split(data.y_tgt[adapt_idx],
                                             args.val_frac, seed)
        a_tr, a_va = adapt_idx[s_tr], adapt_idx[s_va]
        if args.rag:
            ptr = rag_src([src_texts[i] for i in p_tr], p_tr)
            pva = rag_src([src_texts[i] for i in p_va], p_va)
            pos_in_adapt = {int(g): k for k, g in enumerate(adapt_idx)}
            ftr = rag_tgt_pool([tgt_texts[i] for i in a_tr], adapt_idx,
                               [pos_in_adapt[int(i)] for i in a_tr])
            fva = rag_tgt_pool([tgt_texts[i] for i in a_va], adapt_idx,
                               [pos_in_adapt[int(i)] for i in a_va])
            te_texts = rag_tgt_pool([tgt_texts[i] for i in test_idx], adapt_idx)
        else:
            ptr = [src_texts[i] for i in p_tr]; pva = [src_texts[i] for i in p_va]
            ftr = [tgt_texts[i] for i in a_tr]; fva = [tgt_texts[i] for i in a_va]
            te_texts = [tgt_texts[i] for i in test_idx]
        pretrain_phase(ptr, data.y_src[p_tr], pva, data.y_src[p_va])
        trainer = phase(ftr, data.y_tgt[a_tr], fva, data.y_tgt[a_va],
                        "finetune")
    y_true, y_pred, y_proba = predict(trainer, te_texts, data.y_tgt[test_idx])

    point = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=data.labels)
    std = bootstrap_target_std(y_true, y_pred, y_proba, labels=data.labels,
                               n_boot=args.n_bootstrap, seed=seed)
    elapsed = time.time() - t0
    print(f"\n  TARGET ({data.tgt_name}, n={len(y_true)}): "
          f"acc={point['accuracy']:.3f}  bal_acc={point['balanced_acc']:.3f}  "
          f"qwk={point['qwk']:.3f}  mae={point['mae']:.3f}")
    print(f"  elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    model = None  # closures still reference it, so no `del`
    if DEVICE == "cuda":
        import torch as _t
        _t.cuda.empty_cache()
    shutil.rmtree(models_dir / method, ignore_errors=True)
    return method, point, std


def run_within_cls(data, args, models_dir):
    """Within-state K-fold CV. Every fold starts from the same initial head/LoRA
    weights; RAG exemplars come from the fold's training rows only."""
    from transformers import AutoTokenizer

    state = data.tgt_name
    variant = "cls_rag" if args.rag else ("cls_lora" if args.adapt == "lora" else "cls")
    method = f"qwen_{variant}_within{scale_infix(data.scale)}_tgt100_{state}"
    lr = LORA_LR if args.adapt == "lora" else HEAD_LR
    seed = args.seed

    print(f"\n{'=' * 70}")
    print(f"WITHIN-STATE LLM-CLS ({variant}, adapt={args.adapt}): {state}  "
          f"({args.within_cv}-fold CV)")
    print(f"  model={args.model}  classes={data.labels}  device={DEVICE}")
    print(f"  batch={BATCH_SIZE}x{GRAD_ACCUM} eval={EVAL_BATCH_SIZE} "
          f"max_len={MAX_LEN} epochs={EPOCHS} lr={lr}")
    print(f"{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = data.tgt_texts.tolist()
    y = data.y_tgt
    emb = None
    if args.rag:
        dev = "cuda" if (DEVICE == "cuda") else "cpu"
        print("  [rag] embedding state rows for medoid exemplars...")
        emb = _embed_texts(texts, args.embed_model, dev)

    t0 = time.time()
    model = _load_cls_model(args.model, data.n_classes, args.adapt, tokenizer,
                            args.lora_r, args.lora_alpha, args.lora_dropout)
    init_state = _trainable_state(model)

    cache_dir = run_artifact_dir(args.output) / "llm_cls_cache"
    use_feats = (FEATURE_CACHE and args.adapt == "head"
                 and _backbone_is_deterministic(model))
    if use_feats:
        print("  [feature-cache] ON — backbone frozen and deterministic; "
              "training the K-way head on cached pooled states")

    def feats(t):
        return _features_for(model, tokenizer, t, cache_dir, args.model)

    folds = target_cv_folds(y, args.within_cv, seed)
    yt, yp, pr = [], [], []
    for fi, (tr, te) in enumerate(folds, 1):
        print(f"\n  -- fold {fi}/{len(folds)}: train={len(tr)} test={len(te)} --")
        model.load_state_dict(init_state, strict=False)
        s_tr, s_va = stratified_source_split(y[tr], args.val_frac, seed)
        f_tr, f_va = tr[s_tr], tr[s_va]
        if args.rag:
            pool_t = [texts[int(i)] for i in tr]
            pool_y = y[tr]
            pool_e = emb[tr]
            pos_in_tr = {int(g): k for k, g in enumerate(tr)}
            ftr = _rag_wrap([texts[i] for i in f_tr], pool_t, pool_y, pool_e,
                            data.idx_to_label, args.max_exemplar_chars,
                            [pos_in_tr[int(i)] for i in f_tr])
            fva = _rag_wrap([texts[i] for i in f_va], pool_t, pool_y, pool_e,
                            data.idx_to_label, args.max_exemplar_chars,
                            [pos_in_tr[int(i)] for i in f_va])
            fte = _rag_wrap([texts[i] for i in te], pool_t, pool_y, pool_e,
                            data.idx_to_label, args.max_exemplar_chars, None)
        else:
            ftr = [texts[i] for i in f_tr]
            fva = [texts[i] for i in f_va]
            fte = [texts[i] for i in te]
        if use_feats:
            trainer = _phase_features(
                model, feats(ftr), y[f_tr].astype(int), feats(fva), y[f_va],
                data.n_classes, data.idx_to_label, data.labels, lr,
                models_dir / method / f"fold{fi}")
            a, b, c = _predict_features(trainer, feats(fte), y[te],
                                        data.idx_to_label)
        else:
            trainer = _phase(model, tokenizer, ftr, y[f_tr].astype(int),
                             fva, y[f_va], data.n_classes, data.idx_to_label,
                             data.labels, lr, models_dir / method / f"fold{fi}")
            a, b, c = _predict(trainer, tokenizer, fte, y[te], data.idx_to_label)
        yt.append(a); yp.append(b); pr.append(c)
    y_true = np.concatenate(yt)
    y_pred = np.concatenate(yp)
    y_proba = np.concatenate(pr, axis=0)

    point = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=data.labels)
    std = bootstrap_target_std(y_true, y_pred, y_proba, labels=data.labels,
                               n_boot=args.n_bootstrap, seed=seed)
    elapsed = time.time() - t0
    print(f"\n  OOF ({state}, n={len(y_true)}): "
          f"acc={point['accuracy']:.3f}  bal_acc={point['balanced_acc']:.3f}  "
          f"qwk={point['qwk']:.3f}  mae={point['mae']:.3f}")
    print(f"  elapsed: {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    model = None
    if DEVICE == "cuda":
        import torch as _t
        _t.cuda.empty_cache()
    shutil.rmtree(models_dir / method, ignore_errors=True)
    return method, point, std


def _method_tag(args) -> str:
    variant = "cls_rag" if args.rag else ("cls_lora" if args.adapt == "lora" else "cls")
    infix = scale_infix(args.rating_scale)
    if args.within_cv > 0:
        return f"qwen_{variant}_within{infix}_tgt100_{args.tgt_name}"
    src = args.pool_name if args.pool_sources else args.src_name
    pct = int(round(args.target_frac * 100))
    return f"xfer_qwen_{variant}{infix}_tgt{pct}_{src}2{args.tgt_name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3 classification-head transfer (head / +rag / +lora).")
    add_transfer_args(parser)
    parser.add_argument("--adapt", choices=("head", "lora"), default="head",
                        help="'head': frozen backbone, train the K-way head only. "
                             "'lora': LoRA adapters + head.")
    parser.add_argument("--rag", action="store_true",
                        help="Prefix every input with per-class medoid exemplars.")
    parser.add_argument("--model", default=DEFAULT_CLS_LLM_PATH,
                        help=f"Local Qwen checkpoint (default {DEFAULT_CLS_LLM_PATH}).")
    parser.add_argument("--embed-model",
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedder for --rag medoid selection.")
    parser.add_argument("--max-exemplar-chars", type=int, default=1200)
    parser.add_argument("--within-cv", type=int, default=0,
                        help="K-fold CV within one state (pass its CSV as both --source "
                             "and --target). 0 = cross-state transfer.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    if args.within_cv > 0 and args.pool_sources:
        parser.error("--within-cv is single-state; it cannot take --pool-sources.")
    configure_verbosity(args.verbose)
    _resolve_transfer_output(args, "transfer_llm_cls")
    models_dir = run_artifact_dir(args.output) / "llm_cls_models"

    try:
        if args.within_cv > 0:
            data = load_and_prepare(
                args.source, args.target, args.src_name, args.tgt_name,
                compliance_mode=args.compliance, scale=args.rating_scale)
            method, point, std = run_within_cls(data, args, models_dir)
            source = data.tgt_name
        else:
            if args.pool_sources:
                data = load_and_prepare_pooled(
                    parse_pool_sources(args.pool_sources), args.tgt_name, args.target,
                    compliance_mode=args.compliance, scale=args.rating_scale,
                    pool_name=args.pool_name)
            else:
                data = load_and_prepare(
                    args.source, args.target, args.src_name, args.tgt_name,
                    compliance_mode=args.compliance, scale=args.rating_scale)
            method, point, std = run_transfer_cls(data, args, models_dir)
            source = data.src_name
        log_transfer_result(method, point, std, output_path=args.output, notes=method,
                            source=source, target=data.tgt_name, classes=data.labels)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        src = (args.tgt_name if args.within_cv > 0
               else (args.pool_name if args.pool_sources else args.src_name))
        log_failed_transfer(_method_tag(args), output_path=args.output,
                            source=src, target=args.tgt_name,
                            error=f"{type(exc).__name__}: {exc}")
        raise
    print(f"\nDone. See {args.output} for results.")


if __name__ == "__main__":
    main()
