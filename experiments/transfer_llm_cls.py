"""Qwen3-4B with a trained classification head, for cross-state transfer.

Three variants sharing one protocol -- pretrain on the pool, then finetune --
differing only in adaptation capacity and input construction:

  --adapt head            xfer_qwen_cls_*        frozen backbone, K-way head only
  --adapt head --rag      xfer_qwen_cls_rag_*    same, inputs prefixed with
                                                 per-class medoid exemplars
  --adapt lora            xfer_qwen_cls_lora_*   LoRA adapters plus head

All three are discriminatively trained classifiers: the input is the serialized
row, the output is the head's K-way softmax. There is no chat template, rubric,
generation step or verbalizer.

A stratified 20% slice of the target is carved once and scored at every curve
point. At p=0 there is no finetune phase. Each phase early-stops on a stratified
15% slice of its own training rows and uses balanced class weights.

Backbone features and the pool pretrain state are cached under the run's artifact
directory; see the cache notes below.

    python transfer_llm_cls.py --pool-sources WI=data/wi_records_cleaned_raw.csv ... \\
        --target data/nc_records_cleaned_raw.csv --tgt-name NC --rating-scale 5star \\
        --adapt head --target-frac 0.2
    python transfer_llm_cls.py --within-cv 5 --adapt lora \\
        --source data/nc_records_cleaned_raw.csv --target data/nc_records_cleaned_raw.csv \\
        --src-name NC --tgt-name NC
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

import hashlib  # noqa: E402
import itertools  # noqa: E402
import os  # noqa: E402

# Checkpoint for the trained-head methods: dense Qwen3-4B. This is a LOCAL
# directory, not a HuggingFace repo id, because compute nodes generally have no
# outbound network access during a job -- fetch it first with `download_qwen.py`.
#
# Resolution order: --model, then $LLM_CLS_MODEL, then models/qwen3_4b inside the
# repository, so a fresh checkout needs no configuration beyond the download.
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
# Caches
# -----------------------------------------------------------------------------
# Both live under run_artifact_dir(--output), so concurrent jobs writing
# different outputs do not share them.
#
#   PRETRAIN cache  the pooled-source pretrain phase does not depend on p, so it
#     is computed once and reloaded at every curve point.
#   FEATURE cache   with --adapt head the backbone is frozen, so pooled hidden
#     states are extracted once and the head trains on the cached vectors. Exact
#     only when no backbone dropout is active, checked by
#     _backbone_is_deterministic.
PRETRAIN_CACHE = os.environ.get("CCQ_LLMCLS_PRETRAIN_CACHE", "1") == "1"
FEATURE_CACHE = os.environ.get("CCQ_LLMCLS_FEATURE_CACHE", "1") == "1"
# Tolerance for the guard that cached features reproduce the model's own logits.
FEATURE_CHECK_ATOL = float(os.environ.get("CCQ_LLMCLS_FEATURE_ATOL", "2e-2"))
# Relative term, added to the absolute one. bf16 has an 8-bit
# mantissa, so a 2560-wide matmul accumulates an error PROPORTIONAL to the
# logit magnitude — a fixed atol cannot separate that from a genuinely wrong
# pooled position. 5% of |logits|max sits ~2x above the noise actually observed
# (0.020-0.041 on logits of order 1) and ~20x below a mispooling error.
FEATURE_CHECK_RTOL = float(os.environ.get("CCQ_LLMCLS_FEATURE_RTOL", "5e-2"))
# Bumped whenever the extraction MATH changes, so stale entries miss by
# construction. The device and precision are part of the digest as well: fp32
# vectors computed on CPU and bf16 vectors computed under autocast are not
# interchangeable, and nothing else in the key distinguishes them.
_FEATURE_CACHE_VERSION = 2


def _digest_texts(texts, *scalars) -> str:
    """Stable md5 over the exact strings a phase will consume plus the scalar
    knobs that change the resulting weights. Streamed so a 25k-row pool does not
    materialise a giant repr. Same stdlib-md5 convention as the embedding cache: any serialization change is a cache MISS by
    construction rather than a silent stale hit."""
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
    """Write via a temp file + os.replace so a killed job never leaves a
    half-written cache entry that a later job would happily load.

    ``saver(obj, tmp)`` must write to exactly ``tmp``. ``np.save`` appends
    ``.npy`` when the path does not already end in it, so the existence check
    below reports that as a clear error rather than a FileNotFoundError from
    ``os.replace``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per CALL, not merely per process. A failed write leaves its temp
    # behind, and a later write in the same process would otherwise find that
    # stale file at the pid-derived name and rename IT over the good entry.
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
        # Reached with tmp present only when the write failed; a successful
        # os.replace has already consumed it.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _np_save(arr, path) -> None:
    """np.save to an EXACT path (a file handle stops it appending '.npy')."""
    with open(path, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)


# -----------------------------------------------------------------------------
# RAG input construction (plain text, no chat template)
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

    self_positions: optional array mapping each query row to its position in the
    pool (when the queries ARE pool rows, e.g. training on the exemplar pool) so
    a row is never its own exemplar — rows that are their class medoid get that
    class's runner-up instead. None = queries disjoint from the pool (eval).
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

    # fp32 weights with bf16/fp16 AUTOCAST via TrainingArguments -- the same
    # mixed-precision pattern MBERT uses here. Loading the backbone in bf16 and
    # casting the head to fp32 instead would crash the first forward on a Linear
    # dtype mismatch.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes)
    # Qwen has no pad token; seq-cls pools the LAST NON-PAD position, which
    # requires config.pad_token_id to be set or every padded batch mispools.
    model.config.pad_token_id = tokenizer.pad_token_id

    # Place the model on the accelerator at construction. With --adapt head the
    # Trainer is handed only _HeadOnly(model.score), so leaving placement to it
    # would move the head and strand the backbone on CPU.
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
    """One fine-tuning phase: train on (tr), best-by-QWK on (va), restore best.
    Mirrors transfer_plm's sequential _phase; scoring uses the sampler-proof
    predict pairing everywhere else in this file."""
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
        fp16=USE_FP16, bf16=USE_BF16,  # autocast over fp32 weights (MBERT pattern)
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
    """Score texts -> (y_true, y_pred, proba), paired via label_ids (sampler-proof)."""
    ds = _TextDataset(tokenizer(list(texts), truncation=True, padding=False,
                                max_length=MAX_LEN), np.asarray(labels))
    logits, out_labels = predict_with_labels(trainer, ds)
    # Same drift guard transfer_plm._predict_trainer carries: label_ids must be a
    # permutation of the dataset's labels.
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
    """CPU copy of the TRAINABLE parameters only — the restore payload.

    Copying the full ``state_dict()`` would move roughly 16 GB of frozen fp32
    Qwen3-4B weights, for every in-RAM snapshot and every cached entry. Nothing
    outside the trainable set can move: ``--adapt head`` freezes all but
    ``score``, LoRA freezes the base weights and trains adapters +
    ``modules_to_save``, and Qwen has no training-mutated buffers (no
    BatchNorm). So the trainable parameters ARE the delta, and every restore
    site pairs this with ``strict=False``.
    """
    return {k: v.detach().cpu().clone()
            for k, v in model.named_parameters() if v.requires_grad}


# -----------------------------------------------------------------------------
# Pretrain-state cache (speedup #1)
# -----------------------------------------------------------------------------
def _pretrain_cache_path(args, cache_dir: Path, variant: str, tr_texts, tr_labels,
                         va_texts, va_labels, n_classes, lr) -> Path:
    """Path for the post-pretrain state of THIS exact computation.

    The digest covers everything that can change the resulting weights: the
    checkpoint, the adaptation mode (and LoRA geometry), the exact train/val
    texts and labels of the pretrain phase, the class count, and every
    optimisation knob including the autocast precision. Anything not in the key
    is something that provably cannot move the weights.
    """
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
    """Return the cached state dict, or None on a miss / unreadable entry."""
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
        # A cache write failure must never take down a run that already did the
        # expensive part; the next job just re-pretrains.
        print(f"  [pretrain-cache] WARNING: could not save {path.name}: {exc}")


def _predict_only_trainer(model, tokenizer, phase_dir: Path):
    """Trainer used purely for .predict() when the pretrain phase was served
    from cache and no training Trainer was constructed. Same TrainingArguments
    as _phase so batching/precision/pairing behave identically."""
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
# Frozen-backbone feature cache (speedup #2) — `--adapt head` only
# -----------------------------------------------------------------------------
def _backbone_is_deterministic(model) -> bool:
    """True iff no active dropout remains anywhere in the model.

    Caching features is EXACT only if the backbone returns the same vector for a
    row on every epoch. HF Trainer puts the whole model in train() mode, so any
    nn.Dropout with p>0 would resample per epoch and the cache would silently
    change the method. Qwen3 ships attention/hidden dropout at 0.0, but the
    checkpoint is a runtime input — so this is checked, not assumed.
    """
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
    """Wrap the trained K-way head so it consumes precomputed features.

    Shares the SAME nn.Linear object as the full model's `score`, so training
    this module trains the real head in place and the full model stays usable
    for anything downstream.
    """
    import torch.nn as nn
    from transformers.modeling_outputs import SequenceClassifierOutput

    class _HeadOnly(nn.Module):
        def __init__(self, score_module):
            super().__init__()
            self.score = score_module

        def forward(self, features=None, labels=None, **_):
            logits = self.score(features)
            # Return a loss whenever labels are passed. Training does not reach
            # this branch (the weighted trainer pops "labels" first); it exists
            # for the plain Trainer in _phase_features_predictor, whose
            # compute_loss requires a model-side loss. The value is discarded by
            # predict_with_labels.
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.float(), labels.view(-1).long())
            return SequenceClassifierOutput(loss=loss, logits=logits)

    return _HeadOnly(score)


def _extract_features(model, tokenizer, texts):
    """Pooled hidden states the seq-cls head consumes, one row per text.

    Rather than reimplementing (and guessing at) the head's position-pooling
    rule and the tokenizer's padding side, this runs the real model, asks for
    hidden states, derives the pooled vector, and then VERIFIES that
    `score(pooled)` reproduces the model's own logits. A mismatch raises instead
    of silently caching the wrong vectors.
    """
    import torch
    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    enc = tokenizer(list(texts), truncation=True, padding=False, max_length=MAX_LEN)
    n = len(texts)
    order = np.argsort([len(x) for x in enc["input_ids"]])   # length-sorted: less padding
    was_training = model.training
    model.eval()
    score = model.score if hasattr(model, "score") else model.base_model.model.score
    # Tripwire: _make_head_module hands the head-only Trainer the
    # SAME nn.Linear the full model uses, and HF Trainer calls .to(args.device)
    # on whatever model it is given. If that device ever diverges from the
    # backbone's, the hidden states and the head land on different devices and
    # the forward fails deep inside transformers. Fail here instead, naming the
    # cause. _load_cls_model places the model on DEVICE, so this is unreachable
    # unless that changes.
    if score.weight.device != model.device:
        raise RuntimeError(
            f"backbone is on {model.device} but the classification head is on "
            f"{score.weight.device}. The head-only Trainer moved the shared "
            f"`score` Linear off the backbone's device (see _make_head_module). "
            f"Check that _load_cls_model still does model.to(DEVICE) and that "
            f"TrainingArguments resolves to the same device.")
    out = None
    t0 = time.time()
    # Match the training-time forward: bf16/fp16 autocast over fp32 weights.
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
            hs = res.hidden_states[-1]                        # (B, T, H), post-norm
            mask = batch["attention_mask"]
            # The head pools ONE position per row; which one depends on the
            # tokenizer's padding side. Try both, keep whichever actually
            # reproduces the model's logits (checked, not assumed).
            right = mask.sum(-1) - 1
            left = torch.full_like(right, hs.shape[1] - 1)
            b = torch.arange(hs.shape[0], device=hs.device)
            # Tolerance is SCALE-RELATIVE. The check exists to catch a WRONG
            # POOLED POSITION, which produces an error the size of the logits
            # themselves, and it must not fire on bf16 accumulation noise, which
            # scales with |logits|. A flat absolute tolerance cannot separate the
            # two: under autocast the noise reaches a few percent of |logits|,
            # while a mispooled position is off by O(1).
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
    """Cached wrapper around _extract_features, keyed on the exact strings.

    Two layers: an in-process memo (the same text block is asked for more than
    once per run — e.g. the eval block is fetched again at predict time, and a
    ~25k x 2560 fp32 array is ~250 MB to re-read), and the on-disk cache that
    survives across the curve points and across jobs.
    """
    # DEVICE and the cache version are part of the key. USE_BF16/USE_FP16 record
    # the INTENT to autocast, but autocast only applies when the model sits on
    # cuda, so a CPU run produces fp32 vectors that must not share a key with
    # autocast-computed ones -- mixing them would shift the head's inputs between
    # train and test.
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
    """_phase, but over precomputed features and training ONLY the K-way head.

    Deliberately reuses the same TrainingArguments, the same class-weighted
    Trainer subclass, and the same best-by-QWK RAM keeper as the text path, so
    the optimisation protocol is unchanged — only the (frozen, hence constant)
    backbone forward pass is skipped.
    """
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
    """Feature-path analogue of _predict_only_trainer: wraps the (already
    restored) head so a cache-served pretrain can still be scored."""
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
    """_predict over cached features; same sampler-proof label pairing."""
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
    import torch
    from transformers import AutoTokenizer

    cv_folds = args.target_cv_folds
    PCT = 100 if cv_folds > 0 else int(round(args.target_frac * 100))
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

    # --- RAG embeddings (once; MiniLM, the shared featurizer) ---------------
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
        """Exemplars from the TARGET rows at positions pool_pos."""
        pool_t = [tgt_texts[int(i)] for i in pool_pos]
        pool_y = data.y_tgt[pool_pos]
        pool_e = tgt_emb[pool_pos]
        return _rag_wrap(qtexts, pool_t, pool_y, pool_e,
                         data.idx_to_label, args.max_exemplar_chars, qpos_in_pool)

    t0 = time.time()
    model = _load_cls_model(args.model, data.n_classes, args.adapt, tokenizer,
                            args.lora_r, args.lora_alpha, args.lora_dropout)

    # --- cache setup -------------------------------------------
    # Both caches live under the per-target artifact subtree, so concurrent
    # per-target jobs never share (or clobber) an entry.
    cache_dir = run_artifact_dir(args.output) / "llm_cls_cache"
    # Feature caching applies ONLY when the backbone is frozen (--adapt head,
    # with or without --rag) AND nothing in it is stochastic.
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
        """The pooled-source pretrain, served from cache when it has already
        been computed for this exact (pool, variant, hyperparameter) tuple.
        Identical at every curve point, so this is a straight 1-of-N saving."""
        path = _pretrain_cache_path(args, cache_dir, variant, tr_texts,
                                    tr_y, va_texts, va_y, data.n_classes, lr)
        state = _load_pretrain_state(path)
        if state is not None:
            model.load_state_dict(state, strict=False)   # delta-only payload
            return None                     # caller builds a trainer if it needs one
        tr = phase(tr_texts, tr_y, va_texts, va_y, "pretrain")
        _save_pretrain_state(model, path)
        return tr

    if cv_folds > 0:
        # p=100: pretrain ONCE on the full pool, then per-fold finetune+predict.
        p_tr, p_va = stratified_source_split(data.y_src, args.val_frac, seed)
        tr_texts = rag_src([src_texts[i] for i in p_tr], p_tr) if args.rag \
            else [src_texts[i] for i in p_tr]
        va_texts = rag_src([src_texts[i] for i in p_va], p_va) if args.rag \
            else [src_texts[i] for i in p_va]
        pretrain_phase(tr_texts, data.y_src[p_tr], va_texts, data.y_src[p_va])
        pre_state = _trainable_state(model)

        folds = target_cv_folds(data.y_tgt, cv_folds, seed)
        yt, yp, pr = [], [], []
        for fi, (tr, te) in enumerate(folds, 1):
            print(f"\n  -- fold {fi}/{len(folds)}: target train={len(tr)} "
                  f"test={len(te)} --")
            model.load_state_dict(pre_state, strict=False)  # delta-only, see _trainable_state
            s_tr, s_va = stratified_source_split(data.y_tgt[tr], args.val_frac, seed)
            f_tr, f_va = tr[s_tr], tr[s_va]
            if args.rag:
                ftr = rag_tgt_pool([tgt_texts[i] for i in f_tr], tr,
                                   [int(np.where(tr == i)[0][0]) for i in f_tr])
                fva = rag_tgt_pool([tgt_texts[i] for i in f_va], tr,
                                   [int(np.where(tr == i)[0][0]) for i in f_va])
                fte = rag_tgt_pool([tgt_texts[i] for i in te], tr)
            else:
                ftr = [tgt_texts[i] for i in f_tr]
                fva = [tgt_texts[i] for i in f_va]
                fte = [tgt_texts[i] for i in te]
            trainer = phase(ftr, data.y_tgt[f_tr], fva, data.y_tgt[f_va],
                            f"fold{fi}")
            a, b, c = predict(trainer, fte, data.y_tgt[te])
            yt.append(a); yp.append(b); pr.append(c)
        y_true = np.concatenate(yt); y_pred = np.concatenate(yp)
        y_proba = np.concatenate(pr, axis=0)
    else:
        adapt_idx, test_idx = split_target_fewshot(
            data.y_tgt, args.target_frac, args.target_test_frac, seed)
        print(f"  target split: adapt={len(adapt_idx)}  test={len(test_idx)}")
        if PCT == 0:
            # Zero-shot: pretrain on the FULL pool and stop. No target labels
            # exist at p=0, so there is no finetune phase -- which is precisely
            # the source-only pretrain point that MBERT, MBERT-CORN and TabNN
            # report, and is what makes the p=0 row comparable ACROSS method
            # families rather than only down this method's own curve.
            p_tr, p_va = stratified_source_split(data.y_src, args.val_frac, seed)
            if args.rag:
                ptr = rag_src([src_texts[i] for i in p_tr], p_tr)
                pva = rag_src([src_texts[i] for i in p_va], p_va)
                # Eval exemplars at p=0: the SOURCE pool (no target labels).
                te_texts = rag_src([tgt_texts[i] for i in test_idx])
            else:
                ptr = [src_texts[i] for i in p_tr]; pva = [src_texts[i] for i in p_va]
                te_texts = [tgt_texts[i] for i in test_idx]
            trainer = pretrain_phase(ptr, data.y_src[p_tr], pva, data.y_src[p_va])
            if trainer is None:
                # Served from cache: no training Trainer was built, so make a
                # predict-only one over the restored weights.
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

    model = None   # drop the reference (not `del`: the phase/predict closures capture it)
    if DEVICE == "cuda":
        import torch as _t
        _t.cuda.empty_cache()
    shutil.rmtree(models_dir / method, ignore_errors=True)
    return method, point, std


def run_within_cls(data, args, models_dir):
    """Within-state K-fold CV for the classification-head variants.

    Source and target are the same state. The model is loaded once and a pristine
    CPU snapshot taken, which each fold restores, so folds share an identical
    init.

    Per fold: a stratified 15% val carve of the fold's training rows, best-by-QWK
    keeper with patience 2, balanced class weights, then predict the held-out
    fold; out-of-fold predictions are concatenated.

    Under ``--rag`` the exemplars are medoids of the fold's training rows only,
    self-excluded, so a held-out row never contributes one.

    Tags: qwen_cls[_rag|_lora]_within[_5star]_tgt100_<S>.
    """
    import torch
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
    init_state = _trainable_state(model)  # pristine snapshot, restored per fold

    # Feature cache: with --adapt head the backbone is frozen, so
    # the SAME pooled states serve all K folds — one extraction pass instead of
    # K x EPOCHS backbone passes. NB with --rag the exemplar block is per-fold
    # (medoids come from the fold's train rows), so each fold's wrapped strings
    # are distinct and get their own cache entry; the saving there is the 4
    # epochs within a fold rather than reuse across folds.
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
        model.load_state_dict(init_state, strict=False)  # delta-only, see _trainable_state
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

    model = None   # drop the reference (not `del`: the phase/predict closures capture it)
    if DEVICE == "cuda":
        import torch as _t
        _t.cuda.empty_cache()
    shutil.rmtree(models_dir / method, ignore_errors=True)
    return method, point, std


def _method_tag_from_args(a):
    variant = "cls_rag" if getattr(a, "rag", False) else (
        "cls_lora" if getattr(a, "adapt", "head") == "lora" else "cls")
    infix = scale_infix(getattr(a, "rating_scale", "3star"))
    if getattr(a, "within_cv", 0) and a.within_cv > 0:  # within-state mode
        return f"qwen_{variant}_within{infix}_tgt100_{a.tgt_name}"
    PCT = (100 if getattr(a, "target_cv_folds", 0) > 0
           else int(round(getattr(a, "target_frac", 0.0) * 100)))
    src = getattr(a, "pool_name", None) if getattr(a, "pool_sources", None) \
        else getattr(a, "src_name", None)
    return f"xfer_qwen_{variant}{infix}_tgt{PCT}_{src}2{a.tgt_name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM classification-head transfer (head / +rag / +lora), "
                    "sequential pretrain->finetune.")
    add_transfer_args(parser)
    parser.add_argument("--adapt", choices=("head", "lora"), default="head",
                        help="'head': frozen backbone, train only the K-way head. "
                             "'lora': LoRA adapters + head (task_type SEQ_CLS).")
    parser.add_argument("--rag", action="store_true",
                        help="Prefix every input with per-class medoid exemplars "
                             "(pretrain/finetune pools; see module docstring).")
    parser.add_argument("--model", default=DEFAULT_CLS_LLM_PATH,
                        help=f"Dense Qwen checkpoint (default {DEFAULT_CLS_LLM_PATH}; "
                             "fetch with download_qwen.py --repo-id Qwen/Qwen3-4B).")
    parser.add_argument("--embed-model",
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedder for --rag medoid selection (shared MiniLM).")
    parser.add_argument("--max-exemplar-chars", type=int, default=1200)
    # Accepted and ignored. This once carved a fraction of the pool into a
    # manufactured finetune phase at p=0; the zero-shot cell is now a single
    # pretrain phase on the full pool. The flag survives only so that command
    # lines recorded in older logs still parse rather than hard-failing.
    parser.add_argument("--zero-shot-finetune-frac", type=float, default=0.0,
                        help="Ignored (retained for backward compatibility): p=0 "
                             "is a single pretrain phase on the full pool.")
    parser.add_argument("--within-cv", type=int, default=0,
                        help="WITHIN-STATE mode: K-fold CV over ONE "
                             "state (pass its raw CSV as BOTH --source and --target, "
                             "same --src-name/--tgt-name). Fresh head/LoRA per fold "
                             "from a pristine snapshot; --rag draws per-class medoid "
                             "exemplars from the fold's train rows only. Tag "
                             "qwen_cls[_rag|_lora]_within[_5star]_tgt100_<S>. "
                             "0 (default) = normal cross-state transfer.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    configure_verbosity(args.verbose)

    if args.within_cv and args.within_cv > 0:
        if args.pool_sources:
            parser.error("--within-cv is single-state (source == target); it is "
                         "incompatible with --pool-sources.")
        data = load_and_prepare(
            args.source, args.target, args.src_name, args.tgt_name,
            text_mode=args.text_mode, compliance_mode=args.compliance,
            scale=args.rating_scale)
        models_dir = run_artifact_dir(args.output) / "llm_cls_models"
        method, point, std = run_within_cls(data, args, models_dir)
        log_transfer_result(method, point, std, output_path=args.output,
                            notes=method, source=data.tgt_name,
                            target=data.tgt_name, classes=data.labels)
        print(f"\nDone. See {args.output} for within-state LLM-cls scores.")
        return

    if args.pool_sources:
        data = load_and_prepare_pooled(
            parse_pool_sources(args.pool_sources), args.tgt_name, args.target,
            text_mode=args.text_mode, compliance_mode=args.compliance,
            scale=args.rating_scale, pool_name=args.pool_name)
    else:
        data = load_and_prepare(
            args.source, args.target, args.src_name, args.tgt_name,
            text_mode=args.text_mode, compliance_mode=args.compliance,
            scale=args.rating_scale)
    models_dir = run_artifact_dir(args.output) / "llm_cls_models"
    method, point, std = run_transfer_cls(data, args, models_dir)
    log_transfer_result(method, point, std, output_path=args.output, notes=method,
                        source=data.src_name, target=data.tgt_name,
                        classes=data.labels)
    print(f"\nDone. See {args.output} for transfer scores.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — always leave a FAILED row
        traceback.print_exc()
        _p = argparse.ArgumentParser(add_help=False)
        add_transfer_args(_p)
        _p.add_argument("--adapt", choices=("head", "lora"), default="head")
        _p.add_argument("--rag", action="store_true")
        _p.add_argument("--within-cv", type=int, default=0)
        _a, _ = _p.parse_known_args()
        _within = getattr(_a, "within_cv", 0) and _a.within_cv > 0
        _src = (_a.tgt_name if _within
                else (_a.pool_name if _a.pool_sources else _a.src_name))
        log_failed_transfer(_method_tag_from_args(_a), output_path=_a.output,
                            source=_src, target=_a.tgt_name,
                            error=f"{type(exc).__name__}: {exc}")
        raise
