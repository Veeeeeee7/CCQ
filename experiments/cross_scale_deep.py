"""Cross-rubric transfer, deep stage: TabNet, TabPFN, ModernBERT and Qwen3-4B.

Companion to `cross_scale.py`, which holds the tabular half. Every function here
returns a continuous score per test row; none maps a prediction onto the target's
label scale.

The regression variants live here and import their plumbing from the classifier
modules rather than modifying them.

Trainable models run sequentially: pretrain on the source, then finetune on the
target adaptation rows, so each phase sees exactly one rating scale. Targets
arrive already z-scored by `cross_scale.rating_stats`; nothing here
re-standardizes. Early stopping selects on c-index over a stratified 15% slice of
the training phase.
"""
from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ranking_metrics import concordance_index
from transfer_common import SEED, balanced_class_weights, stratified_source_split

# Regression phases are UNWEIGHTED, for the same reason the tabular regressors
# are: balanced sample weights combined with squared-error loss distort a skewed
# ordinal target. Every model in this module is a regressor, so the rule has no
# exceptions here.
UNWEIGHTED_NOTE = "regression phases unweighted by design"


@dataclass
class Phase:
    """One training block: features/texts + the two label views a phase needs.

    `y` is the Z-SCORED training target, produced by the caller from THAT
    state's own `cross_scale.rating_stats` transform — the same transform the
    pooled path uses, so both regimes share one definition of "standardized
    rating" and neither asserts any correspondence between two rubrics' levels.

    `y_raw` is the state's native integer levels, kept because early-stopping
    splits must stratify on real classes (a z-scored float has no strata).
    """
    X: "np.ndarray | list[str]"
    y: np.ndarray                 # z-scored training target
    y_raw: np.ndarray             # native integer ratings, for stratification only

    def __len__(self) -> int:
        return len(self.y)


# Standardization happens ONCE per state, in cross_scale.rating_stats, and
# arrives here already applied in Phase.y. Nothing in this module re-standardizes:
# a per-phase transform would make the sequential path disagree with the pooled
# one about what a standardized rating is.


def _inner_val(y: np.ndarray, val_frac: float = 0.15, seed: int = SEED):
    """Stratified inner split for early stopping, carved from the training phase.

    Stratifies on the integer rating; a phase contains only one scale.
    """
    return stratified_source_split(y.astype(int), val_frac=val_frac, seed=seed)


def predict_scores_ordered(model, tok, texts, max_len: int, batch_size: int,
                           score_fn) -> np.ndarray:
    """Ordered, loss-free inference: one score per input row, in input order.

    Runs the batching by hand rather than through a Trainer, so no sampler, loss
    or labels are involved and the output order is guaranteed. `score_fn` maps a
    batch of raw logits to a 1-D score. Raises if the scored row count disagrees
    with the input count.
    """
    import torch

    model.eval()
    device = next(model.parameters()).device
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = [str(t) for t in texts[i:i + batch_size]]
            enc = tok(chunk, truncation=True, padding=True, max_length=max_len,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            out.append(np.asarray(score_fn(logits.float().cpu()), dtype=float).ravel())
    scores = np.concatenate(out) if out else np.empty(0)
    if len(scores) != len(texts):
        raise RuntimeError(f"scored {len(scores)} rows for {len(texts)} inputs")
    return scores


def make_trainable_keeper(metric: str = "eval_c_index", patience: int = 2):
    """Best-epoch keeper that snapshots only the trainable parameters.

    Cloning a full `state_dict()` is around 15 GB for fp32 Qwen3-4B and will OOM
    a modest node. Frozen parameters cannot change during training, so restoring
    the trainable subset with `strict=False` is equivalent.
    """
    from transformers import TrainerCallback

    class _BestTrainableInRAM(TrainerCallback):
        def __init__(self):
            self.best = None
            self.best_state = None
            self.wait = 0

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
            if metrics is None or model is None:
                return
            val = metrics.get(metric, metrics.get(metric.removeprefix("eval_")))
            if val is None:
                return
            if self.best is None or val > self.best:
                self.best = val
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in model.named_parameters()
                                   if v.requires_grad}
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= patience:
                    control.should_training_stop = True

    return _BestTrainableInRAM()


def _rank_metric(y_true, scores) -> dict:
    """Early-stopping signal for the regression heads: c-index on the inner val
    slice. Selecting on the metric the experiment reports is the point — MSE
    would select for calibration, which this experiment explicitly does not
    measure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"c_index": float(concordance_index(y_true, scores))}


# =============================================================================
# Stage 2 — TabNet / TabPFN over the shared MiniLM embeddings
# =============================================================================
def fit_predict_tabnet(pre: Phase, ft: "Phase | None", X_test: np.ndarray,
                       seed: int = SEED) -> np.ndarray:
    """TabNetRegressor, pretrain -> optional warm-start finetune -> score."""
    import torch

    from pytorch_tabnet.tab_model import TabNetRegressor

    from transfer_tabular import _tabnet_batch_kwargs

    def _fit(model, phase: Phase, warm: bool):
        # Stratify on the RAW integer levels; train on the already-z-scored
        # target, so the warm-started phase 2 is not fighting a scale change.
        tr, va = _inner_val(phase.y_raw, seed=seed)
        y2 = np.asarray(phase.y, dtype=np.float32).reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                phase.X[tr], y2[tr],
                eval_set=[(phase.X[va], y2[va])], eval_metric=["mse"],
                max_epochs=200, patience=20,
                # Without this an undersized phase trains zero batches.
                **_tabnet_batch_kwargs(len(tr)),
                warm_start=warm,
            )
        return model

    model = TabNetRegressor(
        # Mirrors tabular_dl's TabNetClassifier config exactly; only the task
        # changes. entmax + StepLR are pytorch-tabnet's own census_example
        # settings.
        n_d=16, n_a=16, n_steps=3, gamma=1.5, lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam, optimizer_params={"lr": 2e-2},
        scheduler_params={"step_size": 20, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax", seed=seed, verbose=0,
    )
    _fit(model, pre, warm=False)
    if ft is not None and len(ft) > 0:
        _fit(model, ft, warm=True)     # SEQUENTIAL: second phase, target scale
    return np.asarray(model.predict(X_test), dtype=float).ravel()


def fit_predict_tabpfn(pre: Phase, ft: "Phase | None", X_test: np.ndarray,
                       seed: int = SEED) -> np.ndarray:
    """TabPFNRegressor. In-context, so there is no sequential variant: the
    adaptation rows join the context set instead of forming a second phase.

    That means this is the ONE deep model whose context can span two scales at
    p>0, so the adaptation labels are z-scored onto the source's scale first —
    monotone within each domain, hence rank-preserving, and it stops the context
    from carrying two incompatible numeric conventions.
    """
    import torch

    from tabpfn import TabPFNRegressor

    from transfer_tabular import TABPFN_SAMPLE_CAP

    # Both phases arrive already z-scored by their own state's transform
    # (cross_scale.rating_stats), so the in-context set can simply be their
    # concatenation -- no second standardization, and no crosswalk asserted.
    X, y = pre.X, np.asarray(pre.y, dtype=float)
    if ft is not None and len(ft) > 0:
        X = np.vstack([pre.X, ft.X])
        y = np.concatenate([np.asarray(pre.y, float), np.asarray(ft.y, float)])

    if len(y) > TABPFN_SAMPLE_CAP:
        # TabPFN v2's own pretraining limit, applied identically to every TabPFN
        # cell in the suite. Plain random subsample: the target is
        # continuous here, so there is no class to stratify on.
        rng = np.random.RandomState(seed)
        keep = rng.choice(len(y), size=TABPFN_SAMPLE_CAP, replace=False)
        print(f"    [tabpfn] {len(y)} context rows > cap {TABPFN_SAMPLE_CAP}; "
              f"subsampling")
        X, y = X[keep], y[keep]

    model = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
        return np.asarray(model.predict(X_test), dtype=float).ravel()


# =============================================================================
# Stage 3 — ModernBERT regression head, and CORN's native monotone score
# =============================================================================
_FLOAT_DS_CLS = None


def _FloatTextDataset(encodings: dict, labels):  # noqa: N802 — used like a class
    """`llm._TextDataset` casts labels to int64, which would silently turn an MSE
    objective into garbage. This is the same dataset with FLOAT labels.

    Built lazily so importing this module does not require torch: the registry,
    `Phase` and `STAGE` are plain metadata that `cross_scale.py` inspects on the
    CPU-only Stage 1 path, where the deep stack may not be installed at all.
    """
    global _FLOAT_DS_CLS
    if _FLOAT_DS_CLS is None:
        import torch

        class _FloatDS(torch.utils.data.Dataset):
            def __init__(self, encodings: dict, labels):
                self.encodings = encodings
                self.labels = np.asarray(labels, dtype=np.float32)

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, i: int) -> dict:
                item = {k: torch.as_tensor(v[i]) for k, v in self.encodings.items()}
                item["labels"] = torch.as_tensor(self.labels[i])
                return item

        _FLOAT_DS_CLS = _FloatDS
    return _FLOAT_DS_CLS(encodings, labels)


def _plm_args(out_dir: Path, epochs: int, lr: float, batch: int, eval_batch: int):
    from llm import (
        GROUP_BY_LENGTH, NUM_WORKERS, USE_BF16, USE_FP16, WARMUP_RATIO,
        make_training_arguments,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return make_training_arguments(
        output_dir=str(out_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch, per_device_eval_batch_size=eval_batch,
        group_by_length=GROUP_BY_LENGTH, learning_rate=lr,
        warmup_ratio=WARMUP_RATIO, weight_decay=0.01, eval_strategy="epoch",
        # save_strategy="no" for the same reason transfer_plm uses it: the
        # best-epoch weights are kept in RAM by make_best_model_keeper, and
        # per-epoch checkpoints would fill the scratch quota.
        save_strategy="no", logging_steps=50, seed=SEED, report_to=[],
        dataloader_num_workers=NUM_WORKERS, fp16=USE_FP16, bf16=USE_BF16,
    )


def fit_predict_mbert(pre: Phase, ft: "Phase | None", test_texts: list,
                      out_dir: Path, seed: int = SEED) -> np.ndarray:
    """ModernBERT with a 1-unit regression head (HF uses MSE when num_labels==1).

    Sequential: fine-tune on the source, then continue on the target adaptation
    rows from the source weights. Best epoch in each phase is chosen by c-index
    on an inner 15% slice of THAT phase's training rows.
    """
    import torch

    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer,
    )

    from llm import (
        BATCH_SIZE, EPOCHS, EVAL_BATCH_SIZE, LR, MAX_LEN, MODEL_NAME,
        make_best_model_keeper, trainer_tokenizer_kwarg,
    )

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1)          # -> problem_type="regression", MSELoss
    collator = DataCollatorWithPadding(tokenizer=tok)

    def _enc(texts):
        return tok(list(texts), truncation=True, padding=False, max_length=MAX_LEN)

    def _run_phase(phase: Phase, tag: str):
        # Stratify on RAW levels; the target arrives already z-scored (Phase.y).
        tr, va = _inner_val(phase.y_raw, seed=seed)
        yz = np.asarray(phase.y, dtype=float)
        texts = list(phase.X)
        train_ds = _FloatTextDataset(_enc([texts[i] for i in tr]), yz[tr])
        val_ds = _FloatTextDataset(_enc([texts[i] for i in va]), yz[va])
        keeper = make_best_model_keeper(metric="eval_c_index",
                                        greater_is_better=True, patience=2)
        trainer = Trainer(
            model=model, args=_plm_args(out_dir / tag, EPOCHS, LR,
                                        BATCH_SIZE, EVAL_BATCH_SIZE),
            train_dataset=train_ds, eval_dataset=val_ds,
            **trainer_tokenizer_kwarg(tok), data_collator=collator,
            compute_metrics=lambda ep: _rank_metric(
                np.asarray(ep[1]).ravel(), np.asarray(ep[0]).ravel()),
            callbacks=[keeper],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.train()
        if keeper.best_state is not None:
            model.load_state_dict(keeper.best_state)
        return trainer

    _run_phase(pre, "pretrain")
    if ft is not None and len(ft) > 0:
        _run_phase(ft, "finetune")

    return predict_scores_ordered(model, tok, list(test_texts), MAX_LEN,
                                  EVAL_BATCH_SIZE, lambda lg: lg.squeeze(-1))


def _rag_prefix(query_texts, pool_texts, pool_y, pool_emb, max_chars: int = 1200):
    """Prefix each query with ONE medoid exemplar per class, in ascending order.

    Uses the same selector every other RAG variant in the suite uses
    (`llm_common.class_medoid_exemplar_indices`), and the exemplars always come
    from the CURRENT PHASE's training pool — so their labels are on the same
    scale the model is being trained on. That is why RAG needs no special
    handling here: the sequential regime already guarantees one scale per phase.
    """
    from llm_common import class_medoid_exemplar_indices

    rep = class_medoid_exemplar_indices(pool_emb, np.asarray(pool_y).astype(int))
    block = "\n\n".join(
        f"### Example (rating {int(pool_y[i])})\n{str(pool_texts[i])[:max_chars]}"
        for i in rep)
    return [f"{block}\n\n### Provider to rate\n{t}" for t in query_texts]


def fit_predict_qwen(pre: Phase, ft: "Phase | None", test_texts: list,
                     out_dir: Path, *, adapt: str = "head", rag: bool = False,
                     model_name: str = "", pre_emb=None, ft_emb=None,
                     seed: int = SEED) -> np.ndarray:
    """Qwen3-4B, `num_labels=1`, in the three LOSO variants.

    `adapt="head"` freezes the backbone and trains only `score`; `adapt="lora"`
    attaches LoRA adapters and trains those plus `score`. `rag=True` prefixes
    per-class medoid exemplars to every input, drawn from the phase's own pool.

    NO feature cache and no `_HeadOnly` (see the module docstring): the frozen
    variant simply sets `requires_grad=False` on the backbone and trains through
    the ordinary HF Trainer: slower per epoch, and with no cache-key or
    feature-extraction correctness surface at all.
    """
    import torch

    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer,
    )

    from llm import trainer_tokenizer_kwarg
    # DEFAULT_CLS_LLM_PATH, not LLM_CLS_MODEL: the latter is the SHELL variable
    # the drivers export, while the module-level constant it defaults to is
    # DEFAULT_CLS_LLM_PATH -- a LOCAL scratch path to weights fetched by
    # download_qwen.py, not an HF repo id.
    from transfer_llm_cls import (
        BATCH_SIZE, DEFAULT_CLS_LLM_PATH, EPOCHS, EVAL_BATCH_SIZE, GRAD_ACCUM,
        HEAD_LR, LORA_LR, MAX_LEN,
    )

    name = model_name or DEFAULT_CLS_LLM_PATH
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # low_cpu_mem_usage streams the checkpoint shard-by-shard instead of
    # materialising a second full fp32 copy in host RAM during load. Purely a
    # LOADING strategy, so the weights are identical, but it halves peak host
    # memory -- which matters when a 15 GB model is loaded under a 50 GB limit.
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=1, low_cpu_mem_usage=True)
    model.config.pad_token_id = tok.pad_token_id
    if torch.cuda.is_available():
        # Place the model explicitly. Left on CPU, the Trainer would move only
        # the head to cuda and strand the backbone on a different device.
        model = model.to("cuda")

    if adapt == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=["score"]))
    else:
        for p in model.parameters():
            p.requires_grad = False
        head = model.score if hasattr(model, "score") else model.base_model.model.score
        for p in head.parameters():
            p.requires_grad = True

    lr = LORA_LR if adapt == "lora" else HEAD_LR
    collator = DataCollatorWithPadding(tokenizer=tok)

    def _enc(texts):
        return tok(list(texts), truncation=True, padding=False, max_length=MAX_LEN)

    def _texts_for(phase: Phase, emb):
        t = list(phase.X)
        if not rag:
            return t, t
        if emb is None:
            raise ValueError("rag=True needs the phase's retrieval embeddings.")
        return _rag_prefix(t, t, phase.y, emb), t

    def _run_phase(phase: Phase, emb, tag: str):
        wrapped, _ = _texts_for(phase, emb)
        # Stratify on RAW levels; the target arrives already z-scored (Phase.y).
        tr, va = _inner_val(phase.y_raw, seed=seed)
        yz = np.asarray(phase.y, dtype=float)
        # Trainable-only snapshot: KB/MB of host RAM instead of 15 GB, and
        # equivalent, since a frozen parameter cannot move.
        keeper = make_trainable_keeper(metric="eval_c_index", patience=2)
        args = _plm_args(out_dir / tag, EPOCHS, lr, BATCH_SIZE, EVAL_BATCH_SIZE)
        args.gradient_accumulation_steps = GRAD_ACCUM
        train_ds = _FloatTextDataset(_enc([wrapped[i] for i in tr]), yz[tr])
        val_ds = _FloatTextDataset(_enc([wrapped[i] for i in va]), yz[va])
        trainer = Trainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=val_ds,
            **trainer_tokenizer_kwarg(tok), data_collator=collator,
            compute_metrics=lambda ep: _rank_metric(
                np.asarray(ep[1]).ravel(), np.asarray(ep[0]).ravel()),
            callbacks=[keeper],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.train()
        if keeper.best_state is not None:
            # strict=False: the payload is the trainable delta only, exactly as
            # transfer_llm_cls pairs with _trainable_state.
            model.load_state_dict(keeper.best_state, strict=False)
        # Drop the trainer + its tokenized datasets before the next phase; on a
        # 50 G node the finetune phase must not be paying for the pretrain one.
        del trainer, keeper, train_ds, val_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _run_phase(pre, pre_emb, "pretrain")
    last_phase, last_emb = pre, pre_emb
    if ft is not None and len(ft) > 0:
        _run_phase(ft, ft_emb, "finetune")
        last_phase, last_emb = ft, ft_emb

    # Test-time RAG exemplars come from the LAST phase's pool — the same
    # distribution the head was last trained against (source at p=0, target
    # adaptation at p>0), matching the LOSO convention.
    if rag:
        test_in = _rag_prefix(test_texts, list(last_phase.X), last_phase.y, last_emb)
    else:
        test_in = list(test_texts)

    return predict_scores_ordered(model, tok, test_in, MAX_LEN, EVAL_BATCH_SIZE,
                                  lambda lg: lg.squeeze(-1))


# =============================================================================
# Registry — consumed by cross_scale.py
# =============================================================================
# input: "emb" = the shared MiniLM matrix, "text" = serialized rows.
# sequential: True = pretrain(source) then finetune(target adapt), one scale per
# phase. False = one in-context fit (TabPFN), which z-scores per domain instead.
DEEP_MODELS: dict[str, dict] = {
    "tabnet":     {"input": "emb",  "sequential": True,  "fn": fit_predict_tabnet},
    "tabpfn":     {"input": "emb",  "sequential": False, "fn": fit_predict_tabpfn},
    "mbert":      {"input": "text", "sequential": True,  "fn": fit_predict_mbert},
    "qwen":       {"input": "text", "sequential": True,
                   "fn": lambda *a, **k: fit_predict_qwen(*a, adapt="head", **k)},
    "qwen_rag":   {"input": "text", "sequential": True, "needs_emb": True,
                   "fn": lambda *a, **k: fit_predict_qwen(*a, adapt="head", rag=True, **k)},
    "qwen_lora":  {"input": "text", "sequential": True,
                   "fn": lambda *a, **k: fit_predict_qwen(*a, adapt="lora", **k)},
}

STAGE = {"tabnet": 2, "tabpfn": 2, "mbert": 3,
         "qwen": 4, "qwen_rag": 4, "qwen_lora": 4}
