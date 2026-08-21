"""Tabular models for cross-state transfer over a shared embedding space.

The TabNet and TabPFN arms of the cross-state experiment. The states publish
disjoint feature schemas, so both states are serialized to text and embedded into
a fixed-width sentence-embedding vector; the models then train on one state and
apply to the other in that shared space.

TabPFN performs in-context learning and has no sequential variant. TabNet trains
conventionally, with balanced class weights and early stopping, and does.

The split protocol comes from `transfer_common.split_target_fewshot`: a seeded
stratified 20% target slice carved once and scored at every point on the
supervision curve, with adaptation rows drawn from the remaining 80%.

    python transfer_tabular.py \\
        --source data/wi_records_cleaned_raw.csv --target data/ca_records_cleaned_raw.csv \\
        --src-name WI --tgt-name CA \\
        --target-frac 0.0 --target-test-frac 0.2 --models tabnet tabpfn
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
from utils import SEED, _stratification_labels, compute_metrics, run_artifact_dir  # noqa: E402

# Fixed featurizer, deliberately not a flag: every method that consumes
# embeddings must consume the SAME ones for the comparison to mean anything. The
# 384-d output is already under TabPFN v2's feature cap, so no reduction is
# needed.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
TABPFN_SAMPLE_CAP = 10_000   # TabPFN v2's pretraining context limit


# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# -----------------------------------------------------------------------------
# Featurizer: sentence-transformer with chunk-mean-pooling + on-disk cache
# -----------------------------------------------------------------------------
def _safe_model_tag(name: str) -> str:
    return name.replace("/", "_").replace("-", "_")


def _texts_digest(texts) -> str:
    """Content hash of the serialized texts, for the embedding cache key.

    Keying on row count alone would treat a serialization change that preserved
    the row count as a cache hit. Stdlib-only, so any environment computes the
    same key."""
    import hashlib
    h = hashlib.md5()
    for t in texts:
        h.update(str(t).encode("utf-8", "replace"))
        h.update(b"\x1e")  # record separator: ("ab","c") != ("a","bc")
    return h.hexdigest()[:12]


def _cache_path(cache_dir: Path, state: str, text_mode: str, compliance: str,
                model_name: str, digest: str) -> Path:
    key = (f"{state}_{text_mode}_{compliance}_{_safe_model_tag(model_name)}"
           f"_{digest}.npy")
    return cache_dir / key


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """Write the embedding cache via a temp file and `os.replace`.

    A truncated `.npy` passes `path.exists()` but makes `np.load` raise, so a
    partial write would poison every later run touching that state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Temp name ends in .npy so np.save writes exactly here (it appends .npy only
    # when the name lacks that suffix) -> the written path is deterministic.
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npy")
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)          # atomic within the same directory
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _embed_chunked(texts, model, batch_size: int = 64) -> np.ndarray:
    """Embed each string, mean-pooling over non-overlapping token chunks.

    Rows longer than the encoder's max sequence length would otherwise be
    truncated. Each chunk is encoded and unit-normalized, the chunk vectors are
    averaged, and the mean is re-normalized. Single-chunk rows reduce to the
    ordinary normalized embedding.
    """
    tok = model.tokenizer
    max_len = int(getattr(model, "max_seq_length", 256) or 256)
    chunk_tokens = max(8, max_len - 2)

    # 1) Build the flat list of chunk-strings and remember each row's span.
    chunk_strings: list[str] = []
    spans: list[tuple[int, int]] = []
    for t in texts:
        t = "" if t is None else str(t)
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) == 0:
            ids = tok.encode(" ", add_special_tokens=False) or [tok.unk_token_id or 0]
        start = len(chunk_strings)
        for i in range(0, len(ids), chunk_tokens):
            piece = ids[i:i + chunk_tokens]
            chunk_strings.append(tok.decode(piece, skip_special_tokens=True))
        spans.append((start, len(chunk_strings)))

    # 2) Encode all chunks at once (unit-normalized), then mean-pool per row.
    chunk_emb = model.encode(
        chunk_strings, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,  # tqdm floods the .err
    ).astype(np.float32)

    out = np.zeros((len(texts), chunk_emb.shape[1]), dtype=np.float32)
    for r, (s, e) in enumerate(spans):
        v = chunk_emb[s:e].mean(axis=0)
        n = np.linalg.norm(v)
        out[r] = v / n if n > 0 else v
    n_long = sum(1 for s, e in spans if e - s > 1)
    print(f"    [embed] {len(texts)} rows -> {chunk_emb.shape[1]}-d; "
          f"{n_long} rows needed >1 chunk (mean-pooled)")
    return out


def embed_state(texts, state: str, text_mode: str, compliance: str,
                cache_dir: Path, model_name: str = EMBED_MODEL_NAME,
                batch_size: int = 64, device: "str | None" = None,
                model=None) -> np.ndarray:
    """Return the (n_rows, dim) embedding for one state's full serialized text,
    using an on-disk cache keyed by (state, text_mode, compliance, model_name).

    Each state is reused across its 3 sources/targets, so encoding once and
    caching turns 12x2 encodes into 4 (/). The cache key includes a
    CONTENT HASH of the texts so any serialization
    change is a miss by construction; the row-count check below stays as a
    belt-and-suspenders guard."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = list(texts)
    path = _cache_path(cache_dir, state, text_mode, compliance, model_name,
                       _texts_digest(texts))

    if path.exists():
        # a truncated cache (from an earlier interrupted/quota-failed write)
        # passes exists() but raises EOFError/ValueError in np.load. Treat any load
        # failure as a cache miss and re-encode, rather than propagating it and
        # killing an otherwise-healthy cell (this is the exact failure that
        # cascaded across a full-target CV run's WI/GA cells).
        try:
            cached = np.load(path)
        except (EOFError, ValueError, OSError) as exc:
            print(f"    [embed] cache {path.name} unreadable ({type(exc).__name__}: "
                  f"{exc}); re-encoding")
            cached = None
        if cached is not None:
            if cached.shape[0] == len(texts):
                print(f"    [embed] cache hit {path.name} ({cached.shape})")
                return cached.astype(np.float32)
            print(f"    [embed] cache row mismatch for {path.name} "
                  f"({cached.shape[0]} != {len(texts)}); re-encoding")

    if model is None:
        from sentence_transformers import SentenceTransformer
        dev = device or _device()
        print(f"    [embed] loading {model_name} on {dev}")
        model = SentenceTransformer(model_name, device=dev)
    emb = _embed_chunked(texts, model, batch_size=batch_size)
    _atomic_save(path, emb)            # atomic write, never a partial file
    print(f"    [embed] cached -> {path}")
    return emb


def embed_dataframe_cv(df, compliance: str, state: str, output_path=None,
                       cache_dir=None, model_name: str = EMBED_MODEL_NAME):
    """Serialize every row of ``df`` to the shared ``textualized_full`` view and
    sentence-embed it, returning ``(emb_df, feature_cols)`` where ``emb_df`` has
    ``emb_<i>`` columns + the target, in the ORIGINAL row order (positional folds
    stay aligned).

    Shared by the within-state ``--features text`` paths of tabular_dl.py AND
    baselines_ml.py so both produce the SAME embedding matrix (and hit the SAME
    on-disk cache) that transfer_tabular uses cross-state -- one embedder, one
    place to change it."""
    import pandas as pd
    from text_serialization import serialize_dataframe
    from utils import TARGET_COL, run_artifact_dir

    if cache_dir is None:
        cache_dir = run_artifact_dir(output_path) / "embeddings"
    texts = serialize_dataframe(df, compliance).tolist()
    print(f"Serialized {len(texts)} rows ({compliance}); embedding with "
          f"{model_name} (cache: {cache_dir})...")
    emb = embed_state(texts, state, "textualized_full", compliance,
                      cache_dir, model_name)
    feature_cols = [f"emb_{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=feature_cols, index=df.index)
    emb_df[TARGET_COL] = df[TARGET_COL].to_numpy()
    print(f"Using {len(feature_cols)} sentence-embedding features (raw text).")
    return emb_df, feature_cols


# -----------------------------------------------------------------------------
# Proba alignment (so columns match data.labels even if a class is absent from fit)
# -----------------------------------------------------------------------------
def _align_proba(proba: np.ndarray, model_classes, n_classes: int) -> np.ndarray:
    """Re-order/pad predict_proba columns to the canonical 0..n_classes-1 index
    space (== data.labels order). model_classes are the idx-space labels the model
    saw, in column order."""
    aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for col, cls in enumerate(model_classes):
        aligned[:, int(cls)] = proba[:, col]
    # guard against all-zero rows (a class the model never emits) -> renormalize
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return aligned / row_sums


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def _predict_fitted(model, X_test, n_classes):
    """Score a fitted tabular model -> (pred_idx, aligned proba). Split out so the zero-shot multi-target path fits once and predicts
    many targets."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = model.predict_proba(X_test)
    proba = _align_proba(np.asarray(proba), model.classes_, n_classes)
    return proba.argmax(axis=1), proba


def _fit_tabpfn(X_fit, y_fit, n_classes, seed=SEED, balanced_subsample=False):
    """TabPFN v2: .fit() loads the fit set as context; returns the fitted model.

    balanced_subsample=False (default) mirrors tabular_dl.run_tabpfn_cv's plain
    'tabpfn'. balanced_subsample=True downsamples every class in the fit set to
    the smallest class count BEFORE fitting (TabPFN has no native class
    weighting) -- the transfer analogue of tabular_dl's 'tabpfn_balanced'. If the
    fit set still exceeds the sample cap, stratified-subsample to the cap."""
    from tabpfn import TabPFNClassifier

    if balanced_subsample:
        rng = np.random.default_rng(seed)
        classes, counts = np.unique(y_fit, return_counts=True)
        min_count = int(counts.min())
        keep = []
        for cls in classes:
            cls_idx = np.where(y_fit == cls)[0]
            keep.extend(rng.choice(cls_idx, size=min_count, replace=False).tolist())
        keep = np.sort(np.array(keep, dtype=int))
        print(f"    [tabpfn] class-balanced subsample: {len(X_fit)} -> {len(keep)} "
              f"rows ({min_count}/class x {len(classes)} classes)")
        X_fit, y_fit = X_fit[keep], y_fit[keep]

    if len(X_fit) > TABPFN_SAMPLE_CAP:
        print(f"    [tabpfn] fit set {len(X_fit)} > cap {TABPFN_SAMPLE_CAP}; "
              f"stratified-subsampling to {TABPFN_SAMPLE_CAP}")
        # _stratification_labels: rare-class safe — a 1-member
        # class in y_fit would crash the stratified subsample; fold it into its
        # nearest ordinal neighbour for the draw only (labels stay true).
        keep, _ = train_test_split(
            np.arange(len(X_fit)), train_size=TABPFN_SAMPLE_CAP,
            stratify=_stratification_labels(y_fit, 2), random_state=seed)
        X_fit, y_fit = X_fit[keep], y_fit[keep]

    model = TabPFNClassifier(n_estimators=8, device=_device(), random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_fit, y_fit)
    return model


def _fit_predict_tabpfn(X_fit, y_fit, X_test, n_classes, seed=SEED,
                        balanced_subsample=False):
    """Original fit+predict path (unchanged behavior)."""
    model = _fit_tabpfn(X_fit, y_fit, n_classes, seed=seed,
                        balanced_subsample=balanced_subsample)
    return _predict_fitted(model, X_test, n_classes)


def _tabnet_batch_kwargs(n_train: int, batch_size: int = 256,
                         virtual_batch_size: int = 64) -> dict:
    """Batch kwargs sized to `n_train`.

    pytorch-tabnet defaults to `drop_last=True`, so a training set smaller than
    one batch yields zero batches and an untrained network. Below `batch_size`
    the batch shrinks to exactly `n_train`; at or above it, nothing changes.
    """
    if n_train >= batch_size:
        return {"batch_size": batch_size, "virtual_batch_size": virtual_batch_size}
    bs = max(2, int(n_train))
    return {"batch_size": bs, "virtual_batch_size": min(virtual_batch_size, bs)}


def _fit_tabnet(X_fit, y_fit, n_classes, seed=SEED):
    """TabNet (pytorch-tabnet): reuse the tabular_dl.py hyperparameters, balanced
    class weighting (weights=1 -> inverse-frequency), and early stopping on a
    stratified slice carved from the FIT set (never the target test set).
    Returns the fitted model."""
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.metrics import Metric
    from sklearn.metrics import balanced_accuracy_score

    class BalancedAccuracy(Metric):
        def __init__(self):
            self._name = "balanced_accuracy"
            self._maximize = True

        def __call__(self, y_true, y_score):
            return balanced_accuracy_score(y_true, np.argmax(y_score, axis=1))

    # Stratified early-stopping slice from the fit set (15%, like the source-val
    # fraction used elsewhere). Guard tiny fit sets.
    es_frac = 0.15
    if len(X_fit) >= max(20, 2 * n_classes):
        # rare-class safe: same crash mode as the sequential
        # _split — a big-enough fit set can still hold a 1-member class.
        tr_idx, es_idx = stratified_source_split(y_fit, es_frac, seed)
    else:  # too small to stratify-split; train on all, eval on all
        tr_idx = es_idx = np.arange(len(X_fit))
    X_tr, y_tr = X_fit[tr_idx], y_fit[tr_idx]
    X_es, y_es = X_fit[es_idx], y_fit[es_idx]
    # Surface the realized class balance the weighting will counteract.
    _ = balanced_class_weights(y_tr, n_classes)

    model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=3, gamma=1.5, lambda_sparse=1e-4,
        cat_idxs=[], cat_dims=[], cat_emb_dim=4,
        optimizer_fn=torch.optim.Adam, optimizer_params={"lr": 2e-2},
        scheduler_params={"step_size": 20, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax", device_name=_device(), seed=seed, verbose=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric=[BalancedAccuracy],
            max_epochs=200, patience=40,
            **_tabnet_batch_kwargs(len(X_tr)),  # never a zero-batch phase
            weights=1,  # balanced (inverse-frequency) — matches tabular_dl
        )
    return model


def _fit_predict_tabnet(X_fit, y_fit, X_test, n_classes, seed=SEED):
    """Original fit+predict path (unchanged behavior)."""
    model = _fit_tabnet(X_fit, y_fit, n_classes, seed=seed)
    return _predict_fitted(model, X_test, n_classes)


def _fit_predict_tabnet_sequential(X_src, y_src, X_tgt, y_tgt, X_test, n_classes, seed=SEED):
    """SEQUENTIAL TabNet (Phase E): fit on SOURCE, then CONTINUE training
    (warm_start=True) on the TARGET adaptation rows, then predict test. Each phase
    carves its own stratified early-stop slice. This is the tabular analogue of
    transfer_plm.py's sequential pretrain->finetune; the pooled variant unions
    source+target into one fit (run_one_model default). TabPFN has no sequential
    twin -- it is in-context, not trained -- so only TabNet reaches here."""
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.metrics import Metric
    from sklearn.metrics import balanced_accuracy_score

    class BalancedAccuracy(Metric):
        def __init__(self):
            self._name = "balanced_accuracy"
            self._maximize = True

        def __call__(self, y_true, y_score):
            return balanced_accuracy_score(y_true, np.argmax(y_score, axis=1))

    def _split(X, y):
        # stratified_source_split rather than raw train_test_split, because it is
        # rare-class safe: the size guard below counts TOTAL rows only, so a large
        # target set holding a single member of one class passes it and then
        # crashes sklearn's stratify.
        if len(X) >= max(20, 2 * n_classes):
            tr, es = stratified_source_split(y, 0.15, seed)
        else:  # too small to stratify-split; train on all, eval on all
            tr = es = np.arange(len(X))
        return X[tr], y[tr], X[es], y[es]

    model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=3, gamma=1.5, lambda_sparse=1e-4,
        cat_idxs=[], cat_dims=[], cat_emb_dim=4,
        optimizer_fn=torch.optim.Adam, optimizer_params={"lr": 2e-2},
        scheduler_params={"step_size": 20, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax", device_name=_device(), seed=seed, verbose=0,
    )
    fit_kw = dict(eval_metric=[BalancedAccuracy], max_epochs=200, patience=40,
                  weights=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xtr, ytr, Xes, yes = _split(X_src, y_src)   # phase 1: source
        model.fit(Xtr, ytr, eval_set=[(Xes, yes)],
                  **_tabnet_batch_kwargs(len(Xtr)), **fit_kw)
        Xtr, ytr, Xes, yes = _split(X_tgt, y_tgt)   # phase 2: target (continue)
        # The per-phase batch guard matters MOST here: a small adaptation draw
        # can fall under the drop_last floor and silently skip the entire
        # finetune phase.
        model.fit(Xtr, ytr, eval_set=[(Xes, yes)], warm_start=True,
                  **_tabnet_batch_kwargs(len(Xtr)), **fit_kw)
        proba = model.predict_proba(X_test)
    proba = _align_proba(np.asarray(proba), model.classes_, n_classes)
    return proba.argmax(axis=1), proba


def _fit_predict_tabpfn_balanced(X_fit, y_fit, X_test, n_classes, seed=SEED):
    """Class-balanced-subsample TabPFN (see _fit_predict_tabpfn). Registered as a
    distinct model so its method tag reads xfer_tabpfn_balanced_*."""
    return _fit_predict_tabpfn(X_fit, y_fit, X_test, n_classes, seed=seed,
                               balanced_subsample=True)


_MODEL_FNS = {"tabpfn": _fit_predict_tabpfn,
              "tabpfn_balanced": _fit_predict_tabpfn_balanced,
              "tabnet": _fit_predict_tabnet}


def _fit_tabpfn_balanced(X_fit, y_fit, n_classes, seed=SEED):
    return _fit_tabpfn(X_fit, y_fit, n_classes, seed=seed, balanced_subsample=True)


# Fit-only variants (P1 zero-shot dedupe: fit once on source, predict many targets).
_MODEL_FIT_FNS = {"tabpfn": _fit_tabpfn,
                  "tabpfn_balanced": _fit_tabpfn_balanced,
                  "tabnet": _fit_tabnet}


# -----------------------------------------------------------------------------
# Classical sklearn baselines over the shared embeddings (dummy / lr / rf / xgb).
# -----------------------------------------------------------------------------
# The same estimators baselines_ml runs within-state, applied to the shared
# embeddings. Configs are imported from baselines_ml, so there is no second copy
# of the hyperparameters.
SKLEARN_MODELS = ("dummy", "lr", "rf", "xgb")


class _GlobalLabelClassifier:
    """Wrap an sklearn-style estimator so it fits on DENSELY-relabeled targets
    (0..m-1, which XGBoost 2.x requires) while exposing the GLOBAL class ids via
    ``.classes_`` and returning ``predict_proba`` columns in that same order.

    This makes the wrapped estimator a drop-in for _predict_fitted (which reads
    ``.classes_`` + ``.predict_proba`` and then _align_proba-widens to the full
    n_classes space) -- identical plumbing to TabPFN/TabNet. Dense relabeling is
    a no-op when the fit set already holds every class (the common cross-state
    case, since the source is a whole state), so it only matters for a degenerate
    fit set missing an ultra-rare ordinal class."""

    def __init__(self, estimator, balanced_weights: bool = False):
        self._est = estimator
        self._balanced_weights = balanced_weights
        self.classes_ = None

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        present = np.unique(y)                       # global ids in the fit set
        g2d = {int(g): i for i, g in enumerate(present)}
        y_dense = np.array([g2d[int(v)] for v in y])
        if self._balanced_weights:
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight(class_weight="balanced", y=y_dense)
            self._est.fit(X, y_dense, sample_weight=sw)
        else:
            self._est.fit(X, y_dense)
        # est.classes_ is the dense 0..m-1 space; lift each back to its global id
        # so downstream _align_proba scatters proba columns to the right slots.
        self.classes_ = present[self._est.classes_.astype(int)]
        return self

    def predict_proba(self, X):
        return self._est.predict_proba(X)


def _make_sklearn_estimator(name: str, n_classes: int):
    """Return (estimator, balanced_weights) from baselines_ml's factories, so the
    cross-state sklearn transfer uses byte-for-byte the within-state config."""
    from baselines_ml import make_dummy, make_lr, make_rf, make_xgb
    factories = {"dummy": make_dummy, "lr": make_lr, "rf": make_rf, "xgb": make_xgb}
    model, pre_kwargs = factories[name](n_classes)
    return model, bool(pre_kwargs.get("balanced_weights", False))


def _make_sklearn_fit(name: str):
    def _fit(X_fit, y_fit, n_classes, seed=SEED):
        est, balanced = _make_sklearn_estimator(name, n_classes)
        return _GlobalLabelClassifier(est, balanced_weights=balanced).fit(X_fit, y_fit)
    return _fit


def _make_sklearn_fit_predict(name: str):
    _fit = _make_sklearn_fit(name)

    def _fit_predict(X_fit, y_fit, X_test, n_classes, seed=SEED):
        model = _fit(X_fit, y_fit, n_classes, seed=seed)
        return _predict_fitted(model, X_test, n_classes)
    return _fit_predict


for _sk in SKLEARN_MODELS:
    _MODEL_FNS[_sk] = _make_sklearn_fit_predict(_sk)
    _MODEL_FIT_FNS[_sk] = _make_sklearn_fit(_sk)


def method_tag(model_name: str, target_frac: float, src: str, tgt: str,
               scale: str = "3star", no_source: bool = False, mode: str = "pool") -> str:
    PCT = int(round(target_frac * 100))
    seq = "_seq" if mode == "sequential" else ""
    if no_source:
        # target-only baseline. Source-independent (like the LLM), so it is
        # tagged and logged once per TARGET with no `{src}2` segment, giving a
        # same-representation control for the full-target CV delta (QWK with source minus
        # QWK without) instead of confounding it against the within-state native-feature
        # oracle.
        return f"xfer_{model_name}{scale_infix(scale)}{seq}_nosrc_tgt{PCT}_{tgt}"
    return f"xfer_{model_name}{scale_infix(scale)}{seq}_tgt{PCT}_{src}2{tgt}"


def run_one_model(model_name, data, src_emb, tgt_emb, adapt_idx, test_idx,
                  target_frac, n_boot, seed, output_path, no_source=False, mode="pool"):
    """Fit one model on (source + adaptation) embeddings, score the target test
    rows. With ``no_source`` the source is omitted and the fit set is
    the target adaptation rows only (the same-representation target-only baseline,
    ) — requires a non-empty adaptation draw (target_frac > 0).

    ``mode='sequential'`` (Phase E, TabNet only): fit on source then warm-start
    continue on the target adaptation rows (never pooled), tag gets ``_seq``."""
    method = method_tag(model_name, target_frac, data.src_name, data.tgt_name,
                        scale=data.scale, no_source=no_source, mode=mode)
    print(f"\n  ---- {model_name.upper()}  [{method}] ----")

    if mode == "sequential":
        if no_source:
            raise ValueError("sequential is a source->target method; incompatible with --no-source.")
        if model_name != "tabnet":
            raise ValueError(f"sequential tabular transfer is TabNet-only; '{model_name}' is "
                             "in-context (no continued training) -- use the pooled variant.")
        if len(adapt_idx) == 0:
            raise ValueError("sequential needs target adaptation rows (--target-frac > 0).")
        # Scaler fit on all training features (source + adapt), applied per phase.
        scaler = StandardScaler().fit(np.vstack([src_emb, tgt_emb[adapt_idx]]))
        Xs = scaler.transform(src_emb).astype(np.float32)
        Xt = scaler.transform(tgt_emb[adapt_idx]).astype(np.float32)
        X_test_s = scaler.transform(tgt_emb[test_idx]).astype(np.float32)
        y_src = data.y_src.astype(int)
        y_tgt = data.y_tgt[adapt_idx].astype(int)
        print(f"    [seq] source fit={Xs.shape} -> target fit={Xt.shape}  test={X_test_s.shape}")
        t0 = time.time()
        pred_idx, proba = _fit_predict_tabnet_sequential(
            Xs, y_src, Xt, y_tgt, X_test_s, data.n_classes, seed=seed)
        y_pred = np.array([data.idx_to_label[int(i)] for i in pred_idx])
        y_true = np.array([data.idx_to_label[int(i)] for i in data.y_tgt[test_idx]])
        point = compute_metrics(y_true, y_pred, y_proba=proba, labels=data.labels)
        std = bootstrap_target_std(y_true, y_pred, proba, labels=data.labels,
                                   n_boot=n_boot, seed=seed)
        print(f"    TARGET TEST ({data.tgt_name}, n={len(y_true)}): "
              f"qwk={point['qwk']:.3f}  mae={point['mae']:.3f}  ({time.time() - t0:.0f}s)")
        log_transfer_result(method, point, std, output_path=output_path, notes=method,
                            source=data.src_name, target=data.tgt_name, classes=data.labels)
        return method

    if no_source:
        if len(adapt_idx) == 0:
            raise ValueError(
                "--no-source with target_frac=0 leaves an empty fit set; the "
                "target-only baseline needs target labels (use CV mode for full-target CV, "
                "or --target-frac > 0).")
        X_fit = tgt_emb[adapt_idx]
        y_fit = data.y_tgt[adapt_idx].astype(int)
    else:
        # Fit set = all source rows + target adaptation rows. Test = target test rows.
        X_fit = np.vstack([src_emb, tgt_emb[adapt_idx]]) if len(adapt_idx) \
            else src_emb.copy()
        y_fit = np.concatenate([data.y_src, data.y_tgt[adapt_idx]]).astype(int)
    X_test = tgt_emb[test_idx]

    # StandardScaler fit on the FIT set only (never target-test). Helps TabNet,
    # harmless for TabPFN.
    scaler = StandardScaler().fit(X_fit)
    X_fit_s = scaler.transform(X_fit).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    src_n = 0 if no_source or src_emb is None else len(src_emb)
    print(f"    fit={X_fit_s.shape} (src={src_n} + adapt={len(adapt_idx)})  "
          f"test={X_test_s.shape}")

    t0 = time.time()
    pred_idx, proba = _MODEL_FNS[model_name](
        X_fit_s, y_fit, X_test_s, data.n_classes, seed=seed)

    y_pred = np.array([data.idx_to_label[int(i)] for i in pred_idx])
    y_true = np.array([data.idx_to_label[int(i)] for i in data.y_tgt[test_idx]])
    point = compute_metrics(y_true, y_pred, y_proba=proba, labels=data.labels)
    std = bootstrap_target_std(y_true, y_pred, proba, labels=data.labels,
                               n_boot=n_boot, seed=seed)
    print(
        f"    TARGET TEST ({data.tgt_name}, n={len(y_true)}, +/-=bootstrap over {n_boot}):\n"
        f"      acc={point['accuracy']:.3f}+/-{std['accuracy']:.3f}  "
        f"bal_acc={point['balanced_acc']:.3f}+/-{std['balanced_acc']:.3f}  "
        f"qwk={point['qwk']:.3f}+/-{std['qwk']:.3f}  "
        f"mae={point['mae']:.3f}+/-{std['mae']:.3f}  "
        f"ll={point['log_loss']:.3f}  ({time.time() - t0:.0f}s)")

    log_transfer_result(method, point, std, output_path=output_path, notes=method,
                        source=None if no_source else data.src_name,
                        target=data.tgt_name, classes=data.labels)
    return method


def run_one_model_cv(model_name, data, src_emb, tgt_emb, folds,
                     n_boot, seed, output_path, no_source=False, mode="pool"):
    """Full-target CV: for each fold fit on (all source + the other K-1
    target folds) embeddings and predict the held-out fold, then concatenate the
    out-of-fold predictions and score the whole target once. tgt100 tag.

    Embeddings are fold-independent, so this only slices the cached arrays — no
    re-encoding. The StandardScaler is fit per fold on that fold's fit set only
    (never on the held-out rows), matching the single-split path.

    ``mode='sequential'`` (TabNet only): per fold, fit on source then warm-start
    continue on that fold's target-train rows (never pooled); ``_seq`` tag."""
    method = method_tag(model_name, 1.0, data.src_name, data.tgt_name,
                        scale=data.scale, no_source=no_source, mode=mode)  # -> tgt100
    seq = (mode == "sequential")
    if seq and (no_source or model_name != "tabnet"):
        raise ValueError("sequential CV is TabNet source->target only "
                         "(not --no-source, not tabpfn).")
    scope = "target-only" if no_source else "full-target"
    print(f"\n  ---- {model_name.upper()}  [{method}]  {scope} {len(folds)}-fold CV ----")

    t0 = time.time()
    yt_parts, yp_parts, pr_parts = [], [], []
    for fi, (tr, te) in enumerate(folds, 1):
        # no_source -> fit on the other K-1 TARGET folds only (a same-
        # representation within-target CV baseline); otherwise fold the full
        # source into every fold's fit set.
        if seq:
            # Sequential: scaler on source+fold-train; fit source, warm-start fold-train.
            scaler = StandardScaler().fit(np.vstack([src_emb, tgt_emb[tr]]))
            Xs = scaler.transform(src_emb).astype(np.float32)
            Xt = scaler.transform(tgt_emb[tr]).astype(np.float32)
            X_test_s = scaler.transform(tgt_emb[te]).astype(np.float32)
            print(f"    fold {fi}/{len(folds)} [seq]: src={Xs.shape} -> tgt_train={Xt.shape}  "
                  f"test={X_test_s.shape}")
            pred_idx, proba = _fit_predict_tabnet_sequential(
                Xs, data.y_src.astype(int), Xt, data.y_tgt[tr].astype(int),
                X_test_s, data.n_classes, seed=seed)
            yp_parts.append(np.array([data.idx_to_label[int(i)] for i in pred_idx]))
            yt_parts.append(np.array([data.idx_to_label[int(i)] for i in data.y_tgt[te]]))
            pr_parts.append(proba)
            continue
        if no_source:
            X_fit = tgt_emb[tr]
            y_fit = data.y_tgt[tr].astype(int)
            src_n = 0
        else:
            X_fit = np.vstack([src_emb, tgt_emb[tr]])
            y_fit = np.concatenate([data.y_src, data.y_tgt[tr]]).astype(int)
            src_n = len(src_emb)
        X_test = tgt_emb[te]
        scaler = StandardScaler().fit(X_fit)
        X_fit_s = scaler.transform(X_fit).astype(np.float32)
        X_test_s = scaler.transform(X_test).astype(np.float32)
        print(f"    fold {fi}/{len(folds)}: fit={X_fit_s.shape} "
              f"(src={src_n} + tgt_train={len(tr)})  test={X_test_s.shape}")
        pred_idx, proba = _MODEL_FNS[model_name](
            X_fit_s, y_fit, X_test_s, data.n_classes, seed=seed)
        yp_parts.append(np.array([data.idx_to_label[int(i)] for i in pred_idx]))
        yt_parts.append(np.array([data.idx_to_label[int(i)] for i in data.y_tgt[te]]))
        pr_parts.append(proba)

    y_pred = np.concatenate(yp_parts)
    y_true = np.concatenate(yt_parts)
    proba = np.concatenate(pr_parts, axis=0)
    point = compute_metrics(y_true, y_pred, y_proba=proba, labels=data.labels)
    std = bootstrap_target_std(y_true, y_pred, proba, labels=data.labels,
                               n_boot=n_boot, seed=seed)
    print(
        f"    TARGET OOF ({data.tgt_name}, n={len(y_true)}, +/-=bootstrap over {n_boot}):\n"
        f"      acc={point['accuracy']:.3f}+/-{std['accuracy']:.3f}  "
        f"bal_acc={point['balanced_acc']:.3f}+/-{std['balanced_acc']:.3f}  "
        f"qwk={point['qwk']:.3f}+/-{std['qwk']:.3f}  "
        f"mae={point['mae']:.3f}+/-{std['mae']:.3f}  "
        f"ll={point['log_loss']:.3f}  ({time.time() - t0:.0f}s)")

    log_transfer_result(method, point, std, output_path=output_path, notes=method,
                        source=None if no_source else data.src_name,
                        target=data.tgt_name, classes=data.labels)
    return method


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiments 6/7/8: tabular (TabNN/TabPFN) cross-state transfer "
                    "over sentence-transformer features.")
    add_transfer_args(parser)
    parser.add_argument("--models", nargs="+", default=["tabnet", "tabpfn"],
                        choices=["tabnet", "tabpfn", "tabpfn_balanced",
                                 "dummy", "lr", "rf", "xgb"],
                        help="Which tabular models to run over the shared embeddings. "
                             "'tabnet' is balanced (weights=1); 'tabpfn' is unweighted; "
                             "'tabpfn_balanced' class-balances the fit set; "
                             "'dummy'/'lr'/'rf'/'xgb' are the baselines_ml estimators "
                             "(identical configs) applied to the embeddings, so they "
                             "become cross-state methods too (default: tabnet + tabpfn).")
    parser.add_argument("--embed-batch-size", type=int, default=64,
                        help="Sentence-transformer encode batch size (default 64).")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Embedding cache dir (default: <output_dir>/embeddings).")
    parser.add_argument("--no-source", action="store_true",
                        help="target-only baseline: omit the source entirely and "
                             "fit on target rows only (CV: the other K-1 folds; single-"
                             "split: the adaptation draw). Source-independent -> tagged/"
                             "logged once per target as xfer_<model>_nosrc_tgt<PCT>_<TGT>. "
                             "Gives a same-representation control for the full-target CV delta.")
    parser.add_argument("--transfer-mode", choices=("pool", "sequential"), default="pool",
                        help="'pool' (default): fit once on source+target-adapt. "
                             "'sequential' (Phase E, TabNet ONLY): fit on source then "
                             "warm-start continue on the target adapt rows; '_seq' tag. "
                             "tabpfn/tabpfn_balanced have no sequential (in-context).")
    args = parser.parse_args()
    if args.transfer_mode == "sequential":
        bad = [m for m in args.models if m != "tabnet"]
        if bad:
            parser.error(f"--transfer-mode sequential is TabNet-only; drop {bad} "
                         "(TabPFN is in-context, no continued training).")
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
    cache_dir = args.cache_dir or (run_artifact_dir(args.output) / "embeddings")

    cv = args.target_cv_folds > 0
    PCT = 100 if cv else int(round(args.target_frac * 100))
    exp = "EXP9" if cv else "EXP6/7/8"
    print(f"\n{'=' * 70}")
    print(f"{exp} TABULAR: {data.src_name} -> {data.tgt_name}  tgt{PCT}  "
          f"models={args.models}")
    print(f"  featurizer={EMBED_MODEL_NAME}  classes={data.labels}  device={_device()}")
    if cv:
        print(f"  full-target {args.target_cv_folds}-fold CV (source folded into "
              f"every fold's training set)")
    print(f"{'=' * 70}")

    # Choose the target protocol (fold-independent of the embeddings below).
    if cv:
        folds = target_cv_folds(data.y_tgt, args.target_cv_folds, args.seed)
        print(f"  full-target CV: {len(folds)} folds; "
              f"held-out sizes={[len(te) for _, te in folds]}")
    else:
        adapt_idx, test_idx = split_target_fewshot(
            data.y_tgt, args.target_frac, args.target_test_frac, args.seed)
        print(f"  target split: adapt={len(adapt_idx)}  test={len(test_idx)}")

    # Embed both states once (cached). One model instance shared across both.
    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL_NAME, device=_device())
    except Exception as exc:  # noqa: BLE001 — fall back to per-call load on miss
        print(f"  [embed] preload failed ({exc}); will load on cache miss")

    # no_source doesn't touch the source at all, so skip encoding it.
    if args.no_source:
        src_emb = None
        print("  [no-source] skipping source embed (target-only baseline)")
    else:
        src_emb = embed_state(data.src_texts, data.src_name, data.text_mode,
                              data.compliance_mode, cache_dir,
                              batch_size=args.embed_batch_size, model=embedder)
    tgt_emb = embed_state(data.tgt_texts, data.tgt_name, data.text_mode,
                          data.compliance_mode, cache_dir,
                          batch_size=args.embed_batch_size, model=embedder)
    del embedder
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Per-model: a crash in one model logs a FAILED row but does not lose the others.
    tag_frac = 1.0 if cv else args.target_frac  # -> tgt100 in CV mode
    for model_name in args.models:
        try:
            if cv:
                run_one_model_cv(model_name, data, src_emb, tgt_emb, folds,
                                 args.n_bootstrap, args.seed, args.output,
                                 no_source=args.no_source, mode=args.transfer_mode)
            else:
                run_one_model(model_name, data, src_emb, tgt_emb, adapt_idx, test_idx,
                              args.target_frac, args.n_bootstrap, args.seed, args.output,
                              no_source=args.no_source, mode=args.transfer_mode)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            log_failed_transfer(
                method_tag(model_name, tag_frac, data.src_name, data.tgt_name,
                           scale=data.scale, no_source=args.no_source, mode=args.transfer_mode),
                output_path=args.output,
                source=None if args.no_source else data.src_name,
                target=data.tgt_name,
                classes=data.labels, error=f"{type(exc).__name__}: {exc}")

    print(f"\nDone. See {args.output} for transfer scores.")
    print(f"Embedding cache at {cache_dir} (safe to delete; will re-encode).")



def _failed_methods(a) -> list[str]:
    models = getattr(a, "models", None) or ["tabnet", "tabpfn"]
    frac = 1.0 if getattr(a, "target_cv_folds", 0) > 0 else getattr(a, "target_frac", 0.0)
    scale = getattr(a, "rating_scale", "3star")
    no_source = getattr(a, "no_source", False)
    mode = getattr(a, "transfer_mode", "pool")
    # In LOSO mode the drivers pass --pool-sources but never --src-name, so the
    # success-path tag reads ..._LOSO2<TGT>. The FAILED tag must match, or a
    # crashed cell is filed under a method name nothing looks for.
    src = a.pool_name if getattr(a, "pool_sources", None) else a.src_name
    return [method_tag(m, frac, src, a.tgt_name, scale=scale,
                       no_source=no_source, mode=mode) for m in models]


if __name__ == "__main__":
    # a crash BEFORE the per-model loop (e.g. load/embed failure) would
    # otherwise leave no row for any model. Record a FAILED row per requested
    # model and re-raise so the exit code is honest.
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        _p = argparse.ArgumentParser(add_help=False)
        add_transfer_args(_p)
        _p.add_argument("--models", nargs="+", default=["tabnet", "tabpfn"])
        _p.add_argument("--embed-batch-size", type=int, default=64)
        _p.add_argument("--cache-dir", type=Path, default=None)
        _p.add_argument("--no-source", action="store_true")
        _p.add_argument("--transfer-mode", choices=("pool", "sequential"), default="pool")
        _a, _ = _p.parse_known_args()
        _pool_src = _a.pool_name if getattr(_a, "pool_sources", None) else _a.src_name
        _src = None if getattr(_a, "no_source", False) else _pool_src
        for _m in _failed_methods(_a):
            log_failed_transfer(_m, output_path=_a.output, source=_src,
                                target=_a.tgt_name,
                                error=f"{type(exc).__name__}: {exc}")
        raise