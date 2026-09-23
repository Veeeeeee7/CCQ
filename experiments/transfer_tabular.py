"""Tabular models for cross-state transfer over a shared sentence-embedding space.

States publish different feature schemas, so each row is serialized to text and
embedded with MiniLM; the models train and predict in that shared space.

    python transfer_tabular.py --pool-sources NC=data/nc_records_cleaned_raw.csv ... \\
        --target data/wi_records_cleaned_raw.csv --tgt-name WI --rating-scale 5star \\
        --target-frac 0.2 --models dummy lr rf xgb tabpfn_balanced
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
    _resolve_transfer_output,
    add_transfer_args,
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
)
from utils import SEED, _stratification_labels, compute_metrics, run_artifact_dir  # noqa: E402

# Fixed on purpose: every embedding-based method must consume the same vectors.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABPFN_SAMPLE_CAP = 10_000   # TabPFN v2's pretraining context limit


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# -----------------------------------------------------------------------------
# Featurizer
# -----------------------------------------------------------------------------
def _safe_model_tag(name: str) -> str:
    return name.replace("/", "_").replace("-", "_")


def _texts_digest(texts) -> str:
    """Content hash of the texts, so a serialization change is a cache miss."""
    import hashlib
    h = hashlib.md5()
    for t in texts:
        h.update(str(t).encode("utf-8", "replace"))
        h.update(b"\x1e")
    return h.hexdigest()[:12]


def _cache_path(cache_dir: Path, state: str, compliance: str,
                model_name: str, digest: str) -> Path:
    return cache_dir / f"{state}_{compliance}_{_safe_model_tag(model_name)}_{digest}.npy"


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    """Write via a temp file and `os.replace`, so a killed job never leaves a
    truncated cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npy")
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _embed_chunked(texts, model, batch_size: int = 64) -> np.ndarray:
    """Embed each string as the normalized mean of its chunk embeddings, so rows
    longer than the encoder window are not truncated."""
    tok = model.tokenizer
    max_len = int(getattr(model, "max_seq_length", 256) or 256)
    chunk_tokens = max(8, max_len - 2)

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

    chunk_emb = model.encode(
        chunk_strings, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
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


def embed_state(texts, state: str, compliance: str, cache_dir: Path,
                model_name: str = EMBED_MODEL_NAME, batch_size: int = 64,
                device: "str | None" = None, model=None) -> np.ndarray:
    """(n_rows, dim) embeddings of one state's texts, cached on disk by content hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = list(texts)
    path = _cache_path(cache_dir, state, compliance, model_name, _texts_digest(texts))

    if path.exists():
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
    _atomic_save(path, emb)
    print(f"    [embed] cached -> {path}")
    return emb


def embed_dataframe_cv(df, compliance: str, state: str, output_path=None,
                       cache_dir=None, model_name: str = EMBED_MODEL_NAME):
    """Embedding features for the within-state `--features text` runs.

    Returns ``(emb_df, feature_cols)`` with ``emb_<i>`` columns plus the target, in
    the original row order so the positional folds still apply.
    """
    import pandas as pd
    from text_serialization import serialize_dataframe
    from utils import TARGET_COL, run_artifact_dir

    if cache_dir is None:
        cache_dir = run_artifact_dir(output_path) / "embeddings"
    texts = serialize_dataframe(df, compliance).tolist()
    print(f"Serialized {len(texts)} rows ({compliance}); embedding with "
          f"{model_name} (cache: {cache_dir})...")
    emb = embed_state(texts, state, compliance, cache_dir, model_name)
    feature_cols = [f"emb_{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=feature_cols, index=df.index)
    emb_df[TARGET_COL] = df[TARGET_COL].to_numpy()
    print(f"Using {len(feature_cols)} sentence-embedding features (raw text).")
    return emb_df, feature_cols


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
def _align_proba(proba: np.ndarray, model_classes, n_classes: int) -> np.ndarray:
    """Scatter predict_proba columns into the full 0..n_classes-1 label space."""
    aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for col, cls in enumerate(model_classes):
        aligned[:, int(cls)] = proba[:, col]
    row_sums = aligned.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return aligned / row_sums


def _predict_fitted(model, X_test, n_classes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = model.predict_proba(X_test)
    proba = _align_proba(np.asarray(proba), model.classes_, n_classes)
    return proba.argmax(axis=1), proba


def _fit_predict_tabpfn_balanced(X_fit, y_fit, X_test, n_classes, seed=SEED):
    """TabPFN v2 with every class downsampled to the smallest class count (TabPFN
    has no class weighting), then capped at its context limit."""
    from tabpfn import TabPFNClassifier

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
        keep, _ = train_test_split(
            np.arange(len(X_fit)), train_size=TABPFN_SAMPLE_CAP,
            stratify=_stratification_labels(y_fit, 2), random_state=seed)
        X_fit, y_fit = X_fit[keep], y_fit[keep]

    model = TabPFNClassifier(n_estimators=8, device=_device(), random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_fit, y_fit)
    return _predict_fitted(model, X_test, n_classes)


def _tabnet_batch_kwargs(n_train: int, batch_size: int = 256,
                         virtual_batch_size: int = 64) -> dict:
    """Batch sizes capped at `n_train`: pytorch-tabnet drops the last partial
    batch, so a set smaller than one batch would otherwise train on nothing."""
    if n_train >= batch_size:
        return {"batch_size": batch_size, "virtual_batch_size": virtual_batch_size}
    bs = max(2, int(n_train))
    return {"batch_size": bs, "virtual_batch_size": min(virtual_batch_size, bs)}


def _make_tabnet(seed):
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    return TabNetClassifier(
        n_d=16, n_a=16, n_steps=3, gamma=1.5, lambda_sparse=1e-4,
        cat_idxs=[], cat_dims=[], cat_emb_dim=4,
        optimizer_fn=torch.optim.Adam, optimizer_params={"lr": 2e-2},
        scheduler_params={"step_size": 20, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="entmax", device_name=_device(), seed=seed, verbose=0,
    )


def _balanced_accuracy_metric():
    from pytorch_tabnet.metrics import Metric
    from sklearn.metrics import balanced_accuracy_score

    class BalancedAccuracy(Metric):
        def __init__(self):
            self._name = "balanced_accuracy"
            self._maximize = True

        def __call__(self, y_true, y_score):
            return balanced_accuracy_score(y_true, np.argmax(y_score, axis=1))

    return BalancedAccuracy


def _es_split(X, y, n_classes, seed):
    """Train / early-stopping split; sets too small to split use all rows for both."""
    if len(X) >= max(20, 2 * n_classes):
        tr, es = stratified_source_split(y, 0.15, seed)
    else:
        tr = es = np.arange(len(X))
    return X[tr], y[tr], X[es], y[es]


def _fit_predict_tabnet(X_fit, y_fit, X_test, n_classes, seed=SEED):
    """TabNet with balanced sample weights, early-stopped on a slice of the fit set."""
    Xtr, ytr, Xes, yes = _es_split(X_fit, y_fit, n_classes, seed)
    model = _make_tabnet(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            Xtr, ytr, eval_set=[(Xes, yes)], eval_metric=[_balanced_accuracy_metric()],
            max_epochs=200, patience=40, **_tabnet_batch_kwargs(len(Xtr)),
            weights=1,  # inverse-frequency sample weights
        )
    return _predict_fitted(model, X_test, n_classes)


def _fit_predict_tabnet_sequential(X_src, y_src, X_tgt, y_tgt, X_test, n_classes, seed=SEED):
    """TabNet fit on the source, then warm-started on the target adaptation rows."""
    model = _make_tabnet(seed)
    fit_kw = dict(eval_metric=[_balanced_accuracy_metric()], max_epochs=200,
                  patience=40, weights=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xtr, ytr, Xes, yes = _es_split(X_src, y_src, n_classes, seed)
        model.fit(Xtr, ytr, eval_set=[(Xes, yes)],
                  **_tabnet_batch_kwargs(len(Xtr)), **fit_kw)
        Xtr, ytr, Xes, yes = _es_split(X_tgt, y_tgt, n_classes, seed)
        model.fit(Xtr, ytr, eval_set=[(Xes, yes)], warm_start=True,
                  **_tabnet_batch_kwargs(len(Xtr)), **fit_kw)
        proba = model.predict_proba(X_test)
    proba = _align_proba(np.asarray(proba), model.classes_, n_classes)
    return proba.argmax(axis=1), proba


class _GlobalLabelClassifier:
    """Fit an sklearn-style estimator on densely relabeled targets (XGBoost needs
    0..m-1) while exposing the global class ids through ``classes_``."""

    def __init__(self, estimator, balanced_weights: bool = False):
        self._est = estimator
        self._balanced_weights = balanced_weights
        self.classes_ = None

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        present = np.unique(y)
        g2d = {int(g): i for i, g in enumerate(present)}
        y_dense = np.array([g2d[int(v)] for v in y])
        if self._balanced_weights:
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight(class_weight="balanced", y=y_dense)
            self._est.fit(X, y_dense, sample_weight=sw)
        else:
            self._est.fit(X, y_dense)
        self.classes_ = present[self._est.classes_.astype(int)]
        return self

    def predict_proba(self, X):
        return self._est.predict_proba(X)


def _make_sklearn_fit_predict(name: str):
    """The within-state baselines_ml estimator, with identical hyperparameters."""
    def _fit_predict(X_fit, y_fit, X_test, n_classes, seed=SEED):
        from baselines_ml import make_dummy, make_lr, make_rf, make_xgb
        factories = {"dummy": make_dummy, "lr": make_lr, "rf": make_rf, "xgb": make_xgb}
        est, pre_kwargs = factories[name](n_classes)
        balanced = bool(pre_kwargs.get("balanced_weights", False))
        model = _GlobalLabelClassifier(est, balanced_weights=balanced).fit(X_fit, y_fit)
        return _predict_fitted(model, X_test, n_classes)
    return _fit_predict


_MODEL_FNS = {
    "tabpfn_balanced": _fit_predict_tabpfn_balanced,
    "tabnet": _fit_predict_tabnet,
    **{name: _make_sklearn_fit_predict(name) for name in ("dummy", "lr", "rf", "xgb")},
}


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def method_tag(model_name: str, target_frac: float, src: str, tgt: str,
               scale: str = "3star", mode: str = "pool") -> str:
    PCT = int(round(target_frac * 100))
    seq = "_seq" if mode == "sequential" else ""
    return f"xfer_{model_name}{scale_infix(scale)}{seq}_tgt{PCT}_{src}2{tgt}"


def run_one_model(model_name, data, src_emb, tgt_emb, adapt_idx, test_idx,
                  target_frac, n_boot, seed, output_path, mode="pool"):
    """Fit one model on source (+ adaptation) embeddings and score the target test rows.

    ``mode='pool'`` fits once on the union; ``mode='sequential'`` (TabNet only)
    fits on the source and then continues on the adaptation rows.
    """
    method = method_tag(model_name, target_frac, data.src_name, data.tgt_name,
                        scale=data.scale, mode=mode)
    print(f"\n  ---- {model_name.upper()}  [{method}] ----")
    t0 = time.time()

    if mode == "sequential":
        if model_name != "tabnet":
            raise ValueError(f"sequential tabular transfer is TabNet-only, not '{model_name}'.")
        if len(adapt_idx) == 0:
            raise ValueError("sequential needs target adaptation rows (--target-frac > 0).")
        scaler = StandardScaler().fit(np.vstack([src_emb, tgt_emb[adapt_idx]]))
        Xs = scaler.transform(src_emb).astype(np.float32)
        Xt = scaler.transform(tgt_emb[adapt_idx]).astype(np.float32)
        X_test_s = scaler.transform(tgt_emb[test_idx]).astype(np.float32)
        print(f"    [seq] source fit={Xs.shape} -> target fit={Xt.shape}  test={X_test_s.shape}")
        pred_idx, proba = _fit_predict_tabnet_sequential(
            Xs, data.y_src.astype(int), Xt, data.y_tgt[adapt_idx].astype(int),
            X_test_s, data.n_classes, seed=seed)
    else:
        X_fit = np.vstack([src_emb, tgt_emb[adapt_idx]]) if len(adapt_idx) \
            else src_emb.copy()
        y_fit = np.concatenate([data.y_src, data.y_tgt[adapt_idx]]).astype(int)
        scaler = StandardScaler().fit(X_fit)
        X_fit_s = scaler.transform(X_fit).astype(np.float32)
        X_test_s = scaler.transform(tgt_emb[test_idx]).astype(np.float32)
        print(f"    fit={X_fit_s.shape} (src={len(src_emb)} + adapt={len(adapt_idx)})  "
              f"test={X_test_s.shape}")
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
                        source=data.src_name, target=data.tgt_name, classes=data.labels)
    return method


def run(args) -> None:
    if args.pool_sources:
        data = load_and_prepare_pooled(
            parse_pool_sources(args.pool_sources), args.tgt_name, args.target,
            compliance_mode=args.compliance, scale=args.rating_scale,
            pool_name=args.pool_name)
    else:
        data = load_and_prepare(
            args.source, args.target, args.src_name, args.tgt_name,
            compliance_mode=args.compliance, scale=args.rating_scale)
    cache_dir = args.cache_dir or (run_artifact_dir(args.output) / "embeddings")

    PCT = int(round(args.target_frac * 100))
    print(f"\n{'=' * 70}")
    print(f"TABULAR: {data.src_name} -> {data.tgt_name}  tgt{PCT}  models={args.models}")
    print(f"  featurizer={EMBED_MODEL_NAME}  classes={data.labels}  device={_device()}")
    print(f"{'=' * 70}")

    adapt_idx, test_idx = split_target_fewshot(
        data.y_tgt, args.target_frac, args.target_test_frac, args.seed)
    print(f"  target split: adapt={len(adapt_idx)}  test={len(test_idx)}")

    embedder = None
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL_NAME, device=_device())
    except Exception as exc:  # noqa: BLE001
        print(f"  [embed] preload failed ({exc}); will load on cache miss")
    src_emb = embed_state(data.src_texts, data.src_name, data.compliance_mode, cache_dir,
                          batch_size=args.embed_batch_size, model=embedder)
    tgt_emb = embed_state(data.tgt_texts, data.tgt_name, data.compliance_mode, cache_dir,
                          batch_size=args.embed_batch_size, model=embedder)
    del embedder
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # A crash in one model logs a FAILED row without losing the others.
    for model_name in args.models:
        try:
            run_one_model(model_name, data, src_emb, tgt_emb, adapt_idx, test_idx,
                          args.target_frac, args.n_bootstrap, args.seed, args.output,
                          mode=args.transfer_mode)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            log_failed_transfer(
                method_tag(model_name, args.target_frac, data.src_name, data.tgt_name,
                           scale=data.scale, mode=args.transfer_mode),
                output_path=args.output, source=data.src_name, target=data.tgt_name,
                classes=data.labels, error=f"{type(exc).__name__}: {exc}")

    print(f"\nDone. See {args.output} for transfer scores.")
    print(f"Embedding cache at {cache_dir} (safe to delete; will re-encode).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tabular cross-state transfer over sentence embeddings.")
    add_transfer_args(parser)
    parser.add_argument("--models", nargs="+", default=["tabnet", "tabpfn_balanced"],
                        choices=["tabnet", "tabpfn_balanced", "dummy", "lr", "rf", "xgb"])
    parser.add_argument("--embed-batch-size", type=int, default=64,
                        help="Sentence-transformer encode batch size (default 64).")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Embedding cache directory (default: under the run's artifacts).")
    parser.add_argument("--transfer-mode", choices=("pool", "sequential"), default="pool",
                        help="'pool': one fit on source + adaptation rows. 'sequential' "
                             "(TabNet only): fit on source, then continue on the target.")
    args = parser.parse_args()
    if args.transfer_mode == "sequential":
        bad = [m for m in args.models if m != "tabnet"]
        if bad:
            parser.error(f"--transfer-mode sequential is TabNet-only; drop {bad}.")
    configure_verbosity(args.verbose)
    _resolve_transfer_output(args, "transfer_tabular")

    try:
        run(args)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        # A failure before the per-model loop still leaves one FAILED row per model.
        traceback.print_exc()
        src = args.pool_name if args.pool_sources else args.src_name
        for m in args.models:
            log_failed_transfer(
                method_tag(m, args.target_frac, src, args.tgt_name,
                           scale=args.rating_scale, mode=args.transfer_mode),
                output_path=args.output, source=src, target=args.tgt_name,
                error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
