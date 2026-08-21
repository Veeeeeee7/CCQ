"""Tabular deep learning for within-state prediction: TabNet and TabPFN.

Both run 5-fold stratified CV over the shared fold indices. The cross-state
counterparts are in `transfer_tabular`.

TabNet is stochastic, so fold-to-fold variance on the smaller states is higher
than for the tree models. TabPFN performs in-context learning, has no native
class weighting, and is capped at its pretraining context size.

    python tabular_dl.py --input data/wi_records_cleaned_full.csv --output results.csv
    python tabular_dl.py --models tabnet
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from utils import (
    SEED,
    TARGET_COL,
    add_io_args,
    align_proba,
    build_tabnet_arrays,
    compute_metrics,
    configure_verbosity,
    get_feature_columns,
    get_folds,
    load_data,
    maybe_remap,
    log_results,
    nan_metrics,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fraction of each training fold held out for TabNet early stopping.
VAL_FRAC = 0.15


# -----------------------------------------------------------------------------
# TabNet
# -----------------------------------------------------------------------------
def run_tabnet_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[dict],
    weighted: bool = True,
    max_epochs: int = 200,
    patience: int = 40,
) -> list[dict]:
    """Run TabNet across the 5 folds.

    Early stopping monitors balanced accuracy rather than logloss.
    """
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.metrics import Metric
    from sklearn.metrics import balanced_accuracy_score

    # pytorch-tabnet doesn't ship balanced_accuracy; define it for early stopping.
    class BalancedAccuracy(Metric):
        def __init__(self):
            self._name = "balanced_accuracy"
            self._maximize = True

        def __call__(self, y_true, y_score):
            y_pred = np.argmax(y_score, axis=1)
            return balanced_accuracy_score(y_true, y_pred)

    n_classes = df[TARGET_COL].nunique()
    classes_sorted = sorted(df[TARGET_COL].unique())
    label_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    label = "tabnet" if weighted else "tabnet_unweighted"
    print(f"\n{'=' * 60}")
    print(f"Training {label.upper()} ({n_classes} classes, device={DEVICE})")
    print(f"  max_epochs={max_epochs}, patience={patience}, weighted={weighted}")
    print(f"  early-stop metric: balanced_accuracy")
    print(f"{'=' * 60}")

    fold_metrics: list[dict] = []
    t0 = time.time()

    for fold in folds:
        fold_idx = fold["fold"]
        train_idx, val_idx = fold["train_idx"], fold["val_idx"]

        # Model selection watches a VAL_FRAC slice of the TRAINING fold; the
        # scored fold is untouched until predict.
        from transfer_common import stratified_source_split
        train_idx = np.asarray(train_idx)
        y_train_full = df.iloc[train_idx][TARGET_COL].map(label_to_idx).to_numpy()
        inner_tr, inner_va = stratified_source_split(
            y_train_full, VAL_FRAC, SEED)
        fit_idx, es_idx = train_idx[inner_tr], train_idx[inner_va]

        X_train_df = df.iloc[fit_idx]
        X_es_df = df.iloc[es_idx]
        X_val_df = df.iloc[val_idx]
        y_train = df.iloc[fit_idx][TARGET_COL].map(label_to_idx).to_numpy()
        y_es = df.iloc[es_idx][TARGET_COL].map(label_to_idx).to_numpy()
        y_val = df.iloc[val_idx][TARGET_COL].map(label_to_idx).to_numpy()

        # Called twice so both the early-stopping slice and the scored fold are
        # transformed by a preprocessor fit on the fit slice alone. The pipeline
        # is deterministic, so the two fits agree.
        X_train, X_es, cat_idxs, cat_dims, _ = build_tabnet_arrays(
            X_train_df, X_es_df, numerical_cols=feature_cols, categorical_cols=[],
        )
        _, X_val, _, _, _ = build_tabnet_arrays(
            X_train_df, X_val_df, numerical_cols=feature_cols, categorical_cols=[],
        )

        # weights=1 → balanced (inverse-frequency); weights=0 → unweighted.
        weight_arg = 1 if weighted else 0

        model = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=3,
            gamma=1.5,
            lambda_sparse=1e-4,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": 2e-2},
            scheduler_params={"step_size": 20, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
            device_name=DEVICE,
            seed=SEED + fold_idx,  # vary per fold to expose true stability
            verbose=0,
        )

        # TabNet refuses a validation fold containing a class absent from
        # training. Record NaN for the fold rather than failing the run.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # pytorch-tabnet defaults to drop_last=True, so a fold smaller
                # than one batch trains zero batches. Size the batch to the fold.
                from transfer_tabular import _tabnet_batch_kwargs
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_es, y_es)],
                    eval_metric=[BalancedAccuracy],
                    max_epochs=max_epochs,
                    patience=patience,
                    **_tabnet_batch_kwargs(len(X_train)),
                    weights=weight_arg,
                )
            y_pred_idx = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            # Widen proba if this fold trained on < n_classes classes.
            y_proba = align_proba(y_proba, model.classes_, n_classes)
        except ValueError as e:
            print(f"  Fold {fold_idx}: SKIPPED (TabNet could not fit — {e})")
            fold_metrics.append(nan_metrics())
            continue

        y_pred = np.array([idx_to_label[i] for i in y_pred_idx])
        y_true = np.array([idx_to_label[i] for i in y_val])

        m = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=classes_sorted)
        fold_metrics.append(m)
        print(
            f"  Fold {fold_idx}: "
            f"acc={m['accuracy']:.3f}  bal_acc={m['balanced_acc']:.3f}  "
            f"macroF1={m['macro_f1']:.3f}  qwk={m['qwk']:.3f}  "
            f"mae={m['mae']:.3f}  ll={m['log_loss']:.3f}  "
            f"(epochs trained={model.best_epoch})"
        )

    elapsed = time.time() - t0
    agg = {k: np.nanmean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    std = {k: np.nanstd([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    print(
        f"  MEAN:   "
        f"acc={agg['accuracy']:.3f}±{std['accuracy']:.3f}  "
        f"bal_acc={agg['balanced_acc']:.3f}±{std['balanced_acc']:.3f}  "
        f"macroF1={agg['macro_f1']:.3f}±{std['macro_f1']:.3f}  "
        f"qwk={agg['qwk']:.3f}±{std['qwk']:.3f}  "
        f"mae={agg['mae']:.3f}±{std['mae']:.3f}  "
        f"ll={agg['log_loss']:.3f}±{std['log_loss']:.3f}"
    )
    print(f"  Elapsed: {elapsed:.1f}s")
    return fold_metrics


# -----------------------------------------------------------------------------
# TabPFN
# -----------------------------------------------------------------------------
def run_tabpfn_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    folds: list[dict],
    n_estimators: int = 8,
    balanced_subsample: bool = False,
) -> list[dict]:
    """Run TabPFN across the 5 folds.

    `.fit()` stores the training set as context; `.predict()` runs the prior over
    it. `balanced_subsample` downsamples each class to the smallest class size,
    standing in for the class weighting TabPFN does not support.
    """
    from tabpfn import TabPFNClassifier

    n_classes = df[TARGET_COL].nunique()
    classes_sorted = sorted(df[TARGET_COL].unique())
    label_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    label = "tabpfn_balanced" if balanced_subsample else "tabpfn"
    print(f"\n{'=' * 60}")
    print(f"Training {label.upper()} ({n_classes} classes, device={DEVICE})")
    print(f"  n_estimators={n_estimators}, balanced_subsample={balanced_subsample}")
    print(f"{'=' * 60}")

    rng = np.random.RandomState(SEED)

    fold_metrics: list[dict] = []
    t0 = time.time()

    for fold in folds:
        fold_idx = fold["fold"]
        train_idx, val_idx = fold["train_idx"], fold["val_idx"]
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Class-balanced subsampling, done before feature extraction.
        if balanced_subsample:
            y_train_full = train_df[TARGET_COL].to_numpy()
            min_count = pd.Series(y_train_full).value_counts().min()
            balanced_indices = []
            for cls in classes_sorted:
                cls_indices = np.where(y_train_full == cls)[0]
                if len(cls_indices) == 0:
                    # Rare class absent from this train fold — nothing to sample.
                    continue
                chosen = rng.choice(cls_indices, size=min_count, replace=False)
                balanced_indices.extend(chosen)
            train_df = train_df.iloc[np.array(balanced_indices)]
            print(
                f"  Fold {fold_idx}: subsampled train from "
                f"{len(y_train_full)} → {len(train_df)} rows ({min_count} per class)"
            )

        # TabPFN raises beyond its pretraining context limit unless explicitly
        # overridden. The transfer path enforces the same cap.
        from transfer_tabular import TABPFN_SAMPLE_CAP, _stratification_labels
        if len(train_df) > TABPFN_SAMPLE_CAP:
            from sklearn.model_selection import train_test_split
            print(f"  Fold {fold_idx}: fit set {len(train_df)} > cap "
                  f"{TABPFN_SAMPLE_CAP}; stratified-subsampling to "
                  f"{TABPFN_SAMPLE_CAP}")
            y_for_strat = train_df[TARGET_COL].to_numpy()
            keep, _ = train_test_split(
                np.arange(len(train_df)), train_size=TABPFN_SAMPLE_CAP,
                stratify=_stratification_labels(y_for_strat, 2),
                random_state=SEED)
            train_df = train_df.iloc[np.sort(keep)]

        y_train = train_df[TARGET_COL].map(label_to_idx).to_numpy()
        y_val = val_df[TARGET_COL].map(label_to_idx).to_numpy()

        # All features numeric; TabPFN handles NaN natively, so pass raw arrays.
        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        X_val = val_df[feature_cols].to_numpy(dtype=np.float32)

        model = TabPFNClassifier(
            n_estimators=n_estimators,
            device=DEVICE,
            random_state=SEED,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            y_pred_idx = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
        # Widen proba if this fold's train set was missing a rare class.
        y_proba = align_proba(y_proba, model.classes_, n_classes)

        y_pred = np.array([idx_to_label[i] for i in y_pred_idx])
        y_true = np.array([idx_to_label[i] for i in y_val])

        m = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=classes_sorted)
        fold_metrics.append(m)
        print(
            f"  Fold {fold_idx}: "
            f"acc={m['accuracy']:.3f}  bal_acc={m['balanced_acc']:.3f}  "
            f"macroF1={m['macro_f1']:.3f}  qwk={m['qwk']:.3f}  "
            f"mae={m['mae']:.3f}  ll={m['log_loss']:.3f}"
        )

    elapsed = time.time() - t0
    agg = {k: np.nanmean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    std = {k: np.nanstd([fm[k] for fm in fold_metrics]) for k in fold_metrics[0]}
    print(
        f"  MEAN:   "
        f"acc={agg['accuracy']:.3f}±{std['accuracy']:.3f}  "
        f"bal_acc={agg['balanced_acc']:.3f}±{std['balanced_acc']:.3f}  "
        f"macroF1={agg['macro_f1']:.3f}±{std['macro_f1']:.3f}  "
        f"qwk={agg['qwk']:.3f}±{std['qwk']:.3f}  "
        f"mae={agg['mae']:.3f}±{std['mae']:.3f}  "
        f"ll={agg['log_loss']:.3f}±{std['log_loss']:.3f}"
    )
    print(f"  Elapsed: {elapsed:.1f}s")
    return fold_metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
MODEL_RUNNERS: dict[str, Callable] = {
    "tabnet": lambda df, c, f: run_tabnet_cv(df, c, f, weighted=True),
    "tabnet_unweighted": lambda df, c, f: run_tabnet_cv(df, c, f, weighted=False),
    "tabpfn": lambda df, c, f: run_tabpfn_cv(df, c, f, balanced_subsample=False),
    "tabpfn_balanced": lambda df, c, f: run_tabpfn_cv(df, c, f, balanced_subsample=True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_io_args(parser)
    parser.add_argument(
        "--models",
        nargs="+",
        # Weighted/balanced only by default: drop the
        # unweighted TabNet twin and the un-balanced plain TabPFN. `tabnet` is
        # the class-weighted variant; `tabpfn_balanced` is the balanced-subsample
        # variant. Pass the others explicitly via --models if ever needed.
        default=["tabnet", "tabpfn_balanced"],
        choices=list(MODEL_RUNNERS.keys()),
        help="Which models to run (default: weighted/balanced variants)",
    )
    parser.add_argument(
        "--features", choices=("numeric", "text"), default="numeric",
        help="Feature source. 'numeric' (default): use the prepared numeric "
             "columns of the input CSV directly (the preprocessed `_full` data). "
             "'text': serialize each row and sentence-embed it (the raw `_raw` "
             "data path) — reuses transfer_tabular's chunk-mean-pooled embedder "
             "so raw free-text rows land in a fixed-width numeric space.",
    )
    parser.add_argument(
        "--compliance", choices=("verbose", "summary", "abnormal_only"),
        default="verbose",
        help="Compliance-text verbosity for --features text serialization "
             "(matches llm.py / the transfer serialization; default verbose).",
    )
    parser.add_argument(
        "--method-suffix", default="",
        help="Optional suffix appended to every method tag, e.g. "
             "--method-suffix raw -> 'tabnet_raw'. Disambiguates the "
             "preprocessed run (bare tag) from the raw run. Empty by default.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Where to cache --features text embeddings (default: an "
             "'embeddings' subdir next to the results file, shared with "
             "transfer_tabular's cache when the state/text/model keys match).",
    )
    args = parser.parse_args()
    configure_verbosity(args.verbose)

    df = load_data(args.input)
    df = maybe_remap(df, args.remap_state, allow_identity=args.allow_identity, scale=args.rating_scale)
    folds = get_folds(df, folds_path=args.folds)

    if args.features == "text":
        # Raw-data path: serialize every row to the shared `textualized_full`
        # view and sentence-embed it, so tabpfn/tabnet get a fixed-width numeric
        # matrix from the free-text raw CSV. The embedded frame keeps row order
        # (folds are positional, so they stay aligned) and carries only the
        # embedding columns + the target. Shared helper (also used by
        # baselines_ml --features text) so the embedding/cache is identical.
        from transfer_tabular import embed_dataframe_cv               # noqa: E402
        state = args.remap_state or "within"
        df, feature_cols = embed_dataframe_cv(
            df, args.compliance, state, output_path=args.output,
            cache_dir=args.cache_dir)
    else:
        feature_cols = get_feature_columns(df)
        print(f"Using {len(feature_cols)} numeric feature columns (no categoricals).")

    # TabPFN v2 has a soft feature-count limit (~500). Warn so a degraded run
    # isn't a mystery.
    if len(feature_cols) > 500 and any("tabpfn" in m for m in args.models):
        print(
            f"  WARNING: {len(feature_cols)} features exceeds TabPFN's ~500 "
            "limit. TabPFN may error or silently degrade. Consider reducing "
            "features (drop rare columns or apply PCA/SVD)."
        )

    for model_name in args.models:
        runner = MODEL_RUNNERS[model_name]
        fold_results = runner(df, feature_cols, folds)
        tag = f"{model_name}_{args.method_suffix}" if args.method_suffix else model_name
        log_results(tag, fold_results, output_path=args.output, notes=tag)

    print(f"\nDone. See {args.output} for aggregated scores.")


if __name__ == "__main__":
    main()