"""Classical tabular baselines: majority-class dummy, logistic regression, random
forest and gradient boosting.

Runs 5-fold stratified CV over the shared fold indices. All four are trained as
plain classifiers; ordinality appears in the metrics, not the loss. Preprocessing
is rebuilt inside each fold.

`REGRESSOR_REGISTRY` holds regression twins used by `shap_xgb` and
`cross_scale`. `_assert_regressor_parity` checks at import that each twin still
matches its classifier.

    python baselines_ml.py --input data/wi_records_cleaned_full.csv --output results.csv
    python baselines_ml.py --models lr rf xgb
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from utils import (
    SEED,
    TARGET_COL,
    add_io_args,
    align_proba,
    build_preprocessor,
    compute_metrics,
    configure_verbosity,
    get_feature_columns,
    get_folds,
    load_data,
    log_results,
    maybe_remap,
    nan_metrics,
)


# -----------------------------------------------------------------------------
# Model factories — each returns (model, preprocessing_kwargs)
# -----------------------------------------------------------------------------
def make_dummy(n_classes: int) -> tuple[object, dict]:
    """Majority-class baseline. Floor that every other method must beat."""
    model = DummyClassifier(strategy="most_frequent", random_state=SEED)
    return model, {"scale": False, "encoding": "ordinal"}


def make_lr(n_classes: int) -> tuple[object, dict]:
    """L2-regularized multinomial logistic regression, class-weighted."""
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    return model, {"scale": True, "encoding": "onehot"}


def make_lr_unweighted(n_classes: int) -> tuple[object, dict]:
    """Same as make_lr but no class weighting — measures the accuracy-vs-QWK
    tradeoff that class_weight introduces."""
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
        n_jobs=-1,
    )
    return model, {"scale": True, "encoding": "onehot"}


def make_rf(n_classes: int) -> tuple[object, dict]:
    """Random forest at sklearn defaults, with balanced_subsample weighting."""
    model = RandomForestClassifier(
        n_estimators=100,       # sklearn default
        max_depth=None,         # sklearn default
        min_samples_leaf=1,     # sklearn default
        max_features="sqrt",    # sklearn default
        class_weight="balanced_subsample",   # policy, not a default
        random_state=SEED,
        n_jobs=-1,
    )
    return model, {"scale": False, "encoding": "ordinal"}


def make_rf_unweighted(n_classes: int) -> tuple[object, dict]:
    """Random forest without class weighting. Registered but not run by any driver."""
    model = RandomForestClassifier(
        n_estimators=100,       # sklearn default
        max_depth=None,         # sklearn default
        min_samples_leaf=1,     # sklearn default
        max_features="sqrt",    # sklearn default
        random_state=SEED,
        n_jobs=-1,
    )
    return model, {"scale": False, "encoding": "ordinal"}


def make_xgb(n_classes: int) -> tuple[object, dict]:
    """XGBoost. Balanced sample weights are applied per-row in `run_cv`, since
    multiclass XGB has no `class_weight`.
    """
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        # num_class is intentionally NOT set: XGBoost 2.x's sklearn wrapper
        # derives it from the (contiguous, 0-based) labels it is handed, and a
        # hardcoded n_classes conflicts with the per-fold DENSE relabeling in
        # run_cv that a rare-class-missing fold needs (see there).
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    return model, {"scale": False, "encoding": "ordinal", "balanced_weights": True}


def make_xgb_unweighted(n_classes: int) -> tuple[object, dict]:
    """XGBoost, no sample weighting."""
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        # num_class is intentionally NOT set: XGBoost 2.x's sklearn wrapper
        # derives it from the (contiguous, 0-based) labels it is handed, and a
        # hardcoded n_classes conflicts with the per-fold DENSE relabeling in
        # run_cv that a rare-class-missing fold needs (see there).
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    return model, {"scale": False, "encoding": "ordinal"}


MODEL_REGISTRY: dict[str, Callable[[int], tuple[object, dict]]] = {
    "dummy": make_dummy,
    "lr": make_lr,
    "lr_unweighted": make_lr_unweighted,
    "rf": make_rf,
    "rf_unweighted": make_rf_unweighted,
    "xgb": make_xgb,
    "xgb_unweighted": make_xgb_unweighted,
}


# -----------------------------------------------------------------------------
# Regression factories
# -----------------------------------------------------------------------------
# Regression twins of the four classifiers above, used by `cross_scale.py` and
# `shap_xgb.py`. Only the objective differs; every shared hyperparameter is
# copied verbatim.
#
# All four are unweighted, the one place they diverge from the classifiers.
def make_dummy_regressor(n_classes: int) -> tuple[object, dict]:
    """Constant (mean) prediction, the ranking floor.

    Every pair ties, so the c-index is exactly 0.5 by construction rather than
    NaN as with Spearman or Kendall.
    """
    from sklearn.dummy import DummyRegressor
    return DummyRegressor(strategy="mean"), {"scale": False, "encoding": "ordinal"}


def make_ridge(n_classes: int) -> tuple[object, dict]:
    """L2 linear regression, the twin of `make_lr`. Same scale+onehot preprocessing."""
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0, random_state=SEED), {"scale": True, "encoding": "onehot"}


def make_rf_regressor(n_classes: int) -> tuple[object, dict]:
    """Random forest regressor at sklearn defaults, twin of `make_rf`.

    NB `max_features` is 1.0 here against "sqrt" in `make_rf`; both are library
    defaults for their respective estimators.
    """
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=100,       # sklearn default
        max_depth=None,         # sklearn default
        min_samples_leaf=1,     # sklearn default
        max_features=1.0,       # sklearn REGRESSOR default (classifier's is "sqrt")
        random_state=SEED,
        n_jobs=-1,
    )
    return model, {"scale": False, "encoding": "ordinal"}


def make_xgb_regressor(n_classes: int, n_jobs: int = -1) -> tuple[object, dict]:
    """XGBoost regressor, twin of `make_xgb` and identical to
    `shap_xgb.make_xgb_regressor`. Only `objective` differs from the classifier.
    """
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=n_jobs,
    )
    return model, {"scale": False, "encoding": "ordinal"}


REGRESSOR_REGISTRY: dict[str, Callable[[int], tuple[object, dict]]] = {
    "dummy": make_dummy_regressor,
    "ridge": make_ridge,
    "rf": make_rf_regressor,
    "xgb": make_xgb_regressor,
}


def _assert_regressor_parity() -> None:
    """Check that `make_xgb_regressor` matches `shap_xgb`'s copy, and that each
    regressor shares every hyperparameter it has in common with its classifier
    twin. Called at import by cross_scale.
    """
    import shap_xgb

    def _same(a, b) -> bool:
        # XGBoost's `missing` defaults to NaN, and NaN != NaN would report a
        # phantom drift on every call.
        if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
            return True
        return a == b

    mine = make_xgb_regressor(0)[0].get_params()
    theirs = shap_xgb.make_xgb_regressor(n_jobs=-1).get_params()
    drift = {k: (mine.get(k), theirs.get(k)) for k in set(mine) | set(theirs)
             if k != "n_jobs" and not _same(mine.get(k), theirs.get(k))}
    if drift:
        raise AssertionError(
            f"make_xgb_regressor has drifted from shap_xgb.make_xgb_regressor: {drift}")

    for name, (clf_f, reg_f) in {
        "rf": (make_rf, make_rf_regressor),
        "xgb": (make_xgb, make_xgb_regressor),
    }.items():
        c, r = clf_f(3)[0].get_params(), reg_f(3)[0].get_params()
        # Exempt: `criterion`/`objective`/`eval_metric` ARE the objective (RF
        # spells it `criterion`: gini vs squared_error); `class_weight` has no
        # regressor analogue; `max_features` is the documented library-default
        # difference (see make_rf_regressor); `n_jobs` is a runtime knob.
        shared = (set(c) & set(r)) - {"max_features", "class_weight", "objective",
                                      "eval_metric", "criterion", "n_jobs"}
        bad = {k: (c[k], r[k]) for k in shared if not _same(c[k], r[k])}
        if bad:
            raise AssertionError(
                f"{name} classifier/regressor twins disagree on shared "
                f"hyperparameters {bad}; only the objective may differ.")


# -----------------------------------------------------------------------------
# Single-model CV loop
# -----------------------------------------------------------------------------
def run_cv(
    model_name: str,
    df: pd.DataFrame,
    folds: list[dict],
) -> list[dict]:
    """Run 5-fold CV for one model. Returns list of per-fold metric dicts."""
    n_classes = df[TARGET_COL].nunique()
    factory = MODEL_REGISTRY[model_name]

    fold_metrics: list[dict] = []
    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"Training {model_name.upper()} ({n_classes} classes)")
    print(f"{'=' * 60}")

    # XGB needs labels in 0..K-1. Build a stable mapping once.
    classes_sorted = sorted(df[TARGET_COL].unique())
    label_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    num_cols = get_feature_columns(df)  # all non-target columns are numeric

    for fold in folds:
        fold_idx = fold["fold"]
        train_idx, val_idx = fold["train_idx"], fold["val_idx"]

        X_train_df = df.iloc[train_idx]
        X_val_df = df.iloc[val_idx]
        y_train = df.iloc[train_idx][TARGET_COL].map(label_to_idx).to_numpy()
        y_val = df.iloc[val_idx][TARGET_COL].map(label_to_idx).to_numpy()

        try:
            # Fresh model + preprocessor per fold (no leakage)
            model, pre_kwargs = factory(n_classes)
            # 'balanced_weights' is a model-loop directive, not a preprocessing arg.
            use_balanced_weights = pre_kwargs.pop("balanced_weights", False)
            pre = build_preprocessor(
                numerical_cols=num_cols,
                categorical_cols=[],
                **pre_kwargs,
            )

            X_train = pre.fit_transform(X_train_df)
            X_val = pre.transform(X_val_df)

            # Dense per-fold relabeling. A fold missing an ultra-rare class
            # leaves the global label space non-contiguous, which XGBoost
            # rejects. Remap the present classes onto 0..m-1 for fitting and map
            # predictions back. A no-op when all classes are present.
            present = np.unique(y_train)              # global idxs present in train
            g2d = {g: i for i, g in enumerate(present)}
            y_train_fit = np.array([g2d[v] for v in y_train])

            # For XGB-balanced: per-row weights so minority classes count more.
            if use_balanced_weights:
                from sklearn.utils.class_weight import compute_sample_weight
                sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_fit)
                model.fit(X_train, y_train_fit, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train_fit)

            # Predictions come back in the dense space; lift to global idxs.
            y_pred_idx = present[model.predict(X_val)]

            # Dummy's predict_proba is degenerate (1.0 on majority), which makes
            # log_loss explode on minorities — leave it blank (NaN) by design.
            if model_name == "dummy":
                y_proba = None
            elif hasattr(model, "predict_proba"):
                # proba columns follow model.classes_ (dense 0..m-1); map each
                # back to its global idx and widen to the full class space.
                y_proba = model.predict_proba(X_val)
                y_proba = align_proba(y_proba, present[model.classes_], n_classes)
            else:
                y_proba = None
        except Exception as e:
            print(f"  Fold {fold_idx}: SKIPPED ({model_name} could not fit — {e})")
            fold_metrics.append(nan_metrics())
            continue

        # Map back to original ordinal labels for metric computation.
        y_pred = np.array([idx_to_label[i] for i in y_pred_idx])
        y_true = np.array([idx_to_label[i] for i in y_val])

        m = compute_metrics(y_true, y_pred, y_proba=y_proba, labels=classes_sorted)
        fold_metrics.append(m)
        print(
            f"  Fold {fold_idx}: "
            f"acc={m['accuracy']:.3f}  "
            f"bal_acc={m['balanced_acc']:.3f}  "
            f"macroF1={m['macro_f1']:.3f}  "
            f"qwk={m['qwk']:.3f}  "
            f"mae={m['mae']:.3f}  "
            f"ll={m['log_loss']:.3f}  "
            f"(n_train={len(train_idx)}, n_val={len(val_idx)})"
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
def main() -> None:
    parser = argparse.ArgumentParser()
    add_io_args(parser)
    parser.add_argument(
        "--models",
        nargs="+",
        # Weighted/balanced-only by default: the
        # `*_unweighted` twins are dropped from the default sweep — pass them
        # explicitly via --models if an unweighted comparison is ever wanted.
        # `dummy` is kept as the majority-class floor (it has no weighted twin).
        default=["dummy", "lr", "rf", "xgb"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to run (default: weighted/balanced variants + dummy floor)",
    )
    parser.add_argument(
        "--features", choices=("numeric", "text"), default="numeric",
        help="Feature source. 'numeric' (default): the prepared numeric columns "
             "of the input CSV (the preprocessed `_full` data). 'text': serialize "
             "each row and sentence-embed it (the raw `_raw` path) so the same "
             "lr/rf/xgb/dummy estimators run on the MiniLM embeddings -- the "
             "within-state embedding-space ceiling that matches cross-state.",
    )
    parser.add_argument(
        "--compliance", choices=("verbose", "summary", "abnormal_only"),
        default="verbose",
        help="Compliance-text verbosity for --features text serialization "
             "(matches llm.py / tabular_dl / the transfer serialization).",
    )
    parser.add_argument(
        "--method-suffix", default="",
        help="Optional suffix appended to every method tag, e.g. --method-suffix "
             "raw -> 'lr_raw'. Disambiguates the embedding run from the numeric "
             "run (bare tag). Empty by default.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Where to cache --features text embeddings (default: an 'embeddings' "
             "subdir next to the results file, shared with tabular_dl / "
             "transfer_tabular when the state/text/model keys match).",
    )
    args = parser.parse_args()
    configure_verbosity(args.verbose)

    df = load_data(args.input)
    df = maybe_remap(df, args.remap_state, allow_identity=args.allow_identity, scale=args.rating_scale)
    folds = get_folds(df, folds_path=args.folds)

    if args.features == "text":
        # Raw-data path: reuse the shared embedder (identical matrix/cache to
        # tabular_dl --features text and transfer_tabular). run_cv reads
        # get_feature_columns(df) internally, which returns the emb_<i> columns.
        from transfer_tabular import embed_dataframe_cv               # noqa: E402
        state = args.remap_state or "within"
        df, feature_cols = embed_dataframe_cv(
            df, args.compliance, state, output_path=args.output,
            cache_dir=args.cache_dir)
    else:
        feature_cols = get_feature_columns(df)
        print(f"Using {len(feature_cols)} numeric feature columns (no categoricals).")

    for model_name in args.models:
        fold_results = run_cv(model_name, df, folds)
        tag = f"{model_name}_{args.method_suffix}" if args.method_suffix else model_name
        log_results(tag, fold_results, output_path=args.output, notes=tag)

    print(f"\nDone. See {args.output} for aggregated scores.")


if __name__ == "__main__":
    main()