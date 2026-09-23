"""Within-state 5-fold CV for the classical classifiers: majority-class dummy,
logistic regression, random forest and XGBoost.

    python baselines_ml.py --input data/wi_records_cleaned_full.csv --output results.csv
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
    resolve_output,
    setup_logging,
)


_SCRIPT = "baselines_ml"


# ---- Model factories: each returns (model, preprocessing_kwargs) ----
def make_dummy(n_classes: int) -> tuple[object, dict]:
    """Majority-class baseline."""
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


def make_rf(n_classes: int) -> tuple[object, dict]:
    """Random forest at sklearn defaults, with balanced_subsample weighting."""
    # Explicit sklearn defaults; only class_weight is a choice.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    return model, {"scale": False, "encoding": "ordinal"}


def make_xgb(n_classes: int) -> tuple[object, dict]:
    """XGBoost; balanced sample weights are applied in `run_cv`."""
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        # num_class is left unset so it follows run_cv's per-fold dense relabeling.
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    return model, {"scale": False, "encoding": "ordinal", "balanced_weights": True}


MODEL_REGISTRY: dict[str, Callable[[int], tuple[object, dict]]] = {
    "dummy": make_dummy,
    "lr": make_lr,
    "rf": make_rf,
    "xgb": make_xgb,
}


# ---- CV loop ----
def run_cv(
    model_name: str,
    df: pd.DataFrame,
    folds: list[dict],
) -> list[dict]:
    """Run CV for one model; returns a list of per-fold metric dicts."""
    n_classes = df[TARGET_COL].nunique()
    factory = MODEL_REGISTRY[model_name]

    fold_metrics: list[dict] = []
    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"Training {model_name.upper()} ({n_classes} classes)")
    print(f"{'=' * 60}")

    classes_sorted = sorted(df[TARGET_COL].unique())
    label_to_idx = {c: i for i, c in enumerate(classes_sorted)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    num_cols = get_feature_columns(df)

    for fold in folds:
        fold_idx = fold["fold"]
        train_idx, val_idx = fold["train_idx"], fold["val_idx"]

        X_train_df = df.iloc[train_idx]
        X_val_df = df.iloc[val_idx]
        y_train = df.iloc[train_idx][TARGET_COL].map(label_to_idx).to_numpy()
        y_val = df.iloc[val_idx][TARGET_COL].map(label_to_idx).to_numpy()

        try:
            model, pre_kwargs = factory(n_classes)
            use_balanced_weights = pre_kwargs.pop("balanced_weights", False)
            pre = build_preprocessor(
                numerical_cols=num_cols,
                categorical_cols=[],
                **pre_kwargs,
            )

            X_train = pre.fit_transform(X_train_df)
            X_val = pre.transform(X_val_df)

            # XGBoost rejects non-contiguous labels, which a fold missing a rare
            # class produces; fit on dense 0..m-1 and map predictions back.
            present = np.unique(y_train)
            g2d = {g: i for i, g in enumerate(present)}
            y_train_fit = np.array([g2d[v] for v in y_train])

            if use_balanced_weights:
                from sklearn.utils.class_weight import compute_sample_weight
                sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_fit)
                model.fit(X_train, y_train_fit, sample_weight=sample_weight)
            else:
                model.fit(X_train, y_train_fit)

            y_pred_idx = present[model.predict(X_val)]

            # Dummy's one-hot proba makes log_loss infinite; leave it NaN.
            if model_name == "dummy":
                y_proba = None
            elif hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_val)
                y_proba = align_proba(y_proba, present[model.classes_], n_classes)
            else:
                y_proba = None
        except Exception as e:
            print(f"  Fold {fold_idx}: SKIPPED ({model_name} could not fit — {e})")
            fold_metrics.append(nan_metrics())
            continue

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


# ---- Main ----
def main() -> None:
    parser = argparse.ArgumentParser()
    add_io_args(parser)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy", "lr", "rf", "xgb"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Models to run.",
    )
    parser.add_argument(
        "--features", choices=("numeric", "text"), default="numeric",
        help="'numeric': use the preprocessed columns; 'text': serialize each "
             "row and use its sentence embedding.",
    )
    parser.add_argument(
        "--compliance", choices=("verbose", "summary", "abnormal_only"),
        default="verbose",
        help="Compliance-text verbosity for --features text.",
    )
    parser.add_argument(
        "--method-suffix", default="",
        help="Suffix appended to every method tag (e.g. raw -> 'lr_raw').",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Embedding cache for --features text (default: 'embeddings' "
             "next to the results file).",
    )
    args = parser.parse_args()
    configure_verbosity(args.verbose)
    _st = (args.remap_state or "all").lower()
    args.output = resolve_output(args, "within_state",
                                 f"experiment_within_state_{_st}_results.csv")
    setup_logging(args, "within_state", f"{_SCRIPT}_{_st}")

    df = load_data(args.input)
    df = maybe_remap(df, args.remap_state, allow_identity=args.allow_identity, scale=args.rating_scale)
    folds = get_folds(df, folds_path=args.folds)

    if args.features == "text":
        # run_cv picks up the emb_<i> columns via get_feature_columns.
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