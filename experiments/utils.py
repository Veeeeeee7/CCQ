"""Shared foundations: data loading, folds, metrics and results logging.

A dataset is any CSV with a `provider_id` column and an ordinal `qr_rating`
target; every other column is a feature.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    mean_absolute_error,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FOLDS_DIR = PROJECT_ROOT / "fold_indices"

# Model checkpoints, embedding caches and the HuggingFace download cache.
ARTIFACT_ROOT = Path(
    os.environ.get("CCQ_ARTIFACT_ROOT") or (PROJECT_ROOT / "artifacts")
)

# Must run before transformers or huggingface_hub is imported.
if not os.environ.get("HF_HOME"):
    _HF_CACHE = ARTIFACT_ROOT / "hf_cache"
    os.environ["HF_HOME"] = str(_HF_CACHE)
    os.environ.setdefault("HF_HUB_CACHE", str(_HF_CACHE / "hub"))
    print(f"[utils] HF_HOME -> {_HF_CACHE} (keeps model downloads off HOME; "
          f"override by exporting HF_HOME or CCQ_ARTIFACT_ROOT)")


def run_artifact_dir(output_path) -> Path:
    """Per-run artifact subtree under ARTIFACT_ROOT, keyed by the results-file stem."""
    return ARTIFACT_ROOT / Path(output_path).stem

TARGET_COL = "qr_rating"
ID_COL = "provider_id"

# Columns that encode the target or its administration. load_data() refuses them.
LEAKY_COLS = (
    "rating_on_site",
    "award_date",
    "expiration_date",
    "rating_age_days",
    "days_until_rating_expires",
)

SEED = 42
N_SPLITS = 5

DEFAULT_DATA_PATH = DATA_DIR / "data.csv"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "results.csv"
DEFAULT_FOLDS_PATH = FOLDS_DIR / "folds.json"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"

EncodingMode = Literal["onehot", "ordinal", "passthrough"]


# -----------------------------------------------------------------------------
# Output location
# -----------------------------------------------------------------------------
def add_output_args(parser) -> None:
    parser.add_argument(
        "--results", type=Path, default=RESULTS_DIR,
        help=f"Base directory for results CSVs (default: {RESULTS_DIR}).",
    )
    parser.add_argument(
        "--logs", type=Path, default=DEFAULT_LOGS_DIR,
        help=f"Base directory for run logs (default: {DEFAULT_LOGS_DIR}).",
    )
    parser.add_argument(
        "--date", default=None,
        help="Optional subfolder for --results and --logs, e.g. 2026-08-21. "
             "Omit to write directly into the base directories.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Explicit results CSV path. Overrides --results/--date.",
    )


def _dated(base, date: "str | None") -> Path:
    base = Path(base)
    return base / date if date else base


def resolve_output(args, subdir: str, filename: str) -> Path:
    """`--output` if given, else `<results>/[<date>/]<subdir>/<filename>`."""
    if getattr(args, "output", None):
        path = Path(args.output)
    else:
        path = _dated(getattr(args, "results", RESULTS_DIR),
                      getattr(args, "date", None)) / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class _Tee:
    """Write to a stream and a file at once."""

    def __init__(self, stream, handle):
        self._stream, self._handle = stream, handle

    def write(self, text):
        self._stream.write(text)
        self._handle.write(text)
        self._handle.flush()
        return len(text)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def isatty(self):
        return self._stream.isatty()


def setup_logging(args, subdir: str, stem: str) -> "Path | None":
    """Mirror stdout and stderr into `<logs>/[<date>/]<subdir>/<stem>.{out,err}`."""
    base = getattr(args, "logs", None)
    if not base:
        return None
    log_dir = _dated(base, getattr(args, "date", None)) / subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    import atexit
    import sys as _sys

    orig_out, orig_err = _sys.stdout, _sys.stderr
    out_h = open(log_dir / f"{stem}.out", "a", buffering=1)
    err_h = open(log_dir / f"{stem}.err", "a", buffering=1)
    _sys.stdout = _Tee(orig_out, out_h)
    _sys.stderr = _Tee(orig_err, err_h)

    def _restore():
        # Restore the real streams before closing the files, so output written
        # during interpreter shutdown does not hit a closed handle.
        _sys.stdout, _sys.stderr = orig_out, orig_err
        for h in (out_h, err_h):
            try:
                h.close()
            except OSError:
                pass

    atexit.register(_restore)
    return log_dir


# -----------------------------------------------------------------------------
# Shared CLI args
# -----------------------------------------------------------------------------
STATES = ("WI", "NC", "CA", "GA", "CO", "KY", "MD", "MT", "NE", "OK", "SC", "WA")


def add_io_args(parser) -> None:
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_DATA_PATH,
        help=f"Path to the dataset CSV (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--folds", type=Path, default=DEFAULT_FOLDS_PATH,
        help=f"Path to cached fold indices (default: {DEFAULT_FOLDS_PATH}).",
    )
    add_output_args(parser)
    parser.add_argument(
        "--remap-state", default=None, choices=STATES,
        help="Map qr_rating through this state's rating map for --rating-scale, "
             "dropping unrated or out-of-scale rows.",
    )
    parser.add_argument(
        "--allow-identity", action="store_true",
        help="Keep the native scale for a state with no rating map instead of erroring.",
    )
    parser.add_argument(
        "--rating-scale", choices=("3star", "5star"), default="3star",
        help="Rating scale for --remap-state: '5star' (native {1..5}; GA has no "
             "map) or '3star' (GA native, other states collapsed 5->3).",
    )
    add_verbosity_arg(parser)


def add_verbosity_arg(parser) -> None:
    parser.add_argument(
        "--verbose", action="store_true",
        default=os.environ.get("CCQ_VERBOSE", "") not in ("", "0", "false", "False"),
        help="Show third-party warnings and logs (also enabled by CCQ_VERBOSE=1).",
    )


def configure_verbosity(verbose: bool = False) -> None:
    """Silence third-party warnings, logs and progress bars unless `verbose`."""
    if verbose:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        return

    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
    for name in ("xgboost", "datasets", "sklearn"):
        logging.getLogger(name).setLevel(logging.ERROR)


def align_proba(
    y_proba: "np.ndarray",
    present_classes: Sequence[int],
    n_classes: int,
) -> "np.ndarray":
    """Widen a probability matrix to all `n_classes` columns, zero-filling absent classes."""
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2 or y_proba.shape[1] == n_classes:
        return y_proba
    full = np.zeros((y_proba.shape[0], n_classes), dtype=y_proba.dtype)
    full[:, np.asarray(present_classes, dtype=int)] = y_proba
    return full


def nan_metrics() -> "dict[str, float]":
    """All-NaN metrics for a fold that could not be evaluated."""
    return {
        "accuracy": float("nan"),
        "balanced_acc": float("nan"),
        "macro_f1": float("nan"),
        "micro_f1": float("nan"),
        "qwk": float("nan"),
        "mae": float("nan"),
        "log_loss": float("nan"),
    }


def maybe_remap(df: pd.DataFrame, state: "str | None",
                allow_identity: bool = False, scale: str = "3star") -> pd.DataFrame:
    if not state:
        return df
    from transfer_common import remap_ratings  # lazy: avoids an import cycle
    print(f"[remap] within-state reconciliation requested for {state!r} (scale={scale})")
    return remap_ratings(df, state, scale=scale, allow_identity=allow_identity)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_data(
    input_path: "Path | str | None" = None,
    drop_cols: Sequence[str] = (ID_COL,),
) -> pd.DataFrame:
    """Load a dataset CSV, dropping the id column, unrated rows and all-NaN columns."""
    path = Path(input_path) if input_path is not None else DEFAULT_DATA_PATH
    df = pd.read_csv(path, low_memory=False)
    assert TARGET_COL in df.columns, f"{TARGET_COL!r} not in {path.name}"

    present_drop = [c for c in drop_cols if c in df.columns]
    if present_drop:
        print(f"Dropping index-only column(s): {present_drop}")
        df = df.drop(columns=present_drop)

    present_leaky = [c for c in LEAKY_COLS if c in df.columns]
    if present_leaky:
        raise ValueError(
            f"{path.name} contains target-leaking column(s) {present_leaky}. "
            f"These are excluded by the scraper's preprocessing, so this export "
            f"is stale or a new leak has appeared. Regenerate the cleaned CSVs "
            f"before running; refusing to train on or silently drop a leaking "
            f"feature.")

    n_missing_target = df[TARGET_COL].isna().sum()
    if n_missing_target > 0:
        print(f"Dropping {n_missing_target} rows with missing {TARGET_COL}")
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    all_nan_cols = [c for c in feature_cols if df[c].isna().all()]
    if all_nan_cols:
        print(f"Dropping {len(all_nan_cols)} all-NaN feature columns: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {path.name}")
    print(f"Target distribution:\n{df[TARGET_COL].value_counts().sort_index()}")
    return df


# -----------------------------------------------------------------------------
# Folds: created once per dataset, then reused by every method
# -----------------------------------------------------------------------------
FOLDS_FORMAT_VERSION = 2


def _read_folds(folds_path: Path) -> "tuple[list[dict], str] | None":
    """Return (folds, target_digest), or None if the file is not a current cache."""
    with open(folds_path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "target_digest" not in payload:
        return None
    folds = payload["folds"]
    for fold in folds:
        fold["train_idx"] = np.array(fold["train_idx"])
        fold["val_idx"] = np.array(fold["val_idx"])
    return folds, payload["target_digest"]


def _target_digest(df: pd.DataFrame) -> str:
    """Hash of the label sequence in row order.

    Fold indices are positional, so a cache is only valid for the row order it was
    stratified on; a row count cannot detect a re-ordered export.
    """
    y = np.asarray(df[TARGET_COL].to_numpy())
    finite = y[~pd.isna(y)]
    if finite.size and np.all(np.equal(np.mod(finite.astype(float), 1), 0)):
        payload = y.astype(np.int64).tobytes()
    else:
        payload = y.astype(np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _write_folds(folds: list[dict], folds_path: Path, digest: str, n_rows: int) -> None:
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    plain = [
        {
            "fold": f["fold"],
            "train_idx": np.asarray(f["train_idx"]).tolist(),
            "val_idx": np.asarray(f["val_idx"]).tolist(),
        }
        for f in folds
    ]
    with open(folds_path, "w") as f:
        json.dump(
            {
                "version": FOLDS_FORMAT_VERSION,
                "n_rows": n_rows,
                "target_digest": digest,
                "folds": plain,
            },
            f,
        )


def _stratification_labels(y: "np.ndarray", n_splits: int) -> "np.ndarray":
    """Labels to stratify on, with classes smaller than `n_splits` merged into
    their nearest ordinal neighbour. The true labels are untouched."""
    y = np.asarray(y)
    counts = pd.Series(y).value_counts()
    ok = sorted(c for c, n in counts.items() if n >= n_splits)
    if len(ok) == len(counts):
        return y
    if not ok:
        return np.zeros(len(y), dtype=int)
    strat = y.copy()
    for cls, n in counts.items():
        if n >= n_splits:
            continue
        nearest = min(ok, key=lambda c: abs(c - cls))
        strat[y == cls] = nearest
    return strat


def create_folds(
    df: pd.DataFrame,
    folds_path: "Path | str" = DEFAULT_FOLDS_PATH,
) -> list[dict]:
    """Create stratified 5-fold splits and cache them to disk."""
    folds_path = Path(folds_path)
    strat = _stratification_labels(df[TARGET_COL].to_numpy(), N_SPLITS)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(len(df)), strat)
    ):
        folds.append({
            "fold": fold_idx,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        })

    _write_folds(folds, folds_path, _target_digest(df), len(df))
    print(f"Saved {N_SPLITS} folds ({len(df)} rows) to {folds_path}")
    return _read_folds(folds_path)[0]


def get_folds(
    df: pd.DataFrame,
    folds_path: "Path | str" = DEFAULT_FOLDS_PATH,
    overwrite: bool = False,
) -> list[dict]:
    """Load cached folds, regenerating them unless they cover every row of `df`
    exactly once and were built on the same label sequence."""
    folds_path = Path(folds_path)
    if folds_path.exists() and not overwrite:
        cached = _read_folds(folds_path)
        if cached is None:
            print(f"Cached folds at {folds_path} are not in the current format; regenerating.")
            return create_folds(df, folds_path=folds_path)
        folds, digest = cached
        covered = sum(len(f["val_idx"]) for f in folds)
        max_idx = max(int(f["val_idx"].max()) for f in folds)
        if covered != len(df) or max_idx >= len(df):
            print(
                f"Cached folds at {folds_path} don't match this dataset "
                f"(covered={covered}, rows={len(df)}); regenerating."
            )
            return create_folds(df, folds_path=folds_path)
        current = _target_digest(df)
        if digest == current:
            return folds
        print(
            f"Cached folds at {folds_path} were built on a different row order "
            f"({digest[:12]} on disk, {current[:12]} for this frame); regenerating."
        )
    return create_folds(df, folds_path=folds_path)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: "np.ndarray | None" = None,
    labels: "Sequence[int] | None" = None,
) -> dict[str, float]:
    """Metric suite; QWK is the primary metric. `y_proba` columns follow `labels`."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae": mean_absolute_error(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["log_loss"] = log_loss(y_true, np.asarray(y_proba), labels=labels)
    else:
        metrics["log_loss"] = float("nan")
    return metrics


# -----------------------------------------------------------------------------
# Results logging
# -----------------------------------------------------------------------------
def log_results(
    method: str,
    fold_results: list[dict],
    output_path: "Path | str" = DEFAULT_RESULTS_PATH,
    notes: str = "",
) -> None:
    """Append per-fold rows and a mean/std row to a results CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, fr in enumerate(fold_results):
        rows.append({"method": method, "fold": i, **fr, "notes": notes})

    metric_keys = list(fold_results[0].keys())
    agg = {"method": method, "fold": "mean", "notes": notes}
    for k in metric_keys:
        vals = [fr[k] for fr in fold_results]
        # A fold that could not be scored contributes NaN and is skipped.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            agg[k] = np.nanmean(vals)
            agg[f"{k}_std"] = np.nanstd(vals)
    rows.append(agg)

    new_df = pd.DataFrame(rows)
    if output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_csv(output_path)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(output_path, index=False)
    print(f"Logged results for '{method}' to {output_path}")


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COL]


def build_preprocessor(
    numerical_cols: list[str],
    categorical_cols: list[str] | None = None,
    scale: bool = True,
    encoding: EncodingMode = "onehot",
    numeric_impute: str = "median",
    categorical_impute: str = "most_frequent",
) -> ColumnTransformer:
    """Unfit ColumnTransformer: impute (+ scale) numerics, impute + encode categoricals."""
    categorical_cols = list(categorical_cols or [])

    numeric_steps: list = [("impute", SimpleImputer(strategy=numeric_impute))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    if encoding == "onehot":
        cat_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            min_frequency=2,
        )
    elif encoding == "ordinal":
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
    elif encoding == "passthrough":
        cat_encoder = "passthrough"
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    cat_steps: list = [
        ("impute", SimpleImputer(strategy=categorical_impute, fill_value="MISSING")),
    ]
    if cat_encoder != "passthrough":
        cat_steps.append(("encode", cat_encoder))
    categorical_pipe = Pipeline(cat_steps)

    transformers = []
    if numerical_cols:
        transformers.append(("num", numeric_pipe, numerical_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_tabnet_arrays(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[str]]:
    """TabNet inputs: scaled numerics plus categorical codes (0 = unknown, 1..k known).

    Returns (X_train, X_val, cat_idxs, cat_dims, feature_names).
    """
    categorical_cols = list(categorical_cols or [])

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    X_train_num = num_pipe.fit_transform(train_df[numerical_cols]).astype(np.float32)
    X_val_num = num_pipe.transform(val_df[numerical_cols]).astype(np.float32)

    if not categorical_cols:
        cat_idxs: list[int] = []
        cat_dims: list[int] = []
        return X_train_num, X_val_num, cat_idxs, cat_dims, list(numerical_cols)

    cat_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )

    def to_uniform_str(frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame.isna()
        return frame.astype(object).where(~mask, "MISSING").astype(str)

    X_train_cat_raw = cat_encoder.fit_transform(to_uniform_str(train_df[categorical_cols]))
    X_val_cat_raw = cat_encoder.transform(to_uniform_str(val_df[categorical_cols]))

    X_train_cat = (X_train_cat_raw + 1).astype(np.int64)
    X_val_cat = (X_val_cat_raw + 1).astype(np.int64)

    cat_dims = [int(X_train_cat[:, i].max()) + 1 for i in range(len(categorical_cols))]
    for i in range(len(categorical_cols)):
        over = X_val_cat[:, i] >= cat_dims[i]
        if over.any():
            X_val_cat[over, i] = 0

    X_train = np.concatenate([X_train_num, X_train_cat.astype(np.float32)], axis=1)
    X_val = np.concatenate([X_val_num, X_val_cat.astype(np.float32)], axis=1)

    cat_idxs = list(range(len(numerical_cols), len(numerical_cols) + len(categorical_cols)))
    feature_names = list(numerical_cols) + list(categorical_cols)
    return X_train, X_val, cat_idxs, cat_dims, feature_names


# -----------------------------------------------------------------------------
# Build the fold cache for one dataset
# -----------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Create the cached stratified folds for one dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH,
                        help=f"Dataset CSV to build folds from (default: {DEFAULT_DATA_PATH}).")
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS_PATH,
                        help=f"Where to write fold indices (default: {DEFAULT_FOLDS_PATH}).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate folds even if the cache already matches.")
    parser.add_argument("--remap-state", default=None, choices=STATES,
                        help="Map qr_rating through this state's rating map first.")
    parser.add_argument("--allow-identity", action="store_true",
                        help="Keep the native scale for an unmapped state.")
    parser.add_argument(
        "--rating-scale", choices=("3star", "5star"), default="3star",
        help="Rating scale for --remap-state.")
    add_verbosity_arg(parser)
    args = parser.parse_args()
    configure_verbosity(args.verbose)

    df = load_data(args.input)
    df = maybe_remap(df, args.remap_state, allow_identity=args.allow_identity,
                     scale=args.rating_scale)
    get_folds(df, folds_path=args.folds, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
