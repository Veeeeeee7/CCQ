"""Shared foundations: data loading, folds, metrics, results logging.

Every experiment loads its data, obtains its folds, computes its metrics and
records its results through this module.

A dataset is any CSV with a `provider_id` column and an ordinal `qr_rating`
target; every other column is a feature. `load_data` refuses a file containing
any name in `LEAKY_COLS`. Preprocessing helpers are returned unfit and must be
fit inside each fold.

The memory constants below cap what HuggingFace may plan against, which
otherwise sizes itself from node RAM rather than the job's cgroup limit.
"""
from __future__ import annotations

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

# -----------------------------------------------------------------------------
# Dataset contract + defaults
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FOLDS_DIR = PROJECT_ROOT / "fold_indices"

# Model checkpoints, embedding caches and the HuggingFace download cache.
# Defaults inside the repository; set CCQ_ARTIFACT_ROOT to scratch on a cluster.
ARTIFACT_ROOT = Path(
    os.environ.get("CCQ_ARTIFACT_ROOT") or (PROJECT_ROOT / "artifacts")
)


# Redirect the HuggingFace cache under ARTIFACT_ROOT unless HF_HOME is already
# set. Must run before transformers or huggingface_hub is imported.
if not os.environ.get("HF_HOME"):
    _HF_CACHE = ARTIFACT_ROOT / "hf_cache"
    os.environ["HF_HOME"] = str(_HF_CACHE)
    os.environ.setdefault("HF_HUB_CACHE", str(_HF_CACHE / "hub"))
    print(f"[utils] HF_HOME -> {_HF_CACHE} (keeps model downloads off HOME; "
          f"override by exporting HF_HOME or CCQ_ARTIFACT_ROOT)")


# -----------------------------------------------------------------------------
# Memory budget
# -----------------------------------------------------------------------------
# Two distinct ceilings:
#   CCQ_HOST_MEM_LIMIT_GB  system RAM; must be <= the job's allocation.
#   CCQ_GPU_MEM_LIMIT_GB   VRAM; unset autodetects ~92% of the card. Setting it
#                          below the model's weight footprint spills layers to
#                          host RAM or disk.
def _env_float(name: str, default: "float | None") -> "float | None":
    """Read a float from the environment. Unset or empty yields `default`;
    a malformed value exits with a message rather than a traceback.
    """
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(
            f"[utils] {name}={raw!r} is not a number. Unset it, or set it to a "
            f"value in GB (e.g. {name}=46)."
        ) from None


HOST_MEM_LIMIT_GB = _env_float("CCQ_HOST_MEM_LIMIT_GB", 50.0)
# Host RAM accelerate may use as a weight-offload target.
CPU_OFFLOAD_BUDGET_GB = _env_float("CCQ_CPU_OFFLOAD_GB", 6.0)
_GPU_MEM_LIMIT_GB = _env_float("CCQ_GPU_MEM_LIMIT_GB", None)  # None => autodetect


def offload_dir() -> Path:
    """Disk landing zone for weights accelerate cannot fit in VRAM."""
    d = ARTIFACT_ROOT / "offload"
    d.mkdir(parents=True, exist_ok=True)
    return d


def accelerate_max_memory(cpu_gb: "float | None" = None) -> dict:
    """Build a `max_memory` map for `from_pretrained(device_map=...)`.

    Returns e.g. {0: '88GiB', 'cpu': '6GiB'}: integer GPU device indices plus a
    host-RAM allowance.
    """
    import torch  # local import: utils is imported by CPU-only scripts too

    mm: dict = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            cap = _GPU_MEM_LIMIT_GB if _GPU_MEM_LIMIT_GB is not None else total_gb * 0.92
            mm[i] = f"{cap:.0f}GiB"
    mm["cpu"] = f"{(cpu_gb if cpu_gb is not None else CPU_OFFLOAD_BUDGET_GB):.0f}GiB"
    return mm


def run_artifact_dir(output_path) -> Path:
    """Per-run artifact subtree under ARTIFACT_ROOT, keyed by the results-file stem.

    Runs sharing an ``--output`` share the subtree and so share its embedding
    cache; different outputs get disjoint subtrees, so concurrent jobs do not
    collide.
    """
    return ARTIFACT_ROOT / Path(output_path).stem

# The id is index-only and dropped; the target is the ordinal label. Everything
# else is a feature.
TARGET_COL = "qr_rating"
ID_COL = "provider_id"

# Columns encoding the target or its administration rather than the provider.
# Excluded upstream; load_data() raises if one appears.
LEAKY_COLS = (
    "rating_on_site",
    "award_date",
    "expiration_date",
    "rating_age_days",
    "days_until_rating_expires",
)

SEED = 42
N_SPLITS = 5

# Sensible defaults; every script also accepts --input / --output to override.
DEFAULT_DATA_PATH = DATA_DIR / "data.csv"
DEFAULT_RESULTS_PATH = RESULTS_DIR / "results.csv"
DEFAULT_FOLDS_PATH = FOLDS_DIR / "folds.json"

EncodingMode = Literal["onehot", "ordinal", "passthrough"]


# -----------------------------------------------------------------------------
# Shared CLI args: --input / --output (+ --folds)
# -----------------------------------------------------------------------------
def add_io_args(parser) -> None:
    """Attach the shared --input / --output / --folds flags to a parser."""
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_DATA_PATH,
        help=f"Path to the dataset CSV (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_RESULTS_PATH,
        help=f"Path to the results CSV to append to (default: {DEFAULT_RESULTS_PATH}).",
    )
    parser.add_argument(
        "--folds", type=Path, default=DEFAULT_FOLDS_PATH,
        help=f"Path to cached fold indices (default: {DEFAULT_FOLDS_PATH}).",
    )
    parser.add_argument(
        "--remap-state", default=None,
        choices=("WI", "NC", "CA", "GA", "CO", "KY", "MD", "MT", "NE", "OK", "SC", "WA"),
        help="Reconcile qr_rating to the shared scale for this state, so "
             "within-state results are directly comparable to transfer. Also "
             "drops WI rating 0 the same way transfer does. Omit to keep the "
             "native per-state scale.",
    )
    parser.add_argument(
        "--allow-identity", action="store_true",
        help="Permit native-scale passthrough for a state with no rating map "
             "instead of erroring (rarely needed; the four states are mapped).",
    )
    parser.add_argument(
        "--rating-scale", choices=("3star", "5star"), default="3star",
        help="Rating reconciliation scale for --remap-state, mirroring "
             "transfer_common.add_transfer_args: '3star' (default; WI/NC/CA "
             "5->3 collapse, GA native 3-level) or '5star' (native WI/NC/CA "
             "{1..5}, GA unmapped -> hard error if --remap-state GA is used "
             "with --rating-scale 5star). No effect when --remap-state is omitted.",
    )
    add_verbosity_arg(parser)


def add_verbosity_arg(parser) -> None:
    """Attach the shared --verbose flag.

    Kept separate from add_io_args so parsers that don't take the I/O flags
    (utils.py's own fold-builder main) can still opt into the same switch.
    """
    parser.add_argument(
        "--verbose", action="store_true",
        default=os.environ.get("CCQ_VERBOSE", "") not in ("", "0", "false", "False"),
        help="Print ALL third-party warnings/logs (tokenizer sequence-length "
             "notices, sklearn UndefinedMetric, gradient-boosting chatter). "
             "Default is quiet: only this project's own logs are shown. Can "
             "also be enabled for a whole run with the env var CCQ_VERBOSE=1.",
    )


def configure_verbosity(verbose: bool = False) -> None:
    """Silence third-party warnings and logs unless `verbose` is set.

    Suppresses tokenizer length notices, sklearn UndefinedMetricWarning, general
    UserWarning/FutureWarning, and library info-level logging. This project's own
    output is unaffected.
    """
    if verbose:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        return

    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # tqdm writes to STDERR, so the model-load / encode progress bars are what
    # bloat the .err logs. Kill them at every layer.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # HuggingFace transformers: tokenizer notice, model chatter, shard-loading bar.
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    # huggingface_hub: download/resolve bars (separate from transformers').
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
    # These libraries emit through the logging module rather than warnings.
    for name in ("xgboost", "datasets", "sklearn"):
        logging.getLogger(name).setLevel(logging.ERROR)


def align_proba(
    y_proba: "np.ndarray",
    present_classes: Sequence[int],
    n_classes: int,
) -> "np.ndarray":
    """Widen a probability matrix to the full label space, zero-filling absent classes.

    A model fit on a fold missing a rare class returns fewer columns than
    `n_classes`, which `log_loss` rejects. `present_classes` gives the 0-indexed
    class id of each column. No-op when the matrix is already full width.
    """
    y_proba = np.asarray(y_proba)
    if y_proba.ndim != 2 or y_proba.shape[1] == n_classes:
        return y_proba
    full = np.zeros((y_proba.shape[0], n_classes), dtype=y_proba.dtype)
    full[:, np.asarray(present_classes, dtype=int)] = y_proba
    return full


def nan_metrics() -> "dict[str, float]":
    """An all-NaN metric dict shaped like `compute_metrics`.

    Recorded for a fold that could not be evaluated, so the per-fold row count
    stays intact and the aggregate falls back to nan-aware means.
    """
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
    """Reconcile qr_rating to the shared scale, or return df unchanged if `state` is
    falsy. `scale` selects the map set. Imported lazily to avoid an import cycle.
    """
    if not state:
        return df
    from transfer_common import remap_ratings  # lazy: breaks the import cycle
    print(f"[remap] within-state reconciliation requested for {state!r} (scale={scale})")
    return remap_ratings(df, state, scale=scale, allow_identity=allow_identity)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_data(
    input_path: "Path | str | None" = None,
    drop_cols: Sequence[str] = (ID_COL,),
) -> pd.DataFrame:
    """Load a prepared dataset CSV.

    Drops id columns, rows with a missing target, and all-missing feature
    columns. Raises if any `LEAKY_COLS` name is present.
    """
    path = Path(input_path) if input_path is not None else DEFAULT_DATA_PATH
    # low_memory=False avoids the mixed-type DtypeWarning on wide CSVs.
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
# Folds: created once per dataset, then reused across methods
# -----------------------------------------------------------------------------
def _read_folds(folds_path: Path) -> list[dict]:
    with open(folds_path) as f:
        folds = json.load(f)
    for fold in folds:
        fold["train_idx"] = np.array(fold["train_idx"])
        fold["val_idx"] = np.array(fold["val_idx"])
    return folds


def _stratification_labels(y: "np.ndarray", n_splits: int) -> "np.ndarray":
    """Labels to stratify a split on, safe for ultra-rare classes.

    A class with fewer than `min_count` members is folded into its nearest
    ordinal neighbour FOR SPLITTING ONLY; the true labels are untouched.
    """
    y = np.asarray(y)
    counts = pd.Series(y).value_counts()
    ok = sorted(c for c, n in counts.items() if n >= n_splits)
    if len(ok) == len(counts):
        return y  # every class is large enough; stratify on the labels as-is
    if not ok:
        # Degenerate: nothing is frequent enough. Fall back to a single stratum
        # so KFold still produces splits rather than crashing.
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
    """Create stratified 5-fold splits and cache them to disk.

    Splits are positional indices, so one cache serves any row-aligned view of
    the same providers.
    """
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

    folds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(folds_path, "w") as f:
        json.dump(folds, f)
    print(f"Saved {N_SPLITS} folds ({len(df)} rows) to {folds_path}")
    # Re-read so callers get numpy arrays, matching get_folds().
    return _read_folds(folds_path)


def get_folds(
    df: pd.DataFrame,
    folds_path: "Path | str" = DEFAULT_FOLDS_PATH,
    overwrite: bool = False,
) -> list[dict]:
    """Load cached fold indices, regenerating them if they do not fit the input.

    The cache is reused only when every row is covered exactly once and no index
    falls outside the frame.
    """
    folds_path = Path(folds_path)
    if folds_path.exists() and not overwrite:
        folds = _read_folds(folds_path)
        covered = sum(len(f["val_idx"]) for f in folds)
        max_idx = max(int(f["val_idx"].max()) for f in folds)
        if covered == len(df) and max_idx < len(df):
            return folds
        print(
            f"Cached folds at {folds_path} don't match this dataset "
            f"(covered={covered}, rows={len(df)}); regenerating."
        )
    return create_folds(df, folds_path=folds_path)


# -----------------------------------------------------------------------------
# Metrics: ordinal-aware
# -----------------------------------------------------------------------------
def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: "np.ndarray | None" = None,
    labels: "Sequence[int] | None" = None,
) -> dict[str, float]:
    """Compute the metric suite; QWK is the primary metric.

    `y_true` / `y_pred` are in the original label space. `y_proba` is optional
    and its columns must be ordered to match `labels`; log_loss is NaN without
    it.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "micro_f1": f1_score(y_true, y_pred, average="micro"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),  # PRIMARY
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
    """Append per-fold and aggregated (mean ± std) results to a results CSV.

    fold_results: one dict per fold, each as returned by compute_metrics().
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, fr in enumerate(fold_results):
        rows.append({"method": method, "fold": i, **fr, "notes": notes})

    metric_keys = list(fold_results[0].keys())
    agg = {"method": method, "fold": "mean", "notes": notes}
    for k in metric_keys:
        vals = [fr[k] for fr in fold_results]
        # nan-aware: a fold that could not be scored (e.g. a model that refused
        # to fit on a fold missing an ultra-rare class) contributes NaN
        # via nan_metrics(); average over the folds that did run instead of
        # letting one NaN collapse the whole method's mean. An all-NaN method
        # (every fold failed) still yields NaN here, correctly flagging it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            agg[k] = np.nanmean(vals)
            agg[f"{k}_std"] = np.nanstd(vals)
    rows.append(agg)

    new_df = pd.DataFrame(rows)
    # Append only to a NON-EMPTY existing file. A zero-byte file (e.g. a temp
    # output pre-created by mktemp, or a previous run killed before its first
    # write) has no header for pd.read_csv and would raise EmptyDataError;
    # treat it as "start fresh" and let this write lay down the header.
    if output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_csv(output_path)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(output_path, index=False)
    print(f"Logged results for '{method}' to {output_path}")


# -----------------------------------------------------------------------------
# Feature columns
# -----------------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column except the target is a feature."""
    return [c for c in df.columns if c != TARGET_COL]


# -----------------------------------------------------------------------------
# Preprocessor builder (sklearn ColumnTransformer)
# -----------------------------------------------------------------------------
def build_preprocessor(
    numerical_cols: list[str],
    categorical_cols: list[str] | None = None,
    scale: bool = True,
    encoding: EncodingMode = "onehot",
    numeric_impute: str = "median",
    categorical_impute: str = "most_frequent",
) -> ColumnTransformer:
    """Build an unfit sklearn ColumnTransformer.

    `scale` applies StandardScaler to numerics. `encoding` selects one of
    "onehot" (linear models), "ordinal" (trees), or "passthrough".
    `numeric_impute` / `categorical_impute` are SimpleImputer strategies.
    """
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


# -----------------------------------------------------------------------------
# TabNet-specific preprocessing
# -----------------------------------------------------------------------------
def build_tabnet_arrays(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numerical_cols: list[str],
    categorical_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[str]]:
    """Prepare train/val arrays for TabNet.

    TabNet requires NaN-free scaled numerics and non-negative integer codes, so
    code 0 is reserved for unknown/missing and known categories shift to 1..k.

    Returns (X_train, X_val, cat_idxs, cat_dims, feature_names).
    """
    categorical_cols = list(categorical_cols or [])

    # ---- Numerics: impute then scale ----
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

    # ---- Categoricals: non-negative codes (0 = unknown/missing, 1..k known) ----
    cat_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )

    def to_uniform_str(frame: pd.DataFrame) -> pd.DataFrame:
        # Robust NaN-aware string conversion for mixed-dtype categoricals.
        mask = frame.isna()
        return frame.astype(object).where(~mask, "MISSING").astype(str)

    X_train_cat_raw = cat_encoder.fit_transform(to_uniform_str(train_df[categorical_cols]))
    X_val_cat_raw = cat_encoder.transform(to_uniform_str(val_df[categorical_cols]))

    X_train_cat = (X_train_cat_raw + 1).astype(np.int64)
    X_val_cat = (X_val_cat_raw + 1).astype(np.int64)

    cat_dims = [int(X_train_cat[:, i].max()) + 1 for i in range(len(categorical_cols))]
    # Clamp any val code that exceeds the train range into the unknown bucket.
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
# First-time setup: build folds from a prepared dataset
# -----------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Create the shared stratified folds from a prepared dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH,
                        help=f"Dataset CSV to build folds from (default: {DEFAULT_DATA_PATH}).")
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS_PATH,
                        help=f"Where to write fold indices (default: {DEFAULT_FOLDS_PATH}).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate folds even if the cache already matches.")
    parser.add_argument("--remap-state", default=None,
                        choices=("WI", "NC", "CA", "GA", "CO", "KY", "MD", "MT", "NE", "OK", "SC", "WA"),
                        help="Reconcile qr_rating to the shared scale before "
                             "building folds; keeps folds aligned with the "
                             "remapped within-state run.")
    parser.add_argument("--allow-identity", action="store_true",
                        help="Permit native-scale passthrough for an unmapped state.")
    parser.add_argument(
        "--rating-scale", choices=("3star", "5star"), default="3star",
        help="Scale for --remap-state, matching add_io_args/add_transfer_args. "
             "'5star' stratifies the folds on the native {1..5} labels so the "
             "5-star within-state CV is stratified on the scale it is scored on.")
    add_verbosity_arg(parser)
    args = parser.parse_args()
    configure_verbosity(args.verbose)

    df = load_data(args.input)
    df = maybe_remap(df, args.remap_state, allow_identity=args.allow_identity,
                     scale=args.rating_scale)
    get_folds(df, folds_path=args.folds, overwrite=args.overwrite)


if __name__ == "__main__":
    main()