"""Shared data, split and logging plumbing for the cross-state experiments.

Rating-scale reconciliation, text serialization, the supervision-curve split,
class weights, bootstrap error bars and the results schema all live here, so every
cross-state method uses the same ones.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from text_serialization import serialize_dataframe  # noqa: E402

from utils import (  # noqa: E402
    SEED,
    TARGET_COL,
    DEFAULT_RESULTS_PATH,
    add_output_args,
    _stratification_labels,
    add_verbosity_arg,
    compute_metrics,
    configure_verbosity,
    load_data,
)

_METRIC_KEYS = [
    "accuracy", "balanced_acc", "macro_f1", "micro_f1", "qwk", "mae", "log_loss",
]


# -----------------------------------------------------------------------------
# Label-scale reconciliation
# -----------------------------------------------------------------------------
# Keyed by state code. Ratings missing from a map (e.g. WI's 0 = unrated) are dropped.
_COLLAPSE_5_TO_3 = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

RATING_MAPS: dict[str, dict[int, int]] = {
    "WI": dict(_COLLAPSE_5_TO_3),
    "NC": dict(_COLLAPSE_5_TO_3),
    "CA": dict(_COLLAPSE_5_TO_3),
    "GA": {1: 1, 2: 2, 3: 3},
    "CO": dict(_COLLAPSE_5_TO_3),
    "KY": dict(_COLLAPSE_5_TO_3),
    "MD": dict(_COLLAPSE_5_TO_3),
    "MT": dict(_COLLAPSE_5_TO_3),
    "NE": dict(_COLLAPSE_5_TO_3),
    "OK": dict(_COLLAPSE_5_TO_3),
    "SC": dict(_COLLAPSE_5_TO_3),
    "WA": dict(_COLLAPSE_5_TO_3),
}

_REQUIRED_STATES = frozenset({"WI", "NC", "CA", "GA",
                              "CO", "KY", "MD", "MT", "NE", "OK", "SC", "WA"})
assert _REQUIRED_STATES <= set(RATING_MAPS), (
    f"RATING_MAPS is missing required states {sorted(_REQUIRED_STATES - set(RATING_MAPS))}; "
    f"known: {sorted(RATING_MAPS)}"
)

# Native 5-level scale. GA is deliberately absent, so a 5-star GA run raises.
_IDENTITY_5 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

RATING_MAPS_5STAR: dict[str, dict[int, int]] = {
    "WI": dict(_IDENTITY_5),
    "NC": dict(_IDENTITY_5),
    "CA": dict(_IDENTITY_5),
    "CO": dict(_IDENTITY_5),
    "KY": dict(_IDENTITY_5),
    "MD": dict(_IDENTITY_5),
    "MT": dict(_IDENTITY_5),
    "NE": dict(_IDENTITY_5),
    "OK": dict(_IDENTITY_5),
    "SC": dict(_IDENTITY_5),
    "WA": dict(_IDENTITY_5),
}

RATING_MAP_SETS: dict[str, dict[str, dict[int, int]]] = {
    "3star": RATING_MAPS,
    "5star": RATING_MAPS_5STAR,
}

_REQUIRED_STATES_5STAR = frozenset({"WI", "NC", "CA",
                                    "CO", "KY", "MD", "MT", "NE", "OK", "SC", "WA"})
assert _REQUIRED_STATES_5STAR <= set(RATING_MAPS_5STAR), (
    f"RATING_MAPS_5STAR is missing required states "
    f"{sorted(_REQUIRED_STATES_5STAR - set(RATING_MAPS_5STAR))}; "
    f"known: {sorted(RATING_MAPS_5STAR)}"
)


def scale_infix(scale: str) -> str:
    """Method-tag infix: '' for 3-star, '_5star' for 5-star."""
    return "" if scale == "3star" else "_5star"


def remap_ratings(df: pd.DataFrame, state: str, *, scale: str = "3star",
                  allow_identity: bool = False) -> pd.DataFrame:
    """Map qr_rating through the state's map for `scale`, dropping unmapped rows.

    An unmapped state raises unless ``allow_identity=True``.
    """
    try:
        map_set = RATING_MAP_SETS[scale]
    except KeyError:
        raise ValueError(
            f"Unknown rating scale {scale!r}; known: {sorted(RATING_MAP_SETS)}."
        ) from None

    mapping = map_set.get(state)
    out = df.copy()
    ratings = pd.to_numeric(out[TARGET_COL], errors="coerce").round()

    if mapping is None:
        if not allow_identity:
            raise ValueError(
                f"No {scale} rating map for state {state!r}; known: {sorted(map_set)}. "
                f"Add a map for {state!r}, fix --src-name/--tgt-name, or pass "
                f"allow_identity=True. (In 5-star mode GA is intentionally "
                f"unmapped.)"
            )
        keep = ratings.notna() & (ratings != 0)
        out = out.loc[keep].copy()
        out[TARGET_COL] = ratings.loc[keep].astype(int).to_numpy()
        print(f"  [{state}] allow_identity=True (no rating map); kept {len(out)} rows "
              f"(dropped 0/missing). NOTE: native scale is NOT cross-state comparable. "
              f"dist: {out[TARGET_COL].value_counts().sort_index().to_dict()}")
        return out.reset_index(drop=True)

    fmap = {float(k): v for k, v in mapping.items()}
    keep = ratings.isin(list(fmap.keys()))
    n_drop = int((~keep).sum())
    out = out.loc[keep].copy()
    out[TARGET_COL] = ratings.loc[keep].map(fmap).astype(int).to_numpy()
    dist = out[TARGET_COL].value_counts().sort_index().to_dict()
    print(f"  [{state}] remapped qr_rating via {mapping}; dropped {n_drop} rows "
          f"(0 / missing / out-of-scale). new dist: {dist}")
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
@dataclass
class TransferData:
    source_df: pd.DataFrame
    target_df: pd.DataFrame
    src_texts: pd.Series
    tgt_texts: pd.Series
    labels: list                  # union label space, sorted
    label_to_idx: dict
    idx_to_label: dict
    y_src: np.ndarray             # label indices
    y_tgt: np.ndarray
    src_name: str
    tgt_name: str
    compliance_mode: str
    scale: str = "3star"

    @property
    def n_classes(self) -> int:
        return len(self.labels)


def load_and_prepare(
    source_path, target_path, src_name: str, tgt_name: str,
    compliance_mode: str = "verbose", scale: str = "3star",
) -> TransferData:
    """Load, reconcile ratings, serialize to text and build the union label space."""
    print(f"[load] source={src_name} <- {source_path}  (scale={scale})")
    source_df = remap_ratings(load_data(source_path), src_name, scale=scale)
    print(f"[load] target={tgt_name} <- {target_path}  (scale={scale})")
    target_df = remap_ratings(load_data(target_path), tgt_name, scale=scale)

    labels = sorted(set(source_df[TARGET_COL].unique()) | set(target_df[TARGET_COL].unique()))
    label_to_idx = {c: i for i, c in enumerate(labels)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    src_texts = serialize_dataframe(source_df, compliance_mode=compliance_mode)
    tgt_texts = serialize_dataframe(target_df, compliance_mode=compliance_mode)
    y_src = source_df[TARGET_COL].map(label_to_idx).to_numpy()
    y_tgt = target_df[TARGET_COL].map(label_to_idx).to_numpy()

    print(f"[prep] scale={scale}  labels={labels}  source rows={len(source_df)}  "
          f"target rows={len(target_df)}")
    return TransferData(
        source_df=source_df, target_df=target_df,
        src_texts=src_texts, tgt_texts=tgt_texts,
        labels=labels, label_to_idx=label_to_idx, idx_to_label=idx_to_label,
        y_src=y_src, y_tgt=y_tgt,
        src_name=src_name, tgt_name=tgt_name,
        compliance_mode=compliance_mode, scale=scale,
    )


def parse_pool_sources(items: "list[str]") -> "list[tuple[str, str]]":
    """Parse repeated ``--pool-sources NAME=PATH`` values."""
    out = []
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--pool-sources expects NAME=PATH, got {item!r}")
        out.append((name, path))
    return out


def load_and_prepare_pooled(
    source_specs: "list[tuple[str, str]]", tgt_name: str, target_path,
    compliance_mode: str = "verbose", scale: str = "3star", pool_name: str = "LOSO",
) -> TransferData:
    """Leave-one-state-out: pool several source states against a held-out target.

    Returns a ``TransferData`` whose source is the concatenated pool.
    """
    src_dfs, src_names = [], []
    for name, path in source_specs:
        print(f"[load] pool-source={name} <- {path}  (scale={scale})")
        src_dfs.append(remap_ratings(load_data(path), name, scale=scale))
        src_names.append(name)
    print(f"[load] target={tgt_name} <- {target_path}  (scale={scale})")
    target_df = remap_ratings(load_data(target_path), tgt_name, scale=scale)

    src_classes: set = set()
    for df in src_dfs:
        src_classes |= set(df[TARGET_COL].unique())
    labels = sorted(src_classes | set(target_df[TARGET_COL].unique()))
    label_to_idx = {c: i for i, c in enumerate(labels)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    src_texts = pd.concat(
        [serialize_dataframe(df, compliance_mode=compliance_mode) for df in src_dfs],
        ignore_index=True)
    y_src = np.concatenate(
        [df[TARGET_COL].map(label_to_idx).to_numpy() for df in src_dfs])
    tgt_texts = serialize_dataframe(target_df, compliance_mode=compliance_mode)
    y_tgt = target_df[TARGET_COL].map(label_to_idx).to_numpy()

    # The pooled states have different schemas, so only the labels are kept.
    source_df = pd.DataFrame({TARGET_COL: np.concatenate(
        [df[TARGET_COL].to_numpy() for df in src_dfs])})

    print(f"[prep-pool] scale={scale}  labels={labels}  pool='{pool_name}' "
          f"({len(src_dfs)} states: {src_names}, {len(src_texts)} rows) -> "
          f"target={tgt_name} ({len(target_df)} rows)")
    return TransferData(
        source_df=source_df, target_df=target_df,
        src_texts=src_texts, tgt_texts=tgt_texts,
        labels=labels, label_to_idx=label_to_idx, idx_to_label=idx_to_label,
        y_src=y_src, y_tgt=y_tgt,
        src_name=pool_name, tgt_name=tgt_name,
        compliance_mode=compliance_mode, scale=scale,
    )


# -----------------------------------------------------------------------------
# Splits and class weights
# -----------------------------------------------------------------------------
def stratified_source_split(y: np.ndarray, val_frac: float = 0.15,
                            seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stratified positional (train, val) split for early stopping.

    Classes too small to stratify are merged into an ordinal neighbour for the
    split only.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))
    strat = _stratification_labels(y, 2)
    # sklearn also needs each side to hold at least one row per stratum.
    n_va = max(1, int(np.ceil(val_frac * len(y))))
    max_strata = max(1, min(n_va, len(y) - n_va))
    while len(np.unique(strat)) > max_strata:
        vals, counts = np.unique(strat, return_counts=True)
        smallest = vals[np.argmin(counts)]
        others = vals[vals != smallest]
        nearest = others[np.argmin(np.abs(others - smallest))]
        strat = np.where(strat == smallest, nearest, strat)
    tr, va = train_test_split(idx, test_size=val_frac, stratify=strat,
                              random_state=seed)
    return tr, va


def split_target_fewshot(
    y_tgt: np.ndarray, target_frac: float, test_frac: float = 0.2, seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(adapt_idx, test_idx)`` for one point on the supervision curve.

    ``test_idx`` is a stratified ``test_frac`` slice carved independently of
    ``target_frac``, so every curve point scores the same rows. ``adapt_idx`` is a
    stratified ``target_frac`` share of the remaining pool (all of it at >= 1).
    """
    y_tgt = np.asarray(y_tgt)
    n = len(y_tgt)
    full_idx = np.arange(n)

    pool_idx, test_idx = train_test_split(
        full_idx, test_size=test_frac, stratify=y_tgt, random_state=seed)

    if target_frac <= 0:
        return np.array([], dtype=int), test_idx

    if target_frac >= 1.0:
        return pool_idx.copy(), test_idx

    n_adapt = int(round(target_frac * len(pool_idx)))
    n_adapt = max(1, min(n_adapt, len(pool_idx)))
    if n_adapt >= len(pool_idx):
        return pool_idx.copy(), test_idx

    y_pool = y_tgt[pool_idx]
    strat = _fewshot_strata(y_pool, n_adapt)
    sub, _ = train_test_split(
        np.arange(len(pool_idx)), train_size=n_adapt, stratify=strat,
        random_state=seed)
    adapt_idx = pool_idx[sub]
    return adapt_idx, test_idx


def _fewshot_strata(y_pool: np.ndarray, n_adapt: int) -> np.ndarray:
    """Strata for the adaptation draw that `train_test_split` can satisfy: each
    stratum holds >= 2 rows and there are no more strata than rows on either side.
    Offending classes are merged into an ordinal neighbour for the draw only."""
    y_pool = np.asarray(y_pool)
    strat = y_pool.astype(int).copy()
    n_rest = len(y_pool) - n_adapt
    capacity = max(1, min(n_adapt, n_rest))

    def _counts(a) -> dict[int, int]:
        u, c = np.unique(a, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))

    while True:
        counts = _counts(strat)
        if len(counts) <= 1:
            return np.zeros(len(y_pool), dtype=int)
        too_small = any(k < 2 for k in counts.values())
        if not too_small and len(counts) <= capacity:
            return strat
        victim = min(counts, key=lambda c: (counts[c], c))
        present = sorted(counts)
        pos = present.index(victim)
        neighbour = present[pos + 1] if pos + 1 < len(present) else present[pos - 1]
        strat[strat == victim] = neighbour


def target_cv_folds(
    y_tgt: np.ndarray, n_folds: int = 5, seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold ``(train_idx, test_idx)`` pairs over all target rows."""
    from sklearn.model_selection import StratifiedKFold

    y_tgt = np.asarray(y_tgt)
    idx = np.arange(len(y_tgt))
    if n_folds < 2:
        raise ValueError(f"target_cv_folds needs n_folds >= 2, got {n_folds}")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(idx, y_tgt)]


def balanced_class_weights(train_labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced class weights of length n_classes; absent classes keep weight 1.0."""
    w = np.ones(n_classes, dtype=np.float32)
    present = np.unique(train_labels)
    cw = compute_class_weight(class_weight="balanced", classes=present, y=train_labels)
    for c, val in zip(present, cw):
        w[int(c)] = float(val)
    return w


# -----------------------------------------------------------------------------
# Bootstrap error bars and results logging
# -----------------------------------------------------------------------------
def bootstrap_target_std(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: "np.ndarray | None",
    labels, n_boot: int = 1000, seed: int = SEED,
) -> dict[str, float]:
    """Std of each metric over `n_boot` bootstrap resamples of the scored rows."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    acc = {k: [] for k in _METRIC_KEYS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            pr = y_proba[idx] if y_proba is not None else None
            m = compute_metrics(y_true[idx], y_pred[idx], y_proba=pr, labels=labels)
            for k in _METRIC_KEYS:
                acc[k].append(m[k])
    return {k: float(np.nanstd(acc[k])) for k in _METRIC_KEYS}


def _classes_str(classes) -> str:
    if classes is None:
        return ""
    return "|".join(str(c) for c in classes)


def _append_row(output_path: Path, row: dict) -> None:
    """Append one row under an exclusive lock, so concurrent jobs can share a CSV."""
    import fcntl

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            if not output_path.exists() or output_path.stat().st_size == 0:
                new_df.to_csv(output_path, index=False)
                return
            with open(output_path, "r", newline="") as f:
                header = f.readline().rstrip("\r\n")
            if header == ",".join(new_df.columns):
                new_df.to_csv(output_path, mode="a", header=False, index=False)
            else:  # different columns: rewrite with the union
                out = pd.concat([pd.read_csv(output_path), new_df], ignore_index=True)
                out.to_csv(output_path, index=False)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def log_transfer_result(method: str, point: dict, std: dict,
                        output_path=DEFAULT_RESULTS_PATH, notes: str = "",
                        source: "str | None" = None, target: "str | None" = None,
                        classes=None) -> None:
    """Append one row: point estimate plus bootstrap std, status='ok'."""
    output_path = Path(output_path)
    row = {"method": method, "fold": "mean", "notes": notes,
           "source": source if source is not None else "",
           "target": target if target is not None else "",
           "classes": _classes_str(classes), "status": "ok"}
    for k in _METRIC_KEYS:
        row[k] = point[k]
        row[f"{k}_std"] = std.get(k, float("nan"))
    _append_row(output_path, row)
    print(f"Logged transfer result for '{method}' to {output_path}")


def log_failed_transfer(method: str, output_path=DEFAULT_RESULTS_PATH, *,
                        source: "str | None" = None, target: "str | None" = None,
                        classes=None, error: str = "", notes: str = "") -> None:
    """Append a NaN row with status='FAILED', so a crashed cell is visible."""
    output_path = Path(output_path)
    note = (f"{notes} " if notes else "") + f"FAILED: {error}"
    row = {"method": method, "fold": "mean", "notes": note.strip(),
           "source": source if source is not None else "",
           "target": target if target is not None else "",
           "classes": _classes_str(classes), "status": "FAILED"}
    for k in _METRIC_KEYS:
        row[k] = float("nan")
        row[f"{k}_std"] = float("nan")
    _append_row(output_path, row)
    print(f"Recorded FAILED transfer row for '{method}' in {output_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def add_transfer_args(parser) -> None:
    """Flags shared by every cross-state script."""
    parser.add_argument("--source", type=Path, required=False, default=None,
                        help="Source-state raw CSV (not needed with --pool-sources).")
    parser.add_argument("--target", type=Path, required=True,
                        help="Target-state raw CSV.")
    parser.add_argument("--pool-sources", action="append", default=[], metavar="NAME=PATH",
                        help="Leave-one-state-out: pool these source states (repeatable) "
                             "in place of --source.")
    parser.add_argument("--pool-name", default="LOSO",
                        help="Name of the pooled source in method tags (default LOSO).")
    add_output_args(parser)
    parser.add_argument("--src-name", default="src",
                        help="Source state code (selects its rating map).")
    parser.add_argument("--tgt-name", default="tgt",
                        help="Target state code.")
    parser.add_argument("--compliance", choices=("verbose", "summary", "abnormal_only"),
                        default="verbose", help="Rendering of the compliance section.")
    parser.add_argument("--rating-scale", choices=("3star", "5star"), default="3star",
                        help="'5star': native {1..5} (GA has no map). '3star': GA native, "
                             "other states collapsed 5->3.")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Early-stopping fraction of each training phase (default 0.15).")
    parser.add_argument("--target-frac", type=float, default=0.0,
                        help="p/100: share of the non-test target pool used for "
                             "adaptation (default 0.0, zero-shot).")
    parser.add_argument("--target-test-frac", type=float, default=0.2,
                        help="Fixed stratified target test fraction, the same at "
                             "every --target-frac (default 0.2).")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for error bars (default 1000).")
    parser.add_argument("--seed", type=int, default=SEED)
    add_verbosity_arg(parser)


def _resolve_transfer_output(args, script: str):
    """Set ``args.output`` and start logging for a cross-state (or --within-cv) run."""
    from utils import resolve_output, setup_logging
    tgt = (getattr(args, "tgt_name", "") or "tgt").lower()
    if getattr(args, "within_cv", 0):
        sub, fname, stem = ("within_state",
                            f"experiment_within_state_{tgt}_results.csv",
                            f"{script}_{tgt}")
    elif getattr(args, "rating_scale", "3star") == "5star":
        sub, fname, stem = ("5_star/loso_few_shot",
                            f"experiment_loso_few_shot_5star_{tgt}_results.csv",
                            f"{script}_{tgt}")
    else:
        sub, fname, stem = ("3_star/loso_ga_few_shot",
                            "experiment_loso_ga_few_shot_3star_results.csv",
                            f"{script}_ga")
    args.output = resolve_output(args, sub, fname)
    setup_logging(args, sub, stem)
    return args.output
