"""Shared data and evaluation plumbing for the cross-state transfer experiments.

Every cross-state method loads its data, splits it, weights its classes and
reports its metrics through this module.

`RATING_MAPS` / `remap_ratings` reconcile state rating scales and raise on an
unmapped state. `build_texts` and `load_and_prepare` produce the serialized text
view and the canonical union label space. `stratified_source_split` carves the
early-stopping slice, `split_target_fewshot` implements the supervision-curve
protocol, `bootstrap_target_std` produces error bars, and `log_transfer_result`
appends results under one schema.

Depends only on numpy, pandas and scikit-learn, never torch or transformers.
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
# Keyed by --src-name / --tgt-name. GA is natively 3-level; every other state
# takes the 5->3 collapse. Rating 0 means unrated, so it has no key and is
# dropped.
_COLLAPSE_5_TO_3 = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3}

RATING_MAPS: dict[str, dict[int, int]] = {
    "WI": dict(_COLLAPSE_5_TO_3),   # 0 deliberately absent -> dropped
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

# A state missing from RATING_MAPS is an import-time failure. When adding a
# state, add it both above and here.
_REQUIRED_STATES = frozenset({"WI", "NC", "CA", "GA",
                              "CO", "KY", "MD", "MT", "NE", "OK", "SC", "WA"})
assert _REQUIRED_STATES <= set(RATING_MAPS), (
    f"RATING_MAPS is missing required states {sorted(_REQUIRED_STATES - set(RATING_MAPS))}; "
    f"known: {sorted(RATING_MAPS)}"
)

# 5-star scale: identity maps for the eleven natively-5-level states, GA absent.
# Written as explicit dicts so the drop-0/missing/out-of-scale guard still
# applies and GA's exclusion is enforced rather than incidental.
_IDENTITY_5 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

RATING_MAPS_5STAR: dict[str, dict[int, int]] = {
    "WI": dict(_IDENTITY_5),   # 0 deliberately absent -> dropped
    "NC": dict(_IDENTITY_5),
    "CA": dict(_IDENTITY_5),
    # GA deliberately absent: a 5-star GA run raises.
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

# GA intentionally not included.
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
    """Coerce qr_rating onto the scale selected by `scale`.

    Rows whose rating is 0, missing, or not a key in the state's map are dropped.
    An unmapped state raises; pass ``allow_identity=True`` to opt into the native
    scale instead. GA has no 5-star map, so a 5-star GA run raises by design.
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

    fmap = {float(k): v for k, v in mapping.items()}  # match float-coerced keys
    keep = ratings.isin(list(fmap.keys()))
    n_drop = int((~keep).sum())
    out = out.loc[keep].copy()
    out[TARGET_COL] = ratings.loc[keep].map(fmap).astype(int).to_numpy()
    dist = out[TARGET_COL].value_counts().sort_index().to_dict()
    print(f"  [{state}] remapped qr_rating via {mapping}; dropped {n_drop} rows "
          f"(0 / missing / out-of-scale). new dist: {dist}")
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Text views (one string per row) — shared by source and target
# -----------------------------------------------------------------------------
def build_texts(df: pd.DataFrame, mode: str = "textualized_full",
                compliance_mode: str = "verbose") -> pd.Series:
    """Render a dataframe to one input string per row. The same function is used
    for both source and target so the two states are textualized identically."""
    if mode == "curriculum_only":
        if "curriculum" not in df.columns:
            raise KeyError("Expected a 'curriculum' column for curriculum_only mode.")
        return df["curriculum"].fillna("").astype(str)
    if mode == "textualized_full":
        return serialize_dataframe(df, compliance_mode=compliance_mode)
    raise ValueError(f"Unknown text mode: {mode!r}")


# -----------------------------------------------------------------------------
# Bundle returned by load_and_prepare — the single source of truth for "the data"
# -----------------------------------------------------------------------------
@dataclass
class TransferData:
    source_df: pd.DataFrame
    target_df: pd.DataFrame
    src_texts: pd.Series          # serialized strings, source order
    tgt_texts: pd.Series          # serialized strings, target order
    labels: list                  # canonical (union) label space, sorted
    label_to_idx: dict
    idx_to_label: dict
    y_src: np.ndarray             # int-indexed source labels
    y_tgt: np.ndarray             # int-indexed target labels (EVAL ONLY)
    src_name: str
    tgt_name: str
    text_mode: str
    compliance_mode: str
    scale: str = "3star"

    @property
    def n_classes(self) -> int:
        return len(self.labels)


def load_and_prepare(
    source_path, target_path, src_name: str, tgt_name: str,
    text_mode: str = "textualized_full", compliance_mode: str = "verbose",
    scale: str = "3star",
) -> TransferData:
    """Load, rating-reconcile, serialize, and build the union label space.

    Source and target need not share feature columns; serialization adapts to
    whatever each file contains. `scale` picks the rating-map set.
    """
    print(f"[load] source={src_name} <- {source_path}  (scale={scale})")
    source_df = remap_ratings(load_data(source_path), src_name, scale=scale)
    print(f"[load] target={tgt_name} <- {target_path}  (scale={scale})")
    target_df = remap_ratings(load_data(target_path), tgt_name, scale=scale)

    # Canonical label space = union of both states (after reconciliation).
    labels = sorted(set(source_df[TARGET_COL].unique()) | set(target_df[TARGET_COL].unique()))
    label_to_idx = {c: i for i, c in enumerate(labels)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    src_texts = build_texts(source_df, text_mode, compliance_mode)
    tgt_texts = build_texts(target_df, text_mode, compliance_mode)
    y_src = source_df[TARGET_COL].map(label_to_idx).to_numpy()
    y_tgt = target_df[TARGET_COL].map(label_to_idx).to_numpy()

    print(f"[prep] scale={scale}  labels={labels}  source rows={len(source_df)}  "
          f"target rows={len(target_df)}  text_mode={text_mode}")
    return TransferData(
        source_df=source_df, target_df=target_df,
        src_texts=src_texts, tgt_texts=tgt_texts,
        labels=labels, label_to_idx=label_to_idx, idx_to_label=idx_to_label,
        y_src=y_src, y_tgt=y_tgt,
        src_name=src_name, tgt_name=tgt_name,
        text_mode=text_mode, compliance_mode=compliance_mode, scale=scale,
    )



def parse_pool_sources(items: "list[str]") -> "list[tuple[str, str]]":
    """Parse repeated ``--pool-sources NAME=PATH`` values into (name, path) tuples."""
    out = []
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--pool-sources expects NAME=PATH, got {item!r}")
        out.append((name, path))
    return out


def load_and_prepare_pooled(
    source_specs: "list[tuple[str, str]]", tgt_name: str, target_path,
    text_mode: str = "textualized_full", compliance_mode: str = "verbose",
    scale: str = "3star", pool_name: str = "LOSO",
) -> TransferData:
    """Pool many source states into one training set against a held-out target.

    Each source is loaded, reconciled to ``scale`` and serialized; their texts and
    labels are concatenated into one bag. Returns a standard ``TransferData`` whose
    source is the whole pool, so every model runner consumes it unchanged.

    The embedding cache keys on ``src_name`` and row count, so a different
    held-out target is a cache miss rather than a silent reuse.
    """
    src_dfs, src_names = [], []
    for name, path in source_specs:
        print(f"[load] pool-source={name} <- {path}  (scale={scale})")
        src_dfs.append(remap_ratings(load_data(path), name, scale=scale))
        src_names.append(name)
    print(f"[load] target={tgt_name} <- {target_path}  (scale={scale})")
    target_df = remap_ratings(load_data(target_path), tgt_name, scale=scale)

    # Canonical union label space over EVERY pooled source + the target.
    src_classes: set = set()
    for df in src_dfs:
        src_classes |= set(df[TARGET_COL].unique())
    labels = sorted(src_classes | set(target_df[TARGET_COL].unique()))
    label_to_idx = {c: i for i, c in enumerate(labels)}
    idx_to_label = {i: c for c, i in label_to_idx.items()}

    src_texts = pd.concat(
        [build_texts(df, text_mode, compliance_mode) for df in src_dfs],
        ignore_index=True)
    y_src = np.concatenate(
        [df[TARGET_COL].map(label_to_idx).to_numpy() for df in src_dfs])
    tgt_texts = build_texts(target_df, text_mode, compliance_mode)
    y_tgt = target_df[TARGET_COL].map(label_to_idx).to_numpy()

    # source_df is metadata only (used for len()); keep it light -- a 1-col frame
    # of the pooled labels avoids a schema-union NaN blow-up across states.
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
        text_mode=text_mode, compliance_mode=compliance_mode, scale=scale,
    )


# -----------------------------------------------------------------------------
# Source train/val split (early stopping) + class weights
# -----------------------------------------------------------------------------
def stratified_source_split(y: np.ndarray, val_frac: float = 0.15,
                            seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Stratified positional split into (train, val) for early stopping.

    Classes with fewer than 2 members are folded into their nearest ordinal
    neighbour FOR THE SPLIT ONLY; returned indices carry the true classes. No-op
    when every class has at least 2 members.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))
    strat = _stratification_labels(y, 2)
    # sklearn also requires both split sides to hold at least n_strata rows,
    # which the folding above does not guarantee for tiny inputs. Fold the
    # smallest stratum into its nearest ordinal neighbour until the stratum count
    # fits the smaller side. The caller's labels keep their true classes.
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
    """Carve the target into a fixed test set plus a p% adaptation draw.

    Returns ``(adapt_idx, test_idx)`` as positional index arrays.

    ``test_idx`` is a seeded, rating-stratified ``test_frac`` slice of the whole
    target, carved first and independently of ``target_frac``, so every point on
    the curve scores the same rows.

    ``target_frac`` is p/100 as a fraction OF THE ADAPTATION POOL: ``<= 0`` gives
    an empty draw, a value in (0, 1) gives ``round(target_frac * len(pool))``
    stratified rows, and ``>= 1`` gives the whole pool.

    At ``test_frac=0.2`` this yields 16/32/48/64/80% of the target for
    p=20/40/60/80/100, with a 20% scored block throughout.
    """
    y_tgt = np.asarray(y_tgt)
    n = len(y_tgt)
    full_idx = np.arange(n)

    # Escape hatch: test_frac >= 1.0 scores EVERY target row with an empty
    # adaptation draw. The drivers do not use it -- every curve point scores the
    # same fixed test set -- but it remains a valid standalone CLI setting.
    if test_frac >= 1.0:
        return np.array([], dtype=int), full_idx

    # 1) Fixed test set: stratified, seeded, independent of target_frac. This is
    # the block every curve point is scored on.
    pool_idx, test_idx = train_test_split(
        full_idx, test_size=test_frac, stratify=y_tgt, random_state=seed)

    # 2) Zero-shot: no adaptation rows, but the SAME held-out test set.
    if target_frac <= 0:
        return np.array([], dtype=int), test_idx

    # 3) p% adaptation draw, taken from the POOL (not the whole target).
    if target_frac >= 1.0:
        # p=100 hold-out point: the entire adaptation pool.
        return pool_idx.copy(), test_idx

    n_adapt = int(round(target_frac * len(pool_idx)))
    n_adapt = max(1, min(n_adapt, len(pool_idx)))
    if n_adapt >= len(pool_idx):
        return pool_idx.copy(), test_idx

    y_pool = y_tgt[pool_idx]
    # Rare-class safety: a pool class with a single member cannot be stratified
    # into both sides of the draw. Fold such classes into their nearest ordinal
    # neighbour FOR THE DRAW ONLY (true labels are untouched) — same convention
    # as utils._stratification_labels / stratified_source_split.
    strat = _fewshot_strata(y_pool, n_adapt)
    sub, _ = train_test_split(
        np.arange(len(pool_idx)), train_size=n_adapt, stratify=strat,
        random_state=seed)
    adapt_idx = pool_idx[sub]
    return adapt_idx, test_idx


def _fewshot_strata(y_pool: np.ndarray, n_adapt: int) -> np.ndarray:
    """Stratification labels for the adaptation draw that `train_test_split` can
    satisfy.

    It requires every stratum to hold >= 2 members and the stratum count to be
    <= min(train_size, test_size); small draws on small states violate both.
    Offending classes are merged into their nearest ordinal neighbour, for the
    draw only.
    """
    y_pool = np.asarray(y_pool)
    strat = y_pool.astype(int).copy()
    n_rest = len(y_pool) - n_adapt
    capacity = max(1, min(n_adapt, n_rest))

    def _counts(a) -> dict[int, int]:
        u, c = np.unique(a, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))

    # Merge along the ordinal scale until every stratum is viable.
    while True:
        counts = _counts(strat)
        if len(counts) <= 1:
            # One stratum left: an unstratified draw is the only option.
            return np.zeros(len(y_pool), dtype=int)
        too_small = any(k < 2 for k in counts.values())
        if not too_small and len(counts) <= capacity:
            return strat
        # Merge the smallest stratum into its nearest ordinal neighbour.
        victim = min(counts, key=lambda c: (counts[c], c))
        present = sorted(counts)
        pos = present.index(victim)
        neighbour = present[pos + 1] if pos + 1 < len(present) else present[pos - 1]
        strat[strat == victim] = neighbour


def target_cv_folds(
    y_tgt: np.ndarray, n_folds: int = 5, seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold over the whole target, for the full-target CV endpoint.

    Returns ``(train_idx, test_idx)`` pairs. Each row appears in exactly one
    ``test_idx``, so concatenating the held-out predictions scores every target
    row out-of-fold while all target labels are used for training across folds.

    Uses the shared seed, so folds line up with the within-state ones.
    """
    from sklearn.model_selection import StratifiedKFold

    y_tgt = np.asarray(y_tgt)
    idx = np.arange(len(y_tgt))
    if n_folds < 2:
        raise ValueError(f"target_cv_folds needs n_folds >= 2, got {n_folds}")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(idx, y_tgt)]


def balanced_class_weights(train_labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Length-n_classes 'balanced' weights from the source train labels. Classes
    absent from the train split keep weight 1.0 (no CE term, and
    compute_class_weight would raise on a class missing from y). Returns numpy so
    this module stays torch-free; callers wrap it in a tensor."""
    w = np.ones(n_classes, dtype=np.float32)
    present = np.unique(train_labels)
    cw = compute_class_weight(class_weight="balanced", classes=present, y=train_labels)
    for c, val in zip(present, cw):
        w[int(c)] = float(val)
    return w


# -----------------------------------------------------------------------------
# Bootstrap error bars on the TARGET set (uniform across all experiments)
# -----------------------------------------------------------------------------
def bootstrap_target_std(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: "np.ndarray | None",
    labels, n_boot: int = 1000, seed: int = SEED,
) -> dict[str, float]:
    """Resample target rows with replacement n_boot times; return the std of
    each metric. Isolates evaluation uncertainty and works identically for every
    method (trainable, zero-shot, retrieval), so all error bars are comparable."""
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


# -----------------------------------------------------------------------------
# Results logging (one row per method, carrying the per-fold mean)
# -----------------------------------------------------------------------------
def _classes_str(classes) -> str:
    """Serialize a label space to a CSV-safe, parseable string: '1|2|3'."""
    if classes is None:
        return ""
    return "|".join(str(c) for c in classes)


def _append_row(output_path: Path, row: dict) -> None:
    """Append one row, safe against concurrent writers.

    An exclusive ``flock`` on a sidecar file serializes writers; while held, the
    on-disk header is re-checked:

      - file absent or empty     write header + row
      - header matches           append the row
      - header differs           read, union, rewrite
    """
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
            else:  # legacy/mixed schema: preserve old behavior (union of columns)
                out = pd.concat([pd.read_csv(output_path), new_df], ignore_index=True)
                out.to_csv(output_path, index=False)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def log_transfer_result(method: str, point: dict, std: dict,
                        output_path=DEFAULT_RESULTS_PATH, notes: str = "",
                        source: "str | None" = None, target: "str | None" = None,
                        classes=None) -> None:
    """Append one fold=='mean' row: point estimate plus bootstrap std.

    Also records `source`, `target`, the realized `classes` and status='ok'.
    """
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
    """Record a fold=='mean' row with NaN metrics and status='FAILED'.

    A crashed cell that wrote nothing would be indistinguishable from one that
    was never run.
    """
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
# Shared CLI surface (the transfer_plm-style dataset chooser)
# -----------------------------------------------------------------------------
def add_transfer_args(parser) -> None:
    """Attach the dataset-selection flags shared by every transfer experiment.
    Experiment-specific flags (--models and similar) are added by each script
    after calling this."""
    parser.add_argument("--source", type=Path, required=False, default=None,
                        help="RAW source-state CSV (labeled; used for training). "
                             "Optional when --pool-sources is given (LOSO mode).")
    parser.add_argument("--target", type=Path, required=True,
                        help="RAW target-state CSV (labels used ONLY for final scoring).")
    # Leave-one-state-out (LOSO): pool MANY source states into one
    # training set and score the held-out --target. Repeatable NAME=PATH; when
    # present it REPLACES --source. The pooled 'source' is tagged with --pool-name
    # (default LOSO) so method tags read e.g. xfer_tabnet_5star_tgt0_LOSO2WI.
    parser.add_argument("--pool-sources", action="append", default=[], metavar="NAME=PATH",
                        help="LOSO: pool these source states (repeatable NAME=PATH) into "
                             "one training set, held-out target = --target. Replaces --source.")
    parser.add_argument("--pool-name", default="LOSO",
                        help="Short tag for the pooled source in method names (default LOSO).")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH,
                        help=f"Results CSV to append to (default: {DEFAULT_RESULTS_PATH}).")
    parser.add_argument("--src-name", default="src",
                        help="Short tag for the source state (keys RATING_MAPS + method name).")
    parser.add_argument("--tgt-name", default="tgt",
                        help="Short tag for the target state.")
    parser.add_argument("--text-mode", choices=("textualized_full", "curriculum_only"),
                        default="textualized_full", help="Row text view (default textualized_full).")
    parser.add_argument("--compliance", choices=("verbose", "summary", "abnormal_only"),
                        default="verbose", help="Compliance rendering for textualized_full.")
    parser.add_argument("--rating-scale", choices=("3star", "5star"), default="3star",
                        help="Rating reconciliation scale: '3star' (default; WI/NC/CA "
                             "5->3 collapse + GA, all four states) or '5star' (native "
                             "WI/NC/CA {1..5}, GA excluded -> a GA pair errors). Passed "
                             "into load_and_prepare and the method-tag infix.")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Fraction of SOURCE held out for early stopping (default 0.15).")
    # Target supervision curve. Available to EVERY transfer script
    # (PLM, LLM, tabular) so the three blocks share one split protocol.
    parser.add_argument("--target-frac", type=float, default=0.0,
                        help="Fraction of the WHOLE target folded into training "
                             "(0.0 zero-shot / 0.2 / 0.8). Drawn from a pool "
                             "disjoint from the fixed test set (default 0.0).")
    parser.add_argument("--target-test-frac", type=float, default=0.2,
                        help="Fixed held-out target fraction for scoring, identical "
                             "across all --target-frac so the 0/20/80 blocks are "
                             "directly comparable (default 0.2).")
    # Full-target CV. 0 keeps the ordinary single-split behaviour.
    parser.add_argument("--target-cv-folds", type=int, default=0,
                        help="Full-target CV: 0 (default) = single-split "
                             "behavior using --target-frac/--target-test-frac; >0 = "
                             "stratified K-fold over the WHOLE target with the "
                             "source folded into every fold's training set (the "
                             "'100%%' endpoint). When >0, --target-frac/"
                             "--target-test-frac are ignored for the split; pass "
                             "--target-frac 1.0 so the method tag reads tgt100.")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Target bootstrap resamples for error bars (default 1000).")
    parser.add_argument("--seed", type=int, default=SEED)
    add_verbosity_arg(parser)