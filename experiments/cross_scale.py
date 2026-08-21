"""Cross-rubric transfer, tabular stage: regressors scored by rank.

Georgia rates on {1,2,3} and the other eleven states on {1..5}, with no
correspondence between the scales. A regressor is trained on the source's native
rating, predicts a continuous score on the target, and is scored by concordance
index against the target's native labels. No bridge, cutpoints or rounding appear
anywhere in this path.

The deep model roster is in `cross_scale_deep`.

Three modes:
  transfer   --source/--src-name with --target/--tgt-name, over the p-curve
  within     --within/--state, 5-fold CV; the ceiling transfer cells read against
  ablation   transfer with --coarsen-source, simulating a 3-level source from a
             real 5-level one

    python cross_scale.py --source data/ga_records_cleaned_full.csv --src-name GA \\
        --target data/mt_records_cleaned_full.csv --tgt-name MT \\
        --target-frac 0.0 --output results/<date>/cross_scale/results.csv
    python cross_scale.py --within data/mt_records_cleaned_full.csv --state MT --output ...
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from baselines_ml import REGRESSOR_REGISTRY, _assert_regressor_parity
from ranking_metrics import (
    RANKING_METRIC_KEYS,
    bootstrap_ranking_std,
    compute_ranking_metrics,
)
from transfer_common import (
    SEED,
    TARGET_COL,
    _COLLAPSE_5_TO_3,
    _append_row,
    build_texts,
    remap_ratings,
    split_target_fewshot,
)
from utils import get_folds, load_data

# Repo-local credentials (TABPFN_TOKEN) if keys.json is present. Env always
# wins, a missing file is a no-op, and no value is ever printed. This is what
# lets `tabpfn` run without remembering an export before every submit.
try:
    from keys import load_keys
    load_keys()
except ImportError:
    pass

# Fail fast at import if a regression twin has drifted from its classifier or
# from shap_xgb's copy — cheaper than discovering it in a tree of results.
_assert_regressor_parity()

MODELS = ("dummy", "ridge", "rf", "xgb")
# Stages 2-4 live in cross_scale_deep (imported lazily: it pulls torch, and the
# tabular path must stay runnable on a CPU box with no deep stack installed).
# The roster is regressors only. An ordinal classifier cannot join it: its loss
# consumes class indices rather than the z-scored target every model here trains
# on, and its head width is tied to one rubric's K. Ordinal-native cross-rubric
# transfer is named future work.
DEEP_MODEL_NAMES = ("tabnet", "tabpfn", "mbert",
                    "qwen", "qwen_rag", "qwen_lora")
ALL_MODELS = MODELS + DEEP_MODEL_NAMES


def rating_stats(y_ref) -> "tuple[float, float]":
    """(mean, sd) defining one state's z-score transform.

    Computed once per state and reused at every p, so the pooled and sequential
    paths share one definition of a standardized rating.

    Source statistics come from all its rows, which are never scored. Target
    statistics come from the fixed non-test pool only, so no test label enters
    the transform and the transform does not drift with p.
    """
    y_ref = np.asarray(y_ref, dtype=float)
    sd = float(y_ref.std())
    return float(y_ref.mean()), (sd if sd > 1e-12 else 1.0)


def zscore(y, stats: "tuple[float, float]") -> np.ndarray:
    """Apply a `rating_stats` transform. Strictly monotone, so within-state
    ordering and any single-domain c-index are unchanged."""
    mean, sd = stats
    return (np.asarray(y, dtype=float) - mean) / sd


# -----------------------------------------------------------------------------
# Loading — NATIVE labels on both sides, always
# -----------------------------------------------------------------------------
def load_native(path, state: str) -> pd.DataFrame:
    """Load one state at its own granularity: GA on {1,2,3}, others on {1..5}.

    Goes through `remap_ratings` with the identity map for that state, which
    keeps the drop-0/missing guard and the hard error on an unknown state.
    """
    scale = "3star" if state.upper() == "GA" else "5star"
    return remap_ratings(load_data(path), state.upper(), scale=scale)


def coarsen_5_to_3(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Collapse a 5-level source to 3 levels, for the granularity ablation only.

    Never applied to a target and never to evaluation; the scored labels are
    always the target's native ones.
    """
    out = df.copy()
    out[TARGET_COL] = out[TARGET_COL].map(_COLLAPSE_5_TO_3).astype(int)
    print(f"  [{state}] ABLATION: source coarsened 5->3 for training only; "
          f"dist {out[TARGET_COL].value_counts().sort_index().to_dict()}")
    return out


def embed(df: pd.DataFrame, state: str, compliance: str, cache_dir: Path) -> np.ndarray:
    """MiniLM embedding of the serialized rows, the shared feature space.

    Raw numerics cannot be used across states, whose feature columns differ in
    both name and number. The cache is keyed by state and a content hash of the
    texts.
    """
    from transfer_tabular import embed_state
    texts = build_texts(df, "textualized_full", compliance)
    return embed_state(texts, state.upper(), "textualized_full", compliance, cache_dir)


# -----------------------------------------------------------------------------
# Fit / predict
# -----------------------------------------------------------------------------
def fit_predict(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray) -> np.ndarray:
    """Fit one regressor and return continuous scores on X_test.

    Scores are never rounded, thresholded or mapped onto the target's scale; they
    are consumed only by rank statistics.
    """
    model, pre = REGRESSOR_REGISTRY[model_name](0)
    if pre.get("scale"):
        # Embeddings are dense, finite and NaN-free, so the numeric branch of
        # build_preprocessor reduces to exactly this. Fit on train only.
        sc = StandardScaler().fit(X_train)
        X_train, X_test = sc.transform(X_train), sc.transform(X_test)
    model.fit(X_train, y_train.astype(float))
    return np.asarray(model.predict(X_test), dtype=float)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def log_ranking_result(method: str, point: dict, std: dict, output_path: Path,
                       *, source: str, target: str, notes: str = "",
                       extra: "dict | None" = None) -> None:
    """Append one fold=='mean' row in the ranking schema.

    Reuses `transfer_common._append_row` for the flock-protected, schema-union
    append rather than re-implementing concurrency-safe CSV writing. The metric
    columns differ from the classification suite's (`c_index` etc. instead of
    `qwk` etc.), so these land in their OWN results tree and are never mixed
    into a LOSO CSV.
    """
    row = {"method": method, "fold": "mean", "notes": notes,
           "source": source, "target": target, "status": "ok"}
    row.update(extra or {})
    for k in RANKING_METRIC_KEYS:
        row[k] = point[k]
        row[f"{k}_std"] = std.get(k, float("nan"))
    _append_row(Path(output_path), row)
    print(f"  logged '{method}' -> {output_path}")


def log_failed(method: str, output_path: Path, *, source: str, target: str,
               error: str, extra: "dict | None" = None) -> None:
    """Record a FAILED cell so a crash leaves a row instead of a silent hole
    (the same convention transfer_common uses)."""
    row = {"method": method, "fold": "mean", "notes": f"FAILED: {error}",
           "source": source, "target": target, "status": "FAILED"}
    row.update(extra or {})
    for k in RANKING_METRIC_KEYS:
        row[k] = float("nan")
        row[f"{k}_std"] = float("nan")
    _append_row(Path(output_path), row)
    print(f"  recorded FAILED row for '{method}'")


def _dump(path: Path, tag: str, y_true, scores) -> None:
    """Per-row predictions, dumped by default. Any future rank
    statistic, decode rule, or calibration analysis becomes computable without
    re-running a single model.

    NB `path` is ALREADY the predictions directory, so the mkdir applies to it
    directly rather than to its parent.
    """
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / f"{tag}.npz",
                        y_true=np.asarray(y_true), scores=np.asarray(scores))


def _dump_safe(path: Path, tag: str, y_true, scores) -> None:
    """Dump, but NEVER let a dump problem destroy a computed result.

    The scores are already logged by the time this runs, and the .npz is a
    convenience artifact -- a full disk, a quota, or a bad path must cost that
    artifact, not the cell and not the models queued behind it.
    """
    try:
        _dump(path, tag, y_true, scores)
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] prediction dump failed for {tag} "
              f"({type(e).__name__}: {e}); result row is already logged, continuing")


# -----------------------------------------------------------------------------
# Mode 1: cross-state transfer (the 11x2 grid)
# -----------------------------------------------------------------------------
def run_transfer(args) -> None:
    src_df = load_native(args.source, args.src_name)
    tgt_df = load_native(args.target, args.tgt_name)
    resolution = "native"
    if args.coarsen_source:
        if args.src_name.upper() == "GA":
            raise SystemExit("--coarsen-source on GA is meaningless: GA is already 3-level.")
        src_df = coarsen_5_to_3(src_df, args.src_name)
        resolution = "coarsened3"

    needs_text = any(m in DEEP_MODEL_NAMES for m in args.models)
    X_src = embed(src_df, args.src_name, args.compliance, args.cache_dir)
    X_tgt = embed(tgt_df, args.tgt_name, args.compliance, args.cache_dir)
    y_src = src_df[TARGET_COL].to_numpy()
    y_tgt = tgt_df[TARGET_COL].to_numpy()
    txt_src = list(build_texts(src_df, "textualized_full", args.compliance)) if needs_text else None
    txt_tgt = list(build_texts(tgt_df, "textualized_full", args.compliance)) if needs_text else None

    # Same split protocol as the LOSO suite: a stratified 20% target test block
    # carved ONCE and scored at every p, so a c-index column is comparable down
    # the curve; --target-frac is p% of the REMAINING 80%.
    adapt_idx, test_idx = split_target_fewshot(
        y_tgt, args.target_frac, test_frac=args.target_test_frac, seed=args.seed)
    pct = int(round(args.target_frac * 100))
    y_test = y_tgt[test_idx]

    # ONE z-score transform per state, fixed across the whole p-curve, shared by
    # the pooled and sequential paths. Source stats from all its
    # rows (never scored); target stats from the fixed NON-TEST POOL, so no test
    # label ever touches training and the transform does not drift with p.
    pool_idx = np.setdiff1d(np.arange(len(y_tgt)), test_idx, assume_unique=False)
    src_stats = rating_stats(y_src)
    tgt_stats = rating_stats(y_tgt[pool_idx])
    print(f"  z-transform  src {args.src_name}: mean={src_stats[0]:.3f} sd={src_stats[1]:.3f}"
          f"   tgt {args.tgt_name} (non-test pool, n={len(pool_idx)}): "
          f"mean={tgt_stats[0]:.3f} sd={tgt_stats[1]:.3f}")

    print(f"\n[transfer] {args.src_name}->{args.tgt_name}  p={pct}%  "
          f"resolution={resolution}\n  src={len(y_src)} rows + adapt={len(adapt_idx)} "
          f"-> scored={len(y_test)} rows  "
          f"src_levels={sorted(set(y_src.tolist()))} tgt_levels={sorted(set(y_test.tolist()))}")

    # The source and target label spaces are NEVER reconciled: y_src may be
    # {1,2,3} while y_test is {1..5}. That is the entire point -- the model's
    # output range is irrelevant to a rank statistic. What DOES matter is that a
    # single regression target never MIXES the two scales, which `rating_stats`
    # and `zscore` guarantee by standardizing each state once, independently.
    extra = {"src_levels": len(set(y_src.tolist())),
             "tgt_levels": len(set(y_test.tolist())),
             "label_resolution": resolution, "target_pct": pct}
    suffix = "_c3" if args.coarsen_source else ""

    for name in args.models:
        tag = (f"xscale_{name}_reg{suffix}_tgt{pct}_"
               f"{args.src_name.upper()}2{args.tgt_name.upper()}")
        t0 = time.time()
        try:
            if name in DEEP_MODEL_NAMES:
                scores = _run_deep(name, args, X_src, y_src, txt_src,
                                   X_tgt, y_tgt, txt_tgt, adapt_idx, test_idx,
                                   src_stats, tgt_stats)
            else:
                # One-shot fit on source + p% target adaptation. Each state's
                # ratings go through ITS OWN transform, so the two rubrics land
                # on one numeric scale with no correspondence asserted between
                # their levels. Identical transform to the sequential path.
                if len(adapt_idx):
                    X_train = np.vstack([X_src, X_tgt[adapt_idx]])
                    y_train = np.concatenate([zscore(y_src, src_stats),
                                              zscore(y_tgt[adapt_idx], tgt_stats)])
                else:
                    X_train, y_train = X_src, zscore(y_src, src_stats)
                scores = fit_predict(name, X_train, y_train, X_tgt[test_idx])
            point = compute_ranking_metrics(y_test, scores)
            std = bootstrap_ranking_std(y_test, scores, args.n_bootstrap, args.seed)
        except Exception as e:  # noqa: BLE001 — a failed cell must still be recorded
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")
            log_failed(tag, args.output, source=args.src_name.upper(),
                       target=args.tgt_name.upper(), error=f"{type(e).__name__}: {e}",
                       extra=extra)
            continue
        print(f"  {name:10s} c={point['c_index']:.4f}±{std['c_index']:.4f}  "
              f"D={point['somers_d']:+.4f}  c_w={point['c_index_weighted']:.4f}  "
              f"rho={point['spearman']:+.4f}  ({time.time() - t0:.1f}s)")
        log_ranking_result(tag, point, std, args.output,
                           source=args.src_name.upper(), target=args.tgt_name.upper(),
                           notes=tag, extra=extra)
        if args.dump_predictions:
            _dump_safe(Path(args.output).parent / "predictions", tag, y_test, scores)


def _run_deep(name, args, X_src, y_src, txt_src, X_tgt, y_tgt, txt_tgt,
              adapt_idx, test_idx, src_stats, tgt_stats) -> np.ndarray:
    """Dispatch one Stage 2-4 cell.

    Builds the two phases the sequential regime needs: pretrain on the whole
    source, fine-tune on the p% target adaptation draw. At p=0 there is no second
    phase, which is what makes the zero-shot row comparable across families.

    Each Phase carries BOTH the z-scored training target (`y`, from that state's
    own transform — the SAME one the pooled path uses) and the raw integer levels
    (`y_raw`), because early-stopping splits must stratify on real classes.
    """
    from cross_scale_deep import DEEP_MODELS, Phase

    spec = DEEP_MODELS[name]
    use_text = spec["input"] == "text"
    src_X = txt_src if use_text else X_src
    tgt_X = txt_tgt if use_text else X_tgt

    def _take(seq, idx):
        return [seq[i] for i in idx] if use_text else seq[idx]

    pre = Phase(X=src_X, y=zscore(y_src, src_stats), y_raw=y_src)
    ft = (Phase(X=_take(tgt_X, adapt_idx),
                y=zscore(y_tgt[adapt_idx], tgt_stats), y_raw=y_tgt[adapt_idx])
          if len(adapt_idx) else None)
    test_X = _take(tgt_X, test_idx)

    out_dir = (Path(args.artifact_root) / "cross_scale" /
               f"{args.src_name.upper()}2{args.tgt_name.upper()}_tgt"
               f"{int(round(args.target_frac * 100))}_{name}")
    kw: dict = {"seed": args.seed}
    if not use_text:
        return spec["fn"](pre, ft, test_X, **kw)
    kw["out_dir"] = out_dir
    if spec.get("needs_emb"):
        # RAG needs retrieval embeddings for the medoid selector, aligned with
        # each phase's own rows.
        kw["pre_emb"] = X_src
        kw["ft_emb"] = X_tgt[adapt_idx] if len(adapt_idx) else None
    if name.startswith("qwen"):
        kw["model_name"] = args.llm_model
    try:
        return spec["fn"](pre, ft, test_X, **kw)
    finally:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Mode 2: within-state ceiling
# -----------------------------------------------------------------------------
def run_within(args) -> None:
    """5-fold CV inside one state — the oracle every transfer cell reads against.

    Per-fold c-index is computed and then AVERAGED; out-of-fold scores are NOT
    pooled into one ranking. Each fold's regressor has its own arbitrary output
    scale, so ranking rows across folds would compare incomparable scores and
    depress the ceiling for a purely artefactual reason. Averaging per-fold
    values also matches how the rest of the suite aggregates.
    """
    needs_text = any(m in DEEP_MODEL_NAMES for m in args.models)
    df = load_native(args.within, args.state)
    X = embed(df, args.state, args.compliance, args.cache_dir)
    y = df[TARGET_COL].to_numpy()
    txt = list(build_texts(df, "textualized_full", args.compliance)) if needs_text else None
    # Default to the SAME cached native-scale folds experiment_within_state uses,
    # so the ranking ceiling and the QWK oracle are computed on identical splits
    # and the two tables describe the same partition of each state.
    folds_path = args.folds or Path("fold_indices") / f"{args.state.lower()}_native_folds.json"
    folds = get_folds(df, folds_path=folds_path)
    print(f"\n[within] {args.state}  {len(y)} rows  levels={sorted(set(y.tolist()))}  "
          f"{len(folds)} folds")

    extra = {"src_levels": len(set(y.tolist())), "tgt_levels": len(set(y.tolist())),
             "label_resolution": "native", "target_pct": 100}

    for name in args.models:
        tag = f"xscale_{name}_reg_within_{args.state.upper()}"
        per_fold: list[dict] = []
        t0 = time.time()
        try:
            for fold in folds:
                tr, va = fold["train_idx"], fold["val_idx"]
                if name in DEEP_MODEL_NAMES:
                    # One state, one scale, so the ceiling is a SINGLE phase --
                    # there is no cross-scale adaptation to sequence.
                    from cross_scale_deep import DEEP_MODELS, Phase
                    spec = DEEP_MODELS[name]
                    use_text = spec["input"] == "text"
                    src = txt if use_text else X
                    take = (lambda s, i: [s[j] for j in i]) if use_text else (
                        lambda s, i: s[i])
                    kw: dict = {"seed": args.seed}
                    out_dir = (Path(args.artifact_root) / "cross_scale" /
                               f"within_{args.state.upper()}_f{fold['fold']}_{name}")
                    if use_text:
                        kw["out_dir"] = out_dir
                        if spec.get("needs_emb"):
                            kw["pre_emb"] = X[tr]
                        if name.startswith("qwen"):
                            kw["model_name"] = args.llm_model
                    try:
                        scores = spec["fn"](Phase(X=take(src, tr), y=y[tr].astype(float)),
                                            None, take(src, va), **kw)
                    finally:
                        import shutil
                        shutil.rmtree(out_dir, ignore_errors=True)
                else:
                    scores = fit_predict(name, X[tr], y[tr].astype(float), X[va])
                per_fold.append(compute_ranking_metrics(y[va], scores))
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: FAILED ({type(e).__name__}: {e})")
            log_failed(tag, args.output, source=args.state.upper(),
                       target=args.state.upper(), error=f"{type(e).__name__}: {e}",
                       extra=extra)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            point = {k: float(np.nanmean([f[k] for f in per_fold]))
                     for k in RANKING_METRIC_KEYS}
            std = {k: float(np.nanstd([f[k] for f in per_fold]))
                   for k in RANKING_METRIC_KEYS}
        print(f"  {name:6s} c={point['c_index']:.4f}±{std['c_index']:.4f}  "
              f"D={point['somers_d']:+.4f}  rho={point['spearman']:+.4f}  "
              f"({time.time() - t0:.1f}s)")
        log_ranking_result(tag, point, std, args.output,
                           source=args.state.upper(), target=args.state.upper(),
                           notes=f"{tag} (fold-mean, {len(folds)} folds)", extra=extra)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="experiment_cross_scale: cross-rubric transfer scored as ranking.")
    p.add_argument("--source", type=Path, help="Source-state CSV (transfer mode).")
    p.add_argument("--src-name", default="", help="Source state code, e.g. GA.")
    p.add_argument("--target", type=Path, help="Target-state CSV (transfer mode).")
    p.add_argument("--tgt-name", default="", help="Target state code, e.g. MT.")
    p.add_argument("--within", type=Path, help="State CSV for the within-state ceiling.")
    p.add_argument("--state", default="", help="State code for --within.")
    p.add_argument("--folds", type=Path, default=None,
                   help="Cached fold indices for --within (default: utils' convention).")
    p.add_argument("--coarsen-source", action="store_true",
                   help="Granularity ablation: collapse the 5-level SOURCE to 3 levels "
                        "for training only. Target labels are untouched.")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(ALL_MODELS),
                   help="Stage 1 (CPU): dummy ridge rf xgb. Stage 2: tabnet tabpfn. "
                        "Stage 3: mbert mbert_corn. Stage 4: qwen qwen_rag qwen_lora. "
                        "Anything past Stage 1 needs a GPU.")
    p.add_argument("--llm-model", default="",
                   help="Qwen checkpoint for the Stage 4 cells (default: "
                        "transfer_llm_cls.LLM_CLS_MODEL, i.e. $LLM_CLS_MODEL).")
    p.add_argument("--artifact-root", default=None,
                   help="Scratch root for deep-model run dirs (default: "
                        "$CCQ_ARTIFACT_ROOT, else utils.ARTIFACT_ROOT). "
                        "Each cell's dir is deleted when it finishes.")
    p.add_argument("--target-frac", type=float, default=0.0,
                   help="p/100, as a fraction of the non-test target pool (default 0 = zero-shot).")
    p.add_argument("--target-test-frac", type=float, default=0.2,
                   help="Fixed stratified target test block, scored at every p (default 0.2).")
    p.add_argument("--compliance", choices=("verbose", "summary", "abnormal_only"),
                   default="verbose")
    p.add_argument("--cache-dir", type=Path, default=Path("embeddings_cross_scale"),
                   help="Embedding cache (shared across every pair; keyed by state + content hash).")
    p.add_argument("--output", type=Path, required=True, help="Results CSV to append to.")
    p.add_argument("--dump-predictions", action="store_true",
                   help="Also write per-row (y_true, scores) to predictions/<tag>.npz.")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    if args.artifact_root is None:
        # One source of truth: utils resolves $CCQ_ARTIFACT_ROOT, falling back to
        # artifacts/ inside the repository.
        from utils import ARTIFACT_ROOT
        args.artifact_root = str(ARTIFACT_ROOT)

    if args.within:
        if not args.state:
            raise SystemExit("--within requires --state.")
        run_within(args)
    else:
        missing = [f for f, v in [("--source", args.source), ("--target", args.target),
                                  ("--src-name", args.src_name),
                                  ("--tgt-name", args.tgt_name)] if not v]
        if missing:
            raise SystemExit(f"transfer mode requires {', '.join(missing)} "
                             f"(or use --within/--state for the ceiling).")
        run_transfer(args)


if __name__ == "__main__":
    main()
