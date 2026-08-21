"""Per-state feature attribution via SHAP over an XGBoost regressor.

An interpretation layer over the within-state experiment: it writes no rows into
the within-state result CSVs and adds no method tag.

Fits `XGBRegressor(reg:squarederror)` on the ordinal rating rather than the K-way
classifier, so SHAP returns one (n, d) matrix in rating points instead of one
table per class. Hyperparameters mirror `baselines_ml.make_xgb`; only `objective`
differs. `--qwk-check` reports the regressor's CV QWK against the classifier's.

A Boruta-style noise floor is computed alongside: `--n-shadow` real columns are
permuted and re-fit, and the floor is the largest mean |SHAP| any shadow attains.
Shadow sources are drawn across cardinality tiers, so the floor is calibrated for
continuous columns and is strict for binary ones. Above the floor is signal;
below it is not evidence of noise.

Use `dir_corr`, not `mean_shap`, for direction.

    python shap_xgb.py \\
        --input data/nc_records_cleaned_full.csv \\
        --output results/<date>/shap_xgb/shap_xgb_nc_results.csv \\
        --folds fold_indices/nc_native_folds.json \\
        --remap-state NC --rating-scale 5star
    python shap_xgb.py --grid-only --output results/<date>/shap_xgb/grid.csv
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from utils import (
    LEAKY_COLS,
    SEED,
    TARGET_COL,
    add_io_args,
    build_preprocessor,
    compute_metrics,
    configure_verbosity,
    get_feature_columns,
    get_folds,
    load_data,
    maybe_remap,
)

SHADOW_PREFIX = "shadow__"

# The 12 states, in the order the drivers use them, for the grid figure.
GRID_STATES = ["ca", "ga", "nc", "wi", "co", "ky", "md", "mt", "ne", "ok", "sc", "wa"]


def n_jobs_from_env(default: int = 4) -> int:
    """Respect the SLURM allocation instead of grabbing every visible core.

    `baselines_ml` uses n_jobs=-1, which is correct on a whole-node GPU job but
    antisocial on the shared CPU partition this experiment runs on
    (--cpus-per-task=4). Falls back to `default` off-cluster.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val and val.isdigit() and int(val) > 0:
            return int(val)
    return default


def make_xgb_regressor(n_jobs: int):
    """XGBoost regressor for the attribution model.

    Identical to `baselines_ml.make_xgb` except for `objective`, so the model
    explained here is the model the tables report.
    """
    from xgboost import XGBRegressor
    return XGBRegressor(
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


def build_shadow_frame(
    df: pd.DataFrame, feature_cols: list[str], n_shadow: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[str]]:
    """Add permuted shadow copies of real columns, as a noise floor.

    Sources are drawn across cardinality tiers rather than uniformly by rank:
    most columns here are binary, and since the floor is a max over shadows and
    high-cardinality columns win it, rank-uniform sampling would reduce the floor
    to a sample of size one.

    A column with k distinct values offers k-1 split candidates, so the max over
    more draws is larger by chance alone; the tiers keep that comparable.
    """
    if n_shadow <= 0:
        return pd.DataFrame(index=df.index), []

    collide = [c for c in feature_cols if c.startswith(SHADOW_PREFIX)]
    if collide:
        raise ValueError(
            f"Real feature column(s) already use the reserved {SHADOW_PREFIX!r} "
            f"prefix: {collide}. Rename them or change SHADOW_PREFIX.")

    card = df[feature_cols].nunique(dropna=True)
    tiers = {
        "binary": card[card <= 2].index.tolist(),
        "low":    card[(card > 2) & (card <= 10)].index.tolist(),
        "mid":    card[(card > 10) & (card <= 100)].index.tolist(),
        "high":   card[card > 100].index.tolist(),
    }
    tiers = {k: v for k, v in tiers.items() if v}

    def _spread(cols, k, key):
        """k columns spread evenly across `cols` sorted by `key`."""
        if k <= 0 or not cols:
            return []
        s = key[cols].sort_values().index.tolist()
        if len(s) <= k:
            return s
        pos = np.unique(np.linspace(0, len(s) - 1, k).round().astype(int))
        return [s[i] for i in pos]

    prevalence = df[feature_cols].notna().mean() * df[feature_cols].apply(
        lambda c: c.astype(float).fillna(0).clip(0, 1).mean() if c.nunique(dropna=True) <= 2 else 1.0)

    per_tier = max(1, n_shadow // len(tiers))
    chosen: list[str] = []
    for name, cols in tiers.items():
        key = prevalence if name == "binary" else card
        chosen += _spread(cols, per_tier, key)
    # Top up from the highest tier if integer division left us short: the floor
    # is a max, so extra draws at the high end are what stabilise it.
    if len(chosen) < n_shadow:
        top = tiers[list(tiers)[-1]]
        for c in _spread(top, n_shadow, card):
            if c not in chosen and len(chosen) < n_shadow:
                chosen.append(c)

    data = {}
    for col in chosen:
        vals = df[col].to_numpy(copy=True)
        rng.shuffle(vals)  # destroys the label relationship, keeps the marginal
        data[f"{SHADOW_PREFIX}{col}"] = vals

    counts = {n: sum(1 for c in chosen if c in cols) for n, cols in tiers.items()}
    print(f"[shadow] {len(chosen)} shadow columns; per tier {counts}; "
          f"source cardinality {card[chosen].min()}..{card[chosen].max()}")
    return pd.DataFrame(data, index=df.index), chosen


def round_to_scale(pred: np.ndarray, classes_sorted) -> np.ndarray:
    """Round continuous predictions to the nearest valid rating level."""
    lo, hi = min(classes_sorted), max(classes_sorted)
    return np.clip(np.rint(np.asarray(pred, dtype=float)), lo, hi).astype(int)


def fit_and_explain(X, y_float, sample_weight, feature_names, n_jobs):
    """Fit the regressor on all rows and compute TreeSHAP values.

    Returns the per-row SHAP matrix in rating points alongside the fitted model.
    """
    import shap

    model = make_xgb_regressor(n_jobs)
    t0 = time.time()
    model.fit(X, y_float, sample_weight=sample_weight)
    print(f"[fit] {X.shape[0]} rows x {X.shape[1]} features in {time.time() - t0:.1f}s")

    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    sv = np.asarray(sv)
    print(f"[shap] explained {sv.shape} in {time.time() - t0:.1f}s")

    # Single-output regression MUST give (n, d). Older shap returns a list;
    # assert the shape rather than branching on a version string.
    if sv.ndim != 2 or sv.shape != X.shape:
        raise RuntimeError(
            f"Expected SHAP values of shape {X.shape} for a single-output "
            f"regressor, got {sv.shape}. If this is a list-returning shap "
            f"version the normalization above needs updating.")
    if len(feature_names) != sv.shape[1]:
        raise RuntimeError(
            f"Feature-name count {len(feature_names)} != SHAP width "
            f"{sv.shape[1]}; the preprocessor changed the column set.")

    # ADDITIVITY: sum of attributions + base value must reconstruct the model's
    # own prediction exactly. This one identity catches essentially every
    # preprocessing or column-ordering bug, because any misalignment breaks it.
    expected = float(np.ravel(explainer.expected_value)[0])
    recon = sv.sum(axis=1) + expected
    max_err = float(np.abs(recon - model.predict(X)).max())
    print(f"[check] additivity max|Sum(shap)+base - predict| = {max_err:.3e}")
    if max_err > 1e-3:
        raise RuntimeError(
            f"SHAP additivity violated (max error {max_err:.3e}). The explained "
            f"matrix does not reconstruct the model's predictions — do not trust "
            f"the ranking.")
    return sv, model, expected


def feature_direction(sv, X) -> np.ndarray:
    """Per-feature Pearson correlation between feature value and its SHAP value.

    This is the direction statistic, not `mean_shap`. Balanced sample weighting
    offsets `base_score` from the unweighted row mean, which lands in every
    feature's signed value; and a feature with a non-monotone effect can have a
    signed mean near zero while still mattering. The correlation is unaffected by
    both.
    """
    xc = X - X.mean(axis=0)
    sc = sv - sv.mean(axis=0)
    den = np.sqrt((xc ** 2).sum(axis=0) * (sc ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, (xc * sc).sum(axis=0) / den, 0.0)
    return np.nan_to_num(r, nan=0.0)


def summarize(sv, X, feature_names, shadow_names, card=None) -> pd.DataFrame:
    """Collapse the (n, d) matrix to one row per feature."""
    mean_abs = np.abs(sv).mean(axis=0)
    mean_signed = sv.mean(axis=0)

    out = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        # THE direction column. `mean_shap` is retained for completeness but is
        # baseline-offset-confounded — see feature_direction().
        "dir_corr": feature_direction(sv, X),
        "mean_shap": mean_signed,
    })
    # Cardinality + tier travel with the row so a bucketed re-analysis of the
    # floor is possible from the CSV alone, without another fit.
    if card is not None:
        src = out["feature"].str.replace(f"^{SHADOW_PREFIX}", "", regex=True)
        out["cardinality"] = src.map(card)
        out["card_tier"] = pd.cut(
            out["cardinality"], [0, 2, 10, 100, np.inf],
            labels=["binary", "low", "mid", "high"])
    out["is_shadow"] = out["feature"].isin(shadow_names)

    real = out.loc[~out["is_shadow"]]
    total = real["mean_abs_shap"].sum()
    out["shap_share"] = out["mean_abs_shap"] / total if total > 0 else np.nan

    # Rank REAL features only; shadows are diagnostics, not findings.
    out["rank"] = np.nan
    order = real.sort_values("mean_abs_shap", ascending=False).index
    out.loc[order, "rank"] = np.arange(1, len(order) + 1)

    floor = out.loc[out["is_shadow"], "mean_abs_shap"].max() if shadow_names else np.nan
    out["noise_floor"] = floor
    out["above_floor"] = out["mean_abs_shap"] > floor if shadow_names else True

    return out.sort_values(
        ["is_shadow", "mean_abs_shap"], ascending=[True, False]
    ).reset_index(drop=True)


def run_qwk_check(df, feature_cols, folds, classes_sorted, n_jobs, balanced: bool):
    """5-fold CV QWK for the regressor, rounded to the rating scale.

    Optionally reports the delta against the classifier rows in
    `compare_results`.
    """
    fold_metrics = []
    for fold in folds:
        tr, va = fold["train_idx"], fold["val_idx"]
        pre = build_preprocessor(
            numerical_cols=feature_cols, categorical_cols=[],
            scale=False, encoding="ordinal")
        X_tr = pre.fit_transform(df.iloc[tr])
        X_va = pre.transform(df.iloc[va])
        y_tr = df.iloc[tr][TARGET_COL].to_numpy(dtype=float)
        y_va = df.iloc[va][TARGET_COL].to_numpy(dtype=int)

        sw = compute_sample_weight("balanced", y=y_tr.astype(int)) if balanced else None
        model = make_xgb_regressor(n_jobs)
        model.fit(X_tr, y_tr, sample_weight=sw)

        y_pred = round_to_scale(model.predict(X_va), classes_sorted)
        m = compute_metrics(y_va, y_pred, y_proba=None, labels=classes_sorted)
        fold_metrics.append(m)
        print(f"  fold {fold['fold']}: qwk={m['qwk']:.3f} acc={m['accuracy']:.3f} "
              f"mae={m['mae']:.3f}")

    return {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}


def lookup_classifier_qwk(path: "Path | None") -> float:
    """The reported `xgb` mean QWK, for the --qwk-check delta."""
    if not path or not Path(path).exists():
        return float("nan")
    ref = pd.read_csv(path)
    row = ref[(ref["method"] == "xgb") & (ref["fold"].astype(str) == "mean")]
    return float(row["qwk"].iloc[0]) if len(row) else float("nan")


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def plot_state(summary, sv, X, feature_names, state, top_k, figs_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    top = summary[~summary["is_shadow"]].head(top_k).iloc[::-1]
    floor = summary["noise_floor"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 0.38 * len(top) + 1.2))
    # Colour from dir_corr, NOT mean_shap: mean_shap carries the balanced-
    # weighting baseline offset, which made whole states one colour (see
    # feature_direction()).
    colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in top["dir_corr"]]
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
    if np.isfinite(floor):
        ax.axvline(floor, color="#888780", linestyle="--", linewidth=1,
                   label="noise floor")
        ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("mean |SHAP| (rating points)")
    ax.set_title(f"{state.upper()} — top {top_k} features")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figs_dir / f"shap_{state}_bar.{ext}", dpi=200)
    plt.close(fig)

    try:
        import shap
        real_idx = [i for i, f in enumerate(feature_names)
                    if not f.startswith(SHADOW_PREFIX)]
        shap.summary_plot(
            sv[:, real_idx],
            pd.DataFrame(X[:, real_idx], columns=[feature_names[i] for i in real_idx]),
            max_display=top_k, show=False)
        fig = plt.gcf()
        fig.suptitle(f"{state.upper()}", fontsize=10)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figs_dir / f"shap_{state}_beeswarm.{ext}", dpi=200)
        plt.close(fig)
    except Exception as e:  # a beeswarm failure must not lose the CSV
        print(f"  WARNING: beeswarm failed for {state}: {e}")


def _shorten(name: str, maxlen: int = 26) -> str:
    """Middle-elide a feature name so it fits a narrow paper panel.

    Keeps the head AND the tail because the tail is often the discriminating
    part (`accreditation_national_association_for_family_child_care_nafcc` ->
    `accreditation_natio…ce_nafcc`); a plain truncation would drop it.
    """
    name = str(name)
    if len(name) <= maxlen:
        return name
    keep_tail = max(6, maxlen // 3)
    return name[: maxlen - keep_tail - 1] + "…" + name[-keep_tail:]


def plot_grid(results_dir: Path, top_k: int = 10, paper: bool = True):
    """3x4 grid of per-state top-k panels, built from the per-state CSVs.

    Run after every state has finished, since each run knows only its own
    state. ``paper=True`` sizes the figure at 1:1 for a two-column page and
    elides long feature names.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if paper:
        figsize, fs_lab, fs_tick, fs_title, fs_sup, maxlen = \
            (7.16, 6.6), 4.6, 4.4, 6.5, None, 26
        # No suptitle in paper mode: the LaTeX caption carries it, and dropping
        # it buys the top row ~0.3 in.
        margins = dict(left=0.165, right=0.985, top=0.975, bottom=0.045,
                       wspace=1.05, hspace=0.30)
    else:
        figsize, fs_lab, fs_tick, fs_title, fs_sup, maxlen = \
            (20, 12), 7, 7, 11, 13, 60
        margins = dict(left=0.07, right=0.98, top=0.92, bottom=0.05,
                       wspace=0.55, hspace=0.30)

    results_dir = Path(results_dir)
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    n_found = 0
    for ax, st in zip(axes.ravel(), GRID_STATES):
        path = results_dir / f"shap_xgb_{st}_results.csv"
        if not path.exists():
            # Blank the panel but KEEP the axes in the layout. ax.axis("off")
            # drops it from tight_layout's grid, so the panels that DO have
            # data stretch to fill the gap and the figure stops being a
            # comparable 3x4 — states finish at different times, so a partial
            # grid is the normal case, not the exception.
            ax.set_title(f"{st.upper()} — missing", fontsize=fs_title, color="#888780")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        n_found += 1
        d = pd.read_csv(path)
        top = d[~d["is_shadow"]].nsmallest(top_k, "rank").iloc[::-1]
        colors = ["#1D9E75" if v >= 0 else "#D85A30" for v in top["dir_corr"]]
        ax.barh([_shorten(f, maxlen) for f in top["feature"]],
                top["mean_abs_shap"], color=colors)
        floor = top["noise_floor"].iloc[0] if len(top) else np.nan
        if np.isfinite(floor):
            ax.axvline(floor, color="#888780", linestyle="--", linewidth=0.7)
        ax.set_title(st.upper(), fontsize=fs_title, pad=2)
        ax.tick_params(axis="y", labelsize=fs_lab, pad=1, length=0)
        ax.tick_params(axis="x", labelsize=fs_tick, pad=1, length=2)
        if paper:
            # A 4-column grid leaves ~1 in per panel for the x axis; the default
            # tick locator puts 6-7 labels there and they collide.
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_linewidth(0.5)

    if fs_sup is not None:
        fig.suptitle(f"Top-{top_k} features by mean |SHAP| (XGBoost regression), "
                     "dashed line = shadow-feature noise floor", fontsize=fs_sup)
    # FIXED geometry, not tight_layout: tight_layout sizes panels from their
    # content, so empty panels (states not finished yet) collapse and the ones
    # with data stretch to fill the gap — the grid stops being comparable.
    # subplots_adjust keeps every panel the same size regardless.
    fig.subplots_adjust(**margins)
    figs = results_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"shap_grid.{ext}", dpi=200)
    plt.close(fig)
    print(f"Grid figure written for {n_found}/{len(GRID_STATES)} states -> {figs}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    add_io_args(parser)
    parser.add_argument("--top-k", type=int, default=10,
                        help="How many features the top-K report covers (default 10).")
    parser.add_argument("--n-shadow", type=int, default=10,
                        help="Shadow (shuffled-copy) columns for the noise floor. "
                             "0 disables the floor.")
    parser.add_argument("--figs-dir", type=Path, default=None,
                        help="Where figures go (default: a 'figs' subdir next to --output).")
    parser.add_argument("--no-figs", action="store_true", help="Skip figure generation.")
    parser.add_argument("--unweighted", action="store_true",
                        help="Fit WITHOUT balanced sample weights. The suite's "
                             "classical baselines are class-weighted, so weighted "
                             "is the default here too.")
    parser.add_argument("--qwk-check", action="store_true", default=True,
                        help="5-fold CV QWK for the regressor, to show it is a fair "
                             "stand-in for the reported classifier (default on).")
    parser.add_argument("--no-qwk-check", dest="qwk_check", action="store_false")
    parser.add_argument("--compare-results", type=Path, default=None,
                        help="Within-state results CSV for this state; the `xgb` "
                             "mean QWK is read from it for the delta.")
    parser.add_argument("--grid-only", action="store_true",
                        help="Skip fitting; rebuild the 3x4 grid figure from the "
                             "per-state CSVs already in --output's directory.")
    args = parser.parse_args()
    configure_verbosity(args.verbose)

    out_dir = Path(args.output).parent
    if args.grid_only:
        plot_grid(out_dir, top_k=args.top_k)
        return

    state = (args.remap_state or "within").lower()
    df = load_data(args.input)
    df = maybe_remap(df, args.remap_state, allow_identity=args.allow_identity,
                     scale=args.rating_scale)

    feature_cols = get_feature_columns(df)
    # int, not the float the CSV parses to: these become clip bounds in
    # round_to_scale and the label space for compute_metrics, and ratings are
    # integers by definition.
    classes_sorted = sorted(int(c) for c in df[TARGET_COL].unique())
    n_jobs = n_jobs_from_env()
    print(f"[{state}] {len(df)} rows, {len(feature_cols)} features, "
          f"classes={classes_sorted}, n_jobs={n_jobs}")

    # --- shadow features -----------------------------------------------------
    rng = np.random.default_rng(SEED)
    shadow_df, shadow_sources = build_shadow_frame(df, feature_cols, args.n_shadow, rng)
    shadow_names = list(shadow_df.columns)
    aug = pd.concat([df[feature_cols], shadow_df], axis=1)
    all_cols = feature_cols + shadow_names
    # Cardinality of the REAL columns, keyed by source name. Shadows inherit
    # their source's value via the name strip in summarize(), so one map covers
    # both and the CSV can carry a per-feature tier for later re-analysis.
    card_all = df[feature_cols].nunique(dropna=True)

    # --- fit + explain -------------------------------------------------------
    pre = build_preprocessor(numerical_cols=all_cols, categorical_cols=[],
                             scale=False, encoding="ordinal")
    X = pre.fit_transform(aug)
    feature_names = list(pre.get_feature_names_out())
    y_float = df[TARGET_COL].to_numpy(dtype=float)
    sample_weight = (None if args.unweighted
                     else compute_sample_weight("balanced", y=y_float.astype(int)))

    sv, model, expected = fit_and_explain(X, y_float, sample_weight, feature_names, n_jobs)
    summary = summarize(sv, X, feature_names, shadow_names, card=card_all)

    # --- standing assertions -------------------------------------------------
    top = summary[~summary["is_shadow"]].head(args.top_k)
    leaky_hits = [f for f in top["feature"] if f in LEAKY_COLS]
    if leaky_hits:
        raise RuntimeError(
            f"LEAKAGE: {leaky_hits} appear in the top-{args.top_k}. `load_data` "
            f"should have refused this file — investigate before reporting.")

    floor = summary["noise_floor"].iloc[0]
    n_above = int(top["above_floor"].sum())
    cum_share = float(top["shap_share"].sum())
    print(f"\n[{state}] top-{args.top_k}: {n_above}/{len(top)} clear the noise "
          f"floor ({floor:.5f}); they carry {cum_share:.1%} of total attribution")
    for _, r in top.iterrows():
        flag = "" if r["above_floor"] else "   <-- BELOW NOISE FLOOR"
        print(f"  {int(r['rank']):2d}. {r['feature']:<45s} "
              f"|shap|={r['mean_abs_shap']:.4f}  dir={r['dir_corr']:+.2f}"
              f"  [{r.get('card_tier', '?')}]{flag}")

    # A shadow column outranking real features means the model is largely
    # fitting noise, which on a small state is a plausible outcome worth
    # surfacing. The floor is the largest shadow, so this counts real features
    # only.
    n_real_above = int((summary["mean_abs_shap"] > floor).sum()) if np.isfinite(floor) else -1
    if np.isfinite(floor) and n_real_above < args.top_k:
        print(f"  WARNING: only {n_real_above} real features beat a shuffled "
              f"column — this state's model is mostly fitting noise.")

    # --- write ---------------------------------------------------------------
    # NB keep the current run's frame separate from what lands on disk: results
    # logging APPENDS (never dedupes), so the on-disk frame may carry rows from
    # earlier runs and must not be what the figures are drawn from.
    summary.insert(0, "state", state)
    summary.insert(1, "scale", args.rating_scale)
    summary.insert(2, "n_rows", len(df))
    summary.insert(3, "n_features", len(feature_cols))
    summary.insert(4, "n_shadow", len(shadow_names))
    summary["notes"] = ("weighted" if not args.unweighted else "unweighted")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_write = summary
    if out_path.exists() and out_path.stat().st_size > 0:
        to_write = pd.concat([pd.read_csv(out_path), summary], ignore_index=True)
    to_write.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    # --- the regressor-vs-classifier gate ------------------------------------
    if args.qwk_check:
        print(f"\n[{state}] QWK check (regressor, 5-fold CV on the native folds)")
        folds = get_folds(df, folds_path=args.folds)
        agg = run_qwk_check(df, feature_cols, folds, classes_sorted, n_jobs,
                            balanced=not args.unweighted)
        ref = lookup_classifier_qwk(args.compare_results)
        delta = agg["qwk"] - ref if np.isfinite(ref) else float("nan")
        if np.isfinite(ref):
            verdict = "regressor is a fair stand-in" if delta > -0.03 else \
                      "REGRESSOR IS MATERIALLY WORSE for this state"
            print(f"  regressor mean QWK = {agg['qwk']:.4f}   "
                  f"reported classifier = {ref:.4f}   delta = {delta:+.4f}")
            print(f"  -> {verdict}")
        else:
            print(f"  regressor mean QWK = {agg['qwk']:.4f}   "
                  f"(no --compare-results given, so no delta)")

        check_path = out_path.parent / "shap_xgb_qwk_check.csv"
        row = pd.DataFrame([{
            "state": state, "scale": args.rating_scale,
            "weighting": "unweighted" if args.unweighted else "weighted",
            "qwk_regressor": agg["qwk"], "qwk_classifier_reported": ref,
            "delta": delta, "accuracy_regressor": agg["accuracy"],
            "mae_regressor": agg["mae"],
        }])
        if check_path.exists() and check_path.stat().st_size > 0:
            row = pd.concat([pd.read_csv(check_path), row], ignore_index=True)
        row.to_csv(check_path, index=False)
        print(f"Wrote {check_path}")

    if not args.no_figs:
        figs_dir = args.figs_dir or (out_path.parent / "figs")
        plot_state(summary, sv, X, feature_names, state, args.top_k, figs_dir)
        print(f"Figures -> {figs_dir}")


if __name__ == "__main__":
    main()
