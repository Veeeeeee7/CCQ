"""Seed stability of the SHAP noise floor for one state.

Reruns shap_xgb.py's fit under several seeds (the seed drives the shadow-column
shuffle and the XGBoost fit) and counts, for every real feature, the runs in which
it clears the noise floor.

    python shap_seed_check.py --input data/mt_records_cleaned_full.csv \\
        --remap-state MT --rating-scale 5star --n-seeds 30 \\
        --output results/shap_xgb_seed_check/shap_xgb_seed_check_mt.csv
"""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

import shap_xgb
from utils import TARGET_COL, build_preprocessor, get_feature_columns, load_data, maybe_remap


def run_once(df, feature_cols, card, seed, n_shadow, weighted, n_jobs):
    """One shap_xgb fit with `seed`; returns the per-feature summary for this run."""
    shap_xgb.SEED = seed  # make_xgb_regressor reads the module-level seed
    with contextlib.redirect_stdout(io.StringIO()):
        rng = np.random.default_rng(seed)
        shadow_df, _ = shap_xgb.build_shadow_frame(df, feature_cols, n_shadow, rng)
        shadow_names = list(shadow_df.columns)
        aug = pd.concat([df[feature_cols], shadow_df], axis=1)
        pre = build_preprocessor(numerical_cols=feature_cols + shadow_names,
                                 categorical_cols=[], scale=False, encoding="ordinal")
        X = pre.fit_transform(aug)
        names = list(pre.get_feature_names_out())
        y = df[TARGET_COL].to_numpy(dtype=float)
        w = compute_sample_weight("balanced", y=y.astype(int)) if weighted else None
        sv, _, _ = shap_xgb.fit_and_explain(X, y, w, names, n_jobs)
        summary = shap_xgb.summarize(sv, X, names, shadow_names, card=card)
    summary.insert(0, "seed", seed)
    summary.insert(1, "weighted", weighted)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True, help="Cleaned `_full` CSV for one state.")
    p.add_argument("--remap-state", required=True, help="State code, e.g. MT.")
    p.add_argument("--rating-scale", default="5star", choices=["3star", "5star"])
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--n-shadow", type=int, default=10)
    p.add_argument("--unweighted", action="store_true",
                   help="Fit without balanced sample weights.")
    p.add_argument("--output", type=Path, required=True,
                   help="Long CSV: one row per (seed, feature).")
    args = p.parse_args()

    df = load_data(args.input)
    df = maybe_remap(df, args.remap_state, scale=args.rating_scale)
    feature_cols = get_feature_columns(df)
    card = df[feature_cols].nunique(dropna=True)
    n_jobs = shap_xgb.n_jobs_from_env()

    seeds = range(args.first_seed, args.first_seed + args.n_seeds)
    runs = pd.concat([run_once(df, feature_cols, card, s, args.n_shadow,
                               not args.unweighted, n_jobs) for s in seeds],
                     ignore_index=True)
    runs.insert(0, "state", args.remap_state.lower())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output, index=False)

    real = runs[~runs["is_shadow"]]
    floors = runs.groupby("seed")["noise_floor"].first()
    cleared = (real.groupby("feature")["above_floor"].sum()
               .sort_values(ascending=False).astype(int))
    print(f"[{args.remap_state}] {len(seeds)} seeds, "
          f"{'unweighted' if args.unweighted else 'balanced weights'}")
    print(f"  noise floor: median {floors.median():.4f}, "
          f"range {floors.min():.4f}..{floors.max():.4f}")
    print(f"  runs with at least one real feature above the floor: "
          f"{int((real.groupby('seed')['above_floor'].sum() > 0).sum())} of {len(seeds)}")
    print("  runs in which each feature clears the floor:")
    for feat, n in cleared[cleared > 0].items():
        print(f"    {feat:<40s} {n:>3d} / {len(seeds)}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
