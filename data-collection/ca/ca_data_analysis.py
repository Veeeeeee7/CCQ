#!/usr/bin/env python3
"""
target_eda.py -- quick exploratory analysis of the QCC quality-rating target.

Run on the CLEANED csv (output of clean_crawl_errors.py), before any
preprocessing. It characterises `basics_qcc_score`:

  1. raw value distribution, incl. the '-' sentinel and NaN
  2. numeric coercion + identification of out-of-range GARBAGE values
     (QCC scores are 1..5; anything else is a scrape artifact)
  3. the rated-only (1..5) distribution you'll actually model, with
     class proportions and imbalance ratio
  4. a 5-fold StratifiedKFold preview so you can SEE how thin the rare
     classes get per fold (score 1 has only 8 rows total)

Usage:
    python target_eda.py facility_records_clean.csv
    python target_eda.py facility_records_clean.csv --plot target_dist.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "basics_qcc_score"
ERRORS = "errors"
ID = "facility_number"
VALID = [1, 2, 3, 4, 5]  # the only legitimate QCC scores
N_SPLITS = 5
SEED = 42


def hr(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def text_bar(counts: pd.Series, width: int = 40) -> None:
    """Tiny horizontal bar chart for the terminal."""
    if counts.empty:
        print("  (no data)")
        return
    top = counts.max()
    for idx, n in counts.items():
        bar = "#" * int(round(width * n / top)) if top else ""
        print(f"  {str(idx):>6} | {n:6d} | {bar}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", type=Path, help="cleaned CSV")
    ap.add_argument("--plot", type=Path, default=None, help="optional PNG of the rated-only distribution")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols from {args.input.name}")

    # --- 1. raw distribution -------------------------------------------------
    hr("1. Raw value_counts of basics_qcc_score (incl. NaN)")
    raw = df[TARGET].value_counts(dropna=False)
    print(raw.to_string())

    # --- 2. coercion + garbage detection ------------------------------------
    hr("2. Numeric coercion")
    s = df[TARGET].replace("-", np.nan)          # '-' sentinel -> missing
    num = pd.to_numeric(s, errors="coerce")      # non-numeric -> NaN

    n_dash = (df[TARGET] == "-").sum()
    n_nan_raw = df[TARGET].isna().sum()
    n_numeric = num.notna().sum()
    print(f"'-' sentinel (no rating):     {n_dash:,}")
    print(f"NaN in source (scrape fails): {n_nan_raw:,}")
    print(f"coerced to a number:          {n_numeric:,}")

    garbage_mask = num.notna() & ~num.isin(VALID)
    n_garbage = int(garbage_mask.sum())
    print(f"\nGARBAGE values (numeric but not in 1..5): {n_garbage}")
    if n_garbage:
        cols = [c for c in (ID, TARGET) if c in df.columns]
        print(df.loc[garbage_mask, cols].to_string(index=False))
        print("  -> recommend dropping/fixing these at source before modelling.")

    # --- 3. successful-scrape view ------------------------------------------
    hr("3. Target among successful scrapes (errors is NaN)")
    if ERRORS in df.columns:
        ok = df[df[ERRORS].isna()]
        print(f"successful scrapes: {len(ok):,} of {len(df):,}")
        print(ok[TARGET].value_counts(dropna=False).to_string())
    else:
        print(f"(no '{ERRORS}' column found -- skipping)")

    # --- 4. rated-only distribution -----------------------------------------
    hr("4. Rated-only distribution (the modelling set: scores 1..5)")
    rated = num[num.isin(VALID)].astype(int)
    counts = rated.value_counts().sort_index()
    total = len(rated)
    print(f"rated facilities: {total:,}\n")
    print("  score | count | share")
    print("  ------+-------+------")
    for k in VALID:
        c = int(counts.get(k, 0))
        print(f"  {k:>5} | {c:5d} | {c / total:6.1%}")
    if total:
        imb = counts.max() / max(counts.min(), 1)
        print(f"\nimbalance ratio (largest / smallest class): {imb:.1f}x")
        print("\ndistribution:")
        text_bar(counts)

    # --- 5. 5-fold stratified preview ---------------------------------------
    hr("5. 5-fold StratifiedKFold preview (per-fold class counts)")
    if total < N_SPLITS or counts.min() < N_SPLITS:
        print(f"NOTE: smallest class has {int(counts.min())} rows; with {N_SPLITS} "
              f"folds that's ~{counts.min() / N_SPLITS:.1f} per fold.")
    try:
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        y = rated.to_numpy()
        header = "  fold | " + " | ".join(f"s{k}" for k in VALID) + " | total"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for i, (_, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
            vc = pd.Series(y[val_idx]).value_counts()
            cells = " | ".join(f"{int(vc.get(k, 0)):2d}" for k in VALID)
            print(f"   {i:>3} | {cells} | {len(val_idx):5d}")
        print("\nRead the s1 column: that's how many score-1 examples land in each\n"
              "validation fold. If it's 1-2, per-fold metrics on that class will be\n"
              "very noisy -- worth keeping in mind when you read the results.csv stds.")
    except ImportError:
        print("(scikit-learn not installed -- skipping fold preview)")

    # --- optional plot -------------------------------------------------------
    if args.plot and total:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar([str(k) for k in VALID], [int(counts.get(k, 0)) for k in VALID],
                   color="#4C72B0")
            ax.set_xlabel("QCC score")
            ax.set_ylabel("number of facilities")
            ax.set_title(f"Rated-only target distribution (n={total:,})")
            for k in VALID:
                c = int(counts.get(k, 0))
                ax.text(VALID.index(k), c, str(c), ha="center", va="bottom")
            fig.tight_layout()
            fig.savefig(args.plot, dpi=120)
            print(f"\nsaved plot -> {args.plot}")
        except ImportError:
            print("\n(matplotlib not installed -- skipping plot)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())