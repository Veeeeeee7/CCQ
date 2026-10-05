"""Tuned-parameter counts for the within-state Pareto figures.

Counts the parameters each within-state model fits, per state and fold, under the
configurations of baselines_ml.py, tabular_dl.py, llm.py and transfer_llm_cls.py:

  LR          coef_ + intercept_ of the fitted multinomial model (K*d + K)
  RF, XGB     leaves across all trees (XGBoost grows one tree per class per round)
  TabNN       trainable parameters of the TabNet network
  TabPFN      0: it predicts in context and updates no weights
  MBERT       ModernBERT-base plus its K-way head (full fine-tuning)
  MBERT-CORN  the same backbone with a (K-1)-unit CORN head
  LLM-*       the Qwen3-4B score head, plus the LoRA adapters for LLM-LoRA

Per-state counts are 5-fold means; the figures use the mean over the 12 states.
The -txt tabular models read MiniLM embeddings, so `--reps txt` embeds every state.

    python count_tuned_params.py
    python count_tuned_params.py --summary-only
"""
from __future__ import annotations

import argparse
import glob
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (RESULTS_DIR, SEED, TARGET_COL, build_preprocessor,
                   get_feature_columns, get_folds, load_data, maybe_remap,
                   run_artifact_dir)

warnings.filterwarnings("ignore")

STATES = ["ca", "ga", "nc", "wi", "co", "ky", "md", "mt", "ne", "ok", "sc", "wa"]
MBERT_NAME = "answerdotai/ModernBERT-base"
# Qwen3-4B shapes (config.json) and the LoRA settings of transfer_llm_cls.py.
QWEN = dict(hidden=2560, inter=9728, layers=36, q_out=32 * 128, kv_out=8 * 128)
LORA_R = 16


def scale_for(S: str) -> str:
    return "3star" if S == "GA" else "5star"


# ---- model factories: copies of the within-state configurations ----
def make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1,
                                  max_features="sqrt", class_weight="balanced_subsample",
                                  random_state=SEED, n_jobs=-1)


def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, objective="multi:softprob",
                         eval_metric="mlogloss", tree_method="hist", random_state=SEED,
                         n_jobs=-1)


def tabnet_params(d: int, k: int) -> int:
    from pytorch_tabnet.tab_network import TabNet
    from pytorch_tabnet.utils import create_group_matrix
    net = TabNet(input_dim=d, output_dim=k, n_d=16, n_a=16, n_steps=3, gamma=1.5,
                 cat_idxs=[], cat_dims=[], cat_emb_dim=4, n_independent=2, n_shared=2,
                 epsilon=1e-15, virtual_batch_size=128, momentum=0.02, mask_type="entmax",
                 group_attention_matrix=create_group_matrix([], d))
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# ---- per-fold counts for the tabular families ----
def load_state(s: str, rep: str, data_dir: Path, cache_dir: Path):
    S = s.upper()
    kind = "full" if rep == "num" else "raw"
    df = load_data(data_dir / f"{s}_records_cleaned_{kind}.csv")
    df = maybe_remap(df, S, scale=scale_for(S))
    folds = get_folds(df, folds_path=f"fold_indices/{s}_native_folds.json")
    if rep == "txt":
        from transfer_tabular import embed_dataframe_cv
        df, _ = embed_dataframe_cv(df, "verbose", S, cache_dir=cache_dir)
    return df, folds


def count_state(s: str, rep: str, data_dir: Path, cache_dir: Path) -> list[dict]:
    from sklearn.utils.class_weight import compute_sample_weight
    df, folds = load_state(s, rep, data_dir, cache_dir)
    k_all = df[TARGET_COL].nunique()
    l2i = {c: i for i, c in enumerate(sorted(df[TARGET_COL].unique()))}
    cols = get_feature_columns(df)
    rows = []
    for fold in folds:
        tr = np.asarray(fold["train_idx"])
        X_df = df.iloc[tr]
        y = X_df[TARGET_COL].map(l2i).to_numpy()
        present = np.unique(y)          # dense relabeling, as in baselines_ml.run_cv
        y_fit = np.searchsorted(present, y)
        kp = len(present)

        pre_lr = build_preprocessor(numerical_cols=cols, categorical_cols=[],
                                    scale=True, encoding="onehot")
        d = pre_lr.fit_transform(X_df).shape[1]
        lr = kp * d + kp if kp > 2 else d + 1

        pre_tree = build_preprocessor(numerical_cols=cols, categorical_cols=[],
                                      scale=False, encoding="ordinal")
        X = pre_tree.fit_transform(X_df)
        rf = make_rf().fit(X, y_fit)
        xgb = make_xgb().fit(X, y_fit,
                             sample_weight=compute_sample_weight("balanced", y_fit))
        trees = xgb.get_booster().trees_to_dataframe()

        rows.append(dict(
            state=s.upper(), rep=rep, fold=int(fold["fold"]), K=k_all, K_present=kp,
            n_train=len(tr), d=d, lr=lr,
            rf_leaves=int(sum(e.tree_.n_leaves for e in rf.estimators_)),
            rf_nodes=int(sum(e.tree_.node_count for e in rf.estimators_)),
            xgb_leaves=int((trees["Feature"] == "Leaf").sum()),
            xgb_nodes=int(len(trees)), xgb_trees=int(trees["Tree"].nunique()),
            tabnet=tabnet_params(d, k_all)))
        print(rows[-1], flush=True)
    return rows


# ---- language models ----
def mbert_params(k: int) -> int:
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification
    cfg = AutoConfig.from_pretrained(MBERT_NAME, num_labels=k)
    with torch.device("meta"):
        model = AutoModelForSequenceClassification.from_config(cfg)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def qwen_analytic(k: int) -> dict:
    h, i, L = QWEN["hidden"], QWEN["inter"], QWEN["layers"]
    q, kv = QWEN["q_out"], QWEN["kv_out"]
    head = h * k                        # the Qwen `score` layer has no bias
    lora = LORA_R * L * ((h + q) + 2 * (h + kv) + (q + h) + 2 * (h + i) + (i + h))
    return {"cls": head, "rag": head, "lora": lora + head}


def qwen_from_logs(patterns: list[str]) -> dict:
    """{STATE: {'cls': n, 'rag': n, 'lora': n}} from the within-state run logs.
    Later files (sorted by path, i.e. by date) override earlier ones."""
    out = {}
    files = sorted({f for p in patterns for f in glob.glob(p)})
    for f in files:
        text = Path(f).read_text(errors="replace")
        for block in re.split(r"(?m)^==== within-state: ", text)[1:]:
            state = block.split()[0]
            n = [int(x.replace(",", "")) for x in
                 re.findall(r"\[cls\] adapt=\w+\s+trainable params=([\d,]+)", block)]
            if len(n) == 3:
                out[state] = {"cls": n[0], "rag": n[1], "lora": n[2]}
    return out


# ---- summary ----
def summarize(fold_csvs: list[Path], qwen_logs: list[str]) -> pd.DataFrame:
    folds = pd.concat([pd.read_csv(p) for p in fold_csvs], ignore_index=True)
    per_state = folds.groupby(["rep", "state"]).mean(numeric_only=True)
    ks = folds.groupby("state")["K"].first()

    rows = {}
    for rep in ("num", "txt"):
        if rep not in per_state.index.get_level_values(0):
            continue
        ps = per_state.loc[rep]
        for col, name in [("lr", "LR"), ("rf_leaves", "RF"), ("xgb_leaves", "XGB"),
                          ("tabnet", "TabNN")]:
            rows[f"{name}-{rep}"] = ps[col]
        rows[f"TabPFN-{rep}"] = ps["lr"] * 0

    rows["MBERT"] = pd.Series({s: mbert_params(int(k)) for s, k in ks.items()})
    rows["MBERT-CORN"] = pd.Series({s: mbert_params(int(k) - 1) for s, k in ks.items()})
    logged = qwen_from_logs(qwen_logs)
    for key, name in [("cls", "LLM-CLS"), ("rag", "LLM-RAG"), ("lora", "LLM-LoRA")]:
        vals = {}
        for s, k in ks.items():
            analytic = qwen_analytic(int(k))[key]
            if s in logged and logged[s][key] != analytic:
                print(f"  [warn] {name} {s}: log {logged[s][key]:,} != analytic {analytic:,}")
            vals[s] = logged.get(s, {}).get(key, analytic)
        rows[name] = pd.Series(vals)

    table = pd.DataFrame(rows).T[sorted(ks.index)]
    summary = pd.DataFrame({"mean": table.mean(axis=1), "median": table.median(axis=1),
                            "min": table.min(axis=1), "max": table.max(axis=1)})
    return pd.concat([summary, table], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", default=None, help="Writes to results/[<date>/]param_counts/.")
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Embedding cache for --reps txt (default: artifacts/param_counts/).")
    ap.add_argument("--reps", nargs="+", default=["num", "txt"], choices=["num", "txt"])
    ap.add_argument("--states", nargs="+", default=STATES)
    ap.add_argument("--qwen-logs", nargs="*", default=[],
                    help="Within-state run logs (globs); their `trainable params=` lines "
                         "replace the analytic LLM counts, with a warning on a mismatch.")
    ap.add_argument("--summary-only", action="store_true",
                    help="Rebuild the summary from existing per-fold CSVs.")
    args = ap.parse_args()

    out_dir = Path(args.results) / (args.date or "") / "param_counts"
    cache_dir = args.cache_dir or run_artifact_dir("param_counts") / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_csvs = [out_dir / f"param_counts_folds_{rep}.csv" for rep in args.reps]

    if not args.summary_only:
        for rep, path in zip(args.reps, fold_csvs):
            rows = []
            for s in args.states:
                t0 = time.time()
                rows += count_state(s, rep, args.data_dir, cache_dir)
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"[{rep}:{s}] {time.time() - t0:.1f}s", flush=True)

    summary = summarize([p for p in fold_csvs if p.exists()], args.qwen_logs)
    summary.to_csv(out_dir / "param_counts_summary.csv", float_format="%.1f")
    with pd.option_context("display.float_format", "{:,.0f}".format,
                           "display.width", 200):
        print(summary[["mean", "median", "min", "max"]])


if __name__ == "__main__":
    main()
