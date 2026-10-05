"""Elo scores for the within-state and cross-state benchmarks (Figure 5).

Each combination of state and metric (BA, QWK) is one match between every pair
of models; the higher score wins and equal scores draw. Scores are compared as
reported in the paper's tables (percentages to two decimals). Ratings use the
online Elo update (K=20, start 1000), are shifted so that RF-txt is at 1000,
and are averaged over bootstrap rounds that resample the twelve states with
replacement. Cross-state Elo is computed separately at each target supervision
level on the same resampled states and then averaged over the six levels.

    python elo.py --results results --output results/elo
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import numpy as np

STATES = ["CA", "CO", "GA", "KY", "MD", "MT", "NC", "NE", "OK", "SC", "WA", "WI"]
METRICS = ["balanced_acc", "qwk"]

# Model order fixes the order of the matches, which the online update depends on.
WITHIN_MODELS = {
    "dummy": "Dummy", "lr": "LR-num", "rf": "RF-num", "xgb": "XGB-num",
    "tabpfn_balanced": "TabPFN-num", "tabnet": "TabNN-num",
    "lr_raw": "LR-txt", "rf_raw": "RF-txt", "xgb_raw": "XGB-txt",
    "tabpfn_balanced_raw": "TabPFN-txt", "tabnet_raw": "TabNN-txt",
    "bert_textualized_full_verbose": "MBERT", "bert_corn_textualized_full_verbose": "MBERT-CORN",
    "qwen_cls_within": "LLM-CLS", "qwen_cls_rag_within": "LLM-RAG", "qwen_cls_lora_within": "LLM-LoRA",
}
CROSS_MODELS = {
    "dummy": "Dummy", "lr": "LR-txt", "rf": "RF-txt", "xgb": "XGB-txt",
    "tabpfn_balanced": "TabPFN-txt", "tabnet": "TabNN-txt",
    "bert_textualized_full_verbose": "MBERT", "bert_corn_verbose": "MBERT-CORN",
    "qwen_cls": "LLM-CLS", "qwen_cls_rag": "LLM-RAG", "qwen_cls_lora": "LLM-LoRA",
}
ANCHOR = "RF-txt"

LLM_WITHIN_RE = re.compile(r"^(qwen_cls(?:_rag|_lora)?_within)(?:_5star)?_tgt100_[A-Z]{2}$")
LOSO_RE = re.compile(r"^xfer_(?P<base>.+?)(?:_5star)?(?P<seq>_seq)?_tgt(?P<p>\d+)_LOSO2(?P<tgt>[A-Z]{2})$")


def _pct(x: str) -> float:
    return float(f"{float(x) * 100:.2f}") if x not in ("", None) else float("nan")


def _ok(row: dict) -> bool:
    return row.get("status", "") in ("", "ok")


def within_scores(results: Path) -> dict:
    """{state: {model: (BA, QWK)}} from the mean rows; the last ok row of a tag wins."""
    out = {}
    for s in STATES:
        path = results / "within_state" / f"experiment_within_state_{s.lower()}_results.csv"
        scores = {}
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row.get("fold") != "mean" or not _ok(row):
                    continue
                m = LLM_WITHIN_RE.match(row["method"])
                tag = m.group(1) if m else row["method"]
                if tag in WITHIN_MODELS:
                    scores[WITHIN_MODELS[tag]] = tuple(_pct(row[k]) for k in METRICS)
        out[s] = scores
    return out


def cross_scores(results: Path) -> dict:
    """{level: {state: {model: (BA, QWK)}}}. At p > 0 a sequential (_seq) row wins over a pooled one."""
    files = [results / "5_star" / "loso_few_shot" / f"experiment_loso_few_shot_5star_{s.lower()}_results.csv"
             for s in STATES if s != "GA"]
    files.append(results / "3_star" / "loso_ga_few_shot" / "experiment_loso_ga_few_shot_3star_results.csv")
    picked = {}
    for path in files:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                m = LOSO_RE.match(row["method"])
                if not m or not _ok(row) or m["base"] not in CROSS_MODELS:
                    continue
                key = (int(m["p"]), m["tgt"], CROSS_MODELS[m["base"]])
                rank = 1 if (m["seq"] and int(m["p"]) > 0) else 0
                if key not in picked or rank >= picked[key][0]:
                    picked[key] = (rank, tuple(_pct(row[k]) for k in METRICS))
    out = {}
    for (p, tgt, model), (_, vals) in picked.items():
        out.setdefault(p, {}).setdefault(tgt, {})[model] = vals
    return out


def match_records(scores: dict, models: list[str]) -> dict:
    """Per dataset, the (i, j, score_i) matches in model order, one block per metric."""
    idx = {m: i for i, m in enumerate(models)}
    recs = {}
    for ds, by_model in scores.items():
        r = []
        for k in range(len(METRICS)):
            vals = [(m, by_model[m][k]) for m in models if m in by_model and not np.isnan(by_model[m][k])]
            for a in range(len(vals)):
                for b in range(a + 1, len(vals)):
                    va, vb = vals[a][1], vals[b][1]
                    r.append((idx[vals[a][0]], idx[vals[b][0]], 1.0 if va > vb else 0.0 if va < vb else 0.5))
        recs[ds] = r
    return recs


def elo(recs: dict, datasets: list, n_models: int, anchor: int, k: float = 20.0) -> np.ndarray:
    r = [1000.0] * n_models
    for ds in datasets:
        for a, b, s in recs[ds]:
            delta = k * (s - 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / 400.0)))
            r[a] += delta
            r[b] -= delta
    return np.asarray(r) - r[anchor] + 1000.0


def bootstrap(rec_groups: list[dict], models: list[str], n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_models) ratings; each round resamples the states once and averages over groups."""
    random.seed(seed)
    anchor = models.index(ANCHOR)
    rounds = []
    for _ in range(n_boot):
        sample = random.choices(STATES, k=len(STATES))
        rounds.append(np.mean([elo(g, sample, len(models), anchor) for g in rec_groups], axis=0))
    return np.asarray(rounds)


def summarize(ratings: np.ndarray, models: list[str], ci: float) -> list[dict]:
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    rows = [{"model": m, "elo": ratings[:, i].mean(),
             "ci_low": np.percentile(ratings[:, i], lo), "ci_high": np.percentile(ratings[:, i], hi)}
            for i, m in enumerate(models)]
    return sorted(rows, key=lambda r: -r["elo"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--output", type=Path, default=Path("results/elo"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    w_models = list(WITHIN_MODELS.values())
    w = bootstrap([match_records(within_scores(args.results), w_models)], w_models, args.n_boot, args.seed)
    c_models = list(CROSS_MODELS.values())
    cross = cross_scores(args.results)
    c = bootstrap([match_records(cross[p], c_models) for p in sorted(cross)], c_models, args.n_boot, args.seed)

    for name, ratings, models in [("within_state", w, w_models), ("cross_state", c, c_models)]:
        rows = summarize(ratings, models, args.ci)
        path = args.output / f"elo_{name}.csv"
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["model", "elo", "ci_low", "ci_high"])
            wr.writeheader()
            wr.writerows({k: (f"{v:.1f}" if k != "model" else v) for k, v in r.items()} for r in rows)
        print(f"\n{name} (mean Elo, {args.ci:g}% interval, {args.n_boot} rounds) -> {path}")
        for r in rows:
            print(f"  {r['model']:12s} {r['elo']:7.1f}  [{r['ci_low']:7.1f}, {r['ci_high']:7.1f}]")


if __name__ == "__main__":
    main()
