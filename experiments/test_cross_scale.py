"""Self-verification for the cross-rubric experiment.

Runs the whole pipeline -- load, split, fit, rank, log -- with the sentence
encoder replaced by a deterministic hashed projection of each state's numeric
columns. That substitution is the only one; the real splitting, fitting, metric
and logging code runs.

Needs `data/` and no GPU. Exits non-zero on the first failed check.

    python test_cross_scale.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import cross_scale
from ranking_metrics import compute_ranking_metrics
from transfer_common import TARGET_COL
from utils import get_feature_columns

DATA = Path("data")
DIM = 96


def _stub_embed(df, state, compliance, cache_dir):
    """Deterministic shared-space stand-in for the MiniLM encoder.

    Each numeric column is projected onto a fixed random vector seeded by a hash
    of its NAME, so a column present in two states lands on the same axis in
    both — the property that makes cross-state transfer possible at all. Values
    are z-scored per column before projection so scale differences do not
    dominate, and rows are unit-normalized to match embed_state's output.
    """
    cols = get_feature_columns(df)
    X = np.zeros((len(df), DIM), dtype=np.float32)
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        v = np.nan_to_num(v, nan=0.0)
        sd = v.std()
        v = (v - v.mean()) / sd if sd > 1e-12 else v * 0.0
        rng = np.random.RandomState(abs(hash(c)) % (2**31))
        X += np.outer(v, rng.normal(0, 1, DIM)).astype(np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1)


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(dict(
            source=None, src_name="", target=None, tgt_name="", within=None,
            state="", folds=None, coarsen_source=False, models=list(cross_scale.MODELS),
            target_frac=0.0, target_test_frac=0.2, compliance="verbose",
            cache_dir=Path("/tmp/xs_cache"), output=None,
            # dump_predictions=True by default here, on purpose: the runner
            # always passes --dump-predictions, so the test must exercise the
            # same path. Testing a configuration nothing ships is not a test.
            dump_predictions=True,
            n_bootstrap=50, seed=42, llm_model="", artifact_root="/tmp/xs_art"), **kw)


def _raises(fn) -> bool:
    """True if `fn()` raises — used where failing loudly is the contract."""
    try:
        fn()
    except Exception:
        return True
    return False


def main() -> int:
    cross_scale.embed = _stub_embed          # the one substitution
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "xs.csv"
    ok = True

    def check(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'ok  ' if passed else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    print("\n=== 1. native labels, unreconciled ===")
    ga = cross_scale.load_native(DATA / "ga_records_cleaned_full.csv", "GA")
    mt = cross_scale.load_native(DATA / "mt_records_cleaned_full.csv", "MT")
    ga_lv = sorted(ga[TARGET_COL].unique().tolist())
    mt_lv = sorted(mt[TARGET_COL].unique().tolist())
    check("GA stays 3-level", ga_lv == [1, 2, 3], f"{ga_lv}")
    check("MT stays 5-level", mt_lv == [1, 2, 3, 4, 5], f"{mt_lv}")
    check("label spaces differ (no reconciliation)", ga_lv != mt_lv)

    print("\n=== 2-3. transfer GA->MT and MT->GA, p=0 ===")
    for src, tgt in [("GA", "MT"), ("MT", "GA")]:
        cross_scale.run_transfer(_Args(
            source=DATA / f"{src.lower()}_records_cleaned_full.csv", src_name=src,
            target=DATA / f"{tgt.lower()}_records_cleaned_full.csv", tgt_name=tgt, output=out))
    df = pd.read_csv(out)
    check("both directions logged", len(df) == 8, f"{len(df)} rows")
    check("all status ok", (df["status"] == "ok").all())
    check("c_index in [0,1]", df["c_index"].between(0, 1).all(),
          f"range {df['c_index'].min():.3f}..{df['c_index'].max():.3f}")
    dummies = df[df["method"].str.contains("dummy")]["c_index"]
    check("dummy == 0.500 exactly", (dummies == 0.5).all(), f"{dummies.tolist()}")
    check("somers_d == 2c-1", np.allclose(df["somers_d"], 2 * df["c_index"] - 1))
    check("src/tgt level counts recorded",
          set(df["src_levels"]) == {3, 5} and set(df["tgt_levels"]) == {3, 5})

    print("\n=== 4-5. split protocol across the p-curve ===")
    from transfer_common import split_target_fewshot
    y = mt[TARGET_COL].to_numpy()
    tests, overlaps = [], []
    for p in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        a, t = split_target_fewshot(y, p, test_frac=0.2, seed=42)
        tests.append(t)
        overlaps.append(len(np.intersect1d(a, t)))
    check("fixed test block identical at every p",
          all(np.array_equal(tests[0], t) for t in tests), f"n={len(tests[0])}")
    check("adaptation never touches the test block", all(o == 0 for o in overlaps))

    print("\n=== 6. within-state ceiling vs transfer ===")
    # Run on WI (3,750 rows -> a 750-row test block), NOT MT. MT's 37-row test
    # block gives a c-index bootstrap std around 0.05, so "ceiling >= transfer"
    # there is inside noise and would be a coin-flip assertion. This is a
    # STATISTICAL expectation, not an invariant, so it is checked on a state
    # large enough for the comparison to mean something and with the logged
    # bootstrap std as the tolerance.
    cross_scale.run_transfer(_Args(
        source=DATA / "ga_records_cleaned_full.csv", src_name="GA",
        target=DATA / "wi_records_cleaned_full.csv", tgt_name="WI",
        output=out, models=["ridge", "rf", "xgb"]))
    cross_scale.run_within(_Args(within=DATA / "wi_records_cleaned_full.csv", state="WI",
                                 output=out, models=["ridge", "rf", "xgb"]))
    df = pd.read_csv(out)
    for m in ("ridge", "rf", "xgb"):
        ceil = df[df["method"] == f"xscale_{m}_reg_within_WI"]["c_index"].iloc[0]
        row = df[df["method"] == f"xscale_{m}_reg_tgt0_GA2WI"]
        xfer, tol = row["c_index"].iloc[0], 2 * row["c_index_std"].iloc[0]
        check(f"{m}: WI ceiling >= GA->WI transfer", ceil >= xfer - tol,
              f"ceiling {ceil:.3f} vs transfer {xfer:.3f} (tol {tol:.3f})")

    print("\n=== 7. ablation + schema ===")
    cross_scale.run_transfer(_Args(
        source=DATA / "wi_records_cleaned_full.csv", src_name="WI",
        target=DATA / "mt_records_cleaned_full.csv", tgt_name="MT",
        coarsen_source=True, output=out, models=["xgb"]))
    df = pd.read_csv(out)
    abl = df[df["label_resolution"] == "coarsened3"]
    check("ablation row written with coarsened3", len(abl) == 1)
    check("ablation trained on a 3-level source", (abl["src_levels"] == 3).all())
    check("ablation still SCORED on 5-level MT", (abl["tgt_levels"] == 5).all())
    need = {"method", "fold", "notes", "source", "target", "status", "c_index",
            "somers_d", "c_index_weighted", "spearman", "n_scored", "n_pairs",
            "src_levels", "tgt_levels", "label_resolution", "target_pct"}
    check("schema complete", need <= set(df.columns), f"missing {need - set(df.columns)}")
    # 8 (GA<->MT, 4 models each) + 3 (GA->WI) + 3 (within WI) + 1 (ablation) = 15
    check("one header only (clean appends)", len(df) == 15, f"{len(df)} rows")

    print("\n=== 7b. prediction dump ===")
    pred_dir = out.parent / "predictions"
    check("predictions/ directory was created", pred_dir.is_dir())
    npz = sorted(pred_dir.glob("*.npz")) if pred_dir.is_dir() else []
    # 12, not 15: the three within-state CEILING rows do not dump. Their scores
    # are per-fold and each fold's model has its own arbitrary output scale, so
    # a single concatenated vector would not be a meaningful ranking.
    check("one .npz per logged TRANSFER row (ceiling rows do not dump)",
          len(npz) == 12, f"{len(npz)} files")
    check("no transfer tag missing its dump",
          {p.stem for p in npz} == set(
              pd.read_csv(out).query("source != target")["method"]),
          "tag set mismatch")
    if npz:
        z = np.load(npz[0])
        check("npz carries y_true + scores, same length",
              set(z.files) == {"y_true", "scores"} and len(z["y_true"]) == len(z["scores"]))
    # A dump failure must never destroy an already-logged result, nor kill the
    # models queued behind it: the .npz is a convenience artifact, the row is the
    # result.
    before = len(pd.read_csv(out))
    cross_scale._dump_safe(Path("/proc/nonexistent/nope"), "bad", [1, 2], [0.1, 0.2])
    cross_scale.run_transfer(_Args(
        source=DATA / "ga_records_cleaned_full.csv", src_name="GA",
        target=DATA / "mt_records_cleaned_full.csv", tgt_name="MT",
        output=out, models=["ridge", "rf"]))
    check("_dump_safe swallows a bad path", True)
    check("run continues past a dump problem", len(pd.read_csv(out)) == before + 2)

    print("\n=== 8. one z-transform per state, shared by both regimes ===")
    # One transform per state (rating_stats), applied identically by the pooled
    # and sequential paths, with the target's stats taken from its fixed
    # non-test pool.
    ga_y = np.array([1, 2, 3, 1, 2, 3], float)
    wi_y = np.array([1, 2, 3, 4, 5, 3], float)
    gs, ws = cross_scale.rating_stats(ga_y), cross_scale.rating_stats(wi_y)
    zg, zw = cross_scale.zscore(ga_y, gs), cross_scale.zscore(wi_y, ws)
    check("each state centred by its own transform",
          abs(zg.mean()) < 1e-12 and abs(zw.mean()) < 1e-12)
    check("each state unit-scaled", abs(zg.std() - 1) < 1e-12 and abs(zw.std() - 1) < 1e-12)
    check("3-level and 5-level land on ONE numeric scale",
          abs(zg.std() - zw.std()) < 1e-12)
    check("transform is strictly monotone (rank-preserving)",
          np.array_equal(np.argsort(np.argsort(zw)), np.argsort(np.argsort(wi_y))))
    check("constant state does not divide by zero",
          np.all(np.isfinite(cross_scale.zscore(np.ones(4), cross_scale.rating_stats(np.ones(4))))))
    # Determinism down the curve: the target transform must not depend on p.
    from transfer_common import split_target_fewshot
    y_mt = mt[TARGET_COL].to_numpy()
    stats_by_p = []
    for pf in (0.2, 0.6, 1.0):
        a, t = split_target_fewshot(y_mt, pf, test_frac=0.2, seed=42)
        pool = np.setdiff1d(np.arange(len(y_mt)), t)
        stats_by_p.append(cross_scale.rating_stats(y_mt[pool]))
    check("target transform identical at every p", len(set(stats_by_p)) == 1,
          f"{stats_by_p[0]}")
    _, t0 = split_target_fewshot(y_mt, 0.0, test_frac=0.2, seed=42)
    pool0 = np.setdiff1d(np.arange(len(y_mt)), t0)
    check("transform uses NO test rows",
          len(np.intersect1d(pool0, t0)) == 0)

    print("\n=== 9. deep-model registry wiring ===")
    import cross_scale_deep as csd
    check("registry covers DEEP_MODEL_NAMES",
          set(csd.DEEP_MODELS) == set(cross_scale.DEEP_MODEL_NAMES),
          f"{sorted(csd.DEEP_MODELS)}")
    check("every entry declares input/sequential/fn",
          all({"input", "sequential", "fn"} <= set(v) for v in csd.DEEP_MODELS.values()))
    check("inputs are emb|text",
          all(v["input"] in ("emb", "text") for v in csd.DEEP_MODELS.values()))
    check("tabpfn is the only non-sequential deep model",
          [k for k, v in csd.DEEP_MODELS.items() if not v["sequential"]] == ["tabpfn"])
    check("only qwen_rag needs retrieval embeddings",
          [k for k, v in csd.DEEP_MODELS.items() if v.get("needs_emb")] == ["qwen_rag"])
    check("stage map covers the roster",
          set(csd.STAGE) == set(cross_scale.DEEP_MODEL_NAMES)
          and set(csd.STAGE.values()) == {2, 3, 4})
    # The dispatcher must build SINGLE-SCALE phases: pretrain=source only,
    # finetune=target adaptation only. Verified by intercepting the registry.
    seen = {}

    def _spy(pre, ft, test_X, **kw):
        # Levels come from y_raw now; y is the z-scored training target.
        seen.update(pre_levels=sorted(set(pre.y_raw.tolist())),
                    ft_levels=sorted(set(ft.y_raw.tolist())) if ft is not None else None,
                    pre_std=(float(np.mean(pre.y)), float(np.std(pre.y))),
                    ft_std=(float(np.mean(ft.y)), float(np.std(ft.y))) if ft is not None else None,
                    n_test=len(test_X))
        return np.arange(len(test_X), dtype=float)

    csd.DEEP_MODELS["tabnet"] = {"input": "emb", "sequential": True, "fn": _spy}
    cross_scale.run_transfer(_Args(
        source=DATA / "ga_records_cleaned_full.csv", src_name="GA",
        target=DATA / "mt_records_cleaned_full.csv", tgt_name="MT",
        target_frac=0.4, output=out, models=["tabnet"]))
    check("pretrain phase is source-only (GA 3-level)", seen.get("pre_levels") == [1, 2, 3],
          f"{seen.get('pre_levels')}")
    check("finetune phase is target-only (MT 5-level)",
          seen.get("ft_levels") is not None and max(seen["ft_levels"]) == 5,
          f"{seen.get('ft_levels')}")
    check("no phase mixes the two scales",
          seen.get("pre_levels") != seen.get("ft_levels"))
    # Source uses ALL its rows, so it standardizes exactly. The finetune phase is
    # the p% DRAW transformed by the pool's statistics, so it lands NEAR (0, 1)
    # but not exactly -- which is the proof it used the fixed pool transform
    # rather than recomputing its own from the draw.
    pm, psd = seen["pre_std"]; fm, fsd = seen["ft_std"]
    check("source phase standardizes exactly (uses all its rows)",
          abs(pm) < 1e-9 and abs(psd - 1) < 1e-9, f"({pm:.2e}, {psd:.6f})")
    check("both phases land on ONE numeric scale",
          abs(fm) < 0.15 and abs(fsd - 1) < 0.15, f"ft=({fm:.4f}, {fsd:.4f})")
    check("finetune used the fixed POOL transform, not the draw's own stats",
          abs(fm) > 1e-9 or abs(fsd - 1) > 1e-9)
    seen.clear()
    cross_scale.run_transfer(_Args(
        source=DATA / "ga_records_cleaned_full.csv", src_name="GA",
        target=DATA / "mt_records_cleaned_full.csv", tgt_name="MT",
        target_frac=0.0, output=out, models=["tabnet"]))
    check("p=0 runs ONE phase (zero-shot purity)", seen.get("ft_levels") is None)

    print("\n=== 9b. ordered loss-free inference ===")
    # Checks the two properties predict_scores_ordered must have: output order
    # equals input order, and no labels anywhere. The only torch use inside it is
    # `torch.no_grad()`, stubbed here so the real batching and ordering code runs
    # even where torch is absent.
    import contextlib
    import sys
    import types
    stub = None
    if "torch" not in sys.modules:
        stub = types.ModuleType("torch")
        stub.no_grad = contextlib.nullcontext
        sys.modules["torch"] = stub

    class _Vec(list):
        """Minimal stand-in for a logits tensor: .float()/.cpu() are identity."""
        def float(self): return self
        def cpu(self): return self

    class _Ids(list):
        """Stand-in for a token tensor: .to(device) is identity."""
        def to(self, device): return self

    class _FakeTok:
        def __call__(self, chunk, **kw):
            return {"input_ids": _Ids(chunk)}

    class _FakeModel:
        """Emits each row's TRUE global index, so any reordering is visible."""
        cursor = 0

        def eval(self): return self

        def parameters(self):
            return iter([types.SimpleNamespace(device="cpu")])

        def __call__(self, input_ids=None, **kw):
            n = len(input_ids)
            out = _Vec(range(_FakeModel.cursor, _FakeModel.cursor + n))
            _FakeModel.cursor += n
            return types.SimpleNamespace(logits=out)

    try:
        texts = [f"row{i}" for i in range(23)]
        _FakeModel.cursor = 0
        got = csd.predict_scores_ordered(_FakeModel(), _FakeTok(), texts, 128, 5,
                                         lambda lg: lg)
        check("output order == input order", np.array_equal(got, np.arange(23)),
              f"{got[:6].astype(int).tolist()}...")
        check("ragged final batch handled (23 rows, batch 5)", len(got) == 23)
        _FakeModel.cursor = 0
        short = csd.predict_scores_ordered(_FakeModel(), _FakeTok(), texts[:3], 128, 5,
                                           lambda lg: lg)
        check("single partial batch works", np.array_equal(short, np.arange(3)))
        _FakeModel.cursor = 0
        check("row-count mismatch raises rather than returning short",
              _raises(lambda: csd.predict_scores_ordered(
                  _FakeModel(), _FakeTok(), texts, 128, 5, lambda lg: lg[:1])))
    finally:
        if stub is not None:
            del sys.modules["torch"]

    print("\n=== 9c. trainable-only best-model snapshot ===")
    # llm.make_best_model_keeper clones the FULL state_dict, which is 15 GB for
    # fp32 Qwen3-4B and will OOM a modest node. make_trainable_keeper snapshots
    # trainable params only, which is equivalent: a frozen parameter cannot
    # change during training.
    class _P:
        def __init__(self, req, n):
            self.requires_grad, self._n = req, n

        def detach(self): return self

        def cpu(self): return self

        def clone(self): return self

    class _M:
        def named_parameters(self):
            return [("backbone.w", _P(False, 4_000_000_000)),
                    ("score.weight", _P(True, 2560)),
                    ("score.bias", _P(True, 1))]

    # `transformers` is only needed for the TrainerCallback base class; stub it
    # so the keeper's real snapshot/patience logic runs anywhere.
    tf_stub = None
    if "transformers" not in sys.modules:
        tf_stub = types.ModuleType("transformers")
        tf_stub.TrainerCallback = type("TrainerCallback", (), {})
        sys.modules["transformers"] = tf_stub

    k = csd.make_trainable_keeper(metric="eval_c_index", patience=2)
    k.on_evaluate(None, None, None, metrics={"eval_c_index": 0.7}, model=_M())
    snap = set(k.best_state)
    check("snapshot holds ONLY trainable params", snap == {"score.weight", "score.bias"},
          f"{sorted(snap)}")
    check("frozen backbone excluded from the snapshot", "backbone.w" not in snap)

    class _Ctl:
        should_training_stop = False

    ctl = _Ctl()
    k.on_evaluate(None, None, ctl, metrics={"eval_c_index": 0.6}, model=_M())
    check("no early stop after 1 bad epoch (patience 2)", not ctl.should_training_stop)
    k.on_evaluate(None, None, ctl, metrics={"eval_c_index": 0.5}, model=_M())
    check("early-stops after patience is exhausted", ctl.should_training_stop)
    k.on_evaluate(None, None, ctl, metrics={"eval_c_index": 0.9}, model=_M())
    check("best_state tracks the best epoch", abs(k.best - 0.9) < 1e-12)
    qsrc = __import__("inspect").getsource(csd.fit_predict_qwen)
    check("restore uses strict=False (trainable-only payload)",
          "strict=False" in qsrc)
    check("qwen loads with low_cpu_mem_usage", "low_cpu_mem_usage=True" in qsrc)
    check("qwen does not use the full-state_dict keeper",
          "make_best_model_keeper" not in qsrc)
    if tf_stub is not None:
        del sys.modules["transformers"]

    print("\n=== 9d. phases carry standardized y + raw levels ===")
    dsrc = __import__("inspect").getsource(cross_scale._run_deep)
    check("phases are built with the caller's z-transform",
          "zscore(y_src, src_stats)" in dsrc and "zscore(y_tgt[adapt_idx], tgt_stats)" in dsrc)
    check("phases keep raw levels for stratification",
          "y_raw=y_src" in dsrc and "y_raw=y_tgt[adapt_idx]" in dsrc)
    for fn in (csd.fit_predict_mbert, csd.fit_predict_qwen, csd.fit_predict_tabnet):
        fsrc = __import__("inspect").getsource(fn)
        check(f"{fn.__name__} stratifies on y_raw", "_inner_val(phase.y_raw" in fsrc)
        check(f"{fn.__name__} does not re-standardize", "phase_targets" not in fsrc)
    # Code only — the module keeps a comment recording why phase_targets went away.
    csd_code = "\n".join(ln for ln in __import__("inspect").getsource(csd).splitlines()
                         if not ln.lstrip().startswith("#"))
    check("no model re-standardizes an already-standardized phase",
          "phase_targets(" not in csd_code)
    check("mbert_corn is gone from the roster",
          "mbert_corn" not in csd.DEEP_MODELS and "mbert_corn" not in cross_scale.DEEP_MODEL_NAMES)
    check("mbert_corn is gone from the stage map", "mbert_corn" not in csd.STAGE)

    # Grep the CODE, not the docstrings that explain the fix.
    src = [ln for ln in __import__("inspect").getsource(csd).splitlines()
           if not ln.lstrip().startswith("#")]
    code = "\n".join(src)
    check("no predict_with_labels CALL remains", "predict_with_labels(" not in code)
    check("no arange probe remains",
          "np.arange(len(test_texts)" not in code and "np.arange(len(test_in)" not in code)
    check("qwen imports DEFAULT_CLS_LLM_PATH, not the nonexistent LLM_CLS_MODEL",
          "DEFAULT_CLS_LLM_PATH" in code and "LLM_CLS_MODEL," not in code)

    print("\n=== 10. metric invariance sanity on real labels ===")
    rng = np.random.RandomState(0)
    s = y + rng.normal(0, 1, len(y))
    a = compute_ranking_metrics(y, s)
    b = compute_ranking_metrics(y, 5 * s + 3)              # monotone affine
    c = compute_ranking_metrics(y, np.exp(s / 3))          # monotone nonlinear
    check("c-index invariant to monotone score transforms",
          abs(a["c_index"] - b["c_index"]) < 1e-12 and abs(a["c_index"] - c["c_index"]) < 1e-12)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'PASS' if ok else 'FAIL'} — cross_scale verification (Stage 1 end-to-end + Stages 2-4 wiring)\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
