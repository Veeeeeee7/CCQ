"""Rank-association metrics for cross-rubric transfer.

Scores a continuous prediction against ordinal truth without assuming any
correspondence between the two label scales.

    c = (C + 0.5 * T_x) / (C + D + T_x)      over pairs untied on truth

Also provides Somers' D (2c - 1), a distance-weighted c, and Spearman.

Computed exactly rather than by enumerating pairs: truth takes K <= 5 values, so
the pair set partitions into K(K-1)/2 blocks and each block's concordance is a
Mann-Whitney U over that block's scores. Cost is O(K^2 n log n).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import rankdata, spearmanr

# Mirrors the role of transfer_common._METRIC_KEYS for the ranking suite: the
# ordered key list every logger / bootstrap iterates. `c_index` is PRIMARY.
RANKING_METRIC_KEYS = [
    "c_index",            # PRIMARY — P(correctly ordered | state rated them differently)
    "somers_d",           # 2*c_index - 1; signed [-1, 1] presentation of the same number
    "c_index_weighted",   # distance-weighted; SECONDARY, scale-dependent (see below)
    "spearman",           # familiar third; midrank-corrected but ceiling-limited by ties
    "n_scored",           # rows in the scored block
    "n_pairs",            # pairs the state actually orders (the c-index denominator)
]


def _auc_by_class_pair(y_true: np.ndarray, scores: np.ndarray):
    """Yield ``(gap, n_pairs, auc)`` for every ordered class pair (a < b).

    ``auc`` = P(score_b > score_a) + 0.5 * P(score_b == score_a), computed with
    the Mann-Whitney U identity on average ranks. ``gap`` = b - a in the TARGET's
    own label units; ``n_pairs`` = n_a * n_b.
    """
    classes = np.unique(y_true)
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            sa = scores[y_true == a]
            sb = scores[y_true == b]
            n_a, n_b = len(sa), len(sb)
            if n_a == 0 or n_b == 0:
                continue
            # Rank the two groups jointly; ties get average ranks, which is what
            # turns "tied score" into exactly half credit.
            r = rankdata(np.concatenate([sa, sb]))
            u_b = r[n_a:].sum() - n_b * (n_b + 1) / 2.0
            yield float(b - a), n_a * n_b, u_b / (n_a * n_b)


def concordance_index(y_true, scores, *, distance_weighted: bool = False) -> float:
    """Probability the model orders a truth-untied pair the way the state did.

    ``distance_weighted=True`` weights each pair by |y_i - y_j|, restoring the
    magnitude sensitivity QWK's quadratic weights carry (getting a 1-vs-5 pair
    backwards is worse than a 1-vs-2 pair). SECONDARY ONLY: the weights live in
    the target's own label units, so a 5-level target's max gap is 4 and a
    3-level target's is 2 — it partially reintroduces the cross-scale
    incomparability the plain c-index exists to avoid.

    Returns NaN when the state rated every scored row identically (no pair
    carries an ordering, so there is nothing to measure).
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs scores {scores.shape}")
    finite = np.isfinite(y_true) & np.isfinite(scores)
    if not finite.all():
        # A non-finite prediction is a real failure, not something to silently
        # drop: say how many, then score what is left.
        warnings.warn(f"{(~finite).sum()} non-finite y/score rows dropped from c-index",
                      RuntimeWarning, stacklevel=2)
        y_true, scores = y_true[finite], scores[finite]

    num = den = 0.0
    for gap, n_pairs, auc in _auc_by_class_pair(y_true, scores):
        w = (gap if distance_weighted else 1.0) * n_pairs
        num += w * auc
        den += w
    if den == 0:
        return float("nan")
    return float(num / den)


def somers_d(y_true, scores) -> float:
    """Somers' D_yx = 2*c - 1. Same statistic as the c-index on a signed scale:
    +1 perfect, 0 chance, -1 perfectly inverted."""
    return 2.0 * concordance_index(y_true, scores) - 1.0


def n_ordered_pairs(y_true) -> int:
    """Number of pairs the state actually orders — the c-index denominator.
    Reported so a reader can see how much of the data carries ordering signal
    (WA: 34% of pairs; OK: 77%)."""
    _, counts = np.unique(np.asarray(y_true), return_counts=True)
    n = int(counts.sum())
    return int(n * (n - 1) // 2 - (counts * (counts - 1) // 2).sum())


def compute_ranking_metrics(y_true, scores) -> dict[str, float]:
    """Full ranking metric suite for one scored block. Mirrors the dict-return
    shape of ``utils.compute_metrics`` so the logging path is interchangeable.

    ``scores`` is a CONTINUOUS latent-quality prediction, NOT a class label —
    nothing here rounds, thresholds, or maps it onto the target's scale.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    c = concordance_index(y_true, scores)
    with warnings.catch_warnings():
        # Constant input (the `dummy` floor) makes Spearman undefined; c-index
        # still returns a clean 0.5, which is exactly why it is the primary.
        warnings.simplefilter("ignore")
        rho = spearmanr(y_true, scores).statistic
    return {
        "c_index": c,
        "somers_d": 2.0 * c - 1.0,
        "c_index_weighted": concordance_index(y_true, scores, distance_weighted=True),
        "spearman": float(rho) if rho is not None else float("nan"),
        "n_scored": float(len(y_true)),
        "n_pairs": float(n_ordered_pairs(y_true)),
    }


def bootstrap_ranking_std(y_true, scores, n_boot: int = 1000,
                          seed: int = 42) -> dict[str, float]:
    """Resample scored rows with replacement; return each metric's std.

    The ranking analogue of ``transfer_common.bootstrap_target_std``, and the
    same caveat applies as there: this isolates EVALUATION uncertainty only.
    Note pairs are not independent under row resampling, so these are the usual
    approximate c-index error bars, not exact ones.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    acc: dict[str, list[float]] = {k: [] for k in RANKING_METRIC_KEYS}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            m = compute_ranking_metrics(y_true[idx], scores[idx])
            for k in RANKING_METRIC_KEYS:
                acc[k].append(m[k])
    return {k: float(np.nanstd(v)) for k, v in acc.items()}


# -----------------------------------------------------------------------------
# Self-test — `python ranking_metrics.py`
# -----------------------------------------------------------------------------
def _self_test() -> None:
    """Properties that must hold, checked against brute-force pair enumeration."""
    rng = np.random.RandomState(0)

    def brute(y, s):
        """O(n^2) definition, straight from the formula. The reference."""
        y, s = np.asarray(y, float), np.asarray(s, float)
        C = D = T = 0
        for i in range(len(y)):
            dy, ds = y[i] - y[i + 1:], s[i] - s[i + 1:]
            m = dy != 0
            dy, ds = dy[m], ds[m]
            C += int(((dy > 0) & (ds > 0)).sum() + ((dy < 0) & (ds < 0)).sum())
            D += int(((dy > 0) & (ds < 0)).sum() + ((dy < 0) & (ds > 0)).sum())
            T += int((ds == 0).sum())
        return (C + 0.5 * T) / (C + D + T)

    ok = True
    for K, n in [(3, 300), (5, 400), (4, 250)]:
        y = rng.randint(1, K + 1, n)
        for name, s in [("noisy", y + rng.normal(0, 1.2, n)),
                        ("perfect", y + rng.uniform(-.01, .01, n)),
                        ("inverted", -y + rng.uniform(-.01, .01, n)),
                        ("constant", np.zeros(n)),
                        ("ties", np.round(y + rng.normal(0, 1, n)))]:
            fast, ref = concordance_index(y, s), brute(y, s)
            match = abs(fast - ref) < 1e-9
            ok &= match
            print(f"  K={K} {name:9s} c={fast:.6f} brute={ref:.6f} "
                  f"{'ok' if match else 'MISMATCH'}")

    y = rng.randint(1, 6, 500)
    perfect = y + rng.uniform(-.01, .01, 500)
    checks = [
        ("perfect ranker c == 1.0", abs(concordance_index(y, perfect) - 1.0) < 1e-12),
        ("inverted ranker c == 0.0", abs(concordance_index(y, -perfect)) < 1e-12),
        ("constant c == 0.5 exactly", concordance_index(y, np.zeros(500)) == 0.5),
        ("somers_d == 2c-1", abs(somers_d(y, perfect)
                                 - (2 * concordance_index(y, perfect) - 1)) < 1e-12),
        ("monotone-invariant", abs(concordance_index(y, perfect)
                                   - concordance_index(y, np.exp(perfect))) < 1e-12),
        ("label-rescale-invariant", abs(concordance_index(y, perfect)
                                        - concordance_index(3 * y + 7, perfect)) < 1e-12),
        ("single-class -> NaN", np.isnan(concordance_index(np.ones(10), rng.rand(10)))),
    ]
    for name, passed in checks:
        ok &= passed
        print(f"  {name:28s} {'ok' if passed else 'FAILED'}")

    # The property the whole design rests on: c-index is invariant to the label
    # SCALE, so a 3-level and a 5-level view of the same ordering agree.
    print(f"\n  {'PASS' if ok else 'FAIL'} — ranking_metrics self-test")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    _self_test()
