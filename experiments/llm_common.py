"""Exemplar selection shared by the RAG classifier variants."""
from __future__ import annotations

import numpy as np


def class_medoid_exemplar_indices(pool_emb, pool_labels, exclude=None):
    """Return one medoid index per class in `pool_emb`, in ascending class order.

    The medoid is taken as the member most aligned with its class centroid, which
    is equivalent to maximising mean cosine similarity to classmates because the
    embeddings are L2-normalized.

    `exclude` omits pool indices, so a row is never its own exemplar. Classes with
    no eligible member are skipped.
    """
    pool_labels = np.asarray(pool_labels)
    ex = set(int(i) for i in exclude) if exclude is not None else set()
    out = []
    for c in sorted(np.unique(pool_labels).tolist()):
        idx = np.where(pool_labels == c)[0]
        if ex:
            idx = np.array([i for i in idx if int(i) not in ex], dtype=int)
        if len(idx) == 0:
            continue
        sub = pool_emb[idx]
        mu = sub.mean(axis=0)
        out.append(int(idx[int(np.argmax(sub @ mu))]))
    return out
