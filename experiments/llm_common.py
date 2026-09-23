"""Exemplar selection shared by the RAG classifier variants."""
from __future__ import annotations

import numpy as np


def class_medoid_exemplar_indices(pool_emb, pool_labels, exclude=None):
    """Return one medoid pool index per class, in ascending class order.

    Assumes L2-normalized embeddings (argmax against the centroid = max mean cosine).
    `exclude` drops pool indices so a row is never its own exemplar; empty classes are skipped.
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
