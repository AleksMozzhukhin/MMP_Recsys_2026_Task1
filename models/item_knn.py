"""Item-based collaborative filtering with shrunk cosine similarity."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .base import Recommender, binary_csr


class ItemKNN(Recommender):
    """Each liked source item votes for its k most similar other items.

    For binary positive matrix X, let C = X.T X and n_i = C_ii.
    Similarity is C_ij / sqrt(n_i n_j) * C_ij / (C_ij + shrinkage).
    The diagonal is zero. Row pruning retains at most k neighbors per source
    item; equal similarities prefer the lower target-item ID. Scores are X S.
    """

    def __init__(self, k: int = 100, shrinkage: float = 10.0):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be finite and nonnegative")
        self.k = int(k)
        self.shrinkage = float(shrinkage)

    def fit(self, positive):
        self.positive_ = binary_csr(positive, name="positive")
        self._set_shape(self.positive_)
        cooccurrence = (self.positive_.T @ self.positive_).tocsr()
        counts = cooccurrence.diagonal().copy()
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()
        cooccurrence.sort_indices()
        row_ids = np.repeat(np.arange(self.n_items_), np.diff(cooccurrence.indptr))
        values = cooccurrence.data.copy()
        denominator = np.sqrt(counts[row_ids] * counts[cooccurrence.indices])
        cooccurrence.data = values / denominator
        if self.shrinkage:
            cooccurrence.data *= values / (values + self.shrinkage)
        indptr = [0]
        indices, weights = [], []
        for item in range(self.n_items_):
            lo, hi = cooccurrence.indptr[item : item + 2]
            neighbors = cooccurrence.indices[lo:hi]
            similarity = cooccurrence.data[lo:hi]
            keep = np.lexsort((neighbors, -similarity))[: self.k]
            # Keep CSR rows sorted for subsequent sparse arithmetic.
            keep = keep[np.argsort(neighbors[keep])]
            indices.extend(neighbors[keep])
            weights.extend(similarity[keep])
            indptr.append(len(indices))
        self.similarity_ = sparse.csr_matrix(
            (weights, indices, indptr), shape=(self.n_items_, self.n_items_),
            dtype=np.float64,
        )
        return self

    def score(self, user_ids) -> np.ndarray:
        users = self._user_ids(user_ids)
        return (self.positive_[users] @ self.similarity_).toarray()
