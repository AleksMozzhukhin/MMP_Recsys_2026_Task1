"""Shared input validation and deterministic recommendation generation."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def binary_csr(matrix, *, name: str = "interactions") -> sparse.csr_matrix:
    """Return sorted binary CSR, interpreting every positive entry as observed."""
    result = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if not np.all(np.isfinite(result.data)) or np.any(result.data < 0):
        raise ValueError(f"{name} must contain finite nonnegative entries")
    result.data.fill(1.0)
    return result


class Recommender:
    """Known-user full-catalog scorer; smaller item IDs resolve exact ties."""

    def _set_shape(self, matrix: sparse.csr_matrix) -> None:
        self.n_users_, self.n_items_ = matrix.shape
        if min(matrix.shape) < 1:
            raise ValueError("The interaction matrix must have users and items")

    def _user_ids(self, user_ids) -> np.ndarray:
        if not hasattr(self, "n_users_"):
            raise RuntimeError("Call fit before scoring")
        values = np.asarray(user_ids)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError("user_ids must be a one-dimensional integer array")
        if np.any(values < 0) or np.any(values >= self.n_users_):
            raise ValueError("user_ids contains an unknown user")
        return values.astype(np.int64, copy=False)

    def recommend(self, user_ids, seen, k: int = 10, tie_break=None) -> np.ndarray:
        """Return top-k IDs after masking *all* observed training interactions.

        ``seen`` has the complete fitted user/item shape, including low ratings.
        Equal scores are ordered by descending ``tie_break`` (e.g. training
        popularity), then by ascending item ID. Without ``tie_break`` only the ID
        is used. Every requested user must have at least k unseen items.
        """
        users = self._user_ids(user_ids)
        if not isinstance(k, (int, np.integer)) or not 1 <= k <= self.n_items_:
            raise ValueError("k must be between 1 and the catalog size")
        seen = binary_csr(seen, name="seen")
        if seen.shape != (self.n_users_, self.n_items_):
            raise ValueError("seen must have the fitted user/item shape")
        priority = np.zeros(self.n_items_) if tie_break is None else np.asarray(tie_break, float)
        if priority.shape != (self.n_items_,):
            raise ValueError("tie_break must have one value per item")
        selected_seen = seen[users]
        if np.any(self.n_items_ - np.diff(selected_seen.indptr) < k):
            raise ValueError("A requested user has fewer than k eligible items")
        scores = np.array(self.score(users), dtype=np.float64, copy=True)
        if scores.shape != (len(users), self.n_items_):
            raise ValueError("score must return a batch by catalog matrix")
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("Nonfinite model scores")
        rows, columns = selected_seen.nonzero()
        scores[rows, columns] = -np.inf
        result = np.empty((len(users), k), dtype=np.int64)
        for row, values in enumerate(scores):
            threshold = np.partition(values, self.n_items_ - k)[self.n_items_ - k]
            above = np.flatnonzero(values > threshold)
            tied = np.flatnonzero(values == threshold)
            tied = tied[np.lexsort((tied, -priority[tied]))][: k - len(above)]
            candidates = np.concatenate((above, tied))
            order = np.lexsort((candidates, -priority[candidates], -values[candidates]))
            result[row] = candidates[order]
        return result
