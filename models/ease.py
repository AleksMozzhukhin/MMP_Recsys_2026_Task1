"""EASE: dense linear reconstruction with a zero coefficient diagonal."""

from __future__ import annotations

import numpy as np
from scipy import linalg

from .base import Recommender, binary_csr


class EASE(Recommender):
    """Minimize ||X-XB||_F^2 + l2 ||B||_F^2 subject to diag(B)=0.

    With P=(X.T X + l2 I)^(-1), B_ij=-P_ij/P_jj for i != j.
    This implementation solves the positive-definite system directly; it uses
    no recommender or learned regression implementation. Memory is O(items^2).
    """

    def __init__(self, l2: float = 100.0):
        if not np.isfinite(l2) or l2 <= 0:
            raise ValueError("l2 must be finite and positive")
        self.l2 = float(l2)

    def fit(self, positive):
        self.positive_ = binary_csr(positive, name="positive")
        self._set_shape(self.positive_)
        gram = (self.positive_.T @ self.positive_).toarray()
        gram.flat[:: self.n_items_ + 1] += self.l2
        factor = linalg.cho_factor(gram, lower=True, check_finite=False, overwrite_a=True)
        precision = linalg.cho_solve(
            factor, np.eye(self.n_items_), check_finite=False, overwrite_b=True
        )
        diagonal = np.diag(precision).copy()
        precision /= -diagonal[np.newaxis, :]
        np.fill_diagonal(precision, 0.0)
        self.coefficients_ = precision
        return self

    def score(self, user_ids) -> np.ndarray:
        users = self._user_ids(user_ids)
        return np.asarray(self.positive_[users] @ self.coefficients_)
