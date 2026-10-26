"""Implicit-feedback ALS with conjugate-gradient or exact alternating ridge solves."""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse

from .base import Recommender, binary_csr


class IALS(Recommender):
    """Minimize sum_ui c_ui (p_ui - x_u.T y_i)^2 + l2 (||X||^2+||Y||^2).

    ``observed`` marks every training review, including disliked objects.
    Confidence c_ui = 1 + alpha * observed_ui; all unobserved targets are zero.
    ``targets`` specifies p_ui and may be binary, real or signed. Its nonzero
    support must be contained in observed. If omitted, targets equal observed.

    Row normal equations: (Y.T Y + alpha Y_O.T Y_O + l2 I) x_u = (1+alpha) Y.T p_u.
    ``cg_steps=None`` solves them exactly row by row. An integer runs that many
    conjugate-gradient steps for all rows at once, warm-started from the
    previous factors (Takacs, Pilaszy, Tikk, RecSys 2011). Each CG step costs
    O(nnz f + n f^2) and never forms the f x f per-row matrices.
    """

    def __init__(
        self, n_factors: int = 32, alpha: float = 20.0, l2: float = 1.0,
        n_iters: int = 10, seed: int = 42, cg_steps: int | None = 3,
    ):
        if not isinstance(n_factors, (int, np.integer)) or n_factors < 1:
            raise ValueError("n_factors must be a positive integer")
        if not isinstance(n_iters, (int, np.integer)) or n_iters < 1:
            raise ValueError("n_iters must be a positive integer")
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if not np.isfinite(l2) or l2 <= 0:
            raise ValueError("l2 must be finite and positive")
        if cg_steps is not None and (not isinstance(cg_steps, (int, np.integer)) or cg_steps < 1):
            raise ValueError("cg_steps must be None or a positive integer")
        self.n_factors = int(n_factors)
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.n_iters = int(n_iters)
        self.seed = seed
        self.cg_steps = None if cg_steps is None else int(cg_steps)

    def _update_factors(self, fixed, observed, targets, previous=None):
        if self.cg_steps is None or previous is None:
            return self._solve_exact(fixed, observed, targets)
        return self._solve_cg(fixed, observed, targets, previous)

    def _solve_exact(self, fixed, observed, targets):
        result = np.empty((observed.shape[0], self.n_factors), dtype=np.float64)
        shared = fixed.T @ fixed + self.l2 * np.eye(self.n_factors)
        right = (1.0 + self.alpha) * np.asarray(targets @ fixed)
        for row in range(observed.shape[0]):
            lo, hi = observed.indptr[row : row + 2]
            selected = fixed[observed.indices[lo:hi]]
            result[row] = linalg.solve(
                shared + self.alpha * (selected.T @ selected), right[row],
                assume_a="pos", check_finite=False, overwrite_a=True,
            )
        return result

    def _solve_cg(self, fixed, observed, targets, previous):
        shared = fixed.T @ fixed + self.l2 * np.eye(self.n_factors)
        rows = np.repeat(np.arange(observed.shape[0]), np.diff(observed.indptr))
        columns = observed.indices
        selected = fixed[columns]

        def normal_times(vectors):
            # (shared + alpha Y_O.T Y_O) v for every row, via the sparse support.
            weights = np.einsum("ij,ij->i", vectors[rows], selected)
            support = sparse.csr_matrix((weights, columns, observed.indptr), shape=observed.shape)
            return vectors @ shared + self.alpha * (support @ fixed)

        solution = previous.copy()
        residual = (1.0 + self.alpha) * np.asarray(targets @ fixed) - normal_times(solution)
        direction = residual.copy()
        residual_norm = np.einsum("ij,ij->i", residual, residual)
        for _ in range(self.cg_steps):
            product = normal_times(direction)
            curvature = np.einsum("ij,ij->i", direction, product)
            step = np.divide(residual_norm, curvature, out=np.zeros_like(curvature),
                             where=curvature > 0)
            solution += step[:, None] * direction
            residual -= step[:, None] * product
            new_norm = np.einsum("ij,ij->i", residual, residual)
            ratio = np.divide(new_norm, residual_norm, out=np.zeros_like(new_norm),
                              where=residual_norm > 0)
            direction = residual + ratio[:, None] * direction
            residual_norm = new_norm
        return solution

    def fit(self, observed, targets=None):
        self.observed_ = binary_csr(observed, name="observed")
        self._set_shape(self.observed_)
        if targets is None:
            targets = self.observed_
        self.targets_ = sparse.csr_matrix(targets, dtype=np.float64, copy=True)
        self.targets_.sum_duplicates()
        self.targets_.eliminate_zeros()
        self.targets_.sort_indices()
        if self.targets_.shape != self.observed_.shape:
            raise ValueError("targets and observed must have identical shapes")
        if not np.all(np.isfinite(self.targets_.data)):
            raise ValueError("targets must be finite")
        outside = self.targets_ - self.targets_.multiply(self.observed_)
        outside.eliminate_zeros()
        if outside.nnz:
            raise ValueError("Nonzero targets must correspond to observed entries")
        rng = np.random.default_rng(self.seed)
        self.user_factors_ = rng.normal(0, 0.01, (self.n_users_, self.n_factors))
        self.item_factors_ = rng.normal(0, 0.01, (self.n_items_, self.n_factors))
        transposed_observed = self.observed_.T.tocsr()
        transposed_targets = self.targets_.T.tocsr()
        self.loss_history_ = [self.objective()]
        for _ in range(self.n_iters):
            self.user_factors_ = self._update_factors(
                self.item_factors_, self.observed_, self.targets_, self.user_factors_
            )
            self.item_factors_ = self._update_factors(
                self.user_factors_, transposed_observed, transposed_targets, self.item_factors_
            )
            self.loss_history_.append(self.objective())
        return self

    def objective(self) -> float:
        """Exact full-catalog weighted objective, using Gram and sparse identities."""
        users, items = self.user_factors_, self.item_factors_
        value = np.sum((users.T @ users) * (items.T @ items))
        # Add alpha * prediction^2 on all observed coordinates, including p=0.
        rows, columns = self.observed_.nonzero()
        predictions = np.einsum("ij,ij->i", users[rows], items[columns])
        value += self.alpha * np.dot(predictions, predictions)
        rows, columns = self.targets_.nonzero()
        predictions = np.einsum("ij,ij->i", users[rows], items[columns])
        target_values = self.targets_.data
        value += (1.0 + self.alpha) * (
            np.dot(target_values, target_values)
            - 2.0 * np.dot(target_values, predictions)
        )
        value += self.l2 * (np.sum(users * users) + np.sum(items * items))
        return float(value)

    def score(self, user_ids) -> np.ndarray:
        users = self._user_ids(user_ids)
        return self.user_factors_[users] @ self.item_factors_.T
