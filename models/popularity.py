"""Non-personalized reference: most popular items, optionally in a recent window."""

from __future__ import annotations

import numpy as np

from .base import Recommender


class Popularity(Recommender):
    """Score every item by its count of positive reviews in the chosen window.

    ``counts`` is computed by the caller from training history before the cutoff,
    e.g. positives in the last 180 days, or over all time.
    """

    def fit(self, counts, n_users: int):
        self.counts_ = np.asarray(counts, dtype=np.float64)
        self.n_users_, self.n_items_ = int(n_users), len(self.counts_)
        return self

    def score(self, user_ids) -> np.ndarray:
        return np.broadcast_to(self.counts_, (len(self._user_ids(user_ids)), self.n_items_))
