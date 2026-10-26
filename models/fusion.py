"""Weighted sum of per-user standardized scores of already fitted recommenders."""

from __future__ import annotations

import numpy as np

from .base import Recommender


class ScoreFusion(Recommender):
    """score(u) = sum_m w_m z_m(u), z = (s - mean_i s) / std_i s over the catalog.

    Standardization puts models with different score scales on one footing.
    """

    def __init__(self, models, weights):
        if len(models) != len(weights) or not models:
            raise ValueError("one weight per model is required")
        self.models = list(models)
        self.weights = [float(w) for w in weights]
        self.n_users_, self.n_items_ = models[0].n_users_, models[0].n_items_

    def score(self, user_ids) -> np.ndarray:
        users = self._user_ids(user_ids)
        total = np.zeros((len(users), self.n_items_))
        for model, weight in zip(self.models, self.weights):
            scores = np.asarray(model.score(users), dtype=np.float64)
            std = scores.std(axis=1, keepdims=True)
            total += weight * (scores - scores.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1)
        return total
