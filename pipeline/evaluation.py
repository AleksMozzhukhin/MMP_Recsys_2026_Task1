"""Common evaluation cohort, top-K metrics and paired bootstrap."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .data import POSITIVE_RATING, Phase


@dataclass
class EvaluationSet:
    """Users with at least one new positive holdout item; identical for every model.

    ``relevant``: holdout pairs with rating >= 4 that are absent from history.
    ``seen``: every history review (any rating); these items are never recommended.
    ``tie_break``: historical positive count, orders items with equal scores.
    """

    users: np.ndarray
    relevant: sparse.csr_matrix
    seen: sparse.csr_matrix
    tie_break: np.ndarray
    k: int

    @classmethod
    def from_phase(cls, phase: Phase, k: int = 10) -> "EvaluationSet":
        seen = phase.matrix("observed")
        liked = phase.holdout[phase.holdout.rating >= POSITIVE_RATING]
        relevant = sparse.csr_matrix((np.ones(len(liked)), (liked.user, liked.item)), shape=phase.shape)
        relevant = (relevant - relevant.multiply(seen)).tocsr()
        relevant.eliminate_zeros()
        relevant.data[:] = 1.0
        enough = phase.n_items - np.diff(seen.indptr) >= k
        users = np.flatnonzero((np.diff(relevant.indptr) > 0) & enough)
        return cls(users, relevant, seen, phase.popularity(), k)

    def report(self) -> dict:
        counts = np.diff(self.relevant.indptr)[self.users]
        return {"users": len(self.users), "relevant_pairs": int(counts.sum()),
                "relevant_per_user_median": float(np.median(counts)),
                "candidates_mean": float(self.seen.shape[1] - np.diff(self.seen.indptr)[self.users].mean())}


@dataclass
class Result:
    metrics: dict
    per_user: dict
    recommendations: np.ndarray


def score_lists(recommendations: np.ndarray, data: EvaluationSet) -> Result:
    """NDCG, Recall, HitRate per user (binary gains); Coverage over the catalog."""
    users, k = data.users, data.k
    if recommendations.shape != (len(users), k):
        raise ValueError("expected one top-k list per evaluation user")
    if (np.diff(np.sort(recommendations, axis=1), axis=1) == 0).any():
        raise ValueError("duplicate item in a list")
    flat_users = np.repeat(users, k)
    if np.asarray(data.seen[flat_users, recommendations.ravel()]).any():
        raise ValueError("a list contains an already reviewed item")
    hits = np.asarray(data.relevant[flat_users, recommendations.ravel()]).reshape(-1, k) > 0
    n_relevant = np.diff(data.relevant.indptr)[users]
    discount = 1 / np.log2(np.arange(2, k + 2))
    per_user = {
        f"ndcg@{k}": (hits * discount).sum(1) / np.cumsum(discount)[np.minimum(n_relevant, k) - 1],
        f"recall@{k}": hits.sum(1) / n_relevant,
        f"hitrate@{k}": hits.any(1).astype(float),
    }
    metrics = {name: float(values.mean()) for name, values in per_user.items()}
    metrics[f"coverage@{k}"] = len(np.unique(recommendations)) / data.seen.shape[1]
    return Result(metrics, per_user, recommendations)


def evaluate(model, data: EvaluationSet, batch_size: int = 1024) -> Result:
    lists = [model.recommend(data.users[start : start + batch_size], data.seen, data.k,
                             tie_break=data.tie_break)
             for start in range(0, len(data.users), batch_size)]
    return score_lists(np.concatenate(lists), data)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 42) -> dict:
    """95% percentile interval of mean(a - b), resampling users with replacement."""
    difference = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    draws = np.array([difference[rng.integers(len(difference), size=len(difference))].mean()
                      for _ in range(n)])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"difference": float(difference.mean()), "ci_low": float(low), "ci_high": float(high)}
