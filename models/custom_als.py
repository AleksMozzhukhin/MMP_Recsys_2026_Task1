"""ALS with a signed, recency-adjusted rating target and unchanged confidence."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .ials import IALS


class CustomALS(IALS):
    """Use IALS's identical objective and solver with independently built targets.

    q_ui = (rating_ui - 3)/2 * [(1-eta) + eta*2^(-age_ui/h)],
    where age_ui is days before the training cutoff and h is the half-life.
    The signed preference distinguishes strong/weak likes and dislikes. Age
    adjusts the *desired score*, never the confidence: c=1+alpha on every
    review in both models. The persistent component 1-eta retains old evidence.
    Ratings and dates must come exclusively from the current training split.
    """

    @staticmethod
    def target_values(
        ratings, timestamps, cutoff: float, *, half_life_days: float = 365.0,
        recency_mix: float = 0.5,
    ) -> np.ndarray:
        """Build targets from aligned rating and Unix-seconds timestamp arrays."""
        ratings = np.asarray(ratings, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if ratings.shape != timestamps.shape:
            raise ValueError("ratings and timestamps must have identical shapes")
        if not np.all(np.isfinite(ratings)) or np.any((ratings < 1) | (ratings > 5)):
            raise ValueError("ratings must be finite values in [1, 5]")
        if not np.isfinite(cutoff) or not np.all(np.isfinite(timestamps)):
            raise ValueError("cutoff and timestamps must be finite Unix seconds")
        if np.any(timestamps > cutoff):
            raise ValueError("Training timestamps cannot be later than cutoff")
        if not np.isfinite(half_life_days) or half_life_days <= 0:
            raise ValueError("half_life_days must be finite and positive")
        if not np.isfinite(recency_mix) or not 0 <= recency_mix <= 1:
            raise ValueError("recency_mix must be in [0, 1]")
        age_days = (cutoff - timestamps) / 86400.0
        persistence = (1.0 - recency_mix) + recency_mix * np.exp2(-age_days / half_life_days)
        return (ratings - 3.0) / 2.0 * persistence

    @classmethod
    def build_targets(
        cls, ratings, timestamps, cutoff: float, *, half_life_days: float = 365.0,
        recency_mix: float = 0.5,
    ) -> sparse.csr_matrix:
        """CSR convenience wrapper; ratings/timestamps must share exact support."""
        ratings = sparse.csr_matrix(ratings, dtype=np.float64, copy=True)
        timestamps = sparse.csr_matrix(timestamps, dtype=np.float64, copy=True)
        for matrix in (ratings, timestamps):
            matrix.sum_duplicates()
            matrix.sort_indices()
        if (
            ratings.shape != timestamps.shape
            or not np.array_equal(ratings.indptr, timestamps.indptr)
            or not np.array_equal(ratings.indices, timestamps.indices)
        ):
            raise ValueError("ratings and timestamps must share exact CSR support")
        ratings.data = cls.target_values(
            ratings.data, timestamps.data, cutoff,
            half_life_days=half_life_days, recency_mix=recency_mix,
        )
        ratings.eliminate_zeros()
        return ratings
