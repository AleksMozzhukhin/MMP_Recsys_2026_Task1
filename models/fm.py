"""A CPU factorization machine over user, item and point-in-time item features.

Every input field is one-hot. For the active feature indices A the logit is
    w0 + sum(w[j], j in A)
       + 1/2 sum_f[(sum(v[j,f], j in A))**2 - sum(v[j,f]**2, j in A)].
Training minimizes binary cross entropy with Adam. Each epoch uses every
observed training pair once, labelled according to ``positive``, and freshly
samples ``negative_ratio`` unobserved pairs per positive training interaction.
Sampling excludes ALL observed training pairs, including low-rated ones.

Item features are evaluated at the moment of each training event; a sampled
negative inherits the time of the positive it was drawn for. Recommendations use
the same features evaluated at the cutoff. Constant per-item features would be
absorbed by the item embedding; time-varying ones are not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from .base import Recommender


def _binary_csr(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Canonicalize the support, without changing the caller's sparse matrix."""
    result = sp.csr_matrix(matrix, copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if not np.isfinite(result.data).all() or (result.data < 0).any():
        raise ValueError("interaction matrices must contain finite nonnegative entries")
    result.data = np.ones(result.nnz, dtype=np.float32)
    return result


def _contains(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    if not len(sorted_keys):
        return np.zeros(query.shape, dtype=bool)
    positions = np.searchsorted(sorted_keys, query)
    return (positions < len(sorted_keys)) & (
        sorted_keys[np.minimum(positions, len(sorted_keys) - 1)] == query
    )


def _sample_items(
    user_ids: np.ndarray,
    observed: sp.csr_matrix,
    observed_keys: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform rejection sampling; handle dense users without unbounded loops."""
    n_items = observed.shape[1]
    counts = np.diff(observed.indptr)
    users = user_ids[counts[user_ids] < n_items].copy()
    items = rng.integers(n_items, size=len(users), dtype=np.int64)
    remaining = np.flatnonzero(_contains(observed_keys, users * n_items + items))
    # The ordinary sparse case takes one or two vectorized rounds. For dense
    # rows, a bounded fallback samples directly from their complement.
    for _ in range(20):
        if not len(remaining):
            break
        items[remaining] = rng.integers(n_items, size=len(remaining))
        remaining = remaining[
            _contains(observed_keys, users[remaining] * n_items + items[remaining])
        ]
    if len(remaining):
        for user in np.unique(users[remaining]):
            locations = remaining[users[remaining] == user]
            known = observed.indices[observed.indptr[user] : observed.indptr[user + 1]]
            available = np.setdiff1d(np.arange(n_items), known, assume_unique=True)
            items[locations] = rng.choice(available, size=len(locations))
    return users, items


def sample_unobserved_pairs(
    observed: sp.spmatrix,
    positive: sp.spmatrix,
    *,
    negative_ratio: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducible training-only negatives, sampled with replacement.

    User frequency follows the count of positive training interactions. Users
    who have observed the complete catalog are skipped because no negative
    candidate exists. Duplicate sampled pairs are allowed.
    """
    if not isinstance(negative_ratio, (int, np.integer)) or negative_ratio < 0:
        raise ValueError("negative_ratio must be a nonnegative integer")
    observed, positive = _binary_csr(observed), _binary_csr(positive)
    if observed.shape != positive.shape or min(observed.shape) == 0:
        raise ValueError("observed and positive need the same nonempty shape")
    observed_users, observed_items = observed.nonzero()
    keys = observed_users.astype(np.int64) * observed.shape[1] + observed_items
    positive_users, positive_items = positive.nonzero()
    positive_keys = positive_users.astype(np.int64) * observed.shape[1] + positive_items
    if not _contains(keys, positive_keys).all():
        raise ValueError("positive pairs must be a subset of observed training pairs")
    users = np.repeat(positive_users.astype(np.int64), negative_ratio)
    return _sample_items(users, observed, keys, np.random.default_rng(seed))


class _FMNetwork(nn.Module):
    """Generic FM forward pass, independent of the optimized catalog scoring."""

    def __init__(self, n_features: int, factors: int, n_identity_features: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        # Construct without random default initialization. Initialize identity
        # and metadata factors separately so the identity-only ablation starts
        # with exactly the same user/item factors and the same negative stream.
        self.linear = nn.Embedding.from_pretrained(torch.zeros(n_features, 1), freeze=False)
        self.embedding = nn.Embedding.from_pretrained(
            torch.zeros(n_features, factors), freeze=False,
        )
        nn.init.normal_(self.embedding.weight[:n_identity_features], mean=0.0, std=0.01)
        if n_identity_features < n_features:
            nn.init.normal_(self.embedding.weight[n_identity_features:], mean=0.0, std=0.01)

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.embedding(feature_ids)
        pairwise = 0.5 * (
            vectors.sum(dim=1).square() - vectors.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + self.linear(feature_ids).sum(dim=(1, 2)) + pairwise


class FactorizationMachine(Recommender):
    """Factorization machine trained from explicit labels and sampled negatives.

    ``fit(observed, positive, features, times, cutoff)`` accepts CSR matrices of
    shape ``(n_users, n_items)``; their support defines observed and relevant
    pairs. ``features`` has ``encode(items, times) -> int array (n, fields)`` and
    ``n_ids``; it must use only information strictly before the given times.
    ``times`` holds event timestamps on the observed support. Without
    ``features`` the model is the user/item-only ablation with the same
    initial user/item factors and the same negative stream.
    """

    def __init__(
        self,
        factors: int = 32,
        epochs: int = 10,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5,
        negative_ratio: int = 1,
        batch_size: int = 4096,
        seed: int = 42,
        num_threads: int = 4,
    ) -> None:
        if factors < 1 or epochs < 1 or batch_size < 1 or num_threads < 1:
            raise ValueError("factors, epochs, batch_size, num_threads must be positive")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if not isinstance(negative_ratio, (int, np.integer)) or negative_ratio < 0:
            raise ValueError("negative_ratio must be a nonnegative integer")
        self.factors = int(factors)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.negative_ratio = int(negative_ratio)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.num_threads = int(num_threads)

    def fit(
        self,
        observed: sp.spmatrix,
        positive: sp.spmatrix,
        features: Any = None,
        times: sp.spmatrix | None = None,
        cutoff: int | None = None,
        on_epoch: Any = None,
    ) -> "FactorizationMachine":
        """``on_epoch(epoch, model)`` is called after every epoch with scoring ready."""
        observed, positive = _binary_csr(observed), _binary_csr(positive)
        if observed.shape != positive.shape or min(observed.shape) == 0:
            raise ValueError("observed and positive need the same nonempty shape")
        if observed.nnz == 0:
            raise ValueError("at least one observed training pair is required")
        self.n_users_, self.n_items_ = observed.shape
        observed_users, observed_items = observed.nonzero()
        observed_keys = observed_users.astype(np.int64) * self.n_items_ + observed_items
        positive_users, positive_items = positive.nonzero()
        positive_keys = positive_users.astype(np.int64) * self.n_items_ + positive_items
        if not _contains(observed_keys, positive_keys).all():
            raise ValueError("positive pairs must be a subset of observed training pairs")
        labels = _contains(positive_keys, observed_keys).astype(np.float32)
        self.features_ = features
        if features is not None:
            if times is None or cutoff is None:
                raise ValueError("time-dependent features need event times and a cutoff")
            times = sp.csr_matrix(times, dtype=np.float64, copy=True)
            times.sort_indices()
            if (times.shape != observed.shape or not np.array_equal(times.indptr, observed.indptr)
                    or not np.array_equal(times.indices, observed.indices)):
                raise ValueError("times must have exactly the observed support")
            observed_times = times.data.astype(np.int64)
            if observed_times.max() >= cutoff:
                raise ValueError("training events must precede the cutoff")
            positive_times = observed_times[np.searchsorted(observed_keys, positive_keys)]
            self.cutoff_ = int(cutoff)
            self.item_features_ = self._encode(np.arange(self.n_items_), np.full(self.n_items_, cutoff))
            self.n_metadata_features_ = int(features.n_ids)
        else:
            observed_times = positive_times = np.zeros(0, dtype=np.int64)
            self.item_features_ = np.empty((self.n_items_, 0), dtype=np.int64)
            self.n_metadata_features_ = 0
        self.n_features_ = self.n_users_ + self.n_items_ + self.n_metadata_features_
        negative_users = np.repeat(positive_users.astype(np.int64), self.negative_ratio)
        negative_times = np.repeat(positive_times, self.negative_ratio)
        has_negative = np.diff(observed.indptr)[negative_users] < self.n_items_
        negative_users = negative_users[has_negative]
        if features is not None:
            negative_times = negative_times[has_negative]
        rng = np.random.default_rng(self.seed)
        self.loss_history_: list[float] = []
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self.num_threads)
            # Initialization is seeded while preserving the caller's Torch RNG.
            # All later randomness comes from the local NumPy Generator.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                self.network_ = _FMNetwork(
                    self.n_features_, self.factors, self.n_users_ + self.n_items_,
                )
            optimizer = torch.optim.Adam(
                self.network_.parameters(), lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            for epoch in range(self.epochs):
                neg_u, neg_i = _sample_items(negative_users, observed, observed_keys, rng)
                users = np.concatenate((observed_users, neg_u)).astype(np.int64)
                items = np.concatenate((observed_items, neg_i)).astype(np.int64)
                columns = [users, items + self.n_users_]
                if features is not None:
                    metadata = self._encode(items, np.concatenate((observed_times, negative_times)))
                    columns += list((metadata + self.n_users_ + self.n_items_).T)
                feature_ids = torch.from_numpy(np.column_stack(columns))
                target = torch.from_numpy(
                    np.concatenate((labels, np.zeros(len(neg_u), dtype=np.float32))))
                order = torch.from_numpy(rng.permutation(len(users)))
                total_loss = 0.0
                self.network_.train()
                for start in range(0, len(order), self.batch_size):
                    indices = order[start : start + self.batch_size]
                    logits = self.network_(feature_ids[indices])
                    loss = F.binary_cross_entropy_with_logits(logits, target[indices])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite FM training loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(indices)
                self.loss_history_.append(total_loss / len(users))
                self.network_.eval()
                self._cache_scoring_parameters()
                if on_epoch is not None:
                    on_epoch(epoch + 1, self)
        finally:
            torch.set_num_threads(previous_threads)
        return self

    def _encode(self, items: np.ndarray, times: np.ndarray) -> np.ndarray:
        encoded = np.asarray(self.features_.encode(items, times))
        if encoded.ndim != 2 or len(encoded) != len(items) or not np.issubdtype(encoded.dtype, np.integer):
            raise ValueError("features.encode must return an integer array (n, fields)")
        if encoded.size and (encoded.min() < 0 or encoded.max() >= self.features_.n_ids):
            raise ValueError("feature ID outside [0, n_ids)")
        return encoded.astype(np.int64, copy=False)

    def _cache_scoring_parameters(self) -> None:
        """Reduce item fields once; catalog scoring is a single matrix product."""
        embeddings = self.network_.embedding.weight.detach().numpy()
        linear = self.network_.linear.weight.detach().numpy().ravel()
        item_fields = np.concatenate(
            (np.arange(self.n_items_, dtype=np.int64)[:, None] + self.n_users_,
             self.item_features_ + self.n_users_ + self.n_items_), axis=1,
        )
        vectors = embeddings[item_fields]
        self.item_vectors_ = vectors.sum(axis=1)
        self.item_bias_ = (
            linear[item_fields].sum(axis=1)
            + 0.5 * (self.item_vectors_**2 - (vectors**2).sum(axis=1)).sum(axis=1)
            + float(self.network_.bias.detach())
        )
        self.user_vectors_ = embeddings[: self.n_users_].copy()
        self.user_bias_ = linear[: self.n_users_].copy()

    def _check_users(self, user_ids: Any) -> np.ndarray:
        if not hasattr(self, "user_vectors_"):
            raise RuntimeError("fit the FM before scoring")
        return self._user_ids(user_ids)

    def score(self, user_ids: Any) -> np.ndarray:
        """Return B x I logits without materializing a B x I x factors tensor."""
        users = self._check_users(user_ids)
        return (self.user_vectors_[users] @ self.item_vectors_.T
                + self.user_bias_[users, None] + self.item_bias_[None, :])

    def predict_pairs(self, user_ids: Any, item_ids: Any, times: Any = None) -> np.ndarray:
        """Evaluate the generic FM formula; features at ``times`` (default: cutoff)."""
        users = self._check_users(user_ids)
        items = np.asarray(item_ids)
        if items.shape != users.shape or not np.issubdtype(items.dtype, np.integer):
            raise ValueError("item_ids must be an integer array matching user_ids")
        if (items < 0).any() or (items >= self.n_items_).any():
            raise ValueError("item_ids must lie inside the fitted catalog")
        items = items.astype(np.int64, copy=False)
        if self.features_ is None:
            metadata = np.empty((len(items), 0), dtype=np.int64)
        elif times is None:
            metadata = self.item_features_[items]
        else:
            metadata = self._encode(items, np.asarray(times))
        feature_ids = np.concatenate(
            (users[:, None], items[:, None] + self.n_users_,
             metadata + self.n_users_ + self.n_items_), axis=1,
        )
        with torch.no_grad():
            return self.network_(torch.from_numpy(feature_ids)).numpy()

# Short name for notebook tables and convenient imports.
FM = FactorizationMachine
