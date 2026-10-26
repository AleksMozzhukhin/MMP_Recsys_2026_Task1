"""Download, cleaning and temporal phases of Amazon Reviews 2014 — Video Games."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from http.client import HTTPException
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SOURCE_URLS = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/ratings_Video_Games.csv",
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Video_Games.csv",
)
EXPECTED_SHA256 = "f0ca171050c2e0f208d47f5267992476b21ab68a01439c21c8be1a4432982cfa"
EXPECTED_BYTES = 53_949_503
POSITIVE_RATING = 4.0
DAY = 86_400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(path: Path) -> Path:
    """Return a local copy of the CSV whose size and SHA-256 match the experiment."""
    path = Path(path)
    if path.exists():
        if path.stat().st_size != EXPECTED_BYTES or sha256(path) != EXPECTED_SHA256:
            raise ValueError(f"{path} fails the integrity check; delete it and rerun")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    errors = []
    for url in SOURCE_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "recsys-coursework"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            if partial.stat().st_size != EXPECTED_BYTES or sha256(partial) != EXPECTED_SHA256:
                raise ValueError("downloaded bytes differ from the experiment source")
            partial.replace(path)
            return path
        except (OSError, ValueError, HTTPException) as exc:
            errors.append(f"{url}: {exc}")
            partial.unlink(missing_ok=True)
    raise RuntimeError("Download failed:\n" + "\n".join(errors))


def utc_seconds(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp())


def read_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    """Drop invalid rows; keep the earliest review of each user/item pair.

    Exact-time duplicates keep the first row of the file. All ratings 1–5 stay:
    a low rating is still an observed interaction that must not be recommended.
    """
    raw = pd.read_csv(path, header=None, names=["user_id", "item_id", "rating", "timestamp"],
                      dtype={"user_id": "string", "item_id": "string"})
    rating = pd.to_numeric(raw.rating, errors="coerce")
    timestamp = pd.to_numeric(raw.timestamp, errors="coerce")
    valid = (raw.user_id.str.strip().fillna("").ne("") & raw.item_id.str.strip().fillna("").ne("")
             & rating.between(1, 5) & timestamp.ge(0) & timestamp.eq(np.floor(timestamp)))
    clean = pd.DataFrame({"user_id": raw.user_id.str.strip(), "item_id": raw.item_id.str.strip(),
                          "rating": rating.astype(float), "timestamp": timestamp})[valid]
    clean["timestamp"] = clean.timestamp.astype(np.int64)
    clean = clean.sort_values("timestamp", kind="stable")  # stable: file order breaks ties
    duplicate = clean.duplicated(["user_id", "item_id"], keep="first")
    clean = clean[~duplicate].reset_index(drop=True)
    report = {"rows": len(raw), "users": raw.user_id.nunique(), "items": raw.item_id.nunique(),
              "missing_values": int(raw.isna().sum().sum()), "invalid_rows": int((~valid).sum()),
              "duplicate_rows": int(duplicate.sum()), "clean_rows": len(clean)}
    return clean, report


class ItemHistory:
    """Point-in-time item descriptors computed only from reviews strictly before t.

    Fields (one-hot buckets, disjoint ID ranges): ``total`` reviews before t,
    ``recent`` reviews in the last 90 days before t, and ``age`` — days since the
    item's first review (own bucket when there is none yet). Ratings are not used.
    The same function describes training events at their own time and
    candidate items at the recommendation cutoff.
    """

    EDGES = {"total": np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, np.inf]),
             "recent": np.array([0, 1, 2, 4, 8, 16, 32, np.inf]),
             "age": np.array([0, 30, 90, 180, 365, 730, 1460, np.inf])}
    RECENT_DAYS = 90

    def __init__(self, items: np.ndarray, timestamps: np.ndarray, n_items: int, fields=("age",)):
        self.scale = np.int64(1) << 32
        self.keys = np.sort(items.astype(np.int64) * self.scale + timestamps.astype(np.int64))
        self.start = np.searchsorted(self.keys, np.arange(n_items + 1, dtype=np.int64) * self.scale)
        self.fields = tuple(fields)
        # Age has an extra bucket for items without earlier reviews.
        self.sizes = [len(self.EDGES[f]) - (f != "age") for f in self.fields]
        self.offsets = np.concatenate(([0], np.cumsum(self.sizes)[:-1]))
        self.n_ids = int(sum(self.sizes))

    def encode(self, items: np.ndarray, times: np.ndarray) -> np.ndarray:
        items = np.asarray(items, dtype=np.int64)
        times = np.asarray(times, dtype=np.int64)
        base = items * self.scale
        before = np.searchsorted(self.keys, base + times) - self.start[items]
        values = {
            "total": before,
            "recent": before - np.searchsorted(self.keys, base + times - self.RECENT_DAYS * DAY)
                      + self.start[items],
            "age": (times - (self.keys[np.minimum(self.start[items], len(self.keys) - 1)] - base)) / DAY,
        }
        columns = []
        for field, offset in zip(self.fields, self.offsets):
            bucket = np.searchsorted(self.EDGES[field], values[field], side="right") - 1
            if field == "age":
                bucket = np.where(before > 0, bucket, len(self.EDGES["age"]) - 1)
            columns.append(bucket + offset)
        return np.column_stack(columns).astype(np.int64)


@dataclass
class Phase:
    """One temporal slice: axes and history strictly before ``cutoff``, holdout after it.

    The catalog is the top ``catalog_size`` items by review count before the cutoff
    (ties: ascending ASIN); users need ``min_reviews`` reviews of that catalog before
    the cutoff. Holdout rows outside these axes are dropped (cold users/items).
    """

    name: str
    cutoff: int
    history: pd.DataFrame
    holdout: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray
    dropped_holdout_rows: int
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_users, self.n_items

    def _csr(self, frame: pd.DataFrame, values) -> sparse.csr_matrix:
        matrix = sparse.csr_matrix((values, (frame.user, frame.item)), shape=self.shape)
        matrix.sort_indices()
        return matrix

    def matrix(self, signal: str) -> sparse.csr_matrix:
        """``observed``: every review; ``positive``: rating >= 4; ``rating``/``time``: values."""
        if signal not in self._cache:
            frame = self.history
            if signal == "observed":
                self._cache[signal] = self._csr(frame, np.ones(len(frame)))
            elif signal == "positive":
                liked = frame[frame.rating >= POSITIVE_RATING]
                self._cache[signal] = self._csr(liked, np.ones(len(liked)))
            elif signal == "rating":
                self._cache[signal] = self._csr(frame, frame.rating.to_numpy(np.float64))
            elif signal == "time":
                self._cache[signal] = self._csr(frame, frame.timestamp.to_numpy(np.float64))
            else:
                raise ValueError(signal)
        return self._cache[signal]

    def item_history(self, fields=("age",)) -> ItemHistory:
        key = ("item_history", tuple(fields))
        if key not in self._cache:
            self._cache[key] = ItemHistory(self.history.item.to_numpy(),
                                           self.history.timestamp.to_numpy(), self.n_items, fields)
        return self._cache[key]

    def popularity(self, window_days: int | None = None) -> np.ndarray:
        """Positive reviews per item, optionally only in the last ``window_days``."""
        liked = self.history[self.history.rating >= POSITIVE_RATING]
        if window_days is not None:
            liked = liked[liked.timestamp >= self.cutoff - window_days * DAY]
        return np.bincount(liked.item, minlength=self.n_items).astype(np.float64)

    def summary(self) -> dict:
        h, t = self.history, self.holdout
        return {"history_rows": len(h), "users": self.n_users, "items": self.n_items,
                "holdout_rows": len(t), "holdout_dropped_rows": self.dropped_holdout_rows,
                "history_positive_share": float((h.rating >= POSITIVE_RATING).mean()),
                "density": len(h) / (self.n_users * self.n_items)}


def make_phase(clean: pd.DataFrame, name: str, cutoff: str, end: str | None = None,
               catalog_size: int = 4000, min_reviews: int = 3) -> Phase:
    start = utc_seconds(cutoff)
    stop = np.iinfo(np.int64).max if end is None else utc_seconds(end)
    past = clean[clean.timestamp < start]
    counts = past.groupby("item_id").size().reset_index(name="n")
    catalog = counts.sort_values(["n", "item_id"], ascending=[False, True]).item_id.head(catalog_size)
    past = past[past.item_id.isin(catalog)]
    activity = past.groupby("user_id").size()
    user_ids = np.sort(activity.index[activity >= min_reviews].to_numpy(str))
    past = past[past.user_id.isin(user_ids)]
    item_ids = np.sort(past.item_id.unique().astype(str))
    user_map = pd.Series(np.arange(len(user_ids)), index=user_ids)
    item_map = pd.Series(np.arange(len(item_ids)), index=item_ids)

    def encode(frame):
        frame = frame[frame.user_id.isin(user_map.index) & frame.item_id.isin(item_map.index)]
        return frame.assign(user=frame.user_id.map(user_map).to_numpy(np.int64),
                            item=frame.item_id.map(item_map).to_numpy(np.int64)).reset_index(drop=True)

    future = clean[(clean.timestamp >= start) & (clean.timestamp < stop)]
    holdout = encode(future)
    return Phase(name, start, encode(past), holdout, user_ids, item_ids, len(future) - len(holdout))
