"""In-memory token-bucket rate limiter, one bucket per client.

Good enough for a single-process demo. In a real deployment, rate limiting
belongs in the reverse proxy / API gateway (shared across replicas, aware of
the real client address); see SECURITY.md.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """``rate_per_minute`` sustained requests per key, with bursts up to ``burst``.

    The number of tracked keys is bounded (least-recently-used eviction), so a
    flood of distinct client addresses cannot exhaust memory.
    """

    def __init__(
        self,
        rate_per_minute: int,
        burst: int,
        *,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate_per_minute <= 0 or burst <= 0:
            raise ValueError("rate_per_minute and burst must be positive")
        self.rate = rate_per_minute / 60.0
        self.burst = float(burst)
        self.max_keys = max_keys
        self.clock = clock
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()

    def acquire(self, key: str) -> float:
        """Take one token for ``key``. Returns 0 when allowed, else seconds until a token."""
        now = self.clock()
        with self._lock:
            bucket = self._buckets.pop(key, None) or _Bucket(self.burst, now)
            bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
            self._buckets[key] = bucket
            while len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return 0.0
            return (1 - bucket.tokens) / self.rate
