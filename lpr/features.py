"""Per-page features used by the learned eviction model."""
import numpy as np

FEATURES = ["recency", "age", "frequency", "decayed_frequency", "last_gap"]
NEW_PAGE_GAP = 5000  # sentinel: a freshly loaded page has no reuse gap yet


class PageStats:
    """Tracks recency / frequency information for resident pages."""

    def __init__(self, decay=0.99):
        self.decay = decay
        self.d = {}  # page -> [last_access, load_time, count, ewma, last_gap]

    def load(self, p, t):
        self.d[p] = [t, t, 1, 1.0, NEW_PAGE_GAP]

    def hit(self, p, t):
        s = self.d[p]
        gap = t - s[0]
        s[3] = s[3] * self.decay ** gap + 1.0
        s[0], s[2], s[4] = t, s[2] + 1, gap

    def evict(self, p):
        del self.d[p]

    def last_access(self, p):
        return self.d[p][0]

    def matrix(self, pages, t):
        rows = []
        for p in pages:
            s = self.d[p]
            rows.append((t - s[0], t - s[1], s[2], s[3] * self.decay ** (t - s[0]), s[4]))
        return np.log1p(np.asarray(rows, dtype=float))
