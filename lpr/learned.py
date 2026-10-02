"""Learned eviction policy.

Idea: at every eviction we look at each resident page's features
(recency, age, frequency, decayed frequency, last reuse gap) and a classifier
predicts "this page is among the farthest-reused candidates, so it is a good
eviction victim".  We evict the page with the highest predicted probability.
Training labels come from hindsight (Belady-style next-use distance) on a
training trace.

Two variants:
  * static   - trained once, offline, on a pre-shift style workload.
  * adaptive - additionally retrained every `retrain_every` references on the
               most recent `window` references (past data only, no peeking).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import PageStats
from .policies import Policy, next_use_array


def make_model(kind, seed=0):
    if kind == "dt":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=seed)
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    raise ValueError(kind)


class Collector(Policy):
    """Behaviour policy (random eviction) that logs labelled eviction events."""
    name = "collector"

    def __init__(self, horizon, seed=0):
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def reset(self, trace, frames):
        super().reset(trace, frames)
        self.nxt = next_use_array(trace)
        self.n = len(trace)
        self.stats = PageStats()
        self.next_of = {}
        self.X, self.y = [], []

    def on_hit(self, p, t):
        self.stats.hit(p, t)
        self.next_of[p] = self.nxt[t]

    def on_load(self, p, t):
        self.stats.load(p, t)
        self.next_of[p] = self.nxt[t]

    def on_evict(self, p):
        self.stats.evict(p)
        del self.next_of[p]

    def victim(self, resident, t):
        pages = list(resident)
        if t + self.horizon < self.n:  # label is only known if horizon fits in trace
            self.X.append(self.stats.matrix(pages, t))
            nu = np.array([self.next_of[q] for q in pages], dtype=float)
            # rank-relative label: 1 = page is among the farthest-reused 25% of candidates
            # (ties at "never reused" all count), so argmax(proba) targets Belady's choice
            self.y.append(nu >= np.quantile(nu, 0.75))
        # Purely random eviction gives much better state coverage than following LRU
        # (LRU-driven data is biased: it keeps re-creating the states LRU is bad at).
        return pages[self.rng.integers(len(pages))]


def train_model(kind, trace, frames, seed=0):
    """Fit a model on hindsight labels from `trace`. Returns None if data is degenerate."""
    from .policies import simulate
    col = Collector(horizon=frames, seed=seed)
    simulate(trace, frames, col)
    if not col.X:
        return None
    X, y = np.vstack(col.X), np.concatenate(col.y)
    if len(np.unique(y)) < 2:
        return None
    return make_model(kind, seed).fit(X, y)


class LearnedPolicy(Policy):
    def __init__(self, kind, model, retrain_every=None, window=3000, seed=0):
        self.kind, self.model = kind, model
        self.retrain_every, self.window, self.seed = retrain_every, window, seed
        tag = "adaptive" if retrain_every else "static"
        self.name = f"Learned-{kind.upper()} ({tag})"

    def reset(self, trace, frames):
        super().reset(trace, frames)
        self.stats = PageStats()
        self.retrains = 0

    def _maybe_retrain(self, t):
        if self.retrain_every and t > 0 and t % self.retrain_every == 0:
            past = self.trace[max(0, t - self.window):t]  # only references already seen
            m = train_model(self.kind, past, self.frames, self.seed + t)
            if m is not None:
                self.model, self.retrains = m, self.retrains + 1

    def on_hit(self, p, t):
        self.stats.hit(p, t)
        self._maybe_retrain(t)

    def on_load(self, p, t):
        self.stats.load(p, t)
        self._maybe_retrain(t)

    def on_evict(self, p):
        self.stats.evict(p)

    def victim(self, resident, t):
        pages = list(resident)
        rec = np.array([t - self.stats.last_access(q) for q in pages], dtype=float)
        if self.model is None:
            return pages[int(rec.argmax())]  # fall back to LRU
        proba = self.model.predict_proba(self.stats.matrix(pages, t))[:, 1]
        # tiny recency term breaks ties in favour of LRU
        return pages[int(np.argmax(proba + 1e-9 * rec))]
