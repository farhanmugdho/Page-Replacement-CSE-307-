"""Classical page-replacement policies (FIFO, LRU, Belady's Optimal) + simulator."""
from collections import OrderedDict
import numpy as np

INF = 10**12  # "never used again"


def next_use_array(trace):
    """nxt[i] = index of the next access to trace[i] after i (INF if none)."""
    n = len(trace)
    nxt = np.empty(n, dtype=np.int64)
    last = {}
    for i in range(n - 1, -1, -1):
        p = trace[i]
        nxt[i] = last.get(p, INF)
        last[p] = i
    return nxt


class Policy:
    name = "base"

    def reset(self, trace, frames):
        self.trace, self.frames = trace, frames

    def on_hit(self, page, t): ...
    def on_load(self, page, t): ...
    def on_evict(self, page): ...

    def victim(self, resident, t):
        raise NotImplementedError


class FIFO(Policy):
    name = "FIFO"

    def reset(self, trace, frames):
        super().reset(trace, frames)
        self.q = OrderedDict()

    def on_load(self, page, t):
        self.q[page] = t

    def on_evict(self, page):
        del self.q[page]

    def victim(self, resident, t):
        return next(iter(self.q))  # oldest loaded page


class LRU(Policy):
    name = "LRU"

    def reset(self, trace, frames):
        super().reset(trace, frames)
        self.q = OrderedDict()

    def on_hit(self, page, t):
        self.q.move_to_end(page)

    def on_load(self, page, t):
        self.q[page] = t

    def on_evict(self, page):
        del self.q[page]

    def victim(self, resident, t):
        return next(iter(self.q))  # least recently used


class Optimal(Policy):
    """Belady's MIN: evict the resident page whose next use is farthest away."""
    name = "OPT"

    def reset(self, trace, frames):
        super().reset(trace, frames)
        self.nxt = next_use_array(trace)
        self.next_of = {}

    def on_hit(self, page, t):
        self.next_of[page] = self.nxt[t]

    def on_load(self, page, t):
        self.next_of[page] = self.nxt[t]

    def on_evict(self, page):
        del self.next_of[page]

    def victim(self, resident, t):
        return max(resident, key=self.next_of.__getitem__)


def simulate(trace, frames, policy):
    """Run `policy` over `trace` with `frames` frames. Returns a bool array: hit?"""
    trace = list(trace)
    policy.reset(trace, frames)
    resident = set()
    hits = np.zeros(len(trace), dtype=bool)
    for t, p in enumerate(trace):
        if p in resident:
            hits[t] = True
            policy.on_hit(p, t)
        else:
            if len(resident) >= frames:
                v = policy.victim(resident, t)
                resident.remove(v)
                policy.on_evict(v)
            resident.add(p)
            policy.on_load(p, t)
    return hits


def phase_stats(hits, shift):
    """Hit ratio / fault count before and after the workload shift."""
    pre, post = hits[:shift], hits[shift:]
    return {
        "hit_pre": pre.mean(), "hit_post": post.mean(),
        "faults_pre": int((~pre).sum()), "faults_post": int((~post).sum()),
    }
