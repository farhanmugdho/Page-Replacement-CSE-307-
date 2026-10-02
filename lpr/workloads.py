"""Synthetic page-access traces with a deliberate mid-trace workload shift."""
import numpy as np


def phase1_locality(n, rng, universe=200, ws=40, phase_len=800):
    """Locality-heavy + sequential: a drifting working set of `ws` pages that is
    alternately scanned sequentially (loops) and hit with Zipf-skewed accesses."""
    w = 1.0 / np.arange(1, ws + 1)
    w /= w.sum()
    out = []
    while len(out) < n:
        base = int(rng.integers(0, universe - ws))
        pages = base + np.arange(ws)
        hot = rng.permutation(pages)
        end = len(out) + phase_len
        while len(out) < end:
            if rng.random() < 0.5:                       # sequential scan run
                length = int(rng.integers(ws, 2 * ws + 1))
                start = int(rng.integers(0, ws))
                out.extend(int(pages[(start + k) % ws]) for k in range(length))
            else:                                        # skewed hot-set run
                length = int(rng.integers(40, 121))
                out.extend(hot[rng.choice(ws, size=length, p=w)].tolist())
    return out[:n]


def phase2_random(n, rng, universe=400):
    return rng.integers(0, universe, size=n).tolist()


def phase2_bursty(n, rng, universe=400, burst_prob=0.02, group=8):
    """Uniform random background, interrupted by short bursts on a small page group."""
    out = []
    while len(out) < n:
        if rng.random() < burst_prob:
            g = int(rng.integers(0, universe - group)) + np.arange(group)
            out.extend(rng.choice(g, size=int(rng.integers(30, 61))).tolist())
        else:
            out.append(int(rng.integers(0, universe)))
    return out[:n]


SCENARIOS = {"random": phase2_random, "bursty": phase2_bursty}


def make_trace(scenario, n, seed):
    """First half = locality/sequential, second half = `scenario`. Returns (trace, shift_index)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    return phase1_locality(half, rng) + SCENARIOS[scenario](n - half, rng), half


def make_training_trace(n, seed):
    """Pre-shift-style trace used to train the offline (static) models."""
    return phase1_locality(n, np.random.default_rng(seed))
