import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lpr.policies import FIFO, LRU, Optimal, simulate

# Classic textbook reference string (Silberschatz et al.), 3 frames
REF = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]


def faults(policy, frames=3):
    return int((~simulate(REF, frames, policy)).sum())


def test_fifo():
    assert faults(FIFO()) == 15


def test_lru():
    assert faults(LRU()) == 12


def test_optimal():
    assert faults(Optimal()) == 9


def test_opt_is_lower_bound():
    import numpy as np
    rng = np.random.default_rng(0)
    tr = rng.integers(0, 30, 2000).tolist()
    o = (~simulate(tr, 8, Optimal())).sum()
    assert o <= (~simulate(tr, 8, LRU())).sum()
    assert o <= (~simulate(tr, 8, FIFO())).sum()
