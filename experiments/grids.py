"""Shared full tuning grids for serial baselines (see E26/E26b)."""
import itertools
SERIAL_FWD = [(BN, D, nw) for BN in (16, 32, 64, 128) for D in (2, 4, 8, 16, 32) for nw in (1, 2, 4)]


def best_serial(timeit, run_plain, run_tiled, T):
    """min over the plain kernel and the time-tiled kernel on the full grid; returns (ms, cfg)."""
    best = (timeit(run_plain), "plain")
    for BN, D, nw in SERIAL_FWD:
        if T % D: continue
        t = timeit(lambda: run_tiled(BN, D, nw))
        if t < best[0]: best = (t, (BN, D, nw))
    return best
