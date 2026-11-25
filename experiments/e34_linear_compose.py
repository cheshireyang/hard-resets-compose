"""E34: O(n_F + n_g) linear-time composition and O(T log T) exact tree scan.

Verifies Theorem (linear-time composition):
  1. Exact rational equivalence of LinearHardMap.then and tree_scan_states_linear against
     HardMap.then and serial_hard across 3,000 randomized instances (including beta=1,
     beta>1, negative inputs, u_r != 0, and exact breakpoint ties).
  2. Realizable worst-case single-composition scaling: two Proposition-2 segments of length n
     whose constants fall inside the right segment's breakpoints. Measures operation counts
     and wall-clock time for n = 128 .. 32,768: linear O(n) vs binary-search Theta(n log n).
  3. Full binary-tree scan operation count scaling for T = 64 .. 16,384, verifying
     O(T log T) up-sweep work and O(T) down-sweep work.

CPU only (pure Python standard library). Output: results/e34_linear_compose.json
"""
import json
import math
import os
import random
import sys
import time
from fractions import Fraction as Fr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lifscan import exact as E


def old_then_counted(F: E.HardMap, g: E.HardMap, ops: list) -> E.HardMap:
    """HardMap.then with explicit operation counter for fair comparison."""
    A, B = F.A, F.B
    nx, nc = [], []
    if g.x:
        for j, yj in enumerate(g.x):
            zj = (yj - B) / A
            ops[0] += 1
            if F.x and zj >= F.x[0]:
                break
            nx.append(zj)
            nc.append(g.c[j])
    gx = g.x
    for xi, ci in zip(F.x, F.c):
        nx.append(xi)
        if not gx or ci < gx[0]:
            ops[0] += 1
            nc.append(g.A * ci + g.B)
        else:
            lo, hi = 0, len(gx) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                ops[0] += 1
                if gx[mid] <= ci:
                    lo = mid
                else:
                    hi = mid - 1
            nc.append(g.c[lo])
    nx, nc = E._merge(nx, nc)
    return E.HardMap(g.A * A, g.A * B + g.B, nx, nc, F.L + g.L)


def check_exact_rational(n_trials: int = 3000):
    rng = random.Random(2026)
    bad = 0
    for _ in range(n_trials):
        T = rng.randint(1, 40)
        beta = Fr(rng.randint(1, 14), 10)  # covers beta < 1, = 1, > 1
        theta = Fr(rng.choice([-1, 1, 2]), 2)
        u_r = Fr(rng.choice([-3, 0, 0, 2, 7]), 10)
        I = [Fr(rng.randint(-15, 20), 10) for _ in range(T)]
        u0 = Fr(rng.randint(-30, 15), 10)

        us, ss = E.serial_hard(I, beta, theta, u_r, u0)
        ul, sl = E.tree_scan_states_linear(I, beta, theta, u_r, u0)

        # Compare composed root maps directly (HardMap vs LinearHardMap)
        leaves_l = [E.LinearHardMap.step(x, beta, theta, u_r) for x in I]
        leaves_o = [E.HardMap.step(x, beta, theta, u_r) for x in I]

        def build(leaves, lo, hi):
            if hi - lo == 1:
                return leaves[lo]
            mid = (lo + hi) // 2
            return build(leaves, lo, mid).then(build(leaves, mid, hi), merge=True)

        Ml = build(leaves_l, 0, T)
        Mo = build(leaves_o, 0, T)
        inv_ok = (
            Ml.v == sorted(set(Ml.v))
            and len(Ml.x) == len(Ml.cid)
            and len(Ml.v) <= len(Ml.x)
            and Ml.A == Mo.A
            and Ml.B == Mo.B
            and Ml.x == Mo.x
            and Ml.c == Mo.c
        )
        # Test exact breakpoint evaluations (ties)
        bp_ok = all(Ml(bp) == E.serial_hard(I, beta, theta, u_r, bp)[0][-1] for bp in Ml.x)
        if not (us == ul and ss == sl and inv_ok and bp_ok):
            bad += 1
            break
    assert bad == 0, f"Exact rational test failed ({bad})"
    return dict(trials=n_trials, failures=bad)


def bench_single_compose():
    rows = []
    for n in (128, 512, 2048, 8192, 32768):
        beta = 1.0 - 1e-7
        theta = 1.0
        u_r = 0.0
        # Realizable LIF segments of length n (Prop 2): F has n+1 pieces with constants in (0, 0.9),
        # g has n+1 pieces with breakpoints in (0.05, 1.0), so ~95% of F's constants hit inside g's breakpoints.
        I1 = [0.9 / n] * n
        I2 = [0.95 / n] * n

        def mk(I):
            A, B, xs, cs = E.closed_form_segment(I, beta, theta, u_r)
            old = E.HardMap(A, B, xs, cs, len(I))
            uniq = sorted(set(cs))
            vmap = {val: i for i, val in enumerate(uniq)}
            lin = E.LinearHardMap(A, B, xs, uniq, [vmap[c] for c in cs], len(I))
            return old, lin

        Fo, Fl = mk(I1)
        go, gl = mk(I2)
        assert Fo.pieces == n + 1 and go.pieces == n + 1

        # Operation counts
        ops_lin = [0]
        Hl = Fl.then(gl, merge=True, op_counter=ops_lin)
        ops_old = [0]
        Ho = old_then_counted(Fo, go, ops_old)
        assert len(Hl.x) == len(Ho.x) and Hl.x == Ho.x and Hl.c == Ho.c

        # Median wall-clock time (without op_counter instrumentation)
        reps = 15 if n <= 2048 else 5
        tl_list, to_list = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            Fl.then(gl, merge=True)
            tl_list.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            Fo.then(go, merge=True)
            to_list.append((time.perf_counter() - t0) * 1000.0)
        tl_list.sort()
        to_list.sort()
        tl_ms = tl_list[reps // 2]
        to_ms = to_list[reps // 2]

        logn = math.log2(n)
        row = dict(
            n=n,
            out_pieces=Hl.pieces,
            ops_linear=ops_lin[0],
            ops_old=ops_old[0],
            ops_linear_per_n=ops_lin[0] / n,
            ops_old_per_n=ops_old[0] / n,
            ops_old_per_nlogn=ops_old[0] / (n * logn),
            op_ratio=ops_old[0] / ops_lin[0],
            ms_linear=tl_ms,
            ms_old=to_ms,
            wall_speedup=to_ms / tl_ms,
        )
        rows.append(row)
        print(
            f"single n={n:5d} | lin ops/n={row['ops_linear_per_n']:.2f} ({tl_ms:6.2f}ms) | "
            f"old ops/n={row['ops_old_per_n']:5.2f} ({to_ms:6.2f}ms) | "
            f"op ratio={row['op_ratio']:.2f}x | wall={row['wall_speedup']:.2f}x",
            flush=True,
        )
    return rows


def bench_tree_scan():
    """Full binary-tree build + down-sweep op counts on Prop-2 worst-case inputs."""
    rows = []
    for T in (64, 256, 1024, 4096, 16384):
        beta = 1.0 - 1e-6
        theta = 1.0
        u_r = 0.0
        I_val = 0.5 * ((1.0 - beta) * theta + (1.0 - beta) * theta / (1.0 - beta ** T))
        I = [I_val] * T
        leaves = [E.LinearHardMap.step(x, beta, theta, u_r) for x in I]
        ops_up = [0]

        def build(lo, hi):
            if hi - lo == 1:
                return (leaves[lo], None, None)
            mid = (lo + hi) // 2
            Lc, Rc = build(lo, mid), build(mid, hi)
            M = Lc[0].then(Rc[0], merge=True, op_counter=ops_up)
            return (M, Lc, Rc)

        root = build(0, T)
        assert root[0].pieces == T + 1
        # Count down-sweep comparisons directly
        ops_down = [0]

        def down(node, u_in):
            M, Lc, Rc = node
            if Lc is None:
                return
            lm = Lc[0]
            x = lm.x
            if not x or u_in < x[0]:
                ops_down[0] += 1
                umid = lm.A * u_in + lm.B
            else:
                lo, hi = 0, len(x) - 1
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    ops_down[0] += 1
                    if x[mid] <= u_in:
                        lo = mid
                    else:
                        hi = mid - 1
                umid = lm.v[lm.cid[lo]]
            down(Lc, u_in)
            down(Rc, umid)

        # Choose u0 above all breakpoints so every internal node performs full binary search!
        down(root, 2.0)
        logT = T.bit_length() - 1
        row = dict(
            T=T,
            root_pieces=root[0].pieces,
            ops_up=ops_up[0],
            ops_up_per_TlogT=ops_up[0] / (T * logT),
            ops_down=ops_down[0],
            ops_down_per_T=ops_down[0] / T,
        )
        rows.append(row)
        print(
            f"tree   T={T:5d} | up ops/(T log T)={row['ops_up_per_TlogT']:.2f} | "
            f"down ops/T={row['ops_down_per_T']:.2f}",
            flush=True,
        )
    return rows


def main():
    out = dict(
        exact=check_exact_rational(3000),
        single_compose=bench_single_compose(),
        tree_scan=bench_tree_scan(),
    )
    print("exact:", out["exact"])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "e34_linear_compose.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
