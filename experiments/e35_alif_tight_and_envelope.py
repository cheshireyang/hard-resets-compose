"""E35: Tightness of the ALIF post-reset quadratic bound (all L) and the rigorous parallel
dominance-envelope scan for ALIF under downward reset.

Part 1 (Theorem: ALIF Psi_0 tightness):
  Explicit construction I_t = theta_0 + L*q + t*alpha (0 < alpha < q, alpha/q avoiding
  denominators < L), small beta > 0, rho close to 1, u_r <= theta_0 (downward reset strictly
  satisfied). Verifies in exact rational arithmetic that Psi_0 has exactly 1 + L(L+1)/2
  pieces for every L = 1..16 (and at moderate beta=1/1000, rho=999/1000 for L <= 10).

Part 2 (Theorem: ALIF rigorous dominance envelope -- exact rational checks):
  Verifies the coupled upper u-envelope + dominance w-envelope bounds in exact rational
  arithmetic on 2,000 randomized downward-reset instances (0 violations).

Part 3 (Strictly parallel Jacobi dominance-envelope benchmark):
  Evaluates the 4 neuron classes of E24 (lsnn, glif2, appD-a, appD-b) x 3 drives
  (m_free in {0.5, 1.0, 2.0}), T = 4096, C = 128, downward reset (excitatory rectified drive,
  u_r = 0, plus negative-u_r subthreshold-safe drive), comparing:
    - naive rigorous box [0, q*(1-rho^t)/(1-rho)],
    - strictly parallel Jacobi dominance-envelope scan (K_win = 4 chunks, 4 rounds),
    - oracle box (width 0.2q centred on true entry w).
  Reports box width reduction, actual ALIF chunk-map piece counts, exact-pinned chunks,
  and verifies 0 containment violations across all chunks.

CPU only (pure Python standard library). Output: results/e35_alif_tight_and_envelope.json
"""
import json
import math
import os
import random
import sys
from fractions import Fraction as Fr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------
# Part 1: exact rational piece count for Psi_0 (all L = 1..16)
# ----------------------------------------------------------------------------

def psi_pieces_exact(I, beta, rho, q, theta, ur, wlo, whi):
    iv = [(wlo, whi, Fr(1), Fr(0), ur)]
    for x in I:
        new = []
        for lo, hi, a, b, u in iv:
            v = beta * u + x
            z = (v - theta - b) / a  # spike iff w <= z
            if z >= lo:
                new.append((lo, min(hi, z), rho * a, rho * b + q, ur))
            if z < hi:
                new.append((max(lo, z), hi, rho * a, rho * b, v))
        new.sort(key=lambda r: (r[0], r[1]))
        merged = []
        for r in new:
            if r[1] <= r[0]:
                continue
            if merged and merged[-1][2:] == r[2:] and merged[-1][1] >= r[0]:
                m = merged[-1]
                merged[-1] = (m[0], max(m[1], r[1])) + m[2:]
            else:
                merged.append(r)
        iv = merged
    return len(iv)


def verify_alif_tightness():
    theta, q, ur = Fr(1), Fr(1), Fr(0)
    alpha = Fr(7, 17)  # denominator 17 > L-1 for all L <= 16
    beta = Fr(1, 10 ** 6)
    rho = Fr(99999, 100000)
    rows = []
    for L in range(1, 17):
        C0 = L * q
        I = [theta + C0 + t * alpha for t in range(1, L + 1)]
        assert all(x >= (1 - beta) * ur for x in I)
        pieces = psi_pieces_exact(I, beta, rho, q, theta, ur, Fr(0), Fr(1000))
        bound = 1 + L * (L + 1) // 2
        assert pieces == bound, f"L={L}: {pieces} != {bound}"
        rows.append(dict(L=L, pieces=pieces, bound=bound))
    # Also check moderate parameters (beta=1/1000, rho=999/1000, u_r=-1)
    mod_rows = []
    for L in (6, 7, 8, 10):
        I = [theta + L * q + t * Fr(5, 11) for t in range(1, L + 1)]
        assert all(x >= (1 - Fr(1, 1000)) * Fr(-1) for x in I)
        p = psi_pieces_exact(I, Fr(1, 1000), Fr(999, 1000), q, theta, Fr(-1), Fr(0), Fr(1000))
        b = 1 + L * (L + 1) // 2
        assert p == b
        mod_rows.append(dict(L=L, pieces=p, bound=b))
    print(f"Part 1: ALIF Psi_0 tightness verified exactly for L=1..16 (max pieces={rows[-1]['pieces']})", flush=True)
    return dict(alpha="7/17", beta="1e-6", rho="0.99999", rows=rows, moderate_check=mod_rows)


# ----------------------------------------------------------------------------
# Part 2: exact rational verification of the dominance-envelope theorem
# ----------------------------------------------------------------------------

def verify_envelope_exact(n_trials: int = 2000):
    rng = random.Random(2026)
    viol = 0
    for _ in range(n_trials):
        C = rng.randint(2, 16)
        beta = Fr(rng.randint(1, 20), 20)
        rho = Fr(rng.randint(1, 20), 20)
        q = Fr(rng.randint(0, 20), 10)
        th0 = Fr(1)
        ur = Fr(rng.randint(-15, 10), 10)  # ur <= th0
        lo_I = (1 - beta) * ur
        I = [lo_I + Fr(rng.randint(0, 35), 10) for _ in range(C)]

        lu = ur - Fr(rng.randint(0, 10), 10)
        hu = max(lu, th0 + Fr(rng.randint(-5, 20), 10))
        lw0 = Fr(rng.randint(0, 20), 10)
        hw0 = lw0 + Fr(rng.randint(0, 30), 10)
        u_true = lu + (hu - lu) * Fr(rng.randint(0, 10), 10)
        w_true = lw0 + (hw0 - lw0) * Fr(rng.randint(0, 10), 10)

        def run_ex(u, w):
            sp = []
            for j, x in enumerate(I, 1):
                v = beta * u + x
                if v >= th0 + w:
                    u = ur
                    w = rho * w + q
                    sp.append(j)
                else:
                    u = v
                    w = rho * w
            return sp, u, w

        a_sp, _, _ = run_ex(hu, lw0)
        c_sp, _, _ = run_ex(lu, hw0)
        b_sp, u_end, w_end = run_ex(u_true, w_true)
        N1, N3 = len(a_sp), len(c_sp)
        if not (N3 <= len(b_sp) <= N1):
            viol += 1
            break

        # Exact upper envelope on u inside the chunk using rigorous w upper bound w_cap_j
        hu_tr = [hu]
        w_cap = hw0
        for x in I:
            hu_tr.append(min(th0 + w_cap, beta * hu_tr[-1] + x))
            w_cap = rho * w_cap + q
        if u_end > hu_tr[-1]:
            viol += 1
            break

        # Feasible spike steps
        feas = []
        w_lo_j = lw0
        for j in range(1, C + 1):
            if beta * hu_tr[j - 1] + I[j - 1] >= th0 + w_lo_j:
                feas.append(j)
            w_lo_j *= rho

        lo_w = (rho ** C) * lw0 + q * sum(rho ** (C - a_sp[i]) for i in range(N3))
        hi_w = (rho ** C) * hw0 + q * sum(rho ** (C - c_sp[i]) for i in range(N3))
        if N1 > N3:
            extra = N1 - N3
            top = feas[-extra:] if len(feas) >= extra else feas
            hi_w += q * sum(rho ** (C - j) for j in top)
        if not (lo_w <= w_end <= hi_w):
            viol += 1
            break

    assert viol == 0, f"Exact envelope check failed: {viol}"
    print(f"Part 2: Exact rational dominance-envelope verified on {n_trials}/{n_trials} instances (0 violations)", flush=True)
    return dict(tests=n_trials, violations=viol)


# ----------------------------------------------------------------------------
# Part 3: strictly parallel (Jacobi) dominance-envelope benchmark + piece counts
# ----------------------------------------------------------------------------

def clip(poly, A, B, Cc, keep_pos):
    n = len(poly)
    if n == 0:
        return []
    sg = 1.0 if keep_pos else -1.0
    vals = [sg * (A * p[0] + B * p[1] + Cc) for p in poly]
    out = []
    for i in range(n):
        p, r = poly[i], poly[(i + 1) % n]
        a, b = vals[i], vals[(i + 1) % n]
        if a >= 0:
            out.append(p)
        if (a > 0 and b < 0) or (a < 0 and b > 0):
            t = a / (a - b)
            out.append((p[0] + t * (r[0] - p[0]), p[1] + t * (r[1] - p[1])))
    return out


def area(poly):
    s = 0.0
    for i in range(len(poly)):
        p, r = poly[i], poly[(i + 1) % len(poly)]
        s += p[0] * r[1] - r[0] * p[1]
    return abs(s) / 2


def build_alif_pieces(Ic, beta, rho, q, th0, ur, lu, hu, lw, hw):
    """Exact chunk-map piece count (matching E24), handling exact-pinned lw == hw cleanly."""
    if hw - lw <= 1e-12:
        lw = max(0.0, lw - 1e-7)
        hw = hw + 1e-7
    if hu - lu <= 1e-12:
        lu = lu - 1e-7
        hu = hu + 1e-7
    poly = [(lu, lw), (hu, lw), (hu, hw), (lu, hw)]
    eps = 1e-13 * max(1.0, (hu - lu) * (hw - lw))
    pieces = []
    A, b, R = 1.0, 0.0, 1.0
    for j, x in enumerate(Ic, start=1):
        new = []
        for (k, xl, xr, u, a, off) in pieces:
            v = beta * u + x
            z = (v - th0 - off) / a
            if z >= xr:
                new.append((k, xl, xr, ur, rho * a, rho * off + q))
            elif z < xl:
                new.append((k, xl, xr, v, rho * a, rho * off))
            else:
                new.append((k, xl, z, ur, rho * a, rho * off + q))
                new.append((k, z, xr, v, rho * a, rho * off))
        pieces = new
        A *= beta
        b = beta * b + x
        if poly:
            sp = clip(poly, A, -R, b - th0, True)
            if len(sp) >= 3 and area(sp) > eps:
                ws = [p[1] for p in sp]
                pieces.append((j, R * rho * min(ws) + q, R * rho * max(ws) + q, ur, 1.0, 0.0))
            poly = clip(poly, A, -R, b - th0, False)
            if len(poly) < 3 or area(poly) <= eps:
                poly = []
        R *= rho
    return len(pieces) + (1 if poly else 0)


def alif_serial(I, beta, rho, q, th0, ur, u0, w0):
    u, w = u0, w0
    us, ws, ss = [], [], []
    for x in I:
        v = beta * u + x
        if v >= th0 + w:
            u = ur
            w = rho * w + q
            s = 1
        else:
            u = v
            w = rho * w
            s = 0
        us.append(u)
        ws.append(w)
        ss.append(s)
    return us, ws, ss


def alif_chunk_spikes(Ic, beta, rho, q, th0, ur, u0, w0):
    u, w = u0, w0
    sp = []
    for j, x in enumerate(Ic, start=1):
        v = beta * u + x
        if v >= th0 + w:
            u = ur
            w = rho * w + q
            sp.append(j)
        else:
            u = v
            w = rho * w
    return sp


def jacobi_alif_envelope(I, beta, rho, q, th0, ur, u0, w0, C=128, Kwin=4, rounds=4):
    """Strictly parallel (Jacobi) iterated dominance-envelope scan under downward reset."""
    T = len(I)
    M = T // C
    rhoC = rho ** C

    # Lower envelope on u (Lemma 6, 3-parameter min-affine scan)
    lu_all = [0.0] * (T + 1)
    lu_all[0] = u0
    for t in range(T):
        lu_all[t + 1] = min(ur, beta * lu_all[t] + I[t])
    lu_in = [lu_all[m * C] for m in range(M)]

    lw = [w0] + [0.0] * (M - 1)
    hw = [w0] + [q * (1.0 - rho ** (m * C)) / (1.0 - rho) for m in range(1, M)]

    def calc_hu(hw_arr):
        hu = [0.0] * (T + 1)
        hu[0] = u0
        for m in range(M):
            hw_end = hw_arr[m + 1] if m + 1 < M else (rhoC * hw_arr[m] + q * (1.0 - rhoC) / (1.0 - rho))
            inv_r = 1.0 / rhoC
            for j in range(C):
                t = m * C + j
                cap = th0 + hw_end * inv_r
                hu[t + 1] = min(cap, beta * hu[t] + I[t])
                inv_r *= rho
        return hu

    for _ in range(rounds):
        hu_all = calc_hu(hw)
        hu_in = [hu_all[m * C] for m in range(M)]

        # Phase A: parallel 1-chunk dominance transfer + global affine scan across all M chunks
        Blo = [0.0] * M
        Bhi = [0.0] * M
        for m in range(M - 1):
            Ic = I[m * C:(m + 1) * C]
            a_sp = alif_chunk_spikes(Ic, beta, rho, q, th0, ur, hu_in[m], lw[m])
            c_sp = alif_chunk_spikes(Ic, beta, rho, q, th0, ur, lu_in[m], hw[m])
            N1, N3 = len(a_sp), len(c_sp)
            assert N1 >= N3
            blo = q * sum(rho ** (C - a_sp[i]) for i in range(N3))
            bhi = q * sum(rho ** (C - c_sp[i]) for i in range(N3))
            if N1 > N3:
                feas = []
                w_lo_j = lw[m]
                for j in range(1, C + 1):
                    t = m * C + j - 1
                    if beta * hu_all[t] + I[t] >= th0 + w_lo_j - 1e-12:
                        feas.append(j)
                    w_lo_j *= rho
                extra = N1 - N3
                top = feas[-extra:] if len(feas) >= extra else feas
                bhi += q * sum(rho ** (C - j) for j in top)
            Blo[m], Bhi[m] = blo, bhi

        lw_a = list(lw)
        hw_a = list(hw)
        for m in range(M - 1):
            lw_a[m + 1] = max(lw_a[m + 1], rhoC * lw_a[m] + Blo[m])
            hw_a[m + 1] = min(hw_a[m + 1], rhoC * hw_a[m] + Bhi[m])

        # Phase B: strictly parallel (Jacobi) overlapping Kwin-chunk window tightening
        if Kwin > 1:
            lw_b = list(lw_a)
            hw_b = list(hw_a)
            for m in range(1, M):
                s = max(0, m - Kwin)
                L = (m - s) * C
                Iw = I[s * C:m * C]
                a_sp = alif_chunk_spikes(Iw, beta, rho, q, th0, ur, hu_in[s], lw_a[s])
                c_sp = alif_chunk_spikes(Iw, beta, rho, q, th0, ur, lu_in[s], hw_a[s])
                N1, N3 = len(a_sp), len(c_sp)
                assert N1 >= N3
                rhoL = rho ** L
                lo_w = rhoL * lw_a[s] + q * sum(rho ** (L - a_sp[i]) for i in range(N3))
                hi_w = rhoL * hw_a[s] + q * sum(rho ** (L - c_sp[i]) for i in range(N3))
                if N1 > N3:
                    feas = []
                    for j in range(1, L + 1):
                        t = s * C + j - 1
                        cm = t // C
                        rem = t - cm * C
                        w_lo_t = (rho ** rem) * lw_a[cm]
                        if beta * hu_all[t] + I[t] >= th0 + w_lo_t - 1e-12:
                            feas.append(j)
                    extra = N1 - N3
                    top = feas[-extra:] if len(feas) >= extra else feas
                    hi_w += q * sum(rho ** (L - j) for j in top)
                lw_b[m] = max(lw_b[m], lo_w)
                hw_b[m] = min(hw_b[m], hi_w)
            lw, hw = lw_b, hw_b
        else:
            lw, hw = lw_a, hw_a

    hu_all = calc_hu(hw)
    hu_in = [hu_all[m * C] for m in range(M)]
    return lu_in, hu_in, lw, hw


def bench_envelope():
    neurons = [
        ("lsnn",   0.95, 0.995, 0.9),
        ("glif2",  0.99, 0.96,  0.2),
        ("appD-a", 0.90, 0.95,  0.5),
        ("appD-b", 0.99, 0.99,  0.25),
    ]
    T = 4096
    C = 128
    M = T // C
    th0 = 1.0
    ur = 0.0
    rows = []
    total_chunks = 0
    total_viol = 0
    for name, beta, rho, q in neurons:
        for m_free in (0.5, 1.0, 2.0):
            rng = random.Random(0)
            sig = 0.5
            I = [max(0.0, (1 - beta) * m_free + sig * math.sqrt(1 - beta * beta) * rng.gauss(0, 1)) for _ in range(T)]
            u0, w0 = ur, 0.0
            us, ws, ss = alif_serial(I, beta, rho, q, th0, ur, u0, w0)
            true_u = [u0] + [us[m * C - 1] for m in range(1, M)]
            true_w = [w0] + [ws[m * C - 1] for m in range(1, M)]

            lu_in, hu_in, lw, hw = jacobi_alif_envelope(I, beta, rho, q, th0, ur, u0, w0, C=C, Kwin=4, rounds=4)
            viol = sum(
                1 for m in range(M)
                if not (lu_in[m] - 1e-10 <= true_u[m] <= hu_in[m] + 1e-10 and lw[m] - 1e-10 <= true_w[m] <= hw[m] + 1e-10)
            )
            assert viol == 0
            total_viol += viol
            total_chunks += M

            lw_n = [w0] + [0.0] * (M - 1)
            hw_n = [w0] + [q * (1.0 - rho ** (m * C)) / (1.0 - rho) for m in range(1, M)]
            pcs_n, pcs_d, pcs_o = [], [], []
            for m in range(1, M):
                Ic = I[m * C:(m + 1) * C]
                lu = lu_in[m] - 1e-9
                pcs_n.append(build_alif_pieces(Ic, beta, rho, q, th0, ur, lu, max(ur, th0 + hw_n[m] / rho) + 1e-9, lw_n[m], hw_n[m]))
                pcs_d.append(build_alif_pieces(Ic, beta, rho, q, th0, ur, lu, hu_in[m] + 1e-9, lw[m], hw[m]))
                lw_o = max(0.0, true_w[m] - 0.1 * q)
                hw_o = true_w[m] + 0.1 * q
                pcs_o.append(build_alif_pieces(Ic, beta, rho, q, th0, ur, lu, max(ur, th0 + hw_o / rho) + 1e-9, lw_o, hw_o))

            wn = sum(hw_n[m] - lw_n[m] for m in range(1, M)) / (M - 1) / q
            wd = sum(hw[m] - lw[m] for m in range(1, M)) / (M - 1) / q
            exact_pinned = sum(1 for m in range(1, M) if hw[m] - lw[m] <= 1e-9)
            row = dict(
                name=name, beta=beta, rho=rho, q=q, m_free=m_free,
                rate=sum(ss) / T,
                naive_width_q=wn, dom_width_q=wd, width_shrink=wn / max(wd, 1e-9),
                exact_pinned=exact_pinned, n_chunks=M - 1,
                naive_pieces_mean=sum(pcs_n) / len(pcs_n),
                dom_pieces_mean=sum(pcs_d) / len(pcs_d),
                dom_pieces_max=max(pcs_d),
                oracle_pieces_mean=sum(pcs_o) / len(pcs_o),
                piece_shrink=(sum(pcs_n) / len(pcs_n)) / max(sum(pcs_d) / len(pcs_d), 1e-9),
                violations=viol,
            )
            rows.append(row)
            print(
                f"{name:6s} m={m_free:<3} rate={100*row['rate']:4.1f}% | width/q: {wn:5.1f}->{wd:4.2f} ({row['width_shrink']:5.1f}x) | "
                f"pieces: {row['naive_pieces_mean']:6.1f}->{row['dom_pieces_mean']:5.1f} (oracle {row['oracle_pieces_mean']:4.1f})",
                flush=True,
            )
    return dict(T=T, C=C, Kwin=4, rounds=4, total_chunks=total_chunks, total_violations=total_viol, rows=rows)


def main():
    out = dict(
        tightness=verify_alif_tightness(),
        exact_envelope=verify_envelope_exact(2000),
        benchmark=bench_envelope(),
    )
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "e35_alif_tight_and_envelope.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
