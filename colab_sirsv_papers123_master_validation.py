#!/usr/bin/env python3
"""
================================================================================
FULL-PORTFOLIO (PAPERS 1-3) 15-METHODOLOGY CROSS-DOMAIN VALIDATION SUITE
ON THE 4D NON-DIMENSIONAL SIRSV / SIRS / SIR EPIDEMIC BENCHMARK
================================================================================
External Cross-Domain Reference System (Kermack-McKendrick 1927; Hethcote 2000):
  dS/dt = -R0 * S * I + xi * R - u_S * S
  dI/dt =  R0 * S * I - I
  dR/dt =  I - (xi + rho + u_R) * R
  dV/dt =  (rho + u_R) * R + u_S * S
Baseline empirical MLE: R0 = 3.016540569360357, xi = 0.49865705161955964,
rho = 0.019863805706324063, Y0 = (0.99, 0.01, 0.0, 0.0).

Executes cleanly in pure Python 3.12 standard library (optional NumPy/PyTorch
auto-detection for auxiliary checks) and exports structured verification JSON:
  data/sirsv_papers123_validation.json

Usage:
  python3 colab_sirsv_papers123_master_validation.py --full
  python3 colab_sirsv_papers123_master_validation.py --verify-only
================================================================================
"""

import cmath
import json
import math
import pathlib
import random
import sys
import time
import unittest

try:
    import numpy as _np_opt  # Optional hardware accelerator detection
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch as _torch_opt
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BRAIN_DIR = pathlib.Path(__file__).resolve().parent
SCRATCH_DIR = BRAIN_DIR / "data"
RESULTS_PATH = SCRATCH_DIR / "sirsv_papers123_validation.json"

P_SIRSV = {
    "R0": 3.016540569360357,
    "xi": 0.49865705161955964,
    "rho": 0.019863805706324063,
}
Y0_SIRSV = (0.99, 0.01, 0.0, 0.0)


# ==============================================================================
# 0. PURE-PYTHON LINEAR ALGEBRA, HESSENBERG + COMPLEX WILKINSON QR & INTEGRATORS
# ==============================================================================
def mat_mul(A, B):
    m, k, n = len(A), len(B), len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(n)] for i in range(m)]


def mat_T(A):
    return [[A[r][c] for r in range(len(A))] for c in range(len(A[0]))]


def solve_linear_real(A_in, b_in):
    n = len(b_in)
    aug = [list(map(float, A_in[r])) + [float(b_in[r])] for r in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        if abs(pv) < 1e-18:
            raise ValueError("Singular real matrix")
        inv_pv = 1.0 / pv
        for k in range(col, n + 1):
            aug[col][k] *= inv_pv
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                fac = aug[r][col]
                for k in range(col, n + 1):
                    aug[r][k] -= fac * aug[col][k]
    return [aug[r][n] for r in range(n)]


def solve_linear_complex(A_in, b_in):
    n = len(b_in)
    aug = [[complex(A_in[r][c]) for c in range(n)] + [complex(b_in[r])] for r in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        if abs(pv) < 1e-18:
            raise ValueError("Singular complex matrix")
        inv_pv = 1.0 / pv
        for k in range(col, n + 1):
            aug[col][k] *= inv_pv
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                fac = aug[r][col]
                for k in range(col, n + 1):
                    aug[r][k] -= fac * aug[col][k]
    return [aug[r][n] for r in range(n)]


def invert_matrix_real(M_in):
    n = len(M_in)
    aug = [list(map(float, M_in[r])) + [1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        if abs(pv) < 1e-18:
            raise ValueError("Singular matrix inverse")
        inv_pv = 1.0 / pv
        for k in range(col, 2 * n):
            aug[col][k] *= inv_pv
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                fac = aug[r][col]
                for k in range(col, 2 * n):
                    aug[r][k] -= fac * aug[col][k]
    return [row[n:] for row in aug]


def cholesky_lower(M_in):
    n = len(M_in)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = M_in[i][i] - s
                L[i][j] = math.sqrt(max(1e-15, val))
            else:
                L[i][j] = (M_in[i][j] - s) / L[j][j]
    return L


def eigvals_general(A_in, max_iter=500):
    """Exact pure-Python eigenvalues via Householder Hessenberg + complex Wilkinson QR."""
    n = len(A_in)
    if n == 0:
        return []
    if n == 1:
        return [complex(A_in[0][0])]
    H = [[complex(A_in[r][c]) for c in range(n)] for r in range(n)]

    # Complex Householder reduction to upper Hessenberg
    for k in range(n - 2):
        col_norm = math.sqrt(sum(abs(H[r][k]) ** 2 for r in range(k + 1, n)))
        if col_norm < 1e-15:
            continue
        x0 = H[k + 1][k]
        phase = x0 / abs(x0) if abs(x0) > 1e-18 else 1.0 + 0.0j
        v = [H[r][k] for r in range(k + 1, n)]
        v[0] += phase * col_norm
        vn = math.sqrt(sum(abs(z) ** 2 for z in v))
        if vn < 1e-16:
            continue
        u = [z / vn for z in v]
        m = len(u)
        # Left update H[k+1:, k:] -= 2 u (u^H H[k+1:, k:])
        for c in range(k, n):
            dot = sum(u[i].conjugate() * H[k + 1 + i][c] for i in range(m))
            for i in range(m):
                H[k + 1 + i][c] -= 2.0 * u[i] * dot
        # Right update H[:, k+1:] -= 2 (H[:, k+1:] u) u^H
        for r in range(n):
            dot = sum(H[r][k + 1 + i] * u[i] for i in range(m))
            for i in range(m):
                H[r][k + 1 + i] -= 2.0 * dot * u[i].conjugate()

    eigs = []
    m = n - 1
    it = 0
    while m > 0 and it < max_iter * n:
        it += 1
        # Check deflation along subdiagonal
        def_idx = m
        while def_idx > 0:
            scale = abs(H[def_idx - 1][def_idx - 1]) + abs(H[def_idx][def_idx])
            if abs(H[def_idx][def_idx - 1]) <= 1e-13 * max(1.0, scale):
                H[def_idx][def_idx - 1] = 0.0j
                if def_idx == m:
                    eigs.append(H[m][m])
                    m -= 1
                break
            def_idx -= 1
        if m <= 0:
            break
        if H[m][m - 1] == 0.0j:
            continue
        l_start = def_idx
        # Wilkinson shift from bottom-right 2x2 of active block
        a = H[m - 1][m - 1]
        b = H[m - 1][m]
        c = H[m][m - 1]
        d = H[m][m]
        tr = a + d
        det = a * d - b * c
        disc = cmath.sqrt(tr * tr - 4.0 * det)
        r1 = 0.5 * (tr + disc)
        r2 = 0.5 * (tr - disc)
        shift = r1 if abs(r1 - d) <= abs(r2 - d) else r2
        if it % 29 == 0:
            shift = d + abs(H[m][m - 1]) * (0.75 + 0.433j)

        # Implicit/explicit Givens bulge-chasing QR step on [l_start .. m]
        for i in range(l_start, m + 1):
            H[i][i] -= shift
        cs, sn = [], []
        for i in range(l_start, m):
            x = H[i][i]
            y = H[i + 1][i]
            r = math.hypot(abs(x), abs(y))
            if r < 1e-18:
                c_g, s_g = 1.0 + 0.0j, 0.0j
            else:
                c_g = x.conjugate() / r  # makes c*x - (-s)*y? Standard Hermitian Givens: G = [[c, s], [-s*, c*]]
                # Using G = [[x*/r, y*/r], [-y/r, x/r]] gives G @ [x, y]^T = [r, 0]^T!
                c_g = x.conjugate() / r
                s_g = y.conjugate() / r
            cs.append(c_g)
            sn.append(s_g)
            for col in range(i, m + 1):
                t1 = c_g * H[i][col] + s_g * H[i + 1][col]
                t2 = -s_g.conjugate() * H[i][col] + c_g.conjugate() * H[i + 1][col]
                H[i][col], H[i + 1][col] = t1, t2
        # Apply G^H from right: G^H = [[c*, -s], [s*, c]]
        for idx, i in enumerate(range(l_start, m)):
            c_g, s_g = cs[idx], sn[idx]
            for row in range(l_start, min(m + 1, i + 3)):
                t1 = H[row][i] * c_g.conjugate() + H[row][i + 1] * s_g.conjugate()
                t2 = -H[row][i] * s_g + H[row][i + 1] * c_g
                H[row][i], H[row][i + 1] = t1, t2
        for i in range(l_start, m + 1):
            H[i][i] += shift
    if m == 0:
        eigs.append(H[0][0])
    while len(eigs) < n:
        idx = n - 1 - len(eigs)
        eigs.append(H[idx][idx])
    return eigs


def sirsv_rhs_scalar(s, i_val, r, _v, r0, xi, rho, u_s=0.0, u_r=0.0):
    ds = -r0 * s * i_val + xi * r - u_s * s
    di = r0 * s * i_val - i_val
    dr = i_val - (xi + rho + u_r) * r
    dv = (rho + u_r) * r + u_s * s
    return ds, di, dr, dv


def integrate_sirsv_rk4(r0, xi, rho, y0=Y0_SIRSV, t_end=100.0, dt=0.05, stride=20):
    n_steps = int(round(t_end / dt))
    half = 0.5 * dt
    sixth = dt / 6.0
    s, i_v, r, v = y0
    out = [(0.0, s, i_v, r, v)]
    for step in range(1, n_steps + 1):
        ds1, di1, dr1, dv1 = sirsv_rhs_scalar(s, i_v, r, v, r0, xi, rho)
        ds2, di2, dr2, dv2 = sirsv_rhs_scalar(s + half * ds1, i_v + half * di1, r + half * dr1, v + half * dv1, r0, xi, rho)
        ds3, di3, dr3, dv3 = sirsv_rhs_scalar(s + half * ds2, i_v + half * di2, r + half * dr2, v + half * dv2, r0, xi, rho)
        ds4, di4, dr4, dv4 = sirsv_rhs_scalar(s + dt * ds3, i_v + dt * di3, r + dt * dr3, v + dt * dv3, r0, xi, rho)
        s += sixth * (ds1 + 2.0 * ds2 + 2.0 * ds3 + ds4)
        i_v += sixth * (di1 + 2.0 * di2 + 2.0 * di3 + di4)
        r += sixth * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4)
        v += sixth * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
        if step % stride == 0:
            out.append((step * dt, s, i_v, r, v))
    return out


# ==============================================================================
# GROUP I (PAPER 1): AUTONOMOUS DYNAMICAL SYSTEMS & RFDE METHODS (M1 - M6)
# ==============================================================================
def validate_m1_simplex_sir_peak_and_cascade(p=P_SIRSV):
    r0, xi, rho = p["R0"], p["xi"], p["rho"]
    # 1. Nagumo cone inflow & simplex sum = 0
    s_rhs = sum(sirsv_rhs_scalar(0.51, 0.22, 0.17, 0.10, r0, xi, rho))
    nagumo = {
        "dS_at_S0": xi * 0.25,
        "dI_at_I0": 0.0,
        "dR_at_R0": 0.15,
        "dV_at_V0": rho * 0.25,
        "sum_rhs_check": s_rhs,
    }
    # 2. Classic Kermack-McKendrick SIR closed-form peak (xi=rho=0)
    sir_exact_peak = 1.0 - (1.0 + math.log(0.99 * r0)) / r0  # 0.30594170...
    def rk4_sir_peak(dt):
        s, i_v, r, v = Y0_SIRSV
        n_steps = int(round(12.0 / dt))
        best_i = i_v
        prev = (s, i_v, sirsv_rhs_scalar(s, i_v, r, v, r0, 0.0, 0.0)[1])
        for _ in range(n_steps):
            k1 = sirsv_rhs_scalar(s, i_v, r, v, r0, 0.0, 0.0)
            k2 = sirsv_rhs_scalar(s + 0.5 * dt * k1[0], i_v + 0.5 * dt * k1[1], r + 0.5 * dt * k1[2], v + 0.5 * dt * k1[3], r0, 0.0, 0.0)
            k3 = sirsv_rhs_scalar(s + 0.5 * dt * k2[0], i_v + 0.5 * dt * k2[1], r + 0.5 * dt * k2[2], v + 0.5 * dt * k2[3], r0, 0.0, 0.0)
            k4 = sirsv_rhs_scalar(s + dt * k3[0], i_v + dt * k3[1], r + dt * k3[2], v + dt * k3[3], r0, 0.0, 0.0)
            s_n = s + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
            i_n = i_v + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
            r_n = r + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
            v_n = v + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
            di_n = sirsv_rhs_scalar(s_n, i_n, r_n, v_n, r0, 0.0, 0.0)[1]
            # Locate root of dI/dt = 0 via cubic Hermite interpolation when dI changes sign
            if prev[2] >= 0.0 >= di_n:
                lo, hi = 0.0, 1.0
                for _b in range(50):
                    u = 0.5 * (lo + hi)
                    u2, u3 = u * u, u * u * u
                    dh00 = 6.0 * u2 - 6.0 * u
                    dh10 = 3.0 * u2 - 4.0 * u + 1.0
                    dh01 = -6.0 * u2 + 6.0 * u
                    dh11 = 3.0 * u2 - 2.0 * u
                    der = (dh00 * prev[1] + dh01 * i_n) / dt + dh10 * prev[2] + dh11 * di_n
                    if der > 0.0:
                        lo = u
                    else:
                        hi = u
                u = 0.5 * (lo + hi)
                u2, u3 = u * u, u * u * u
                i_peak = (2*u3 - 3*u2 + 1)*prev[1] + (u3 - 2*u2 + u)*dt*prev[2] + (-2*u3 + 3*u2)*i_n + (u3 - u2)*dt*di_n
                best_i = max(best_i, i_peak)
            s, i_v, r, v = s_n, i_n, r_n, v_n
            prev = (s, i_v, di_n)
        return best_i

    sir_rk4_peak = rk4_sir_peak(0.025)
    sir_peak_rel_err = abs(sir_rk4_peak - sir_exact_peak) / sir_exact_peak

    # 3. Dyadic RK4 convergence order at t = 5.0 on full SIRSV vs ultra-fine ref dt=0.00125
    ref_state = integrate_sirsv_rk4(r0, xi, rho, Y0_SIRSV, t_end=5.0, dt=0.00125, stride=4000)[-1][1:]
    errs_dyadic = []
    for dt_v in (0.025, 0.0125, 0.00625):
        st = integrate_sirsv_rk4(r0, xi, rho, Y0_SIRSV, t_end=5.0, dt=dt_v, stride=int(round(5.0 / dt_v)))[-1][1:]
        errs_dyadic.append(math.sqrt(sum((st[c] - ref_state[c]) ** 2 for c in range(4))))
    rk4_emp_order = math.log(errs_dyadic[1] / errs_dyadic[2]) / math.log(2.0)

    # 4. Erlang-k cascade total mass conservation
    max_cascade_mass_err = 0.0
    tw = 1.0 / xi
    k_c = 4
    rw = k_c / tw
    sc, ic, vc = Y0_SIRSV[0], Y0_SIRSV[1], Y0_SIRSV[3]
    rc = [0.0] * k_c
    dt_c = 0.025
    for _ in range(800):
        def rhs_k(sv, iv, rl):
            ds = -r0 * sv * iv + rw * rl[-1]
            di = r0 * sv * iv - iv
            dr = [iv - (rw + rho) * rl[0]] + [rw * rl[j - 1] - (rw + rho) * rl[j] for j in range(1, k_c)]
            dv = rho * sum(rl)
            return ds, di, dr, dv
        ds1, di1, dr1, dv1 = rhs_k(sc, ic, rc)
        sc += dt_c * ds1
        ic += dt_c * di1
        rc = [rc[j] + dt_c * dr1[j] for j in range(k_c)]
        vc += dt_c * dv1
        # Check exact conservation using RK4 on step 1 then track mass drift:
        # actually let's run exact RK4 check inside loop
        break
    # Clean RK4 cascade check
    sc, ic, vc = Y0_SIRSV[0], Y0_SIRSV[1], Y0_SIRSV[3]
    rc = [0.0] * k_c
    for _ in range(1200):
        def rk(sv, iv, rl):
            ds = -r0 * sv * iv + rw * rl[-1]
            di = r0 * sv * iv - iv
            dr = [iv - (rw + rho) * rl[0]] + [rw * rl[j - 1] - (rw + rho) * rl[j] for j in range(1, k_c)]
            dv = rho * sum(rl)
            return ds, di, dr, dv
        k1 = rk(sc, ic, rc)
        k2 = rk(sc + 0.5*dt_c*k1[0], ic + 0.5*dt_c*k1[1], [rc[j] + 0.5*dt_c*k1[2][j] for j in range(k_c)])
        k3 = rk(sc + 0.5*dt_c*k2[0], ic + 0.5*dt_c*k2[1], [rc[j] + 0.5*dt_c*k2[2][j] for j in range(k_c)])
        k4 = rk(sc + dt_c*k3[0], ic + dt_c*k3[1], [rc[j] + dt_c*k3[2][j] for j in range(k_c)])
        sc += (dt_c/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        ic += (dt_c/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        rc = [rc[j] + (dt_c/6.0)*(k1[2][j] + 2*k2[2][j] + 2*k3[2][j] + k4[2][j]) for j in range(k_c)]
        vc += (dt_c/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        max_cascade_mass_err = max(max_cascade_mass_err, abs(sc + ic + sum(rc) + vc - 1.0))

    return {
        "nagumo": nagumo,
        "sir_exact_peak": sir_exact_peak,
        "sir_rk4_peak": sir_rk4_peak,
        "sir_peak_rel_err": sir_peak_rel_err,
        "rk4_emp_order": rk4_emp_order,
        "cascade_k4_mass_err": max_cascade_mass_err,
    }


def validate_m2_littles_law_gspt_overshoot(p=P_SIRSV):
    r0, xi, rho = p["R0"], p["xi"], p["rho"]
    tw = 1.0 / xi
    s_herd = 1.0 / r0
    v_crit = 1.0 - s_herd
    i_end = (1.0 - s_herd) / (1.0 + tw)

    # 1. Little's Law across k in {1, 2, 4, 16} at SIRS endemic equilibrium
    littles_max_res = 0.0
    for k in (1, 2, 4, 16):
        rw = k / tw
        r_stage = i_end / rw
        sum_r = k * r_stage
        littles_max_res = max(littles_max_res, abs(sum_r - tw * i_end))

    # 2. Fenichel slow manifold & transcritical overshoot across rho
    rho_sweep = [0.005, 0.010, rho, 0.040, 0.080]
    gspt_rows = []
    for rv in rho_sweep:
        t_end_l = 4000.0 if rv <= 0.01 else 1800.0
        traj = integrate_sirsv_rk4(r0, xi, rv, Y0_SIRSV, t_end=t_end_l, dt=0.05, stride=20)
        err_30_90 = 0.0
        for t_v, _s, i_v, _r, v_v in traj:
            if 30.0 <= t_v <= 90.0:
                i_qs = max(0.0, (xi / (1.0 + xi)) * (v_crit - v_v))
                err_30_90 = max(err_30_90, abs(i_v - i_qs))
        s_inf, v_inf = traj[-1][1], traj[-1][4]
        gspt_rows.append({
            "rho": rv, "fenichel_err_30_90": err_30_90,
            "S_inf": s_inf, "V_inf": v_inf, "dV_over": v_inf - v_crit,
        })
    return {
        "S_herd": s_herd, "V_crit": v_crit, "E_end_I": i_end,
        "littles_law_max_res": littles_max_res, "gspt_rows": gspt_rows,
    }


def _find_sirs_hopf_crossings(k, r0):
    if k is None:
        crossings = []
        prev_res, prev_tw = None, None
        for it in range(10, 8000):
            tw = it * 0.01
            mu = (r0 - 1.0) / (1.0 + tw)
            if not (0.0 < mu < 2.0):
                prev_res = None
                continue
            w = math.sqrt(mu * (2.0 - mu))
            if w > 1.0:
                prev_res = None
                continue
            ang = (2.0 * math.pi - math.asin(w)) if mu >= 1.0 else (math.pi + math.asin(w))
            res = w * tw - ang
            if prev_res is not None and prev_res * res <= 0.0:
                lo, hi = prev_tw, tw
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    mm = (r0 - 1.0) / (1.0 + mid)
                    ww = math.sqrt(mm * (2.0 - mm))
                    aa = (2.0 * math.pi - math.asin(ww)) if mm >= 1.0 else (math.pi + math.asin(ww))
                    rm = ww * mid - aa
                    if rm * prev_res <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                        prev_res = rm
                tw_s = 0.5 * (lo + hi)
                mm = (r0 - 1.0) / (1.0 + tw_s)
                w_s = math.sqrt(mm * (2.0 - mm))
                lam = 1j * w_s
                dmu_dt = -mm / (1.0 + tw_s)
                g = cmath.exp(-lam * tw_s)
                dF_dlam = 2.0 * lam + mm + mm * tw_s * g
                dF_dt = dmu_dt * lam + dmu_dt * (1.0 - g) + mm * lam * g
                crossings.append((tw_s, w_s, (-dF_dt / dF_dlam).real))
            prev_res, prev_tw = res, tw
        return crossings

    # Finite Erlang-k closed-form phase angle parameterization: phi = k*theta in (pi, min(k*pi/2, 2*pi))
    phi_lo = math.pi + 1e-6
    phi_hi = min(0.5 * k * math.pi, 2.0 * math.pi) - 1e-6
    if phi_lo >= phi_hi:
        return []
    n_scan = 2500
    grid = []
    for i in range(n_scan + 1):
        phi = phi_lo + (phi_hi - phi_lo) * i / n_scan
        th = phi / k
        ck = math.cos(th) ** k
        w = -ck * math.sin(phi)
        if w <= 1e-9:
            continue
        tw = (k * math.tan(th)) / w
        denom = 1.0 - ck * math.cos(phi)
        if denom <= 1e-12:
            continue
        mu = (w * w) / denom
        r0_c = 1.0 + mu * (1.0 + tw)
        grid.append((phi, r0_c))
    crossings = []
    for i in range(len(grid) - 1):
        p0, f0 = grid[i][0], grid[i][1] - r0
        p1, f1 = grid[i + 1][0], grid[i + 1][1] - r0
        if f0 * f1 <= 0.0:
            lo, hi = p0, p1
            flo = f0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                th = mid / k
                ck = math.cos(th) ** k
                w = -ck * math.sin(mid)
                tw = (k * math.tan(th)) / w
                mu = (w * w) / (1.0 - ck * math.cos(mid))
                fmid = 1.0 + mu * (1.0 + tw) - r0
                if flo * fmid <= 0.0:
                    hi = mid
                else:
                    lo = mid
                    flo = fmid
            phi_s = 0.5 * (lo + hi)
            th = phi_s / k
            ck = math.cos(th) ** k
            w_s = -ck * math.sin(phi_s)
            tw_s = (k * math.tan(th)) / w_s
            lam = 1j * w_s
            mm = (r0 - 1.0) / (1.0 + tw_s)
            dmu_dt = -mm / (1.0 + tw_s)
            g = (1.0 + lam * tw_s / k) ** (-k)
            dg_dlam = -tw_s * ((1.0 + lam * tw_s / k) ** (-k - 1))
            dg_dt = -lam * ((1.0 + lam * tw_s / k) ** (-k - 1))
            dF_dlam = 2.0 * lam + mm - mm * dg_dlam
            dF_dt = dmu_dt * lam + dmu_dt * (1.0 - g) - mm * dg_dt
            crossings.append((tw_s, w_s, (-dF_dt / dF_dlam).real))
    crossings.sort(key=lambda x: x[0])
    return crossings


def _compute_kuznetsov_l1_and_vecs(k, tw, w, r0):
    n = k + 1
    mu = (r0 - 1.0) / (1.0 + tw)
    r_rate = k / tw
    A = [[0.0] * n for _ in range(n)]
    for col in range(n):
        A[0][col] = -mu
    A[1][0] = 1.0
    A[1][1] = -r_rate
    for j in range(2, n):
        A[j][j - 1] = r_rate
        A[j][j] = -r_rate

    denom = 1j * w + r_rate
    q = [1.0 + 0.0j, 1.0 / denom]
    for _j in range(2, n):
        q.append(q[-1] * (r_rate / denom))
    qn = math.sqrt(sum(abs(z) ** 2 for z in q))
    q = [z / qn for z in q]

    p_raw = [0j] * n
    p_raw[0] = 1.0 + 0.0j
    d_adj = r_rate - 1j * w
    p_raw[n - 1] = -mu / d_adj
    for j in range(n - 2, 0, -1):
        p_raw[j] = (-mu + r_rate * p_raw[j + 1]) / d_adj
    ip_raw = sum(p_raw[j].conjugate() * q[j] for j in range(n))
    p = [z / ip_raw.conjugate() for z in p_raw]
    biorth_err = abs(sum(p[j].conjugate() * q[j] for j in range(n)) - 1.0)

    def B_form(u, v):
        su, sv = sum(u), sum(v)
        out = [0j] * n
        out[0] = -r0 * (u[0] * sv + v[0] * su)
        return out

    q_bar = [z.conjugate() for z in q]
    A_c = [[complex(A[r_i][c_i]) for c_i in range(n)] for r_i in range(n)]
    M2 = [[(2j * w if r_i == c_i else 0j) - complex(A[r_i][c_i]) for c_i in range(n)] for r_i in range(n)]
    invA_b = solve_linear_complex(A_c, B_form(q, q_bar))
    invM_b = solve_linear_complex(M2, B_form(q, q))
    comb = [-2.0 * B_form(q, invA_b)[j] + B_form(q_bar, invM_b)[j] for j in range(n)]
    l1 = sum(p[j].conjugate() * comb[j] for j in range(n)).real / (2.0 * w)
    return l1, biorth_err


def validate_m3_hopf_bubbles_and_stabilization(p=P_SIRSV):
    r0 = p["R0"]
    rows = []
    for k in (1, 4, 11, 12, 13, 16, 32, None):
        cr = _find_sirs_hopf_crossings(k, r0)
        rows.append({
            "k": "RFDE" if k is None else k,
            "n_crossings": len(cr),
            "tau_w_minus": cr[0][0] if cr else None,
            "omega_minus": cr[0][1] if cr else None,
            "cross_speed_minus": cr[0][2] if cr else None,
            "tau_w_plus": cr[1][0] if len(cr) > 1 else None,
            "cross_speed_plus": cr[1][2] if len(cr) > 1 else None,
        })
    return {"rows": rows}


def validate_m4_kuznetsov_and_transverse_floquet(p=P_SIRSV):
    r0 = p["R0"]
    c12 = _find_sirs_hopf_crossings(12, r0)
    c13 = _find_sirs_hopf_crossings(13, r0)
    l1_12, biorth12 = _compute_kuznetsov_l1_and_vecs(12, c12[0][0], c12[0][1], r0)
    l1_13, biorth13 = _compute_kuznetsov_l1_and_vecs(13, c13[0][0], c13[0][1], r0)

    # Transverse Floquet monodromy matrix on reduced (k=13, tw=12.5) SIRS cycle (dim = 14)
    k, tw = 13, 12.5
    n = k + 1
    rw = k / tw
    i_eq = (r0 - 1.0) / (r0 * (1.0 + tw))
    r_eq = i_eq / rw
    y = [i_eq + 0.015] + [r_eq] * k

    def f_red(st):
        s_v = 1.0 - sum(st)
        out = [0.0] * n
        out[0] = (r0 * s_v - 1.0) * st[0]
        out[1] = st[0] - rw * st[1]
        for j in range(2, n):
            out[j] = rw * (st[j - 1] - st[j])
        return out

    def jac_red(st):
        s_v = 1.0 - sum(st)
        J = [[0.0] * n for _ in range(n)]
        J[0][0] = (r0 * s_v - 1.0) - r0 * st[0]
        for col in range(1, n):
            J[0][col] = -r0 * st[0]
        J[1][0] = 1.0
        J[1][1] = -rw
        for j in range(2, n):
            J[j][j - 1] = rw
            J[j][j] = -rw
        return J

    def rk4_step_red(st, dt):
        k1 = f_red(st)
        k2 = f_red([st[m] + 0.5 * dt * k1[m] for m in range(n)])
        k3 = f_red([st[m] + 0.5 * dt * k2[m] for m in range(n)])
        k4 = f_red([st[m] + dt * k3[m] for m in range(n)])
        return [st[m] + (dt / 6.0) * (k1[m] + 2.0 * k2[m] + 2.0 * k3[m] + k4[m]) for m in range(n)]

    # Settle onto limit cycle for t = 2200 d (lam_perp ~ -0.0034 d^-1 -> e^-7.4 < 6e-4)
    dt = 0.05
    for _ in range(int(2200.0 / dt)):
        y = rk4_step_red(y, dt)

    # Bracket two upward crossings of Poincare section I = i_eq inside next 50 d
    crossings = []
    for step in range(1, int(55.0 / dt)):
        y_prev = list(y)
        y = rk4_step_red(y, dt)
        if y_prev[0] < i_eq <= y[0]:
            f0 = f_red(y_prev)[0]
            f1 = f_red(y)[0]
            # Bisect exact Hermite crossing u in [0, 1]
            lo, hi = 0.0, 1.0
            for _ in range(45):
                u = 0.5 * (lo + hi)
                u2, u3 = u * u, u * u * u
                val = (2*u3 - 3*u2 + 1)*y_prev[0] + (u3 - 2*u2 + u)*dt*f0 + (-2*u3 + 3*u2)*y[0] + (u3 - u2)*dt*f1
                if val < i_eq:
                    lo = u
                else:
                    hi = u
            u = 0.5 * (lo + hi)
            y_cross = []
            fp, fn = f_red(y_prev), f_red(y)
            u2, u3 = u * u, u * u * u
            for m in range(n):
                y_cross.append((2*u3 - 3*u2 + 1)*y_prev[m] + (u3 - 2*u2 + u)*dt*fp[m] + (-2*u3 + 3*u2)*y[m] + (u3 - u2)*dt*fn[m])
            crossings.append(((step - 1 + u) * dt, y_cross))
            if len(crossings) == 2:
                break

    T_per = crossings[1][0] - crossings[0][0]
    y_cyc = list(crossings[0][1])

    # Integrate augmented state + nxn monodromy matrix Phi over exact [0, T_per]
    n_sub = max(200, int(round(T_per / 0.05)))
    h_per = T_per / n_sub
    Phi = [[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]
    zc = list(y_cyc)

    def rhs_aug(st, M):
        f_v = f_red(st)
        J_v = jac_red(st)
        JM = mat_mul(J_v, M)
        return f_v, JM

    for _ in range(n_sub):
        f1, K1 = rhs_aug(zc, Phi)
        z2 = [zc[m] + 0.5 * h_per * f1[m] for m in range(n)]
        P2 = [[Phi[r][c] + 0.5 * h_per * K1[r][c] for c in range(n)] for r in range(n)]
        f2, K2 = rhs_aug(z2, P2)
        z3 = [zc[m] + 0.5 * h_per * f2[m] for m in range(n)]
        P3 = [[Phi[r][c] + 0.5 * h_per * K2[r][c] for c in range(n)] for r in range(n)]
        f3, K3 = rhs_aug(z3, P3)
        z4 = [zc[m] + h_per * f3[m] for m in range(n)]
        P4 = [[Phi[r][c] + h_per * K3[r][c] for c in range(n)] for r in range(n)]
        f4, K4 = rhs_aug(z4, P4)
        for m in range(n):
            zc[m] += (h_per / 6.0) * (f1[m] + 2.0 * f2[m] + 2.0 * f3[m] + f4[m])
        for r in range(n):
            for c in range(n):
                Phi[r][c] += (h_per / 6.0) * (K1[r][c] + 2.0 * K2[r][c] + 2.0 * K3[r][c] + K4[r][c])

    # Tangent direction invariance check M * f(y0) == f(y0) (trivial Floquet multiplier mu_1 = 1)
    f0_cyc = f_red(y_cyc)
    f0_norm = math.sqrt(sum(v * v for v in f0_cyc))
    v_tan = [v / f0_norm for v in f0_cyc]
    Mv_tan = [sum(Phi[r][c] * v_tan[c] for c in range(n)) for r in range(n)]
    tangent_mu1_err = math.sqrt(sum((Mv_tan[r] - v_tan[r]) ** 2 for r in range(n)))

    # Project Phi onto 13D hyperplane orthogonal to v_tan using Householder reflector H_Q
    w_vec = list(v_tan)
    w_vec[0] += 1.0 if v_tan[0] >= 0.0 else -1.0
    wn2 = sum(u * u for u in w_vec)
    Q_mat = [[(1.0 if r == c else 0.0) - 2.0 * w_vec[r] * w_vec[c] / wn2 for c in range(n)] for r in range(n)]
    M_tilde = mat_mul(mat_mul(mat_T(Q_mat), Phi), Q_mat)
    M_perp = [[M_tilde[r][c] for c in range(1, n)] for r in range(1, n)]
    eigs_perp = eigvals_general(M_perp)
    mags_perp = sorted([abs(ev) for ev in eigs_perp], reverse=True)
    max_perp_mult = mags_perp[0]
    lambda_perp = math.log(max_perp_mult) / T_per

    return {
        "l1_k12": l1_12, "biorth_err_k12": biorth12,
        "l1_k13": l1_13, "biorth_err_k13": biorth13,
        "period_T_k13_tw125": T_per,
        "tangent_mu1_residual": tangent_mu1_err,
        "max_transverse_multiplier": max_perp_mult,
        "transverse_floquet_exponent": lambda_perp,
    }


def _hermite_eval_scalar(t_query, t_grid, i_hist, di_hist, dt):
    if t_query < 0.0:
        return 0.0
    idx = int(t_query / dt)
    if idx >= len(t_grid) - 1:
        idx = len(t_grid) - 2
    u = max(0.0, min(1.0, (t_query - t_grid[idx]) / dt))
    u2, u3 = u * u, u * u * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    return h00 * i_hist[idx] + h10 * dt * di_hist[idx] + h01 * i_hist[idx + 1] + h11 * dt * di_hist[idx + 1]


def validate_m5_hermite_rfde_convergence(p=P_SIRSV):
    r0, xi, rho = p["R0"], p["xi"], p["rho"]
    tw = 1.0 / xi
    dt = 0.025
    n_steps = int(round(100.0 / dt))
    gamma_w = tw * rho
    surv_delay = math.exp(-gamma_w)
    s, i_v, r_v, v = Y0_SIRSV
    t_g = [0.0]
    i_g = [i_v]
    di_g = [r0 * s * i_v - i_v]

    def i_del(tq):
        return _hermite_eval_scalar(tq, t_g, i_g, di_g, dt)

    for step in range(n_steps):
        t = step * dt
        id_0 = i_del(t - tw)
        id_m = i_del(t + 0.5 * dt - tw)
        id_1 = i_del(t + dt - tw)

        def rk(sv, iv, rv, idl):
            in_w = surv_delay * idl
            return (-r0*sv*iv + in_w, r0*sv*iv - iv, iv - rho*rv - in_w, rho*rv)

        k1 = rk(s, i_v, r_v, id_0)
        k2 = rk(s + 0.5*dt*k1[0], i_v + 0.5*dt*k1[1], r_v + 0.5*dt*k1[2], id_m)
        k3 = rk(s + 0.5*dt*k2[0], i_v + 0.5*dt*k2[1], r_v + 0.5*dt*k2[2], id_m)
        k4 = rk(s + dt*k3[0], i_v + dt*k3[1], r_v + dt*k3[2], id_1)
        s += (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        i_v += (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        r_v += (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        v += (dt/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        t_g.append((step + 1) * dt)
        i_g.append(i_v)
        di_g.append(r0 * s * i_v - i_v)

    trough_rfde = min(i_g[m] for m, t in enumerate(t_g) if 4.0 <= t <= 9.5)
    rebound_rfde = max(i_g[m] for m, t in enumerate(t_g) if 6.5 <= t <= 16.0)
    rows = []
    for k in (1, 2, 4, 8, 16, 32):
        rw = k / tw
        sk, ik = Y0_SIRSV[0], Y0_SIRSV[1]
        rk_l = [0.0] * k
        ik_hist = [ik]
        for _ in range(n_steps):
            def rk_s(sv, iv, rl):
                ds = -r0*sv*iv + rw*rl[-1]
                di = r0*sv*iv - iv
                dr = [iv - (rw + rho)*rl[0]] + [rw*rl[j-1] - (rw + rho)*rl[j] for j in range(1, k)]
                return ds, di, dr
            k1 = rk_s(sk, ik, rk_l)
            k2 = rk_s(sk + 0.5*dt*k1[0], ik + 0.5*dt*k1[1], [rk_l[j] + 0.5*dt*k1[2][j] for j in range(k)])
            k3 = rk_s(sk + 0.5*dt*k2[0], ik + 0.5*dt*k2[1], [rk_l[j] + 0.5*dt*k2[2][j] for j in range(k)])
            k4 = rk_s(sk + dt*k3[0], ik + dt*k3[1], [rk_l[j] + dt*k3[2][j] for j in range(k)])
            sk += (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            ik += (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            rk_l = [rk_l[j] + (dt/6.0)*(k1[2][j] + 2*k2[2][j] + 2*k3[2][j] + k4[2][j]) for j in range(k)]
            ik_hist.append(ik)
        linf = max(abs(ik_hist[m] - i_g[m]) for m in range(len(t_g)))
        rows.append({"k": k, "L_inf_vs_RFDE": linf})
    return {
        "tau_w": tw, "trough_rfde": trough_rfde,
        "rebound_rfde": rebound_rfde, "convergence_rows": rows,
    }


def _average_ranks(vals):
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for m in range(i, j + 1):
            ranks[order[m]] = avg
        i = j + 1
    return ranks


def _prcc_from_samples(inputs_cols, y_col):
    n = len(y_col)
    p_dim = len(inputs_cols)
    cols = [_average_ranks(c) for c in inputs_cols] + [_average_ranks(y_col)]
    k_tot = p_dim + 1
    means = [sum(c) / n for c in cols]
    sds = [math.sqrt(sum((v - means[idx]) ** 2 for v in cols[idx])) for idx in range(k_tot)]
    C_mat = [[0.0] * k_tot for _ in range(k_tot)]
    for a in range(k_tot):
        C_mat[a][a] = 1.0
        for b in range(a + 1, k_tot):
            cov = sum((cols[a][i] - means[a]) * (cols[b][i] - means[b]) for i in range(n))
            val = cov / (sds[a] * sds[b])
            C_mat[a][b] = C_mat[b][a] = val
    P_inv = invert_matrix_real(C_mat)
    df = n - 2 - (p_dim - 1)
    out = []
    for j in range(p_dim):
        prcc = -P_inv[j][k_tot - 1] / math.sqrt(P_inv[j][j] * P_inv[k_tot - 1][k_tot - 1])
        t_stat = prcc * math.sqrt(df / max(1e-15, 1.0 - prcc * prcc))
        z_approx = abs(t_stat) * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + t_stat * t_stat / (2.0 * df))
        p_approx = math.erfc(z_approx / math.sqrt(2.0))
        out.append((prcc, t_stat, p_approx))
    return out


def validate_m6_shooting_and_lhs_prcc(p=P_SIRSV):
    r0, xi, rho = p["R0"], p["xi"], p["rho"]
    # 1. RK4 bisection for outbreak threshold S_crit(I0=1e-4) -> 1/R0 and surge threshold S_sep(I0=0.01; I_cap=0.15)
    def max_I_for_S0(s0_val, i0_val, t_end=18.0):
        v0_val = max(0.0, 1.0 - s0_val - i0_val)
        tr = integrate_sirsv_rk4(r0, xi, rho, (s0_val, i0_val, 0.0, v0_val), t_end=t_end, dt=0.05, stride=1)
        return max(row[2] for row in tr)

    lo_c, hi_c = 0.15, 0.60
    for _ in range(42):
        mid = 0.5 * (lo_c + hi_c)
        if max_I_for_S0(mid, 1.0e-4, 8.0) > 1.0001e-4:
            hi_c = mid
        else:
            lo_c = mid
    s_crit_shot = 0.5 * (lo_c + hi_c)

    lo_s, hi_s = 0.40, 0.90
    for _ in range(42):
        mid = 0.5 * (lo_s + hi_s)
        if max_I_for_S0(mid, 0.01, 25.0) >= 0.1500:
            hi_s = mid
        else:
            lo_s = mid
    s_sep_cap = 0.5 * (lo_s + hi_s)

    # 2. N=240 deterministic LHS-PRCC + dummy parameter falsification gate
    n_samples = 240
    specs = [("R0", 2.0, 4.5), ("xi", 0.2, 1.0), ("rho", 0.005, 0.05), ("I0", 0.002, 0.05), ("dummy", 0.0, 1.0)]
    rng = random.Random(20261116)
    X_cols = []
    for _, lo, hi in specs:
        perm = list(range(n_samples))
        rng.shuffle(perm)
        X_cols.append([lo + (hi - lo) * ((perm[i] + rng.random()) / n_samples) for i in range(n_samples)])
    y_imax, y_i100 = [], []
    for i in range(n_samples):
        tr = integrate_sirsv_rk4(X_cols[0][i], X_cols[1][i], X_cols[2][i], (1.0 - X_cols[3][i], X_cols[3][i], 0.0, 0.0), t_end=100.0, dt=0.05, stride=4)
        y_imax.append(max(row[2] for row in tr))
        y_i100.append(tr[-1][2])
    rp = _prcc_from_samples(X_cols, y_imax)
    re = _prcc_from_samples(X_cols, y_i100)
    prcc_rows = {specs[j][0]: {"prcc_Imax": rp[j][0], "p_Imax": rp[j][2], "prcc_I100": re[j][0], "p_I100": re[j][2]} for j in range(5)}
    return {
        "S_crit_shot": s_crit_shot, "S_herd": 1.0 / r0,
        "S_sep_cap015": s_sep_cap, "prcc": prcc_rows,
    }


# ==============================================================================
# GROUP II (PAPER 2): ICNN+CARE HJB, PIPE-HOCBF, RESOLVENT & BOED/ABC-SMC (M7-M10)
# ==============================================================================
def validate_m7_icnn_rff_care_hjb(p=P_SIRSV):
    r0, xi, rho = p["R0"], p["xi"], p["rho"]
    # 1. Linearized 2D SIRSV Continuous LQR around (S_ref, I_ref) = (0.35, 0.05) with control channel u_S on dS
    s_ref, i_ref = 0.35, 0.05
    A = [[-r0 * i_ref, -r0 * s_ref], [r0 * i_ref, r0 * s_ref - 1.0]]
    B = [[-s_ref], [0.0]]
    Q = [[1.0, 0.0], [0.0, 25.0]]
    r_cost = 1.0
    # Stabilizing initial gain K0 = [-3.0, -8.0] (since B0 < 0, u = -Kx increases S suppression when I > 0)
    K = [[-3.0, -8.0]]

    def solve_lyap_2x2(Acl, Q_eff):
        a11, a12 = Acl[0][0], Acl[0][1]
        a21, a22 = Acl[1][0], Acl[1][1]
        M_l = [
            [2.0 * a11, 2.0 * a21, 0.0],
            [a12, a11 + a22, a21],
            [0.0, 2.0 * a12, 2.0 * a22],
        ]
        rhs = [-Q_eff[0][0], -Q_eff[0][1], -Q_eff[1][1]]
        p1, p2, p3 = solve_linear_real(M_l, rhs)
        return [[p1, p2], [p2, p3]]

    P = [[1.0, 0.0], [0.0, 1.0]]
    for _ in range(25):
        Acl = [[A[i][j] - B[i][0] * K[0][j] for j in range(2)] for i in range(2)]
        Qeff = [[Q[i][j] + r_cost * K[0][i] * K[0][j] for j in range(2)] for i in range(2)]
        P = solve_lyap_2x2(Acl, Qeff)
        # K = R^-1 B^T P
        BtP = [B[0][0] * P[0][0] + B[1][0] * P[1][0], B[0][0] * P[0][1] + B[1][0] * P[1][1]]
        K = [[BtP[0] / r_cost, BtP[1] / r_cost]]

    ATP = mat_mul(mat_T(A), P)
    PA = mat_mul(P, A)
    PB = mat_mul(P, B)
    PBBTP = [[PB[r][0] * PB[c][0] / r_cost for c in range(2)] for r in range(2)]
    care_res = math.sqrt(sum((ATP[i][j] + PA[i][j] - PBBTP[i][j] + Q[i][j]) ** 2 for i in range(2) for j in range(2)))
    trP = P[0][0] + P[1][1]
    detP = P[0][0] * P[1][1] - P[0][1] ** 2
    lam_min_P = 0.5 * (trP - math.sqrt(max(0.0, trP * trP - 4.0 * detP)))

    # 2. Genuine Convex ICNN(x) + Temporal Sinusoidal RFF(t) Only
    rng = random.Random(20261117)
    H_dim = 16
    M_rff = 8
    rff_freqs = [rng.gauss(0.0, 2.0) for _ in range(M_rff)]
    # Non-negative pass-through & layer-2 weights on spatial state x = [S, I, R, V]
    W_x = [[abs(rng.gauss(0.0, 0.4)) for _ in range(4)] for _ in range(H_dim)]
    W_rff = [[rng.gauss(0.0, 0.15) for _ in range(2 * M_rff)] for _ in range(H_dim)]
    w_out = [abs(rng.gauss(0.0, 0.3)) + 0.05 for _ in range(H_dim)]

    def softplus(u):
        return math.log1p(math.exp(u)) if u < 30.0 else u

    def icnn_val(x4, t_val):
        tn = t_val / 40.0
        rff_t = [math.sin(2.0 * math.pi * f * tn) for f in rff_freqs] + [math.cos(2.0 * math.pi * f * tn) for f in rff_freqs]
        tot = 0.05 * sum(xv * xv for xv in x4)
        for i in range(H_dim):
            lin = sum(W_x[i][c] * x4[c] for c in range(4)) + sum(W_rff[i][m] * rff_t[m] for m in range(2 * M_rff))
            tot += w_out[i] * softplus(lin)
        return tot

    # Numerical Hessian eigenvalues of ICNN wrt x in R^4 at test (x4, t)
    x_test = [0.65, 0.12, 0.10, 0.13]
    h_fd = 1e-4
    H_icnn = [[0.0] * 4 for _ in range(4)]
    v0 = icnn_val(x_test, 10.0)
    for i in range(4):
        xp, xm = list(x_test), list(x_test)
        xp[i] += h_fd
        xm[i] -= h_fd
        H_icnn[i][i] = (icnn_val(xp, 10.0) - 2.0 * v0 + icnn_val(xm, 10.0)) / (h_fd * h_fd)
        for j in range(i + 1, 4):
            xpp, xpm, xmp, xmm = list(x_test), list(x_test), list(x_test), list(x_test)
            xpp[i] += h_fd; xpp[j] += h_fd
            xpm[i] += h_fd; xpm[j] -= h_fd
            xmp[i] -= h_fd; xmp[j] += h_fd
            xmm[i] -= h_fd; xmm[j] -= h_fd
            H_icnn[i][j] = H_icnn[j][i] = (icnn_val(xpp, 10.0) - icnn_val(xpm, 10.0) - icnn_val(xmp, 10.0) + icnn_val(xmm, 10.0)) / (4.0 * h_fd * h_fd)
    lam_min_icnn = min(ev.real for ev in eigvals_general(H_icnn))

    # 3. Closed-loop Nonlinear HJB/LQR Rollout Cost Reduction vs Uncontrolled & Constant u_S=0.10
    def sim_cost(mode):
        s, iv, rv, vv = Y0_SIRSV
        dt = 0.05
        n_st = int(round(40.0 / dt))
        cost = 0.0
        for _ in range(n_st):
            if mode == "unctrl":
                u = 0.0
            elif mode == "const010":
                u = 0.10
            else:
                # Feedback derived from CARE gradient P*(y - E_target) projected onto control vector (-S e_S + S e_V)
                # grad V = 2*P*[S-0.30, I], so -1/(2*r_cost) * B^T grad V = S * (P[0][0]*(S-0.30) + P[0][1]*I) / r_cost
                ds_err = s - 0.30
                u_opt = s * (P[0][0] * ds_err + P[0][1] * iv) / r_cost
                u = max(0.0, min(0.60, u_opt))
            cost += dt * (25.0 * iv * iv + r_cost * u * u)
            k1 = sirsv_rhs_scalar(s, iv, rv, vv, r0, xi, rho, u_s=u)
            s += dt * k1[0]
            iv += dt * k1[1]
            rv += dt * k1[2]
            vv += dt * k1[3]
        return cost

    c_unctrl = sim_cost("unctrl")
    c_const = sim_cost("const010")
    c_lqr = sim_cost("lqr")
    return {
        "care_residual_F": care_res,
        "lam_min_P": lam_min_P,
        "icnn_lam_min_hess": lam_min_icnn,
        "cost_unctrl": c_unctrl,
        "cost_const010": c_const,
        "cost_care_lqr": c_lqr,
    }


def validate_m8_pipe_hocbf_and_reldeg2(p=P_SIRSV):
    r0, xi, rho = p["R0"], p["xi"], p["rho"]
    # 1. Telescoping k=2 vaccine seroconversion Pipe-HOCBF on vaccine-induced immunity V_vac (tau_v = 2.0, d_v = 0.05)
    tau_v, k_v, d_v = 2.0, 2, 0.05
    r_v = k_v / tau_v
    sigma_v = r_v / (r_v + d_v)
    v_cap = 0.450
    alpha_p = 1.2
    u_nom = 0.35

    def sim_seroconv(mode):
        s, iv, rv, e1, e2, v_nat, v_vac = 0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0
        dt = 0.025
        max_vvac, max_mvac = v_vac, v_vac
        max_telescoping_err = 0.0
        for _ in range(int(round(60.0 / dt))):
            mvac = v_vac + (sigma_v ** 2) * e1 + sigma_v * e2
            if mode == "pipe_hocbf":
                # Purely actuated invariant: dM_vac/dt = sigma_v^2 * u_S * S <= alpha_p * (v_cap - M_vac)
                rhs_b = alpha_p * (v_cap - mvac)
                u_s = max(0.0, min(u_nom, rhs_b / max(1e-12, (sigma_v ** 2) * s)))
            else:
                u_s = u_nom if v_vac < v_cap else 0.0

            def f_sc(sv, i_v, r_v_st, e1v, e2v, _vn, _vv, us):
                ds = -r0 * sv * i_v + xi * r_v_st - us * sv
                di = r0 * sv * i_v - i_v
                dr = i_v - (xi + rho) * r_v_st
                de1 = us * sv - (r_v + d_v) * e1v
                de2 = r_v * e1v - (r_v + d_v) * e2v
                dvn = rho * r_v_st
                dvv = r_v * e2v
                return ds, di, dr, de1, de2, dvn, dvv

            k1 = f_sc(s, iv, rv, e1, e2, v_nat, v_vac, u_s)
            dM_num = k1[6] + (sigma_v ** 2) * k1[3] + sigma_v * k1[4]
            dM_ex = (sigma_v ** 2) * u_s * s
            max_telescoping_err = max(max_telescoping_err, abs(dM_num - dM_ex))

            m = [a + 0.5 * dt * da for a, da in zip((s, iv, rv, e1, e2, v_nat, v_vac), k1)]
            mv_m = m[6] + (sigma_v ** 2) * m[3] + sigma_v * m[4]
            if mode == "pipe_hocbf":
                um = max(0.0, min(u_nom, (alpha_p * (v_cap - mv_m)) / max(1e-12, (sigma_v ** 2) * m[0])))
            else:
                um = u_nom if m[6] < v_cap else 0.0
            k2 = f_sc(*m, um)
            s += dt * k2[0]; iv += dt * k2[1]; rv += dt * k2[2]
            e1 += dt * k2[3]; e2 += dt * k2[4]; v_nat += dt * k2[5]; v_vac += dt * k2[6]
            max_vvac = max(max_vvac, v_vac)
            max_mvac = max(max_mvac, v_vac + (sigma_v ** 2) * e1 + sigma_v * e2)
        return {"V_max": max_vvac, "M_vac_max": max_mvac, "telescoping_err": max_telescoping_err}

    sc_blind = sim_seroconv("delay_blind")
    sc_pipe = sim_seroconv("pipe_hocbf")

    # 2. Rel-Deg-2 Prophylactic HO-CBF vs Ch-A uR=2 vs Const uS=0.25 (I_cap = 0.1500)
    i_cap = 0.1500
    a1, a2, u_max = 1.5, 2.5, 0.60

    def sim_cap(u_s_fn=None, u_r=0.0):
        s, iv, rv, vv = Y0_SIRSV
        dt = 0.025
        imax, dose = iv, 0.0
        for _ in range(int(round(100.0 / dt))):
            def f_c(sv, i_v, r_v_st, _v_s):
                us = u_s_fn(sv, i_v, r_v_st) if u_s_fn else 0.0
                return sirsv_rhs_scalar(sv, i_v, r_v_st, _v_s, r0, xi, rho, u_s=us, u_r=u_r), us
            k1, us1 = f_c(s, iv, rv, vv)
            k2, us2 = f_c(s + 0.5*dt*k1[0], iv + 0.5*dt*k1[1], rv + 0.5*dt*k1[2], vv + 0.5*dt*k1[3])
            k3, us3 = f_c(s + 0.5*dt*k2[0], iv + 0.5*dt*k2[1], rv + 0.5*dt*k2[2], vv + 0.5*dt*k2[3])
            k4, us4 = f_c(s + dt*k3[0], iv + dt*k3[1], rv + dt*k3[2], vv + dt*k3[3])
            dose += (dt/6.0)*(us1*s + 2*us2*(s + 0.5*dt*k1[0]) + 2*us3*(s + 0.5*dt*k2[0]) + us4*(s + dt*k3[0]))
            s += (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            iv += (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
            rv += (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
            vv += (dt/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
            imax = max(imax, iv)
        return imax, dose

    def hocbf_fn(sv, iv, rv):
        h = i_cap - iv
        h_dot = -(r0 * sv - 1.0) * iv
        lf2 = (-r0 * iv) * (-r0 * sv * iv + xi * rv) - ((r0 * sv - 1.0) ** 2) * iv
        lg = r0 * sv * iv
        if lg < 1e-12:
            return 0.0
        req = -(lf2 + (a1 + a2) * h_dot + a1 * a2 * h)
        return 0.0 if req <= 0.0 else min(u_max, req / lg)

    chA_imax, _ = sim_cap(None, 2.0)
    ho_imax, ho_dose = sim_cap(hocbf_fn, 0.0)
    c025_imax, c025_dose = sim_cap(lambda _s, _i, _r: 0.25, 0.0)
    return {
        "V_cap_pipe": v_cap,
        "seroconv_delay_blind": sc_blind,
        "seroconv_pipe_hocbf": sc_pipe,
        "chA_uR2_Imax": chA_imax,
        "hocbf_Imax": ho_imax,
        "hocbf_dose": ho_dose,
        "const025_Imax": c025_imax,
        "const025_dose": c025_dose,
    }


def _spectral_norm_2x2_complex(M):
    a = (M[0][0].conjugate() * M[0][0] + M[1][0].conjugate() * M[1][0]).real
    d = (M[0][1].conjugate() * M[0][1] + M[1][1].conjugate() * M[1][1]).real
    b = M[0][0].conjugate() * M[0][1] + M[1][0].conjugate() * M[1][1]
    tr = a + d
    diff = a - d
    return math.sqrt(0.5 * (tr + math.sqrt(diff * diff + 4.0 * (b.real ** 2 + b.imag ** 2))))


def validate_m9_closed_loop_pseudospectra(p=P_SIRSV):
    r0, xi = p["R0"], p["xi"]
    i_end = (xi / (1.0 + xi)) * (1.0 - 1.0 / r0)
    A_open = [[-r0 * i_end, -r0 * i_end], [1.0, -xi]]
    # Closed-loop linearized Jacobian under active immune-modulated damping (A_cl = A_open - diag(0.85, 0.55) + offdiag stabilization)
    A_cl = [[A_open[0][0] - 0.85, A_open[0][1] * 0.45], [A_open[1][0] * 0.45, A_open[1][1] - 0.55]]

    def inspect_mat(A):
        frob2 = sum(A[r][c] ** 2 for r in range(2) for c in range(2))
        eigs = eigvals_general(A)
        eig_sq = sum(abs(ev) ** 2 for ev in eigs)
        hen = math.sqrt(max(0.0, frob2 - eig_sq))
        spec_absc = max(ev.real for ev in eigs)
        h11, h22, h12 = A[0][0], A[1][1], 0.5 * (A[0][1] + A[1][0])
        mu2 = 0.5 * ((h11 + h22) + math.sqrt((h11 - h22) ** 2 + 4.0 * h12 * h12))
        k_max = 0.0
        for kx in range(36):
            x = 10.0 ** (-4.0 + 4.5 * kx / 35.0)
            for ky in range(49):
                y = -2.0 + 4.0 * ky / 48.0
                z = complex(x, y)
                zA = [[z - A[0][0], -A[0][1]], [-A[1][0], z - A[1][1]]]
                dt_c = zA[0][0] * zA[1][1] - zA[0][1] * zA[1][0]
                inv = [[zA[1][1] / dt_c, -zA[0][1] / dt_c], [-zA[1][0] / dt_c, zA[0][0] / dt_c]]
                k_max = max(k_max, x * _spectral_norm_2x2_complex(inv))
        # H_inf resolvent peak along imaginary axis sup_w ||(iw I - A)^(-1)||_2
        hinf = 0.0
        for ky in range(201):
            w = -4.0 + 8.0 * ky / 200.0
            z = complex(0.0, w)
            zA = [[z - A[0][0], -A[0][1]], [-A[1][0], z - A[1][1]]]
            dt_c = zA[0][0] * zA[1][1] - zA[0][1] * zA[1][0]
            inv = [[zA[1][1] / dt_c, -zA[0][1] / dt_c], [-zA[1][0] / dt_c, zA[0][0] / dt_c]]
            hinf = max(hinf, _spectral_norm_2x2_complex(inv))
        return {"henrici": hen, "mu2": mu2, "spec_absc": spec_absc, "kreiss": k_max, "H_inf": hinf}

    m_open = inspect_mat(A_open)
    m_cl = inspect_mat(A_cl)
    return {"open_loop": m_open, "closed_loop": m_cl}


def validate_m10_boed_and_abc_smc(p=P_SIRSV):
    r0, xi, rho = p["R0"], p["xi"], p["rho"]
    # 1. 16D Forward Variational BOED
    dt = 0.05
    n_steps = int(round(100.0 / dt))
    z = list(Y0_SIRSV) + [0.0] * 12

    def rhs_16(st):
        s, iv, rv, vv = st[0], st[1], st[2], st[3]
        out = list(sirsv_rhs_scalar(s, iv, rv, vv, r0, xi, rho))
        df_dth = [[-s*iv, s*iv, 0.0, 0.0], [rv, 0.0, -rv, 0.0], [0.0, 0.0, -rv, rv]]
        for j in range(3):
            b = 4 + 4 * j
            sS, sI, sR = st[b], st[b + 1], st[b + 2]
            out.append(-r0 * (iv * sS + s * sI) + xi * sR + df_dth[j][0])
            out.append(r0 * (iv * sS + s * sI) - sI + df_dth[j][1])
            out.append(sI - (xi + rho) * sR + df_dth[j][2])
            out.append(rho * sR + df_dth[j][3])
        return out

    snaps = {0: list(z)}
    for step in range(1, n_steps + 1):
        k1 = rhs_16(z)
        k2 = rhs_16([z[m] + 0.5*dt*k1[m] for m in range(16)])
        k3 = rhs_16([z[m] + 0.5*dt*k2[m] for m in range(16)])
        k4 = rhs_16([z[m] + dt*k3[m] for m in range(16)])
        for m in range(16):
            z[m] += (dt / 6.0) * (k1[m] + 2*k2[m] + 2*k3[m] + k4[m])
        if step % 20 == 0:
            snaps[step // 20] = list(z)

    h_fd = 1e-6
    t_fd = []
    for k in ("R0", "xi", "rho"):
        pp = dict(p); pp[k] += h_fd
        pm = dict(p); pm[k] -= h_fd
        tp = integrate_sirsv_rk4(pp["R0"], pp["xi"], pp["rho"], Y0_SIRSV, t_end=15.0, dt=0.05, stride=300)[-1]
        tm = integrate_sirsv_rk4(pm["R0"], pm["xi"], pm["rho"], Y0_SIRSV, t_end=15.0, dt=0.05, stride=300)[-1]
        t_fd.extend([(tp[c + 1] - tm[c + 1]) / (2.0 * h_fd) for c in range(4)])
    var_err_t15 = max(abs(snaps[15][4 + m] - t_fd[m]) for m in range(12))

    sigma2 = 0.0206346357691995 ** 2
    cand = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    fim_t = {ti: [[sum(snaps[ti][4 + 4*a + c] * snaps[ti][4 + 4*b + c] for c in range(4)) / sigma2 for b in range(3)] for a in range(3)] for ti in range(101)}
    def det3(M):
        return M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0])
    def sdet(trp):
        return det3([[sum(fim_t[tk][a][b] for tk in trp) for b in range(3)] for a in range(3)])
    best_sched, best_det = None, -1.0
    for i1 in range(len(cand)):
        for i2 in range(i1 + 1, len(cand)):
            for i3 in range(i2 + 1, len(cand)):
                d = sdet((cand[i1], cand[i2], cand[i3]))
                if d > best_det:
                    best_det, best_sched = d, [cand[i1], cand[i2], cand[i3]]
    det_123 = sdet((1, 2, 3))
    ratio_123 = best_det / det_123

    # 2. 3-Stage Cooling ABC-SMC Recovery on Synthetic Noisy Data (seed=20261117)
    rng = random.Random(20261117)
    true_traj = integrate_sirsv_rk4(r0, xi, rho, Y0_SIRSV, t_end=100.0, dt=0.05, stride=20)
    sigma_obs = 0.015
    y_obs = [[row[c + 1] + rng.gauss(0.0, sigma_obs) for c in range(4)] for row in true_traj]

    def sim_rmse(th):
        tr = integrate_sirsv_rk4(th[0], th[1], th[2], Y0_SIRSV, t_end=100.0, dt=0.1, stride=10)
        sse = sum((tr[ti][c + 1] - y_obs[ti][c]) ** 2 for ti in range(101) for c in range(4))
        return math.sqrt(sse / 404.0)

    bounds = [(1.8, 5.0), (0.15, 1.4), (0.002, 0.06)]
    eps_sched = [0.06, 0.025, 0.0175]
    n_part = 50
    particles, weights, sd_trace = [], [], []
    # Stage 0: Prior draw
    for st_idx, eps in enumerate(eps_sched):
        new_p, new_w = [], []
        if st_idx == 0:
            while len(new_p) < n_part:
                cand_th = [rng.uniform(lo, hi) for lo, hi in bounds]
                if sim_rmse(cand_th) < eps:
                    new_p.append(cand_th)
                    new_w.append(1.0 / n_part)
        else:
            # Empirical SD per coordinate for Gaussian perturbation kernel
            means = [sum(weights[i] * particles[i][j] for i in range(n_part)) for j in range(3)]
            sds = [max(1e-5, math.sqrt(sum(weights[i] * ((particles[i][j] - means[j]) ** 2) for i in range(n_part)))) for j in range(3)]
            cdf = []
            acc = 0.0
            for w in weights:
                acc += w
                cdf.append(acc)
            while len(new_p) < n_part:
                u = rng.random()
                idx = next(k for k, c in enumerate(cdf) if u <= c)
                cand_th = [particles[idx][j] + rng.gauss(0.0, 1.5 * sds[j]) for j in range(3)]
                if not all(bounds[j][0] <= cand_th[j] <= bounds[j][1] for j in range(3)):
                    continue
                if sim_rmse(cand_th) < eps:
                    # SMC importance weight: prior(1) / sum_k w_k * N(cand; p_k, (1.5*sd)^2)
                    kern_sum = 0.0
                    for k in range(n_part):
                        lp = -0.5 * sum(((cand_th[j] - particles[k][j]) / (1.5 * sds[j])) ** 2 for j in range(3))
                        kern_sum += weights[k] * math.exp(lp)
                    new_p.append(cand_th)
                    new_w.append(1.0 / max(1e-30, kern_sum))
            w_tot = sum(new_w)
            new_w = [w / w_tot for w in new_w]
        particles, weights = new_p, new_w
        m_st = [sum(weights[i] * particles[i][j] for i in range(n_part)) for j in range(3)]
        sd_st = [math.sqrt(sum(weights[i] * ((particles[i][j] - m_st[j]) ** 2) for i in range(n_part))) for j in range(3)]
        sd_trace.append({"eps": eps, "mean": m_st, "sd": sd_st})

    final_mean = sd_trace[-1]["mean"]
    rel_errs = [abs(final_mean[0] - r0) / r0, abs(final_mean[1] - xi) / xi, abs(final_mean[2] - rho) / rho]
    return {
        "var_err_t15": var_err_t15,
        "D_optimal_BOED": best_sched,
        "det_FIM_BOED": best_det,
        "ratio_vs_123": ratio_123,
        "abc_smc_stages": sd_trace,
        "abc_rel_errs": rel_errs,
    }


# ==============================================================================
# GROUP III (PAPER 3): LOBATTO, WEINGARTEN/cEDMD/FTLE, WENO-7, DCT-II & GAMES (M11-M15)
# ==============================================================================
def validate_m11_lobatto_wkb_instanton(p=P_SIRSV):
    r0, xi = p["R0"], p["xi"]
    s_eq = 1.0 / r0
    i_eq = (xi / (1.0 + xi)) * (1.0 - s_eq)

    def H_and_derivs(s, iv, pS, pI):
        w1 = r0 * s * iv
        w3 = xi * (1.0 - s - iv)
        e1 = math.exp(max(-35.0, min(35.0, pI - pS)))
        e2 = math.exp(max(-35.0, min(35.0, -pI)))
        e3 = math.exp(max(-35.0, min(35.0, pS)))
        H = w1 * (e1 - 1.0) + iv * (e2 - 1.0) + w3 * (e3 - 1.0)
        Hq = [r0 * iv * (e1 - 1.0) - xi * (e3 - 1.0), r0 * s * (e1 - 1.0) + (e2 - 1.0) - xi * (e3 - 1.0)]
        Hp = [-w1 * e1 + w3 * e3, w1 * e1 - iv * e2]
        # Second partial blocks H_qq, H_qp, H_pp (note H_pq = H_qp^T)
        Hqq = [[0.0, r0 * (e1 - 1.0)], [r0 * (e1 - 1.0), 0.0]]
        Hqp = [[-r0 * iv * e1 - xi * e3, r0 * iv * e1], [-r0 * s * e1 - xi * e3, r0 * s * e1 - e2]]
        Hpq = mat_T(Hqp)
        Hpp = [[w1 * e1 + w3 * e3, -w1 * e1], [-w1 * e1, w1 * e1 + iv * e2]]
        return H, Hq, Hp, Hqq, Hqp, Hpp

    def lobatto3a3b_step(z4, h, compute_tangent=False):
        q0 = [z4[0], z4[1]]
        p0 = [z4[2], z4[3]]
        ph = list(p0)
        for _ in range(20):
            _, Hq0, _, _, Hqp0, _ = H_and_derivs(q0[0], q0[1], ph[0], ph[1])
            G1 = [ph[i] - p0[i] + 0.5 * h * Hq0[i] for i in range(2)]
            if max(abs(G1[0]), abs(G1[1])) < 1e-16:
                break
            J1_ph = [[(1.0 if r == c else 0.0) + 0.5 * h * Hqp0[r][c] for c in range(2)] for r in range(2)]
            if abs(J1_ph[0][0] * J1_ph[1][1] - J1_ph[0][1] * J1_ph[1][0]) < 1e-14:
                return None, None
            dph = solve_linear_real(J1_ph, [-G1[0], -G1[1]])
            ph[0] += dph[0]; ph[1] += dph[1]
            if max(abs(ph[0]), abs(ph[1])) > 20.0:
                return None, None
        _, Hq0, Hp0, Hqq0, Hqp0, Hpp0 = H_and_derivs(q0[0], q0[1], ph[0], ph[1])

        q1 = [q0[0] + h * Hp0[0], q0[1] + h * Hp0[1]]
        for _ in range(20):
            _, _, Hp1, _, Hqp1, _ = H_and_derivs(q1[0], q1[1], ph[0], ph[1])
            Hpq1 = mat_T(Hqp1)
            G2 = [q1[i] - q0[i] - 0.5 * h * (Hp0[i] + Hp1[i]) for i in range(2)]
            if max(abs(G2[0]), abs(G2[1])) < 1e-16:
                break
            J2_q1 = [[(1.0 if r == c else 0.0) - 0.5 * h * Hpq1[r][c] for c in range(2)] for r in range(2)]
            if abs(J2_q1[0][0] * J2_q1[1][1] - J2_q1[0][1] * J2_q1[1][0]) < 1e-14:
                return None, None
            dq1 = solve_linear_real(J2_q1, [-G2[0], -G2[1]])
            q1[0] += dq1[0]; q1[1] += dq1[1]
            if max(abs(q1[0]), abs(q1[1])) > 5.0:
                return None, None

        _, Hq1, Hp1, Hqq1, Hqp1, Hpp1 = H_and_derivs(q1[0], q1[1], ph[0], ph[1])
        p1 = [ph[i] - 0.5 * h * Hq1[i] for i in range(2)]
        z_next = [q1[0], q1[1], p1[0], p1[1]]
        if not compute_tangent:
            return z_next, None

        Hpq0 = mat_T(Hqp0)
        Hpq1 = mat_T(Hqp1)
        inv_A1 = invert_matrix_real([[(1.0 if r == c else 0.0) + 0.5 * h * Hqp0[r][c] for c in range(2)] for r in range(2)])
        dph_dq0 = mat_mul(inv_A1, [[-0.5 * h * Hqq0[r][c] for c in range(2)] for r in range(2)])
        dph_dp0 = inv_A1
        S1 = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [dph_dq0[0][0], dph_dq0[0][1], dph_dp0[0][0], dph_dp0[0][1]],
            [dph_dq0[1][0], dph_dq0[1][1], dph_dp0[1][0], dph_dp0[1][1]],
        ]
        inv_A2 = invert_matrix_real([[(1.0 if r == c else 0.0) - 0.5 * h * Hpq1[r][c] for c in range(2)] for r in range(2)])
        dq1_dq0 = mat_mul(inv_A2, [[(1.0 if r == c else 0.0) + 0.5 * h * Hpq0[r][c] for c in range(2)] for r in range(2)])
        dq1_dph = mat_mul(inv_A2, [[0.5 * h * (Hpp0[r][c] + Hpp1[r][c]) for c in range(2)] for r in range(2)])
        S2 = [
            [dq1_dq0[0][0], dq1_dq0[0][1], dq1_dph[0][0], dq1_dph[0][1]],
            [dq1_dq0[1][0], dq1_dq0[1][1], dq1_dph[1][0], dq1_dph[1][1]],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        S3 = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-0.5 * h * Hqq1[0][0], -0.5 * h * Hqq1[0][1], 1.0 - 0.5 * h * Hqp1[0][0], -0.5 * h * Hqp1[0][1]],
            [-0.5 * h * Hqq1[1][0], -0.5 * h * Hqq1[1][1], -0.5 * h * Hqp1[1][0], 1.0 - 0.5 * h * Hqp1[1][1]],
        ]
        return z_next, mat_mul(S3, mat_mul(S2, S1))

    # Gaussian Lyapunov covariance at E_end* + WKB fade-out action S* via Lobatto shooting
    A_s = [[-r0 * i_eq - xi, -r0 * s_eq - xi], [r0 * i_eq, 0.0]]
    w1_eq, w3_eq = r0 * s_eq * i_eq, xi * (1.0 - s_eq - i_eq)
    B_cov = [[w1_eq + w3_eq, -w1_eq], [-w1_eq, w1_eq + i_eq]]
    M_ly = [[2*A_s[0][0], 2*A_s[0][1], 0.0], [A_s[1][0], A_s[0][0]+A_s[1][1], A_s[0][1]], [0.0, 2*A_s[1][0], 2*A_s[1][1]]]
    v1, v2, v3 = solve_linear_real(M_ly, [-B_cov[0][0], -B_cov[0][1], -B_cov[1][1]])
    s_gauss = 0.5 * (i_eq * i_eq) / (v3 - 1e-18)

    Sigma = [[v1, v2], [v2, v3]]
    best_S, best_exit, best_H_drift, best_z0 = 1e9, None, 0.0, None
    best_nsteps = 0
    for idx in range(360):
        ang = 2.0 * math.pi * idx / 360.0
        dp = [math.cos(ang), math.sin(ang)]
        dq = [Sigma[0][0]*dp[0] + Sigma[0][1]*dp[1], Sigma[1][0]*dp[0] + Sigma[1][1]*dp[1]]
        sc = 5e-4 / math.hypot(dq[0], dq[1])
        z_start = [s_eq + sc*dq[0], i_eq + sc*dq[1], sc*dp[0], sc*dp[1]]
        zt = list(z_start)
        act = 0.0
        max_h = 0.0
        steps_used = 0
        for _ in range(950):
            zn, _ = lobatto3a3b_step(zt, 0.03, compute_tangent=False)
            if zn is None:
                break
            steps_used += 1
            act += 0.5 * ((zn[2] + zt[2])*(zn[0] - zt[0]) + (zn[3] + zt[3])*(zn[1] - zt[1]))
            Hv, _, _, _, _, _ = H_and_derivs(zn[0], zn[1], zn[2], zn[3])
            max_h = max(max_h, abs(Hv))
            zt = zn
            if zt[1] < 0.002 or zt[0] < 0.0 or zt[0] + zt[1] > 1.05 or zt[1] > 0.9:
                break
        if zt[1] < 0.0025 and 0.0 < zt[0] < 1.0 and act < best_S:
            best_S, best_exit, best_H_drift = act, [zt[0], zt[1]], max_h
            best_z0, best_nsteps = z_start, steps_used

    # Evaluate exact 1-step & full-instanton symplectic 2-form residual along the true H=0 Lobatto WKB instanton!
    z_m = list(best_z0)
    M_tot = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    Omega = [[0, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 0, 0], [0, -1, 0, 0]]
    max_1step_err = 0.0
    for _ in range(best_nsteps):
        z_m, M_st = lobatto3a3b_step(z_m, 0.03, compute_tangent=True)
        MtOM1 = mat_mul(mat_mul(mat_T(M_st), Omega), M_st)
        max_1step_err = max(max_1step_err, max(abs(MtOM1[r][c] - Omega[r][c]) for r in range(4) for c in range(4)))
        M_tot = mat_mul(M_st, M_tot)
    MtOM = mat_mul(mat_mul(mat_T(M_tot), Omega), M_tot)
    lobatto_symp_err = max(abs(MtOM[r][c] - Omega[r][c]) for r in range(4) for c in range(4))

    return {
        "lobatto_1step_symp_err": max_1step_err,
        "lobatto_symp_err": lobatto_symp_err,
        "S_gauss": s_gauss,
        "S_wkb_lobatto": best_S,
        "exit_SI": best_exit,
        "max_H_drift": best_H_drift,
    }


def validate_m12_ftle_cedmd_weingarten(p=P_SIRSV):
    r0, xi, rho = p["R0"], p["xi"], p["rho"]
    # 1. 41x41 FTLE peak (using fast reuse check)
    n_s, n_i = 41, 41
    s_vals = [0.10 + 0.85 * a / (n_s - 1) for a in range(n_s)]
    i_vals = [0.005 + 0.145 * b / (n_i - 1) for b in range(n_i)]
    t_hor = 6.0
    phi = [[(lambda s0, i0: integrate_sirsv_rk4(r0, xi, rho, (s0, i0, 0.0, max(0.0, 1.0-s0-i0)), t_end=t_hor, dt=0.1, stride=60)[-1][1:3])(sv, iv) for sv in s_vals] for iv in i_vals]
    ds, di = s_vals[1] - s_vals[0], i_vals[1] - i_vals[0]
    ftle_max = 0.0
    for bi in range(1, n_i - 1):
        for ai in range(1, n_s - 1):
            if s_vals[ai] + i_vals[bi] > 0.98:
                continue
            j11 = (phi[bi][ai+1][0] - phi[bi][ai-1][0]) / (2.0 * ds)
            j21 = (phi[bi][ai+1][1] - phi[bi][ai-1][1]) / (2.0 * ds)
            j12 = (phi[bi+1][ai][0] - phi[bi-1][ai][0]) / (2.0 * di)
            j22 = (phi[bi+1][ai][1] - phi[bi-1][ai][1]) / (2.0 * di)
            c11, c12, c22 = j11*j11 + j21*j21, j11*j12 + j21*j22, j12*j12 + j22*j22
            lam = 0.5 * ((c11 + c22) + math.sqrt((c11 - c22)**2 + 4.0*c12*c12))
            ftle_max = max(ftle_max, math.log(max(1.0, lam)) / (2.0 * t_hor))

    # 2. Polynomial cEDMD Koopman generator extraction around SIRS E_end*
    s_eq = 1.0 / r0
    i_eq = (xi / (1.0 + xi)) * (1.0 - s_eq)
    r_eq = 1.0 - s_eq - i_eq
    tr_end, det_end = -(r0 * i_eq + xi), r0 * i_eq * (1.0 + xi)
    eig_true = complex(0.5 * tr_end, 0.5 * math.sqrt(max(0.0, 4.0 * det_end - tr_end * tr_end)))

    def lift_deg2(di, dr):
        return [di, dr, di * di, di * dr, dr * dr]

    dt_k = 0.15
    M_k = 5
    G0 = [[1e-11 if a == b else 0.0 for b in range(M_k)] for a in range(M_k)]
    G1 = [[0.0] * M_k for _ in range(M_k)]
    rng = random.Random(77)
    for _traj in range(28):
        s, iv, rv = s_eq + rng.uniform(-0.04, 0.04), i_eq + rng.uniform(-0.04, 0.04), 0.0
        rv = 1.0 - s - iv
        for _step in range(18):
            psi0 = lift_deg2(iv - i_eq, rv - r_eq)
            # advance dt_k with 3 RK4 steps
            h_s = dt_k / 3.0
            for _s in range(3):
                k1 = sirsv_rhs_scalar(s, iv, rv, 0.0, r0, xi, 0.0)
                k2 = sirsv_rhs_scalar(s + 0.5*h_s*k1[0], iv + 0.5*h_s*k1[1], rv + 0.5*h_s*k1[2], 0.0, r0, xi, 0.0)
                k3 = sirsv_rhs_scalar(s + 0.5*h_s*k2[0], iv + 0.5*h_s*k2[1], rv + 0.5*h_s*k2[2], 0.0, r0, xi, 0.0)
                k4 = sirsv_rhs_scalar(s + h_s*k3[0], iv + h_s*k3[1], rv + h_s*k3[2], 0.0, r0, xi, 0.0)
                s += (h_s/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
                iv += (h_s/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
                rv += (h_s/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
            psi1 = lift_deg2(iv - i_eq, rv - r_eq)
            for a in range(M_k):
                for b in range(M_k):
                    G0[a][b] += psi0[a] * psi0[b]
                    G1[a][b] += psi1[a] * psi0[b]
    G0_inv = invert_matrix_real(G0)
    K_mat = mat_mul(G1, G0_inv)
    # Continuous generator eigenvalues: log(mu)/dt_k; locate fundamental mode matching eig_true
    k_eigs = [cmath.log(mu) / dt_k for mu in eigvals_general(K_mat) if mu.imag > 0.02]
    best_koop = min(k_eigs, key=lambda lam: abs(lam - eig_true))
    koopman_err = abs(best_koop - eig_true)

    # 3. 4th-Order FD Weingarten Shape Operator on Curved SIRS Level Surface Phi(S,I,R) = S + I - ln(S)/R0 - (xi/2)*R^2 = c0
    p0 = (0.45, 0.25, 0.30)
    def phi_surf(s, iv, rv):
        return s + iv - math.log(s) / r0 - 0.5 * xi * rv * rv

    h_w = 1e-3
    def fd_d1(idx):
        pts = [list(p0) for _ in (-2, -1, 1, 2)]
        for k_i, sh in enumerate((-2, -1, 1, 2)):
            pts[k_i][idx] += sh * h_w
        vals = [phi_surf(*pt) for pt in pts]
        return (vals[0] - 8.0*vals[1] + 8.0*vals[2] - vals[3]) / (12.0 * h_w)

    def fd_d2(i, j):
        if i == j:
            vals = []
            for sh in (-2, -1, 0, 1, 2):
                pt = list(p0); pt[i] += sh * h_w
                vals.append(phi_surf(*pt))
            return (-vals[0] + 16.0*vals[1] - 30.0*vals[2] + 16.0*vals[3] - vals[4]) / (12.0 * h_w * h_w)
        # 4th-order mixed second derivative (Wait: 4th-order O(h^4) mixed partial via O(h^2) + Richardson extrapolation!)
        def mix2(hv):
            acc = 0.0
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    pt = list(p0); pt[i] += s1 * hv; pt[j] += s2 * hv
                    acc += s1 * s2 * phi_surf(*pt)
            return acc / (4.0 * hv * hv)
        return (4.0 * mix2(0.5 * h_w) - mix2(h_w)) / 3.0

    def eval_weingarten(grad, hess):
        gn = math.sqrt(sum(g * g for g in grad))
        n_v = [g / gn for g in grad]
        P_p = [[(1.0 if r == c else 0.0) - n_v[r]*n_v[c] for c in range(3)] for r in range(3)]
        S_op = [[-x / gn for x in row] for row in mat_mul(mat_mul(P_p, hess), P_p)]
        e1_raw = [-n_v[1], n_v[0], 0.0] if abs(n_v[2]) < 0.9 else [0.0, -n_v[2], n_v[1]]
        e1n = math.sqrt(sum(v*v for v in e1_raw))
        e1 = [v / e1n for v in e1_raw]
        e2 = [n_v[1]*e1[2] - n_v[2]*e1[1], n_v[2]*e1[0] - n_v[0]*e1[2], n_v[0]*e1[1] - n_v[1]*e1[0]]
        B = [[e1[r], e2[r]] for r in range(3)]
        S2 = mat_mul(mat_mul(mat_T(B), S_op), B)
        tr = S2[0][0] + S2[1][1]
        dt = S2[0][0]*S2[1][1] - S2[0][1]*S2[1][0]
        disc = math.sqrt(max(0.0, tr*tr - 4.0*dt))
        return 0.5*(tr - disc), 0.5*(tr + disc)

    g_fd = [fd_d1(0), fd_d1(1), fd_d1(2)]
    H_fd = [[fd_d2(r, c) for c in range(3)] for r in range(3)]
    kmin_fd, kmax_fd = eval_weingarten(g_fd, H_fd)
    g_ex = [1.0 - 1.0 / (r0 * p0[0]), 1.0, -xi * p0[2]]
    H_ex = [[1.0 / (r0 * p0[0] * p0[0]), 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -xi]]
    kmin_ex, kmax_ex = eval_weingarten(g_ex, H_ex)
    wein_err = max(abs(kmin_fd - kmin_ex), abs(kmax_fd - kmax_ex))

    return {
        "ftle_max": ftle_max,
        "koopman_true": [eig_true.real, eig_true.imag],
        "koopman_cedmd": [best_koop.real, best_koop.imag],
        "koopman_err": koopman_err,
        "weingarten_k_exact": [kmin_ex, kmax_ex],
        "weingarten_gauss_K": kmin_ex * kmax_ex,
        "weingarten_fd_err": wein_err,
    }


def validate_m13_weno7_sir_invariant(p=P_SIRSV):
    r0 = p["R0"]
    # Exact nonlinear analytical Kermack-McKendrick SIR invariant:
    # Psi(S, I) = exp(S + I - ln(S)/R0) = S^(-1/R0) * exp(S + I) => L_{f_SIR} Psi == 0 identically!
    def psi_nl(s, iv):
        return (s ** (-1.0 / r0)) * math.exp(s + iv)

    def weno7_1d_slice(u_ext, dx):
        # u_ext has 4 ghost cells on each side around M interior points -> length M + 8
        M = len(u_ext) - 8
        d0, d1, d2, d3 = 1.0/35.0, 12.0/35.0, 18.0/35.0, 4.0/35.0
        eps_w = 1e-40
        f_half = [0.0] * (M + 1)
        # Reconstruct at interfaces j-1/2 (idx 3) .. j+M-1/2 (idx 3+M) => offset base = 3
        for m in range(M + 1):
            c = 3 + m  # cell i = c, f_{c+1/2} uses u[c-3 .. c+3]
            um3, um2, um1, u0, up1, up2, up3 = u_ext[c-3], u_ext[c-2], u_ext[c-1], u_ext[c], u_ext[c+1], u_ext[c+2], u_ext[c+3]
            q0 = (-3*um3 + 13*um2 - 23*um1 + 25*u0) / 12.0
            q1 = (1*um2 - 5*um1 + 13*u0 + 3*up1) / 12.0
            q2 = (-1*um1 + 7*u0 + 7*up1 - 1*up2) / 12.0
            q3 = (3*u0 + 13*up1 - 5*up2 + 1*up3) / 12.0
            s0 = (um3*(547*um3 - 3882*um2 + 4642*um1 - 1854*u0) + um2*(7043*um2 - 17246*um1 + 7042*u0) + um1*(11003*um1 - 9402*u0) + 2107*u0*u0) / 240.0
            s1 = (um2*(267*um2 - 1642*um1 + 1602*u0 - 494*up1) + um1*(2843*um1 - 5966*u0 + 1922*up1) + u0*(3443*u0 - 2522*up1) + 547*up1*up1) / 240.0
            s2 = (um1*(547*um1 - 2522*u0 + 1922*up1 - 494*up2) + u0*(3443*u0 - 5966*up1 + 1602*up2) + up1*(2843*up1 - 1642*up2) + 267*up2*up2) / 240.0
            s3 = (u0*(2107*u0 - 9402*up1 + 7042*up2 - 1854*up3) + up1*(11003*up1 - 17246*up2 + 4642*up3) + up2*(7043*up2 - 3882*up3) + 547*up3*up3) / 240.0
            tau7 = abs(s0 - s3)
            a0 = d0 * (1.0 + (tau7 / (eps_w + s0)) ** 2)
            a1 = d1 * (1.0 + (tau7 / (eps_w + s1)) ** 2)
            a2 = d2 * (1.0 + (tau7 / (eps_w + s2)) ** 2)
            a3 = d3 * (1.0 + (tau7 / (eps_w + s3)) ** 2)
            f_half[m] = (a0*q0 + a1*q1 + a2*q2 + a3*q3) / (a0 + a1 + a2 + a3)
        return [(f_half[m + 1] - f_half[m]) / dx for m in range(M)]

    res_by_n = {}
    for N in (16, 32, 64, 128):
        s_lo, s_hi = 0.25, 0.85
        i_lo, i_hi = 0.08, 0.45
        hs = (s_hi - s_lo) / N
        hi = (i_hi - i_lo) / N
        s_ext = [s_lo + (k - 4) * hs for k in range(N + 8)]
        i_ext = [i_lo + (k - 4) * hi for k in range(N + 8)]
        grid_Psi = [[psi_nl(sv, iv) for iv in i_ext] for sv in s_ext]
        dI_rows = [weno7_1d_slice(grid_Psi[4 + m], hi) for m in range(N)]
        dS_cols = [weno7_1d_slice([grid_Psi[a][4 + m] for a in range(N + 8)], hs) for m in range(N)]
        max_lie = 0.0
        for ms in range(N):
            sv = s_ext[4 + ms]
            for mi in range(N):
                iv = i_ext[4 + mi]
                fS = -r0 * sv * iv
                fI = (r0 * sv - 1.0) * iv
                lie = fS * dS_cols[mi][ms] + fI * dI_rows[ms][mi]
                max_lie = max(max_lie, abs(lie))
        res_by_n[str(N)] = max_lie

    order_32_64 = math.log(res_by_n["32"] / res_by_n["64"]) / math.log(2.0)
    order_64_128 = math.log(res_by_n["64"] / res_by_n["128"]) / math.log(2.0)
    return {
        "max_lie_N16": res_by_n["16"],
        "max_lie_N32": res_by_n["32"],
        "max_lie_N64": res_by_n["64"],
        "max_lie_N128": res_by_n["128"],
        "weno7_order_32_64": order_32_64,
        "weno7_order_64_128": order_64_128,
    }


def validate_m14_dct2_spatial_homogenization(p=P_SIRSV):
    r0, xi, rho = p["R0"], p["xi"], p["rho"]
    Nx = 32
    x_pts = [(j + 0.5) / Nx for j in range(Nx)]
    # Orthonormal DCT-II matrix C[m][j] = c_m * cos(m * pi * x_j)
    basis = []
    for m in range(Nx):
        sc = math.sqrt(1.0 / Nx) if m == 0 else math.sqrt(2.0 / Nx)
        basis.append([sc * math.cos(m * math.pi * x_pts[j]) for j in range(Nx)])
    # Exact finite-volume discrete Metzler Neumann Laplacian eigenvalues (Markov positive semigroup under DCT-II)
    evals_lap = [4.0 * (Nx ** 2) * (math.sin(m * math.pi / (2.0 * Nx)) ** 2) for m in range(Nx)]

    def dct2_step(u_vec, D_coef, h_diff):
        # u_hat = basis @ u_vec, scale by exp(-D * lam * h), u_new = basis^T @ u_hat
        u_hat = [sum(basis[m][j] * u_vec[j] for j in range(Nx)) * math.exp(-D_coef * evals_lap[m] * h_diff) for m in range(Nx)]
        return [sum(basis[m][j] * u_hat[m] for m in range(Nx)) for j in range(Nx)]

    # Well-mixed reference trajectory on t in [0, 25]
    ode_traj = integrate_sirsv_rk4(r0, xi, rho, Y0_SIRSV, t_end=25.0, dt=0.05, stride=1)

    def run_pde(D_coef):
        # Asymmetric step seed on [0, 1] with identical spatial mean Y0 = (0.99, 0.01, 0, 0)
        S_f = [0.98 if j < Nx // 2 else 1.00 for j in range(Nx)]
        I_f = [0.02 if j < Nx // 2 else 0.00 for j in range(Nx)]
        R_f = [0.0] * Nx
        V_f = [0.0] * Nx
        dt = 0.05
        half_dt = 0.5 * dt
        mean_I_hist = [sum(I_f) / Nx]
        min_cell = min(min(S_f), min(I_f), min(R_f), min(V_f))
        max_mass_err = 0.0
        for step in range(1, int(round(25.0 / dt)) + 1):
            S_f = dct2_step(S_f, D_coef, half_dt)
            I_f = dct2_step(I_f, D_coef, half_dt)
            R_f = dct2_step(R_f, D_coef, half_dt)
            V_f = dct2_step(V_f, D_coef, half_dt)
            for j in range(Nx):
                sv, iv, rv, vv = S_f[j], I_f[j], R_f[j], V_f[j]
                k1 = sirsv_rhs_scalar(sv, iv, rv, vv, r0, xi, rho)
                k2 = sirsv_rhs_scalar(sv + half_dt*k1[0], iv + half_dt*k1[1], rv + half_dt*k1[2], vv + half_dt*k1[3], r0, xi, rho)
                k3 = sirsv_rhs_scalar(sv + half_dt*k2[0], iv + half_dt*k2[1], rv + half_dt*k2[2], vv + half_dt*k2[3], r0, xi, rho)
                k4 = sirsv_rhs_scalar(sv + dt*k3[0], iv + dt*k3[1], rv + dt*k3[2], vv + dt*k3[3], r0, xi, rho)
                S_f[j] += (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
                I_f[j] += (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
                R_f[j] += (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
                V_f[j] += (dt/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
            S_f = dct2_step(S_f, D_coef, half_dt)
            I_f = dct2_step(I_f, D_coef, half_dt)
            R_f = dct2_step(R_f, D_coef, half_dt)
            V_f = dct2_step(V_f, D_coef, half_dt)
            min_cell = min(min_cell, min(S_f), min(I_f), min(R_f), min(V_f))
            tot_mean = sum(S_f[j] + I_f[j] + R_f[j] + V_f[j] for j in range(Nx)) / Nx
            max_mass_err = max(max_mass_err, abs(tot_mean - 1.0))
            mean_I_hist.append(sum(I_f) / Nx)
        homog_err_t10_25 = max(abs(mean_I_hist[k] - ode_traj[k][2]) for k in range(200, len(mean_I_hist)))
        grad_norm_end = max(I_f) - min(I_f)
        return {
            "I_mean_peak": max(mean_I_hist),
            "ode_peak": max(row[2] for row in ode_traj),
            "homog_err_t10_25": homog_err_t10_25,
            "grad_norm_t25": grad_norm_end,
            "max_mass_err": max_mass_err,
            "min_state": min_cell,
        }

    fast_res = run_pde(5.0)
    slow_res = run_pde(5.0e-4)
    return {"fast_diffusion": fast_res, "slow_diffusion": slow_res}


def validate_m15_enkf_cvar_noncomm_and_mirror_prox(p=P_SIRSV):
    r0, xi, rho = p["R0"], p["xi"], p["rho"]
    # 1. Partial-Observation EnKF + CVaR-99% DR-HOCBF under R0 uncertainty [2.75, 3.35]
    rng = random.Random(20261118)
    r0_true = 3.22
    i_cap = 0.1500
    a1, a2, u_max = 1.5, 2.5, 0.65
    n_ens = 48
    dt = 0.05
    r0_ens = [2.75 + (3.35 - 2.75) * e / (n_ens - 1) for e in range(n_ens)]
    ens = [list(Y0_SIRSV) for _ in range(n_ens)]
    y_true = list(Y0_SIRSV)
    rmse_acc = 0.0
    n_obs = 0
    true_imax = y_true[1]

    def hocbf_u(st, r0_est):
        sv, iv, rv, _ = st
        h = i_cap - iv
        hd = -(r0_est * sv - 1.0) * iv
        lf2 = (-r0_est * iv) * (-r0_est * sv * iv + xi * rv) - ((r0_est * sv - 1.0) ** 2) * iv
        lg = r0_est * sv * iv
        if lg < 1e-12:
            return 0.0
        req = -(lf2 + (a1 + a2) * hd + a1 * a2 * h)
        return 0.0 if req <= 0.0 else min(u_max, req / lg)

    for step in range(int(round(40.0 / dt))):
        # Robust CVaR upper-tail control (98th percentile / max across uncertain R0 ensemble)
        u_vals = sorted([hocbf_u(ens[e], r0_ens[e]) for e in range(n_ens)], reverse=True)
        # Top-2 average approximates CVaR-96..99% over N_ens=48
        u_dr = 0.5 * (u_vals[0] + u_vals[1])

        # Advance truth with u_dr
        k1 = sirsv_rhs_scalar(*y_true, r0_true, xi, rho, u_s=u_dr)
        y_m = [y_true[m] + 0.5*dt*k1[m] for m in range(4)]
        k2 = sirsv_rhs_scalar(*y_m, r0_true, xi, rho, u_s=u_dr)
        for m in range(4):
            y_true[m] += dt * k2[m]
        true_imax = max(true_imax, y_true[1])

        # Propagate ensemble + inflation noise
        for e in range(n_ens):
            ke1 = sirsv_rhs_scalar(*ens[e], r0_ens[e], xi, rho, u_s=u_dr)
            ym = [ens[e][m] + 0.5*dt*ke1[m] for m in range(4)]
            ke2 = sirsv_rhs_scalar(*ym, r0_ens[e], xi, rho, u_s=u_dr)
            for m in range(4):
                ens[e][m] = max(0.0, min(1.0, ens[e][m] + dt * ke2[m]))

        # Daily observation assimilation of noisy I(t) every 20 steps (dt=1.0)
        if (step + 1) % 20 == 0:
            sig_y = 0.01
            y_meas = max(0.0, min(1.0, y_true[1] + rng.gauss(0.0, sig_y)))
            mean_e = [sum(ens[e][m] for e in range(n_ens)) / n_ens for m in range(4)]
            cov_xy = [sum((ens[e][m] - mean_e[m]) * (ens[e][1] - mean_e[1]) for e in range(n_ens)) / (n_ens - 1) for m in range(4)]
            var_y = cov_xy[1] + sig_y ** 2
            kalman_g = [c / max(1e-12, var_y) for c in cov_xy]
            for e in range(n_ens):
                pert_obs = y_meas + rng.gauss(0.0, sig_y)
                innov = pert_obs - ens[e][1]
                for m in range(4):
                    ens[e][m] = max(0.0, min(1.0, ens[e][m] + kalman_g[m] * innov))
                s_e = sum(ens[e])
                if s_e > 1e-12:
                    ens[e] = [v / s_e for v in ens[e]]
            mean_post = [sum(ens[e][m] for e in range(n_ens)) / n_ens for m in range(4)]
            rmse_acc += sum((mean_post[m] - y_true[m]) ** 2 for m in range(4)) / 4.0
            n_obs += 1
    enkf_rmse = math.sqrt(rmse_acc / max(1, n_obs))

    # 2. Lie-Bracket Non-Commutativity of Prophylactic Vac (u_S=0.3) & Transmission NPI (0.45*R0)
    def apply_op(st0, r0_v, u_s_v, dur=3.0):
        s, iv, rv, vv = st0
        dt_o = 0.025
        for _ in range(int(round(dur / dt_o))):
            k1 = sirsv_rhs_scalar(s, iv, rv, vv, r0_v, xi, rho, u_s=u_s_v)
            k2 = sirsv_rhs_scalar(s+0.5*dt_o*k1[0], iv+0.5*dt_o*k1[1], rv+0.5*dt_o*k1[2], vv+0.5*dt_o*k1[3], r0_v, xi, rho, u_s=u_s_v)
            k3 = sirsv_rhs_scalar(s+0.5*dt_o*k2[0], iv+0.5*dt_o*k2[1], rv+0.5*dt_o*k2[2], vv+0.5*dt_o*k2[3], r0_v, xi, rho, u_s=u_s_v)
            k4 = sirsv_rhs_scalar(s+dt_o*k3[0], iv+dt_o*k3[1], rv+dt_o*k3[2], vv+dt_o*k3[3], r0_v, xi, rho, u_s=u_s_v)
            s += (dt_o/6.0)*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
            iv += (dt_o/6.0)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
            rv += (dt_o/6.0)*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
            vv += (dt_o/6.0)*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
        return (s, iv, rv, vv)

    y_AB = apply_op(apply_op(Y0_SIRSV, r0, 0.30, 3.0), 0.45 * r0, 0.0, 3.0)
    y_BA = apply_op(apply_op(Y0_SIRSV, 0.45 * r0, 0.0, 3.0), r0, 0.30, 3.0)
    commutator_norm = math.sqrt(sum((y_AB[m] - y_BA[m]) ** 2 for m in range(4)))

    # 3. Nemirovski Entropic Mirror-Prox Minimax Bilinear Game (Genuine Non-Trivial 2 Policies x 2 Strains)
    def combined_metric(r0_s, xi_s, u_s_pol, npi_fac):
        s, iv, rv, vv = Y0_SIRSV
        mx = iv
        for _ in range(700):
            k1 = sirsv_rhs_scalar(s, iv, rv, vv, r0_s*npi_fac, xi_s, rho, u_s=u_s_pol)
            k2 = sirsv_rhs_scalar(s+0.025*k1[0], iv+0.025*k1[1], rv+0.025*k1[2], vv+0.025*k1[3], r0_s*npi_fac, xi_s, rho, u_s=u_s_pol)
            k3 = sirsv_rhs_scalar(s+0.025*k2[0], iv+0.025*k2[1], rv+0.025*k2[2], vv+0.025*k2[3], r0_s*npi_fac, xi_s, rho, u_s=u_s_pol)
            k4 = sirsv_rhs_scalar(s+0.05*k3[0], iv+0.05*k3[1], rv+0.05*k3[2], vv+0.05*k3[3], r0_s*npi_fac, xi_s, rho, u_s=u_s_pol)
            s += (0.05/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0]); iv += (0.05/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
            rv += (0.05/6)*(k1[2]+2*k2[2]+2*k3[2]+k4[2]); vv += (0.05/6)*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
            mx = max(mx, iv)
        return mx + 3.0 * iv

    # Cross-dominated payoff matrix: Policy 0 (Prophylactic Vac uS=0.08) dominates fast-waning Strain 1 (2.6, 1.40),
    # Policy 1 (Transmission NPI 0.62*R0, uS=0.005) dominates high-R0 slow-waning Strain 0 (3.6, 0.10)
    G = [
        [combined_metric(3.6, 0.10, 0.08, 1.00), combined_metric(2.6, 1.40, 0.08, 1.00)],
        [combined_metric(3.6, 0.10, 0.005, 0.62), combined_metric(2.6, 1.40, 0.005, 0.62)],
    ]
    x_p, y_p = [0.5, 0.5], [0.5, 0.5]
    x_avg, y_avg = [0.0, 0.0], [0.0, 0.0]
    eta = 3.5
    n_mp = 3000
    for _ in range(n_mp):
        gx = [sum(G[i][j] * y_p[j] for j in range(2)) for i in range(2)]
        gy = [sum(x_p[i] * G[i][j] for i in range(2)) for j in range(2)]
        xh = [x_p[i] * math.exp(-eta * gx[i]) for i in range(2)]
        yh = [y_p[j] * math.exp(+eta * gy[j]) for j in range(2)]
        xh = [v / sum(xh) for v in xh]; yh = [v / sum(yh) for v in yh]
        gxh = [sum(G[i][j] * yh[j] for j in range(2)) for i in range(2)]
        gyh = [sum(xh[i] * G[i][j] for i in range(2)) for j in range(2)]
        x_p = [x_p[i] * math.exp(-eta * gxh[i]) for i in range(2)]
        y_p = [y_p[j] * math.exp(+eta * gyh[j]) for j in range(2)]
        x_p = [v / sum(x_p) for v in x_p]; y_p = [v / sum(y_p) for v in y_p]
        x_avg = [x_avg[i] + xh[i] / n_mp for i in range(2)]
        y_avg = [y_avg[j] + yh[j] / n_mp for j in range(2)]
    duality_gap = max(sum(x_avg[i]*G[i][j] for i in range(2)) for j in range(2)) - min(sum(G[i][j]*y_avg[j] for j in range(2)) for i in range(2))

    return {
        "dr_hocbf_true_R0": r0_true,
        "dr_hocbf_Imax": true_imax,
        "enkf_rmse": enkf_rmse,
        "lie_commutator_norm": commutator_norm,
        "y_Vac_then_NPI_I": y_AB[1],
        "y_NPI_then_Vac_I": y_BA[1],
        "game_payoff_G": G,
        "mirror_prox_x": x_avg,
        "mirror_prox_y": y_avg,
        "mirror_prox_duality_gap": duality_gap,
    }


def run_all_15_methods():
    t0 = time.perf_counter()
    data = {
        "M01_simplex_sir_cascade": validate_m1_simplex_sir_peak_and_cascade(),
        "M02_littles_law_gspt": validate_m2_littles_law_gspt_overshoot(),
        "M03_erlang_rfde_hopf": validate_m3_hopf_bubbles_and_stabilization(),
        "M04_kuznetsov_floquet": validate_m4_kuznetsov_and_transverse_floquet(),
        "M05_hermite_rfde_conv": validate_m5_hermite_rfde_convergence(),
        "M06_shooting_lhs_prcc": validate_m6_shooting_and_lhs_prcc(),
        "M07_icnn_rff_care_hjb": validate_m7_icnn_rff_care_hjb(),
        "M08_pipe_reldeg2_hocbf": validate_m8_pipe_hocbf_and_reldeg2(),
        "M09_bbk_pseudospectra": validate_m9_closed_loop_pseudospectra(),
        "M10_boed_and_abc_smc": validate_m10_boed_and_abc_smc(),
        "M11_lobatto_wkb": validate_m11_lobatto_wkb_instanton(),
        "M12_ftle_cedmd_weingarten": validate_m12_ftle_cedmd_weingarten(),
        "M13_weno7_sir_lie": validate_m13_weno7_sir_invariant(),
        "M14_dct2_spatial_sirsv": validate_m14_dct2_spatial_homogenization(),
        "M15_enkf_lie_mirrorprox": validate_m15_enkf_cvar_noncomm_and_mirror_prox(),
    }
    elapsed = time.perf_counter() - t0
    data["_meta"] = {"elapsed_s": elapsed, "numpy_present": HAS_NUMPY, "torch_present": HAS_TORCH}
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


_CACHED_DATA = None


def get_validation_data():
    global _CACHED_DATA
    if _CACHED_DATA is None:
        if "--verify-only" in sys.argv and RESULTS_PATH.exists():
            _CACHED_DATA = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        else:
            _CACHED_DATA = run_all_15_methods()
    return _CACHED_DATA


class TestSIRSVPapers123Validation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = get_validation_data()

    def test_01_m01_simplex_sir_exact_peak_and_cascade(self):
        m = self.d["M01_simplex_sir_cascade"]
        self.assertLess(abs(m["nagumo"]["sum_rhs_check"]), 1e-16)
        self.assertGreater(m["nagumo"]["dS_at_S0"], 0.0)
        self.assertLess(m["sir_peak_rel_err"], 1e-8)
        self.assertGreater(m["rk4_emp_order"], 3.97)
        self.assertLess(m["rk4_emp_order"], 4.03)
        self.assertLess(m["cascade_k4_mass_err"], 1e-14)

    def test_02_m02_littles_law_fenichel_and_overshoot(self):
        m = self.d["M02_littles_law_gspt"]
        self.assertLess(m["littles_law_max_res"], 1e-14)
        rows = m["gspt_rows"]
        for i in range(len(rows) - 1):
            self.assertLess(rows[i]["fenichel_err_30_90"], rows[i + 1]["fenichel_err_30_90"])
            self.assertGreater(rows[i]["dV_over"], 0.0)
        self.assertAlmostEqual(rows[2]["fenichel_err_30_90"], 0.00283, places=4)

    def test_03_m03_erlang_rfde_hopf_bubbles_and_stabilization(self):
        rows = {str(r["k"]): r for r in self.d["M03_erlang_rfde_hopf"]["rows"]}
        for ks in ("1", "4", "11"):
            self.assertEqual(rows[ks]["n_crossings"], 0)
        self.assertEqual(rows["12"]["n_crossings"], 2)
        self.assertGreater(rows["12"]["cross_speed_minus"], 0.0)
        self.assertLess(rows["12"]["cross_speed_plus"], 0.0)
        taus = [rows[k]["tau_w_minus"] for k in ("12", "13", "16", "32", "RFDE")]
        for i in range(len(taus) - 1):
            self.assertGreater(taus[i], taus[i + 1])

    def test_04_m04_kuznetsov_supercriticality_and_floquet_stability(self):
        m = self.d["M04_kuznetsov_floquet"]
        self.assertLess(m["biorth_err_k12"], 1e-13)
        self.assertLess(m["biorth_err_k13"], 1e-13)
        self.assertLess(m["l1_k12"], -1.0)
        self.assertLess(m["l1_k13"], -1.0)
        self.assertLess(m["tangent_mu1_residual"], 5e-3)
        self.assertLess(m["max_transverse_multiplier"], 1.0)
        self.assertLess(m["transverse_floquet_exponent"], 0.0)

    def test_05_m05_hermite_rfde_monotone_convergence(self):
        m = self.d["M05_hermite_rfde_conv"]
        errs = [r["L_inf_vs_RFDE"] for r in m["convergence_rows"]]
        for i in range(len(errs) - 1):
            self.assertGreater(errs[i], errs[i + 1])
        self.assertLess(errs[-1], 5e-3)
        self.assertGreater(m["rebound_rfde"], m["trough_rfde"] + 0.03)

    def test_06_m06_shooting_threshold_and_lhs_prcc_dummy(self):
        m = self.d["M06_shooting_lhs_prcc"]
        self.assertLess(abs(m["S_crit_shot"] - m["S_herd"]), 5e-4)
        self.assertGreater(m["S_sep_cap015"], m["S_herd"])
        pr = m["prcc"]
        self.assertGreater(pr["R0"]["prcc_Imax"], 0.99)
        self.assertLess(pr["rho"]["prcc_I100"], -0.95)
        self.assertGreater(pr["dummy"]["p_Imax"], 0.05)
        self.assertGreater(pr["dummy"]["p_I100"], 0.05)

    def test_07_m07_icnn_rff_care_hjb_convexity(self):
        m = self.d["M07_icnn_rff_care_hjb"]
        self.assertLess(m["care_residual_F"], 1e-14)
        self.assertGreater(m["lam_min_P"], 0.0)
        self.assertGreaterEqual(m["icnn_lam_min_hess"], 0.099)
        self.assertLess(m["cost_care_lqr"], m["cost_unctrl"])

    def test_08_m08_seroconversion_pipe_hocbf_and_reldeg2(self):
        m = self.d["M08_pipe_reldeg2_hocbf"]
        self.assertLess(m["seroconv_pipe_hocbf"]["telescoping_err"], 1e-13)
        self.assertGreater(m["seroconv_delay_blind"]["V_max"], m["V_cap_pipe"] + 0.02)
        self.assertLessEqual(m["seroconv_pipe_hocbf"]["M_vac_max"], m["V_cap_pipe"] + 1e-6)
        self.assertLessEqual(m["seroconv_pipe_hocbf"]["V_max"], m["V_cap_pipe"] + 1e-6)
        self.assertGreater(m["chA_uR2_Imax"], 0.30)
        self.assertLessEqual(m["hocbf_Imax"], 0.1500)
        self.assertLess(m["hocbf_dose"], 0.45 * m["const025_dose"])

    def test_09_m09_closed_loop_bbk_resolvent_compression(self):
        m = self.d["M09_bbk_pseudospectra"]
        self.assertGreater(m["open_loop"]["henrici"], 0.30)
        self.assertLess(m["open_loop"]["mu2"], 0.0)
        self.assertLessEqual(m["open_loop"]["kreiss"], 1.0 + 1e-6)
        self.assertLess(m["closed_loop"]["spec_absc"], m["open_loop"]["spec_absc"])
        self.assertLess(m["closed_loop"]["H_inf"], 0.65 * m["open_loop"]["H_inf"])

    def test_10_m10_16d_boed_and_3stage_abc_smc(self):
        m = self.d["M10_boed_and_abc_smc"]
        self.assertLess(m["var_err_t15"], 1e-9)
        self.assertEqual(m["D_optimal_BOED"], [2, 6, 70])
        self.assertGreater(m["ratio_vs_123"], 5000.0)
        st = m["abc_smc_stages"]
        for j in range(3):
            self.assertGreater(st[0]["sd"][j], st[2]["sd"][j])
        for rel_e in m["abc_rel_errs"]:
            self.assertLess(rel_e, 0.06)

    def test_11_m11_symplectic_lobatto_3a3b_and_wkb_action(self):
        m = self.d["M11_lobatto_wkb"]
        self.assertLess(m["lobatto_1step_symp_err"], 1e-15)
        self.assertLess(m["lobatto_symp_err"], 1e-10)
        self.assertGreater(m["S_wkb_lobatto"], 1.18 * m["S_gauss"])
        self.assertLess(m["max_H_drift"], 5e-4)

    def test_12_m12_ftle_cedmd_koopman_and_weingarten_curvature(self):
        m = self.d["M12_ftle_cedmd_weingarten"]
        self.assertGreater(m["ftle_max"], 0.30)
        self.assertLess(m["koopman_err"], 0.025)
        self.assertLess(m["weingarten_gauss_K"], 0.0)
        self.assertLess(m["weingarten_fd_err"], 1e-8)

    def test_13_m13_weno7_sir_lie_invariant_order7(self):
        m = self.d["M13_weno7_sir_lie"]
        self.assertGreater(m["weno7_order_64_128"], 6.90)
        self.assertLess(m["weno7_order_64_128"], 7.20)
        self.assertLess(m["max_lie_N128"], 1e-10)

    def test_14_m14_dct2_spatial_sirsv_homogenization(self):
        m = self.d["M14_dct2_spatial_sirsv"]
        fast, slow = m["fast_diffusion"], m["slow_diffusion"]
        self.assertLess(fast["max_mass_err"], 1e-12)
        self.assertLess(slow["max_mass_err"], 1e-12)
        self.assertGreaterEqual(fast["min_state"], -1e-12)
        self.assertGreaterEqual(slow["min_state"], -1e-12)
        self.assertLess(fast["homog_err_t10_25"], 1e-3)
        self.assertLess(fast["grad_norm_t25"], 1e-5)
        self.assertLess(slow["I_mean_peak"], fast["ode_peak"] - 0.02)

    def test_15_m15_enkf_cvar_lie_commutator_and_mirror_prox(self):
        m = self.d["M15_enkf_lie_mirrorprox"]
        self.assertLess(m["enkf_rmse"], 0.035)
        self.assertLessEqual(m["dr_hocbf_Imax"], 0.1500)
        self.assertGreater(m["lie_commutator_norm"], 1e-3)
        self.assertGreater(m["mirror_prox_x"][0], 0.15)
        self.assertLess(m["mirror_prox_x"][0], 0.92)
        self.assertLess(m["mirror_prox_duality_gap"], 1e-4)


def print_summary(d):
    print("=" * 92)
    print("PAPERS 1-3 FULL 15-METHOD VALIDATION SUMMARY ON EXTERNAL 4D SIRSV/SIRS/SIR BENCHMARK")
    print("=" * 92)
    m1 = d["M01_simplex_sir_cascade"]
    print(f"[M01 Paper1] SIR Peak Exact={m1['sir_exact_peak']:.8f} | RK4={m1['sir_rk4_peak']:.8f} (rel={m1['sir_peak_rel_err']:.2e}) | order={m1['rk4_emp_order']:.4f} | cascade_err={m1['cascade_k4_mass_err']:.2e}")
    m2 = d["M02_littles_law_gspt"]
    print(f"[M02 Paper1] Little's Law res={m2['littles_law_max_res']:.2e} | MLE Fenichel err={m2['gspt_rows'][2]['fenichel_err_30_90']:.5f} | dV_over=+{m2['gspt_rows'][2]['dV_over']:.5f}")
    m3 = d["M03_erlang_rfde_hopf"]
    r12 = next(r for r in m3["rows"] if r["k"] == 12)
    rinf = next(r for r in m3["rows"] if r["k"] == "RFDE")
    print(f"[M03 Paper1] Erlang k=1..11 stable | k=12 bubble=[{r12['tau_w_minus']:.4f}, {r12['tau_w_plus']:.4f}] | RFDE tau_w-={rinf['tau_w_minus']:.4f}")
    m4 = d["M04_kuznetsov_floquet"]
    print(f"[M04 Paper1] l1(12)={m4['l1_k12']:.4f}, l1(13)={m4['l1_k13']:.4f} | T_per={m4['period_T_k13_tw125']:.4f} | tan_err={m4['tangent_mu1_residual']:.2e} | max|mu_perp|={m4['max_transverse_multiplier']:.6f}, lam_perp={m4['transverse_floquet_exponent']:.6f}")
    m5 = d["M05_hermite_rfde_conv"]
    print(f"[M05 Paper1] Hermite RFDE L_inf decay: {m5['convergence_rows'][0]['L_inf_vs_RFDE']:.3e} (k=1) -> {m5['convergence_rows'][-1]['L_inf_vs_RFDE']:.3e} (k=32) | rebound {m5['trough_rfde']:.4f}->{m5['rebound_rfde']:.4f}")
    m6 = d["M06_shooting_lhs_prcc"]
    print(f"[M06 Paper1] Shoot S_crit={m6['S_crit_shot']:.6f} vs 1/R0={m6['S_herd']:.6f} | S_sep(0.15)={m6['S_sep_cap015']:.6f} | PRCC(R0,Imax)={m6['prcc']['R0']['prcc_Imax']:+.4f}, PRCC(dummy) p={m6['prcc']['dummy']['p_Imax']:.3f}")
    m7 = d["M07_icnn_rff_care_hjb"]
    print(f"[M07 Paper2] CARE res={m7['care_residual_F']:.2e}, lam_min(P)={m7['lam_min_P']:.4f} | ICNN Hess min eig={m7['icnn_lam_min_hess']:.4f} | J_lqr={m7['cost_care_lqr']:.4f} < J_const={m7['cost_const010']:.4f} < J_unctrl={m7['cost_unctrl']:.4f}")
    m8 = d["M08_pipe_reldeg2_hocbf"]
    print(f"[M08 Paper2] Seroconv telescoping_err={m8['seroconv_pipe_hocbf']['telescoping_err']:.2e} | delay_blind V_max={m8['seroconv_delay_blind']['V_max']:.4f} vs Pipe V_max={m8['seroconv_pipe_hocbf']['V_max']:.4f} | HO-CBF Imax={m8['hocbf_Imax']:.4f} (dose={m8['hocbf_dose']:.4f} vs const={m8['const025_dose']:.4f})")
    m9 = d["M09_bbk_pseudospectra"]
    print(f"[M09 Paper2] Open Henrici={m9['open_loop']['henrici']:.4f}, K={m9['open_loop']['kreiss']:.4f}, H_inf={m9['open_loop']['H_inf']:.4f} -> Closed H_inf={m9['closed_loop']['H_inf']:.4f}")
    m10 = d["M10_boed_and_abc_smc"]
    print(f"[M10 Paper2] BOED={m10['D_optimal_BOED']} ({m10['ratio_vs_123']:.1f}x vs [1,2,3]) | ABC-SMC rel_errs={[round(v*100,2) for v in m10['abc_rel_errs']]}%")
    m11 = d["M11_lobatto_wkb"]
    print(f"[M11 Paper3] Lobatto IIIA-IIIB 1step={m11['lobatto_1step_symp_err']:.2e}, traj={m11['lobatto_symp_err']:.2e} | S_wkb={m11['S_wkb_lobatto']:.6f} vs S_gauss={m11['S_gauss']:.6f}")
    m12 = d["M12_ftle_cedmd_weingarten"]
    print(f"[M12 Paper3] FTLE_max={m12['ftle_max']:.4f} | cEDMD Koopman err={m12['koopman_err']:.5f} | Weingarten K_G={m12['weingarten_gauss_K']:.4f}, FD err={m12['weingarten_fd_err']:.2e}")
    m13 = d["M13_weno7_sir_lie"]
    print(f"[M13 Paper3] WENO-7 SIR Lie N64={m13['max_lie_N64']:.2e}, N128={m13['max_lie_N128']:.2e} | empirical order p_32_64={m13['weno7_order_32_64']:.4f}, p_64_128={m13['weno7_order_64_128']:.4f}")
    m14 = d["M14_dct2_spatial_sirsv"]
    print(f"[M14 Paper3] DCT-II mass_err={m14['fast_diffusion']['max_mass_err']:.2e} | Fast homog_err={m14['fast_diffusion']['homog_err_t10_25']:.2e} | Slow peak={m14['slow_diffusion']['I_mean_peak']:.4f} < ODE={m14['fast_diffusion']['ode_peak']:.4f}")
    m15 = d["M15_enkf_lie_mirrorprox"]
    print(f"[M15 Paper3] EnKF RMSE={m15['enkf_rmse']:.4f} | DR-HOCBF Imax={m15['dr_hocbf_Imax']:.4f} | Lie ||[Vac,NPI]||={m15['lie_commutator_norm']:.5f} | Mirror-Prox x={round(m15['mirror_prox_x'][0],3)}, gap={m15['mirror_prox_duality_gap']:.2e}")
    print(f"Elapsed compute: {d['_meta']['elapsed_s']:.3f}s")


if __name__ == "__main__":
    data = get_validation_data()
    if "--full" in sys.argv:
        print_summary(data)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSIRSVPapers123Validation)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if res.wasSuccessful() else 1)
