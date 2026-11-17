#!/usr/bin/env python3
"""
Standalone Pure-Stdlib Repository Verification Suite for Paper II (0-180 Days Controlled Suite).
Verifies analytical identities, canonical V1-V7 & SIRSV M07-M10 gates, Gaussian (99.0%) /
one-sided Cantelli (84.40%) SOCP duality, telemetry JSON invariants, and vector atlas assets
using strictly local repository files with zero hardcoded machine paths.
"""

import argparse
import json
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
sys.path.insert(0, str(ROOT))

import colab_paper2_controlled_gpu_master as p2  # noqa: E402


class TestPaper2ControlledRepo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.telem = json.loads((DATA_DIR / "paper2_telemetry_summary.json").read_text(encoding="utf-8"))
        cls.sirsv = json.loads((DATA_DIR / "sirsv_papers123_validation.json").read_text(encoding="utf-8"))

    def test_01_decreasing_weight_lyapunov_and_erlang_telescoping(self):
        """Proposition 1 decreasing weights & Theorem 2 exact telescoping across k in {1, 2, 4}."""
        p = dict(p2.P_NOM_PAPER2)
        p["d_a"] = 0.03
        for kv in (1, 2, 4):
            zk = [1900.0, 45.0, 220.0] + [110.0 - 15.0 * j for j in range(kv)] + [780.0]
            dz = p2.rhs_erlang_controlled(0.0, zk, p, tau=p["tau"], k=kv, u_ctrl=520.0, u_B_ctrl=180.0)
            met = p2.pipe_hocbf_exact_2d(zk, 0.0, 0.0, p, tau=p["tau"], k=kv)
            sig = met["sigma_k"] ** (1.0 / kv)
            dM_num = sum((sig ** (kv - j)) * dz[3 + j] for j in range(kv)) + dz[3 + kv]
            dM_ex = met["f_comm"] + (sig ** kv) * 520.0 + 180.0
            self.assertLess(abs(dM_num - dM_ex), 1.0e-11)

            w = [2.0 - float(j) / float(kv) for j in range(kv)]
            r_tau = kv / p["tau"]
            dW = sum(w[j] * dz[3 + j] for j in range(kv)) + dz[3 + kv]
            bound = (
                2.0 * 520.0 + 180.0 + 2.0 * p["rho"] * (zk[1] + zk[2]) * zk[3 + kv]
                - sum((r_tau / float(kv) + w[j] * p["d_a"]) * zk[3 + j] for j in range(kv))
                - (p["omega"] * zk[0] + p["d_c"]) * zk[3 + kv] - p["theta"] * (zk[3 + kv] ** 2)
            )
            self.assertLessEqual(dW, bound + 1.0e-11)

    def test_02_lie_derivative_fd_parity_and_terminal_authority(self):
        """Theorem 3 analytical L_{f0} h_comp - gamma_A u_A - gamma_B u_B vs central FD & gamma_B floor."""
        p = dict(p2.P_NOM_PAPER2)
        uA, uB = 380.0, 210.0
        dt_fd = 1.0e-6
        for zt in ([500.0, 190.0, 480.0, 90.0, 70.0, 1220.0], [2100.0, 25.0, 150.0, 100.0, 80.0, 940.0]):
            met = p2.pipe_hocbf_exact_2d(zt, 0.0, 0.0, p=p)
            dh_ex = met["Lf_h"] - met["gamma_A"] * uA - met["gamma_B"] * uB
            f_tot = p2.rhs_erlang_controlled(0.0, zt, p=p, u_ctrl=uA, u_B_ctrl=uB)
            zp = [zt[i] + dt_fd * f_tot[i] for i in range(6)]
            zm = [zt[i] - dt_fd * f_tot[i] for i in range(6)]
            dh_fd = (p2.pipe_hocbf_exact_2d(zp, p=p)["h_comp"] - p2.pipe_hocbf_exact_2d(zm, p=p)["h_comp"]) / (2.0 * dt_fd)
            self.assertLess(abs(dh_ex - dh_fd), 1.0e-5)
            self.assertGreaterEqual(met["gamma_B"], p["d_c"] / met["lambda_dec"] - 1.0e-12)

    def test_03_canonical_care_double_integrator_and_sirsv_gates(self):
        """Canonical 2D CARE (<1e-14), unilateral double integrator (<=10.0), & SIRSV M07-M10."""
        P_c, res_c = p2.solve_2d_canonical_care()
        self.assertLess(res_c, 1.0e-14)
        self.assertAlmostEqual(P_c[0][0], math.sqrt(3.0), places=12)

        # Unilateral double integrator: dp/dt = v, dv/dt = -0.6 v + u (u in [0, 5], p_safe = 10.0)
        p_val, v_val, dt = 7.0, 1.5, 0.005
        p_max = p_val
        for _ in range(800):
            u_cap = max(0.0, min(5.0, 0.8 * (10.0 - p_val - max(0.0, v_val) / 0.6)))
            p_val += dt * v_val
            v_val += dt * (-0.6 * v_val + u_cap)
            p_max = max(p_max, p_val)
        self.assertLessEqual(p_max, 10.0 + 1e-6)

        self.assertLess(self.sirsv["M07_icnn_rff_care_hjb"]["care_residual_F"], 1.0e-14)
        self.assertLessEqual(self.sirsv["M08_pipe_reldeg2_hocbf"]["seroconv_pipe_hocbf"]["V_max"], 0.4500 + 1e-6)
        self.assertEqual(self.sirsv["M10_boed_and_abc_smc"]["D_optimal_BOED"], [2, 6, 70])

    def test_04_dual_gaussian_and_cantelli_socp_feasibility(self):
        """Proposition 3: kappa=2.3263 -> 99.0% Gaussian, 84.40% Cantelli, and feasible 2D SOCP filter."""
        kappa = 2.3263
        p_gauss = 0.5 * (1.0 + math.erf(kappa / math.sqrt(2.0)))
        p_cantelli = 1.0 - 1.0 / (1.0 + kappa * kappa)
        self.assertAlmostEqual(p_gauss, 0.9900, places=4)
        self.assertAlmostEqual(p_cantelli, 0.8440, places=4)

        p = dict(p2.P_NOM_PAPER2)
        p["d_a"] = 0.12
        z_test = [1600.0, 40.0, 220.0, 80.0, 60.0, 1180.0]
        std_xi = (0.008, 1.5e-4, 0.35, 0.045)
        uA, uB, slack = p2.dr_pipe_hocbf_cantelli_socp(
            z_test, 350.0, 800.0, p=p, std_xi=std_xi, kappa=kappa, allow_uB=True
        )
        self.assertGreaterEqual(uA, 0.0)
        self.assertGreaterEqual(uB, 0.0)
        self.assertLessEqual(uA, p["u_max"] + 1e-9)
        self.assertLessEqual(uB, p["u_max"] + 1e-9)
        self.assertLessEqual(slack, 1.0e-8)

    def test_05_longitudinal_0_to_180d_telemetry_and_atlas(self):
        """Verify 0-180d cost hierarchy, 5,184-traj Pareto 0%, RFDE, 6D SVD, regimes, BOED/ABC & atlas PDF."""
        cb = self.telem["channel_benchmark"]
        self.assertGreater(cb["delay_blind"]["C_max"], 1733.0)
        self.assertLessEqual(cb["pipe_joint2d"]["M_comm_max"], 1650.0)

        p1 = self.telem["phase1_hjb"]
        self.assertGreater(p1["min_hess_eig_with_ridge"], 0.12)
        self.assertLess(p1["care_5d_residual"], 1.0e-18)
        pol = p1["policies"]
        self.assertLess(pol["PMP_Pipe"]["J"], pol["CARE_LQR_Pipe"]["J"])
        self.assertLess(pol["CARE_LQR_Pipe"]["J"], pol["ICNN_Pipe"]["J"])
        self.assertLess(pol["ICNN_Pipe"]["J"], pol["Pulse1200_Pipe"]["J"])
        self.assertLess(pol["Pulse1200_Pipe"]["J"], pol["Constant500_Pipe"]["J"])
        self.assertLess(pol["Constant500_Pipe"]["J"], pol["Uncontrolled"]["J"])

        dr = self.telem["dr_socp_benchmark"]
        self.assertLess(dr["dr_breach_pct"], dr["nom_breach_pct"] * 0.1)

        par = self.telem["phase3_pareto"]
        self.assertEqual(par["total_patients"], 5184)
        self.assertTrue(all(r["pipe_breach"] == 0.0 for r in par["table_rows"]))

        l2c = self.telem["phase4_rfde"]["errs_l2_C"]
        vals = [l2c[str(k)] for k in (1, 2, 4, 8, 16, 32)]
        self.assertTrue(all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)))

        ps = self.telem["phase5_pseudospectra"]
        self.assertLess(ps["pipe_metrics"]["mu2"], ps["open_metrics"]["mu2"])
        self.assertLess(ps["pipe_metrics"]["eff3_H_inf"], ps["open_metrics"]["eff3_H_inf"])

        cohorts = self.telem["phase6a_cohorts"]
        self.assertEqual(len(cohorts), 6)
        self.assertTrue(all(c["T_final_cl"] <= 3.1e-4 for c in cohorts))

        boed = self.telem["phase6b_boed"]
        self.assertEqual(boed["t_early"], [2, 30, 45, 120])
        self.assertGreater(boed["info_gain_vs_7d"], 2.0e6)

        abc = self.telem["phase6c_abc_smc"]
        self.assertTrue(all(err < 0.06 for err in abc["rel_errs"]))

        atlas = FIG_DIR / "paper2_controlled_6panel_atlas.pdf"
        self.assertTrue(atlas.exists())
        self.assertGreater(atlas.stat().st_size, 100_000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Paper II repository code, benchmarks, and telemetry")
    parser.add_argument("--verify-only", action="store_true", help="Run full verification suite")
    args, _ = parser.parse_known_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPaper2ControlledRepo)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)
