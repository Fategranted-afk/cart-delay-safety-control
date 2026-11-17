# Delay-Aware Safety-Critical Optimal Control & Longitudinal Identifiability (`0–180 d`)

Code, non-tautological verification suites, `V1–V7` & cross-domain `SIRSV` benchmarks, `0–180 d` longitudinal telemetry, and vector 6-panel figure atlas for dual-channel controlled $(4+k)\mathrm{D}$ Erlang-cascade and discrete-delay RFDE solid-tumor cellular immunotherapy.

## Core Mathematical & Computational Architecture

Six numerical operators ($\mathcal{O}_1\text{–}\mathcal{O}_6$) govern the longitudinal `[0, 180]` day synthesis pipeline:

| Operator | Methodological Scope | Key Analytical & Computational Guarantees |
| :---: | :--- | :--- |
| $\mathcal{O}_1$ | Pontryagin Adjoint (`PMP`) & Reduced $5\mathrm{D}$ `CARE-LQR` | Eliminates rank-deficient naive immune $I{=}0$ eigenspace; Newton–Kleinman Riccati residual $3.46 \times 10^{-22}$; `PMP` cost $J = 0.4612$ vs.\ uncontrolled $10.8977$. |
| $\mathcal{O}_2$ | Strictly Spatially Convex `ICNN-FBSDE` (Time-Only RFF) | Sinusoidal Random Fourier Features restricted strictly to time $t$; certifies uniform spatial Hessian floor $\nabla_{\tilde{\mathbf{z}}}^2 V \succeq 0.12\,\mathbf{I} \succ 0$ ($\lambda_{\min} = 0.2508$ observed). |
| $\mathcal{O}_3$ | Smooth Two-Tier `Pipe-HOCBF` + 2D KKT QP & Cantelli/Gaussian SOCP | Exact telescoping conservation ($\dot{M}_{\mathrm{comm}} = f_{\mathrm{comm}} + \sigma_s^k u_A + u_B$); $\gamma_B \ge d_c/\lambda_{\mathrm{dec}} > 0$; continuous unilateral shutoff contraction; $\kappa = \Phi^{-1}(0.99) = 2.3263$ ($99.0\%$ Gaussian / $84.40\%$ worst-case Cantelli compliance) cutting epistemic $C_{\max}$ breaches $44.50\% \to 1.25\%$. |
| $\mathcal{O}_4$ | Synchronous Leibniz–Hermite `RK4` RFDE Solver | Differential Leibniz pipeline law immune to odd–even Simpson quadrature aliasing; strictly monotonic $L^2(C): 76.98 \to 15.67$ across $k = 1\dots 32$. |
| $\mathcal{O}_5$ | True $6\mathrm{D}$ Equilibrium & Complex-128 SVD Resolvent Pseudospectra | Evaluates $\bar{\mathbf{z}}$ ($\|\mathbf{f}\|_2 = 0$); active barrier deepens $\mu_2: -0.0173 \to -0.1179$ and cuts effector $\mathcal{H}_\infty$ peak $3.414 \to 1.964$ ($42.5\%$ attenuation). |
| $\mathcal{O}_6$ | $24\mathrm{D}$ Variational Sensitivity + 495-Schedule BOED & 3-Stage `ABC-SMC` | Ranks all $\binom{12}{4} = 495$ protocols over $0\text{–}180\text{ d}$; optimal `[2, 30, 45, 120] d` yields $2.03 \times 10^6\times$ Fisher determinant gain over acute $7\text{ d}$ and recovers $(\delta_C, \rho, \tau)$ within $0.67\%\text{–}5.30\%$. |

## Repository Contents

| Path | Description |
| :--- | :--- |
| `colab_paper2_controlled_gpu_master.py` | Full controlled ODE/SDE/RFDE engine, `12/12` `Stage 1` test suite, `5,184`-trajectory LHS Pareto sweep, $6$ synthetic phenotypic regimes, pseudospectra, BOED, `ABC-SMC`, and GPU-scalable PyTorch marathon mode. |
| `verify_paper2_controlled_suite.py` | Zero-dependency standalone `unittest` verifier (`5/5` tests across Lyapunov/telescoping, Lie FD parity, `V1–V5` / `SIRSV M07–M10`, Gaussian/Cantelli SOCP duality, and `0–180 d` JSON telemetry). |
| `colab_sirsv_papers123_master_validation.py` | Standalone `15/15` cross-domain `SIRSV / SIRS / SIR` operator gatekeeper (`M01–M15`). |
| `data/paper2_telemetry_summary.json` | Verified `0–180 d` benchmark & cohort telemetry snapshot. |
| `data/sirsv_papers123_validation.json` | Certified `SIRSV` cross-domain telemetry archive. |
| `figures/paper2_controlled_6panel_atlas.pdf` | Publication-grade 6-panel vector synthesis (`Figures 1–6`). |

## Canonical Benchmark Verification (`V1–V7`)

- `V1` ($\mathcal{O}_1$): Canonical 2D Continuous Algebraic Riccati Equation ($P_{11} = P_{22} = \sqrt{3}$, residual $4.44 \times 10^{-16} < 10^{-14}$).
- `V2` ($\mathcal{O}_3$): Unilateral Double Integrator safety filter ($u \ge 0$, $p_{\max} = 9.9968 \le p_{\mathrm{safe}} = 10.0$).
- `V3` ($\mathcal{O}_3$): Analytical Lie derivative $\dot{h}_{\mathrm{comp}}$ central finite-difference parity ($< 10^{-5}$) and $\gamma_B \ge d_c/\lambda_{\mathrm{dec}}$.
- `V4` ($\mathcal{O}_2$): Strictly convex `ICNN` 6D spatial Hessian floor under nonzero saddle ridge ($\lambda_{\min} = 0.2508 > 0.12$).
- `V5` ($\mathcal{O}_3$): Dual Gaussian ($99.0\%$) / Cantelli ($84.40\%$) SOCP boundary feasibility ($\psi_{\mathrm{DR}} \le 0$).
- `V6` ($\mathcal{O}_6$): $24\mathrm{D}$ variational sensitivity parity against perturbed `RK4` central finite differences ($< 5 \times 10^{-4}$).
- `V7` ($\mathcal{O}_1\text{–}\mathcal{O}_6$): Delayed `SIRSV` epidemiological cascade (`CARE` $2.55 \times 10^{-15}$, $V_{\max} = 0.4500$, $\mathcal{H}_\infty: 1.7538 \to 0.9164$, `ABC-SMC` $< 1\%$).

## Quickstart & CLI Modes

```bash
# Fast pure-Python / NumPy verification gates
python3 colab_paper2_controlled_gpu_master.py --verify-only
python3 verify_paper2_controlled_suite.py --verify-only
python3 colab_sirsv_papers123_master_validation.py --verify-only

# End-to-end smoke test & CPU scientific JSON export
python3 colab_paper2_controlled_gpu_master.py --smoke
python3 colab_paper2_controlled_gpu_master.py --numpy-scientific --export-json data/paper2_telemetry_summary.json

# Full PyTorch GPU run (Colab / CUDA workstation)
python3 colab_paper2_controlled_gpu_master.py --full
```
