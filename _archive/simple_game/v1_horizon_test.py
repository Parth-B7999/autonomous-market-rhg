"""
v1_horizon_test.py — The CR-explosion test and its exact decoupling fix.

Demonstrates the plan's v1 result:
  1. NAIVE full-horizon mpQP: per-agent CR count grows ~(single-step count)^H as we
     put the whole H-step price forecast into θ.  Measured for H = 1..H_MAX.
  2. EXACT decoupling reduction: with no ramp/storage/H₂ the H-step game is H
     independent copies of the v0 single-step game.  Build the single-step variational
     map ONCE, apply per step → CR count flat in H, θ stays 1-D.
  3. VALIDATION: the decoupled map reproduces (a) the naive full-horizon variational
     map (checked at H=2) and (b) ADMM on the full coupled H-step game (checked at
     H=6), to tolerance.

Run:  python simple_game/v1_horizon_test.py
Outputs: table to stdout + simple_game/out/v1_cr_explosion.png
"""

from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
for pth in (str(_HERE.parent / "src"), str(_HERE)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from amrhg.solvers.mp_solver import solve_all_agents_mp
from amrhg.solvers.gne_combiner import build_variational_gne_solution
from amrhg.solvers.admm_solver import admm_solve

from v0_game import build_v0_game
from v1_game import build_v1_game

OUT = _HERE / "out"; OUT.mkdir(exist_ok=True)

H_MAX_NAIVE = 5          # sweep naive per-agent mpQP for H = 1..H_MAX_NAIVE
H_VALIDATE  = 6          # validate decoupling against ADMM at this horizon
SEED        = 0


def decoupled_eval(g_step, H, lam_vec):
    """Full-horizon GNE by applying the single-step variational map per step.
    Returns x in v1 stacking order: x[i*H + k] = p_{i,k}."""
    N = g_step.N
    x = np.full(N * H, np.nan)
    for k in range(H):
        s_k = g_step.evaluate(np.array([lam_vec[k]]))   # [p_0k, p_1k, ..., p_{N-1,k}]
        if s_k is None:
            return None
        for i in range(N):
            x[i * H + k] = s_k[i]
    return x


def main():
    print("=" * 70)
    print("v1 — horizon CR-explosion test + exact decoupling reduction")
    print("=" * 70)

    # ── 1. Naive full-horizon mpQP: per-agent CR count vs H ───────────────────
    print("\n[1] Naive full-horizon per-agent mpQP — CR count vs H")
    print(f"    (θ_i = [x_-i ((N-1)H); λ (H)];  expect CRs ~ single-step^H)\n")
    Hs, cr_naive, t_naive = [], [], []
    for H in range(1, H_MAX_NAIVE + 1):
        game, fleet = build_v1_game(H)
        t0 = time.perf_counter()
        try:
            sols = solve_all_agents_mp(game, verbose=False)
            dt = time.perf_counter() - t0
            counts = [s.n_cr for s in sols]
            theta_dim = (game.N - 1) * H + H
            Hs.append(H); cr_naive.append(max(counts)); t_naive.append(dt)
            print(f"    H={H}: θ_i={theta_dim:3d}  per-agent CRs={counts}  "
                  f"max={max(counts):5d}  ({dt:.2f}s)")
        except Exception as e:
            print(f"    H={H}: FAILED/too big: {type(e).__name__}: {e}")
            break

    # ── 2. Decoupling reduction: single-step variational map (built ONCE) ─────
    print("\n[2] Decoupling reduction — single-step variational map (v0)")
    g0, fleet = build_v0_game()
    step_sols = solve_all_agents_mp(g0, verbose=False)
    g_step = build_variational_gne_solution(g0, step_sols, verbose=False)
    print(f"    single-step map: {g_step.n_cr} variational CRs, θ=1  (FLAT in H)")

    # ── 3a. Validate decoupled == naive full-horizon variational map (H=2) ────
    print("\n[3a] Validate: decoupled == naive full-horizon variational map (H=2)")
    H2 = 2
    game2, _ = build_v1_game(H2)
    sols2 = solve_all_agents_mp(game2, verbose=False)
    gne2 = build_variational_gne_solution(game2, sols2, verbose=False)
    rng = np.random.default_rng(SEED)
    lo, hi = float(g0.p_lb[0]), float(g0.p_ub[0])
    err_naive = 0.0
    for _ in range(200):
        lam = rng.uniform(lo, hi, size=H2)
        x_full = gne2.evaluate(lam)
        x_dec  = decoupled_eval(g_step, H2, lam)
        if x_full is not None and x_dec is not None:
            err_naive = max(err_naive, float(np.max(np.abs(x_full - x_dec))))
    print(f"    naive-map({gne2.n_cr} CRs) vs decoupled: max|Δ| = {err_naive:.2e} kW")

    # ── 3b. Validate decoupled == ADMM on full coupled H-step game (H=6) ──────
    print(f"\n[3b] Validate: decoupled == ADMM on full coupled game (H={H_VALIDATE})")
    gameH, _ = build_v1_game(H_VALIDATE)
    err_admm = 0.0; worst = None
    for _ in range(50):
        lam = rng.uniform(lo, hi, size=H_VALIDATE)
        x_dec = decoupled_eval(g_step, H_VALIDATE, lam)
        res = admm_solve(gameH, lam, rho=0.5, max_iter=3000, tol=1e-8)
        e = float(np.max(np.abs(x_dec - res.x_stacked)))
        if e > err_admm:
            err_admm = e; worst = lam
    print(f"    decoupled vs ADMM (full {H_VALIDATE}-step game): max|Δ| = {err_admm:.2e} kW")

    gate = (err_naive < 1e-3 and err_admm < 1e-3)
    print("\n  GATE (decoupling exact vs naive-map AND vs ADMM): "
          + ("PASS ✅" if gate else "FAIL ❌"))

    # ── figure: CR count vs H, naive (exponential) vs decoupled (flat) ───────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(Hs, cr_naive, "o-", color="crimson", lw=2,
                label="naive full-horizon (per-agent CRs)")
    ax.axhline(g_step.n_cr, color="seagreen", ls="--", lw=2,
               label=f"decoupled single-step map ({g_step.n_cr} CRs, flat)")
    ref = [cr_naive[0] * (cr_naive[1] / cr_naive[0]) ** (h - 1) for h in Hs] \
        if len(cr_naive) > 1 else cr_naive
    ax.semilogy(Hs, ref, ":", color="gray", lw=1, label="geometric reference")
    ax.set_xlabel("horizon H (5-min steps)")
    ax.set_ylabel("critical regions per agent (log)")
    ax.set_title("v1 CR-explosion: naive full-horizon θ vs exact per-step decoupling")
    ax.set_xticks(Hs); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(OUT / "v1_cr_explosion.png", dpi=130)
    plt.close(fig)
    print(f"\n  Figure → {OUT / 'v1_cr_explosion.png'}")


if __name__ == "__main__":
    main()
