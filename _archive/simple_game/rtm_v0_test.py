"""
rtm_v0_test.py — validate the v0 ERCOT small-scale RTM game (FORMULATION.md).

Proves:
  1. Per-agent mpQP → Mode 1 (combiner + KKT variational filter) → single-valued map.
  2. Plain map lookup == ADMM == centralized QP over random θ in the n_p = 1+N box.
     THE key check: the pipeline handles the NEW one-sided L_min-only coupling.
  3. (optional) Mode 3 (FACET) == Mode 1.  OFF by default: the combiner (Mode 1) is
     EXACT and instant at this scale; FACET is the offline-SCALING path (v3+) and
     currently hits the parallel-seed spawn-hang noted in PLAN_mp_focapo_rtm.md.
     Enable with RUN_FACET=1.

ADMM note: γ=5e-3 is small ⇒ ADMM needs ~1900 iters (~1.2 s) per θ; keep the θ sample
modest.  The plot overlays the fast `centralized` reference instead of ADMM.

Run:  python simple_game/rtm_v0_test.py
Outputs: table + simple_game/out/rtm_v0_map.png
"""
from __future__ import annotations
import os, sys, time
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
from amrhg.solvers.gne_combiner import build_gne_solution, filter_variational_kkt
from amrhg.solvers.admm_solver import admm_solve

from rtm_v0_game import build_rtm_v0_game, centralized_gne

OUT = _HERE / "out"; OUT.mkdir(exist_ok=True)
TOL_GATE = 1e-3
N_THETA = 60
RUN_FACET = bool(int(os.environ.get("RUN_FACET", "0")))
RNG = np.random.default_rng(0)


def _rand_theta(game, n):
    lb, ub = np.asarray(game.p_lb), np.asarray(game.p_ub)
    return lb + RNG.random((n, len(lb))) * (ub - lb)


def main():
    print("=" * 74)
    print("rtm v0 — ERCOT small-scale, grid-only, H=1, L_min-only coupling")
    print("=" * 74)

    game, grid = build_rtm_v0_game()
    N = game.N
    l_min = -float(game.d[0])
    lam_lb, lam_ub = float(game.p_lb[0]), float(game.p_ub[0])
    print(f"\nFleet: {[s.name for s in grid]}  (p_max={[s.p_max for s in grid]} kW, "
          f"a={[s.a for s in grid]} $/MWh)")
    print(f"n_p = {game.n_p}  θ = [λ, p_DA×{N}]   coupling: Σp ≥ L_min={l_min:.0f} (L_max dropped)")
    print(f"λ box = [{lam_lb:.0f}, {lam_ub:.0f}] $/MWh")

    print("\n[1] Per-agent mpQP ...")
    t0 = time.perf_counter()
    sols = solve_all_agents_mp(game, verbose=False)
    print(f"    per-agent CRs = {[s.n_cr for s in sols]}  ({time.perf_counter()-t0:.2f}s)")

    print("[2] Mode 1 — single-valued variational map (KKT filter) ...")
    t0 = time.perf_counter()
    gne_full = build_gne_solution(game, sols, verbose=False, equilibrium_select="potential")
    gne1 = filter_variational_kkt(gne_full, game, verbose=False)
    print(f"    {gne_full.n_cr} GNE CRs → {gne1.n_cr} variational CRs "
          f"({time.perf_counter()-t0:.2f}s)")

    gne3 = None
    if RUN_FACET:
        print("[3] Mode 3 — FACET variational map ...")
        try:
            from amrhg.solvers.facet_gne import (find_all_agent_cr_neighbors,
                                                 build_gne_solution_facet)
            t0 = time.perf_counter()
            sf = find_all_agent_cr_neighbors(sols, method="facet_adjacency", verbose=False)
            fr = build_gne_solution_facet(game, sf, verbose=False,
                                          equilibrium_select="potential")
            gne3 = filter_variational_kkt(fr.gne_sol, game, verbose=False)
            print(f"    FACET {fr.gne_sol.n_cr} → {gne3.n_cr} variational "
                  f"({time.perf_counter()-t0:.2f}s)")
        except Exception as e:
            print(f"    Mode 3 FAILED: {e}")
    else:
        print("[3] Mode 3 (FACET) — SKIPPED (set RUN_FACET=1 to enable; Mode 1 is exact here)")

    print(f"\n[4] Cross-check over {N_THETA} random θ (plain lookup vs ADMM & centralized) ...")
    thetas = _rand_theta(game, N_THETA)
    e_admm = e_cent = e13 = 0.0
    misses = conf = n_ok = 0
    worst = None
    for th in thetas:
        ks = gne1.locate_all(th)
        x1 = gne1.evaluate(th)
        if x1 is None:
            misses += 1; continue
        n_ok += 1
        if len(ks) > 1:
            xs = [gne1.regions[k].evaluate(th) for k in ks]
            if max(np.max(np.abs(xs[0] - xj)) for xj in xs[1:]) > 1e-6:
                conf += 1
        xa = admm_solve(game, th, rho=0.5, max_iter=8000, tol=1e-10).x_stacked
        xc = centralized_gne(game, grid, th)
        ea = float(np.max(np.abs(x1 - xa))); ec = float(np.max(np.abs(x1 - xc)))
        if ea > e_admm:
            e_admm = ea; worst = (th, x1, xa, xc)
        e_cent = max(e_cent, ec)
        if gne3 is not None:
            x3 = gne3.evaluate(th)
            if x3 is not None:
                e13 = max(e13, float(np.max(np.abs(x1 - x3))))

    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    print(f"  per-agent CRs                : {[s.n_cr for s in sols]}")
    print(f"  Mode 1 variational CRs       : {gne1.n_cr}")
    if gne3 is not None:
        print(f"  Mode 3 (FACET) variational   : {gne3.n_cr}"
              f"  ({'== Mode 1' if gne3.n_cr == gne1.n_cr else 'differs'})")
        print(f"  max |Mode1 − Mode3|          : {e13:.2e} kW")
    print(f"  random θ checked             : {n_ok} (misses {misses})")
    print(f"  conflicting overlaps         : {conf} / {n_ok} (single-valued if 0)")
    print(f"  max |map − centralized|      : {e_cent:.2e} kW  (gate < {TOL_GATE})")
    print(f"  max |map − ADMM|             : {e_admm:.2e} kW  (gate < {TOL_GATE})")
    if worst:
        th, x1, xa, xc = worst
        print(f"  worst-vs-ADMM θ=λ{th[0]:.1f} p_DA{np.round(th[1:],1)}: "
              f"map={np.round(x1,2)} admm={np.round(xa,2)} cent={np.round(xc,2)}")
    gate = (e_admm < TOL_GATE and e_cent < TOL_GATE and misses == 0 and conf == 0)
    print("\n  CORE GATE (single-valued, plain lookup == ADMM == centralized, 0 misses): "
          + ("PASS ✅" if gate else "FAIL ❌"))

    _plot(game, grid, gne1)
    print(f"\n  Map figure → {OUT / 'rtm_v0_map.png'}")


def _plot(game, grid, gmap):
    """p_i vs λ at a fixed p_DA — map (line) vs centralized (markers, fast ref)."""
    N = game.N
    lam_lb, lam_ub = float(game.p_lb[0]), float(game.p_ub[0])
    p_da = np.array([0.4 * s.p_max for s in grid])       # fixed anchor for the slice
    lams = np.linspace(lam_lb, lam_ub, 120)
    lam_mk = np.linspace(lam_lb, lam_ub, 24)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, N))
    for i, s in enumerate(grid):
        pm = [gmap.evaluate(np.concatenate([[lam], p_da]))[game.x_slice(i).start] for lam in lams]
        pc = [centralized_gne(game, grid, np.concatenate([[lam], p_da]))[i] for lam in lam_mk]
        ax.plot(lams, pm, "-", color=colors[i], lw=2, label=f"{s.name} (a={s.a:.0f})")
        ax.plot(lam_mk, pc, "o", color=colors[i], ms=4, mfc="none")
        ax.axvline(s.a, color=colors[i], ls=":", lw=0.8, alpha=0.5)
    l_min = -float(game.d[0])
    ax.set_xlabel("RT price  λ  [$/MWh]"); ax.set_ylabel("grid buy  p  [kW]")
    ax.set_title(f"rtm v0 map — grid buy vs λ  (p_DA fixed, Σp≥L_min={l_min:.0f})\n"
                 "lines = explicit map, ○ = centralized,  : = break-even a_i", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "rtm_v0_map.png", dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    main()
