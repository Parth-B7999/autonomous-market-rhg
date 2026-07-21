"""
v2_test.py — Renewable agents: the 2-D (λ, g) single-step variational GNE map.

Deliverables (per PLAN_mp_focapo_rtm.md, v2):
  1. Build the single-step v2 game (2 grid-only + 1 renewable) → per-agent mpQP →
     single-valued variational map (Mode 1) and FACET (Mode 3).
  2. Validate the map == ADMM == centralized over the 2-D (λ, g) parameter box,
     using PLAIN lookup (no costs online).
  3. 2-D map figure over the (λ, g) plane.
  Decoupling across the horizon holds by the SAME structure proved in v1 (no ramp /
  storage / H₂ ⇒ steps independent); the 2-D single-step map is applied per step.

Run:  python simple_game/v2_test.py
Outputs: table + simple_game/out/v2_map.png
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
from amrhg.solvers.gne_combiner import (
    build_gne_solution, filter_variational_kkt,
)
from amrhg.solvers.facet_gne import find_all_agent_cr_neighbors, build_gne_solution_facet
from amrhg.solvers.admm_solver import admm_solve

from v2_game import build_v2_step_game

OUT = _HERE / "out"; OUT.mkdir(exist_ok=True)
TOL_GATE = 1e-3


def main():
    print("=" * 70)
    print("v2 — renewable agents: 2-D (λ, g) single-step variational GNE map")
    print("=" * 70)

    game, specs, R = build_v2_step_game()
    N = game.N
    l_max, l_min = float(game.d[0]), float(-game.d[1])
    lam_lb, lam_ub = float(game.p_lb[0]), float(game.p_ub[0])
    g_lb, g_ub = float(game.p_lb[1]), float(game.p_ub[1])
    print(f"\nFleet: {[s.name for s in specs]}  (n_p={game.n_p} = [λ, g])")
    print(f"Coupling {l_min:.0f} ≤ Σp ≤ {l_max:.0f} | λ∈[{lam_lb:.0f},{lam_ub:.0f}] | g∈[{g_lb:.0f},{g_ub:.0f}]")

    # ── 1. mpQP → variational maps ────────────────────────────────────────────
    print("\n[1] Per-agent mpQP ...")
    sols = solve_all_agents_mp(game, verbose=False)
    print(f"    per-agent CRs = {[s.n_cr for s in sols]}")

    print("[2] Mode 1 — single-valued variational map (KKT filter) ...")
    t0 = time.perf_counter()
    gne_full = build_gne_solution(game, sols, verbose=False, equilibrium_select="potential")
    gne1 = filter_variational_kkt(gne_full, game, verbose=False)
    print(f"    {gne_full.n_cr} GNE CRs → {gne1.n_cr} variational CRs "
          f"({time.perf_counter()-t0:.2f}s)")

    # Single-valuedness: plain locate() must be unambiguous online (no costs).
    conf = 0
    for lam in np.linspace(lam_lb, lam_ub, 41):
        for g in np.linspace(g_lb, g_ub, 41):
            th = np.array([lam, g]); ks = gne1.locate_all(th)
            if len(ks) > 1:
                xs = [gne1.regions[k].evaluate(th) for k in ks]
                if max(np.max(np.abs(xs[0] - xj)) for xj in xs[1:]) > 1e-6:
                    conf += 1
    print(f"    conflicting overlaps: {conf}/1681 "
          f"({'single-valued ✅' if conf == 0 else 'NOT single-valued'})")

    print("[3] Mode 3 — FACET single-valued variational map (KKT filter) ...")
    gne3 = None
    try:
        sf = find_all_agent_cr_neighbors(sols, method="facet_adjacency", verbose=False)
        fr = build_gne_solution_facet(game, sf, verbose=False, equilibrium_select="potential")
        gne3 = filter_variational_kkt(fr.gne_sol, game, verbose=False)
        print(f"    FACET {fr.gne_sol.n_cr} → {gne3.n_cr} variational "
              f"(checked {fr.n_combos_checked}/{fr.n_combos_total})")
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"    Mode 3 FAILED: {e}")

    # ── 4. validate over 2-D (λ, g) grid — PLAIN lookup (no costs online) ─────
    print("\n[4] Cross-check over (λ, g) grid: single-valued map (plain lookup) vs ADMM ...")
    lam_grid = np.linspace(lam_lb, lam_ub, 31)
    g_grid = np.linspace(g_lb, g_ub, 31)
    e13 = e1a = 0.0; misses = 0; n = 0
    worst = None
    for lam in lam_grid:
        for g in g_grid:
            th = np.array([lam, g]); n += 1
            x1 = gne1.evaluate(th)                     # PLAIN locate() + affine
            if x1 is None:
                misses += 1; continue
            xa = admm_solve(game, th, rho=0.5, max_iter=3000, tol=1e-8).x_stacked
            e1a_ = float(np.max(np.abs(x1 - xa)))
            if e1a_ > e1a:
                e1a = e1a_; worst = (lam, g, x1, xa)
            if gne3 is not None:
                x3 = gne3.evaluate(th)
                if x3 is not None:
                    e13 = max(e13, float(np.max(np.abs(x1 - x3))))

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  per-agent CRs                 : {[s.n_cr for s in sols]}")
    print(f"  Mode 1 variational CRs        : {gne1.n_cr}")
    print(f"  Mode 3 (FACET) variational CRs: {gne3.n_cr if gne3 is not None else 'N/A'}"
          f"  ({'== Mode 1' if gne3 is not None and gne3.n_cr == gne1.n_cr else 'differs'})")
    print(f"  conflicting overlaps          : {conf} / 1681 (single-valued if 0)")
    print(f"  lookup misses                 : {misses} / {n}")
    print(f"  max |Mode1 − Mode3|           : {e13:.2e} kW")
    print(f"  max |map (plain lookup) − ADMM|: {e1a:.2e} kW  (gate < {TOL_GATE})")
    if worst:
        lam, g, x1, xa = worst
        print(f"  worst-vs-ADMM (λ={lam:.1f}, g={g:.1f}): map={np.round(x1,1)} admm={np.round(xa,1)}")
    gate = (e1a < TOL_GATE and misses == 0 and conf == 0)
    print("\n  GATE (single-valued map, plain lookup == ADMM, 0 misses, 0 overlaps): "
          + ("PASS ✅" if gate else "FAIL ❌"))

    _plot_2d(game, gne1, lam_lb, lam_ub, g_lb, g_ub)
    print(f"\n  Map figure → {OUT / 'v2_map.png'}")


def _plot_2d(game, gmap, lam_lb, lam_ub, g_lb, g_ub):
    ng = 90
    lam_grid = np.linspace(lam_lb, lam_ub, ng)
    g_grid = np.linspace(g_lb, g_ub, ng)
    LAM, G = np.meshgrid(lam_grid, g_grid)
    N = game.N
    ren_idx = game.agents[-1].index                     # renewable agent (last)
    ren_p_off = game.x_slice(ren_idx).start
    ren_cv_off = ren_p_off + 1
    AGG = np.full_like(LAM, np.nan)
    PELEC = np.full_like(LAM, np.nan)
    PREN = np.full_like(LAM, np.nan)
    for a in range(ng):
        for b in range(ng):
            x = gmap.evaluate(np.array([LAM[a, b], G[a, b]]))
            if x is None:
                continue
            AGG[a, b] = sum(x[game.x_slice(i).start] for i in range(N))   # Σ grid p
            PREN[a, b] = x[ren_p_off]                                     # renewable grid import
            PELEC[a, b] = x[ren_p_off] + G[a, b] - x[ren_cv_off]          # renewable p_elec = p+g-cv

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, Z, title, cmap in [
        (axs[0], AGG, "aggregate grid import  Σp*  [kW]", "viridis"),
        (axs[1], PREN, "renewable agent grid import  p*  [kW]", "plasma"),
        (axs[2], PELEC, "renewable electrolyzer load  p+g−cv  [kW]", "cividis"),
    ]:
        pc = ax.pcolormesh(LAM, G, Z, cmap=cmap, shading="auto")
        fig.colorbar(pc, ax=ax, fraction=0.046)
        ax.set_xlabel("price λ  [$/MWh]"); ax.set_ylabel("renewable g  [kW]")
        ax.set_title(title, fontsize=10)
    fig.suptitle("v2 explicit variational GNE map over (λ, g)  —  single step, applied per horizon step",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "v2_map.png", dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    main()
