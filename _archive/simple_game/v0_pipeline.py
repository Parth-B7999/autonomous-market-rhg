"""
v0_pipeline.py — Build and CROSS-CHECK the v0 explicit GNE map.

Runs the full offline pipeline on the v0 game and verifies the map agrees with
two independent references, per the plan's acceptance gate:

    max |p_map − p_ADMM| < 1e-3 kW   over a λ sweep + zero lookup misses.

Pipeline:
  1. mp_solver.solve_all_agents_mp(game)      → per-agent explicit best responses
  2. gne_combiner.build_gne_solution(...)      → Mode 1 (exhaustive) explicit GNE map
  3. facet_gne (neighbors + build_gne_solution_facet) → Mode 3 (FACET) explicit GNE map
  4. For a λ sweep:  compare  Mode-1 map ↔ Mode-3 map ↔ ADMM ↔ centralized QP.

Outputs: a PASS/FAIL table to stdout and simple_game/out/v0_map.png (the p_i*(λ) map).

Run:  python simple_game/v0_pipeline.py
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_SRC = str(_HERE.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from amrhg.solvers.mp_solver import solve_all_agents_mp
from amrhg.solvers.gne_combiner import (
    build_gne_solution, filter_variational, verify_gne_at_p,
)
from amrhg.solvers.facet_gne import (
    find_all_agent_cr_neighbors, build_gne_solution_facet,
)
from amrhg.solvers.admm_solver import admm_solve

from v0_game import build_v0_game, centralized_gne


OUT = _HERE / "out"
OUT.mkdir(exist_ok=True)

TOL_GATE = 1e-3   # kW — plan acceptance gate on |p_map − p_ADMM|


def _agg(game, p):
    """Σ p_i for a stacked decision vector."""
    return float(np.sum(p))


def main() -> None:
    print("=" * 70)
    print("v0 pipeline — simplest market GNE, iteration-free RTM")
    print("=" * 70)

    game, fleet = build_v0_game()
    N = game.N
    l_max, l_min = float(game.d[0]), float(-game.d[1])
    lam_lb, lam_ub = float(game.p_lb[0]), float(game.p_ub[0])
    print(f"\nFleet ({N} grid-only agents):")
    for s in fleet:
        print(f"  {s.name:5s}  p_max={s.p_max:6.1f} kW  η={s.eta:.3f}  "
              f"γ={s.gamma:.3f}  a=r_H2·η·1000={s.a:.1f} $/MWh")
    print(f"Coupling: {l_min:.0f} ≤ Σp ≤ {l_max:.0f} kW   |   λ ∈ [{lam_lb:.0f}, {lam_ub:.0f}] $/MWh")

    # ── 1. per-agent mpQP ────────────────────────────────────────────────────
    print("\n[1] Per-agent mpQP (PPOPT) ...")
    t0 = time.perf_counter()
    agent_sols = solve_all_agents_mp(game, verbose=True)
    t_mp = time.perf_counter() - t0
    cr_counts = [s.n_cr for s in agent_sols]
    print(f"    per-agent CRs = {cr_counts}   (offline {t_mp:.2f}s)")

    # ── 2. Mode 1 — exhaustive combiner → SINGLE-VALUED VARIATIONAL map ──────
    # build_gne_solution(select="potential") fixes the within-combo degenerate pick;
    # filter_variational then drops the overlapping non-variational corner GNE OFFLINE,
    # leaving a single-valued map.  Online = locate() + affine eval, NO cost sharing.
    print("\n[2] Mode 1 — exhaustive combiner → single-valued variational map ...")
    t0 = time.perf_counter()
    gne_full = build_gne_solution(game, agent_sols, verbose=True,
                                  equilibrium_select="potential")
    gne1 = filter_variational(gne_full, game, verbose=True)
    t_m1 = time.perf_counter() - t0
    print(f"    Mode 1: {gne_full.n_cr} GNE CRs → {gne1.n_cr} variational CRs "
          f"(offline {t_m1:.2f}s)")

    # single-valuedness check: no p covered by two kept CRs with different x*
    dup_conflicts = 0
    for lam in np.linspace(float(game.p_lb[0]), float(game.p_ub[0]), 400):
        ks = gne1.locate_all(np.array([lam]))
        if len(ks) > 1:
            xs = [gne1.regions[k].evaluate(np.array([lam])) for k in ks]
            if max(np.max(np.abs(xs[0] - xj)) for xj in xs[1:]) > 1e-6:
                dup_conflicts += 1
    print(f"    single-valued check: {dup_conflicts} conflicting-overlap points "
          f"(want 0)")

    # ── 3. Mode 3 — FACET (LP-refined neighbors + variational selection) ─────
    # FACET builds the SAME variational map via the neighbor-graph BFS (the scalable
    # path).  Both facet_gne.py multiprocessing bugs are now fixed (2026-07-10), so no
    # explicit seed / hyperplane workaround is needed.  Post-filter to variational.
    print("\n[3] Mode 3 — FACET (facet_adjacency + variational) ...")
    t0 = time.perf_counter()
    gne3 = None
    facet_res = None
    try:
        agent_sols_fac = find_all_agent_cr_neighbors(
            agent_sols, method="facet_adjacency", verbose=False)
        facet_res = build_gne_solution_facet(
            game, agent_sols_fac, verbose=False, equilibrium_select="potential")
        gne3 = filter_variational(facet_res.gne_sol, game, verbose=False)
        t_m3 = time.perf_counter() - t0
        print(f"    Mode 3: {facet_res.gne_sol.n_cr} GNE CRs → {gne3.n_cr} variational "
              f"(combos checked {facet_res.n_combos_checked}/{facet_res.n_combos_total}, "
              f"offline {t_m3:.2f}s)")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"    Mode 3 FAILED: {type(e).__name__}: {e}")

    # ── 4. cross-check over a λ sweep ────────────────────────────────────────
    print("\n[4] Cross-check over λ sweep (Mode1 ↔ Mode3 ↔ ADMM ↔ centralized) ...")
    lam_grid = np.linspace(lam_lb, lam_ub, 200)

    max_err_13   = 0.0   # Mode1 vs Mode3
    max_err_1a   = 0.0   # Mode1 vs ADMM
    max_err_1c   = 0.0   # Mode1 vs centralized
    max_resid    = 0.0   # equilibrium residual of Mode1 map
    misses_1     = 0
    misses_3     = 0
    worst = {}

    for t, lam in enumerate(lam_grid):
        p = np.array([lam])

        x1 = gne1.evaluate(p)   # PLAIN lookup on the single-valued variational map
        if x1 is None:
            misses_1 += 1
            continue
        x3 = gne3.evaluate(p) if gne3 is not None else None
        if gne3 is not None and x3 is None:
            misses_3 += 1

        res_admm = admm_solve(game, p, rho=0.5, max_iter=2000, tol=1e-7)
        xa = res_admm.x_stacked
        xc = centralized_gne(game, fleet, lam)

        e13 = np.max(np.abs(x1 - x3)) if x3 is not None else 0.0
        e1a = np.max(np.abs(x1 - xa))
        e1c = np.max(np.abs(x1 - xc))
        # equilibrium residual is cheap; check on a coarse grid
        if t % 10 == 0:
            vk  = verify_gne_at_p(p, gne1, game, agent_sols, verbose=False)
            max_resid = max(max_resid, vk.get("residual", np.nan))

        if e1a > max_err_1a:
            worst = {"lam": lam, "x1": x1, "xa": xa, "xc": xc}
        max_err_13 = max(max_err_13, e13)
        max_err_1a = max(max_err_1a, e1a)
        max_err_1c = max(max_err_1c, e1c)

    # ── report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  per-agent CRs                : {cr_counts}")
    print(f"  Mode 1 GNE CRs               : {gne1.n_cr}")
    print(f"  Mode 3 GNE CRs               : {gne3.n_cr if gne3 is not None else 'N/A (FACET failed)'}")
    if facet_res is not None:
        print(f"  FACET combos checked         : {facet_res.n_combos_checked}"
              f" / {facet_res.n_combos_total}")
    print(f"  Mode1 lookup misses          : {misses_1} / {len(lam_grid)}")
    print(f"  Mode3 lookup misses          : {misses_3} / {len(lam_grid)}")
    print(f"  max |Mode1 − Mode3|          : {max_err_13:.2e} kW")
    print(f"  max |Mode1 − ADMM|           : {max_err_1a:.2e} kW  (gate < {TOL_GATE})")
    print(f"  max |Mode1 − centralized|    : {max_err_1c:.2e} kW")
    print(f"  max equilibrium residual     : {max_resid:.2e}")
    if worst:
        print(f"  worst-λ ({worst['lam']:.2f}): "
              f"map={np.round(worst['x1'],2)}  admm={np.round(worst['xa'],2)}  "
              f"cent={np.round(worst['xc'],2)}")

    # Core gate: the explicit Mode-1 map reproduces the iterative ADMM equilibrium
    # with zero lookup misses.  Mode-3 (FACET) agreement is reported when available.
    gate_ok = (max_err_1a < TOL_GATE and max_err_1c < TOL_GATE and misses_1 == 0)
    facet_ok = (gne3 is not None and max_err_13 < TOL_GATE and misses_3 == 0)
    print("\n  CORE GATE (Mode1 == ADMM == centralized, 0 misses): "
          + ("PASS ✅" if gate_ok else "FAIL ❌"))
    print("  FACET (Mode3 == Mode1): "
          + ("PASS ✅" if facet_ok else "see findings (FACET multiprocessing bugs)"))

    # ── figure: p_i*(λ) map + aggregate with L_min/L_max ─────────────────────
    _plot_map(game, fleet, gne1, lam_lb, lam_ub, l_min, l_max)
    print(f"\n  Map figure → {OUT / 'v0_map.png'}")


def _plot_map(game, fleet, gne1, lam_lb, lam_ub, l_min, l_max):
    N = game.N
    lam_grid = np.linspace(lam_lb, lam_ub, 400)
    P = np.full((len(lam_grid), N), np.nan)
    agg = np.full(len(lam_grid), np.nan)
    for t, lam in enumerate(lam_grid):
        x = gne1.evaluate(np.array([lam]))
        if x is not None:
            P[t] = x
            agg[t] = np.sum(x)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for i, s in enumerate(fleet):
        ax1.plot(lam_grid, P[:, i], lw=2, label=f"{s.name} (p_max={s.p_max:.0f})")
    ax1.set_ylabel("agent power  p_i*  [kW]")
    ax1.set_title("v0 explicit GNE map:  p_i*(λ)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(lam_grid, agg, lw=2, color="k", label="Σ p_i*")
    ax2.axhline(l_max, color="red",   ls="--", lw=1, label=f"L_max={l_max:.0f}")
    ax2.axhline(l_min, color="blue",  ls="--", lw=1, label=f"L_min={l_min:.0f}")
    ax2.set_xlabel("real-time price  λ  [$/MWh]")
    ax2.set_ylabel("aggregate  Σp*  [kW]")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "v0_map.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
