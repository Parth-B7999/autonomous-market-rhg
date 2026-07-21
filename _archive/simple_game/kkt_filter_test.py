"""
kkt_filter_test.py — Validate the rigorous single-valued variational map
(filter_variational_kkt) for n_p = 1 (v0) and n_p = 2 (v2, the case the
Chebyshev-centre filter could not make single-valued).

Checks, per game, over a θ grid:
  • the KKT-filtered map is SINGLE-VALUED (no conflicting overlaps), so plain
    locate() suffices online — no costs, no exchange;
  • plain locate()+affine == ADMM (ground truth) to < 1e-3;
  • also via FACET (Mode 3).

Run:  python simple_game/kkt_filter_test.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
for pth in (str(_HERE.parent / "src"), str(_HERE)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from amrhg.solvers.mp_solver import solve_all_agents_mp
from amrhg.solvers.gne_combiner import (
    build_gne_solution, filter_variational, filter_variational_kkt,
)
from amrhg.solvers.facet_gne import find_all_agent_cr_neighbors, build_gne_solution_facet
from amrhg.solvers.admm_solver import admm_solve

from v0_game import build_v0_game
from v2_game import build_v2_step_game


def _grid(game, n):
    axes = [np.linspace(float(game.p_lb[k]), float(game.p_ub[k]), n)
            for k in range(game.n_p)]
    mesh = np.meshgrid(*axes)
    return np.stack([m.ravel() for m in mesh], axis=1)


def _conflicts(gmap, thetas):
    c = 0
    for th in thetas:
        ks = gmap.locate_all(th)
        if len(ks) > 1:
            xs = [gmap.regions[k].evaluate(th) for k in ks]
            if max(np.max(np.abs(xs[0] - xj)) for xj in xs[1:]) > 1e-6:
                c += 1
    return c


def _vs_admm(gmap, game, thetas):
    err = 0.0; misses = 0
    for th in thetas:
        x = gmap.evaluate(th)
        if x is None:
            misses += 1; continue
        xa = admm_solve(game, th, rho=0.5, max_iter=3000, tol=1e-8).x_stacked
        err = max(err, float(np.max(np.abs(x - xa))))
    return err, misses


def run(tag, game, n_grid):
    print(f"\n{'='*60}\n{tag}  (n_p={game.n_p})\n{'='*60}")
    sols = solve_all_agents_mp(game, verbose=False)
    full = build_gne_solution(game, sols, verbose=False, equilibrium_select="potential")
    thetas = _grid(game, n_grid)

    old = filter_variational(full, game, verbose=False)
    kkt = filter_variational_kkt(full, game, verbose=False)
    print(f"  full GNE CRs        : {full.n_cr}")
    print(f"  Chebyshev-filter CRs: {old.n_cr}  | conflicts: {_conflicts(old, thetas)}/{len(thetas)}")
    print(f"  KKT-filter CRs      : {kkt.n_cr}  | conflicts: {_conflicts(kkt, thetas)}/{len(thetas)}")

    e, m = _vs_admm(kkt, game, thetas)
    print(f"  KKT map (plain lookup) vs ADMM: max|Δ|={e:.2e} kW, misses={m}/{len(thetas)}")

    # FACET (Mode 3) with the same KKT filter
    sf = find_all_agent_cr_neighbors(sols, method="facet_adjacency", verbose=False)
    fr = build_gne_solution_facet(game, sf, verbose=False, equilibrium_select="potential")
    kkt3 = filter_variational_kkt(fr.gne_sol, game, verbose=False)
    e3, m3 = _vs_admm(kkt3, game, thetas)
    print(f"  FACET KKT CRs       : {kkt3.n_cr}  | conflicts: {_conflicts(kkt3, thetas)}/{len(thetas)}")
    print(f"  FACET KKT vs ADMM   : max|Δ|={e3:.2e} kW, misses={m3}/{len(thetas)}")

    ok = (_conflicts(kkt, thetas) == 0 and e < 1e-3 and m == 0
          and _conflicts(kkt3, thetas) == 0 and e3 < 1e-3 and m3 == 0)
    print(f"  → {tag}: " + ("PASS ✅" if ok else "FAIL ❌"))
    return ok


def main():
    r0 = run("v0 single-step", build_v0_game()[0], n_grid=200)
    r2 = run("v2 renewable (λ,g)", build_v2_step_game()[0], n_grid=31)
    print("\n" + "=" * 60)
    print("OVERALL: " + ("ALL PASS ✅" if (r0 and r2) else "SOME FAIL ❌"))


if __name__ == "__main__":
    main()
