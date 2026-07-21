"""
v3_test.py — H₂ cumulative demand + DAM anchor: the horizon becomes a REAL θ dimension.

Deliverables (per PLAN_mp_focapo_rtm.md, v3):
  1. Build the H-step v3 game (per-agent H₂ demand + p_DA anchor, ERCOT 15-min, H=2) →
     per-agent mpQP → single-valued variational map (Mode 1 KKT filter) and FACET (Mode 3).
  2. Validate the map (PLAIN lookup, no costs online) == ADMM == centralized QP over
     RANDOM θ in the n_p=7 box (7-D → sample, not grid).  Report CRs / conflicts /
     misses / max error.
  3. Prove the horizon now COUPLES the steps: cross-step sensitivity ∂p_0/∂λ_1 ≠ 0 at a
     demand-binding θ (it was EXACTLY 0 in v1/v2, where the map decoupled per step).
  4. Figure: as H₂ demand rises the agent buys more and preferentially in the CHEAP
     step (impossible for a per-step map) — map certified against ADMM.

Run:  python simple_game/v3_test.py
Outputs: table + simple_game/out/v3_map.png
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
from amrhg.solvers.gne_combiner import build_gne_solution, filter_variational_kkt
from amrhg.solvers.facet_gne import find_all_agent_cr_neighbors, build_gne_solution_facet
from amrhg.solvers.admm_solver import admm_solve

from v3_game import build_v3_game, centralized_qp

OUT = _HERE / "out"; OUT.mkdir(exist_ok=True)
TOL_GATE = 1e-3
RNG = np.random.default_rng(7)


def _p(game, meta, x, i, k):
    """grid import p_{i,k} for agent i, step k, from stacked x."""
    return x[game.x_slice(i).start + k]


def _rand_theta(game, n):
    lb, ub = np.asarray(game.p_lb), np.asarray(game.p_ub)
    return lb + RNG.random((n, len(lb))) * (ub - lb)


def main():
    print("=" * 74)
    print("v3 — H₂ cumulative demand + DAM anchor: horizon becomes a real θ dimension")
    print("=" * 74)

    H = 2
    game, meta = build_v3_game(H=H)
    N, R = meta.N, meta.R
    l_max, l_min = float(game.d[0]), float(-game.d[H])
    lam_lb, lam_ub = float(game.p_lb[0]), float(game.p_ub[0])
    print(f"\nFleet: {[s.name for s in meta.specs]}  H={H} (ERCOT 15-min → 30-min window)")
    print(f"n_p = {game.n_p}  θ = [λ({H}), g({R*H}), D({N})]")
    print(f"Coupling per step: {l_min:.0f} ≤ Σp_k ≤ {l_max:.0f} | λ∈[{lam_lb:.0f},{lam_ub:.0f}]")
    print(f"H₂ demand ranges D_i∈[0, {[s.d_max for s in meta.specs]}] kg  (per-agent, +N θ)")
    print(f"p_DA anchor (baked constant): {[s.p_da for s in meta.specs]}")

    # ── 1. per-agent mpQP → variational maps ──────────────────────────────────
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

    print("[3] Mode 3 — FACET single-valued variational map (KKT filter) ...")
    gne3 = None
    try:
        t0 = time.perf_counter()
        sf = find_all_agent_cr_neighbors(sols, method="facet_adjacency", verbose=False)
        fr = build_gne_solution_facet(game, sf, verbose=False, equilibrium_select="potential")
        gne3 = filter_variational_kkt(fr.gne_sol, game, verbose=False)
        print(f"    FACET {fr.gne_sol.n_cr} → {gne3.n_cr} variational "
              f"(checked {fr.n_combos_checked}/{fr.n_combos_total}, {time.perf_counter()-t0:.2f}s)")
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"    Mode 3 FAILED: {e}")

    # ── 4. validate over RANDOM θ (n_p=7) — plain lookup vs ADMM & centralized QP ─
    print("\n[4] Cross-check over random θ (plain lookup vs ADMM & centralized QP) ...")
    thetas = _rand_theta(game, 350)
    e_admm = e_cqp = e13 = 0.0
    misses = conf = n_ok = 0
    worst = None
    for th in thetas:
        ks = gne1.locate_all(th)
        x1 = gne1.evaluate(th)
        xq = centralized_qp(game, th)          # exact variational reference
        if xq is None:                          # infeasible θ (shouldn't happen in box)
            continue
        if x1 is None:
            misses += 1; continue
        n_ok += 1
        # single-valuedness: overlapping CRs must agree
        if len(ks) > 1:
            xs = [gne1.regions[k].evaluate(th) for k in ks]
            if max(np.max(np.abs(xs[0] - xj)) for xj in xs[1:]) > 1e-6:
                conf += 1
        xa = admm_solve(game, th, rho=0.5, max_iter=5000, tol=1e-9).x_stacked
        ea = float(np.max(np.abs(x1 - xa))); eq = float(np.max(np.abs(x1 - xq)))
        if ea > e_admm:
            e_admm = ea; worst = (th, x1, xa, xq)
        e_cqp = max(e_cqp, eq)
        if gne3 is not None:
            x3 = gne3.evaluate(th)
            if x3 is not None:
                e13 = max(e13, float(np.max(np.abs(x1 - x3))))

    # ── 5. horizon-coupling proof: ∂p_0/∂λ_1 ≠ 0 at a demand-binding θ ─────────
    print("[5] Horizon-coupling check: does step-0 buy depend on step-1 price? ...")
    th_b = np.zeros(game.n_p)
    th_b[meta.lam_col(0)] = 65.0; th_b[meta.lam_col(1)] = 65.0     # both above a_PEM=60
    for r in range(R):
        for k in range(H):
            th_b[meta.g_col(r, k)] = 40.0
    for i in range(N):
        th_b[meta.d_col(i)] = meta.specs[i].d_max                  # demand at max → binding
    x_lo = gne1.evaluate(th_b.copy())
    th_hi = th_b.copy(); th_hi[meta.lam_col(1)] = 78.0             # raise ONLY step-1 price
    x_hi = gne1.evaluate(th_hi)
    dp0 = _p(game, meta, x_hi, 0, 0) - _p(game, meta, x_lo, 0, 0)
    dp1 = _p(game, meta, x_hi, 0, 1) - _p(game, meta, x_lo, 0, 1)
    coupled = abs(dp0) > 1e-4
    print(f"    raise λ_1 65→78 (PEM):  Δp_0={dp0:+.3f} kW, Δp_1={dp1:+.3f} kW")
    print(f"    → step-0 decision {'DOES' if coupled else 'does NOT'} respond to step-1 price "
          f"({'horizon COUPLES ✅' if coupled else 'still decoupled ❌'})")

    # ── results ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    print(f"  per-agent CRs                  : {[s.n_cr for s in sols]}")
    print(f"  Mode 1 variational CRs         : {gne1.n_cr}")
    print(f"  Mode 3 (FACET) variational CRs : {gne3.n_cr if gne3 is not None else 'N/A'}"
          f"  ({'== Mode 1' if gne3 is not None and gne3.n_cr == gne1.n_cr else 'differs'})")
    print(f"  random θ checked               : {n_ok} (misses {misses})")
    print(f"  conflicting overlaps           : {conf} / {n_ok} (single-valued if 0)")
    print(f"  max |Mode1 − Mode3|            : {e13:.2e} kW")
    print(f"  max |map − centralized QP|     : {e_cqp:.2e} kW  (gate < {TOL_GATE})")
    print(f"  max |map − ADMM|               : {e_admm:.2e} kW  (gate < {TOL_GATE})")
    if worst:
        th, x1, xa, xq = worst
        print(f"  worst-vs-ADMM θ=λ{np.round(th[:H],1)} g{np.round(th[H:H+R*H],1)} "
              f"D{np.round(th[H+R*H:],1)}")
    gate = (e_admm < TOL_GATE and e_cqp < TOL_GATE and misses == 0 and conf == 0 and coupled)
    print("\n  GATE (single-valued, plain lookup == ADMM == centralized, 0 misses, "
          "horizon couples): " + ("PASS ✅" if gate else "FAIL ❌"))

    _plot(game, meta, gne1)
    print(f"\n  Map figure → {OUT / 'v3_map.png'}")


def _plot(game, meta, gmap):
    """Left: H₂ demand shifts buying to the cheap step (map vs ADMM).
       Right: p_PEM,0 over (λ_0,λ_1) at binding demand — tilted ⇒ cross-step coupling."""
    H = meta.H
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ── Left: sweep PEM demand with λ_0 < λ_1 (cheap now, dear later) ──────────
    lam0, lam1 = 40.0, 70.0
    base = np.zeros(game.n_p)
    base[meta.lam_col(0)] = lam0; base[meta.lam_col(1)] = lam1
    for r in range(meta.R):
        for k in range(H): base[meta.g_col(r, k)] = 30.0
    base[meta.d_col(1)] = meta.specs[1].d_max * 0.5
    for ri in meta.ren_indices: base[meta.d_col(ri)] = meta.specs[ri].d_max * 0.5
    Ds = np.linspace(0.0, meta.specs[0].d_max, 40)
    p0m, p1m, p0a, p1a = [], [], [], []
    for D in Ds:
        th = base.copy(); th[meta.d_col(0)] = D
        x = gmap.evaluate(th)
        p0m.append(_p(game, meta, x, 0, 0)); p1m.append(_p(game, meta, x, 0, 1))
        xa = admm_solve(game, th, rho=0.5, max_iter=5000, tol=1e-9).x_stacked
        p0a.append(_p(game, meta, xa, 0, 0)); p1a.append(_p(game, meta, xa, 0, 1))
    ax = axs[0]
    ax.plot(Ds, p0m, "-", color="tab:blue", lw=2, label=f"p₀ (cheap step, λ={lam0:.0f})")
    ax.plot(Ds, p1m, "-", color="tab:red", lw=2, label=f"p₁ (dear step, λ={lam1:.0f})")
    ax.plot(Ds, p0a, "o", color="tab:blue", ms=3, mfc="none")
    ax.plot(Ds, p1a, "o", color="tab:red", ms=3, mfc="none")
    ax.set_xlabel("PEM H₂ demand  D  [kg]"); ax.set_ylabel("grid buy  p  [kW]")
    ax.set_title("H₂ demand shifts buying to the cheap step\n(lines = explicit map, ○ = ADMM)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

    # ── Right: p_PEM,0 over (λ_0,λ_1) at binding demand ───────────────────────
    ng = 80
    l0 = np.linspace(0, 80, ng); l1 = np.linspace(0, 80, ng)
    L0, L1 = np.meshgrid(l0, l1)
    Z = np.full_like(L0, np.nan)
    thb = np.zeros(game.n_p)
    for r in range(meta.R):
        for k in range(H): thb[meta.g_col(r, k)] = 30.0
    for i in range(meta.N): thb[meta.d_col(i)] = meta.specs[i].d_max
    for a in range(ng):
        for b in range(ng):
            th = thb.copy(); th[meta.lam_col(0)] = L0[a, b]; th[meta.lam_col(1)] = L1[a, b]
            x = gmap.evaluate(th)
            if x is not None:
                Z[a, b] = _p(game, meta, x, 0, 0)
    ax = axs[1]
    pc = ax.pcolormesh(L0, L1, Z, cmap="viridis", shading="auto")
    fig.colorbar(pc, ax=ax, fraction=0.046, label="PEM step-0 buy  p₀  [kW]")
    ax.plot([0, 80], [0, 80], "w--", lw=0.8, alpha=0.6)
    ax.set_xlabel("step-0 price  λ₀"); ax.set_ylabel("step-1 price  λ₁")
    ax.set_title("p₀ depends on λ₁ (tilted regions)\n⇒ horizon couples (flat bands in v1/v2)",
                 fontsize=10)
    fig.suptitle("v3 explicit variational GNE map — H₂ cumulative demand couples the horizon",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "v3_map.png", dpi=125)
    plt.close(fig)


if __name__ == "__main__":
    main()
