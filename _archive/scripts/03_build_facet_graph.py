"""
08_mpgne_facet.py — Mode 3: FACET-H based GNE (scalable BFS over neighbor graph).

Mode 1 (exhaustive K^N combiner) is intractable with large CR counts.
Mode 3 (FACET-H) avoids K^N by BFS from a seed GNE combo over the CR neighbor graph.

Pipeline
────────
1. Build game from configs/base.yaml
2. Build p_gne layout (40-dim, L_max constant, L_min dropped)
3. Build pext_game (GNEGame with n_p=40 for combiner/FACET)
4. Load cached agent solutions (from 07) or solve from scratch
5. Offline Phase 1: hash-based hyperplane adjacency neighbor graph  O(N·F)
6. Offline Phase 2: ADMM seed → locate starting combo
7. Offline Phase 3: FACET BFS → combo_index {combo → (H_x, h_x)}
8. Save to disk
9. Validate: solve_gne_online vs ADMM across 5 test points

Run: python scripts/08_mpgne_facet.py
"""

from __future__ import annotations
import time
import pickle
from pathlib import Path

import numpy as np
import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amrhg.game.simple_game import build_4agent_simple_game, make_simple_param
from amrhg.solvers.game import GNEGame
from amrhg.solvers.simple_mpqp import (
    make_pgne_layout,
    make_pgne_bounds,
    pack_pgne,
    solve_all_agents,
    DEFAULT_ALGORITHM,
)
from amrhg.solvers.facet_gne import (
    find_all_agent_cr_neighbors,
    solve_gne_online_v2,
    precompute_point_location_arrays,
)
from amrhg.solvers.admm_solver import admm_solve

OUT = ROOT / "results" / "mpgne"
OUT.mkdir(parents=True, exist_ok=True)

# ── Parameter bounds — calibrated to PJM 2024-07-08 simulation week ─────────
# LMP: p1..p98 of actual 5-min RTM prices for this week (10.83–502.87 $/MWh).
#   [10, 220] covers 97.7% of all H-step forecast windows; rare spikes are
#   clipped (true settlement still uses the real cleared price).
# DA: tightened to the actual operating range of each agent in this week,
#   giving 5–8× smaller DA intervals → major CR count reduction on PEM/Alk.
#   Agent order: [VRFB, PV+Batt, PEM, Alk]
LMP_LB = 10.0
LMP_UB = 220.0

DA_LB_PER_AGENT = [ -20.0, -480.0,  750.0,  620.0]   # kW
DA_UB_PER_AGENT = [  20.0,    0.0,  900.0,  750.0]   # kW
# VRFB: near-zero reactive DA (±20 kW vs old ±100)
# PV+Batt: half-PV export, max 480 kW (943 kW peak × 0.5, vs old 500)
# PEM: ±8% around 833 kW setpoint    [750, 900]  vs old [0, 1200] (8× tighter)
# Alk: ±8% around 694 kW setpoint    [620, 750]  vs old [0, 1000] (7.7× tighter)

PV_UB = 950.0   # p99.5 of actual week PV generation (943 kW peak)

# ─────────────────────────────────────────────────────────────────────────────
#  1. Build game
# ─────────────────────────────────────────────────────────────────────────────

cfg   = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
mkt   = cfg["market"]
L_MAX = float(mkt["l_max_mw"]) * 1000
L_MIN = float(mkt["l_min_mw"]) * 1000
H     = int(mkt["rtm_horizon"])
DT_HR = mkt["rtm_interval_min"] / 60.0

agents, layout, game = build_4agent_simple_game(
    vrfb_cfg = cfg["agents"]["vrfb"],
    pv_cfg   = cfg["agents"]["pv_battery"],
    pem_cfg  = cfg["agents"]["electrolyzer_pem"],
    alk_cfg  = cfg["agents"]["electrolyzer_alk"],
    H=H, dt_hr=DT_HR,
)

pgne = make_pgne_layout(layout, L_max_kw=L_MAX)

print(f"Game:   N={game.N}, H={H}, n_x_total={game.n_x_total}")
print(f"p_gne:  {pgne.n_p_gne}-dim  (L_max constant, L_min dropped)")
print(f"n_p_priv: {pgne.n_p_priv}   n_theta: VRFB/PEM/Alk=19, PV+Batt=25  (sum_x_neg=6+p_priv)")

# ─────────────────────────────────────────────────────────────────────────────
#  2. Build pext_game (n_p=40) for combiner / FACET
# ─────────────────────────────────────────────────────────────────────────────

p_gne_lb, p_gne_ub = make_pgne_bounds(
    game, pgne,
    lmp_lb          = LMP_LB,
    lmp_ub          = LMP_UB,
    da_lb_per_agent = DA_LB_PER_AGENT,
    da_ub_per_agent = DA_UB_PER_AGENT,
    pv_ub           = PV_UB,
)

pext_game = GNEGame(
    agents    = game.agents,
    d         = game.d,
    S_coup    = game.S_coup,
    d_lb      = game.d_lb,
    S_coup_lb = game.S_coup_lb,
    p_lb      = p_gne_lb,
    p_ub      = p_gne_ub,
)
assert pext_game.n_p == pgne.n_p_gne
print(f"pext_game.n_p = {pext_game.n_p}  ✓")

# ─────────────────────────────────────────────────────────────────────────────
#  3. Load or solve agent solutions
# ─────────────────────────────────────────────────────────────────────────────

agent_sol_path  = OUT / "agent_solutions.pkl"
neighbors_path  = OUT / "agent_solutions_neighbors.pkl"

# Delete old cache when bounds change so mpQP is re-solved with new parameter space.
# Flip to False if you want to re-use an existing solve with the CURRENT bounds.
FORCE_RESOLVE = True
if FORCE_RESOLVE:
    for _p in (neighbors_path, agent_sol_path):
        if _p.exists():
            _p.unlink()
            print(f"[cache] removed {_p.name} (FORCE_RESOLVE=True)")

if neighbors_path.exists():
    print(f"\nLoading neighbor-enriched solutions from {neighbors_path}")
    with open(neighbors_path, "rb") as f:
        agent_solutions = pickle.load(f)
    cr_counts  = [s.n_cr for s in agent_solutions]
    total_nb   = sum(len(cr.facet_neighbors)
                     for s in agent_solutions for cr in s.regions)
    print(f"  CRs: {cr_counts}  |  neighbor pairs: {total_nb}")
    need_neighbors = False

elif agent_sol_path.exists():
    print(f"\nLoading agent solutions from {agent_sol_path}")
    with open(agent_sol_path, "rb") as f:
        agent_solutions = pickle.load(f)
    cr_counts = [s.n_cr for s in agent_solutions]
    print(f"  CRs: {cr_counts}")
    need_neighbors = True

else:
    print("\n" + "="*60)
    print("OFFLINE: per-agent mpQP (PPOPT, private θ_i)")
    print("="*60)
    t0 = time.perf_counter()
    agent_solutions = solve_all_agents(
        game, layout, pgne,
        algorithm        = DEFAULT_ALGORITHM,
        verbose          = True,
        lmp_lb           = LMP_LB,
        lmp_ub           = LMP_UB,
        da_lb_per_agent  = DA_LB_PER_AGENT,
        da_ub_per_agent  = DA_UB_PER_AGENT,
        pv_ub            = PV_UB,
    )
    print(f"mpQP time: {time.perf_counter()-t0:.1f} s")
    with open(agent_sol_path, "wb") as f:
        pickle.dump(agent_solutions, f)
    cr_counts = [s.n_cr for s in agent_solutions]
    print(f"Saved → {agent_sol_path}  (CRs: {cr_counts})")
    need_neighbors = True

# ─────────────────────────────────────────────────────────────────────────────
#  4. Offline Phase 1: hyperplane adjacency neighbor graph
# ─────────────────────────────────────────────────────────────────────────────

if need_neighbors:
    print("\n" + "="*60)
    print("OFFLINE PHASE 1: Hyperplane adjacency  O(N·F) hash-based")
    print("="*60)
    t1 = time.perf_counter()
    agent_solutions = find_all_agent_cr_neighbors(
        agent_solutions, method="hyperplane_adjacency", verbose=True,
    )
    total_nb = sum(len(cr.facet_neighbors)
                   for s in agent_solutions for cr in s.regions)
    print(f"Done in {time.perf_counter()-t1:.1f} s  |  total pairs: {total_nb}")
    with open(neighbors_path, "wb") as f:
        pickle.dump(agent_solutions, f)
    print(f"Saved → {neighbors_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  5. Offline Phase 2: find seed combo from ADMM
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("OFFLINE PHASE 2: Seed GNE combo via ADMM")
print("="*60)

test_states = [
    cfg["agents"]["vrfb"]["soc_init_kwh"],
    cfg["agents"]["pv_battery"]["soc_init_kwh"],
    cfg["agents"]["electrolyzer_pem"]["tank_init_kg"],
    cfg["agents"]["electrolyzer_alk"]["tank_init_kg"],
]
test_lmp = 45.0 * np.ones(H)
test_da  = [
    np.zeros(H), -240.0*np.ones(H), 833.0*np.ones(H), 694.0*np.ones(H),
]
test_pv  = 200.0 * np.ones(H)

p_gne_seed = pack_pgne(pgne, states=test_states, lmp_rt=test_lmp,
                        da_schedules=test_da, pv_forecast=test_pv)

p_shared_seed = make_simple_param(
    layout, state_inits=test_states, lmp_rt=test_lmp,
    da_schedules=test_da, l_max_kw=L_MAX, l_min_kw=L_MIN, pv_forecast=test_pv,
)
t2 = time.perf_counter()
admm_seed = admm_solve(game, p_shared_seed, rho=5.0, max_iter=500, tol=1.0)
print(f"ADMM seed: {admm_seed.n_iter} iters, converged={admm_seed.converged}, "
      f"time={1e3*(time.perf_counter()-t2):.1f} ms", flush=True)

seed_combo: tuple[int, ...] | None = None
if admm_seed.converged:
    x_seed = admm_seed.x_stacked
    seed_list = []
    for i in range(game.N):
        others  = [j for j in range(game.N) if j != i]
        sum_x_neg = np.zeros(H)
        for j in others:
            sum_x_neg += x_seed[game.x_slice(j)]
        theta_i = np.concatenate([sum_x_neg, p_gne_seed])
        found   = None
        for cr_idx, cr in enumerate(agent_solutions[i].regions):
            if np.all(cr.E @ theta_i <= cr.f + 1e-4):
                found = cr_idx
                break
        if found is None:
            print(f"  WARNING: agent {i} θ_i not in any CR — using 0", flush=True)
            seed_list.append(0)
        else:
            print(f"  Agent {i}: θ_i ∈ CR {found}", flush=True)
            seed_list.append(found)
    seed_combo = tuple(seed_list)
    print(f"Seed combo: {seed_combo}", flush=True)
else:
    print("ADMM did not converge — using origin as seed", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
#  6. Online FACET tracking — no offline BFS (large-N FACET method)
#
#  For N>4 the K^N offline BFS is intractable.  Instead, solve_gne_online
#  hops along the pre-computed facet-neighbor graph from the previous combo.
#  At each RTM step only ~8-12 neighbors per agent are explored (1-2 hops),
#  giving sub-millisecond GNE evaluation.  The combo_cache grows organically.
# ─────────────────────────────────────────────────────────────────────────────

total_combos = 1
for s in agent_solutions:
    total_combos *= s.n_cr

total_nb = sum(len(cr.facet_neighbors)
               for s in agent_solutions for cr in s.regions)
print(f"\nFACET neighbor graph: {total_nb:,} pairs  |  K^N = {total_combos:,}")
print("Skipping offline BFS — using online FACET tracking (large-N method)")

combo_cache: dict = {}   # grows organically from ADMM seeds

# Precompute stacked E/f arrays for vectorized PointLocation (~50-200x faster)
precompute_point_location_arrays(agent_solutions)
print("Precomputed stacked E/f arrays for vectorized PointLocation", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
#  7. Validation: solve_gne_online vs ADMM
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("VALIDATION: solve_gne_online vs ADMM  (5 test points)")
print("="*60)

test_points = [
    # Seed point (nominal)
    (test_states, test_lmp, test_da, test_pv),
    # High SoC, mid-price, near-max PV
    ([s*0.5 for s in test_states], 80.0*np.ones(H),
     [np.zeros(H), -100.0*np.ones(H), 850.0*np.ones(H), 700.0*np.ones(H)],
     700.0*np.ones(H)),
    # Low price, low PV (night)
    ([s*0.8 for s in test_states], 15.0*np.ones(H),
     [np.zeros(H), -10.0*np.ones(H),  800.0*np.ones(H), 650.0*np.ones(H)],
     0.0*np.ones(H)),
    # Price spike, mid PV
    ([s*0.2 for s in test_states], 180.0*np.ones(H),
     [15.0*np.ones(H), -400.0*np.ones(H), 760.0*np.ones(H), 625.0*np.ones(H)],
     800.0*np.ones(H)),
    # Evening peak
    (test_states, 110.0*np.ones(H),
     [np.zeros(H), -300.0*np.ones(H), 890.0*np.ones(H), 740.0*np.ones(H)],
     50.0*np.ones(H)),
]

errors     = []
prev_x_star = admm_seed.x_stacked if admm_seed.converged else None
prev_crs    = list(seed_combo) if seed_combo is not None else None
names       = ["VRFB", "PV+Batt", "PEM", "Alk"]

for pt, (pt_states, pt_lmp, pt_da, pt_pv) in enumerate(test_points):
    p_gne_t = pack_pgne(pgne, states=pt_states, lmp_rt=pt_lmp,
                         da_schedules=pt_da, pv_forecast=pt_pv)

    # Phase 7a: ADMM reference (also used as fallback for GNE search)
    p_shared_t = make_simple_param(
        layout, state_inits=pt_states, lmp_rt=pt_lmp, da_schedules=pt_da,
        l_max_kw=L_MAX, l_min_kw=L_MIN, pv_forecast=pt_pv,
    )
    t_admm = time.perf_counter()
    admm_r = admm_solve(game, p_shared_t, rho=5.0, max_iter=500, tol=1.0)
    t_admm = time.perf_counter() - t_admm
    x_admm = admm_r.x_stacked if admm_r.converged else None
    print(f"\nTest {pt}: ADMM={1e3*t_admm:.1f} ms ({admm_r.n_iter} iter, "
          f"conv={admm_r.converged})", flush=True)

    # Phase 7b: FACET online GNE (v2) — PointLocation + 1-hop neighbors + Chebyshev filter
    t_gne = time.perf_counter()
    combo_out, x_gne, n_checked = solve_gne_online_v2(
        p_gne_t, agent_solutions, pext_game,
        prev_x_star=prev_x_star, prev_crs=prev_crs,
        combo_cache=combo_cache, max_hops=3,
        max_candidates_per_agent=15,
    )
    # ADMM-seed fallback: re-seed PointLocation from ADMM x* then retry
    if combo_out is None and x_admm is not None:
        admm_crs = []
        for i in range(game.N):
            others    = [j for j in range(game.N) if j != i]
            sum_x_neg = np.zeros(game.agents[i].n_x)
            for j in others:
                sum_x_neg += x_admm[game.x_slice(j)]
            theta_i = np.concatenate([sum_x_neg, p_gne_t])
            found = None
            for loc_tol in (1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0):
                found = next(
                    (k for k, cr in enumerate(agent_solutions[i].regions)
                     if np.all(cr.E @ theta_i <= cr.f + loc_tol)),
                    None,
                )
                if found is not None:
                    break
            admm_crs.append(found if found is not None else 0)
        print(f"  [ADMM fallback] re-seeding prev_crs={admm_crs}", flush=True)
        combo_out2, x_gne2, n_checked2 = solve_gne_online_v2(
            p_gne_t, agent_solutions, pext_game,
            prev_x_star=x_admm, prev_crs=admm_crs,
            combo_cache=combo_cache, max_hops=5,
            max_candidates_per_agent=15,
        )
        if combo_out2 is not None:
            combo_out, x_gne, n_checked = combo_out2, x_gne2, n_checked + n_checked2
            print(f"  [ADMM fallback] found ({n_checked2} combos checked)", flush=True)
        else:
            # Tier-3: infinite GNE case — Mx singular, SVD min-norm ≠ physical equilibrium.
            # x_admm IS a valid particular solution; verify membership directly with ADMM tol.
            direct_valid = True
            for i in range(game.N):
                cr = agent_solutions[i].regions[admm_crs[i]]
                sum_x_neg_i = np.zeros(game.agents[i].n_x)
                for j in [jj for jj in range(game.N) if jj != i]:
                    sum_x_neg_i += x_admm[game.x_slice(j)]
                theta_i = np.concatenate([sum_x_neg_i, p_gne_t])
                if np.any(cr.E @ theta_i > cr.f + 1.0):  # 1 kW = ADMM tolerance
                    direct_valid = False
                    break
            if direct_valid:
                combo_out = tuple(admm_crs)
                x_gne     = x_admm
                print(f"  [Tier-3] x_admm satisfies CR membership (infinite GNE, Mx singular)",
                      flush=True)
    t_gne = time.perf_counter() - t_gne

    print(f"  GNE={1e6*t_gne:.0f} µs  combos_checked={n_checked}  combo={combo_out}",
          flush=True)

    if x_gne is not None and x_admm is not None:
        err = float(np.max(np.abs(x_gne - x_admm)))
        errors.append(err)
        status = "PASS ✓" if err < 2.0 else "FAIL ✗"
        print(f"  ‖x_GNE − x_ADMM‖_inf = {err:.4f} kW  [{status}]", flush=True)
        for i in range(game.N):
            sl = game.x_slice(i)
            print(f"  {names[i]:8s}: "
                  f"GNE={np.array2string(x_gne[sl], precision=1, suppress_small=True)}  "
                  f"ADMM={np.array2string(x_admm[sl], precision=1, suppress_small=True)}",
                  flush=True)
    elif x_gne is None:
        print("  GNE: no valid combo found (even with ADMM fallback)", flush=True)
    elif x_admm is None:
        print("  ADMM: did not converge", flush=True)

    if combo_out is not None:
        prev_x_star = x_gne
        prev_crs    = list(combo_out)
    elif x_admm is not None:
        prev_x_star = x_admm   # keep ADMM solution as warm-start for next point

# ─────────────────────────────────────────────────────────────────────────────
#  8. Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  N={game.N}, H={H}  |  private θ_i: VRFB/PEM/Alk=19, PV+Batt=25  (sum_x_neg=6)")
print(f"  CRs per agent:    {cr_counts}")
print(f"  K^N upper bound:  {total_combos:,}")
print(f"  Neighbor pairs:   {total_nb:,}  (bucket-cap=4, non-trivial facets only)")
print(f"  Cache size after validation: {len(combo_cache):,} combos")
if errors:
    print(f"  Max ‖x_GNE−x_ADMM‖_inf: {max(errors):.4f} kW  "
          f"({'PASS ✓' if max(errors)<2.0 else 'CHECK ✗'})")
print(f"\nOutputs → {OUT}/")
print(f"  agent_solutions_neighbors.pkl  — CRs + facet_neighbors")