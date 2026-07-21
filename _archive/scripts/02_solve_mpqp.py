"""
07_mpgne_solve.py — Mode 1: per-agent mpQP + exhaustive GNE combiner.

With corrected private θ_i (31/37-dim), CR counts should be much lower than
the 26K seen with the wrong 58-dim formulation.  If K^N is still too large,
switch to script 08 (FACET-H BFS).

Pipeline
────────
1. Build 4-agent simple game from configs/base.yaml
2. Build p_gne layout (40-dim, L_max fixed constant, L_min dropped)
3. Build pext_game (GNEGame with n_p=40 for combiner)
4. Offline: solve each agent's mpQP via PPOPT  (private 31/37-dim θ_i)
   → PPOPT solves in private space, CRs expanded to 58-dim for combiner
5. Save AgentSolutions to disk
6. Offline: exhaustive GNE combiner (Mode 1) — only run if K^N is feasible
7. Sanity check: GNE map vs ADMM at a test parameter

Run: python scripts/07_mpgne_solve.py
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
from amrhg.solvers.gne_combiner import build_gne_solution
from amrhg.solvers.admm_solver import admm_solve

OUT = ROOT / "results" / "mpgne"
OUT.mkdir(parents=True, exist_ok=True)

LMP_LB = 0.0    # $/MWh — tight bounds to avoid CR explosion
LMP_UB = 250.0

# ─────────────────────────────────────────────────────────────────────────────
#  1. Build game
# ─────────────────────────────────────────────────────────────────────────────

cfg   = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
mkt   = cfg["market"]
L_MAX = float(mkt["l_max_mw"]) * 1000   # kW
H     = int(mkt["rtm_horizon"])          # 6
DT_HR = mkt["rtm_interval_min"] / 60.0  # 1/12

agents, layout, game = build_4agent_simple_game(
    vrfb_cfg = cfg["agents"]["vrfb"],
    pv_cfg   = cfg["agents"]["pv_battery"],
    pem_cfg  = cfg["agents"]["electrolyzer_pem"],
    alk_cfg  = cfg["agents"]["electrolyzer_alk"],
    H=H, dt_hr=DT_HR,
)

pgne = make_pgne_layout(layout, L_max_kw=L_MAX)

print(f"Game:   N={game.N}, H={H}, n_x_total={game.n_x_total}")
print(f"p_gne:  {pgne.n_p_gne}-dim  (L_max={L_MAX} kW constant, L_min dropped)")
print(f"n_p_priv per agent: {pgne.n_p_priv}  (VRFB/PEM/Alk=13, PV+Batt=19)")
print(f"n_theta per agent:  VRFB/PEM/Alk=19, PV+Batt=25  (sum_x_neg=6 + p_priv)")
print(f"Expanded θ_i for combiner: 46-dim  [sum_x_neg=6, p_gne=40]")

# ─────────────────────────────────────────────────────────────────────────────
#  2. Build pext_game (n_p=40) for gne_combiner
#
#  combiner splits cr.A = [A_i_x | A_i_p] using game.n_p.
#  Expanded CRs have A_i_p ∈ ℝ^{6×40}, so we need game.n_p=40.
# ─────────────────────────────────────────────────────────────────────────────

p_gne_lb, p_gne_ub = make_pgne_bounds(game, pgne, lmp_lb=LMP_LB, lmp_ub=LMP_UB)

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
print(f"\npext_game.n_p = {pext_game.n_p}  ✓")

# ─────────────────────────────────────────────────────────────────────────────
#  3. Offline: per-agent mpQP
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("OFFLINE: per-agent mpQP (PPOPT, private θ_i)")
print("="*60)

t0 = time.perf_counter()
agent_solutions = solve_all_agents(
    game, layout, pgne,
    algorithm = DEFAULT_ALGORITHM,
    verbose   = True,
    lmp_lb    = LMP_LB,
    lmp_ub    = LMP_UB,
)
t_mpqp = time.perf_counter() - t0
print(f"\nPer-agent mpQP time: {t_mpqp:.1f} s")

sol_path = OUT / "agent_solutions.pkl"
with open(sol_path, "wb") as f:
    pickle.dump(agent_solutions, f)
print(f"Saved → {sol_path}")

cr_counts   = [s.n_cr for s in agent_solutions]
total_combos = int(np.prod(cr_counts))
print(f"\nCRs per agent: {cr_counts}")
print(f"K^N upper bound: {total_combos:,}  (N={game.N})")

# ─────────────────────────────────────────────────────────────────────────────
#  4. Exhaustive GNE combiner — only if K^N is tractable
# ─────────────────────────────────────────────────────────────────────────────

COMBINER_LIMIT = 1_000_000   # skip if more than 1M combinations

if total_combos <= COMBINER_LIMIT:
    print("\n" + "="*60)
    print(f"OFFLINE: GNE combiner  ({total_combos:,} combinations)")
    print("="*60)

    t1 = time.perf_counter()
    gne_solution = build_gne_solution(pext_game, agent_solutions)
    t_gne = time.perf_counter() - t1

    print(f"GNE combiner time: {t_gne:.1f} s")
    print(f"GNE critical regions: {gne_solution.n_cr}")

    gne_path = OUT / "gne_solution.pkl"
    with open(gne_path, "wb") as f:
        pickle.dump(gne_solution, f)
    print(f"Saved → {gne_path}")
else:
    print(f"\nK^N={total_combos:,} exceeds limit ({COMBINER_LIMIT:,}) — "
          f"skipping exhaustive combiner.  Use script 08 (FACET-H BFS).")
    gne_solution = None

# ─────────────────────────────────────────────────────────────────────────────
#  5. Sanity check: GNE map vs ADMM
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SANITY CHECK: GNE map vs ADMM at test parameter")
print("="*60)

test_states = [
    cfg["agents"]["vrfb"]["soc_init_kwh"],
    cfg["agents"]["pv_battery"]["soc_init_kwh"],
    cfg["agents"]["electrolyzer_pem"]["tank_init_kg"],
    cfg["agents"]["electrolyzer_alk"]["tank_init_kg"],
]
test_lmp = 45.0 * np.ones(H)
test_da  = [
    np.zeros(H),
    -300.0 * np.ones(H),
    800.0  * np.ones(H),
    694.0  * np.ones(H),
]
test_pv  = 200.0 * np.ones(H)

p_gne_test = pack_pgne(pgne, states=test_states, lmp_rt=test_lmp,
                        da_schedules=test_da, pv_forecast=test_pv)

# ADMM reference
p_shared = make_simple_param(
    layout,
    state_inits  = test_states,
    lmp_rt       = test_lmp,
    da_schedules = test_da,
    l_max_kw     = L_MAX,
    l_min_kw     = float(cfg["market"]["l_min_mw"]) * 1000,
    pv_forecast  = test_pv,
)
t2 = time.perf_counter()
result = admm_solve(game, p_shared, rho=5.0, max_iter=500, tol=1.0)
t_admm = time.perf_counter() - t2

x_admm = result.x_stacked if result.converged else None
print(f"ADMM: {result.n_iter} iters, converged={result.converged}, "
      f"time={t_admm*1e3:.1f} ms")

if x_admm is not None:
    names = ["VRFB", "PV+Batt", "PEM", "Alk"]
    for i in range(game.N):
        sl = game.x_slice(i)
        print(f"  {names[i]:8s}: {np.array2string(x_admm[sl], precision=1, suppress_small=True)}")

# GNE map lookup (if combiner ran)
if gne_solution is not None and gne_solution.n_cr > 0:
    x_gne = gne_solution.evaluate(p_gne_test)
    if x_gne is not None and x_admm is not None:
        err = np.max(np.abs(x_gne - x_admm))
        print(f"\n‖x_GNE − x_ADMM‖_inf = {err:.4f} kW  "
              f"({'PASS ✓' if err < 1.0 else 'FAIL ✗'})")
    elif x_gne is None:
        print("\nGNE map: p_gne outside precomputed CRs")

# ─────────────────────────────────────────────────────────────────────────────
#  6. Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  N={game.N} agents, H={H}")
print(f"  Private θ_i dims:  VRFB/PEM/Alk=19, PV+Batt=25  (sum_x_neg=6 + p_priv)")
print(f"  Expanded θ_i dims: 46  [sum_x_neg=6, p_gne=40]")
print(f"  CRs per agent:     {cr_counts}")
print(f"  K^N upper bound:   {total_combos:,}")
print(f"  mpQP time:         {t_mpqp:.1f} s")
if gne_solution is not None:
    print(f"  GNE CRs:           {gne_solution.n_cr}")
print(f"\nOutputs saved to: {OUT}/")