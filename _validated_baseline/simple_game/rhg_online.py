"""
rhg_online.py — ONLINE FACET GNE for the H=4 receding-horizon game.

Uses the cached per-agent solutions (rhg_offline.py).  Per query θ (public p_gne,
24-dim) it locates each agent's CR (FACET point-location, warm-started), gathers
current+1-hop neighbor combos, solves each via the equilibrium linear system, and
returns the VARIATIONAL (min-potential) feasible GNE.

Heterogeneous n_x handled correctly: sum_x_neg_i = Σ_{j≠i} p_j (per step) uses ONLY the
first H (grid-import) components of each agent's x_j — not the full x_j.
"""
from __future__ import annotations
import sys, pickle, itertools
from collections import deque
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
import rhg_mpqp as R
from amrhg.solvers.gne_combiner import _stacked_cost, _centralized_constraints, _solve_equilibrium
from amrhg.solvers.facet_gne import (find_all_agent_cr_neighbors,
                                     precompute_point_location_arrays, _locate_cr_fast)
from amrhg.solvers.admm_solver import admm_solve

H = R.H


def _assemble_rhg(combo, sols, game):
    """Equilibrium system M_x x = M_p θ + M_1, sum-mode with HETEROGENEOUS n_x:
    sum_x_neg = Σ_{j≠i} p_j maps to the FIRST H (grid-import) columns of each x_j."""
    N = game.N; ntot = game.n_x_total; npar = game.n_p
    Mx = np.zeros((ntot, ntot)); Mp = np.zeros((ntot, npar)); M1 = np.zeros(ntot)
    for i, ji in enumerate(combo):
        cr = sols[i][ji]; nxi = game.agents[i].n_x
        nxneg = cr.A.shape[1] - npar          # = H (sum mode)
        A_x = cr.A[:, :nxneg]; A_p = cr.A[:, nxneg:]
        rs = game.x_slice(i).start; re = game.x_slice(i).stop
        Mx[rs:re, rs:re] = np.eye(nxi)
        for j in range(N):
            if j == i:
                continue
            cs = game.x_slice(j).start
            Mx[rs:re, cs:cs + H] += -A_x       # sum_x_neg → p-part of x_j
        Mp[rs:re, :] = A_p; M1[rs:re] = cr.b
    return Mx, Mp, M1


def load_and_prepare():
    sols = pickle.load(open(HERE / "out" / "rhg_agent_sols.pkl", "rb"))
    precompute_point_location_arrays(sols)
    game = R.build_pgne_game()
    return sols, game


def _sumneg(game, x, i):
    """sum_x_neg_i(k) = Σ_{j≠i} p_{j,k}  (first H of each x_j)."""
    s = np.zeros(H)
    for j in range(game.N):
        if j != i:
            s += x[game.x_slice(j).start: game.x_slice(j).start + H]
    return s


def _locate_all(theta, sols, game, x, crs):
    out = []
    for i in range(game.N):
        ti = np.concatenate([_sumneg(game, x, i), theta])
        out.append(_locate_cr_fast(sols[i], ti, 1e-6, hint_idx=(crs[i] if crs else None)))
    return out


def _solve_combo(combo, theta, sols, game):
    Mx, Mp, M1 = _assemble_rhg(combo, sols, game)
    try:
        return np.linalg.solve(Mx, Mp @ theta + M1)
    except np.linalg.LinAlgError:
        return None


def solve_online(theta, sols, game, prev_x=None, prev_crs=None, max_nbr=3, tol=1e-6,
                 stats=None, one_shot=False):
    """Return (x_star, crs). Self-consistency fixed-point on the CR combo, warm-started
    from prev step; then a light 1-hop variational refinement.  ALL of this is LOCAL
    computation on the onboard maps — the block solve (_solve_combo) returns EVERY
    agent's decision at once, so re-checking the combo needs no new agent message.

    Communication accounting (if `stats` dict given):
      stats['broadcasts']   += 1 per step, ALWAYS.  Inter-agent communication is exactly
        one decision broadcast per agent per step (the iteration-free ideal); it does NOT
        grow with the internal solve rounds below.
      stats['solve_rounds'] += number of internal self-consistency rounds (LOCAL compute,
        no broadcast).  1 when the warm-started combo is already correct; +1 per region
        boundary crossing.  one_shot=True forces this to exactly 1.
      stats['coldstart'] += 1 when a centralized cold-start seed was needed (t=0 only).
      stats['fallback']  += 1 when the step failed entirely.
    """
    N = game.N
    Q, c, F = _stacked_cost(game); q = c + F @ theta
    if prev_x is not None:
        x = prev_x.copy()
    else:                                     # cold-start seed (one-time / offline)
        if stats is not None:
            stats["coldstart"] = stats.get("coldstart", 0) + 1
        xc = centralized(game, theta)
        x = xc if xc is not None else np.zeros(game.n_x_total)
    crs = list(prev_crs) if prev_crs else None
    # ── self-consistency rounds on the CR combination (LOCAL compute, no broadcast) ──
    # one_shot=True → exactly ONE round: evaluate maps at the PREVIOUS step's aggregate
    # (one-step lag, absorbed by the receding horizon).  Inter-agent communication is a
    # single decision broadcast per step regardless of how many rounds run here.
    n_iter = 1 if one_shot else (5 if prev_x is not None else 15)
    n_done = 0
    for _ in range(n_iter):
        n_done += 1
        new = _locate_all(theta, sols, game, x, crs)     # local: form aggregate from on-hand x
        xn = _solve_combo(tuple(new), theta, sols, game)  # local: one affine solve for ALL agents
        if xn is None:
            break
        stable = (new == crs)          # located combo == combo we entered with → self-consistent
        x, crs = xn, new
        if stable:
            break
    if stats is not None:
        stats["broadcasts"] = stats.get("broadcasts", 0) + 1        # 1 decision broadcast / step
        stats["solve_rounds"] = stats.get("solve_rounds", 0) + n_done   # LOCAL compute rounds
        stats["steps"] = stats.get("steps", 0) + 1
        if n_done == 1:
            stats["oneround"] = stats.get("oneround", 0) + 1
    if crs is None:
        if stats is not None:
            stats["fallback"] = stats.get("fallback", 0) + 1
        return None, None
    # ── variational refinement: 1-hop neighbor combos, min-potential feasible ──
    # (pure LOCAL computation on the onboard explicit map — zero communication)
    cand = [[crs[i]] + list(sols[i].regions[crs[i]].facet_neighbors)[:max_nbr] for i in range(N)]
    best_x, bestJ, best_combo = None, np.inf, crs
    for combo in itertools.product(*cand):
        xx = _solve_combo(combo, theta, sols, game)
        if xx is not None and game.all_feasible(xx, theta, tol=1e-4):
            J = float(0.5 * xx @ Q @ xx + q @ xx)
            if J < bestJ - 1e-9:
                bestJ, best_x, best_combo = J, xx, list(combo)
    if best_x is None:
        if stats is not None:
            stats["fallback"] = stats.get("fallback", 0) + 1
        return None, crs
    return best_x, best_combo


def _solve_combo_vgne(combo, theta, sols, game, cost):
    """Solve the block-linear equilibrium for `combo` and return x(θ).

    Uses the VARIATIONAL-GNE selection (Hall & Bemporad 2025, §II-C-3): when the
    equilibrium matrix M_x is rank-deficient — which happens exactly when the shared
    coupling binds for ≥2 agents (their Def. 2), i.e. Σ_i p_i = L_max or L_min — the
    system has infinitely many equilibria.  `_solve_equilibrium(select="potential")`
    picks the one that minimises the game potential Σ_i J_i over the solution manifold
    = the variational GNE = social optimum (matches centralized/ADMM).  The naive
    np.linalg.solve used elsewhere fails on that singular boundary system; this does not.
    """
    Mx, Mp, M1 = _assemble_rhg(tuple(combo), sols, game)
    eq = _solve_equilibrium(Mx, Mp, M1, select="potential", cost=cost)
    if not eq.solvable:
        return None
    return eq.H_x @ theta + eq.h_x


# Two tolerances [kW]:
#   STRICT_TOL — acceptance for the walk + min-potential refinement.  Keeps the online
#     solution matching the centralized equilibrium to ~1e-4 kW; NOT relaxed, so a
#     marginally-infeasible (lower-potential) combo can never win the refinement.
#   TOL_ACCEPT — last-resort acceptance of the POINT-LOCATED combo only, at a degenerate
#     vertex (coupling binds for ≥2 agents → rank-deficient M_x) where the variational
#     solution carries a small bounded residual (Hall & Bemporad note).  Physically
#     negligible on a 900 kW system and far more accurate than an ADMM fallback — so we
#     accept it rather than iterate, but ONLY for the located combo (never in refinement).
STRICT_TOL = 1e-4
TOL_ACCEPT = 1e-1


def _feas_resid(game, x, theta):
    """Max constraint violation of x [kW] — local box/balance + shared coupling."""
    r = 0.0
    for a in game.agents:
        xi = x[game.x_slice(a.index)]
        v = a.A_loc @ xi - (a.b_loc + a.S_loc @ theta)
        if v.size:
            r = max(r, float(v.max()))
    lhs = game.coupling_lhs(x)
    if lhs is not None:
        v = lhs - (game.d + game.S_coup @ theta)
        r = max(r, float(v.max()))
    return r


def _membership_ok(combo, x, theta, sols, game, tol=1e-6):
    """True if x is in every agent's CR `combo[i]`: for each i, θ_i=[Σ_{j≠i}p_j ; θ]
    satisfies E_i θ_i ≤ f_i (Hall & Bemporad 2025, membership check).  Unlike point-
    location `== combo`, this TOLERATES OVERLAPPING critical regions — which is exactly
    what happens on the coupling boundary (rank-deficient case, §II-C), where several
    combos' CRs cover the same θ.  A pure matvec per agent — no LP, no re-location."""
    for i in range(game.N):
        th_i = np.concatenate([_sumneg(game, x, i), theta])
        cr = sols[i].regions[combo[i]]
        if np.any(cr.E @ th_i > cr.f + tol):
            return False
    return True


def _equilibrium_x(combo, theta, sols, game, cost, tol=STRICT_TOL):
    """Solve the block-linear system for `combo` (variational-GNE selection); return x
    iff it is feasible (strict) AND a genuine equilibrium (x lies in every agent's CR —
    membership, which tolerates the overlapping CRs of the coupling boundary). Else None."""
    x = _solve_combo_vgne(combo, theta, sols, game, cost)
    if x is None or not game.all_feasible(x, theta, tol=tol):
        return None
    if not _membership_ok(combo, x, theta, sols, game, tol=tol):
        return None
    return x


def solve_step(theta, sols, game, prev_x, prev_combo, stats=None, max_hops=3, max_rounds=4):
    """One RTM clearing step, robust near the coupling boundary.

    Tier 1 — POINT-LOCATION self-consistency (the reliable primitive): from the warm
      solution `prev_x`, locate each agent's CR at [Σ_{j≠i}p_j ; θ], v-GNE-solve that
      combo, and iterate (re-locate from the new x) until the combo is stable.  Point
      location finds the correct combo directly even when consecutive steps' combos are
      graph-far apart, which is what breaks a pure prev→neighbour walk on the boundary.
    Tier 1b — if point location doesn't yield an accepted combo, BFS the facet-neighbour
      graph (1st/2nd/3rd neighbour) from the located combo.
    A 1-hop min-potential refinement then locks the VARIATIONAL (social-optimum) combo.
    Tier 2 — ADMM fallback, used only if all of the above fail (expected very rare; the
      ONLY iteration).  A non-converged ADMM result is NOT propagated as a warm start.

    Equilibria are solved with the v-GNE (potential-min) selection so rank-deficient
    boundary systems are handled correctly (Hall & Bemporad 2025, §II-C-3).

    DATA-TRANSFER accounting (`stats`):
      stats['transfers']  one entry PER STEP = inter-agent transfer rounds that step:
                          **1** for a map step (one decision broadcast; all the location/
                          walk/refinement is LOCAL), **n_iter** for an ADMM-fallback step.
      stats['map_steps'] / stats['fallback'] / stats['fallback_rounds'] / stats['combos_checked'].
    Returns (x, combo).
    """
    Qc, c, F = _stacked_cost(game); qc = c + F @ theta; cost = (Qc, c, F)
    N = game.N; nck = 0
    found_x, found_combo = None, None
    # best point-located combo seen (for the bounded-residual last resort)
    loc_x, loc_combo, loc_resid = None, None, np.inf

    def _track(cc, xx):
        nonlocal loc_x, loc_combo, loc_resid
        r = _feas_resid(game, xx, theta)
        if r < loc_resid:
            loc_x, loc_combo, loc_resid = xx, cc, r

    # ── Tier 1: point-location self-consistency from the warm solution (STRICT) ──
    x = prev_x.copy() if prev_x is not None else np.zeros(game.n_x_total)
    combo = tuple(prev_combo) if prev_combo is not None else None
    for _ in range(max_rounds):
        loc = tuple(_locate_all(theta, sols, game, x, list(combo) if combo else None))
        nck += 1
        xx = _solve_combo_vgne(loc, theta, sols, game, cost)
        if xx is None:
            break
        _track(loc, xx)
        if game.all_feasible(xx, theta, tol=STRICT_TOL) and _membership_ok(loc, xx, theta, sols, game, tol=STRICT_TOL):
            found_x, found_combo = xx, loc
            if tuple(_locate_all(theta, sols, game, xx, list(loc))) == loc:   # self-consistent
                break
        x, combo = xx, loc

    # ── Tier 1b: BFS neighbour walk from the located combo (STRICT) ──
    if found_x is None and combo is not None:
        base = tuple(combo); seen = {base}; q = deque([(base, 0)])
        while q:
            cc, d = q.popleft(); nck += 1
            xx = _solve_combo_vgne(cc, theta, sols, game, cost)
            if xx is not None:
                _track(cc, xx)
                if game.all_feasible(xx, theta, tol=STRICT_TOL) and _membership_ok(cc, xx, theta, sols, game, tol=STRICT_TOL):
                    found_x, found_combo = xx, cc; break
            if d >= max_hops:
                continue
            for i in range(N):
                for nbr in sols[i].regions[cc[i]].facet_neighbors:
                    if nbr < 0 or nbr >= len(sols[i].regions): continue
                    nxt = cc[:i] + (nbr,) + cc[i + 1:]
                    if nxt not in seen:
                        seen.add(nxt); q.append((nxt, d + 1))

    if found_x is not None:
        # ── 1-hop min-potential refinement → variational (social-opt) combo (STRICT) ──
        cand = [[found_combo[i]] + [nbr for nbr in sols[i].regions[found_combo[i]].facet_neighbors if 0 <= nbr < len(sols[i].regions)][:2] for i in range(N)]
        bestJ = float(0.5 * found_x @ Qc @ found_x + qc @ found_x); best_x, best = found_x, list(found_combo)
        for cc in itertools.product(*cand):
            xx = _solve_combo_vgne(cc, theta, sols, game, cost)
            if xx is not None and game.all_feasible(xx, theta, tol=STRICT_TOL):
                J = float(0.5 * xx @ Qc @ xx + qc @ xx)
                if J < bestJ - 1e-9:
                    bestJ, best_x, best = J, xx, list(cc)
        if stats is not None:
            stats.setdefault('transfers', []).append(1)
            stats['broadcasts'] = stats.get('broadcasts', 0) + 1
            stats['map_steps'] = stats.get('map_steps', 0) + 1
            stats['combos_checked'] = stats.get('combos_checked', 0) + nck
            stats['steps'] = stats.get('steps', 0) + 1
        return best_x, tuple(best)

    # ── Tier 2: ADMM fallback (rare; the ONLY iterative data transfer) ──
    # Warm-start from the best v-GNE iterate the map produced, so ADMM converges in a few
    # rounds instead of stalling from zero.
    x_warm = loc_x if loc_x is not None else (x if prev_x is not None else None)
    res = admm_solve(game, theta, rho=0.5, max_iter=500, tol=1e-3,
                     x_init=x_warm, qp_solver="osqp")
    x = res.x_stacked
    combo = tuple(_locate_all(theta, sols, game, x, None)) if res.converged else \
            (tuple(prev_combo) if prev_combo is not None else tuple(_locate_all(theta, sols, game, x, None)))
    if stats is not None:
        stats.setdefault('transfers', []).append(int(res.n_iter))
        stats['broadcasts'] = stats.get('broadcasts', 0) + 1
        stats['fallback'] = stats.get('fallback', 0) + 1
        stats['fallback_rounds'] = stats.get('fallback_rounds', 0) + int(res.n_iter)
        stats['combos_checked'] = stats.get('combos_checked', 0) + nck
        stats['steps'] = stats.get('steps', 0) + 1
    return x, combo


def centralized(game, theta):
    import gurobipy as gp
    from gurobipy import GRB
    Q, c, F = _stacked_cost(game); G, w0, W = _centralized_constraints(game)
    q = c + F @ theta; rhs = w0 + W @ theta; n = Q.shape[0]
    m = gp.Model(); m.Params.OutputFlag = 0
    x = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    m.setObjective(0.5 * (x @ Q @ x) + q @ x, GRB.MINIMIZE)
    m.addConstr(G @ x <= rhs); m.optimize()
    return np.array(x.X) if m.Status == GRB.OPTIMAL else None


def _rand_theta(game, rng):
    """Random public θ in the p_gne box."""
    return game.p_lb + rng.random(game.n_p) * (game.p_ub - game.p_lb)


if __name__ == "__main__":
    import time
    print("loading + neighbor graphs ...", flush=True)
    t = time.perf_counter(); sols, game = load_and_prepare()
    print(f"  ready ({time.perf_counter()-t:.1f}s)  n_p={game.n_p} CRs={[s.n_cr for s in sols]}")
    rng = np.random.default_rng(0)
    lb, ub = game.p_lb, game.p_ub
    th = lb + rng.random(game.n_p) * (ub - lb)          # start of a smooth θ walk
    prev_x = centralized(game, th); prev_crs = None     # seed step 0 (one-time offline)
    emax = 0.0; miss = 0; nok = 0; tt = 0.0
    for it in range(80):
        # smooth random walk (mimics RTM: consecutive θ are close → warm-start)
        th = np.clip(th + rng.normal(0, 0.04, game.n_p) * (ub - lb), lb, ub)
        s = time.perf_counter()
        x, crs = solve_online(th, sols, game, prev_x=prev_x, prev_crs=prev_crs)
        tt += time.perf_counter() - s
        xc = centralized(game, th)
        if xc is None:
            continue
        if x is None:
            miss += 1; continue
        nok += 1; prev_x = x; prev_crs = crs
        emax = max(emax, float(np.max(np.abs(x - xc))))
    print(f"\n[online FACET, warm-started θ-walk] checked {nok}  misses {miss}")
    print(f"  max |online − centralized| = {emax:.2e} kW")
    print(f"  online solve {1e3*tt/max(nok+miss,1):.1f} ms/step")
    print("  GATE:", "PASS" if miss == 0 and emax < 1e-3 else "CHECK")
