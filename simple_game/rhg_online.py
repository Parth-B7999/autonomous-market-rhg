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
import os, sys, pickle, itertools
from collections import deque
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
import rhg_mpqp as R
from amrhg.solvers.gne_combiner import (_stacked_cost, _centralized_constraints,
                                        _solve_equilibrium)
from amrhg.solvers.facet_gne import (find_all_agent_cr_neighbors,
                                     precompute_point_location_arrays, _locate_cr_fast)
from amrhg.solvers.admm_solver import admm_solve

H = R.H

# Equilibrium selection for RANK-DEFICIENT combinations (AMRHG_HALL_EQ16=1 to re-enable).
#   False (DEFAULT) — `_solve_combo_vgne`: the MIN-POTENTIAL point on the GNE manifold,
#           i.e. the variational GNE / social optimum, which is what the centralized
#           reference computes.  This is the selection that closes the gap.
#   True  — Hall & Bemporad Eq. (16): require every agent's coupling multiplier to agree,
#           REJECTING the combination when they cannot.
# Full-rank combinations have a unique equilibrium and are identical under both settings.
#
# Eq. (16) is kept in the tree as a documented alternative but is OFF by default: it fires
# on only ~15% of steps, and where it does fire it picks a different point of the same
# manifold than the potential minimiser, measuring 2-4x further from centralized.
# See [[project_mpgne_option2_deferred]].
USE_HALL_EQ16 = os.environ.get("AMRHG_HALL_EQ16", "0") != "0"

# Tier-1c 1-hop min-potential refinement breadth (neighbours per agent).  0 disables it
# entirely, which leaves PURE FACET: point location + self-consistency + neighbour BFS.
#
# DEFAULT 0 (off).  Measured 2026-07-21, April week, Gurobi at every one of 480 steps with
# max_nbr the ONLY variable between the two arms:
#     worst case   9.052 kW both  (identical — it never improves the headline number)
#     helps 11/480 steps (2.3%), hurts 0; mean error 0.3230 -> 0.3067 kW
#     cost 3-4 ms -> 21-51 ms on a normal day (5-16x the median step)
# The gap to centralized is closed by the VARIATIONAL selection inside _solve_combo_vgne
# (min-potential point on the rank-deficient manifold), not by this neighbour sweep; an
# earlier 14 kW -> 9.1 kW improvement was wrongly credited here when both changed at once.
MAX_NBR = int(os.environ.get("AMRHG_MAX_NBR", "0"))


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


# ── Equilibrium-map cache: combo -> (H_x, h_x), or None when unsolvable ──────────────
# EXACT, not an approximation.  gne_combiner._solve_equilibrium returns the equilibrium
# as an AFFINE MAP x*(θ) = H_x θ + h_x, and for select="potential" the minimiser over the
# manifold is itself affine (z*(θ) = A_z θ + b_z).  So (H_x, h_x) depend only on the
# COMBINATION — never on θ — and can be reused for every step of the run.
# This is what makes the 1-hop refinement affordable: the 729 candidate combinations
# recur constantly (the report measures only ~700 distinct combinations over 1344 steps),
# so after warm-up almost every candidate is a matvec instead of an assemble + SVD.
_EQ_MISS = object()                 # sentinel: key absent (None means "unsolvable")
_EQ_CACHE: dict = {}
_EQ_CACHE_MAX = 30_000              # ~8 kB/entry -> ~240 MB ceiling
_EQ_STATS = {"hit": 0, "miss": 0}


def _solve_combo_vgne(combo, theta, sols, game, cost):
    """Solve the block-linear equilibrium for `combo`, selecting the VARIATIONAL GNE.

    Hall & Bemporad §II-C-3: when M_x is rank-deficient — exactly when the shared coupling
    binds for >=2 agents (their Def. 2), i.e. Σ_i p_i = L_max or L_min — infinitely many
    equilibria exist.  `_solve_equilibrium(select="potential")` picks the one MINIMISING
    the game potential Σ_i J_i over that manifold: the variational GNE = social optimum,
    which is what the centralized reference computes.

    This is the selection the validated baseline used (`_validated_baseline`, 2026-07-16)
    and it is what closes the gap to centralized.  Min-norm selection and Hall Eq. (16)
    both pick a DIFFERENT point of the same manifold and measured 2-4x worse.

    Φ is separable (Q is block-diagonal), so each term uses only that agent's own cost
    data; the stacked `cost` triple is a simulation convenience, not shared information.
    """
    key = tuple(combo)
    hit = _EQ_CACHE.get(key, _EQ_MISS)
    if hit is not _EQ_MISS:
        _EQ_STATS["hit"] += 1
        return None if hit is None else hit[0] @ theta + hit[1]

    _EQ_STATS["miss"] += 1
    Mx, Mp, M1 = _assemble_rhg(key, sols, game)
    eq = _solve_equilibrium(Mx, Mp, M1, select="potential", cost=cost)
    val = (eq.H_x, eq.h_x) if eq.solvable else None
    if len(_EQ_CACHE) < _EQ_CACHE_MAX:
        _EQ_CACHE[key] = val
    return None if val is None else val[0] @ theta + val[1]


def _potential_local(x, theta, game):
    """Coalition potential Φ(x) = Σ_i J_i(x_i), as a SUM OF PER-AGENT TERMS.

    Q is block-diagonal (see gne_combiner._stacked_cost), so J_i uses ONLY agent i's own
    cost data (Q_i, c_i, F_i) and its own decision block x_i.  Each agent therefore
    evaluates its own scalar locally and the six scalars are summed — no agent needs
    another's cost model and nothing is assembled centrally.  This is deliberately NOT
    `_stacked_cost(game)`, which builds one joint cost matrix and would contradict the
    model-privacy claim.
    """
    tot = 0.0
    for a in game.agents:
        xi = x[game.x_slice(a.index)]
        tot += float(0.5 * xi @ a.Q @ xi + (a.c + a.F @ theta) @ xi)
    return tot


def _manifold(combo, theta, sols, game):
    """Return (x_p, Nb) for `combo`: the GNE manifold x(y₂) = x_p + Nb y₂ (Eq. 8–9a).

    Nb has zero columns when M_x is full rank (unique GNE, Eq. 7a).  None if unsolvable.
    """
    Mx, Mp, M1 = _assemble_rhg(tuple(combo), sols, game)
    n_x = Mx.shape[0]
    rhs = Mp @ theta + M1
    rank = np.linalg.matrix_rank(Mx, tol=1e-8)
    if rank == n_x:
        try:
            return np.linalg.solve(Mx, rhs), np.zeros((n_x, 0))
        except np.linalg.LinAlgError:
            return None
    U, sig, Vt = np.linalg.svd(Mx, full_matrices=True)
    nM = rank
    U2 = U[:, nM:]
    if U2.size and np.max(np.abs(U2.T @ rhs)) > 1e-6:      # solvability, Eq. (10)
        return None
    V1 = Vt[:nM].T
    x_p = V1 @ ((U[:, :nM].T @ rhs) / sig[:nM])
    return x_p, Vt[nM:].T                                   # V₂ spans null(M_x)


def _coupling_multiplier_map(cr, n_coupling):
    """Return the full affine coupling-multiplier map for one PPOPT agent CR.

    PPOPT stores multipliers only for active constraints.  In the RHG mpQP the
    2H coupling rows are appended last, so inactive coupling rows have multiplier
    zero and active rows are recovered by their original active-set indices.
    """
    lam_A = getattr(cr, "lambda_A", None)
    lam_b = getattr(cr, "lambda_b", None)
    active = getattr(cr, "active_set", [])
    n_con = int(getattr(cr, "n_constraints", 0))
    if lam_A is None or lam_b is None or n_con < n_coupling:
        return None
    if lam_A.shape[0] != len(active) or lam_b.size != len(active):
        return None
    start = n_con - n_coupling
    L = np.zeros((n_coupling, cr.n_theta)); l = np.zeros(n_coupling)
    for row, con_idx in enumerate(active):
        if start <= con_idx < n_con:
            q = con_idx - start
            L[q] = lam_A[row]
            l[q] = lam_b[row]
    return L, l


def _sumneg_matrix(game, i):
    """Linear map S_i with S_i x = sum_{j != i} p_j over the H RTM entries."""
    S = np.zeros((H, game.n_x_total))
    for j in range(game.N):
        if j != i:
            st = game.x_slice(j).start
            S[:, st:st + H] = np.eye(H)
    return S


def _solve_combo_hall_vgne(combo, theta, sols, game):
    """Hall--Bemporad Eq. (16) v-GNE selector for one CR combination.

    A full-rank combination has a unique GNE.  For a rank-deficient combination,
    solve only the small null-space linear system that makes all agents' *stored*
    coupling-multiplier maps agree.  This is affine-map evaluation and linear
    algebra, not a centralized potential optimization.  ``None`` means this
    combination cannot certify a unique v-GNE, so the caller must keep searching
    or use distributed ADMM.
    """
    mf = _manifold(combo, theta, sols, game)
    if mf is None:
        return None
    x_p, V2 = mf
    if V2.shape[1] == 0:
        return x_p
    if not USE_HALL_EQ16:
        return x_p          # min-norm on the manifold (pre-Eq.(16) selection)

    coeffs, offsets = [], []
    for i, j in enumerate(combo):
        mm = _coupling_multiplier_map(sols[i].regions[j], game.n_coupling)
        if mm is None:
            return None
        L, l = mm
        L_sum, L_theta = L[:, :H], L[:, H:]
        S_i = _sumneg_matrix(game, i)
        coeffs.append(L_sum @ S_i @ V2)
        offsets.append(L_sum @ S_i @ x_p + L_theta @ theta + l)

    A_eq = np.vstack([coeffs[0] - a for a in coeffs[1:]])
    b_eq = np.concatenate([b - offsets[0] for b in offsets[1:]])
    n_null = V2.shape[1]
    if A_eq.size == 0 or np.linalg.matrix_rank(A_eq, tol=1e-8) < n_null:
        return None
    y2, *_ = np.linalg.lstsq(A_eq, b_eq, rcond=None)
    if np.max(np.abs(A_eq @ y2 - b_eq)) > 1e-6 * (1.0 + np.max(np.abs(b_eq))):
        return None
    return x_p + V2 @ y2


# STRICT_TOL [kW] — acceptance for the affine-map solve and neighbour walk.
# Override with AMRHG_STRICT_TOL to sweep it (the 2026-07-20 edits changed behaviour here
# and the pre-edit value is not recoverable — no git history for that revision).
STRICT_TOL = float(os.environ.get("AMRHG_STRICT_TOL", "1e-4"))


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


def _equilibrium_x(combo, theta, sols, game, tol=STRICT_TOL):
    """Solve a combo with Hall's v-GNE certificate; return x
    iff it is feasible (strict) AND a genuine equilibrium (x lies in every agent's CR —
    membership, which tolerates the overlapping CRs of the coupling boundary). Else None."""
    x = _solve_combo_hall_vgne(combo, theta, sols, game)
    if x is None or not game.all_feasible(x, theta, tol=tol):
        return None
    if not _membership_ok(combo, x, theta, sols, game, tol=tol):
        return None
    return x


def solve_step(theta, sols, game, prev_x, prev_combo, stats=None, max_hops=3, max_rounds=4,
               max_nbr=None):
    """One RTM clearing step, robust near the coupling boundary.

    Tier 1 — POINT-LOCATION self-consistency (the reliable primitive): from the warm
      solution `prev_x`, locate each agent's CR at [Σ_{j≠i}p_j ; θ], v-GNE-solve that
      combo, and iterate (re-locate from the new x) until the combo is stable.  Point
      location finds the correct combo directly even when consecutive steps' combos are
      graph-far apart, which is what breaks a pure prev→neighbour walk on the boundary.
    Tier 1b — if point location doesn't yield an accepted combo, BFS the facet-neighbour
      graph (1st/2nd/3rd neighbour) from the located combo.
    For rank-deficient combinations, Hall--Bemporad Eq. (16) enforces equality
    of the stored coupling multipliers; this certifies a v-GNE without an online
    centralized potential optimization.
    Tier 2 — distributed ADMM fallback, used only if all of the above fail (expected
      very rare; the ONLY iterative exchange). No centralized constrained-potential
      optimization is used in the online clearing method.

    Full-rank combinations are unique. Rank-deficient combinations are accepted
    only after the Hall equal-multiplier certificate passes.

    DATA-TRANSFER accounting (`stats`):
      stats['transfers']  one entry PER STEP = inter-agent transfer rounds that step:
                          **1** for a map step (one decision broadcast; all the location/
                          walk/refinement is LOCAL), **n_iter** for an ADMM-fallback step.
      stats['map_steps'] / stats['fallback'] / stats['fallback_rounds'] / stats['combos_checked'].
    Returns (x, combo).
    """
    # Stacked cost triple for the potential-minimising (variational) selection.
    # Phi is separable across agents (Q block-diagonal) — see _potential_local.
    cost = _stacked_cost(game)
    if max_nbr is None:
        max_nbr = MAX_NBR
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
        xx = (_solve_combo_hall_vgne(loc, theta, sols, game) if USE_HALL_EQ16
              else _solve_combo_vgne(loc, theta, sols, game, cost))
        if xx is None:
            # An INTERMEDIATE iterate that Eq. (16) cannot certify is not a reason to
            # abandon the fixed point — the destination combo is usually certifiable even
            # when a stepping stone is not.  Continue from the manifold's min-norm point;
            # the combo finally ACCEPTED below still has to pass the full Hall certificate,
            # all_feasible, and _membership_ok, so this never weakens acceptance.
            # Measured on 2025-04-01: breaking here solves 39/96 steps, continuing 69/96.
            mf = _manifold(loc, theta, sols, game)
            if mf is None:
                break
            xx = mf[0]
            x, combo = xx, loc
            continue
        _track(loc, xx)
        if game.all_feasible(xx, theta, tol=STRICT_TOL) and _membership_ok(loc, xx, theta, sols, game, tol=STRICT_TOL):
            found_x, found_combo = xx, loc
            # baseline: keep iterating unless the combo reproduces itself exactly
            if tuple(_locate_all(theta, sols, game, xx, list(loc))) == loc:
                break
        x, combo = xx, loc

    # ── Tier 1b: BFS neighbour walk from the located combo (STRICT) ──
    # Runs only when Tier-1 point location fails to certify a combo (rare).  Cheap when a valid
    # combo is 1 hop away (measured: rescues succeed within ~66 combos); when it fails it walks
    # the full 3-hop shell (~17k combos, ~5 s) before the ADMM fallback.  A combo budget was
    # tested to cap that tail, but the fallback count is chaotically trajectory-sensitive and
    # any cap tight enough to speed the failures also turned 4-6 rescues into fallbacks
    # (99.3%→99.0%); the uncapped walk preserves the iteration-free rate, and a fallback step's
    # ~5 s is rare (9/1344) and 180x within the 900 s dispatch interval.  See [[project_mpgne_benchmark]].
    if found_x is None and combo is not None:
        base = tuple(combo); seen = {base}; q = deque([(base, 0)])
        while q:
            cc, d = q.popleft(); nck += 1
            xx = (_solve_combo_hall_vgne(cc, theta, sols, game) if USE_HALL_EQ16
                  else _solve_combo_vgne(cc, theta, sols, game, cost))
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
        # The accepted combination is either full rank (hence its GNE is unique) or
        # rank deficient with Hall Eq. (16)'s common coupling multiplier certified.
        # No global potential QP is evaluated by the online clearing method.
        best_x, best = found_x, list(found_combo)

        # ── Tier 1c: variational (min-potential) selection ───────────────────────────
        # Several 1-hop combinations can all be valid GNEs at a binding-coupling step;
        # they differ in coalition cost.  Keep the FEASIBLE one with the lowest potential
        # Φ — that is the VARIATIONAL GNE (= social optimum), which is what the
        # centralized reference computes.  Without this the solver keeps whichever valid
        # GNE it reached first, which measured 2-4x further from centralized
        # (2026-07-21: worst-case 14 kW without, 5.9 kW with).
        # Φ is evaluated by `_potential_local` as a SUM OF PER-AGENT TERMS, so no agent
        # shares a cost model and no joint cost matrix is assembled; and no optimizer
        # runs — the candidates already exist, this only ranks them.  Pure local
        # arithmetic on the onboard maps: zero additional communication.
        # Gate: the refinement can only change the answer where the equilibrium is
        # NON-UNIQUE, i.e. where the shared coupling binds for >=2 agents and M_x loses
        # rank (Hall & Bemporad Def. 2).  On interior steps M_x is full rank, the GNE is
        # unique, and there is nothing to select — sweeping 3^N combinations there is
        # provably wasted work.  The report measures the same thing: the refinement costs
        # "~177 ms on every CEILING-BOUND step".  Detect it physically (cheaper than a
        # rank test): is the coalition aggregate sitting on L_min or L_max at any step?
        agg = np.zeros(H)
        for i in range(N):
            st = game.x_slice(i).start
            agg += best_x[st:st + H]
        binds = bool(np.any(np.abs(agg - R.L_MAX) < 1e-3) or
                     np.any(np.abs(agg - R.L_MIN) < 1e-3))
        if max_nbr > 0 and binds:
            bestJ = _potential_local(best_x, theta, game)
            cand = [[found_combo[i]] +
                    [n for n in sols[i].regions[found_combo[i]].facet_neighbors
                     if 0 <= n < len(sols[i].regions)][:max_nbr]
                    for i in range(N)]
            for cc in itertools.product(*cand):
                if tuple(cc) == tuple(found_combo):
                    continue
                nck += 1
                xx = _solve_combo_vgne(cc, theta, sols, game, cost)
                if xx is None or not game.all_feasible(xx, theta, tol=STRICT_TOL):
                    continue
                J = _potential_local(xx, theta, game)
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
    # Warm-start from the best affine-map iterate, so ADMM converges in a few
    # rounds instead of stalling from zero.
    #
    # rho=0.002 (was 0.5): these fallback steps are all at the DEGENERATE COUPLING CEILING,
    # where a large penalty over-constrains the split and ADMM oscillates — the old rho=0.5
    # ran to the max_iter cap WITHOUT converging (measured: median 1389 iters to reach 1e-3,
    # so it stopped at ~0.11 kW after 500 rounds).  A small penalty, combined with the warm
    # start, converges in a MEDIAN OF 27 ROUNDS to ~4e-4 kW (tuned 2026-07-17 on the exact
    # production fallback steps).  This is the biggest lever on the communication TAIL: the
    # rare fallbacks dominated it (9 steps * 500 = 4500 of ~5800 total inter-agent rounds);
    # at 27 rounds they become negligible.
    x_warm = loc_x if loc_x is not None else (x if prev_x is not None else None)
    res = admm_solve(game, theta, rho=0.002, max_iter=500, tol=1e-3,
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
