"""
simple_mpqp.py — Per-agent mpQP for explicit GNE on the 4-agent RTM simple game.

θ_i per agent (PRIVATE — only parameters that appear in agent i's own QP)
──────────────────────────────────────────────────────────────────────────
  θ_i = [sum_x_neg(H=6)  |  p_priv_i]

  sum_x_neg = Σ_{j≠i} x_j ∈ ℝ^H  — SUM of other agents' power decisions.
              Agent i's coupling constraint only involves this sum, not individual
              x_j values → 6-dim instead of 18-dim (3 agents × H = 18).
              Eliminated by M_x⁻¹ in the GNE solve.

  p_priv_i = LOCAL parameters: only what appears in agent i's own cost/constraints
    VRFB (0):    [SoC_0(1), λ^RT(6), p^DA_0(6)]           →  n_p_priv = 13
    PV+Batt (1): [SoC_1(1), λ^RT(6), p^DA_1(6), g^PV(6)]  →  n_p_priv = 19
    PEM (2):     [Inv_0(1), λ^RT(6), p^DA_2(6)]            →  n_p_priv = 13
    Alk (3):     [Inv_0(1), λ^RT(6), p^DA_3(6)]            →  n_p_priv = 13

  n_theta_i:  VRFB/PEM/Alk = 19,  PV+Batt = 25

GNE parameter space (combined, after equilibrium solve)
────────────────────────────────────────────────────────
  p_gne = union of all private params (40-dim):
    [0]      SoC_0   [1]      SoC_1   [2]      Inv_2   [3]      Inv_3
    [4..9]   λ^RT(6)
    [10..15] p^DA_0(6)   [16..21] p^DA_1(6)
    [22..27] p^DA_2(6)   [28..33] p^DA_3(6)
    [34..39] g^PV(6)

  L_max (1500 kW) is a FIXED CONSTANT in b_coup (not in p_gne).
  L_min is DROPPED — non-binding and causes CR explosion in mpQP.

CR expansion for combiner compatibility
────────────────────────────────────────
  After PPOPT solves in private 19/25-dim θ_i-space, we EXPAND each CR:
    A_expanded: (n_x_i, H + 40) = (6, 46)  — zeros for non-private p_gne cols
    E_expanded: (n_ineq, H + 40) = (n_ineq, 46)

  gne_combiner / facet_gne auto-detect sum mode via cr.E.shape[1] - game.n_p = 6.

Constraints (30 per agent, not 36)
─────────────────────────────────
  0..23   local:     SoC/Inv bounds (×2) + power box (×2)  = 4H rows
  24..29  coupling:  p_i + sum_x_neg ≤ L_max                = H rows  (upper only)
  (L_min removed — non-binding, causes CR inflation)
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ppopt.mpqp_program import MPQP_Program
from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm

from .game import GNEGame
from .cr_store import AgentCR, AgentSolution, agent_solution_from_ppopt
from ..game.simple_game import SimpleParameterLayout


DEFAULT_ALGORITHM = mpqp_algorithm.combinatorial_parallel

# ─────────────────────────────────────────────────────────────────────────────
#  p_gne layout  (combined GNE parameter space, 40-dim)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PGNELayout:
    """
    Defines the combined GNE parameter space p_gne ∈ ℝ^{n_p_gne} and maps
    each agent's private parameters into it.

    Layout (fixed for N=4, H=6):
      [0..3]   initial states: SoC_0, SoC_1, Inv_2, Inv_3
      [4..9]   λ^RT prices (6 steps)
      [10..13] p^DA_0, p^DA_1, p^DA_2, p^DA_3  — ONE scalar per agent
      [14..19] g^PV forecast (6 steps)

    agent_embed[i] : list of indices into p_gne for agent i's private params,
                     in the ORDER they appear in p_priv_i after x_{-i}.
    """
    n_p_gne:      int               # 20
    H:            int               # 6
    n_agents:     int               # 4
    state_indices: tuple[int, ...]  # (0, 1, 2, 3)
    price_start:  int               # 4
    da_starts:    tuple[int, ...]   # (10, 11, 12, 13)
    pv_start:     int | None        # 14
    L_max_kw:     float             # fixed constant
    agent_embed:  tuple[tuple[int, ...], ...]  # per-agent p_gne index lists
    n_p_priv:     tuple[int, ...]   # (8, 14, 8, 8)


def make_pgne_layout(
    layout: SimpleParameterLayout,
    L_max_kw: float,
) -> PGNELayout:
    """
    Build PGNELayout from SimpleParameterLayout.

    p_gne does NOT include L_max or L_min (both are constants).
    The layout mirrors p_ext from the old code but is now the authoritative
    combined GNE parameter space.
    """
    N = layout.n_agents
    H = layout.H

    # Fixed slot assignments
    state_indices = tuple(range(N))                       # [0, 1, 2, 3]
    price_start   = N                                      # 4
    da_starts     = tuple(N + H + i for i in range(N))   # [10, 11, 12, 13] — 1 per agent
    pv_start      = N + H + N                             # 14
    n_p_gne       = pv_start + H                          # 20

    # Per-agent embedding: indices in p_gne for each agent's private params
    # Order: [state_i(1), λ^RT(6), p^DA_i(1), (g^PV(6) if PV agent)]
    embeds = []
    for i in range(N):
        em = [state_indices[i]]                           # state (1)
        em += list(range(price_start, price_start + H))  # λ^RT (6)
        em += [da_starts[i]]                             # p^DA_i (1 scalar)
        if i == 1:                                        # PV+Batt agent
            em += list(range(pv_start, pv_start + H))    # g^PV (6)
        embeds.append(tuple(em))

    n_p_priv = tuple(len(em) for em in embeds)  # (8, 14, 8, 8)

    return PGNELayout(
        n_p_gne      = n_p_gne,
        H            = H,
        n_agents     = N,
        state_indices= state_indices,
        price_start  = price_start,
        da_starts    = da_starts,
        pv_start     = pv_start,
        L_max_kw     = L_max_kw,
        agent_embed  = tuple(embeds),
        n_p_priv     = n_p_priv,
    )


def pack_pgne(
    pgne: PGNELayout,
    states: list[float],
    lmp_rt: np.ndarray,
    da_schedules: list[np.ndarray],
    pv_forecast: np.ndarray | None = None,
) -> np.ndarray:
    """Pack the combined GNE parameter vector p_gne ∈ ℝ^40."""
    p = np.zeros(pgne.n_p_gne)
    for i, s in enumerate(states):
        p[pgne.state_indices[i]] = s
    p[pgne.price_start:pgne.price_start + pgne.H] = lmp_rt
    for i, da in enumerate(da_schedules):
        p[pgne.da_starts[i]] = float(da) if np.ndim(da) == 0 else float(da[0])
    if pv_forecast is not None and pgne.pv_start is not None:
        p[pgne.pv_start:pgne.pv_start + pgne.H] = pv_forecast
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Build per-agent mpQP in PRIVATE θ_i space
# ─────────────────────────────────────────────────────────────────────────────

def _private_theta_layout(i: int, pgne: PGNELayout) -> dict:
    """
    Return index layout within private θ_i = [sum_x_neg(H) | p_priv_i].

    sum_x_neg = Σ_{j≠i} x_j ∈ ℝ^H occupies the first H positions.
    Only the SUM appears in agent i's coupling constraint, not individual x_j.

    Private p_priv_i ordering (after sum_x_neg):
      [0]      state_i
      [1..6]   λ^RT (6 steps)
      [7]      p^DA_i (1 scalar — current hourly DA value)
      [8..13]  g^PV (6 steps) — PV+Batt only

    Returns dict with:
      n_x_neg : H = 6  (sum coupling dims)
      n_p_priv: 8 or 14
      n_theta : 14 or 20
      state_theta : index in θ_i for state
      price_theta : slice in θ_i for λ^RT
      da_theta    : int index in θ_i for DA scalar
      pv_theta    : slice in θ_i for g^PV (or None)
    """
    H       = pgne.H
    n_x_neg = H
    is_pv   = (i == 1)
    n_p_priv = 1 + H + 1 + (H if is_pv else 0)  # 8 or 14
    n_theta  = n_x_neg + n_p_priv                # 14 or 20

    state_theta = n_x_neg + 0
    price_theta = slice(n_x_neg + 1, n_x_neg + 1 + H)
    da_theta    = n_x_neg + 1 + H               # single int
    pv_theta    = slice(n_x_neg + 1 + H + 1, n_theta) if is_pv else None

    return dict(
        n_x_neg     = n_x_neg,
        n_p_priv    = n_p_priv,
        n_theta     = n_theta,
        state_theta = state_theta,
        price_theta = price_theta,
        da_theta    = da_theta,
        pv_theta    = pv_theta,
        is_pv       = is_pv,
    )


def build_agent_matrices(
    game: GNEGame,
    i: int,
    layout: SimpleParameterLayout,
    pgne: PGNELayout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Build PPOPT (Q, H_pp, c, G, b, F) for agent i's mpQP with PRIVATE θ_i.

    θ_i ∈ ℝ^{19 or 25}:  [sum_x_neg(H) | state_i(1) | λ^RT(6) | DA_i(6) | (PV(6))]

    Constraint rows (30 total):
      0..23   local: SoC/Inv bounds + power box
      24..29  coupling upper: p_i + sum_x_neg ≤ L_max  (L_min dropped)

    Returns (Q, H_pp, c, G, b, F) in PPOPT convention:
      min  ½ p^T Q p + (H_pp θ + c)^T p
      s.t. G p ≤ b + F θ
    """
    ai      = game.agents[i]
    H       = layout.H
    dt      = layout.dt_hr
    gamma   = float(ai.Q[0, 0])
    th      = _private_theta_layout(i, pgne)
    n_theta = th["n_theta"]

    # ── Cost ──────────────────────────────────────────────────────────────────
    Q   = ai.Q.copy()   # γ I_H
    c   = ai.c.copy()   # constant linear cost (0 for batteries; < 0 for electrolyzers)

    H_pp = np.zeros((H, n_theta))
    # Energy cost: +dt * λ^RT_k  (price at step k in private θ_i)
    for k in range(H):
        H_pp[k, th["price_theta"].start + k] = dt
    # DA tracking: -γ * p^DA_i  (scalar — same column for all horizon steps)
    H_pp[:, th["da_theta"]] = -gamma

    # ── Local constraints (24 rows) ───────────────────────────────────────────
    G_loc = ai.A_loc.copy()           # (4H, H)
    b_loc = ai.b_loc.copy()           # (4H,)
    F_loc = np.zeros((4 * H, n_theta))

    # Map S_loc (4H, n_p_full) → F_loc using layout indices
    # Rows 0..H-1   : SoC/Inv upper bound  — parametric in state_i (and PV for agent 1)
    # Rows H..2H-1  : SoC/Inv lower bound  — parametric in state_i (and PV for agent 1)
    # Rows 2H..4H-1 : power box            — parametric in g^PV (agent 1 only)
    state_full_idx = layout.state_init_indices[i]
    F_loc[:H,  th["state_theta"]] = ai.S_loc[:H,  state_full_idx]   # upper soc
    F_loc[H:2*H, th["state_theta"]] = ai.S_loc[H:2*H, state_full_idx]  # lower soc

    if th["is_pv"] and layout.pv_start is not None:
        pv_theta_sl = th["pv_theta"]
        pv_full_sl  = layout.pv_slice   # slice in full p
        for k in range(H):
            # Upper SoC:  -Γ g^PV_k  (H rows, each depends on one g^PV_k)
            F_loc[:H,  pv_theta_sl.start + k] = ai.S_loc[:H,  pv_full_sl.start + k]
            # Lower SoC:  +Γ g^PV_k
            F_loc[H:2*H, pv_theta_sl.start + k] = ai.S_loc[H:2*H, pv_full_sl.start + k]
            # Power box:  upper p_net ≤ P_batt - g^PV_k
            F_loc[2*H + k, pv_theta_sl.start + k] = ai.S_loc[2*H + k, pv_full_sl.start + k]
            # Power box:  lower  -p_net ≤ P_batt + g^PV_k
            F_loc[3*H + k, pv_theta_sl.start + k] = ai.S_loc[3*H + k, pv_full_sl.start + k]

    # ── Coupling constraint — upper only (6 rows) ─────────────────────────────
    # p_i + sum_x_neg ≤ L_max  where sum_x_neg = Σ_{j≠i} x_j
    # → I p_i ≤ L_max·1 − sum_x_neg   (sum_x_neg = θ_i[:H])
    L_max    = pgne.L_max_kw
    G_coup   = np.eye(H)
    b_coup   = L_max * np.ones(H)
    F_coup   = np.zeros((H, n_theta))
    # θ_i[:H] = sum_x_neg → F_coup[:H, :H] = -I_H
    F_coup[:H, :H] = -np.eye(H)

    # ── Stack ─────────────────────────────────────────────────────────────────
    G = np.vstack([G_loc, G_coup])           # (30, H)
    b = np.concatenate([b_loc, b_coup])      # (30,)
    F = np.vstack([F_loc, F_coup])           # (30, n_theta)

    return Q, H_pp, c, G, b, F


def build_parameter_space(
    game: GNEGame,
    i: int,
    layout: SimpleParameterLayout,
    pgne: PGNELayout,
    lmp_lb: float = 0.0,
    lmp_ub: float = 250.0,
    da_lb: float | None = None,
    da_ub: float | None = None,
    pv_ub: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tight box constraints on private θ_i = [sum_x_neg(H) | p_priv_i].

    Bounds are physically motivated — no large artificial limits.

    Parameters
    ----------
    lmp_lb / lmp_ub : $/MWh price range.  Tighter range → fewer CRs.
    da_lb / da_ub   : Override for p^DA_i bounds (kW).  None = use agent power limits.
    pv_ub           : Override for g^PV upper bound (kW, PV+Batt only).  None = battery cap.

    Returns A_t (2·n_theta, n_theta), b_t (2·n_theta, 1) for PPOPT.
    """
    H       = layout.H
    th      = _private_theta_layout(i, pgne)
    n_theta = th["n_theta"]

    theta_min = np.zeros(n_theta)
    theta_max = np.zeros(n_theta)

    # ── sum_x_neg bounds: sum of other agents' power boxes (6-dim) ───────────
    # sum_lb(k) = Σ_{j≠i} p_min_j(k),  sum_ub(k) = Σ_{j≠i} p_max_j(k)
    sum_lb = np.zeros(H)
    sum_ub = np.zeros(H)
    for j in range(game.N):
        if j == i:
            continue
        aj = game.agents[j]
        sum_ub += aj.b_loc[2 * H:3 * H]    # +p_max_j
        sum_lb -= aj.b_loc[3 * H:4 * H]    # -(-p_min_j) = +p_min_j
        if j == 1 and pv_ub is not None:   # PV agent
            sum_lb -= pv_ub
    theta_min[:H] = sum_lb
    theta_max[:H] = sum_ub

    # ── state_i: [0, state_max] ───────────────────────────────────────────────
    ai = game.agents[i]
    # b_loc[0] = SoC_max (or Inv_max + q_step for electrolyzers) — valid upper
    theta_min[th["state_theta"]] = 0.0
    theta_max[th["state_theta"]] = float(ai.b_loc[0])

    # ── λ^RT: tight physical price range ─────────────────────────────────────
    ps = th["price_theta"]
    theta_min[ps] = lmp_lb
    theta_max[ps] = lmp_ub

    # ── p^DA_i: DA scalar (or caller override) ───────────────────────────────
    ds = th["da_theta"]   # single int
    theta_min[ds] = da_lb if da_lb is not None else -ai.b_loc[3 * H]
    theta_max[ds] = da_ub if da_ub is not None else  ai.b_loc[2 * H]

    # ── g^PV: [0, PV capacity] (PV+Batt only) ────────────────────────────────
    if th["is_pv"]:
        pv = th["pv_theta"]
        theta_min[pv] = 0.0
        theta_max[pv] = pv_ub if pv_ub is not None else float(ai.b_loc[2 * H])

    A_t = np.vstack([ np.eye(n_theta), -np.eye(n_theta)])
    b_t = np.concatenate([theta_max, -theta_min]).reshape(-1, 1)
    return A_t, b_t


# ─────────────────────────────────────────────────────────────────────────────
#  CR expansion: private (31/37-dim) → full (58-dim) for combiner
# ─────────────────────────────────────────────────────────────────────────────

def _expand_cr(
    cr: AgentCR,
    n_x_neg: int,
    embed: tuple[int, ...],
    n_p_gne: int,
) -> AgentCR:
    """
    Expand a CR from private θ_i-space to full (n_x_neg + n_p_gne)-dim space.

    Private θ_i = [sum_x_neg(H) | p_priv(n_p_priv)]   n_x_neg = H = 6
    Full θ_i    = [sum_x_neg(H) | p_gne(n_p_gne)]      46-dim

    gne_combiner / facet_gne detect sum mode via cr.E.shape[1] - game.n_p = H = 6.
    """
    n_theta_full = n_x_neg + n_p_gne

    E_full = np.zeros((cr.E.shape[0], n_theta_full))
    A_full = np.zeros((cr.A.shape[0], n_theta_full))
    lambda_A_full = (None if cr.lambda_A is None
                     else np.zeros((cr.lambda_A.shape[0], n_theta_full)))

    # sum_x_neg block (unchanged — first n_x_neg=H cols)
    E_full[:, :n_x_neg] = cr.E[:, :n_x_neg]
    A_full[:, :n_x_neg] = cr.A[:, :n_x_neg]
    if lambda_A_full is not None:
        lambda_A_full[:, :n_x_neg] = cr.lambda_A[:, :n_x_neg]

    # Private p block: embed into p_gne positions
    for k, gne_idx in enumerate(embed):
        E_full[:, n_x_neg + gne_idx] += cr.E[:, n_x_neg + k]
        A_full[:, n_x_neg + gne_idx] += cr.A[:, n_x_neg + k]
        if lambda_A_full is not None:
            lambda_A_full[:, n_x_neg + gne_idx] += cr.lambda_A[:, n_x_neg + k]

    return AgentCR(
        E=E_full, f=cr.f.copy(),
        A=A_full, b=cr.b.copy(),
        lambda_A=lambda_A_full,
        lambda_b=(None if cr.lambda_b is None else cr.lambda_b.copy()),
        active_set=list(cr.active_set), n_constraints=cr.n_constraints,
        index=cr.index,
    )


def _expand_solution(
    sol: AgentSolution,
    pgne: PGNELayout,
    i: int,
) -> AgentSolution:
    """Expand all CRs from private (19/25-dim) to full (46-dim) θ_i space."""
    n_x_neg = pgne.H   # H = 6 (sum coupling dims)
    embed   = pgne.agent_embed[i]

    expanded_regions = [
        _expand_cr(cr, n_x_neg, embed, pgne.n_p_gne)
        for cr in sol.regions
    ]
    return AgentSolution(
        agent_index = sol.agent_index,
        n_x_i       = sol.n_x_i,
        n_theta_i   = n_x_neg + pgne.n_p_gne,   # 46
        regions     = expanded_regions,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Solve one / all agents
# ─────────────────────────────────────────────────────────────────────────────

def solve_agent_mpqp(
    game: GNEGame,
    i: int,
    layout: SimpleParameterLayout,
    pgne: PGNELayout,
    algorithm: mpqp_algorithm = DEFAULT_ALGORITHM,
    verbose: bool = True,
    lmp_lb: float = 0.0,
    lmp_ub: float = 250.0,
    da_lb: float | None = None,
    da_ub: float | None = None,
    pv_ub: float | None = None,
) -> AgentSolution:
    """
    Solve agent i's best-response mpQP with PRIVATE θ_i.

    Solves in (19 or 25)-dim space (fewer CRs), then expands to 46-dim
    for combiner compatibility.

    Returns AgentSolution with CRs in 46-dim θ_i-space (sum coupling).
    """
    th      = _private_theta_layout(i, pgne)
    n_theta = th["n_theta"]
    n_x_i   = game.agents[i].n_x

    Q, H_pp, c, G, b, F = build_agent_matrices(game, i, layout, pgne)
    A_t, b_t = build_parameter_space(game, i, layout, pgne, lmp_lb, lmp_ub,
                                      da_lb=da_lb, da_ub=da_ub, pv_ub=pv_ub)

    if verbose:
        lam_min = float(np.linalg.eigvalsh(Q).min())
        names   = ["VRFB", "PV+Batt", "PEM", "Alk"]
        print(f"\n[simple_mpqp] Agent {i} ({names[i]}):")
        print(f"  n_x={n_x_i}, n_theta_priv={n_theta} "
              f"[sum_x_neg={th['n_x_neg']}, p_priv={th['n_p_priv']}]")
        print(f"  n_constraints={G.shape[0]}  (24 local + 6 coupling-upper)")
        print(f"  lambda_min(Q)={lam_min:.4f}")
        print(f"  lmp bounds=[{lmp_lb}, {lmp_ub}] $/MWh")
        if da_lb is not None or da_ub is not None:
            print(f"  DA bounds=[{da_lb}, {da_ub}] kW (override)")

    problem = MPQP_Program(
        G,
        b.reshape(-1, 1),
        c.reshape(-1, 1),
        H_pp,
        Q,
        A_t,
        b_t,
        F,
    )

    solution = solve_mpqp(problem, algorithm=algorithm)
    n_cr = len(solution.critical_regions)

    if verbose:
        print(f"  → {n_cr} critical regions  (private {n_theta}-dim space, sum coupling)")

    sol_priv = agent_solution_from_ppopt(
        solution,
        agent_index = i,
        n_x_i       = n_x_i,
        n_theta_i   = n_theta,
        n_constraints = G.shape[0],
    )

    sol_full = _expand_solution(sol_priv, pgne, i)

    if verbose:
        print(f"  → expanded to 46-dim θ_i for combiner  ({sol_full.n_cr} CRs)")

    return sol_full


def solve_all_agents(
    game: GNEGame,
    layout: SimpleParameterLayout,
    pgne: PGNELayout,
    algorithm: mpqp_algorithm = DEFAULT_ALGORITHM,
    verbose: bool = True,
    lmp_lb: float = 0.0,
    lmp_ub: float = 250.0,
    da_lb_per_agent: list[float | None] | None = None,
    da_ub_per_agent: list[float | None] | None = None,
    pv_ub: float | None = None,
) -> list[AgentSolution]:
    """
    Solve mpQP for all N agents sequentially.
    Returns list[AgentSolution] with 46-dim CRs (sum coupling), ready for gne_combiner.

    da_lb_per_agent / da_ub_per_agent : per-agent DA schedule bounds (kW).
        None entries use the agent's full power limit.  Length must equal game.N.
    pv_ub : upper bound on PV forecast parameter (kW), applies to PV+Batt agent only.
    """
    solutions = []
    for i in range(game.N):
        da_lb_i = da_lb_per_agent[i] if da_lb_per_agent is not None else None
        da_ub_i = da_ub_per_agent[i] if da_ub_per_agent is not None else None
        sol = solve_agent_mpqp(
            game, i, layout, pgne,
            algorithm=algorithm,
            verbose=verbose,
            lmp_lb=lmp_lb,
            lmp_ub=lmp_ub,
            da_lb=da_lb_i,
            da_ub=da_ub_i,
            pv_ub=pv_ub,
        )
        solutions.append(sol)

    if verbose:
        total = sum(s.n_cr for s in solutions)
        names = ["VRFB", "PV+Batt", "PEM", "Alk"]
        print(f"\n[simple_mpqp] All agents solved — {total} total CRs")
        for s in solutions:
            print(f"  agent {s.agent_index} ({names[s.agent_index]}): {s.n_cr} CRs")

    return solutions


# ─────────────────────────────────────────────────────────────────────────────
#  p_gne bounds for pext_game construction (used by combiner / FACET scripts)
# ─────────────────────────────────────────────────────────────────────────────

def make_pgne_bounds(
    game: GNEGame,
    pgne: PGNELayout,
    lmp_lb: float = 0.0,
    lmp_ub: float = 250.0,
    da_lb_per_agent: list[float | None] | None = None,
    da_ub_per_agent: list[float | None] | None = None,
    pv_ub: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build p_gne_lb and p_gne_ub (each shape (n_p_gne,)) for constructing pext_game.

    Bounds must match those used in build_parameter_space / solve_all_agents
    so the Chebyshev LP in facet_gne and the mpQP parameter boxes are consistent.

    da_lb_per_agent / da_ub_per_agent : per-agent DA schedule bounds (kW).
        None entries use the agent's full power limit.  Length must equal game.N.
    pv_ub : upper bound on PV forecast parameter (kW), PV+Batt only.

    Returns (p_gne_lb, p_gne_ub).
    """
    H   = pgne.H
    lb  = np.zeros(pgne.n_p_gne)
    ub  = np.zeros(pgne.n_p_gne)

    for i, ag in enumerate(game.agents):
        # state: [0, b_loc[0]]  (SoC_max or Inv_max + q)
        lb[pgne.state_indices[i]] = 0.0
        ub[pgne.state_indices[i]] = float(ag.b_loc[0])

        # DA schedule — use override if provided, else agent power limits
        da_sl = slice(pgne.da_starts[i], pgne.da_starts[i] + H)
        da_lb_i = da_lb_per_agent[i] if da_lb_per_agent is not None else None
        da_ub_i = da_ub_per_agent[i] if da_ub_per_agent is not None else None
        lb[da_sl] = da_lb_i if da_lb_i is not None else -ag.b_loc[3 * H]
        ub[da_sl] = da_ub_i if da_ub_i is not None else  ag.b_loc[2 * H]

    # λ^RT: tight physical range
    ps = pgne.price_start
    lb[ps:ps + H] = lmp_lb
    ub[ps:ps + H] = lmp_ub

    # g^PV: [0, pv_ub or battery capacity]  (PV+Batt agent)
    if pgne.pv_start is not None:
        pv_ag = game.agents[1]
        pv_sl = slice(pgne.pv_start, pgne.pv_start + H)
        lb[pv_sl] = 0.0
        ub[pv_sl] = pv_ub if pv_ub is not None else float(pv_ag.b_loc[2 * H])

    return lb, ub


# ─────────────────────────────────────────────────────────────────────────────
#  Pack θ_i at runtime (for validation / ADMM comparison)
# ─────────────────────────────────────────────────────────────────────────────

def pack_theta_i(
    game: GNEGame,
    i: int,
    x_all: np.ndarray,
    p_gne: np.ndarray,
) -> np.ndarray:
    """
    Pack the FULL (58-dim) θ_i = [x_{-i}(18); p_gne(40)] for agent i.

    Used after expansion — for combiner evaluation and validation.
    x_all : (n_x_total=24,) stacked decisions of all agents
    p_gne : (40,) combined GNE parameter vector
    """
    others = [j for j in range(game.N) if j != i]
    x_neg  = np.concatenate([x_all[game.x_slice(j)] for j in others])
    return np.concatenate([x_neg, p_gne])


# =============================================================================
#  DAILY two-timescale mode — DA schedule fixed as constant
#
#  After DA clearing (once per day), the DA schedule p^DA_i is treated as a
#  fixed constant rather than a free parameter.  This removes 6 dims from θ_i:
#
#    Standard mode:  θ_i = [sum_x_neg(6) | state_i(1) | λ^RT(6) | DA_i(6) | PV?(6)]
#                          = 19 or 25-dim   →  46-dim after expansion
#
#    Daily mode:     θ_i = [sum_x_neg(6) | state_i(1) | λ^RT(6) | PV?(6)]
#                          = 13 or 19-dim   →  22-dim after expansion
#
#  p_gne_daily = [states(4) | λ^RT(6) | g^PV(6)] = 16-dim  (no DA, no L_max/L_min)
#
#  DA schedule is absorbed into the constant cost vector c:
#    c_eff[k] = c[k] − γ · p^DA_i[k]
#
#  CRs are recomputed once per day after DA clearing and valid for the full day.
# =============================================================================

from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True)
class PGNELayoutDaily:
    """
    Daily GNE parameter space with DA schedules fixed as constants.

    p_gne_daily ∈ ℝ^{n_p_gne}  (16-dim for N=4, H=6):
      [0..3]    initial states (SoC / H2 inventory)
      [4..9]    λ^RT prices (6 steps)
      [10..15]  g^PV forecast (6 steps)

    DA schedules are NOT in p_gne_daily — they are baked into CR affine maps.

    agent_embed[i] : indices into p_gne_daily for agent i's private params,
                     in the order they appear in p_priv_i after sum_x_neg.
      VRFB/PEM/Alk: [state_i(1), λ^RT(6)]             → n_p_priv = 7
      PV+Batt:      [state_i(1), λ^RT(6), g^PV(6)]    → n_p_priv = 13
    """
    n_p_gne:       int
    H:             int
    n_agents:      int
    state_indices: tuple[int, ...]
    price_start:   int
    pv_start:      int
    L_max_kw:      float
    agent_embed:   tuple[tuple[int, ...], ...]
    n_p_priv:      tuple[int, ...]


def make_pgne_layout_daily(
    layout: "SimpleParameterLayout",
    L_max_kw: float,
) -> PGNELayoutDaily:
    """Build the reduced daily p_gne layout (no DA schedules)."""
    N = layout.n_agents
    H = layout.H

    state_indices = tuple(range(N))          # [0, 1, 2, 3]
    price_start   = N                         # 4
    pv_start      = N + H                    # 10
    n_p_gne       = pv_start + H             # 16

    embeds = []
    for i in range(N):
        em  = [state_indices[i]]
        em += list(range(price_start, price_start + H))
        if i == 1:
            em += list(range(pv_start, pv_start + H))
        embeds.append(tuple(em))

    n_p_priv = tuple(len(em) for em in embeds)   # (7, 13, 7, 7)

    return PGNELayoutDaily(
        n_p_gne      = n_p_gne,
        H            = H,
        n_agents     = N,
        state_indices= state_indices,
        price_start  = price_start,
        pv_start     = pv_start,
        L_max_kw     = L_max_kw,
        agent_embed  = tuple(embeds),
        n_p_priv     = n_p_priv,
    )


def pack_pgne_daily(
    pgne_daily: PGNELayoutDaily,
    states: list[float],
    lmp_rt: np.ndarray,
    pv_forecast: np.ndarray | None = None,
) -> np.ndarray:
    """Pack the 16-dim daily GNE parameter vector (no DA, no L_max)."""
    p = np.zeros(pgne_daily.n_p_gne)
    for i, s in enumerate(states):
        p[pgne_daily.state_indices[i]] = s
    p[pgne_daily.price_start:pgne_daily.price_start + pgne_daily.H] = lmp_rt
    if pv_forecast is not None:
        p[pgne_daily.pv_start:pgne_daily.pv_start + pgne_daily.H] = pv_forecast
    return p


def _private_theta_layout_daily(i: int, pgne_daily: PGNELayoutDaily) -> dict:
    """
    Index layout for private θ_i in daily mode.

    θ_i_daily = [sum_x_neg(H) | state_i(1) | λ^RT(H) | (g^PV(H) if PV)]
                = 13 or 19-dim  (no DA block)
    """
    H       = pgne_daily.H
    n_x_neg = H
    is_pv   = (i == 1)
    n_p_priv = 1 + H + (H if is_pv else 0)
    n_theta  = n_x_neg + n_p_priv

    state_theta = n_x_neg
    price_theta = slice(n_x_neg + 1, n_x_neg + 1 + H)
    pv_theta    = slice(n_x_neg + 1 + H, n_theta) if is_pv else None

    return dict(
        n_x_neg     = n_x_neg,
        n_p_priv    = n_p_priv,
        n_theta     = n_theta,
        state_theta = state_theta,
        price_theta = price_theta,
        pv_theta    = pv_theta,
        is_pv       = is_pv,
    )


def build_agent_matrices_daily(
    game: GNEGame,
    i: int,
    layout: "SimpleParameterLayout",
    pgne_daily: PGNELayoutDaily,
    da_schedule_i: np.ndarray,
) -> tuple:
    """
    Build PPOPT (Q, H_pp, c, G, b, F) for agent i in daily mode.

    DA schedule is fixed: absorbed into c as  c_eff[k] = c[k] − γ · da_i[k].
    θ_i_daily = [sum_x_neg(H) | state_i(1) | λ^RT(H) | (g^PV(H) if PV)]

    Constraint rows (30 total, same as standard mode):
      0..23   local:    SoC/Inv bounds + power box
      24..29  coupling: p_i + sum_x_neg ≤ L_max  (upper only)
    """
    ai     = game.agents[i]
    H      = layout.H
    dt     = layout.dt_hr
    gamma  = float(ai.Q[0, 0])
    th     = _private_theta_layout_daily(i, pgne_daily)
    n_theta = th["n_theta"]

    Q = ai.Q.copy()
    # Absorb fixed DA reference into constant cost vector
    c = ai.c.copy() - gamma * np.asarray(da_schedule_i)[:H]

    # H_pp: only LMP energy cost (no DA term — DA is now constant)
    H_pp = np.zeros((H, n_theta))
    for k in range(H):
        H_pp[k, th["price_theta"].start + k] = dt

    # Local constraints — same structure as standard mode
    G_loc = ai.A_loc.copy()
    b_loc = ai.b_loc.copy()
    F_loc = np.zeros((4 * H, n_theta))

    state_full_idx = layout.state_init_indices[i]
    F_loc[:H,    th["state_theta"]] = ai.S_loc[:H,    state_full_idx]
    F_loc[H:2*H, th["state_theta"]] = ai.S_loc[H:2*H, state_full_idx]

    if th["is_pv"] and layout.pv_start is not None:
        pv_theta_sl = th["pv_theta"]
        pv_full_sl  = layout.pv_slice
        for k in range(H):
            F_loc[:H,    pv_theta_sl.start + k] = ai.S_loc[:H,    pv_full_sl.start + k]
            F_loc[H:2*H, pv_theta_sl.start + k] = ai.S_loc[H:2*H, pv_full_sl.start + k]
            F_loc[2*H+k, pv_theta_sl.start + k] = ai.S_loc[2*H+k, pv_full_sl.start + k]
            F_loc[3*H+k, pv_theta_sl.start + k] = ai.S_loc[3*H+k, pv_full_sl.start + k]

    # Coupling upper bound: p_i + sum_x_neg ≤ L_max
    G_coup = np.eye(H)
    b_coup = pgne_daily.L_max_kw * np.ones(H)
    F_coup = np.zeros((H, n_theta))
    F_coup[:H, :H] = -np.eye(H)   # sum_x_neg block

    G = np.vstack([G_loc, G_coup])
    b = np.concatenate([b_loc, b_coup])
    F = np.vstack([F_loc, F_coup])

    return Q, H_pp, c, G, b, F


def build_parameter_space_daily(
    game: GNEGame,
    i: int,
    layout: "SimpleParameterLayout",
    pgne_daily: PGNELayoutDaily,
    lmp_lb: float = 0.0,
    lmp_ub: float = 250.0,
    pv_ub: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Box constraints on θ_i_daily = [sum_x_neg(H) | state_i(1) | λ^RT(H) | (g^PV if PV)].

    No DA bounds (DA is a fixed constant, not a parameter).
    """
    H       = layout.H
    th      = _private_theta_layout_daily(i, pgne_daily)
    n_theta = th["n_theta"]

    theta_min = np.zeros(n_theta)
    theta_max = np.zeros(n_theta)

    # sum_x_neg: sum of other agents' power limits
    sum_lb = np.zeros(H)
    sum_ub = np.zeros(H)
    for j in range(game.N):
        if j == i:
            continue
        aj = game.agents[j]
        sum_ub += aj.b_loc[2 * H:3 * H]
        sum_lb -= aj.b_loc[3 * H:4 * H]
        if j == 1 and pv_ub is not None:
            sum_lb -= pv_ub
    theta_min[:H] = sum_lb
    theta_max[:H] = sum_ub

    # state_i: [0, state_max]
    ai = game.agents[i]
    theta_min[th["state_theta"]] = 0.0
    theta_max[th["state_theta"]] = float(ai.b_loc[0])

    # λ^RT
    ps = th["price_theta"]
    theta_min[ps] = lmp_lb
    theta_max[ps] = lmp_ub

    # g^PV (PV+Batt only)
    if th["is_pv"]:
        pv = th["pv_theta"]
        theta_min[pv] = 0.0
        theta_max[pv] = pv_ub if pv_ub is not None else float(ai.b_loc[2 * H])

    A_t = np.vstack([ np.eye(n_theta), -np.eye(n_theta)])
    b_t = np.concatenate([theta_max, -theta_min]).reshape(-1, 1)
    return A_t, b_t


def _expand_cr_daily(
    cr: "AgentCR",
    n_x_neg: int,
    embed: tuple[int, ...],
    n_p_gne_daily: int,
) -> "AgentCR":
    """
    Expand CR from private (13/19-dim) θ_i_daily to full (22-dim) daily space.

    Private: [sum_x_neg(6) | p_priv(7 or 13)]
    Full:    [sum_x_neg(6) | p_gne_daily(16)]   → 22-dim
    """
    n_theta_full = n_x_neg + n_p_gne_daily   # 6 + 16 = 22

    E_full = np.zeros((cr.E.shape[0], n_theta_full))
    A_full = np.zeros((cr.A.shape[0], n_theta_full))
    lambda_A_full = (None if cr.lambda_A is None
                     else np.zeros((cr.lambda_A.shape[0], n_theta_full)))

    E_full[:, :n_x_neg] = cr.E[:, :n_x_neg]
    A_full[:, :n_x_neg] = cr.A[:, :n_x_neg]
    if lambda_A_full is not None:
        lambda_A_full[:, :n_x_neg] = cr.lambda_A[:, :n_x_neg]

    for k, gne_idx in enumerate(embed):
        E_full[:, n_x_neg + gne_idx] += cr.E[:, n_x_neg + k]
        A_full[:, n_x_neg + gne_idx] += cr.A[:, n_x_neg + k]
        if lambda_A_full is not None:
            lambda_A_full[:, n_x_neg + gne_idx] += cr.lambda_A[:, n_x_neg + k]

    return AgentCR(
        E=E_full, f=cr.f.copy(),
        A=A_full, b=cr.b.copy(),
        lambda_A=lambda_A_full,
        lambda_b=(None if cr.lambda_b is None else cr.lambda_b.copy()),
        active_set=list(cr.active_set), n_constraints=cr.n_constraints,
        index=cr.index,
    )


def _expand_solution_daily(
    sol: "AgentSolution",
    pgne_daily: PGNELayoutDaily,
    i: int,
) -> "AgentSolution":
    """Expand all CRs from private (13/19-dim) to full 22-dim daily θ_i space."""
    n_x_neg = pgne_daily.H
    embed   = pgne_daily.agent_embed[i]

    expanded = [
        _expand_cr_daily(cr, n_x_neg, embed, pgne_daily.n_p_gne)
        for cr in sol.regions
    ]
    return AgentSolution(
        agent_index = sol.agent_index,
        n_x_i       = sol.n_x_i,
        n_theta_i   = n_x_neg + pgne_daily.n_p_gne,   # 22
        regions     = expanded,
    )


def solve_agent_mpqp_daily(
    game: GNEGame,
    i: int,
    layout: "SimpleParameterLayout",
    pgne_daily: PGNELayoutDaily,
    da_schedule_i: np.ndarray,
    algorithm: "mpqp_algorithm" = None,
    verbose: bool = True,
    lmp_lb: float = 0.0,
    lmp_ub: float = 250.0,
    pv_ub: float | None = None,
) -> "AgentSolution":
    """
    Solve agent i's best-response mpQP in daily mode (DA fixed as constant).

    Solves in (13 or 19)-dim private θ_i_daily space, then expands to 22-dim
    for combiner compatibility.

    da_schedule_i : (H,) array — daily representative DA reference for agent i.
                    Absorbed into the constant cost c; not a free parameter.
    """
    if algorithm is None:
        algorithm = DEFAULT_ALGORITHM

    th      = _private_theta_layout_daily(i, pgne_daily)
    n_theta = th["n_theta"]
    n_x_i   = game.agents[i].n_x

    Q, H_pp, c, G, b, F = build_agent_matrices_daily(
        game, i, layout, pgne_daily, da_schedule_i
    )
    A_t, b_t = build_parameter_space_daily(
        game, i, layout, pgne_daily, lmp_lb, lmp_ub, pv_ub=pv_ub
    )

    if verbose:
        names = ["VRFB", "PV+Batt", "PEM", "Alk"]
        lam_min = float(np.linalg.eigvalsh(Q).min())
        print(f"\n[daily_mpqp] Agent {i} ({names[i]}):")
        print(f"  n_x={n_x_i}, n_theta_priv={n_theta} "
              f"[sum_x_neg={th['n_x_neg']}, p_priv={th['n_p_priv']}]  (no DA dim)")
        print(f"  n_constraints={G.shape[0]}  lambda_min(Q)={lam_min:.4f}")
        print(f"  lmp_bounds=[{lmp_lb}, {lmp_ub}]  "
              f"da_fixed_mean={float(np.mean(da_schedule_i)):.1f} kW")

    problem = MPQP_Program(
        G, b.reshape(-1, 1), c.reshape(-1, 1), H_pp, Q, A_t, b_t, F,
    )

    solution  = solve_mpqp(problem, algorithm=algorithm)
    n_cr      = len(solution.critical_regions)

    if verbose:
        print(f"  → {n_cr} critical regions  (private {n_theta}-dim, daily mode)")

    sol_priv = agent_solution_from_ppopt(
        solution,
        agent_index = i,
        n_x_i       = n_x_i,
        n_theta_i   = n_theta,
        n_constraints = G.shape[0],
    )

    sol_full = _expand_solution_daily(sol_priv, pgne_daily, i)

    if verbose:
        print(f"  → expanded to 22-dim θ_i_daily  ({sol_full.n_cr} CRs)")

    return sol_full


def solve_all_agents_daily(
    game: GNEGame,
    layout: "SimpleParameterLayout",
    pgne_daily: PGNELayoutDaily,
    da_schedules_fixed: list[np.ndarray],
    algorithm: "mpqp_algorithm" = None,
    verbose: bool = True,
    lmp_lb: float = 0.0,
    lmp_ub: float = 250.0,
    pv_ub: float | None = None,
) -> list["AgentSolution"]:
    """
    Solve mpQP for all N agents in daily mode (DA fixed per agent).

    da_schedules_fixed : list of (H,) arrays, one per agent.
        Typically the daily mean of each agent's DA schedule, broadcast to H steps.

    Returns list[AgentSolution] with 22-dim CRs, ready for solve_gne_online_v2.
    """
    if algorithm is None:
        algorithm = DEFAULT_ALGORITHM

    solutions = []
    for i in range(game.N):
        sol = solve_agent_mpqp_daily(
            game, i, layout, pgne_daily,
            da_schedule_i = np.asarray(da_schedules_fixed[i])[:layout.H],
            algorithm     = algorithm,
            verbose       = verbose,
            lmp_lb        = lmp_lb,
            lmp_ub        = lmp_ub,
            pv_ub         = pv_ub,
        )
        solutions.append(sol)

    if verbose:
        total = sum(s.n_cr for s in solutions)
        names = ["VRFB", "PV+Batt", "PEM", "Alk"]
        print(f"\n[daily_mpqp] All agents solved — {total} total CRs (daily mode)")
        for s in solutions:
            print(f"  agent {s.agent_index} ({names[s.agent_index]}): {s.n_cr} CRs")

    return solutions


def make_pgne_bounds_daily(
    game: GNEGame,
    pgne_daily: PGNELayoutDaily,
    lmp_lb: float = 0.0,
    lmp_ub: float = 250.0,
    pv_ub: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build p_gne_daily_lb and p_gne_daily_ub (each shape (16,)) for pext_game_daily.

    Bounds mirror those in build_parameter_space_daily.
    """
    H  = pgne_daily.H
    lb = np.zeros(pgne_daily.n_p_gne)
    ub = np.zeros(pgne_daily.n_p_gne)

    for i, ag in enumerate(game.agents):
        lb[pgne_daily.state_indices[i]] = 0.0
        ub[pgne_daily.state_indices[i]] = float(ag.b_loc[0])

    ps = pgne_daily.price_start
    lb[ps:ps + H] = lmp_lb
    ub[ps:ps + H] = lmp_ub

    pv_ag = game.agents[1]
    pv_sl = slice(pgne_daily.pv_start, pgne_daily.pv_start + H)
    lb[pv_sl] = 0.0
    ub[pv_sl] = pv_ub if pv_ub is not None else float(pv_ag.b_loc[2 * H])

    return lb, ub
