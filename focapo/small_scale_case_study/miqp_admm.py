"""
miqp_admm.py  –  Unified MIQP solver for semi-continuous electrolyzer bidding.

Both DAM and RTM formulations contain TWO sets of binaries:
  u_{i}   ∈ {0,1}   hourly participation  →  p_i ∈ {0} ∪ [L_min, P_max]
  δ_{i,k} ∈ {0,1}   battery mode          →  p_ch·p_dis = 0 (no simultaneous)

GNE solver — distributed proximal best-response MIQP-ADMM
  Each agent solves its own local MIQP per round (+proximal term).
  Only the aggregate Σp is exchanged between agents.
  Converges to ε-equilibrium for exact potential games (Fabiani et al. 2022).
  This game is an exact potential: costs are separable and the only coupling
  is the shared constraint Σp ≥ L_min, so the variational GNE = joint optimum
  (Facchinei & Kanzow 2010).

Individual solver — N independent single-agent MIQPs; no coupling.

PJM 65-min rule
  u is committed once per operating hour via a dedicated MIQP (GNE-coordinated
  or per-agent) at the 65-min gate, then held FIXED for the twelve 5-min
  dispatch steps.  δ is free every 5-min step (battery re-decided with live SoC).

References
  Fabiani, Franci, Sagratella, Schmidt, Staudigl (2022)
    "Proximal-like algorithms for equilibrium seeking in mixed-integer Nash
     equilibrium problems." arXiv:2203.15410
  Boyd, Parikh, Chu, Peleato, Eckstein (2011) "Distributed optimization and
    statistical learning via ADMM." Found. Trends Mach. Learn. 3(1).
  Facchinei & Kanzow (2010) "Generalized Nash equilibrium problems."
    4OR 5(3):173-210.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from agents import ElzAgent


# ── Gurobi environment ─────────────────────────────────────────────────────────
_GRB_ENV: Optional[gp.Env] = None

def _env() -> gp.Env:
    global _GRB_ENV
    if _GRB_ENV is None:
        _GRB_ENV = gp.Env(empty=True)
        _GRB_ENV.setParam('OutputFlag',   0)
        _GRB_ENV.setParam('LogToConsole', 0)
        _GRB_ENV.setParam('Threads',      2)
        _GRB_ENV.start()
    return _GRB_ENV


# ── Result types ───────────────────────────────────────────────────────────────
@dataclass
class AgentSol:
    p:        np.ndarray   # (H,) net grid import [kW]
    p_ch:     np.ndarray   # (H,) battery charging [kW]
    p_dis:    np.ndarray   # (H,) battery discharging [kW]
    u:        int          # hourly participation binary
    feasible: bool


@dataclass
class GNEResult:
    p:          np.ndarray   # (N, H)
    p_ch:       np.ndarray   # (N, H)
    p_dis:      np.ndarray   # (N, H)
    u:          np.ndarray   # (N,) int
    n_iter:     int
    converged:  bool
    primal_res: float
    dual_res:   float


@dataclass
class IndResult:
    p:     np.ndarray   # (N, H)
    p_ch:  np.ndarray   # (N, H)
    p_dis: np.ndarray   # (N, H)
    u:     np.ndarray   # (N,) int


# ── Optional-coalition aggregate projection ────────────────────────────────────
def _project_sc_agg(s: np.ndarray, l_min: float, l_max: float) -> np.ndarray:
    """
    Project per-step aggregate s_k onto the optional coalition set {0}∪[l_min,l_max].

    For each step k the nearest point in {0}∪[l_min,l_max] to s_k is:
      0           if |s_k| ≤ distance(s_k, [l_min,l_max])
      clip(s,…)   otherwise
    """
    if l_min <= 0.0:
        return np.clip(s, 0.0, l_max)
    s_star    = np.clip(s, l_min, l_max)
    dist_zero = np.abs(s)
    dist_iv   = np.where(s < l_min, l_min - s,
                         np.where(s > l_max, s - l_max, 0.0))
    return np.where(dist_zero <= dist_iv, 0.0, s_star)


# ── Core: single-agent MIQP ────────────────────────────────────────────────────
def _solve_agent(
    ag:        ElzAgent,
    H:         int,
    dt:        float,
    E_0:       float,          # initial battery SoC [kWh]
    lam:       np.ndarray,     # (H,) price forecast [$/MWh]
    p_da:      float,          # DA anchor [kW]
    g_ren:     np.ndarray,     # (H,) renewable forecast [kW]
    rho:       float,          # ADMM augmentation penalty
    z_i:       np.ndarray,     # (H,) ADMM consensus target
    mu_i:      np.ndarray,     # (H,) ADMM dual variable
    u_fixed:   Optional[int],  # None = u is a variable; 0/1 = fixed participation
    gamma:     float,          # deviation penalty (0 for DAM, ag.gamma for RTM)
    add_h2:    bool,           # True → add hard daily H₂ floor (DAM only)
    l_min:     float,
) -> AgentSol:
    """
    Single-agent MIQP with ADMM augmentation and proximal term.

    Objective (per step k):
      (γ+ρ)/2 · p_k²  +  [c_k + μ_k − γ·p_DA − ρ·z_k] · p_k
      − r·η·dt · p_dis_k                    (H₂ revenue from discharge)
      + r·η·dt/η_rt · p_ch_k               (H₂ opportunity cost of charge)
      + r·η·dt · cv_k                       (curtailment opportunity cost)

    The (ρ/2)‖p−z‖² ADMM proximal term is what provides convergence for the
    MI case (Fabiani et al. 2022, Theorem 1) when the game has an exact potential.

    Binaries:
      u    ∈ {0,1}    hourly participation  (variable if u_fixed is None)
      δ_k  ∈ {0,1}    battery mode per step (always a variable when has_battery)
    """
    c    = (lam / 1000.0 - ag.r_h2 * ag.eta_kg_per_kwh) * dt   # (H,) linear cost
    h2dt = ag.r_h2 * ag.eta_kg_per_kwh * dt
    p_ub = ag.p_elec_max_kw + (ag.bat_p_ch_max_kw if ag.has_battery else 0.0)

    m = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)

    p = m.addVars(H, lb=0.0, ub=float(p_ub), name='p')

    # ── semi-continuous participation ─────────────────────────────────────────
    if u_fixed is None:
        u_var = m.addVar(vtype=GRB.BINARY, name='u')
        for k in range(H):
            m.addConstr(p[k] >= float(l_min) * u_var, name=f'sc_lb_{k}')
            m.addConstr(p[k] <= float(p_ub)  * u_var, name=f'sc_ub_{k}')
    elif u_fixed == 0:
        u_var = None
        for k in range(H):
            m.addConstr(p[k] == 0.0, name=f'out_{k}')
    else:  # u_fixed == 1
        u_var = None
        for k in range(H):
            m.addConstr(p[k] >= float(l_min), name=f'lb_{k}')

    # ── renewable curtailment ─────────────────────────────────────────────────
    if ag.has_renewable:
        cv = m.addVars(H, lb=0.0, name='cv')
        for k in range(H):
            m.addConstr(cv[k] <= max(0.0, float(g_ren[k])), name=f'cv_ub_{k}')
    else:
        cv = None

    # ── battery variables with exact charge/discharge exclusivity ─────────────
    if ag.has_battery:
        pch  = m.addVars(H, lb=0.0, ub=ag.bat_p_ch_max_kw,  name='pch')
        pdis = m.addVars(H, lb=0.0, ub=ag.bat_p_dis_max_kw, name='pdis')
        E    = m.addVars(H + 1, lb=0.0, ub=ag.bat_cap_kwh,  name='E')
        d    = m.addVars(H, vtype=GRB.BINARY, name='d')   # 1=charge, 0=discharge
        m.addConstr(E[0] == float(E_0), name='soc0')
        for k in range(H):
            m.addConstr(pch[k]  <= ag.bat_p_ch_max_kw  * d[k],       name=f'me_ch_{k}')
            m.addConstr(pdis[k] <= ag.bat_p_dis_max_kw * (1 - d[k]), name=f'me_dis_{k}')
            m.addConstr(
                E[k + 1] == E[k]
                          + ag.bat_eta_ch * pch[k] * dt
                          - pdis[k] / ag.bat_eta_dis * dt,
                name=f'soc_{k}'
            )
    else:
        pch = pdis = E = d = None

    # ── electrolyzer bounds ───────────────────────────────────────────────────
    for k in range(H):
        gk   = float(g_ren[k])
        cvk  = cv[k]   if cv   is not None else 0.0
        disk = pdis[k] if pdis is not None else 0.0
        chk  = pch[k]  if pch  is not None else 0.0
        pe   = p[k] + gk - cvk + disk - chk
        m.addConstr(pe >= ag.p_elec_min_kw, name=f'elb_{k}')
        m.addConstr(pe <= ag.p_elec_max_kw, name=f'eub_{k}')

    # ── hard H₂ daily floor (DAM only) ───────────────────────────────────────
    if add_h2 and ag.h2_target_kg_day > 0:
        h2_expr = gp.LinExpr()
        for k in range(H):
            e_dt = ag.eta_kg_per_kwh * dt
            gk   = float(g_ren[k])
            h2_expr.add(p[k], e_dt)
            h2_expr.addConstant(gk * e_dt)
            if cv   is not None: h2_expr.add(cv[k],   -e_dt)
            if pdis is not None: h2_expr.add(pdis[k],  e_dt)
            if pch  is not None: h2_expr.add(pch[k],  -e_dt)
        m.addConstr(h2_expr >= ag.h2_target_kg_day, name='h2_floor')

    # ── objective: cost + ADMM augmentation ──────────────────────────────────
    obj = gp.QuadExpr()
    for k in range(H):
        q = 0.5 * (gamma + rho)
        l = (float(c[k]) + float(mu_i[k])
             - gamma * float(p_da)
             - rho   * float(z_i[k]))
        obj.add(q * p[k] * p[k])
        obj.add(l * p[k])
        if ag.has_battery:
            eta_rt = ag.bat_eta_ch * ag.bat_eta_dis
            obj.add(-h2dt * pdis[k])
            obj.add((h2dt / eta_rt) * pch[k])
        if cv is not None:
            obj.add(h2dt * cv[k])
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        p_sol    = np.array([p[k].X for k in range(H)])
        pch_sol  = np.array([pch[k].X  for k in range(H)]) if pch  is not None else np.zeros(H)
        pdis_sol = np.array([pdis[k].X for k in range(H)]) if pdis is not None else np.zeros(H)
        if u_var is not None:
            u_val = int(round(u_var.X))
        else:
            u_val = 0 if u_fixed == 0 else 1
        ok = True
    else:
        p_sol    = np.zeros(H) if u_fixed == 0 else np.full(H, float(l_min))
        pch_sol  = np.zeros(H)
        pdis_sol = np.zeros(H)
        u_val    = 0 if u_fixed == 0 else 1
        ok       = False

    m.dispose()
    return AgentSol(p=p_sol, p_ch=pch_sol, p_dis=pdis_sol, u=u_val, feasible=ok)


# ── GNE: distributed proximal best-response MIQP-ADMM ─────────────────────────
def solve_gne(
    agents:   list[ElzAgent],
    lam:      np.ndarray,           # (H,) price forecast [$/MWh]
    p_da:     np.ndarray,           # (N,) DA anchor per agent [kW]
    g_ren:    np.ndarray,           # (N,H) renewable forecast [kW]
    H:        int,
    dt:       float,
    bat_soc:  np.ndarray,           # (N,) current battery SoC [kWh]
    rho:      float,
    max_iter: int,
    tol:      float,
    l_min:    float,
    l_max:    float,
    u_fixed:  Optional[np.ndarray], # (N,) int or None (u free)
    mode:     str,                  # 'dam' or 'rtm'
) -> GNEResult:
    """
    Distributed proximal best-response MIQP-ADMM for GNE.

    Algorithm (Fabiani et al. 2022 adapted):
      Initialise: p = 0, z = 0, μ = 0
      Repeat:
        x-update: each agent i solves its local MIQP (in parallel)
                  with (ρ/2)‖p_i − z_i‖² proximal/ADMM term
        z-update: project aggregate V = Σ(p_i + μ_i/ρ) onto {0}∪[l_min,l_max]
                  distribute correction equally: z_i = V_i + (s*−s)/N
        μ-update: μ_i += ρ(p_i − z_i)
      Until: max_i ‖p_i^{t+1} − p_i^t‖∞ < tol

    Convergence: guaranteed to ε-equilibrium for exact potential games.
    This game is an exact potential (separable costs + shared coupling constraint).
    """
    N         = len(agents)
    add_h2    = (mode == 'dam')
    gamma_arr = [0.0 if mode == 'dam' else ag.gamma for ag in agents]

    # Warm-start z from the per-agent unconstrained minimiser (individual-optimal
    # point), not from zero.  Starting at z=0 creates a downward bias: agents are
    # pulled toward p=0 until ADMM builds z up to the true solution.  Warm-starting
    # at the individual optimal means z begins near the true answer, requiring far
    # fewer iterations and eliminating the systematic import-suppression bias.
    c_warm   = np.stack([ag.linear_cost(lam, dt) for ag in agents])   # (N,H)
    p_warm   = []
    gamma_arr = [0.0 if mode == 'dam' else ag.gamma for ag in agents]
    for i, ag in enumerate(agents):
        g_i   = gamma_arr[i]
        if g_i + rho > 1e-12:
            p_unc = (g_i * float(p_da[i]) - c_warm[i]) / (g_i + rho)
        else:
            p_unc = np.zeros(H)
        # Clip to per-agent semi-continuous feasible range:
        # u_fixed=0 → p=0; u_fixed=1 → [0, P_max]; u_fixed=None → [0, P_max]
        # (L_min is an aggregate constraint, not an individual one)
        if u_fixed is not None and int(u_fixed[i]) == 0:
            p_unc_clipped = np.zeros(H)
        else:
            p_unc_clipped = np.clip(p_unc, 0.0, ag.p_elec_max_kw)
        p_warm.append(p_unc_clipped)

    p     = [pw.copy() for pw in p_warm]
    p_ch  = [np.zeros(H) for _ in range(N)]
    p_dis = [np.zeros(H) for _ in range(N)]
    u_sol = np.ones(N, dtype=int)
    z     = [pw.copy() for pw in p_warm]   # warm-start z at individual optimal
    mu    = [np.zeros(H) for _ in range(N)]

    primal_res = dual_res = np.inf
    converged  = False
    itr        = 0

    for itr in range(max_iter):
        z_prev = [zi.copy() for zi in z]
        p_prev = [pi.copy() for pi in p]

        # ── x-update: each agent solves its local MIQP ───────────────────────
        for i, ag in enumerate(agents):
            u_i = None if u_fixed is None else int(u_fixed[i])
            sol = _solve_agent(
                ag, H, dt, float(bat_soc[i]),
                lam, float(p_da[i]), g_ren[i],
                rho, z[i], mu[i],
                u_fixed=u_i, gamma=gamma_arr[i],
                add_h2=add_h2, l_min=0.0,  # GNE local problem has no L_min (handled by ADMM z)
            )
            p[i]     = sol.p
            p_ch[i]  = sol.p_ch
            p_dis[i] = sol.p_dis
            u_sol[i] = sol.u

        # ── z-update: project aggregate, distribute ONLY over in-market agents ─
        # When u_fixed is provided, out-of-market agents have p=0 fixed and
        # cannot absorb a z-correction. Distributing δ/N to them wastes the
        # coupling signal and under-corrects in-market agents (they get δ/N
        # instead of δ/N_in).  Fix: restrict both the aggregate and the
        # correction to in-market agents; keep z=0, μ=0 for out-of-market ones.
        if u_fixed is not None:
            in_mkt = [i for i in range(N) if int(u_fixed[i]) == 1]
        else:
            in_mkt = list(range(N))

        if in_mkt:
            V_in   = np.stack([p[i] + mu[i] / rho for i in in_mkt])  # (N_in,H)
            s      = V_in.sum(axis=0)                                   # (H,)
            if u_fixed is not None:
                # u is locked: coalition is in/out, floor applies only to in-market
                s_star = np.clip(s, l_min if l_min > 0 else 0.0, l_max)
            else:
                # commitment phase: optional coalition semi-continuous projection
                s_star = _project_sc_agg(s, l_min, l_max)
            delta = (s_star - s) / len(in_mkt)
            for idx, i in enumerate(in_mkt):
                z[i] = V_in[idx] + delta
        # out-of-market agents: z=0, μ=0 (no coupling signal needed)
        for i in range(N):
            if i not in in_mkt:
                z[i]  = np.zeros(H)
                mu[i] = np.zeros(H)

        # ── μ-update (in-market agents only) ─────────────────────────────────
        for i in in_mkt:
            mu[i] = mu[i] + rho * (p[i] - z[i])

        # ── convergence: measure only in-market agents ────────────────────────
        # Out-of-market agents always have p=0 (trivially "converged"); including
        # them would mask slow convergence of the agents that actually matter.
        if in_mkt:
            primal_res = float(max(np.max(np.abs(p[i] - p_prev[i])) for i in in_mkt))
            dual_res   = float(rho * max(np.max(np.abs(z[i] - z_prev[i])) for i in in_mkt))
        else:
            primal_res = 0.0; dual_res = 0.0
        converged = primal_res < tol
        if converged:
            break

    return GNEResult(
        p         = np.array(p),
        p_ch      = np.array(p_ch),
        p_dis     = np.array(p_dis),
        u         = u_sol,
        n_iter    = itr + 1,
        converged = converged,
        primal_res= primal_res,
        dual_res  = dual_res,
    )


# ── Individual: N independent agent MIQPs ─────────────────────────────────────
def solve_individual(
    agents:  list[ElzAgent],
    lam:     np.ndarray,           # (H,)
    p_da:    np.ndarray,           # (N,)
    g_ren:   np.ndarray,           # (N,H)
    H:       int,
    dt:      float,
    bat_soc: np.ndarray,           # (N,)
    l_min:   float,
    u_fixed: Optional[np.ndarray], # (N,) int or None
    mode:    str,
) -> IndResult:
    """Solve N independent single-agent MIQPs (no coupling, no ADMM)."""
    N         = len(agents)
    add_h2    = (mode == 'dam')
    p     = np.zeros((N, H))
    p_ch  = np.zeros((N, H))
    p_dis = np.zeros((N, H))
    u_out = np.ones(N, dtype=int)

    for i, ag in enumerate(agents):
        gamma = 0.0 if mode == 'dam' else ag.gamma
        u_i   = None if u_fixed is None else int(u_fixed[i])
        sol   = _solve_agent(
            ag, H, dt, float(bat_soc[i]),
            lam, float(p_da[i]), g_ren[i],
            rho=0.0, z_i=np.zeros(H), mu_i=np.zeros(H),
            u_fixed=u_i, gamma=gamma,
            add_h2=add_h2, l_min=l_min,
        )
        p[i]     = sol.p
        p_ch[i]  = sol.p_ch
        p_dis[i] = sol.p_dis
        u_out[i] = sol.u

    return IndResult(p=p, p_ch=p_ch, p_dis=p_dis, u=u_out)


# ── Hour-ahead commitment: GNE (coordinated MIQP) ─────────────────────────────
def commit_u_gne(
    agents:  list[ElzAgent],
    lam:     np.ndarray,    # (H,) hour-ahead price forecast
    g_ren:   np.ndarray,    # (N,H) hour-ahead renewable forecast
    H:       int,
    dt:      float,
    bat_soc: np.ndarray,    # (N,) SoC at commitment time
    l_min:   float,
    l_max:   float,
) -> np.ndarray:
    """
    Coordinated GNE hourly commitment (PJM 65-min gate).

    One centralized MIQP over all N agents for the upcoming operating hour.
    Binaries: coalition gate U, per-agent u_i (with u_i ≤ U), battery δ.
    Economic objective (γ=0) — decides whether participating this hour is
    collectively profitable.  Only u is returned; continuous vars are discarded.

    Optional participation: U=0 → whole coalition opts out (Σp=0).
    """
    N = len(agents)
    m = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)

    U = m.addVar(vtype=GRB.BINARY, name='U')   # coalition gate

    p_v   = {}   # agent i → addVars(H)
    u_v   = {}   # agent i → addVar BINARY
    pch_v = {}
    pdis_v = {}
    cv_v  = {}

    for i, ag in enumerate(agents):
        p_ub = ag.p_elec_max_kw + (ag.bat_p_ch_max_kw if ag.has_battery else 0.0)
        p_v[i] = m.addVars(H, lb=0.0, ub=float(p_ub), name=f'p{i}')
        u_v[i] = m.addVar(vtype=GRB.BINARY, name=f'u{i}')
        m.addConstr(u_v[i] <= U, name=f'gate{i}')

        for k in range(H):
            # No individual l_min floor! Only p_v <= p_ub * u_v
            m.addConstr(p_v[i][k] <= float(p_ub)  * u_v[i], name=f'sc_ub{i}_{k}')

        if ag.has_renewable:
            cv_v[i] = m.addVars(H, lb=0.0, name=f'cv{i}')
            for k in range(H):
                m.addConstr(cv_v[i][k] <= max(0.0, float(g_ren[i][k])), name=f'cvub{i}_{k}')

        if ag.has_battery:
            pch  = m.addVars(H, lb=0.0, ub=ag.bat_p_ch_max_kw,  name=f'pch{i}')
            pdis = m.addVars(H, lb=0.0, ub=ag.bat_p_dis_max_kw, name=f'pdis{i}')
            E    = m.addVars(H + 1, lb=0.0, ub=ag.bat_cap_kwh,  name=f'E{i}')
            d    = m.addVars(H, vtype=GRB.BINARY, name=f'd{i}')
            pch_v[i] = pch;  pdis_v[i] = pdis
            m.addConstr(E[0] == float(bat_soc[i]), name=f'soc0_{i}')
            for k in range(H):
                m.addConstr(pch[k]  <= ag.bat_p_ch_max_kw  * d[k],       name=f'mech{i}_{k}')
                m.addConstr(pdis[k] <= ag.bat_p_dis_max_kw * (1 - d[k]), name=f'medis{i}_{k}')
                m.addConstr(
                    E[k + 1] == E[k]
                              + ag.bat_eta_ch * pch[k] * dt
                              - pdis[k] / ag.bat_eta_dis * dt,
                    name=f'soc{i}_{k}'
                )

        for k in range(H):
            gk   = float(g_ren[i][k])
            cvk  = cv_v[i][k]  if i in cv_v  else 0.0
            disk = pdis_v[i][k] if i in pdis_v else 0.0
            chk  = pch_v[i][k]  if i in pch_v  else 0.0
            pe   = p_v[i][k] + gk - cvk + disk - chk
            m.addConstr(pe >= ag.p_elec_min_kw, name=f'elb{i}_{k}')
            m.addConstr(pe <= ag.p_elec_max_kw, name=f'eub{i}_{k}')

    # PCC coupling — gated by coalition binary U
    for k in range(H):
        agg_k = gp.quicksum(p_v[i][k] for i in range(N))
        m.addConstr(agg_k >= float(l_min) * U, name=f'coup_lo_{k}')
        if np.isfinite(l_max):
            m.addConstr(agg_k <= float(l_max), name=f'coup_hi_{k}')

    # Economic objective (γ=0)
    obj = gp.LinExpr()
    for i, ag in enumerate(agents):
        c_i  = (lam / 1000.0 - ag.r_h2 * ag.eta_kg_per_kwh) * dt
        h2dt = ag.r_h2 * ag.eta_kg_per_kwh * dt
        for k in range(H):
            obj.add(p_v[i][k], float(c_i[k]))
            if i in pdis_v:
                eta_rt = ag.bat_eta_ch * ag.bat_eta_dis
                obj.add(pdis_v[i][k], -h2dt)
                obj.add(pch_v[i][k],   h2dt / eta_rt)
            if i in cv_v:
                obj.add(cv_v[i][k], h2dt)
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        u_out = np.array([int(round(u_v[i].X)) for i in range(N)], dtype=int)
    else:
        u_out = np.zeros(N, dtype=int)   # opt-out (always feasible)
    m.dispose()
    return u_out


# ── Hour-ahead commitment: individual per-agent ────────────────────────────────
def commit_u_individual(
    ag:    ElzAgent,
    lam:   np.ndarray,   # (H,) hour-ahead price forecast
    g_ren: np.ndarray,   # (H,) hour-ahead renewable forecast
    H:     int,
    dt:    float,
    E_0:   float,
    l_min: float,
) -> int:
    """
    Individual agent hourly commitment (PJM 65-min gate).
    Solves agent's MIQP with u free and γ=0 (pure economic decision).
    Returns u ∈ {0,1}.
    """
    sol = _solve_agent(
        ag, H, dt, E_0, lam, 0.0, g_ren,
        rho=0.0, z_i=np.zeros(H), mu_i=np.zeros(H),
        u_fixed=None, gamma=0.0, add_h2=False, l_min=l_min,
    )
    return sol.u
