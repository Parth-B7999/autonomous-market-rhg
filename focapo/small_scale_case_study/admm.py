"""
admm.py — Multi-agent ADMM with Gurobi QP for both DAM and RTM x-updates.

Supports four agent types:
  - Electrolyzer only
  - Renewable + Electrolyzer
  - Battery + Electrolyzer
  - Battery + Renewable + Electrolyzer

Problem
-------
  min_{p_1,...,p_N}  sum_i  f_i(p_i, p_ch_i, p_dis_i)
  s.t.  L_MIN <= sum_i p_i,k <= L_MAX   (bilateral PCC coupling, per step k)

Cost (includes H2 revenue from battery discharge):
  f_i = sum_k {
    0.5*gamma_i*(p_i,k - p_DA_i)^2
    + c_i,k * p_i,k                           [grid import cost - H2 revenue from p_i]
    - r_h2*eta_i*dt * p_dis_i,k               [H2 revenue from battery discharge]
    + r_h2*eta_i*dt * p_ch_i,k                [H2 opportunity cost of battery charging]
  }
  c_i,k = (lambda_k/1000 - r_h2*eta_i) * dt

ADMM iterations
---------------
  p_i-update  [Gurobi QP, mode-dependent]:

    Non-battery (H=12 RTM or H=24 DAM):
      min  sum_k { 0.5*(gamma+rho)*(p_k)^2  +  q_k*p_k }
      s.t. lb_k <= p_k <= ub_k
           [DAM] sum_k (p_k + g_k)*eta*dt >= H2_target

    Battery (H=12 RTM or H=24 DAM):
      min  same objective in p_k  +  battery objective terms
      s.t. lb_k <= p_k,  p_k <= ub_k              (no-sell + loose capacity)
           0 <= p_ch_k <= P_ch_max
           0 <= p_dis_k <= P_dis_max
           0 <= E_k <= E_cap
           E_{k+1} = E_k + eta_ch*p_ch_k*dt - p_dis_k/eta_dis*dt
           P_min <= p_k + g_k + p_dis_k - p_ch_k <= P_max   (electrolyzer bounds)
           [DAM] sum_k (p_k + g_k + p_dis_k - p_ch_k)*eta*dt >= H2_target

  z_i-update  [bilateral projection, analytic]:
    v_i,k = p_i,k + mu_i,k / rho
    s_k   = sum_i v_i,k
    s*_k  = clip(s_k, L_MIN, L_MAX)
    z_i,k = v_i,k + (s*_k - s_k) / N

  mu_i-update:
    mu_i,k += rho * (p_i,k - z_i,k)

Convergence:
  max_i ||p_i^{t+1} - p_i^t||_inf < tol
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from agents import ElzAgent, L_MIN_KW, L_MAX_KW

# ── Shared Gurobi environment ─────────────────────────────────────────────────
_GRB_ENV: Optional[gp.Env] = None

def _env() -> gp.Env:
    global _GRB_ENV
    if _GRB_ENV is None:
        _GRB_ENV = gp.Env(empty=True)
        _GRB_ENV.setParam('OutputFlag',   0)
        _GRB_ENV.setParam('LogToConsole', 0)
        _GRB_ENV.setParam('Threads',      1)
        _GRB_ENV.start()
    return _GRB_ENV


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ADMMResult:
    p:          np.ndarray   # (N, H) optimal net import [kW]
    p_ch:       np.ndarray   # (N, H) battery charging power (0 for non-battery)
    p_dis:      np.ndarray   # (N, H) battery discharging power (0 for non-battery)
    n_iter:     int
    converged:  bool
    primal_res: float
    dual_res:   float


# ── Non-battery Gurobi QP ─────────────────────────────────────────────────────

def _x_gurobi_simple(ag: ElzAgent,
                     p_da: float,
                     c:    np.ndarray,    # (H,) linear cost
                     g_ren:np.ndarray,    # (H,) renewable forecast
                     z_i:  np.ndarray,    # (H,) consensus
                     mu_i: np.ndarray,    # (H,) dual
                     lb:   np.ndarray,
                     ub:   np.ndarray,
                     rho:  float,
                     dt:   float,
                     add_h2:   bool  = False,
                     gamma_eff: float = None) -> np.ndarray:
    """
    Gurobi QP for agents without battery.

    DAM (gamma_eff=0): pure LP objective + ADMM augmentation only.
      min  sum_k { 0.5*rho*p_k^2  +  (c_k + mu_k - rho*z_k)*p_k }

    RTM (gamma_eff=ag.gamma): quadratic anchor to DA schedule.
      min  sum_k { 0.5*(gamma+rho)*p_k^2  +  (c_k + mu_k - gamma*p_DA - rho*z_k)*p_k }
    """
    H   = len(c)
    g   = ag.gamma if gamma_eff is None else gamma_eff
    m   = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)

    p = m.addVars(H, lb=lb.tolist(), ub=ub.tolist(), name='p')

    # Renewable curtailment: required so the participation floor p ≥ L_min stays
    # feasible when g + L_min > P_max (else the box [L_min, P_max-g] is empty).
    # For legacy calls (ub = P_max - g) the optimal cv is 0, so behaviour is
    # unchanged; cv only activates when the caller raises ub above P_max - g.
    has_cv = ag.has_renewable
    if has_cv:
        cv = m.addVars(H, lb=0.0, name='cv')
        for k in range(H):
            m.addConstr(cv[k] <= max(0.0, float(g_ren[k])), name=f'cv_ub_{k}')
            m.addConstr(p[k] + float(g_ren[k]) - cv[k] <= ag.p_elec_max_kw, name=f'eub_{k}')

    obj = gp.QuadExpr()
    h2_dt = ag.r_h2 * ag.eta_kg_per_kwh * dt
    for k in range(H):
        q_coeff = 0.5 * (g + rho)
        l_coeff = float(c[k]) + float(mu_i[k]) - g * float(p_da) - rho * float(z_i[k])
        obj.add(q_coeff * p[k] * p[k])
        obj.add(l_coeff * p[k])
        if has_cv:
            obj.add(h2_dt * cv[k])   # H₂ opportunity cost of curtailment
    m.setObjective(obj, GRB.MINIMIZE)

    if add_h2:
        h2_ren  = float(np.sum(g_ren * ag.eta_kg_per_kwh * dt))
        h2_need = max(0.0, ag.h2_target_kg_day - h2_ren)
        if h2_need > 1e-6:
            e_dt = ag.eta_kg_per_kwh * dt
            m.addConstr(
                gp.quicksum(p[k] * e_dt for k in range(H)) >= h2_need,
                name='h2_floor'
            )

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        sol = np.array([p[k].X for k in range(H)])
    else:
        p_unc = (g * p_da - c - mu_i + rho * z_i) / (g + rho) if g + rho > 0 else np.zeros(H)
        sol   = np.clip(p_unc, lb, ub)

    m.dispose()
    return sol


# ── Battery Gurobi QP ─────────────────────────────────────────────────────────

def _x_gurobi_battery(ag:   ElzAgent,
                      p_da: float,
                      c:    np.ndarray,    # (H,) linear cost for p_grid
                      g_ren:np.ndarray,    # (H,) renewable forecast
                      z_i:  np.ndarray,    # (H,) consensus
                      mu_i: np.ndarray,    # (H,) dual
                      rho:  float,
                      dt:   float,
                      E_0:  float,         # initial SoC [kWh]
                      add_h2:    bool  = False,
                      gamma_eff: float = None,
                      u_i:   Optional[int] = None,   # fixed participation: 1=in, 0=out, None=legacy
                      l_min: float = 0.0
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gurobi QP for battery agents (ADMM iterations).  Returns (p_grid, p_ch, p_dis).

    Decision variables per step k:
      p[k]     : net grid import  >= 0
      p_ch[k]  : battery charge   >= 0
      p_dis[k] : battery discharge >= 0
      E[k]     : SoC at start of step k
      cv[k]    : renewable curtailment >= 0  (planned, based on forecast)

    Objective:
      min  sum_k { 0.5*(gamma+rho)*p[k]^2 + q_k*p[k]
                   - r_h2*eta*dt * p_dis[k]      (H2 revenue from discharge)
                   + r_h2*eta*dt/eta_rt * p_ch[k] (H2 opportunity cost of charging)
                   + r_h2*eta*dt * cv[k]          (H2 opportunity cost of curtailment)
                 }

    Constraints:
      p[k] >= 0
      0 <= p_ch[k] <= P_ch_max
      0 <= p_dis[k] <= P_dis_max
      0 <= cv[k] <= g[k]                          (curtailment bounded by renewable)
      0 <= E[k] <= E_cap
      E[k+1] = E[k] + eta_ch*p_ch[k]*dt - p_dis[k]/eta_dis*dt
      P_min <= p[k] + (g[k]-cv[k]) + p_dis[k] - p_ch[k] <= P_max
      [DAM] sum_k (p[k]+(g[k]-cv[k])+p_dis[k]-p_ch[k])*eta*dt >= H2_target

    Note: mutual exclusivity (pch*pdis=0) is NOT enforced here to keep the
    problem a convex QP for ADMM convergence.  A one-shot MIQP (_x_gurobi_battery_miqp)
    is called after ADMM convergence to enforce it exactly.
    """
    H = len(c)
    m = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)

    # Decision variables.  Participation commitment u_i sets the p bounds:
    #   u_i == 0 → out of market, p = 0 ;  u_i == 1 → p ∈ [l_min, P_max+P_ch]
    #   u_i is None → legacy, p ∈ [0, P_max+P_ch]
    p_ub = ag.p_elec_max_kw + ag.bat_p_ch_max_kw
    p_lo = 0.0 if u_i != 1 else float(l_min)
    p_hi = 0.0 if u_i == 0 else float(p_ub)
    p    = m.addVars(H, lb=p_lo, ub=p_hi,               name='p')
    pch  = m.addVars(H, lb=0.0, ub=ag.bat_p_ch_max_kw,  name='pch')
    pdis = m.addVars(H, lb=0.0, ub=ag.bat_p_dis_max_kw, name='pdis')
    E    = m.addVars(H + 1, lb=0.0, ub=ag.bat_cap_kwh,  name='E')
    cv   = m.addVars(H, lb=0.0,                          name='cv')  # curtailment

    # Curtailment upper bound: cannot curtail more than available renewable
    for k in range(H):
        m.addConstr(cv[k] <= float(g_ren[k]), name=f'cv_ub_{k}')

    # Fix initial SoC
    m.addConstr(E[0] == float(E_0), name='soc_init')

    # SoC dynamics
    for k in range(H):
        m.addConstr(
            E[k + 1] == E[k]
                      + ag.bat_eta_ch * pch[k] * dt
                      - pdis[k] / ag.bat_eta_dis * dt,
            name=f'soc_{k}'
        )

    # Electrolyzer bounds: P_min <= p + (g-cv) + pdis - pch <= P_max
    for k in range(H):
        g_k = float(g_ren[k])
        m.addConstr(p[k] + g_k - cv[k] + pdis[k] - pch[k] >= ag.p_elec_min_kw,
                    name=f'elb_{k}')
        m.addConstr(p[k] + g_k - cv[k] + pdis[k] - pch[k] <= ag.p_elec_max_kw,
                    name=f'eub_{k}')

    # H2 daily production floor (DAM only)
    if add_h2:
        h2_expr = gp.LinExpr()
        for k in range(H):
            g_k  = float(g_ren[k])
            e_dt = ag.eta_kg_per_kwh * dt
            h2_expr.add(p[k],    e_dt)
            h2_expr.add(pdis[k], e_dt)
            h2_expr.add(pch[k], -e_dt)
            h2_expr.add(cv[k],  -e_dt)   # curtailment reduces H2 production
            h2_expr.addConstant(g_k * e_dt)
        m.addConstr(h2_expr >= ag.h2_target_kg_day, name='h2_floor')

    # Objective
    obj      = gp.QuadExpr()
    g_ag     = ag.gamma if gamma_eff is None else gamma_eff
    h2_dt    = ag.r_h2 * ag.eta_kg_per_kwh * dt   # H2 revenue per kW [$/kW]
    eta_rt   = ag.bat_eta_ch * ag.bat_eta_dis       # round-trip efficiency
    h2_dt_ch = h2_dt / eta_rt                       # opportunity cost of charging
    for k in range(H):
        q_coeff = 0.5 * (g_ag + rho)
        l_coeff = (float(c[k]) + float(mu_i[k])
                   - g_ag * float(p_da) - rho * float(z_i[k]))
        obj.add(q_coeff * p[k] * p[k])
        obj.add(l_coeff * p[k])
        obj.add(-h2_dt    * pdis[k])
        obj.add( h2_dt_ch * pch[k])
        obj.add( h2_dt    * cv[k])    # curtailment opportunity cost
    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        p_sol    = np.array([p[k].X    for k in range(H)])
        pch_sol  = np.array([pch[k].X  for k in range(H)])
        pdis_sol = np.array([pdis[k].X for k in range(H)])
    else:
        # Fallback: unconstrained minimiser for p (respecting participation), zero battery
        p_unc    = (g_ag * p_da - c - mu_i + rho * z_i) / (g_ag + rho) if g_ag + rho > 0 else np.zeros(H)
        p_sol    = np.clip(p_unc, p_lo, p_hi if p_hi > 0 else 0.0)
        pch_sol  = np.zeros(H)
        pdis_sol = np.zeros(H)

    m.dispose()
    return p_sol, pch_sol, pdis_sol


# ── Battery MIQP (one-shot after ADMM convergence) ────────────────────────────

def _x_gurobi_battery_miqp(ag:   ElzAgent,
                            p_da: float,
                            c:    np.ndarray,
                            g_ren:np.ndarray,
                            z_i:  np.ndarray,
                            mu_i: np.ndarray,
                            rho:  float,
                            dt:   float,
                            E_0:  float,
                            add_h2:    bool  = False,
                            gamma_eff: float = None,
                            u_i:   Optional[int] = None,
                            l_min: float = 0.0
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One-shot MIQP called after ADMM convergence to enforce exact mutual
    exclusivity of charge and discharge via binary variable b[k].

    b[k] = 1 → charging allowed, discharge forced to zero
    b[k] = 0 → discharging allowed, charge forced to zero

    Also includes curtailment cv[k] (same as QP).
    Uses the converged dual (mu_i) and consensus (z_i) as fixed parameters.
    Returns (p_grid, p_ch, p_dis) with guaranteed pch[k]*pdis[k] = 0 ∀k.
    Falls back to the QP solution if the MIQP is infeasible.
    """
    H = len(c)
    m = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)

    p_ub = ag.p_elec_max_kw + ag.bat_p_ch_max_kw
    p_lo = 0.0 if u_i != 1 else float(l_min)
    p_hi = 0.0 if u_i == 0 else float(p_ub)
    p    = m.addVars(H, lb=p_lo, ub=p_hi,               name='p')
    pch  = m.addVars(H, lb=0.0, ub=ag.bat_p_ch_max_kw,  name='pch')
    pdis = m.addVars(H, lb=0.0, ub=ag.bat_p_dis_max_kw, name='pdis')
    E    = m.addVars(H + 1, lb=0.0, ub=ag.bat_cap_kwh,  name='E')
    b    = m.addVars(H, vtype=GRB.BINARY,                name='b')   # 1=charge, 0=discharge
    cv   = m.addVars(H, lb=0.0,                          name='cv')

    # Exact mutual exclusivity via big-M (M = rated capacity)
    for k in range(H):
        m.addConstr(pch[k]  <= ag.bat_p_ch_max_kw  * b[k],       name=f'me_ch_{k}')
        m.addConstr(pdis[k] <= ag.bat_p_dis_max_kw * (1 - b[k]), name=f'me_dis_{k}')

    # Curtailment bounded by renewable
    for k in range(H):
        m.addConstr(cv[k] <= float(g_ren[k]), name=f'cv_ub_{k}')

    # SoC
    m.addConstr(E[0] == float(E_0), name='soc_init')
    for k in range(H):
        m.addConstr(
            E[k + 1] == E[k]
                      + ag.bat_eta_ch * pch[k] * dt
                      - pdis[k] / ag.bat_eta_dis * dt,
            name=f'soc_{k}'
        )

    # Electrolyzer bounds
    for k in range(H):
        g_k = float(g_ren[k])
        m.addConstr(p[k] + g_k - cv[k] + pdis[k] - pch[k] >= ag.p_elec_min_kw,
                    name=f'elb_{k}')
        m.addConstr(p[k] + g_k - cv[k] + pdis[k] - pch[k] <= ag.p_elec_max_kw,
                    name=f'eub_{k}')

    # H2 floor (DAM only)
    if add_h2:
        h2_expr = gp.LinExpr()
        for k in range(H):
            g_k  = float(g_ren[k])
            e_dt = ag.eta_kg_per_kwh * dt
            h2_expr.add(p[k],    e_dt)
            h2_expr.add(pdis[k], e_dt)
            h2_expr.add(pch[k], -e_dt)
            h2_expr.add(cv[k],  -e_dt)
            h2_expr.addConstant(g_k * e_dt)
        m.addConstr(h2_expr >= ag.h2_target_kg_day, name='h2_floor')

    # Objective (identical to QP)
    obj      = gp.QuadExpr()
    g_ag     = ag.gamma if gamma_eff is None else gamma_eff
    h2_dt    = ag.r_h2 * ag.eta_kg_per_kwh * dt
    eta_rt   = ag.bat_eta_ch * ag.bat_eta_dis
    h2_dt_ch = h2_dt / eta_rt
    for k in range(H):
        q_coeff = 0.5 * (g_ag + rho)
        l_coeff = (float(c[k]) + float(mu_i[k])
                   - g_ag * float(p_da) - rho * float(z_i[k]))
        obj.add(q_coeff * p[k] * p[k])
        obj.add(l_coeff * p[k])
        obj.add(-h2_dt    * pdis[k])
        obj.add( h2_dt_ch * pch[k])
        obj.add( h2_dt    * cv[k])
    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        p_sol    = np.array([p[k].X    for k in range(H)])
        pch_sol  = np.array([pch[k].X  for k in range(H)])
        pdis_sol = np.array([pdis[k].X for k in range(H)])
    else:
        p_sol = pch_sol = pdis_sol = None   # caller falls back to QP solution

    m.dispose()
    return p_sol, pch_sol, pdis_sol


# ── z-update (bilateral projection) ──────────────────────────────────────────

def _z_update(p_list:  list[np.ndarray],
              mu_list: list[np.ndarray],
              rho:     float,
              l_min:   float,
              l_max:   float) -> list[np.ndarray]:
    """Project aggregate onto [l_min, l_max] per step, shift uniformly."""
    N      = len(p_list)
    V      = np.stack([p_list[i] + mu_list[i] / rho for i in range(N)])  # (N,H)
    s      = V.sum(axis=0)
    s_star = np.clip(s, l_min, l_max)
    delta  = (s_star - s) / N
    return [V[i] + delta for i in range(N)]


# ── Public solver ─────────────────────────────────────────────────────────────

def solve(
    agents:   list[ElzAgent],
    lam:      np.ndarray,           # (H,) price forecast [$/MWh]
    p_da:     np.ndarray,           # (N,) DA anchor per agent [kW]
    g_ren:    np.ndarray,           # (N,H) renewable forecast [kW]
    H:        int,
    dt:       float,
    bat_soc:  Optional[np.ndarray] = None,   # (N,) current battery SoC [kWh]
    rho:      float = 0.05,
    max_iter: int   = 300,
    tol:      float = 1.0,
    mode:     str   = 'rtm',        # 'rtm' or 'dam'
    warm:     Optional[np.ndarray] = None,
    l_min:    float = L_MIN_KW,
    l_max:    float = L_MAX_KW,
    u:        Optional[np.ndarray] = None,   # (N,) fixed hourly participation 1=in/0=out; None=legacy
) -> ADMMResult:
    """
    Multi-agent ADMM.

    RTM (H=12, mode='rtm'): Gurobi box QP (non-battery) or battery QP.  No H2 constraint.
    DAM (H=24, mode='dam'): Gurobi QP + H2 daily production constraint.

    bat_soc : current SoC [kWh] per agent.  Required for battery agents.
              Non-battery agents ignore this parameter.
    """
    N      = len(agents)
    c      = np.stack([ag.linear_cost(lam, dt) for ag in agents])   # (N,H)
    bounds = [agents[i].bounds(g_ren[i]) for i in range(N)]

    # Apply fixed hourly participation commitment u to the per-agent p bounds.
    #   u_i == 0 → out of market, p = 0
    #   u_i == 1 → participate, p ∈ [l_min, P_max(+P_ch)]  (renewable curtailment
    #              via cv keeps the floor feasible when g + l_min > P_max)
    # Non-battery bounds are overridden here; battery bounds are handled inside
    # the battery QP/MIQP via (u_i, l_min).  u is None ⇒ legacy behaviour.
    if u is not None:
        for i in range(N):
            if not agents[i].has_battery:
                if int(u[i]) == 0:
                    bounds[i] = (np.zeros(H), np.zeros(H))
                else:
                    p_hi = agents[i].p_elec_max_kw
                    bounds[i] = (np.full(H, float(l_min)), np.full(H, float(p_hi)))
        # Optional coalition: if no agent is committed this hour, the coalition is
        # out of the market — drop the aggregate floor so the z-update doesn't
        # force phantom imports against the p=0 bounds.
        if not np.any(np.asarray(u) == 1):
            l_min = 0.0

    if bat_soc is None:
        bat_soc = np.array([ag.bat_soc_kwh for ag in agents])

    # Initialise p
    if warm is not None:
        p = [np.clip(warm[i].copy(), bounds[i][0], bounds[i][1]) for i in range(N)]
    else:
        p = [np.clip(np.zeros(H), bounds[i][0], bounds[i][1]) for i in range(N)]

    # Battery charge/discharge (only relevant for battery agents, initialise to 0)
    p_ch  = [np.zeros(H) for _ in range(N)]
    p_dis = [np.zeros(H) for _ in range(N)]

    z  = [pi.copy() for pi in p]
    mu = [np.zeros(H) for _ in range(N)]

    primal_res = dual_res = np.inf
    converged  = False
    add_h2     = (mode == 'dam')
    # DAM: pure LP (gamma=0, no quadratic anchor — no prior commitment exists).
    # RTM: quadratic anchor to the DA schedule (imbalance penalty).
    gamma_eff  = 0.0 if mode == 'dam' else None   # None → use ag.gamma

    for k in range(max_iter):
        z_prev = [zi.copy() for zi in z]
        p_prev = [pi.copy() for pi in p]

        # p-update: dispatch to correct QP
        for i, ag in enumerate(agents):
            lb, ub = bounds[i]
            if ag.has_battery:
                p[i], p_ch[i], p_dis[i] = _x_gurobi_battery(
                    ag, float(p_da[i]), c[i], g_ren[i],
                    z[i], mu[i], rho, dt,
                    E_0=float(bat_soc[i]), add_h2=add_h2,
                    gamma_eff=gamma_eff,
                    u_i=(None if u is None else int(u[i])), l_min=l_min
                )
            else:
                p[i] = _x_gurobi_simple(
                    ag, float(p_da[i]), c[i], g_ren[i],
                    z[i], mu[i], lb, ub, rho, dt, add_h2=add_h2,
                    gamma_eff=gamma_eff
                )

        # z-update: bilateral projection
        z = _z_update(p, mu, rho, l_min, l_max)

        # mu-update
        for i in range(N):
            mu[i] += rho * (p[i] - z[i])

        # Convergence
        agg       = sum(p[i] for i in range(N))
        p_change  = float(max(np.max(np.abs(p[i] - p_prev[i])) for i in range(N)))
        coup_viol = float(np.max(np.abs(agg - np.clip(agg, l_min, l_max))))
        primal_res = max(p_change, coup_viol)
        dual_res   = float(rho * max(np.max(np.abs(z[i] - z_prev[i])) for i in range(N)))
        converged  = primal_res < tol
        if converged:
            break

    # One-shot MIQP cleanup: enforce exact pch[k]*pdis[k]=0 for battery agents
    # using the converged dual (mu) and consensus (z) as fixed parameters.
    # This does not re-run ADMM; it finds the best binary-feasible solution at
    # the converged dual point, guaranteeing mutual exclusivity in the output.
    for i, ag in enumerate(agents):
        if ag.has_battery:
            p_m, pch_m, pdis_m = _x_gurobi_battery_miqp(
                ag, float(p_da[i]), c[i], g_ren[i],
                z[i], mu[i], rho, dt,
                E_0=float(bat_soc[i]), add_h2=add_h2,
                gamma_eff=gamma_eff,
                u_i=(None if u is None else int(u[i])), l_min=l_min
            )
            if p_m is not None:
                p[i]     = p_m
                p_ch[i]  = pch_m
                p_dis[i] = pdis_m

    return ADMMResult(
        p         = np.array(p),
        p_ch      = np.array(p_ch),
        p_dis     = np.array(p_dis),
        n_iter    = k + 1,
        converged = converged,
        primal_res= primal_res,
        dual_res  = dual_res,
    )


# ── Hour-ahead coordinated participation commitment (GNE) ─────────────────────

def commit_gne(
    agents:  list[ElzAgent],
    lam:     np.ndarray,          # (H,) hour-ahead price forecast [$/MWh]
    g_ren:   np.ndarray,          # (N,H) hour-ahead renewable forecast [kW]
    H:       int,
    dt:      float,
    bat_soc: np.ndarray,          # (N,) battery SoC at commitment time [kWh]
    l_min:   float = L_MIN_KW,
    l_max:   float = L_MAX_KW,
) -> np.ndarray:
    """
    Coordinated hour-ahead participation commitment for the GNE coalition.

    Single centralized MIQP over all N agents for the one-hour window, with one
    binary u_i per agent (offer locked for the whole operating hour):

        100·u_i ≤ p_{i,k} ≤ P_ub_i·u_i     ∀k   (semi-continuous per agent)
        l_min ≤ Σ_i p_{i,k} ≤ l_max         ∀k   (PCC coupling)
        electrolyzer / curtailment / battery feasibility per agent

    This is the integer-recovery step: it isolates the non-convexity to a single
    hourly solve.  The committed u is then held fixed across the twelve 5-min RT
    dispatch steps, so the RT ADMM stays convex.  Economic objective (gamma=0).

    Optional coalition participation: an aggregate hourly binary U gates the
    whole coalition.  If U=0 the coalition stays out of the market entirely
    (Σp=0); if U=1 the coupling Σp ≥ l_min applies.  Per-agent u_i ≤ U, so an
    agent can bid only when the coalition is in.

    Returns u ∈ {0,1}^N.
    """
    N = len(agents)
    m = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)

    U = m.addVar(vtype=GRB.BINARY, name='U_coalition')   # aggregate opt-out gate

    p   = {}      # (i) -> tupledict over k
    u   = {}
    cv  = {}
    pch = {}
    pdis = {}
    for i, ag in enumerate(agents):
        p_ub = ag.p_elec_max_kw + (ag.bat_p_ch_max_kw if ag.has_battery else 0.0)
        p[i] = m.addVars(H, lb=0.0, ub=float(p_ub), name=f'p_{i}')
        u[i] = m.addVar(vtype=GRB.BINARY, name=f'u_{i}')
        m.addConstr(u[i] <= U, name=f'agent_gate_{i}')   # bid only if coalition is in
        for k in range(H):
            m.addConstr(p[i][k] >= float(l_min) * u[i], name=f'sc_lb_{i}_{k}')
            m.addConstr(p[i][k] <= float(p_ub)  * u[i], name=f'sc_ub_{i}_{k}')

        if ag.has_renewable:
            cv[i] = m.addVars(H, lb=0.0, name=f'cv_{i}')
            for k in range(H):
                m.addConstr(cv[i][k] <= max(0.0, float(g_ren[i][k])), name=f'cvub_{i}_{k}')

        if ag.has_battery:
            pch[i]  = m.addVars(H, lb=0.0, ub=ag.bat_p_ch_max_kw,  name=f'pch_{i}')
            pdis[i] = m.addVars(H, lb=0.0, ub=ag.bat_p_dis_max_kw, name=f'pdis_{i}')
            E       = m.addVars(H + 1, lb=0.0, ub=ag.bat_cap_kwh,  name=f'E_{i}')
            b       = m.addVars(H, vtype=GRB.BINARY, name=f'b_{i}')
            for k in range(H):
                m.addConstr(pch[i][k]  <= ag.bat_p_ch_max_kw  * b[k],       name=f'mech_{i}_{k}')
                m.addConstr(pdis[i][k] <= ag.bat_p_dis_max_kw * (1 - b[k]), name=f'medis_{i}_{k}')
            m.addConstr(E[0] == float(bat_soc[i]), name=f'soc0_{i}')
            for k in range(H):
                m.addConstr(
                    E[k + 1] == E[k]
                              + ag.bat_eta_ch * pch[i][k] * dt
                              - pdis[i][k] / ag.bat_eta_dis * dt,
                    name=f'soc_{i}_{k}'
                )

        # Electrolyzer feasibility per agent
        for k in range(H):
            g_k = float(g_ren[i][k])
            cvk = cv[i][k] if ag.has_renewable else 0.0
            if ag.has_battery:
                expr = p[i][k] + g_k - cvk + pdis[i][k] - pch[i][k]
            else:
                expr = p[i][k] + g_k - cvk
            m.addConstr(expr >= ag.p_elec_min_kw, name=f'elb_{i}_{k}')
            m.addConstr(expr <= ag.p_elec_max_kw, name=f'eub_{i}_{k}')

    # PCC coupling per step — gated by the coalition opt-out binary U.
    #   U=1 → l_min ≤ Σp ≤ l_max ;  U=0 → Σp = 0 (all u_i ≤ U = 0 ⇒ all p=0)
    for k in range(H):
        agg_k = gp.quicksum(p[i][k] for i in range(N))
        m.addConstr(agg_k >= float(l_min) * U, name=f'coup_lo_{k}')
        if np.isfinite(l_max):
            m.addConstr(agg_k <= float(l_max), name=f'coup_hi_{k}')

    # Economic objective (gamma=0): grid cost − H₂ revenue (+ battery/curtail terms)
    obj = gp.LinExpr()
    for i, ag in enumerate(agents):
        c_i   = ag.linear_cost(lam, dt)
        h2_dt = ag.r_h2 * ag.eta_kg_per_kwh * dt
        for k in range(H):
            obj.add(p[i][k], float(c_i[k]))
            if ag.has_battery:
                eta_rt = ag.bat_eta_ch * ag.bat_eta_dis
                obj.add(pdis[i][k], -h2_dt)
                obj.add(pch[i][k],   h2_dt / eta_rt)
            if ag.has_renewable:
                obj.add(cv[i][k], h2_dt)
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        u_out = np.array([int(round(u[i].X)) for i in range(N)], dtype=int)
    else:
        # Fallback: coalition opts out (always feasible under optional participation)
        u_out = np.zeros(N, dtype=int)
    m.dispose()
    return u_out
