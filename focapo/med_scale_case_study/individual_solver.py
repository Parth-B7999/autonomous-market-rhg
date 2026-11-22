"""
individual_solver.py — Single-agent Gurobi QP for individual market participation.

Each agent faces its own local constraints PLUS a minimum market
participation constraint:  p_i,k ≥ L_min  at every time step.

No coupling with other agents — no ADMM needed.

Key difference from admm.py:
  - No z-update, no dual variable, no consensus
  - L_min enforced as a lower bound on p (not as an aggregate coupling)
  - Curtailment variable added for ALL renewable agents (not just battery+renewable)
    to handle feasibility when g_ren + L_min > P_max
  - MIQP cleanup for battery agents (same as admm.py)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

import sys
from pathlib import Path
_PARENT = str(Path(__file__).resolve().parent.parent / 'admm_elz_market')
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PARENT not in sys.path:
    sys.path.append(_PARENT)  # parent is fallback only (avoid shadowing med-local modules)

import gurobipy as gp                   # noqa: E402
from gurobipy import GRB               # noqa: E402
from agents import ElzAgent            # noqa: E402


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


@dataclass
class IndividualResult:
    """Result from a single-agent optimization."""
    p:      np.ndarray   # (H,) net grid import [kW]
    p_ch:   np.ndarray   # (H,) battery charging [kW]
    p_dis:  np.ndarray   # (H,) battery discharging [kW]


# ── Unified single-agent QP ──────────────────────────────────────────────────

def solve_individual(
    ag:     ElzAgent,
    lam:    np.ndarray,     # (H,) price forecast [$/MWh]
    p_da:   float,          # DA anchor [kW] (0.0 for DAM solve)
    g_ren:  np.ndarray,     # (H,) renewable forecast [kW]
    H:      int,
    dt:     float,
    E_0:    float,          # initial battery SoC [kWh]
    l_min:  float = 100.0,  # minimum market participation [kW]
    mode:   str   = 'rtm',  # 'rtm' or 'dam'
    u:      Optional[int] = None,  # fixed hourly participation: 1=in, 0=out, None=legacy (always in)
) -> IndividualResult:
    """
    Single-agent Gurobi QP, semi-continuous market participation.

    Participation rule  p ∈ {0} ∪ [L_min, P_max]  is governed by the fixed
    hourly commitment ``u`` (decided ahead by ``commit_individual``):
      - u is None : legacy behaviour, p ≥ L_min always (always participate)
      - u == 1    : participate this hour, p ∈ [L_min, P_max]
      - u == 0    : out of market this hour, p = 0 (self-consume renewable/battery)

    Handles all four agent types:
      - Electrolyzer only:       p ≤ P_max
      - Renewable + Elec:        p + g - cv ≤ P_max, 0 ≤ cv ≤ g
      - Battery + Elec:          p + pdis - pch ≤ P_max, SoC dynamics
      - Battery + Renewable + E: p + g - cv + pdis - pch ≤ P_max, cv, SoC

    DAM: pure LP (gamma=0) with H₂ floor constraint.
    RTM: QP with DA anchor (gamma > 0), no H₂ floor.
    """
    c     = ag.linear_cost(lam, dt)
    gamma = 0.0 if mode == 'dam' else ag.gamma
    add_h2 = (mode == 'dam')

    m = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)

    # ── Decision variables ────────────────────────────────────────────────────
    # p: net grid import.  Bounds set by the fixed participation commitment u.
    p_ub = ag.p_elec_max_kw + (ag.bat_p_ch_max_kw if ag.has_battery else 0.0)
    if u == 0:
        p = m.addVars(H, lb=0.0, ub=0.0, name='p')              # out of market
    else:
        p = m.addVars(H, lb=float(l_min), ub=float(p_ub), name='p')  # in market (u=1 or legacy)

    # Curtailment for renewable agents (needed when g + L_min > P_max)
    has_cv = ag.has_renewable
    if has_cv:
        cv = m.addVars(H, lb=0.0, name='cv')
        for k in range(H):
            m.addConstr(cv[k] <= max(0.0, float(g_ren[k])), name=f'cv_ub_{k}')

    # Battery variables
    if ag.has_battery:
        pch  = m.addVars(H,     lb=0.0, ub=ag.bat_p_ch_max_kw,  name='pch')
        pdis = m.addVars(H,     lb=0.0, ub=ag.bat_p_dis_max_kw, name='pdis')
        E    = m.addVars(H + 1, lb=0.0, ub=ag.bat_cap_kwh,      name='E')

        m.addConstr(E[0] == float(E_0), name='soc_init')
        for k in range(H):
            m.addConstr(
                E[k + 1] == E[k]
                          + ag.bat_eta_ch * pch[k] * dt
                          - pdis[k] / ag.bat_eta_dis * dt,
                name=f'soc_{k}'
            )

    # ── Electrolyzer bounds ───────────────────────────────────────────────────
    for k in range(H):
        g_k = float(g_ren[k])
        if ag.has_battery and has_cv:
            # Battery + Renewable: p + g - cv + pdis - pch in [0, P_max]
            m.addConstr(p[k] + g_k - cv[k] + pdis[k] - pch[k] >= ag.p_elec_min_kw,
                        name=f'elb_{k}')
            m.addConstr(p[k] + g_k - cv[k] + pdis[k] - pch[k] <= ag.p_elec_max_kw,
                        name=f'eub_{k}')
        elif ag.has_battery and not has_cv:
            # Battery only (no renewable): p + pdis - pch in [0, P_max]
            m.addConstr(p[k] + pdis[k] - pch[k] >= ag.p_elec_min_kw,
                        name=f'elb_{k}')
            m.addConstr(p[k] + pdis[k] - pch[k] <= ag.p_elec_max_kw,
                        name=f'eub_{k}')
        elif has_cv and not ag.has_battery:
            # Renewable only (no battery): p + g - cv in [0, P_max]
            m.addConstr(p[k] + g_k - cv[k] >= ag.p_elec_min_kw,
                        name=f'elb_{k}')
            m.addConstr(p[k] + g_k - cv[k] <= ag.p_elec_max_kw,
                        name=f'eub_{k}')
        else:
            # No renewable, no battery: p in [L_min, P_max]
            # (lb = L_min already set; just enforce P_max)
            m.addConstr(p[k] <= ag.p_elec_max_kw, name=f'eub_{k}')

    # ── H₂ floor (DAM only) ──────────────────────────────────────────────────
    if add_h2 and ag.h2_target_kg_day > 0:
        h2_expr = gp.LinExpr()
        for k in range(H):
            e_dt = ag.eta_kg_per_kwh * dt
            h2_expr.add(p[k], e_dt)
            h2_expr.addConstant(float(g_ren[k]) * e_dt)
            if has_cv:
                h2_expr.add(cv[k], -e_dt)
            if ag.has_battery:
                h2_expr.add(pdis[k], e_dt)
                h2_expr.add(pch[k], -e_dt)
        m.addConstr(h2_expr >= ag.h2_target_kg_day, name='h2_floor')

    # ── Objective ─────────────────────────────────────────────────────────────
    obj   = gp.QuadExpr()
    h2_dt = ag.r_h2 * ag.eta_kg_per_kwh * dt  # H₂ revenue per kW per step

    for k in range(H):
        # Quadratic DA-deviation penalty (RTM only; gamma=0 in DAM)
        obj.add(0.5 * gamma * p[k] * p[k])
        # Linear term
        l_coeff = float(c[k]) - gamma * float(p_da)
        obj.add(l_coeff * p[k])

        # Battery terms
        if ag.has_battery:
            eta_rt = ag.bat_eta_ch * ag.bat_eta_dis
            obj.add(-h2_dt * pdis[k])              # H₂ revenue from discharge
            obj.add((h2_dt / eta_rt) * pch[k])     # H₂ opportunity cost of charge

        # Curtailment opportunity cost
        if has_cv:
            obj.add(h2_dt * cv[k])

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    # ── Extract solution ──────────────────────────────────────────────────────
    if m.Status == GRB.OPTIMAL:
        p_sol = np.array([p[k].X for k in range(H)])
        if ag.has_battery:
            pch_sol  = np.array([pch[k].X  for k in range(H)])
            pdis_sol = np.array([pdis[k].X for k in range(H)])
        else:
            pch_sol  = np.zeros(H)
            pdis_sol = np.zeros(H)
    else:
        # Fallback: p=0 if out of market, else L_min import; zero battery
        p_sol    = np.zeros(H) if u == 0 else np.full(H, l_min)
        pch_sol  = np.zeros(H)
        pdis_sol = np.zeros(H)

    m.dispose()
    return IndividualResult(p=p_sol, p_ch=pch_sol, p_dis=pdis_sol)


# ── MIQP cleanup for battery agents ──────────────────────────────────────────

def solve_individual_miqp(
    ag:     ElzAgent,
    lam:    np.ndarray,
    p_da:   float,
    g_ren:  np.ndarray,
    H:      int,
    dt:     float,
    E_0:    float,
    l_min:  float = 100.0,
    mode:   str   = 'rtm',
    u:      Optional[int] = None,   # fixed hourly participation: 1=in, 0=out, None=legacy
) -> Optional[IndividualResult]:
    """
    One-shot MIQP for battery agents: enforces pch[k] * pdis[k] = 0
    via binary variable b[k].

    Same objective and constraints as solve_individual, plus:
      pch[k]  ≤ P_ch_max * b[k]
      pdis[k] ≤ P_dis_max * (1 - b[k])

    Returns None if infeasible (caller keeps QP solution).
    """
    if not ag.has_battery:
        return None

    c     = ag.linear_cost(lam, dt)
    gamma = 0.0 if mode == 'dam' else ag.gamma
    add_h2 = (mode == 'dam')

    m = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)

    p_ub = ag.p_elec_max_kw + ag.bat_p_ch_max_kw
    if u == 0:
        p = m.addVars(H, lb=0.0, ub=0.0, name='p')              # out of market
    else:
        p = m.addVars(H, lb=float(l_min), ub=float(p_ub), name='p')
    pch  = m.addVars(H, lb=0.0, ub=ag.bat_p_ch_max_kw,  name='pch')
    pdis = m.addVars(H, lb=0.0, ub=ag.bat_p_dis_max_kw, name='pdis')
    E    = m.addVars(H + 1, lb=0.0, ub=ag.bat_cap_kwh,  name='E')
    b    = m.addVars(H, vtype=GRB.BINARY, name='b')

    has_cv = ag.has_renewable
    if has_cv:
        cv = m.addVars(H, lb=0.0, name='cv')
        for k in range(H):
            m.addConstr(cv[k] <= max(0.0, float(g_ren[k])), name=f'cv_ub_{k}')

    # Mutual exclusivity
    for k in range(H):
        m.addConstr(pch[k]  <= ag.bat_p_ch_max_kw  * b[k],       name=f'me_ch_{k}')
        m.addConstr(pdis[k] <= ag.bat_p_dis_max_kw * (1 - b[k]), name=f'me_dis_{k}')

    # SoC
    m.addConstr(E[0] == float(E_0), name='soc_init')
    for k in range(H):
        m.addConstr(
            E[k + 1] == E[k]
                      + ag.bat_eta_ch * pch[k] * dt
                      - pdis[k] / ag.bat_eta_dis * dt,
            name=f'soc_{k}'
        )

    # Electrolyzer
    for k in range(H):
        g_k = float(g_ren[k])
        if has_cv:
            m.addConstr(p[k] + g_k - cv[k] + pdis[k] - pch[k] >= ag.p_elec_min_kw,
                        name=f'elb_{k}')
            m.addConstr(p[k] + g_k - cv[k] + pdis[k] - pch[k] <= ag.p_elec_max_kw,
                        name=f'eub_{k}')
        else:
            m.addConstr(p[k] + pdis[k] - pch[k] >= ag.p_elec_min_kw,
                        name=f'elb_{k}')
            m.addConstr(p[k] + pdis[k] - pch[k] <= ag.p_elec_max_kw,
                        name=f'eub_{k}')

    # H₂ floor
    if add_h2 and ag.h2_target_kg_day > 0:
        h2_expr = gp.LinExpr()
        for k in range(H):
            e_dt = ag.eta_kg_per_kwh * dt
            h2_expr.add(p[k], e_dt)
            h2_expr.add(pdis[k], e_dt)
            h2_expr.add(pch[k], -e_dt)
            if has_cv:
                h2_expr.add(cv[k], -e_dt)
            h2_expr.addConstant(float(g_ren[k]) * e_dt)
        m.addConstr(h2_expr >= ag.h2_target_kg_day, name='h2_floor')

    # Objective (identical to QP)
    obj   = gp.QuadExpr()
    h2_dt = ag.r_h2 * ag.eta_kg_per_kwh * dt
    eta_rt = ag.bat_eta_ch * ag.bat_eta_dis
    for k in range(H):
        obj.add(0.5 * gamma * p[k] * p[k])
        l_coeff = float(c[k]) - gamma * float(p_da)
        obj.add(l_coeff * p[k])
        obj.add(-h2_dt * pdis[k])
        obj.add((h2_dt / eta_rt) * pch[k])
        if has_cv:
            obj.add(h2_dt * cv[k])
    m.setObjective(obj, GRB.MINIMIZE)

    m.optimize()

    if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        p_sol    = np.array([p[k].X    for k in range(H)])
        pch_sol  = np.array([pch[k].X  for k in range(H)])
        pdis_sol = np.array([pdis[k].X for k in range(H)])
        m.dispose()
        return IndividualResult(p=p_sol, p_ch=pch_sol, p_dis=pdis_sol)

    m.dispose()
    return None


# ── Hour-ahead participation commitment (semi-continuous binary) ──────────────

def commit_individual(
    ag:     ElzAgent,
    lam:    np.ndarray,     # (H,) hour-ahead price forecast [$/MWh]
    g_ren:  np.ndarray,     # (H,) hour-ahead renewable forecast [kW]
    H:      int,
    dt:     float,
    E_0:    float,          # battery SoC at commitment time [kWh]
    p_da:   float = 0.0,    # DA anchor [kW] (0 in commitment — no prior RT commitment)
    l_min:  float = 100.0,
) -> int:
    """
    Hour-ahead participation decision for one agent (PJM 65-min gate).

    Solves a one-hour MIQP over the H forecast steps with a SINGLE shared
    binary u (the offer is locked for the whole operating hour):

        100·u ≤ p_k ≤ p_ub·u   ∀k          (semi-continuous: p_k ∈ {0}∪[100,p_ub])
        electrolyzer / curtailment / battery feasibility as usual

    Returns u ∈ {0,1}: 1 = submit a market bid for the hour, 0 = self-consume.

    The committed u is then held FIXED across the twelve 5-min RT dispatch steps
    of that hour, keeping the RT solve convex.

    The commitment is a pure economic decision (gamma=0, like the DAM LP): is it
    profitable to run the electrolyzer on grid power this hour?  The RT deviation
    penalty (gamma>0, anchored to the DA schedule) is applied later in dispatch,
    not here — including it here with p_da=0 would penalise all participation.
    """
    c = ag.linear_cost(lam, dt)
    gamma = 0.0   # economic commitment only; deviation penalty applied in RT dispatch

    m = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)

    p_ub = ag.p_elec_max_kw + (ag.bat_p_ch_max_kw if ag.has_battery else 0.0)
    p = m.addVars(H, lb=0.0, ub=float(p_ub), name='p')
    u = m.addVar(vtype=GRB.BINARY, name='u')          # single hourly participation

    # Semi-continuous coupling: p_k ∈ {0} ∪ [l_min, p_ub]
    for k in range(H):
        m.addConstr(p[k] >= float(l_min) * u, name=f'sc_lb_{k}')
        m.addConstr(p[k] <= float(p_ub)  * u, name=f'sc_ub_{k}')

    has_cv = ag.has_renewable
    if has_cv:
        cv = m.addVars(H, lb=0.0, name='cv')
        for k in range(H):
            m.addConstr(cv[k] <= max(0.0, float(g_ren[k])), name=f'cv_ub_{k}')

    if ag.has_battery:
        pch  = m.addVars(H, lb=0.0, ub=ag.bat_p_ch_max_kw,  name='pch')
        pdis = m.addVars(H, lb=0.0, ub=ag.bat_p_dis_max_kw, name='pdis')
        E    = m.addVars(H + 1, lb=0.0, ub=ag.bat_cap_kwh,  name='E')
        b    = m.addVars(H, vtype=GRB.BINARY, name='b')
        for k in range(H):
            m.addConstr(pch[k]  <= ag.bat_p_ch_max_kw  * b[k],       name=f'me_ch_{k}')
            m.addConstr(pdis[k] <= ag.bat_p_dis_max_kw * (1 - b[k]), name=f'me_dis_{k}')
        m.addConstr(E[0] == float(E_0), name='soc_init')
        for k in range(H):
            m.addConstr(
                E[k + 1] == E[k]
                          + ag.bat_eta_ch * pch[k] * dt
                          - pdis[k] / ag.bat_eta_dis * dt,
                name=f'soc_{k}'
            )

    # Electrolyzer feasibility (same branching as solve_individual)
    for k in range(H):
        g_k = float(g_ren[k])
        if ag.has_battery and has_cv:
            m.addConstr(p[k] + g_k - cv[k] + pdis[k] - pch[k] >= ag.p_elec_min_kw, name=f'elb_{k}')
            m.addConstr(p[k] + g_k - cv[k] + pdis[k] - pch[k] <= ag.p_elec_max_kw, name=f'eub_{k}')
        elif ag.has_battery and not has_cv:
            m.addConstr(p[k] + pdis[k] - pch[k] >= ag.p_elec_min_kw, name=f'elb_{k}')
            m.addConstr(p[k] + pdis[k] - pch[k] <= ag.p_elec_max_kw, name=f'eub_{k}')
        elif has_cv and not ag.has_battery:
            m.addConstr(p[k] + g_k - cv[k] >= ag.p_elec_min_kw, name=f'elb_{k}')
            m.addConstr(p[k] + g_k - cv[k] <= ag.p_elec_max_kw, name=f'eub_{k}')
        else:
            m.addConstr(p[k] <= ag.p_elec_max_kw, name=f'eub_{k}')

    # Objective (RT-style: deviation penalty + linear cost + H₂ terms)
    obj   = gp.QuadExpr()
    h2_dt = ag.r_h2 * ag.eta_kg_per_kwh * dt
    for k in range(H):
        obj.add(0.5 * gamma * p[k] * p[k])
        obj.add((float(c[k]) - gamma * float(p_da)) * p[k])
        if ag.has_battery:
            eta_rt = ag.bat_eta_ch * ag.bat_eta_dis
            obj.add(-h2_dt * pdis[k])
            obj.add((h2_dt / eta_rt) * pch[k])
        if has_cv:
            obj.add(h2_dt * cv[k])
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    u_val = int(round(u.X)) if m.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL) else 0
    m.dispose()
    return u_val
