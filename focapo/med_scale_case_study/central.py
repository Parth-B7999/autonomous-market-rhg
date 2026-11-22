"""
central.py — Centralized benchmark (B5, --scenario central).

One joint MIQP over all agents for the DAM and for every RTM step.  Each agent
block is the same model as miqp_admm._solve_agent with rho = 0 and no local
market floor (as in the GNE, where the floor is on the aggregate).  The
aggregate market rules are exact constraints with binaries:

  DAM  (u free):   agg_k ∈ {0} ∪ [L_min, L_max]                (U_k per step)
  RTM  (u fixed):  agg_k ∈ [L_min, P_DA] ∪ [P_DA + L_min, L_max] (w_k per step)
                   P_DA = Σ_{in-market} p_DA,i ; same set as _project_rule_b_agg

Objective = Σ_i agent cost (the GNE potential).  If the exact aggregate floor
is physically infeasible at a step (ramp limits), a heavily penalised slack
keeps the model feasible; the amount used is reported in CentralResult.agg_slack.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from miqp_admm import _env, H2_SLACK_PENALTY
from agents import ElzAgent

AGG_SLACK_PENALTY = 1.0e3   # $ per kW per step of aggregate-floor violation


@dataclass
class CentralResult:
    p:      np.ndarray   # (N, H)
    p_ch:   np.ndarray
    p_dis:  np.ndarray
    p_elec: np.ndarray
    u:      np.ndarray   # (N,) 1 if agent imports > 1 kW in any step
    n_iter: int = 1
    converged: bool = True
    agg_slack: float = 0.0     # total aggregate-floor slack used [kW]
    mip_gap: float = 0.0
    n_skipped: int = 0         # agents left out because their own block was infeasible (-1 = model infeasible)


def _add_agent(m, i, ag: ElzAgent, H, dt, E_0, lam, p_da, g_ren, gamma, add_h2,
               u_fixed, p_elec_init, h2_target_rtm):
    """Agent block (mirrors _solve_agent, rho = 0, l_min = 0). Returns vars, obj."""
    c    = (lam / 1000.0 - ag.r_h2 * ag.eta_kg_per_kwh) * dt
    h2dt = ag.r_h2 * ag.eta_kg_per_kwh * dt
    p_ub = ag.p_elec_max_kw + (ag.bat_p_ch_max_kw if ag.has_battery else 0.0)
    out  = (u_fixed is not None and int(u_fixed) == 0)

    p = m.addVars(H, lb=0.0, ub=0.0 if out else float(p_ub), name=f'p{i}')
    cv = m.addVars(H, lb=0.0, name=f'cv{i}') if ag.has_renewable else None
    if cv is not None:
        for k in range(H):
            m.addConstr(cv[k] <= max(0.0, float(g_ren[k])))
    if ag.has_battery:
        pch  = m.addVars(H, lb=0.0, ub=ag.bat_p_ch_max_kw,  name=f'pch{i}')
        pdis = m.addVars(H, lb=0.0, ub=ag.bat_p_dis_max_kw, name=f'pdis{i}')
        E    = m.addVars(H + 1, lb=0.0, ub=ag.bat_cap_kwh,  name=f'E{i}')
        d    = m.addVars(H, vtype=GRB.BINARY, name=f'd{i}')
        m.addConstr(E[0] == float(E_0))
        for k in range(H):
            m.addConstr(pch[k]  <= ag.bat_p_ch_max_kw  * d[k])
            m.addConstr(pdis[k] <= ag.bat_p_dis_max_kw * (1 - d[k]))
            m.addConstr(E[k + 1] == E[k] + ag.bat_eta_ch * pch[k] * dt
                        - pdis[k] / ag.bat_eta_dis * dt)
    else:
        pch = pdis = None

    pe = m.addVars(H, lb=0.0, ub=ag.p_elec_max_kw, name=f'pe{i}')
    ramp = ag.ramp_rate_kw_per_hr * dt
    for k in range(H):
        m.addConstr(pe[k] == p[k] + float(g_ren[k])
                    - (cv[k] if cv is not None else 0.0)
                    + (pdis[k] if pdis is not None else 0.0)
                    - (pch[k] if pch is not None else 0.0))
        prev = pe[k - 1] if k > 0 else float(p_elec_init)
        m.addConstr(pe[k] - prev <= ramp)
        m.addConstr(prev - pe[k] <= ramp)

    if add_h2 and ag.h2_target_kg_day > 0:
        m.addConstr(gp.quicksum(pe[k] for k in range(H)) * ag.eta_kg_per_kwh * dt
                    >= ag.h2_target_kg_day)

    obj = gp.QuadExpr()
    for k in range(H):
        obj.add(0.5 * gamma * p[k] * p[k])
        obj.add((float(c[k]) - gamma * float(p_da)) * p[k])
        if pdis is not None:
            obj.add(-h2dt * pdis[k])
            obj.add((h2dt / (ag.bat_eta_ch * ag.bat_eta_dis)) * pch[k])
        if cv is not None:
            obj.add(h2dt * cv[k])
    if h2_target_rtm is not None:
        s = m.addVars(H, lb=0.0, name=f's_h2_{i}')
        cum = gp.LinExpr()
        for k in range(H):
            cum.add(pe[k], ag.eta_kg_per_kwh * dt)
            m.addConstr(cum + s[k] >= float(h2_target_rtm[k]))
            obj.add(s[k], H2_SLACK_PENALTY)
    return dict(p=p, pch=pch, pdis=pdis, pe=pe, p_ub=p_ub), obj


def solve_central(
    agents: list[ElzAgent], lam: np.ndarray, p_da: np.ndarray, g_ren: np.ndarray,
    H: int, dt: float, bat_soc: np.ndarray, l_min: float, l_max: float,
    u_fixed: Optional[np.ndarray], mode: str,
    p_elec_init: Optional[np.ndarray] = None,
    h2_target_rtm: Optional[np.ndarray] = None,
    skip: frozenset = frozenset(),
    couple: bool = True,
) -> CentralResult:
    """Joint MIQP.  Agents in `skip` are left out (p = 0), mirroring the GNE,
    where an agent whose local MIQP is infeasible (e.g. it cannot ramp down
    fast enough) falls back to p = 0 while the others still optimise."""
    N = len(agents)
    m = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)
    obj = gp.QuadExpr()
    V = {}
    for i, ag in enumerate(agents):
        if i in skip:
            continue
        gamma = 0.0 if mode == 'dam' else ag.gamma
        v, o = _add_agent(
            m, i, ag, H, dt, float(bat_soc[i]), lam, float(p_da[i]), g_ren[i], gamma,
            add_h2=(mode == 'dam'),
            u_fixed=None if u_fixed is None else int(u_fixed[i]),
            p_elec_init=0.0 if p_elec_init is None else float(p_elec_init[i]),
            h2_target_rtm=None if h2_target_rtm is None else h2_target_rtm[i])
        V[i] = v
        obj.add(o)

    in_all = list(range(N)) if u_fixed is None else [i for i in range(N) if int(u_fixed[i]) == 1]
    in_mkt = [i for i in in_all if i in V]
    M = sum(V[i]['p_ub'] for i in in_mkt)
    if np.isfinite(l_max):
        M = min(M, float(l_max))
    slack = {}
    if couple and in_mkt and l_min > 0.0:
        for k in range(H):
            slack[k] = m.addVar(lb=0.0, name=f's_agg{k}')
            obj.add(slack[k], AGG_SLACK_PENALTY)
            agg = gp.quicksum(V[i]['p'][k] for i in in_mkt)
            if np.isfinite(l_max):
                m.addConstr(agg <= float(l_max))
            if u_fixed is None:
                # DAM: optional participation per step, {0} ∪ [L_min, L_max]
                U = m.addVar(vtype=GRB.BINARY)
                m.addConstr(agg + slack[k] >= float(l_min) * U)
                m.addConstr(agg <= M * U)
            else:
                m.addConstr(agg + slack[k] >= float(l_min))
                pda = float(sum(float(p_da[i]) for i in in_all))
                if pda > 1.0:
                    # RT increment rule on the aggregate (per step)
                    lo_hi = max(float(l_min), pda)
                    w = m.addVar(vtype=GRB.BINARY)
                    m.addConstr(agg <= lo_hi + (max(M, lo_hi) - lo_hi) * w)
                    m.addConstr(agg + slack[k] >= (pda + float(l_min)) * w)

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    solved = m.SolCount > 0
    Z = np.zeros((N, H))
    P, PE, PCH, PDS = Z.copy(), Z.copy(), Z.copy(), Z.copy()
    if solved:
        for i, v in V.items():
            P[i]  = [v['p'][k].X for k in range(H)]
            PE[i] = [v['pe'][k].X for k in range(H)]
            if v['pch'] is not None:
                PCH[i] = [v['pch'][k].X for k in range(H)]
                PDS[i] = [v['pdis'][k].X for k in range(H)]
        sl  = float(sum(v.X for v in slack.values()))
        gap = float(m.MIPGap) if m.IsMIP else 0.0
    m.dispose()

    if not solved and couple and not skip:
        # find agents whose own block is infeasible, drop them, re-solve
        bad = frozenset(
            i for i in range(N)
            if solve_central(agents, lam, p_da, g_ren, H, dt, bat_soc, l_min, l_max,
                             u_fixed, mode, p_elec_init, h2_target_rtm,
                             skip=frozenset(set(range(N)) - {i}), couple=False).n_skipped < 0)
        if bad:
            r = solve_central(agents, lam, p_da, g_ren, H, dt, bat_soc, l_min, l_max,
                              u_fixed, mode, p_elec_init, h2_target_rtm, skip=bad)
            r.n_skipped = len(bad)
            return r
    if not solved:
        return CentralResult(p=Z, p_ch=Z, p_dis=Z, p_elec=Z, u=np.zeros(N, dtype=int),
                             converged=False, agg_slack=float('nan'), mip_gap=float('nan'),
                             n_skipped=-1)
    return CentralResult(p=P, p_ch=PCH, p_dis=PDS, p_elec=PE,
                         u=(P > 1.0).any(axis=1).astype(int),
                         converged=True, agg_slack=sl, mip_gap=gap, n_skipped=len(skip))
