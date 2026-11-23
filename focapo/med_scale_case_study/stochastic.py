"""
stochastic.py — scenario-based (two-stage) versions of the agent MIQP, the
individual solver and the GNE MIQP-ADMM  (--scenarios S, needs --forecaster xgb).

Scenarios s = 1..S (probability pi_s) come from the forecaster: point forecast
plus a past out-of-sample forecast error (forecaster/provider.py).  A scenario
is a joint draw: one price path shared by all agents, one renewable path per
agent, so the coalition constraint can be written per scenario.

Non-anticipativity (what must be decided before the uncertainty is known):
  DAM  (nonant='all')   : grid import p_k for all 24 h and the market binaries
                          are first-stage (the DA bid); battery, curtailment
                          and electrolyzer load adapt per scenario.
  RTM  (nonant='first') : step-0 import, charge and discharge (what is executed
                          now) are first-stage; steps 1..H-1 adapt per scenario.

Objective = expected cost  sum_s pi_s * (same per-step cost as _solve_agent).
GNE: ADMM with z, mu of shape (S, H); the z-update projects each scenario's
aggregate exactly as the deterministic solver does (per step).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from miqp_admm import (_env, H2_SLACK_PENALTY, GNEResult, IndResult,
                       _project_sc_agg, _project_rule_b_agg)
from agents import ElzAgent

# The scenario MIQPs are ~10x larger than the deterministic ones; a few
# (battery agent, RT increment rule with per-step binaries) find a good
# solution fast but take minutes to prove optimality.  Stop at a 0.1 % gap or
# after STOCH_TIME_LIMIT s (best solution found is used; hits are counted).
STOCH_MIP_GAP = 1e-3
STOCH_TIME_LIMIT = 3.0
STATS = {'solves': 0, 'time_limit_hits': 0}


def _solve_agent_stoch(
    ag: ElzAgent, H: int, dt: float, E_0: float,
    lam: np.ndarray,          # (S,H) price scenarios [$/MWh]
    p_da: float,
    g_ren: np.ndarray,        # (S,H) renewable scenarios [kW]
    prob: np.ndarray,         # (S,)
    rho: float, z_i: np.ndarray, mu_i: np.ndarray,   # (S,H)
    u_fixed: Optional[int], gamma: float, add_h2: bool, l_min: float,
    nonant: str,
    p_elec_init: float = 0.0,
    h2_target_rtm: Optional[np.ndarray] = None,      # (H,)
    l_min_dev: float = 0.0,
    u_per_step: bool = False, rule_b_per_step: bool = False,
):
    S = lam.shape[0]
    c = (lam / 1000.0 - ag.r_h2 * ag.eta_kg_per_kwh) * dt          # (S,H)
    h2dt = ag.r_h2 * ag.eta_kg_per_kwh * dt
    p_ub = ag.p_elec_max_kw + (ag.bat_p_ch_max_kw if ag.has_battery else 0.0)
    K_p = H if nonant == 'all' else 1          # steps with first-stage import
    K_b = 0 if nonant == 'all' else 1          # steps with first-stage battery

    m = gp.Model(env=_env())
    m.setParam('OutputFlag', 0)
    m.setParam('MIPGap', STOCH_MIP_GAP)
    m.setParam('TimeLimit', STOCH_TIME_LIMIT)
    p = m.addVars(S, H, lb=0.0, ub=float(p_ub), name='p')
    for s in range(1, S):
        for k in range(K_p):
            m.addConstr(p[s, k] == p[0, k])

    # market participation (first-stage binaries are shared across scenarios)
    u_var = u_steps = None
    if u_fixed is None and u_per_step:
        u_steps = m.addVars(H, vtype=GRB.BINARY, name='u_k')
        for s in range(S):
            for k in range(H):
                m.addConstr(p[s, k] >= float(l_min) * u_steps[k])
                m.addConstr(p[s, k] <= float(p_ub) * u_steps[k])
    elif u_fixed is None:
        u_var = m.addVar(vtype=GRB.BINARY, name='u')
        for s in range(S):
            for k in range(H):
                m.addConstr(p[s, k] >= float(l_min) * u_var)
                m.addConstr(p[s, k] <= float(p_ub) * u_var)
    elif int(u_fixed) == 0:
        for s in range(S):
            for k in range(H):
                m.addConstr(p[s, k] == 0.0)
    else:
        for s in range(S):
            for k in range(H):
                m.addConstr(p[s, k] >= float(l_min))
        pdv = float(p_da)
        if l_min_dev > 0.0 and pdv > 1.0:
            for s in range(S):
                if rule_b_per_step:
                    v = m.addVars(H, vtype=GRB.BINARY)
                    vk = lambda k: v[k]                       # noqa: E731
                else:
                    v1 = m.addVar(vtype=GRB.BINARY)
                    vk = lambda k: v1                         # noqa: E731
                for k in range(H):
                    m.addConstr(p[s, k] <= pdv + (float(p_ub) - pdv) * vk(k))
                    m.addConstr(p[s, k] >= (pdv + float(l_min_dev)) * vk(k))

    cv = m.addVars(S, H, lb=0.0, name='cv') if ag.has_renewable else None
    if cv is not None:
        for s in range(S):
            for k in range(H):
                m.addConstr(cv[s, k] <= max(0.0, float(g_ren[s, k])))

    if ag.has_battery:
        pch = m.addVars(S, H, lb=0.0, ub=ag.bat_p_ch_max_kw, name='pch')
        pdis = m.addVars(S, H, lb=0.0, ub=ag.bat_p_dis_max_kw, name='pdis')
        E = m.addVars(S, H + 1, lb=0.0, ub=ag.bat_cap_kwh, name='E')
        d = m.addVars(S, H, vtype=GRB.BINARY, name='d')
        for s in range(S):
            m.addConstr(E[s, 0] == float(E_0))
            for k in range(H):
                m.addConstr(pch[s, k] <= ag.bat_p_ch_max_kw * d[s, k])
                m.addConstr(pdis[s, k] <= ag.bat_p_dis_max_kw * (1 - d[s, k]))
                m.addConstr(E[s, k + 1] == E[s, k] + ag.bat_eta_ch * pch[s, k] * dt
                            - pdis[s, k] / ag.bat_eta_dis * dt)
        for s in range(1, S):
            for k in range(K_b):
                m.addConstr(pch[s, k] == pch[0, k])
                m.addConstr(pdis[s, k] == pdis[0, k])
    else:
        pch = pdis = None

    pe = m.addVars(S, H, lb=0.0, ub=ag.p_elec_max_kw, name='p_elec')
    ramp = ag.ramp_rate_kw_per_hr * dt
    ramp_binding = ramp < ag.p_elec_max_kw      # hourly DAM: 6*Pmax >= Pmax, redundant
    for s in range(S):
        for k in range(H):
            m.addConstr(pe[s, k] == p[s, k] + float(g_ren[s, k])
                        - (cv[s, k] if cv is not None else 0.0)
                        + (pdis[s, k] if pdis is not None else 0.0)
                        - (pch[s, k] if pch is not None else 0.0))
            if ramp_binding:
                prev = pe[s, k - 1] if k > 0 else float(p_elec_init)
                m.addConstr(pe[s, k] - prev <= ramp)
                m.addConstr(prev - pe[s, k] <= ramp)
        if add_h2 and ag.h2_target_kg_day > 0:
            m.addConstr(gp.quicksum(pe[s, k] for k in range(H)) * ag.eta_kg_per_kwh * dt
                        >= ag.h2_target_kg_day)

    obj = gp.QuadExpr()
    for s in range(S):
        w = float(prob[s])
        for k in range(H):
            obj.add(w * 0.5 * (gamma + rho) * p[s, k] * p[s, k])
            obj.add(w * (float(c[s, k]) + float(mu_i[s, k]) - gamma * float(p_da)
                         - rho * float(z_i[s, k])) * p[s, k])
            if pdis is not None:
                obj.add(-w * h2dt * pdis[s, k])
                obj.add(w * (h2dt / (ag.bat_eta_ch * ag.bat_eta_dis)) * pch[s, k])
            if cv is not None:
                obj.add(w * h2dt * cv[s, k])
        if h2_target_rtm is not None:
            sl = m.addVars(H, lb=0.0)
            cum = gp.LinExpr()
            for k in range(H):
                cum.add(pe[s, k], ag.eta_kg_per_kwh * dt)
                m.addConstr(cum + sl[k] >= float(h2_target_rtm[k]))
                obj.add(sl[k], w * H2_SLACK_PENALTY)
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    STATS['solves'] += 1
    STATS['time_limit_hits'] += int(m.Status == GRB.TIME_LIMIT)

    Z = np.zeros((S, H))
    if m.SolCount > 0:
        P = np.array([[p[s, k].X for k in range(H)] for s in range(S)])
        PE = np.array([[pe[s, k].X for k in range(H)] for s in range(S)])
        PCH = np.array([[pch[s, k].X for k in range(H)] for s in range(S)]) if pch is not None else Z
        PDS = np.array([[pdis[s, k].X for k in range(H)] for s in range(S)]) if pdis is not None else Z
        if u_var is not None:
            u = int(round(u_var.X))
        elif u_steps is not None:
            u = int(max(round(u_steps[k].X) for k in range(H)))
        else:
            u = 0 if (u_fixed is not None and int(u_fixed) == 0) else 1
        ok = True
    else:
        out = (u_fixed is not None and int(u_fixed) == 0)
        P = Z.copy() if out else np.full((S, H), float(l_min))
        PE, PCH, PDS, u, ok = Z.copy(), Z.copy(), Z.copy(), (0 if out else 1), False
    m.dispose()
    return dict(p=P, p_elec=PE, p_ch=PCH, p_dis=PDS, u=u, feasible=ok)


def _expect(x: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """(N,S,H) → (N,H) probability-weighted mean (step 0 is identical over s in RTM)."""
    return np.tensordot(x, prob, axes=([1], [0]))


def solve_individual_stoch(agents, lam, p_da, g_ren, prob, H, dt, bat_soc, l_min,
                           u_fixed, mode, p_elec_init=None, h2_target_rtm=None,
                           dam_u_per_step=False, rt_rule_b_per_step=False) -> IndResult:
    """lam (S,H); g_ren (N,S,H)."""
    N, S = len(agents), lam.shape[0]
    out = {k: np.zeros((N, S, H)) for k in ('p', 'p_elec', 'p_ch', 'p_dis')}
    u = np.ones(N, dtype=int)
    Z = np.zeros((S, H))
    for i, ag in enumerate(agents):
        r = _solve_agent_stoch(
            ag, H, dt, float(bat_soc[i]), lam, float(p_da[i]), g_ren[i], prob,
            0.0, Z, Z, None if u_fixed is None else int(u_fixed[i]),
            0.0 if mode == 'dam' else ag.gamma, mode == 'dam', l_min,
            nonant='all' if mode == 'dam' else 'first',
            p_elec_init=0.0 if p_elec_init is None else float(p_elec_init[i]),
            h2_target_rtm=None if h2_target_rtm is None else h2_target_rtm[i],
            l_min_dev=l_min if mode == 'rtm' else 0.0,
            u_per_step=(dam_u_per_step and mode == 'dam'),
            rule_b_per_step=(rt_rule_b_per_step and mode == 'rtm'))
        for k in out:
            out[k][i] = r[k]
        u[i] = r['u']
    E = {k: _expect(v, prob) for k, v in out.items()}
    return IndResult(p=E['p'], p_ch=E['p_ch'], p_dis=E['p_dis'], p_elec=E['p_elec'], u=u)


def solve_gne_stoch(agents, lam, p_da, g_ren, prob, H, dt, bat_soc, rho, max_iter, tol,
                    l_min, l_max, u_fixed, mode, p_elec_init=None,
                    h2_target_rtm=None) -> GNEResult:
    """Stochastic MIQP-ADMM. lam (S,H); g_ren (N,S,H). Agents share only the
    per-scenario aggregate import."""
    N, S = len(agents), lam.shape[0]
    nonant = 'all' if mode == 'dam' else 'first'
    gam = [0.0 if mode in ('dam', 'commit') else ag.gamma for ag in agents]

    # warm start: unconstrained per-scenario minimiser, as in solve_gne
    p = []
    for i, ag in enumerate(agents):
        c = (lam / 1000.0 - ag.r_h2 * ag.eta_kg_per_kwh) * dt
        pu = (gam[i] * float(p_da[i]) - c) / (gam[i] + rho) if gam[i] + rho > 1e-12 else np.zeros((S, H))
        if nonant == 'all':
            pu = np.repeat((prob @ pu)[None, :], S, axis=0)
        else:
            pu[:, 0] = prob @ pu[:, 0]
        if u_fixed is not None and int(u_fixed[i]) == 0:
            pu = np.zeros((S, H))
        p.append(np.clip(pu, 0.0, ag.p_elec_max_kw))
    z = [x.copy() for x in p]
    mu = [np.zeros((S, H)) for _ in range(N)]
    res = [None] * N
    in_mkt = list(range(N)) if u_fixed is None else [i for i in range(N) if int(u_fixed[i]) == 1]
    t_agents, s_star = [], None
    primal = dual = np.inf
    converged, itr = False, 0
    for itr in range(max_iter):
        z_prev = [x.copy() for x in z]
        t_it = np.zeros(N)
        for i, ag in enumerate(agents):
            t0 = time.perf_counter()
            res[i] = _solve_agent_stoch(
                ag, H, dt, float(bat_soc[i]), lam, float(p_da[i]), g_ren[i], prob,
                rho, z[i], mu[i], None if u_fixed is None else int(u_fixed[i]),
                gam[i], mode == 'dam', 0.0, nonant,
                p_elec_init=0.0 if p_elec_init is None else float(p_elec_init[i]),
                h2_target_rtm=None if h2_target_rtm is None else h2_target_rtm[i])
            t_it[i] = time.perf_counter() - t0
            p[i] = res[i]['p']
        t_agents.append(t_it)
        if in_mkt:
            V = np.stack([p[i] + mu[i] / rho for i in in_mkt])        # (n_in,S,H)
            agg = V.sum(axis=0)                                        # (S,H)
            if u_fixed is not None:
                pda = sum(float(p_da[i]) for i in in_mkt)
                s_star = np.vstack([_project_rule_b_agg(agg[s], l_min if l_min > 0 else 0.0,
                                                        l_max, pda) for s in range(S)])
            else:
                s_star = _project_sc_agg(agg, l_min, l_max)
            delta = (s_star - agg) / len(in_mkt)
            for j, i in enumerate(in_mkt):
                z[i] = V[j] + delta
        for i in range(N):
            if i not in in_mkt:
                z[i] = np.zeros((S, H)); mu[i] = np.zeros((S, H))
        for i in in_mkt:
            mu[i] = mu[i] + rho * (p[i] - z[i])
        if in_mkt:
            primal = float(max(np.max(np.abs(p[i] - z[i])) for i in in_mkt))
            dual = float(rho * max(np.max(np.abs(z[i] - z_prev[i])) for i in in_mkt))
        else:
            primal = dual = 0.0
        converged = primal < tol and dual < tol
        if converged:
            break
    st = {k: np.stack([r[k] for r in res]) for k in ('p', 'p_elec', 'p_ch', 'p_dis')}
    E = {k: _expect(v, prob) for k, v in st.items()}
    return GNEResult(p=E['p'], p_ch=E['p_ch'], p_dis=E['p_dis'], p_elec=E['p_elec'],
                     u=np.array([r['u'] for r in res], dtype=int), n_iter=itr + 1,
                     converged=converged, primal_res=primal, dual_res=dual,
                     agg_proj=None if s_star is None else prob @ s_star,
                     t_agents=np.array(t_agents))
