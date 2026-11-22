"""
diag_simultaneous.py
Quantifies simultaneous charge/discharge in the RAW ADMM output (before any
post-processing in simulation.py) for both DAM and RTM over N_DAYS.
Then re-runs with two larger quadratic penalties on pch/pdis to see whether
the events can be eliminated without distorting the dispatch.

Usage:  python diag_simultaneous.py

Output: per-run tables (DAM + RTM) + a side-by-side comparison.
"""
from __future__ import annotations
import sys, time
import numpy as np
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

proj = Path(__file__).parent
sys.path.insert(0, str(proj))

from agents import (ElzAgent, build_fleet,
                    DT_RT, DT_DA, H_RT, H_DA, STEPS_PER_DAY)
from forecasting import load_pjm_week, get_day, dam_forecast, rt_forecast
import admm as admm_mod
from admm import _env

# ── Config ────────────────────────────────────────────────────────────────────
N_DAYS     = 2
START_DATE = '2024-07-08'
SEED       = 42
RHO_RT     = 0.05
RHO_DA     = 1e-4
L_MIN      = 100.0
L_MAX      = 200.0
H2_PROFILE = [1.00, 1.00, 1.00, 1.00, 0.90, 0.65, 0.45]

# Anything above this is counted as "simultaneous" (kW)
THRESH = 1e-3

# Penalty levels to compare: 0 = baseline, 1e-3 = light, 1e-2 = strong
EPS_VALUES = [0.0, 1e-3, 1e-2]


# ── Patched battery QP ────────────────────────────────────────────────────────

def make_battery_qp_fn(eps_quad: float):
    """
    Returns a drop-in replacement for admm._x_gurobi_battery.
    Identical to the original except it adds eps_quad*(pch[k]^2 + pdis[k]^2)
    to the objective.  eps_quad=0 is bit-for-bit equivalent to the original.
    """
    def _fn(ag, p_da, c, g_ren, z_i, mu_i, rho, dt, E_0,
            add_h2=False, gamma_eff=None):
        H = len(c)
        m = gp.Model(env=_env())
        m.setParam('OutputFlag', 0)

        p_ub = ag.p_elec_max_kw + ag.bat_p_ch_max_kw
        p    = m.addVars(H, lb=0.0, ub=p_ub,                name='p')
        pch  = m.addVars(H, lb=0.0, ub=ag.bat_p_ch_max_kw, name='pch')
        pdis = m.addVars(H, lb=0.0, ub=ag.bat_p_dis_max_kw, name='pdis')
        E    = m.addVars(H+1, lb=0.0, ub=ag.bat_cap_kwh,   name='E')

        bat_max = min(ag.bat_p_ch_max_kw, ag.bat_p_dis_max_kw)
        for k in range(H):
            m.addConstr(pch[k] + pdis[k] <= bat_max)
        m.addConstr(E[0] == float(E_0))
        for k in range(H):
            m.addConstr(
                E[k+1] == E[k]
                         + ag.bat_eta_ch  * pch[k]  * dt
                         - pdis[k] / ag.bat_eta_dis * dt
            )
        for k in range(H):
            gk = float(g_ren[k])
            m.addConstr(p[k] + gk + pdis[k] - pch[k] >= ag.p_elec_min_kw)
            m.addConstr(p[k] + gk + pdis[k] - pch[k] <= ag.p_elec_max_kw)

        if add_h2:
            expr = gp.LinExpr()
            for k in range(H):
                gk   = float(g_ren[k])
                e_dt = ag.eta_kg_per_kwh * dt
                expr.add(p[k],     e_dt)
                expr.add(pdis[k],  e_dt)
                expr.add(pch[k],  -e_dt)
                expr.addConstant(gk * e_dt)
            m.addConstr(expr >= ag.h2_target_kg_day)

        g_ag     = ag.gamma if gamma_eff is None else gamma_eff
        h2_dt    = ag.r_h2 * ag.eta_kg_per_kwh * dt
        eta_rt   = ag.bat_eta_ch * ag.bat_eta_dis
        h2_dt_ch = h2_dt / eta_rt

        obj = gp.QuadExpr()
        for k in range(H):
            qc = 0.5 * (g_ag + rho)
            lc = (float(c[k]) + float(mu_i[k])
                  - g_ag * float(p_da) - rho * float(z_i[k]))
            obj.add(qc * p[k]  * p[k])
            obj.add(lc * p[k])
            obj.add(-h2_dt    * pdis[k])
            obj.add( h2_dt_ch * pch[k])
            if eps_quad > 0.0:
                obj.add(eps_quad * pch[k]  * pch[k])
                obj.add(eps_quad * pdis[k] * pdis[k])

        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        if m.Status == GRB.OPTIMAL:
            p_s   = np.array([p[k].X    for k in range(H)])
            pch_s = np.array([pch[k].X  for k in range(H)])
            pd_s  = np.array([pdis[k].X for k in range(H)])
        else:
            denom = g_ag + rho
            u = (g_ag * p_da - c - mu_i + rho * z_i) / denom if denom > 0 else np.zeros(H)
            p_s   = np.clip(u, 0.0, ag.p_elec_max_kw)
            pch_s = np.zeros(H)
            pd_s  = np.zeros(H)

        m.dispose()
        return p_s, pch_s, pd_s

    return _fn


# ── Stats collector ───────────────────────────────────────────────────────────

class Stats:
    def __init__(self, agents: list[ElzAgent]):
        self.agents = agents
        N = len(agents)

        # DAM: one call per day
        self.dam_n_simul  = np.zeros(N, int)   # simultaneous (agent,h) pairs total
        self.dam_n_total  = np.zeros(N, int)   # total (agent,h) pairs checked
        self.dam_mags     = [[] for _ in range(N)]  # min(pch,pdis) when simultaneous

        # RTM: 288 calls per day
        self.rtm_n_simul      = np.zeros(N, int)
        self.rtm_n_total      = np.zeros(N, int)
        self.rtm_calls_w_simul = np.zeros(N, int)  # how many calls had any simultaneous
        self.rtm_n_calls      = np.zeros(N, int)
        self.rtm_mags         = [[] for _ in range(N)]

    def record(self, mode: str, p_ch: np.ndarray, p_dis: np.ndarray):
        """p_ch, p_dis : (N, H) raw ADMM output."""
        N, H = p_ch.shape
        if mode == 'dam':
            ns, nt, mg = self.dam_n_simul, self.dam_n_total, self.dam_mags
        else:
            ns, nt, mg = self.rtm_n_simul, self.rtm_n_total, self.rtm_mags

        for i in range(N):
            mask = (p_ch[i] > THRESH) & (p_dis[i] > THRESH)
            n    = int(mask.sum())
            ns[i] += n
            nt[i] += H
            if n > 0:
                mg[i].extend(np.minimum(p_ch[i][mask], p_dis[i][mask]).tolist())
            if mode == 'rtm':
                self.rtm_n_calls[i] += 1
                if n > 0:
                    self.rtm_calls_w_simul[i] += 1


# ── Simulation loop ───────────────────────────────────────────────────────────

def run_sim(eps_quad: float) -> Stats:
    """
    Run N_DAYS of DAM+RTM, recording raw ADMM p_ch/p_dis before any post-processing.
    Monkeypatches admm._x_gurobi_battery for the duration, then restores it.
    """
    orig_fn = admm_mod._x_gurobi_battery
    admm_mod._x_gurobi_battery = make_battery_qp_fn(eps_quad)

    agents  = build_fleet()
    N       = len(agents)
    rng     = np.random.default_rng(SEED)
    stats   = Stats(agents)
    h2_base = [0.75 * ag.p_elec_max_kw * ag.eta_kg_per_kwh * 24 for ag in agents]

    print(f"  Loading PJM data ... ", end='', flush=True)
    week = load_pjm_week(START_DATE, N_DAYS)
    print("done.")

    carry_soc = np.array([ag.bat_soc_init_frac * ag.bat_cap_kwh for ag in agents])

    try:
        for d in range(N_DAYS):
            # Reset per-day state
            for i, ag in enumerate(agents):
                ag.h2_inv_kg        = 0.0
                ag.bat_soc_kwh      = float(carry_soc[i])
                ag.h2_target_kg_day = h2_base[i] * H2_PROFILE[d % len(H2_PROFILE)]

            # ── DAM ───────────────────────────────────────────────────────────
            dam_fc      = dam_forecast(week, d, agents, rng)
            bat_soc_dam = np.array([ag.bat_soc_kwh for ag in agents])
            dam_res = admm_mod.solve(
                agents   = agents,
                lam      = dam_fc['da_price'],
                p_da     = np.zeros(N),
                g_ren    = dam_fc['ren'],
                H        = H_DA,
                dt       = DT_DA,
                bat_soc  = bat_soc_dam,
                rho      = RHO_DA,
                max_iter = 2000,
                tol      = 5.0,
                mode     = 'dam',
                l_min    = L_MIN,
                l_max    = L_MAX,
            )
            stats.record('dam', dam_res.p_ch, dam_res.p_dis)
            print(f"    Day {d+1} DAM : {dam_res.n_iter:4d} iters "
                  f"({'conv' if dam_res.converged else 'NOCONV'})", end='  |  ')

            # ── RTM ───────────────────────────────────────────────────────────
            cur_soc = np.array([ag.bat_soc_kwh for ag in agents])
            warm    = None

            for t in range(STEPS_PER_DAY):
                hour   = t // 12
                p_da_t = dam_res.p[:, hour]

                g_fc  = np.zeros((N, H_RT))
                lam_t = np.zeros(H_RT)
                for i, ag in enumerate(agents):
                    fc = rt_forecast(t, week, d, H_RT,
                                     ag.renewable_type, ag.renewable_cap_kw, rng)
                    g_fc[i] = fc['ren']
                    if i == 0:
                        lam_t = fc['lam']

                rtm = admm_mod.solve(
                    agents   = agents,
                    lam      = lam_t,
                    p_da     = p_da_t,
                    g_ren    = g_fc,
                    H        = H_RT,
                    dt       = DT_RT,
                    bat_soc  = cur_soc,
                    rho      = RHO_RT,
                    max_iter = 300,
                    tol      = 1.0,
                    mode     = 'rtm',
                    warm     = warm,
                    l_min    = L_MIN,
                    l_max    = L_MAX,
                )
                stats.record('rtm', rtm.p_ch, rtm.p_dis)

                # Minimal SoC update for next call (no curtailment / post-processing)
                pch0 = rtm.p_ch[:, 0].copy()
                pdi0 = rtm.p_dis[:, 0].copy()
                for i, ag in enumerate(agents):
                    if ag.has_battery:
                        pch0[i] = np.clip(pch0[i], 0.0, ag.bat_p_ch_max_kw)
                        pdi0[i] = np.clip(pdi0[i], 0.0, ag.bat_p_dis_max_kw)
                        cur_soc[i] = np.clip(
                            cur_soc[i]
                            + ag.bat_eta_ch  * pch0[i] * DT_RT
                            - pdi0[i] / ag.bat_eta_dis * DT_RT,
                            0.0, ag.bat_cap_kwh
                        )
                warm = rtm.p

            carry_soc = cur_soc.copy()
            print(f"RTM done.")

    finally:
        admm_mod._x_gurobi_battery = orig_fn   # always restore

    return stats


# ── Reporting helpers ─────────────────────────────────────────────────────────

def report_one(stats: Stats, label: str):
    agents   = stats.agents
    bat_idx  = [i for i, ag in enumerate(agents) if ag.has_battery]
    total_rtm_calls = N_DAYS * STEPS_PER_DAY

    print(f"\n{'─'*74}")
    print(f"  Run : {label}")
    print(f"{'─'*74}")

    # DAM
    print(f"\n  DAM (H=24, {N_DAYS} call(s) total):")
    hdr = f"  {'Agent':<16} {'Simul steps':>12} {'/ Total':>8} {'%':>7}  {'Mean kW':>9}  {'Max kW':>9}"
    print(hdr)
    print("  " + "-"*60)
    for i in bat_idx:
        n   = stats.dam_n_simul[i]
        t   = stats.dam_n_total[i]
        mg  = stats.dam_mags[i]
        pct = n / t * 100 if t else 0.0
        print(f"  {agents[i].name:<16} {n:>12} {t:>8} {pct:>6.1f}%  "
              f"{np.mean(mg) if mg else 0:>9.4f}  {np.max(mg) if mg else 0:>9.4f}")

    # RTM
    print(f"\n  RTM (H=12, {total_rtm_calls} calls total [{STEPS_PER_DAY}/day × {N_DAYS} days]):")
    hdr2 = (f"  {'Agent':<16} {'Simul steps':>12} {'/ Total':>8} {'%':>7}  "
            f"{'Calls w/simul':>14} {'/ '+str(total_rtm_calls):>8} {'%':>7}  "
            f"{'Mean kW':>9}  {'Max kW':>9}")
    print(hdr2)
    print("  " + "-"*74)
    for i in bat_idx:
        n   = stats.rtm_n_simul[i]
        t   = stats.rtm_n_total[i]
        cw  = stats.rtm_calls_w_simul[i]
        mg  = stats.rtm_mags[i]
        ps  = n  / t               * 100 if t               else 0.0
        pc  = cw / total_rtm_calls * 100 if total_rtm_calls else 0.0
        print(f"  {agents[i].name:<16} {n:>12} {t:>8} {ps:>6.1f}%  "
              f"{cw:>14} {total_rtm_calls:>8} {pc:>6.1f}%  "
              f"{np.mean(mg) if mg else 0:>9.4f}  {np.max(mg) if mg else 0:>9.4f}")


def report_comparison(all_stats: list[Stats], labels: list[str]):
    agents   = all_stats[0].agents
    bat_idx  = [i for i, ag in enumerate(agents) if ag.has_battery]
    total_t  = N_DAYS * STEPS_PER_DAY * H_RT   # total (step,h) pairs in RTM

    print(f"\n\n{'='*74}")
    print("  COMPARISON — RTM simultaneous step % (raw ADMM, before post-processing)")
    print(f"{'='*74}")

    short_labels = [lb.replace('eps_quad = ', 'ε=') for lb in labels]
    col_w = 13

    print(f"  {'Agent':<16}", end='')
    for sl in short_labels:
        print(f"  {sl:>{col_w}}", end='')
    print(f"  {'Reduction':>{col_w}}")
    print("  " + "-"*(16 + (col_w + 2) * (len(labels) + 1)))

    for i in bat_idx:
        print(f"  {agents[i].name:<16}", end='')
        pcts = []
        for st in all_stats:
            n   = st.rtm_n_simul[i]
            t   = st.rtm_n_total[i]
            pct = n / t * 100 if t else 0.0
            pcts.append(pct)
            print(f"  {pct:>{col_w}.3f}%", end='')
        red = f"{100*(pcts[0]-pcts[-1])/pcts[0]:.0f}%" if pcts[0] > 0 else "—"
        print(f"  {red:>{col_w}}")

    print(f"\n  Same for DAM:")
    print(f"  {'Agent':<16}", end='')
    for sl in short_labels:
        print(f"  {sl:>{col_w}}", end='')
    print(f"  {'Reduction':>{col_w}}")
    print("  " + "-"*(16 + (col_w + 2) * (len(labels) + 1)))
    for i in bat_idx:
        print(f"  {agents[i].name:<16}", end='')
        pcts = []
        for st in all_stats:
            n   = st.dam_n_simul[i]
            t   = st.dam_n_total[i]
            pct = n / t * 100 if t else 0.0
            pcts.append(pct)
            print(f"  {pct:>{col_w}.3f}%", end='')
        red = f"{100*(pcts[0]-pcts[-1])/pcts[0]:.0f}%" if pcts[0] > 0 else "—"
        print(f"  {red:>{col_w}}")

    print(f"\n  Threshold = {THRESH} kW  |  Values = min(p_ch, p_dis) when both > threshold")
    print(f"  These are RAW ADMM values — simulation.py zeros the smaller one afterward.")
    print(f"  eps = quadratic penalty added to both pch^2 and pdis^2 in the QP objective.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_stats = []
    labels    = []

    for eps in EPS_VALUES:
        lbl = f"eps_quad = {eps:.3f}"
        print(f"\n{'='*74}")
        print(f"  Running: {lbl}")
        print(f"{'='*74}")
        t0 = time.time()
        st = run_sim(eps)
        elapsed = time.time() - t0
        print(f"  Wall time: {elapsed:.1f} s")
        all_stats.append(st)
        labels.append(lbl)
        report_one(st, lbl)

    report_comparison(all_stats, labels)
    print("\nDone.")
