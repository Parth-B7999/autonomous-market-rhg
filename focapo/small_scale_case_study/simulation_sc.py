"""
simulation_sc.py — Closed-loop two-timescale simulation with semi-continuous
                   market participation.

Two entry points:
  run_gne()        — GNE coalition (MIQP-ADMM, Fabiani et al. 2022)
  run_individual() — independent agents (N separate MIQPs)

Three timescales per day
  1. DAM MIQP  (once/day, 24h)     → p_DA anchor  +  u_DA participation
  2. 65-min commitment MIQP        → u_h  locked for the operating hour
  3. RTM MIQP  (every 5 min, 1h)  → executed dispatch  (u fixed, δ free)

PJM 65-min gate
  u for hour h is committed at step t = 12·h − 13 (= 65 min before hour h
  starts). Hours 0 and 1 are pre-committed at day start. Thereafter u for
  hour h+2 is committed at the last step of hour h (t%12 == 11).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from agents import ElzAgent, H_RT, H_DA, DT_RT, DT_DA, STEPS_PER_DAY
from forecasting import load_pjm_week, get_day, dam_forecast, rt_forecast
from miqp_admm import (solve_gne, solve_individual,
                        commit_u_gne, commit_u_individual)


# ── Result container ───────────────────────────────────────────────────────────
@dataclass
class SimResults:
    """All time-series outputs.  D=n_days, T=288, N=n_agents."""
    p_net:    np.ndarray   # (D, T, N)  net grid import [kW]
    p_elec:   np.ndarray   # (D, T, N)  electrolyzer power [kW]
    p_ch:     np.ndarray   # (D, T, N)  battery charging [kW]
    p_dis:    np.ndarray   # (D, T, N)  battery discharging [kW]
    bat_soc:  np.ndarray   # (D, T+1, N) battery SoC [kWh]
    ren:      np.ndarray   # (D, T, N)  true renewable [kW]
    h2_step:  np.ndarray   # (D, T, N)  H₂ produced per step [kg]
    h2_cum:   np.ndarray   # (D, T, N)  cumulative H₂ within day [kg]
    h2_daily: np.ndarray   # (D, N)     total H₂ per day [kg]
    h2_target:np.ndarray   # (D, N)     daily H₂ target [kg]
    lmp_rt:   np.ndarray   # (D, T)     true RT LMP [$/MWh]
    lmp_da:   np.ndarray   # (D, 24)    true DA LMP [$/MWh]
    cost_rt:  np.ndarray   # (D, T, N)  RT energy cost per step [$]
    p_da:     np.ndarray   # (D, 24, N) DA schedule [kW]
    da_iters: np.ndarray   # (D,)       DAM iterations
    da_conv:  np.ndarray   # (D,)       bool
    rt_iters: np.ndarray   # (D, T)     RTM iterations per step
    rt_conv:  np.ndarray   # (D, T)     bool
    agg:      np.ndarray   # (D, T)     aggregate net import [kW]
    lmin_bind:np.ndarray   # (D, T)     bool  aggregate at L_min
    lmax_bind:np.ndarray   # (D, T)     bool  aggregate at L_max
    curtail:  np.ndarray   # (D, T, N)  curtailed renewable [kW]
    participation: np.ndarray = field(default=None)  # (D, 24, N) hourly u ∈{0,1}


def _alloc(n_days: int, T: int, N: int) -> dict:
    """Allocate all output arrays."""
    return {
        'p_net':    np.zeros((n_days, T, N)),
        'p_elec':   np.zeros((n_days, T, N)),
        'p_ch':     np.zeros((n_days, T, N)),
        'p_dis':    np.zeros((n_days, T, N)),
        'bat_soc':  np.zeros((n_days, T + 1, N)),
        'ren':      np.zeros((n_days, T, N)),
        'h2_step':  np.zeros((n_days, T, N)),
        'h2_cum':   np.zeros((n_days, T, N)),
        'h2_daily': np.zeros((n_days, N)),
        'h2_target':np.zeros((n_days, N)),
        'lmp_rt':   np.zeros((n_days, T)),
        'lmp_da':   np.zeros((n_days, 24)),
        'cost_rt':  np.zeros((n_days, T, N)),
        'p_da':     np.zeros((n_days, 24, N)),
        'da_iters': np.zeros(n_days, dtype=int),
        'da_conv':  np.zeros(n_days, dtype=bool),
        'rt_iters': np.zeros((n_days, T), dtype=int),
        'rt_conv':  np.zeros((n_days, T), dtype=bool),
        'agg':      np.zeros((n_days, T)),
        'lmin_b':   np.zeros((n_days, T), dtype=bool),
        'lmax_b':   np.zeros((n_days, T), dtype=bool),
        'curtail':  np.zeros((n_days, T, N)),
        'u_sched':  np.zeros((n_days, 24, N), dtype=int),
    }


def _execute_step(
    agents:    list[ElzAgent],
    p_planned: np.ndarray,   # (N,) planned step-0 import from solver
    pch_plan:  np.ndarray,   # (N,) planned step-0 charge
    pdis_plan: np.ndarray,   # (N,) planned step-0 discharge
    g_true:    np.ndarray,   # (N,) true renewable at this step
    cur_soc:   np.ndarray,   # (N,) current battery SoC
    u_now:     np.ndarray,   # (N,) fixed hourly participation
    l_min:     float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply true renewable to planned dispatch, return executed values + curtailment.

    Returns (p_exec, pch_exec, pdis_exec, curtail) each shape (N,).
    """
    N         = len(agents)
    p_exec    = p_planned.copy()
    pch_exec  = pch_plan.copy()
    pdis_exec = pdis_plan.copy()
    curtail   = np.zeros(N)

    for i, ag in enumerate(agents):
        u_i = int(u_now[i])

        if u_i == 0:
            # Out of market: no grid import; electrolyzer runs on renewable/battery
            p_exec[i]   = 0.0
            pch_exec[i] = 0.0
            pdis_exec[i] = 0.0
        else:
            # In market: enforce l_min floor
            p_exec[i] = max(float(l_min), p_exec[i])
            if ag.has_battery:
                pch_exec[i]  = np.clip(pch_exec[i],  0.0, ag.bat_p_ch_max_kw)
                pdis_exec[i] = np.clip(pdis_exec[i], 0.0,
                                        min(ag.bat_p_dis_max_kw,
                                            cur_soc[i] / (DT_RT / ag.bat_eta_dis + 1e-9)))

        # Overflow: true renewable may exceed forecast → curtail renewable
        if ag.has_battery:
            excess = p_exec[i] + g_true[i] + pdis_exec[i] - pch_exec[i] - ag.p_elec_max_kw
            if excess > 1e-6:
                dis_cut       = min(excess, pdis_exec[i])
                pdis_exec[i] -= dis_cut
                excess        -= dis_cut
            if excess > 1e-6:
                ch_room = min(
                    ag.bat_p_ch_max_kw - pch_exec[i],
                    max(0.0, (ag.bat_cap_kwh - cur_soc[i]) / (DT_RT * ag.bat_eta_ch + 1e-9))
                )
                absorbed     = min(excess, ch_room)
                pch_exec[i] += absorbed
                excess       -= absorbed
            curtail[i] = max(0.0, excess)
        else:
            overflow = p_exec[i] + g_true[i] - ag.p_elec_max_kw
            curtail[i] = max(0.0, overflow)

    return p_exec, pch_exec, pdis_exec, curtail


# ── Shared inner loop ──────────────────────────────────────────────────────────
def _run(
    scenario:   str,          # 'gne' or 'individual'
    agents:     list[ElzAgent],
    n_days:     int,
    start_date: str,
    seed:       int,
    rho_rt:     float,
    rho_da:     float,
    tol_rt:     float,
    tol_da:     float,
    l_min:      float,
    l_max:      float,
    h2_profile: list,
) -> tuple[SimResults, list[ElzAgent]]:

    N, T = len(agents), STEPS_PER_DAY
    rng  = np.random.default_rng(seed)
    # Separate RNG for 65-min-ahead commitment forecasts (keeps DA/RT streams
    # identical between GNE and individual for a fair comparison).
    crng = np.random.default_rng(seed + 1000)

    print(f"Loading PJM data from {start_date} ({n_days} days)…")
    week = load_pjm_week(start_date, n_days)
    arr  = _alloc(n_days, T, N)

    h2_base   = [0.75 * ag.p_elec_max_kw * ag.eta_kg_per_kwh * 24 for ag in agents]
    carry_soc = np.array([ag.bat_soc_init_frac * ag.bat_cap_kwh for ag in agents])

    for d in range(n_days):
        # ── daily reset ───────────────────────────────────────────────────────
        for i, ag in enumerate(agents):
            ag.h2_inv_kg   = 0.0
            ag.bat_soc_kwh = float(carry_soc[i])
        prof_d = h2_profile[d % len(h2_profile)]
        for i, ag in enumerate(agents):
            ag.h2_target_kg_day = h2_base[i] * prof_d

        day = get_day(week, d, agents)
        arr['lmp_rt'][d] = day['rt_price']
        arr['lmp_da'][d] = day['da_price']
        arr['h2_target'][d] = [ag.h2_target_kg_day for ag in agents]

        # ── DAM MIQP (once per day) ───────────────────────────────────────────
        dam_fc      = dam_forecast(week, d, agents, rng)
        bat_soc_dam = np.array([ag.bat_soc_kwh for ag in agents])

        if scenario == 'gne':
            dam_res = solve_gne(
                agents, dam_fc['da_price'], np.zeros(N), dam_fc['ren'],
                H_DA, DT_DA, bat_soc_dam,
                rho=rho_da, max_iter=2000, tol=tol_da,
                l_min=l_min, l_max=l_max,
                u_fixed=None, mode='dam',
            )
        else:  # individual
            dam_res = solve_individual(
                agents, dam_fc['da_price'], np.zeros(N), dam_fc['ren'],
                H_DA, DT_DA, bat_soc_dam,
                l_min=l_min, u_fixed=None, mode='dam',
            )

        arr['p_da'][d]    = dam_res.p.T      # (24, N)
        arr['da_iters'][d] = getattr(dam_res, 'n_iter', 1)
        arr['da_conv'][d]  = getattr(dam_res, 'converged', True)

        # ── RTM receding horizon ──────────────────────────────────────────────
        cur_soc = np.array([ag.bat_soc_kwh for ag in agents])
        arr['bat_soc'][d, 0] = cur_soc.copy()
        h2_acc = np.zeros(N)

        # Helper: commit u for hour h using 65-min-ahead forecast + current SoC
        def _commit(h: int, soc: np.ndarray) -> None:
            if h < 0 or h >= 24:
                return
            g_c   = np.zeros((N, H_RT))
            lam_c = np.zeros(H_RT)
            for i, ag in enumerate(agents):
                fc = rt_forecast(12 * h, week, d, H_RT,
                                 ag.renewable_type, ag.renewable_cap_kw, crng)
                g_c[i] = fc['ren']
                if i == 0:
                    lam_c = fc['lam']
            if scenario == 'gne':
                arr['u_sched'][d, h] = commit_u_gne(
                    agents, lam_c, g_c, H_RT, DT_RT, soc, l_min, l_max
                )
            else:
                for i, ag in enumerate(agents):
                    arr['u_sched'][d, h, i] = commit_u_individual(
                        ag, lam_c, g_c[i], H_RT, DT_RT, float(soc[i]), l_min
                    )

        # Pre-commit hours 0 and 1 (no 65-min lead available at day start)
        _commit(0, cur_soc)
        _commit(1, cur_soc)

        for t in range(T):
            hour    = t // 12
            p_da_t  = arr['p_da'][d, hour]       # (N,) DA anchor this hour
            u_now   = arr['u_sched'][d, hour]     # (N,) participation this hour

            # 65-min-ahead gate: at last step of hour h, commit hour h+2
            if t % 12 == 11:
                _commit(hour + 2, cur_soc)

            # Generate RT forecasts (shared price, per-agent renewable)
            g_fc  = np.zeros((N, H_RT))
            lam_t = np.zeros(H_RT)
            for i, ag in enumerate(agents):
                fc = rt_forecast(t, week, d, H_RT,
                                 ag.renewable_type, ag.renewable_cap_kw, rng)
                g_fc[i] = fc['ren']
                if i == 0:
                    lam_t = fc['lam']

            # Gate DA anchor: if out of market this hour, anchor to 0
            p_da_eff = p_da_t * u_now   # (N,)

            # RTM MIQP dispatch (u locked, δ free every step)
            if scenario == 'gne':
                rtm = solve_gne(
                    agents, lam_t, p_da_eff, g_fc,
                    H_RT, DT_RT, cur_soc,
                    rho=rho_rt, max_iter=300, tol=tol_rt,
                    l_min=l_min * int(u_now.any()),   # drop coupling if all opt-out
                    l_max=l_max,
                    u_fixed=u_now, mode='rtm',
                )
            else:  # individual
                rtm = solve_individual(
                    agents, lam_t, p_da_eff, g_fc,
                    H_RT, DT_RT, cur_soc,
                    l_min=l_min, u_fixed=u_now, mode='rtm',
                )

            arr['rt_iters'][d, t] = getattr(rtm, 'n_iter', 1)
            arr['rt_conv'][d, t]  = getattr(rtm, 'converged', True)

            # Execute step 0 with true renewable + safety clamps
            g_true = day['ren'][:, t]
            exec_l_min = l_min if scenario == 'individual' else 0.0
            p_exec, pch_exec, pdis_exec, curtail_t = _execute_step(
                agents, rtm.p[:, 0], rtm.p_ch[:, 0], rtm.p_dis[:, 0],
                g_true, cur_soc, u_now, exec_l_min
            )

            # H₂ production and SoC update
            g_eff    = g_true - curtail_t
            p_elec_t = p_exec + g_eff + pdis_exec - pch_exec
            eta_arr  = np.array([ag.eta_kg_per_kwh for ag in agents])
            h2_s     = np.maximum(p_elec_t, 0.0) * eta_arr * DT_RT
            h2_acc  += h2_s

            for i, ag in enumerate(agents):
                if ag.has_battery:
                    cur_soc[i] = np.clip(
                        cur_soc[i] + ag.bat_eta_ch * pch_exec[i] * DT_RT
                                   - pdis_exec[i] / ag.bat_eta_dis * DT_RT,
                        0.0, ag.bat_cap_kwh
                    )

            # Store
            arr['p_net'][d, t]    = p_exec
            arr['p_elec'][d, t]   = p_elec_t
            arr['p_ch'][d, t]     = pch_exec
            arr['p_dis'][d, t]    = pdis_exec
            arr['bat_soc'][d, t+1]= cur_soc.copy()
            arr['ren'][d, t]      = g_true
            arr['h2_step'][d, t]  = h2_s
            arr['h2_cum'][d, t]   = h2_acc.copy()
            arr['cost_rt'][d, t]  = (day['rt_price'][t] / 1000.0 * p_exec * DT_RT
                                     + day['rt_price'][t] / 1000.0 * curtail_t * DT_RT)
            arr['curtail'][d, t]  = curtail_t
            arr['agg'][d, t]      = p_exec.sum()
            arr['lmin_b'][d, t]   = arr['agg'][d, t] <= l_min + 1.0
            arr['lmax_b'][d, t]   = arr['agg'][d, t] >= l_max - 1.0

        arr['h2_daily'][d] = h2_acc
        carry_soc          = cur_soc.copy()

        tgts = arr['h2_target'][d]
        pcts = [f"{arr['h2_daily'][d,i]/tgts[i]*100:.0f}%" if tgts[i] > 0 else "N/A"
                for i in range(N)]
        names = [ag.name for ag in agents]
        if scenario == 'gne':
            u_rate = arr['u_sched'][d].mean(axis=0)
            print(
                f"Day {d+1}/{n_days}:  GNE  "
                f"DAM {arr['da_iters'][d]:3d}it ({'OK' if arr['da_conv'][d] else 'NO'})  |  "
                f"RTM conv {arr['rt_conv'][d].mean()*100:.0f}%  "
                f"mean-iter {arr['rt_iters'][d].mean():.1f}  |  "
                f"u-rate {' '.join(f'{n}={r:.2f}' for n,r in zip(names, u_rate))}  |  "
                f"H2 {' '.join(f'{n}={p}' for n,p in zip(names, pcts))}"
            )
        else:
            u_rate = arr['u_sched'][d].mean(axis=0)
            print(
                f"Day {d+1}/{n_days}:  Individual  "
                f"u-rate {' '.join(f'{n}={r:.2f}' for n,r in zip(names, u_rate))}  |  "
                f"H2 {' '.join(f'{n}={p}' for n,p in zip(names, pcts))}"
            )

    res = SimResults(
        p_net=arr['p_net'],   p_elec=arr['p_elec'],
        p_ch=arr['p_ch'],     p_dis=arr['p_dis'],
        bat_soc=arr['bat_soc'], ren=arr['ren'],
        h2_step=arr['h2_step'], h2_cum=arr['h2_cum'],
        h2_daily=arr['h2_daily'], h2_target=arr['h2_target'],
        lmp_rt=arr['lmp_rt'],   lmp_da=arr['lmp_da'],
        cost_rt=arr['cost_rt'], p_da=arr['p_da'],
        da_iters=arr['da_iters'], da_conv=arr['da_conv'],
        rt_iters=arr['rt_iters'], rt_conv=arr['rt_conv'],
        agg=arr['agg'],
        lmin_bind=arr['lmin_b'], lmax_bind=arr['lmax_b'],
        curtail=arr['curtail'],  participation=arr['u_sched'],
    )
    return res, agents


# ── Public entry points ────────────────────────────────────────────────────────
def run_gne(
    agents:     list[ElzAgent],
    n_days:     int         = 7,
    start_date: str         = '2024-07-08',
    seed:       int         = 42,
    rho:        float       = 0.05,
    rho_da:     float       = 1e-4,
    tol_rt:     float       = 1.0,
    tol_da:     float       = 5.0,
    l_min:      float       = 100.0,
    l_max:      float       = 200.0,
    h2_profile: list | None = None,
) -> tuple[SimResults, list[ElzAgent]]:
    """Run GNE coalition simulation (MIQP-ADMM, Fabiani et al. 2022)."""
    if h2_profile is None:
        h2_profile = [1.00, 1.00, 1.00, 1.00, 0.90, 0.65, 0.45]
    return _run('gne', agents, n_days, start_date, seed,
                rho, rho_da, tol_rt, tol_da, l_min, l_max, h2_profile)


def run_individual(
    agents:     list[ElzAgent],
    n_days:     int         = 7,
    start_date: str         = '2024-07-08',
    seed:       int         = 42,
    l_min:      float       = 100.0,
    h2_profile: list | None = None,
) -> tuple[SimResults, list[ElzAgent]]:
    """Run individual market participation simulation (N independent MIQPs)."""
    if h2_profile is None:
        h2_profile = [1.00, 1.00, 1.00, 1.00, 0.90, 0.65, 0.45]
    return _run('individual', agents, n_days, start_date, seed,
                0.0, 0.0, 1.0, 5.0, l_min, float('inf'), h2_profile)
