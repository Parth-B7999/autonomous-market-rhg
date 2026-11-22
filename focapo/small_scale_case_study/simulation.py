"""
simulation.py — Closed-loop two-timescale simulation using real PJM data.

Pipeline per day
----------------
1. DAM ADMM (runs once, 24-hour horizon)
     Input : noisy 24-hour DA forecast (prices + hourly renewables)
             Battery agents: initial SoC (50% of capacity, reset each day)
     Output: p_DA_i  (24-vector per agent)

2. RTM ADMM (every 5 min, H=12 step horizon = 1 hour)
     Input : noisy 12-step RT forecast, DA anchor, current battery SoC
     Output: first-step action executed; battery SoC updated

True executed values always come from real PJM data.
Battery SoC is updated from executed p_ch[0] and p_dis[0] each step.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import admm
from agents import (ElzAgent, build_fleet,
                    L_MIN_KW, L_MAX_KW, DT_RT, DT_DA, H_RT, H_DA, STEPS_PER_DAY)
from forecasting import load_pjm_week, get_day, dam_forecast, rt_forecast


@dataclass
class SimResults:
    """All time-series outputs.  D=n_days, T=288, N=6."""
    p_net:    np.ndarray   # (D, T, N) net grid import [kW]
    p_elec:   np.ndarray   # (D, T, N) electrolyzer power [kW]
    p_ch:     np.ndarray   # (D, T, N) battery charging [kW]
    p_dis:    np.ndarray   # (D, T, N) battery discharging [kW]
    bat_soc:  np.ndarray   # (D, T+1, N) battery SoC [kWh]
    ren:      np.ndarray   # (D, T, N) true renewable [kW]
    h2_step:  np.ndarray   # (D, T, N) H2 produced per step [kg]
    h2_cum:   np.ndarray   # (D, T, N) cumulative H2 within day [kg]
    h2_daily: np.ndarray   # (D, N)   total H2 per day [kg]
    h2_target:np.ndarray   # (D, N)   daily H2 target (varies each day) [kg]
    lmp_rt:   np.ndarray   # (D, T)   true RT LMP [$/MWh]
    lmp_da:   np.ndarray   # (D, 24)  true DA LMP [$/MWh]
    cost_rt:  np.ndarray   # (D, T, N) RT energy cost per step [$]
    p_da:     np.ndarray   # (D, 24, N) DA schedule [kW]
    da_iters: np.ndarray   # (D,)  DAM ADMM iterations
    da_conv:  np.ndarray   # (D,)  bool
    rt_iters: np.ndarray   # (D, T)
    rt_conv:  np.ndarray   # (D, T)
    agg:      np.ndarray   # (D, T) aggregate net import [kW]
    lmin_bind:np.ndarray   # (D, T) bool
    lmax_bind:np.ndarray   # (D, T) bool
    curtail:  np.ndarray   # (D, T, N) curtailed renewable per step [kW]
    participation: np.ndarray = None  # (D, 24, N) hour-ahead participation commitment u∈{0,1}


def run(
    n_days:      int         = 7,
    start_date:  str         = '2024-07-08',
    seed:        int         = 42,
    rho:         float       = 0.05,
    rho_da:      float       = 1e-4,
    tol_rt:      float       = 1.0,
    tol_da:      float       = 5.0,
    agents:      list | None = None,
    l_min:       float       = L_MIN_KW,
    l_max:       float       = L_MAX_KW,
    h2_profile:  list | None = None,
) -> tuple[SimResults, list[ElzAgent]]:
    """Run closed-loop simulation.

    agents     : custom fleet list; if None, uses build_fleet().
    l_min/l_max: PCC coupling bounds [kW]; default from agents module.
    rho_da     : ADMM penalty for DAM (smaller than RTM rho).
    h2_profile : 7-element day-of-week demand multipliers (Mon…Sun).
    """
    if agents is None:
        agents = build_fleet()
    if h2_profile is None:
        h2_profile = [1.00, 1.00, 1.00, 1.00, 0.90, 0.65, 0.45]
    N, T   = len(agents), STEPS_PER_DAY
    rng    = np.random.default_rng(seed)
    # Separate RNG for hour-ahead commitment forecasts (keeps DA/RT streams and
    # GNE-vs-individual parity intact).  Same seed offset as the individual sim.
    commit_rng = np.random.default_rng(seed + 1000)

    print(f"Loading PJM data from {start_date} ({n_days} days)...")
    week = load_pjm_week(start_date, n_days)

    # Hour-ahead participation commitment u∈{0,1}: (n_days, 24 hours, N agents)
    u_sched = np.zeros((n_days, 24, N), dtype=int)

    # Allocate output arrays
    p_net    = np.zeros((n_days, T, N))
    p_elec   = np.zeros((n_days, T, N))
    p_ch_arr = np.zeros((n_days, T, N))
    p_dis_arr= np.zeros((n_days, T, N))
    bat_soc  = np.zeros((n_days, T + 1, N))
    ren_arr  = np.zeros((n_days, T, N))
    h2_step  = np.zeros((n_days, T, N))
    h2_cum   = np.zeros((n_days, T, N))
    h2_daily  = np.zeros((n_days, N))
    h2_target = np.zeros((n_days, N))
    lmp_rt   = np.zeros((n_days, T))
    lmp_da   = np.zeros((n_days, 24))
    cost_rt  = np.zeros((n_days, T, N))
    p_da_arr = np.zeros((n_days, 24, N))
    da_iters = np.zeros(n_days, dtype=int)
    da_conv  = np.zeros(n_days, dtype=bool)
    rt_iters = np.zeros((n_days, T), dtype=int)
    rt_conv  = np.zeros((n_days, T), dtype=bool)
    agg      = np.zeros((n_days, T))
    lmin_b      = np.zeros((n_days, T), dtype=bool)
    lmax_b      = np.zeros((n_days, T), dtype=bool)
    curtail_arr = np.zeros((n_days, T, N))

    # Base H2 target: 75% of each agent's maximum possible daily production.
    h2_base = [0.75 * ag.p_elec_max_kw * ag.eta_kg_per_kwh * 24 for ag in agents]

    # Battery SoC carried across days: day 0 starts at init fraction, subsequent
    # days start from the actual SoC at the end of the previous day's RT execution.
    carry_soc = np.array([ag.bat_soc_init_frac * ag.bat_cap_kwh for ag in agents])

    for d in range(n_days):
        for i, ag in enumerate(agents):
            ag.h2_inv_kg   = 0.0
            ag.bat_soc_kwh = float(carry_soc[i])

        # Apply day-specific demand multiplier before the ADMM solve.
        profile_d = h2_profile[d % len(h2_profile)]
        for i, ag in enumerate(agents):
            ag.h2_target_kg_day = h2_base[i] * profile_d

        # True values for this day
        day = get_day(week, d, agents)
        lmp_rt[d] = day['rt_price']
        lmp_da[d] = day['da_price']

        h2_target[d] = [ag.h2_target_kg_day for ag in agents]

        # ── DAM ADMM (once per day) ───────────────────────────────────────────
        dam_fc      = dam_forecast(week, d, agents, rng)
        # DAM is a pure LP (gamma=0 in admm.solve); p_da is multiplied by gamma=0
        # so its value is irrelevant — pass zeros for correctness.
        p_da_anchor = np.zeros(N)
        bat_soc_dam = np.array([ag.bat_soc_kwh for ag in agents])

        # DAM uses a smaller rho so the linear cost (c_k ~ 0.002 $/kW) dominates
        # the ADMM augmentation and the LP reaches the correct optimum.
        dam_res = admm.solve(
            agents   = agents,
            lam      = dam_fc['da_price'],
            p_da     = p_da_anchor,
            g_ren    = dam_fc['ren'],
            H        = H_DA,
            dt       = DT_DA,
            bat_soc  = bat_soc_dam,
            rho      = rho_da,
            max_iter = 2000,
            tol      = tol_da,
            mode     = 'dam',
            l_min    = l_min,
            l_max    = l_max,
        )
        p_da_arr[d] = dam_res.p.T    # (24, N)
        da_iters[d] = dam_res.n_iter
        da_conv[d]  = dam_res.converged

        # ── RTM ADMM (288 steps, receding horizon) ────────────────────────────
        warm   = None
        h2_acc = np.zeros(N)

        # Initial SoC for RTM (battery state at day start, already reset above)
        cur_soc = np.array([ag.bat_soc_kwh for ag in agents])
        bat_soc[d, 0] = cur_soc.copy()

        # Hour-ahead coordinated participation commitment (PJM 65-min gate).
        # u for hour h is committed 13 RT steps (65 min) before hour h starts via
        # a single coordinated MIQP, then held fixed across the hour's 12 steps.
        def _commit_hour(h: int, soc_est: np.ndarray) -> None:
            if h < 0 or h >= 24:
                return
            g_c   = np.zeros((N, H_RT))
            lam_c = np.zeros(H_RT)
            for i, ag in enumerate(agents):
                fc = rt_forecast(12 * h, week, d, H_RT,
                                 ag.renewable_type, ag.renewable_cap_kw, commit_rng)
                g_c[i] = fc['ren']
                if i == 0:
                    lam_c = fc['lam']
            u_sched[d, h] = admm.commit_gne(
                agents, lam_c, g_c, H_RT, DT_RT, soc_est,
                l_min=l_min, l_max=l_max,
            )

        _commit_hour(0, cur_soc)
        _commit_hour(1, cur_soc)

        for t in range(T):
            hour   = t // 12
            p_da_t = p_da_arr[d, hour]    # (N,) DA anchor for this hour
            u_now  = u_sched[d, hour]     # (N,) participation for this hour

            # 65-min-ahead commitment: at the last step of hour h, lock hour h+2.
            if t % 12 == 11:
                _commit_hour(hour + 2, cur_soc)

            # Noisy H-step forecast per agent
            g_fc  = np.zeros((N, H_RT))
            lam_t = np.zeros(H_RT)
            for i, ag in enumerate(agents):
                fc = rt_forecast(t, week, d, H_RT,
                                 ag.renewable_type, ag.renewable_cap_kw, rng)
                g_fc[i] = fc['ren']
                if i == 0:
                    lam_t = fc['lam']    # shared price signal from agent 0

            # Gate the DA anchor by participation: out-of-market agents anchor to 0.
            p_da_eff = p_da_t * u_now

            rtm = admm.solve(
                agents   = agents,
                lam      = lam_t,
                p_da     = p_da_eff,
                g_ren    = g_fc,
                H        = H_RT,
                dt       = DT_RT,
                bat_soc  = cur_soc,
                rho      = rho,
                max_iter = 300,
                tol      = tol_rt,
                mode     = 'rtm',
                warm     = warm,
                l_min    = l_min,
                l_max    = l_max,
                u        = u_now,
            )

            # Execute first step using TRUE renewable (safety clamp)
            p_i   = rtm.p[:, 0].copy()
            pch_i = rtm.p_ch[:, 0].copy()
            pdi_i = rtm.p_dis[:, 0].copy()
            g_i   = day['ren'][:, t]

            curtail_t = np.zeros(N)   # renewable curtailed this step [kW]

            for i, ag in enumerate(agents):
                if ag.has_battery:
                    # p_net is market-settled; only p_ch / p_dis are local control.
                    p_i[i]   = max(0.0, p_i[i])
                    pch_i[i] = np.clip(pch_i[i], 0.0, ag.bat_p_ch_max_kw)
                    pdi_i[i] = np.clip(pdi_i[i], 0.0,
                                       min(ag.bat_p_dis_max_kw,
                                           cur_soc[i] / (DT_RT / ag.bat_eta_dis + 1e-9)))

                    # Overflow check: ADMM used g_hat; true renewable g_true may differ.
                    # The MIQP already guarantees pch*pdis=0 from the ADMM output, so
                    # the steps below can only ever increase one variable at a time.
                    #   Step 1 — reduce discharge to absorb excess (keeps pdis->0 first)
                    #   Step 2 — increase charge only after pdis reaches zero
                    #   Step 3 — curtail whatever the battery cannot absorb
                    excess = p_i[i] + g_i[i] + pdi_i[i] - pch_i[i] - ag.p_elec_max_kw
                    if excess > 1e-6:
                        dis_cut    = min(excess, pdi_i[i])
                        pdi_i[i]  -= dis_cut
                        excess     -= dis_cut
                    if excess > 1e-6:
                        # pdis is now 0; increasing pch cannot create simultaneous operation
                        ch_headroom = min(
                            ag.bat_p_ch_max_kw - pch_i[i],
                            max(0.0, (ag.bat_cap_kwh - cur_soc[i]) / (DT_RT * ag.bat_eta_ch + 1e-9))
                        )
                        absorbed   = min(excess, ch_headroom)
                        pch_i[i]  += absorbed
                        excess     -= absorbed
                    curtail_t[i] = max(0.0, excess)
                else:
                    # Participation gate: out of market ⇒ p=0; in market ⇒ p≥L_min
                    # with renewable curtailment to fit electrolyzer capacity.
                    u_i = int(u_now[i])
                    if u_i == 0:
                        p_i[i] = 0.0
                    else:
                        p_i[i] = float(np.clip(p_i[i], l_min, ag.p_elec_max_kw))
                    pch_i[i] = 0.0
                    pdi_i[i] = 0.0

                    # If g_true + p exceeds electrolyzer capacity, curtail renewable.
                    p_elec_check = p_i[i] + g_i[i] - ag.p_elec_max_kw
                    if p_elec_check > 1e-6:
                        curtail_t[i] = p_elec_check

            # H2 production: p_elec = p_grid + (g_true - curtailed) + p_dis - p_ch
            g_eff    = g_i - curtail_t   # effective renewable after curtailment
            p_elec_t = p_i + g_eff + pdi_i - pch_i
            h2_s     = np.maximum(p_elec_t, 0.0) * np.array([ag.eta_kg_per_kwh for ag in agents]) * DT_RT
            h2_acc  += h2_s

            # Update battery SoC
            for i, ag in enumerate(agents):
                if ag.has_battery:
                    cur_soc[i] = np.clip(
                        cur_soc[i] + ag.bat_eta_ch * pch_i[i] * DT_RT
                                   - pdi_i[i] / ag.bat_eta_dis * DT_RT,
                        0.0, ag.bat_cap_kwh
                    )

            # Store
            p_net[d, t]    = p_i
            p_elec[d, t]   = p_elec_t
            p_ch_arr[d, t] = pch_i
            p_dis_arr[d, t]= pdi_i
            bat_soc[d, t+1]= cur_soc.copy()
            ren_arr[d, t]  = g_i
            h2_step[d, t]  = h2_s
            h2_cum[d, t]   = h2_acc.copy()
            cost_rt[d, t]  = (day['rt_price'][t] / 1000.0 * p_i * DT_RT
                               + day['rt_price'][t] / 1000.0 * curtail_t * DT_RT)
            curtail_arr[d, t] = curtail_t
            agg[d, t]      = p_i.sum()
            lmin_b[d, t]   = agg[d, t] <= l_min + 1.0
            lmax_b[d, t]   = agg[d, t] >= l_max - 1.0
            rt_iters[d, t] = rtm.n_iter
            rt_conv[d, t]  = rtm.converged
            warm            = rtm.p

        h2_daily[d] = h2_acc
        carry_soc   = cur_soc.copy()   # pass end-of-day SoC to next day's DAM
        tgts  = h2_target[d]
        pcts  = [f"{h2_daily[d,i]/tgts[i]*100:.0f}%" if tgts[i] > 0 else "N/A"
                 for i in range(N)]
        print(
            f"Day {d+1}/{n_days}:  "
            f"DAM {da_iters[d]:3d}it ({'OK' if da_conv[d] else 'NO'})  |  "
            f"RTM conv {rt_conv[d].mean()*100:.0f}%  "
            f"mean-iter {rt_iters[d].mean():.1f}  |  "
            f"L_max hits {lmax_b[d].sum():3d}/288  |  "
            f"H2 {' '.join(f'{n}={p}' for n,p in zip([ag.name for ag in agents], pcts))}"
        )

    return SimResults(
        p_net=p_net, p_elec=p_elec, p_ch=p_ch_arr, p_dis=p_dis_arr,
        bat_soc=bat_soc, ren=ren_arr,
        h2_step=h2_step, h2_cum=h2_cum, h2_daily=h2_daily, h2_target=h2_target,
        lmp_rt=lmp_rt, lmp_da=lmp_da, cost_rt=cost_rt,
        p_da=p_da_arr, da_iters=da_iters, da_conv=da_conv,
        rt_iters=rt_iters, rt_conv=rt_conv,
        agg=agg, lmin_bind=lmin_b, lmax_bind=lmax_b,
        curtail=curtail_arr, participation=u_sched,
    ), agents
