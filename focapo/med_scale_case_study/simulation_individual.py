"""
simulation_individual.py — Individual market participation simulation.

Each agent participates alone in DAM + RTM with p_i ≥ L_min.
No coupling with other agents.  No ADMM.

Same closed-loop receding-horizon structure as simulation.py, but each
agent's DAM and RTM are solved independently via Gurobi QP (individual_solver).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import sys
from pathlib import Path
_PARENT = str(Path(__file__).resolve().parent.parent / 'admm_elz_market')
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PARENT not in sys.path:
    sys.path.append(_PARENT)  # parent is fallback only (avoid shadowing med-local modules)

from agents import ElzAgent, DT_RT, DT_DA, H_RT, H_DA, STEPS_PER_DAY  # noqa: E402
from forecasting import load_pjm_week, get_day, dam_forecast, rt_forecast  # noqa: E402
from simulation import SimResults  # noqa: E402  — reuse the same result container
from individual_solver import (solve_individual, solve_individual_miqp,  # noqa: E402
                               commit_individual)


def run_individual(
    agents:      list[ElzAgent],
    n_days:      int         = 7,
    start_date:  str         = '2024-07-08',
    seed:        int         = 42,
    l_min:       float       = 100.0,
    h2_profile:  list | None = None,
) -> tuple[SimResults, list[ElzAgent]]:
    """
    Run individual market participation simulation.

    Each agent solves its own QP with p_i ≥ l_min (no coupling).
    Forecasts are generated with the same RNG sequence as the GNE
    simulation for a fair comparison.
    """
    if h2_profile is None:
        h2_profile = [1.00, 1.00, 1.00, 1.00, 0.90, 0.65, 0.45]

    N, T = len(agents), STEPS_PER_DAY
    rng  = np.random.default_rng(seed)
    # Separate RNG for hour-ahead commitment forecasts so the DA/RT forecast
    # streams (and GNE-vs-individual parity) are unaffected.
    commit_rng = np.random.default_rng(seed + 1000)

    print(f"Loading PJM data from {start_date} ({n_days} days)...")
    week = load_pjm_week(start_date, n_days)

    # Hour-ahead participation commitment u∈{0,1}: (n_days, 24 hours, N agents)
    u_sched = np.zeros((n_days, 24, N), dtype=int)

    # Allocate output arrays (same shape as simulation.py)
    p_net       = np.zeros((n_days, T, N))
    p_elec      = np.zeros((n_days, T, N))
    p_ch_arr    = np.zeros((n_days, T, N))
    p_dis_arr   = np.zeros((n_days, T, N))
    bat_soc     = np.zeros((n_days, T + 1, N))
    ren_arr     = np.zeros((n_days, T, N))
    h2_step     = np.zeros((n_days, T, N))
    h2_cum      = np.zeros((n_days, T, N))
    h2_daily    = np.zeros((n_days, N))
    h2_target   = np.zeros((n_days, N))
    lmp_rt      = np.zeros((n_days, T))
    lmp_da      = np.zeros((n_days, 24))
    cost_rt     = np.zeros((n_days, T, N))
    p_da_arr    = np.zeros((n_days, 24, N))
    da_iters    = np.zeros(n_days, dtype=int)       # not used (no ADMM), set to 1
    da_conv     = np.ones(n_days, dtype=bool)
    rt_iters    = np.ones((n_days, T), dtype=int)   # always 1 (direct solve)
    rt_conv     = np.ones((n_days, T), dtype=bool)
    agg         = np.zeros((n_days, T))
    lmin_b      = np.zeros((n_days, T), dtype=bool)
    lmax_b      = np.zeros((n_days, T), dtype=bool)  # never binds (no L_max)
    curtail_arr = np.zeros((n_days, T, N))

    # Base H₂ target: 75% of maximum possible daily production
    h2_base = [0.75 * ag.p_elec_max_kw * ag.eta_kg_per_kwh * 24 for ag in agents]

    # Battery SoC carried across days
    carry_soc = np.array([ag.bat_soc_init_frac * ag.bat_cap_kwh for ag in agents])

    for d in range(n_days):
        # ── Daily pre-processing ──────────────────────────────────────────────
        for i, ag in enumerate(agents):
            ag.h2_inv_kg   = 0.0
            ag.bat_soc_kwh = float(carry_soc[i])

        profile_d = h2_profile[d % len(h2_profile)]
        for i, ag in enumerate(agents):
            ag.h2_target_kg_day = h2_base[i] * profile_d

        day = get_day(week, d, agents)
        lmp_rt[d] = day['rt_price']
        lmp_da[d] = day['da_price']
        h2_target[d] = [ag.h2_target_kg_day for ag in agents]

        # ── DAM: each agent independently ─────────────────────────────────────
        # Generate DA forecast for all agents (same RNG sequence as GNE)
        dam_fc = dam_forecast(week, d, agents, rng)

        for i, ag in enumerate(agents):
            E_0_i = float(carry_soc[i])

            # Solve individual DAM (pure LP, gamma=0)
            dam_res = solve_individual(
                ag, dam_fc['da_price'], 0.0, dam_fc['ren'][i],
                H_DA, DT_DA, E_0_i, l_min, mode='dam'
            )
            # MIQP cleanup for battery agents
            if ag.has_battery:
                miqp_res = solve_individual_miqp(
                    ag, dam_fc['da_price'], 0.0, dam_fc['ren'][i],
                    H_DA, DT_DA, E_0_i, l_min, mode='dam'
                )
                if miqp_res is not None:
                    dam_res = miqp_res

            p_da_arr[d, :, i] = dam_res.p

        da_iters[d] = 1  # no ADMM iterations

        # ── RTM: receding horizon, each agent independently ───────────────────
        cur_soc = np.array([ag.bat_soc_kwh for ag in agents])
        bat_soc[d, 0] = cur_soc.copy()
        h2_acc = np.zeros(N)

        # Hour-ahead participation commitment (PJM 65-min gate).
        # u for hour h is committed 13 RT steps (65 min) before hour h starts,
        # using an hour-ahead forecast + the battery SoC available at that time,
        # then held fixed across the twelve 5-min dispatch steps of the hour.
        def _commit_hour(h: int, soc_est: np.ndarray) -> None:
            if h < 0 or h >= 24:
                return
            lam_c = None
            for i, ag in enumerate(agents):
                fc = rt_forecast(12 * h, week, d, H_RT,
                                 ag.renewable_type, ag.renewable_cap_kw, commit_rng)
                if i == 0:
                    lam_c = fc['lam']
                u_sched[d, h, i] = commit_individual(
                    ag, lam_c, fc['ren'], H_RT, DT_RT,
                    float(soc_est[i]), p_da=0.0, l_min=l_min
                )

        # Pre-commit hours 0 and 1 (no 65-min lead available at day start).
        _commit_hour(0, cur_soc)
        _commit_hour(1, cur_soc)

        for t in range(T):
            hour   = t // 12
            p_da_t = p_da_arr[d, hour]  # (N,) DA anchor
            u_now  = u_sched[d, hour]   # (N,) participation for this hour

            # 65-min-ahead commitment: at the last step of hour h, lock hour h+2.
            if t % 12 == 11:
                _commit_hour(hour + 2, cur_soc)

            # Generate RT forecasts for all agents (same RNG sequence as GNE)
            g_fc  = np.zeros((N, H_RT))
            lam_t = np.zeros(H_RT)
            for i, ag in enumerate(agents):
                fc = rt_forecast(t, week, d, H_RT,
                                 ag.renewable_type, ag.renewable_cap_kw, rng)
                g_fc[i] = fc['ren']
                if i == 0:
                    lam_t = fc['lam']

            # Solve RTM for each agent independently
            p_i   = np.zeros(N)
            pch_i = np.zeros(N)
            pdi_i = np.zeros(N)

            for i, ag in enumerate(agents):
                u_i = int(u_now[i])
                # Gate the DA anchor: if out of market this hour, anchor to 0.
                p_da_eff = float(p_da_t[i]) if u_i == 1 else 0.0
                rtm_res = solve_individual(
                    ag, lam_t, p_da_eff, g_fc[i],
                    H_RT, DT_RT, float(cur_soc[i]), l_min, mode='rtm', u=u_i
                )
                # MIQP cleanup
                if ag.has_battery:
                    miqp_res = solve_individual_miqp(
                        ag, lam_t, p_da_eff, g_fc[i],
                        H_RT, DT_RT, float(cur_soc[i]), l_min, mode='rtm', u=u_i
                    )
                    if miqp_res is not None:
                        rtm_res = miqp_res

                p_i[i]   = rtm_res.p[0]
                pch_i[i] = rtm_res.p_ch[0]
                pdi_i[i] = rtm_res.p_dis[0]

            # ── Execute using TRUE renewable (local controller) ───────────────
            g_i       = day['ren'][:, t]
            curtail_t = np.zeros(N)

            for i, ag in enumerate(agents):
                u_i = int(u_now[i])
                # Participation floor only applies when committed in-market.
                p_floor = float(l_min) if u_i == 1 else 0.0
                if ag.has_battery:
                    p_i[i]   = 0.0 if u_i == 0 else max(p_floor, p_i[i])
                    pch_i[i] = np.clip(pch_i[i], 0.0, ag.bat_p_ch_max_kw)
                    pdi_i[i] = np.clip(pdi_i[i], 0.0,
                                       min(ag.bat_p_dis_max_kw,
                                           cur_soc[i] / (DT_RT / ag.bat_eta_dis + 1e-9)))

                    # Overflow: ADMM/individual used forecast; true g may differ
                    excess = p_i[i] + g_i[i] + pdi_i[i] - pch_i[i] - ag.p_elec_max_kw
                    if excess > 1e-6:
                        dis_cut    = min(excess, pdi_i[i])
                        pdi_i[i]  -= dis_cut
                        excess     -= dis_cut
                    if excess > 1e-6:
                        ch_headroom = min(
                            ag.bat_p_ch_max_kw - pch_i[i],
                            max(0.0, (ag.bat_cap_kwh - cur_soc[i])
                                     / (DT_RT * ag.bat_eta_ch + 1e-9))
                        )
                        absorbed   = min(excess, ch_headroom)
                        pch_i[i]  += absorbed
                        excess     -= absorbed
                    curtail_t[i] = max(0.0, excess)
                else:
                    # Non-battery: clip to feasible range, handle curtailment
                    p_i[i]   = 0.0 if u_i == 0 else max(p_floor, p_i[i])
                    pch_i[i] = 0.0
                    pdi_i[i] = 0.0

                    # If g_true + p causes electrolyzer overflow, curtail renewable
                    p_elec_check = p_i[i] + g_i[i] - ag.p_elec_max_kw
                    if p_elec_check > 1e-6:
                        curtail_t[i] = p_elec_check

            # ── State update ──────────────────────────────────────────────────
            g_eff    = g_i - curtail_t
            p_elec_t = p_i + g_eff + pdi_i - pch_i
            h2_s     = (np.maximum(p_elec_t, 0.0)
                        * np.array([ag.eta_kg_per_kwh for ag in agents])
                        * DT_RT)
            h2_acc  += h2_s

            # Battery SoC update
            for i, ag in enumerate(agents):
                if ag.has_battery:
                    cur_soc[i] = np.clip(
                        cur_soc[i] + ag.bat_eta_ch * pch_i[i] * DT_RT
                                   - pdi_i[i] / ag.bat_eta_dis * DT_RT,
                        0.0, ag.bat_cap_kwh
                    )

            # Store
            p_net[d, t]       = p_i
            p_elec[d, t]      = p_elec_t
            p_ch_arr[d, t]    = pch_i
            p_dis_arr[d, t]   = pdi_i
            bat_soc[d, t + 1] = cur_soc.copy()
            ren_arr[d, t]     = g_i
            h2_step[d, t]     = h2_s
            h2_cum[d, t]      = h2_acc.copy()
            cost_rt[d, t]     = (day['rt_price'][t] / 1000.0 * p_i * DT_RT
                                 + day['rt_price'][t] / 1000.0 * curtail_t * DT_RT)
            curtail_arr[d, t] = curtail_t
            agg[d, t]         = p_i.sum()
            lmin_b[d, t]      = False  # individual: L_min per agent, not aggregate
            lmax_b[d, t]      = False  # no L_max

        h2_daily[d] = h2_acc
        carry_soc   = cur_soc.copy()

        tgts = h2_target[d]
        pcts = [f"{h2_daily[d,i]/tgts[i]*100:.0f}%" if tgts[i] > 0 else "N/A"
                for i in range(N)]
        print(
            f"Day {d+1}/{n_days}:  "
            f"Individual (no ADMM)  |  "
            f"H2 {' '.join(f'{ag.name}={p}' for ag, p in zip(agents, pcts))}"
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
