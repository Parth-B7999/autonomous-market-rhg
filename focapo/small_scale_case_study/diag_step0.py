"""
diag_step0.py
Focused check: for every RTM call, print step-0 pch and pdis for battery agents
BEFORE any post-processing.  Uses threshold = 0 (any positive value counts).
Also runs the full simulation.py run() and checks the stored (post-processed) arrays.

Usage: python diag_step0.py
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

proj = Path(__file__).parent
sys.path.insert(0, str(proj))

from agents import (build_fleet, DT_RT, DT_DA, H_RT, H_DA, STEPS_PER_DAY)
from forecasting import load_pjm_week, dam_forecast, rt_forecast
import admm as admm_mod
from simulation import run

N_DAYS     = 7
START_DATE = '2024-07-08'
SEED       = 42
RHO_RT     = 0.05
RHO_DA     = 1e-4
L_MIN      = 100.0
L_MAX      = 200.0
H2_PROFILE = [1.00, 1.00, 1.00, 1.00, 0.90, 0.65, 0.45]
TINY       = 1e-6   # treat anything below this as zero (numerical noise)

# ── Part 1: raw ADMM step-0 before any post-processing ────────────────────────

print("=" * 70)
print("PART 1 — Raw ADMM step-0 output (before simulation.py post-processing)")
print("=" * 70)

agents  = build_fleet()
N       = len(agents)
rng     = np.random.default_rng(SEED)
h2_base = [0.75 * ag.p_elec_max_kw * ag.eta_kg_per_kwh * 24 for ag in agents]

bat_names  = {i: ag.name for i, ag in enumerate(agents) if ag.has_battery}
bat_idxs   = list(bat_names.keys())

week = load_pjm_week(START_DATE, N_DAYS)
carry_soc = np.array([ag.bat_soc_init_frac * ag.bat_cap_kwh for ag in agents])

# Per-agent tallies
step0_events  = {i: [] for i in bat_idxs}   # list of (day, t, pch, pdis)
step0_max_min = {i: 0.0 for i in bat_idxs}  # max of min(pch,pdis) over all events

for d in range(N_DAYS):
    for i, ag in enumerate(agents):
        ag.h2_inv_kg        = 0.0
        ag.bat_soc_kwh      = float(carry_soc[i])
        ag.h2_target_kg_day = h2_base[i] * H2_PROFILE[d % len(H2_PROFILE)]

    # DAM
    dam_fc      = dam_forecast(week, d, agents, rng)
    bat_soc_dam = np.array([ag.bat_soc_kwh for ag in agents])
    dam_res = admm_mod.solve(
        agents=agents, lam=dam_fc['da_price'], p_da=np.zeros(N),
        g_ren=dam_fc['ren'], H=H_DA, dt=DT_DA, bat_soc=bat_soc_dam,
        rho=RHO_DA, max_iter=2000, tol=5.0, mode='dam',
        l_min=L_MIN, l_max=L_MAX,
    )

    # Check DAM — all 24 hourly slots
    print(f"\nDay {d+1} DAM (all 24 hours):")
    for i in bat_idxs:
        simul_hrs = []
        for h in range(H_DA):
            pc = dam_res.p_ch[i, h]
            pd = dam_res.p_dis[i, h]
            if pc > TINY and pd > TINY:
                simul_hrs.append((h, pc, pd))
        if simul_hrs:
            for h, pc, pd in simul_hrs:
                print(f"  {agents[i].name:<16}  hr={h:02d}  pch={pc:.6f}  pdis={pd:.6f}  "
                      f"min={min(pc,pd):.6f}  *** SIMUL ***")
        else:
            # summarise cleanly — just show max values
            max_pc = dam_res.p_ch[i].max()
            max_pd = dam_res.p_dis[i].max()
            print(f"  {agents[i].name:<16}  no simultaneous  "
                  f"(max pch={max_pc:.4f}  max pdis={max_pd:.4f})")

    # RTM loop
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
            agents=agents, lam=lam_t, p_da=p_da_t, g_ren=g_fc,
            H=H_RT, dt=DT_RT, bat_soc=cur_soc,
            rho=RHO_RT, max_iter=300, tol=1.0, mode='rtm',
            warm=warm, l_min=L_MIN, l_max=L_MAX,
        )

        # Step-0 raw values BEFORE any clipping or post-processing
        for i in bat_idxs:
            pc = rtm.p_ch[i, 0]
            pd = rtm.p_dis[i, 0]
            if pc > TINY and pd > TINY:
                step0_events[i].append((d+1, t, float(pc), float(pd)))
                step0_max_min[i] = max(step0_max_min[i], min(pc, pd))

        # Minimal SoC update
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

print(f"\n{'─'*70}")
print("RTM step-0 simultaneous events (threshold = {:.0e} kW):".format(TINY))
print(f"{'─'*70}")
for i in bat_idxs:
    evts = step0_events[i]
    print(f"\n  {agents[i].name}:  {len(evts)} event(s)  "
          f"(max min(pch,pdis) = {step0_max_min[i]:.6f} kW)")
    if evts:
        # Sort by min(pch, pdis) descending so worst cases appear first
        evts_sorted = sorted(evts, key=lambda e: min(e[2], e[3]), reverse=True)
        noise  = [e for e in evts_sorted if min(e[2], e[3]) < 1e-3]
        real   = [e for e in evts_sorted if min(e[2], e[3]) >= 1e-3]

        if real:
            print(f"    ── GENUINE (min ≥ 0.001 kW): {len(real)} event(s) ──")
            print(f"    {'Day':>5}  {'Step t':>7}  {'Time':>8}  {'pch (kW)':>12}  {'pdis (kW)':>12}  {'min':>10}")
            for day, t, pc, pd in real[:20]:
                hh = (t * 5) // 60
                mm = (t * 5) % 60
                print(f"    {day:>5}  {t:>7}  {hh:02d}:{mm:02d}    {pc:>12.4f}  {pd:>12.4f}  {min(pc,pd):>10.4f}")
            if len(real) > 20:
                print(f"    ... ({len(real)-20} more genuine not shown)")

        print(f"    ── NOISE   (min < 0.001 kW): {len(noise)} event(s), "
              f"max = {max(min(e[2],e[3]) for e in noise):.6f} kW ──")


# ── Part 2: check post-processed simulation output ────────────────────────────

print(f"\n\n{'='*70}")
print("PART 2 — Post-processed simulation output (simulation.run() stored arrays)")
print("=" * 70)

fleet2 = build_fleet()
res, agents2 = run(
    agents      = fleet2,
    n_days      = N_DAYS,
    start_date  = START_DATE,
    seed        = SEED,
    rho         = RHO_RT,
    rho_da      = RHO_DA,
    tol_rt      = 1.0,
    tol_da      = 5.0,
    l_min       = L_MIN,
    l_max       = L_MAX,
    h2_profile  = H2_PROFILE,
)

print(f"\nChecking res.p_ch and res.p_dis (shape {res.p_ch.shape}) ...")
bat_idxs2 = [i for i, ag in enumerate(agents2) if ag.has_battery]

post_events = {i: [] for i in bat_idxs2}
for d in range(N_DAYS):
    for t in range(STEPS_PER_DAY):
        for i in bat_idxs2:
            pc = res.p_ch[d, t, i]
            pd = res.p_dis[d, t, i]
            if pc > TINY and pd > TINY:
                post_events[i].append((d+1, t, float(pc), float(pd)))

print(f"\nPost-processed simultaneous events (both p_ch > {TINY:.0e} AND p_dis > {TINY:.0e}):")
for i in bat_idxs2:
    evts = post_events[i]
    print(f"\n  {agents2[i].name}:  {len(evts)} event(s)")
    if evts:
        print(f"    {'Day':>5}  {'Step t':>7}  {'Time':>8}  {'p_ch':>10}  {'p_dis':>10}  {'min':>10}")
        for day, t, pc, pd in evts[:30]:
            hh = (t * 5) // 60
            mm = (t * 5) % 60
            print(f"    {day:>5}  {t:>7}  {hh:02d}:{mm:02d}    {pc:>10.6f}  {pd:>10.6f}  {min(pc,pd):>10.6f}")
        if len(evts) > 30:
            print(f"    ... ({len(evts)-30} more)")

print("\nDone.")
