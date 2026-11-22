"""
case_study_gne.py — GNE Coalition (MIQP-ADMM) with small-scale agents.

Same ADMM framework as med_scale_case_study/case_study_gne.py, using the
original 6-agent small-scale fleet (50–50 kW) with L_min=100 kW, L_max=200 kW.

Run cell-by-cell:  VS Code  →  Ctrl+Enter   |   Spyder  →  F9
Each run auto-creates a timestamped subfolder under results/ and saves:
  • All figures (PDF + PNG)
  • h2_production.csv, financials.csv, coupling_stats.csv, run_params.txt
  • results.npz — full NumPy arrays
"""

# %% [0]  Imports & paths
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
import csv

from agents import ElzAgent, DT_RT
import main as _fig
from forecasting import load_pjm_week
from simulation_sc import run_gne

PROJ    = Path(__file__).parent
RESULTS = PROJ / "results"
RESULTS.mkdir(exist_ok=True)


# %% [1]  Market & ADMM parameters
# ─────────────────────────────────────────────────────────────────────────────
L_MIN   = 100.0   # [kW]  coalition aggregate floor
L_MAX   = 200.0   # [kW]  PCC interconnection capacity

H2_PRICE = 3.0    # [$/kg] hydrogen wholesale price

RHO_RT  = 0.05    # ADMM penalty — real-time market
RHO_DA  = 1e-4    # ADMM penalty — day-ahead market
TOL_RT  = 1.0     # [kW]  RTM convergence tolerance
TOL_DA  = 5.0     # [kW]  DAM convergence tolerance

H2_DEMAND_PROFILE = [1.00, 1.00, 1.00, 1.00, 0.90, 0.65, 0.45]


# %% [2]  Fleet parameters  (small-scale, 6 agents)
# ─────────────────────────────────────────────────────────────────────────────
A1 = dict(
    name              = "PEM_Elec",
    renewable_type    = None,
    p_elec_max_kw     = 50.0,
    p_elec_min_kw     = 0.0,
    renewable_cap_kw  = 0.0,
    eta_kg_per_kwh    = 0.020,
    gamma             = 5e-3,
    r_h2              = H2_PRICE,
)

A2 = dict(
    name              = "PEM_PV",
    renewable_type    = "pv",
    p_elec_max_kw     = 25.0,
    p_elec_min_kw     = 0.0,
    renewable_cap_kw  = 25.0,
    eta_kg_per_kwh    = 0.020,
    gamma             = 5e-3,
    r_h2              = H2_PRICE,
)

A3 = dict(
    name              = "PEM_Wind",
    renewable_type    = "wind",
    p_elec_max_kw     = 50.0,
    p_elec_min_kw     = 0.0,
    renewable_cap_kw  = 50.0,
    eta_kg_per_kwh    = 0.020,
    gamma             = 4e-3,
    r_h2              = H2_PRICE,
)

A4 = dict(
    name              = "ALK_Bat",
    renewable_type    = None,
    p_elec_max_kw     = 40.0,
    p_elec_min_kw     = 0.0,
    renewable_cap_kw  = 0.0,
    eta_kg_per_kwh    = 0.018,
    gamma             = 5e-3,
    r_h2              = H2_PRICE,
    bat_cap_kwh       = 30.0,
    bat_p_ch_max_kw   = 7.5,
    bat_p_dis_max_kw  = 7.5,
)

A5 = dict(
    name              = "PEM_PV_Bat",
    renewable_type    = "pv",
    p_elec_max_kw     = 25.0,
    p_elec_min_kw     = 0.0,
    renewable_cap_kw  = 25.0,
    eta_kg_per_kwh    = 0.020,
    gamma             = 5e-3,
    r_h2              = H2_PRICE,
    bat_cap_kwh       = 15.0,
    bat_p_ch_max_kw   = 7.5,
    bat_p_dis_max_kw  = 7.5,
)

A6 = dict(
    name              = "PEM_Wind_Bat",
    renewable_type    = "wind",
    p_elec_max_kw     = 50.0,
    p_elec_min_kw     = 0.0,
    renewable_cap_kw  = 50.0,
    eta_kg_per_kwh    = 0.020,
    gamma             = 4e-3,
    r_h2              = H2_PRICE,
    bat_cap_kwh       = 30.0,
    bat_p_ch_max_kw   = 15.0,
    bat_p_dis_max_kw  = 15.0,
)

FLEET_DICTS = [A1, A2, A3, A4, A5, A6]


# %% [3]  Run settings
# ─────────────────────────────────────────────────────────────────────────────
N_DAYS     = 7
START_DATE = '2024-07-08'
SEED       = 42


# %% [4]  Build fleet & run simulation
# ─────────────────────────────────────────────────────────────────────────────
def _make_agent(d: dict) -> ElzAgent:
    return ElzAgent(h2_target_kg_day=0.0, **d)

fleet = [_make_agent(d) for d in FLEET_DICTS]

_ts      = datetime.now().strftime('%Y%m%d_%H%M')
RUN_NAME = f"gne_{_ts}_Lmin{L_MIN:.0f}_Lmax{L_MAX:.0f}_{N_DAYS}d"
RUN_DIR  = RESULTS / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"  Scenario : GNE Coalition (MIQP-ADMM, semi-continuous bidding)")
print(f"  Run name : {RUN_NAME}")
print(f"  Output   : {RUN_DIR}")
print(f"  L_min={L_MIN:.0f} kW  |  L_max={L_MAX:.0f} kW  |  {N_DAYS} days  |  seed={SEED}")
print(f"  Agents   : {[d['name'] for d in FLEET_DICTS]}")

res, agents = run_gne(
    agents      = fleet,
    n_days      = N_DAYS,
    start_date  = START_DATE,
    seed        = SEED,
    rho         = RHO_RT,
    rho_da      = RHO_DA,
    tol_rt      = TOL_RT,
    tol_da      = TOL_DA,
    l_min       = L_MIN,
    l_max       = L_MAX,
    h2_profile  = H2_DEMAND_PROFILE,
)


# %% [5]  Save all figures to RUN_DIR
# ─────────────────────────────────────────────────────────────────────────────
_fig.RESULTS   = RUN_DIR
_fig.L_MIN_KW  = L_MIN
_fig.L_MAX_KW  = L_MAX

week    = load_pjm_week(START_DATE, N_DAYS)
fc_rng  = np.random.default_rng(SEED + 1)

print("\nGenerating figures ...")
for d in range(N_DAYS):
    _fig.fig_power(res, agents, d)
    _fig.fig_elz_breakdown(res, agents, d)
    _fig.fig_da_vs_rt(res, agents, d)
    _fig.fig_lmp_cost(res, agents, d)
    _fig.fig_bat_soc(res, agents, d)
    _fig.fig_dam_forecast(week, agents, fc_rng, d)
    _fig.fig_rtm_forecast(week, agents, fc_rng, d)

_fig.fig_h2(res, agents)
_fig.fig_admm_conv(res)
_fig.fig_all_days_aggregate(res)
print(f"  All figures → {RUN_DIR}/")


# %% [6]  Export CSV data & parameter log
# ─────────────────────────────────────────────────────────────────────────────
DAYS_FULL = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# ── h2_production.csv
with open(RUN_DIR / "h2_production.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Agent"] + [f"{DAYS_FULL[d % 7]}_actual_kg" for d in range(N_DAYS)]
                           + [f"{DAYS_FULL[d % 7]}_target_kg" for d in range(N_DAYS)]
                           + ["avg_actual_kg", "avg_target_kg", "attainment_%"])
    for i, ag in enumerate(agents):
        actual = res.h2_daily[:, i]
        target = res.h2_target[:, i]
        att    = actual.mean() / target.mean() * 100 if target.mean() > 0 else float('nan')
        w.writerow([ag.name]
                   + [f"{v:.2f}" for v in actual]
                   + [f"{v:.2f}" for v in target]
                   + [f"{actual.mean():.2f}", f"{target.mean():.2f}", f"{att:.1f}"])
    tot_act = res.h2_daily.sum(axis=1)
    tot_tgt = res.h2_target.sum(axis=1)
    att_tot = tot_act.mean() / tot_tgt.mean() * 100 if tot_tgt.mean() > 0 else float('nan')
    w.writerow(["TOTAL"]
               + [f"{v:.2f}" for v in tot_act]
               + [f"{v:.2f}" for v in tot_tgt]
               + [f"{tot_act.mean():.2f}", f"{tot_tgt.mean():.2f}", f"{att_tot:.1f}"])

# ── financials.csv
with open(RUN_DIR / "financials.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Agent", "energy_cost_$", "h2_revenue_$", "net_profit_$"])
    for i, ag in enumerate(agents):
        ec  = float(res.cost_rt[:, :, i].sum())
        rev = float((res.p_elec[:, :, i] * ag.eta_kg_per_kwh * DT_RT * ag.r_h2).sum())
        w.writerow([ag.name, f"{ec:.2f}", f"{rev:.2f}", f"{rev - ec:.2f}"])
    total_ec  = sum(float(res.cost_rt[:, :, i].sum()) for i in range(len(agents)))
    total_rev = sum(
        float((res.p_elec[:, :, i] * agents[i].eta_kg_per_kwh * DT_RT * agents[i].r_h2).sum())
        for i in range(len(agents))
    )
    w.writerow(["TOTAL", f"{total_ec:.2f}", f"{total_rev:.2f}", f"{total_rev - total_ec:.2f}"])

# ── coupling_stats.csv
with open(RUN_DIR / "coupling_stats.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Day", "Date_approx", "Demand_profile",
                "lmax_steps", "lmax_%", "lmin_steps", "lmin_%"])
    from datetime import timedelta, date as _date
    base = _date.fromisoformat(START_DATE)
    for d in range(N_DAYS):
        dt     = base + timedelta(days=d)
        lmax_n = int(res.lmax_bind[d].sum())
        lmin_n = int(res.lmin_bind[d].sum())
        T      = res.lmin_bind.shape[1]
        prof   = H2_DEMAND_PROFILE[d % len(H2_DEMAND_PROFILE)]
        w.writerow([d + 1, dt.isoformat(), f"{prof:.2f}",
                    lmax_n, f"{lmax_n/T*100:.1f}",
                    lmin_n, f"{lmin_n/T*100:.1f}"])
    tot_lmax = int(res.lmax_bind.sum())
    tot_lmin = int(res.lmin_bind.sum())
    total_T  = N_DAYS * T
    w.writerow(["TOTAL", "", "",
                tot_lmax, f"{tot_lmax/total_T*100:.1f}",
                tot_lmin, f"{tot_lmin/total_T*100:.1f}"])

# ── run_params.txt
with open(RUN_DIR / "run_params.txt", "w") as f:
    f.write(f"Run: {RUN_NAME}\n")
    f.write(f"Scenario: GNE Coalition (MIQP-ADMM)\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
    f.write(f"[Market]\n")
    f.write(f"L_MIN={L_MIN} kW\nL_MAX={L_MAX} kW\nH2_PRICE={H2_PRICE} $/kg\n")
    f.write(f"RHO_RT={RHO_RT}\nRHO_DA={RHO_DA}\nTOL_RT={TOL_RT}\nTOL_DA={TOL_DA}\n")
    f.write(f"H2_DEMAND_PROFILE={H2_DEMAND_PROFILE}\n\n")
    f.write(f"[Simulation]\n")
    f.write(f"N_DAYS={N_DAYS}\nSTART_DATE={START_DATE}\nSEED={SEED}\n\n")
    f.write(f"[Fleet]\n")
    for d in FLEET_DICTS:
        f.write(f"  {d['name']}: p_max={d['p_elec_max_kw']} kW, "
                f"eta={d['eta_kg_per_kwh']} kg/kWh, "
                f"ren={d.get('renewable_cap_kw', 0)} kW ({d.get('renewable_type') or 'none'}), "
                f"bat={d.get('bat_cap_kwh', 0)} kWh\n")

# ── results.npz
np.savez_compressed(
    RUN_DIR / "results.npz",
    p_net=res.p_net, p_elec=res.p_elec,
    p_ch=res.p_ch, p_dis=res.p_dis,
    bat_soc=res.bat_soc, ren=res.ren,
    h2_step=res.h2_step, h2_cum=res.h2_cum,
    h2_daily=res.h2_daily, h2_target=res.h2_target,
    lmp_rt=res.lmp_rt, lmp_da=res.lmp_da, cost_rt=res.cost_rt,
    p_da=res.p_da, agg=res.agg,
    lmin_bind=res.lmin_bind, lmax_bind=res.lmax_bind,
    curtail=res.curtail, participation=res.participation,
    agent_names=[ag.name for ag in agents],
)

print(f"\nCSVs, parameter log, and results.npz saved to {RUN_DIR}/")
print("Done — GNE Coalition (small-scale).")

# %%
