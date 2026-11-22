"""
case_study.py — ADMM Green Hydrogen Market: configurable case-study runner.

Run cell-by-cell:  VS Code  →  Ctrl+Enter   |   Spyder  →  F9
Each run auto-creates a timestamped subfolder under results/ and saves:
  • All figures (PDF + PNG)
  • h2_production.csv   — daily H₂ actual vs target per agent
  • financials.csv      — energy cost / H₂ revenue / net profit per agent
  • coupling_stats.csv  — per-day L_min / L_max binding counts & %
  • run_params.txt      — full parameter log for reproducibility
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

from agents import ElzAgent
from simulation import run
import main as _fig          # reuse all figure-drawing functions from main.py

PROJ    = Path(__file__).parent
RESULTS = PROJ / "results"
RESULTS.mkdir(exist_ok=True)


# %% [1]  Market & ADMM parameters
# ─────────────────────────────────────────────────────────────────────────────
L_MIN   = 100.0   # [kW]   coalition floor  — each agent must be < this to force GNE
L_MAX   = 200.0   # [kW]   PCC interconnection capacity

H2_PRICE = 3.0    # [$/kg] hydrogen wholesale price

RHO_RT  = 0.05    # ADMM penalty — real-time market (larger → faster convergence)
RHO_DA  = 1e-4    # ADMM penalty — day-ahead market  (small → LP optimum preserved)
TOL_RT  = 1.0     # [kW]  RTM convergence tolerance
TOL_DA  = 5.0     # [kW]  DAM convergence tolerance

# Weekday/weekend H₂ demand multiplier — one value per day starting from START_DATE
# Default: full Mon–Thu, slight Fri dip, reduced Sat/Sun
H2_DEMAND_PROFILE = [1.00, 1.00, 1.00, 1.00, 0.90, 0.65, 0.45]


# %% [2]  Fleet parameters  (one dict per agent — edit any value freely)
# ─────────────────────────────────────────────────────────────────────────────
# Keys mirror ElzAgent fields exactly.
# renewable_type : 'pv' | 'wind' | None
# bat_cap_kwh    : set to 0.0 to disable battery (bat_* fields ignored)

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
N_DAYS     = 7             # number of simulation days
START_DATE = '2024-07-08'  # first day (must have PJM data available)
SEED       = 42            # random seed for forecast noise

# Auto-generated folder: results/YYYYMMDD_HHMM_Lmin{X}_Lmax{Y}_{N}d/
_ts      = datetime.now().strftime('%Y%m%d_%H%M')
RUN_NAME = f"{_ts}_Lmin{L_MIN:.0f}_Lmax{L_MAX:.0f}_{N_DAYS}d"
RUN_DIR  = RESULTS / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"  Run name : {RUN_NAME}")
print(f"  Output   : {RUN_DIR}")
print(f"  L_min={L_MIN:.0f} kW  |  L_max={L_MAX:.0f} kW  |  {N_DAYS} days  |  seed={SEED}")
print(f"  Agents   : {[d['name'] for d in FLEET_DICTS]}")


# %% [4]  Build fleet & run simulation
# ─────────────────────────────────────────────────────────────────────────────
def _make_agent(d: dict) -> ElzAgent:
    """Build an ElzAgent from a parameter dict.
    h2_target_kg_day is set to 0 here; simulation.run() overrides it
    each day via the H2_DEMAND_PROFILE multiplier.
    """
    return ElzAgent(h2_target_kg_day=0.0, **d)

fleet = [_make_agent(d) for d in FLEET_DICTS]

res, agents = run(
    agents       = fleet,
    n_days       = N_DAYS,
    start_date   = START_DATE,
    seed         = SEED,
    rho          = RHO_RT,
    rho_da       = RHO_DA,
    tol_rt       = TOL_RT,
    tol_da       = TOL_DA,
    l_min        = L_MIN,
    l_max        = L_MAX,
    h2_profile   = H2_DEMAND_PROFILE,
)


# %% [5]  Save all figures to RUN_DIR
# ─────────────────────────────────────────────────────────────────────────────
# Redirect main.py's save target and coupling-bound labels to this run's values
_fig.RESULTS   = RUN_DIR
_fig.L_MIN_KW  = L_MIN
_fig.L_MAX_KW  = L_MAX

from forecasting import load_pjm_week
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
days_hdr  = [f"Day{d+1}" for d in range(N_DAYS)]
DAYS_FULL = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# ── h2_production.csv ────────────────────────────────────────────────────────
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
    # Fleet total row
    tot_act = res.h2_daily.sum(axis=1)
    tot_tgt = res.h2_target.sum(axis=1)
    att_tot = tot_act.mean() / tot_tgt.mean() * 100 if tot_tgt.mean() > 0 else float('nan')
    w.writerow(["TOTAL"]
               + [f"{v:.2f}" for v in tot_act]
               + [f"{v:.2f}" for v in tot_tgt]
               + [f"{tot_act.mean():.2f}", f"{tot_tgt.mean():.2f}", f"{att_tot:.1f}"])

# ── financials.csv ────────────────────────────────────────────────────────────
from agents import DT_RT
with open(RUN_DIR / "financials.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Agent", "energy_cost_$", "h2_revenue_$", "net_profit_$"])
    for i, ag in enumerate(agents):
        ec  = float(res.cost_rt[:, :, i].sum())
        rev = float((res.p_elec[:, :, i] * ag.eta_kg_per_kwh * DT_RT * ag.r_h2).sum())
        w.writerow([ag.name, f"{ec:.2f}", f"{rev:.2f}", f"{rev - ec:.2f}"])
    # Fleet total
    total_ec  = sum(float(res.cost_rt[:, :, i].sum()) for i in range(len(agents)))
    total_rev = sum(
        float((res.p_elec[:, :, i] * agents[i].eta_kg_per_kwh * DT_RT * agents[i].r_h2).sum())
        for i in range(len(agents))
    )
    w.writerow(["TOTAL", f"{total_ec:.2f}", f"{total_rev:.2f}", f"{total_rev - total_ec:.2f}"])

# ── coupling_stats.csv ────────────────────────────────────────────────────────
with open(RUN_DIR / "coupling_stats.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Day", "Date_approx", "Demand_profile",
                "lmax_steps", "lmax_%", "lmin_steps", "lmin_%"])
    from datetime import timedelta
    from datetime import date as _date
    base = _date.fromisoformat(START_DATE)
    for d in range(N_DAYS):
        dt     = base + timedelta(days=d)
        lmax_n = int(res.lmax_bind[d].sum())
        lmin_n = int(res.lmin_bind[d].sum())
        T      = res.lmax_bind.shape[1]
        prof   = H2_DEMAND_PROFILE[d % len(H2_DEMAND_PROFILE)]
        w.writerow([d + 1, dt.isoformat(), f"{prof:.2f}",
                    lmax_n, f"{lmax_n/T*100:.1f}",
                    lmin_n, f"{lmin_n/T*100:.1f}"])
    # Summary row
    tot_lmax = int(res.lmax_bind.sum())
    tot_lmin = int(res.lmin_bind.sum())
    total_T  = N_DAYS * T
    w.writerow(["TOTAL", "", "",
                tot_lmax, f"{tot_lmax/total_T*100:.1f}",
                tot_lmin, f"{tot_lmin/total_T*100:.1f}"])

# ── run_params.txt ────────────────────────────────────────────────────────────
with open(RUN_DIR / "run_params.txt", "w") as f:
    f.write(f"Run: {RUN_NAME}\n")
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

print(f"\nCSVs and parameter log saved to {RUN_DIR}/")
print("Done.")

# %%
