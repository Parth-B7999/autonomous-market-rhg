"""
case_study_individual.py — Scenario A: Individual Market Participation.

Each medium-scale agent participates in DAM + RTM ALONE with p_i ≥ L_min.
No coupling between agents.  No ADMM.

Run cell-by-cell:  VS Code  →  Ctrl+Enter   |   Spyder  →  F9
Each run auto-creates a timestamped subfolder under results/ and saves:
  • All figures (PDF + PNG)
  • h2_production.csv, financials.csv, run_params.txt
  • results.npz — full NumPy arrays for compare.py
"""

# %% [0]  Imports & paths
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
import csv, sys

PROJ    = Path(__file__).parent
RESULTS = PROJ / "results"
RESULTS.mkdir(exist_ok=True)

_PARENT = str(PROJ.parent / 'admm_elz_market')
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from agents import ElzAgent, DT_RT
import main as _fig
from forecasting import load_pjm_week

from agents_med import (
    FLEET_DICTS, L_MIN, H2_PRICE,
    N_DAYS, START_DATE, SEED, H2_DEMAND_PROFILE,
)
from simulation_sc import run_individual


# %% [1]  Build fleet & run simulation
# ─────────────────────────────────────────────────────────────────────────────
def _make_agent(d: dict) -> ElzAgent:
    return ElzAgent(h2_target_kg_day=0.0, **d)

fleet = [_make_agent(d) for d in FLEET_DICTS]

_ts      = datetime.now().strftime('%Y%m%d_%H%M')
RUN_NAME = f"individual_{_ts}_Lmin{L_MIN:.0f}_{N_DAYS}d"
RUN_DIR  = RESULTS / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"  Scenario : Individual Market Participation (no ADMM)")
print(f"  Run name : {RUN_NAME}")
print(f"  Output   : {RUN_DIR}")
print(f"  L_min={L_MIN:.0f} kW (per agent)  |  {N_DAYS} days  |  seed={SEED}")
print(f"  Agents   : {[d['name'] for d in FLEET_DICTS]}")

res, agents = run_individual(
    agents       = fleet,
    n_days       = N_DAYS,
    start_date   = START_DATE,
    seed         = SEED,
    l_min        = L_MIN,
    h2_profile   = H2_DEMAND_PROFILE,
)


# %% [2]  Save all figures to RUN_DIR
# ─────────────────────────────────────────────────────────────────────────────
_fig.RESULTS   = RUN_DIR
_fig.L_MIN_KW  = L_MIN
_fig.L_MAX_KW  = float('inf')

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
_fig.fig_all_days_aggregate(res)
print(f"  All figures → {RUN_DIR}/")


# %% [3]  Export CSV data & parameter log
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

# ── per_agent_lmin.csv  (track when L_min binds per agent)
with open(RUN_DIR / "per_agent_lmin.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Agent", "Day", "lmin_steps", "lmin_%", "avg_import_kW"])
    T = 288
    for d in range(N_DAYS):
        for i, ag in enumerate(agents):
            p_agent = res.p_net[d, :, i]
            lmin_count = int(np.sum(p_agent <= L_MIN + 1.0))
            avg_import = float(p_agent.mean())
            w.writerow([ag.name, d + 1, lmin_count,
                        f"{lmin_count/T*100:.1f}", f"{avg_import:.1f}"])

# ── run_params.txt
with open(RUN_DIR / "run_params.txt", "w") as f:
    f.write(f"Run: {RUN_NAME}\n")
    f.write(f"Scenario: Individual Market Participation (no ADMM)\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
    f.write(f"[Market]\n")
    f.write(f"L_MIN={L_MIN} kW (per agent)\nL_MAX=∞ (not applicable)\n")
    f.write(f"H2_PRICE={H2_PRICE} $/kg\n")
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
print("Done — Scenario A (Individual).")

# %%
