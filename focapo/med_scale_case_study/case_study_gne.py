"""
case_study_gne.py — Scenario B: GNE Coalition (ADMM) with medium-scale agents.

Same ADMM framework as admm_elz_market/case_study.py, but:
  - Medium-scale agents (125–250 kW) from agents_med.py
  - No L_max constraint (l_max → ∞)
  - L_min = 100 kW aggregate coupling

Run cell-by-cell:  VS Code  →  Ctrl+Enter   |   Spyder  →  F9
Each run auto-creates a timestamped subfolder under results/ and saves:
  • All figures (PDF + PNG)
  • h2_production.csv, financials.csv, coupling_stats.csv, run_params.txt
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

# Add parent project to path for shared modules
from _shared_path import SHARED_DIR as _SHARED_DIR  # noqa: E402
_PARENT = str(_SHARED_DIR)
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from agents import ElzAgent, DT_RT
import main as _fig
from forecasting import load_pjm_week

from agents_med import (
    FLEET_DICTS, L_MIN, H2_PRICE,
    RHO_RT, RHO_DA, TOL_RT, TOL_DA,
    N_DAYS, START_DATE, SEED, H2_DEMAND_PROFILE,
)
from simulation_sc import run_gne


# %% [1]  Market parameters
# ─────────────────────────────────────────────────────────────────────────────
L_MAX = float('inf')  # No L_max constraint


# %% [2]  Build fleet & run simulation
# ─────────────────────────────────────────────────────────────────────────────
def _make_agent(d: dict) -> ElzAgent:
    return ElzAgent(h2_target_kg_day=0.0, **d)

fleet = [_make_agent(d) for d in FLEET_DICTS]

_ts      = datetime.now().strftime('%Y%m%d_%H%M')
RUN_NAME = f"gne_{_ts}_Lmin{L_MIN:.0f}_noLmax_{N_DAYS}d"
RUN_DIR  = RESULTS / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"  Scenario : GNE Coalition (MIQP-ADMM, semi-continuous bidding)")
print(f"  Run name : {RUN_NAME}")
print(f"  Output   : {RUN_DIR}")
print(f"  L_min={L_MIN:.0f} kW  |  L_max=∞  |  {N_DAYS} days  |  seed={SEED}")
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


# %% [3]  Save all figures to RUN_DIR
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


# %% [4]  Export CSV data & parameter log
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
                "lmin_steps", "lmin_%"])
    from datetime import timedelta, date as _date
    base = _date.fromisoformat(START_DATE)
    for d in range(N_DAYS):
        dt     = base + timedelta(days=d)
        lmin_n = int(res.lmin_bind[d].sum())
        T      = res.lmin_bind.shape[1]
        prof   = H2_DEMAND_PROFILE[d % len(H2_DEMAND_PROFILE)]
        w.writerow([d + 1, dt.isoformat(), f"{prof:.2f}",
                    lmin_n, f"{lmin_n/T*100:.1f}"])
    tot_lmin = int(res.lmin_bind.sum())
    total_T  = N_DAYS * T
    w.writerow(["TOTAL", "", "",
                tot_lmin, f"{tot_lmin/total_T*100:.1f}"])

# ── run_params.txt
with open(RUN_DIR / "run_params.txt", "w") as f:
    f.write(f"Run: {RUN_NAME}\n")
    f.write(f"Scenario: GNE Coalition (ADMM)\n")
    f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
    f.write(f"[Market]\n")
    f.write(f"L_MIN={L_MIN} kW\nL_MAX=∞ (no upper bound)\nH2_PRICE={H2_PRICE} $/kg\n")
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

# ── results.npz (full arrays for compare.py)
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
print("Done — Scenario B (GNE).")

# %%
