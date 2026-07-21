"""
06_simple_game_results.py
=========================
Generate all results and figures for the *simple game* formulation.

Runs a 7-day rolling-horizon ADMM simulation on real PJM-style data,
then writes 9 figures to results/figures/simple_*.pdf.

Usage:
  python scripts/06_simple_game_results.py
"""

import os, sys, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from amrhg.game.simple_game import (
    build_4agent_simple_game,
    make_simple_param,
)
from amrhg.solvers.admm_solver import admm_solve

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

AGENT_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
AGENT_NAMES  = ["VRFB", "PV+Batt", "PEM Elz.", "Alk. Elz."]
FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Config & game
# ─────────────────────────────────────────────────────────────────────────────
with open("configs/base.yaml") as f:
    cfg = yaml.safe_load(f)

VRFB_CFG = cfg["agents"]["vrfb"]
PV_CFG   = cfg["agents"]["pv_battery"]
PEM_CFG  = cfg["agents"]["electrolyzer_pem"]
ALK_CFG  = cfg["agents"]["electrolyzer_alk"]

H      = 6
DT_HR  = 5 / 60
L_MAX  = cfg["market"]["l_max_mw"] * 1000    # 1500 kW
L_MIN  = cfg["market"]["l_min_mw"] * 1000    # -10000 kW (non-binding)
GAMMA  = 1.0

agents, layout, game = build_4agent_simple_game(
    VRFB_CFG, PV_CFG, PEM_CFG, ALK_CFG,
    H=H, dt_hr=DT_HR, gamma_imb=GAMMA,
)
print(f"Game: N={game.N}, n_x/agent={H}, n_p(shared)={layout.n_p}")
print(f"  Coupling: L_min={L_MIN:.0f} kW, L_max={L_MAX:.0f} kW")

# ─────────────────────────────────────────────────────────────────────────────
#  Load data (use synthetic week if available, fall back to generating)
# ─────────────────────────────────────────────────────────────────────────────
data_path = "data/processed/synthetic_week.npz"
d = np.load(data_path)
rtm_lmp   = d["rtm_lmp"]     # (2016,)  $/MWh
da_lmp    = d["dam_lmp"]     # (168,)   $/MWh
pv_cf     = d["pv_capacity_factor"]  # (2016,)  [0,1]

PV_CAP_KW = PV_CFG["pv_capacity_kw"]   # 1000 kW
pv_kw     = pv_cf * PV_CAP_KW          # (2016,) kW actual PV

N_STEPS = len(rtm_lmp)  # 2016
N_DAYS  = N_STEPS // 288
print(f"Data: {N_DAYS} days, {N_STEPS} RTM steps, "
      f"RT price range=[{rtm_lmp.min():.0f},{rtm_lmp.max():.0f}] $/MWh")

# ─────────────────────────────────────────────────────────────────────────────
#  Build DA schedules per agent (constant baselines)
# ─────────────────────────────────────────────────────────────────────────────
# VRFB: discharge when DA price > median, charge when below median
da_med = np.median(da_lmp)
vrfb_da_hr = np.where(da_lmp > da_med, -600.0, 200.0)   # (168,) kW (neg=export)

# PV+Battery: net = -(PV at each hour) as a baseline (solar-first)
pv_hr = np.array([pv_kw[h * 12:(h + 1) * 12].mean() for h in range(168)])
pv_da_hr = -pv_hr  # export solar

# PEM: 833 kW constant (400 kg/day / (0.02 kg/kWh * 24 h) = 833 kW)
pem_da_hr = np.full(168, PEM_CFG["h2_daily_target_kg"] /
                    (PEM_CFG["h2_production_kg_per_kwh"] * 24))
# Alk: 694 kW constant
alk_da_hr = np.full(168, ALK_CFG["h2_daily_target_kg"] /
                    (ALK_CFG["h2_production_kg_per_kwh"] * 24))

# Expand to 5-min (repeat each hour value 12 times)
def hr_to_5min(arr):
    return np.repeat(arr, 12)

vrfb_da = hr_to_5min(vrfb_da_hr)
pv_da   = hr_to_5min(pv_da_hr)
pem_da  = hr_to_5min(pem_da_hr)
alk_da  = hr_to_5min(alk_da_hr)

da_schedules_5min = [vrfb_da, pv_da, pem_da, alk_da]

# ─────────────────────────────────────────────────────────────────────────────
#  Closed-loop rolling-horizon ADMM simulation
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# Initial states
soc_vrfb = VRFB_CFG["soc_init_kwh"]          # 3500 kWh
soc_pv   = PV_CFG["soc_init_kwh"]            # 2000 kWh
h2_pem   = PEM_CFG["tank_init_kg"]           # 200 kg
h2_alk   = ALK_CFG["tank_init_kg"]           # 320 kg

# Storage dynamics params
eta_vrfb   = VRFB_CFG["roundtrip_efficiency"]    # 0.75
eta_pv     = PV_CFG["roundtrip_efficiency"]      # 0.92
eta_pem    = PEM_CFG["h2_production_kg_per_kwh"] # 0.020
eta_alk    = ALK_CFG["h2_production_kg_per_kwh"] # 0.018

# H2 offtake per step (same as in build_4agent_simple_game)
STEPS_PER_DAY = 288
pem_offtake = PEM_CFG["h2_daily_target_kg"] / STEPS_PER_DAY  # 1.389 kg/step
alk_offtake = ALK_CFG["h2_daily_target_kg"] / STEPS_PER_DAY  # 1.042 kg/step

soc_min_v, soc_max_v = VRFB_CFG["soc_min_kwh"],  VRFB_CFG["soc_max_kwh"]
soc_min_p, soc_max_p = PV_CFG["soc_min_kwh"],    PV_CFG["soc_max_kwh"]
h2_min_pe,  h2_max_pe = PEM_CFG["tank_min_kg"],   PEM_CFG["tank_max_kg"]
h2_min_al,  h2_max_al = ALK_CFG["tank_min_kg"],   ALK_CFG["tank_max_kg"]

# Logs
log_p       = np.zeros((4, N_STEPS))
log_states  = np.zeros((4, N_STEPS + 1))
log_iters   = np.zeros(N_STEPS, dtype=int)
log_conv    = np.zeros(N_STEPS, dtype=bool)
log_agg     = np.zeros(N_STEPS)
log_iters_hist = []

log_states[:, 0] = [soc_vrfb, soc_pv, h2_pem, h2_alk]

print("\nRunning 7-day rolling-horizon simulation...")
t_start = time.time()

RT_PRICE_SIGMA = 5.0  # $/MWh forecast noise

for t in range(N_STEPS):
    # Build price forecast (H steps ahead with noise)
    price_idx = np.arange(t, t + H) % N_STEPS
    lmp_fcast = rtm_lmp[price_idx] + rng.normal(0, RT_PRICE_SIGMA, H)
    lmp_fcast = np.clip(lmp_fcast, 0.0, 500.0)

    # PV forecast (H steps ahead with noise)
    pv_fcast = pv_kw[price_idx] + rng.normal(0, 30.0, H)
    pv_fcast = np.clip(pv_fcast, 0.0, PV_CAP_KW)

    # DA schedules for this horizon window
    da_slices = [da_schedules_5min[i][price_idx] for i in range(4)]

    # Pack parameter vector
    p_vec = make_simple_param(
        layout=layout,
        state_inits=[soc_vrfb, soc_pv, h2_pem, h2_alk],
        lmp_rt=lmp_fcast,
        da_schedules=da_slices,
        l_max_kw=L_MAX,
        l_min_kw=L_MIN,
        pv_forecast=pv_fcast,
    )

    # Solve GNE via ADMM
    res = admm_solve(game, p_vec, rho=5.0, max_iter=500, tol=1.0, verbose=False)

    # Extract first-step actions for all agents
    p_applied = np.array([res.x_sol[i][0] for i in range(4)])
    log_p[:, t]      = p_applied
    log_iters[t]     = res.n_iter
    log_conv[t]      = res.converged
    log_agg[t]       = p_applied.sum()
    log_iters_hist.append(res.n_iter)

    # Advance states (true dynamics with actual PV and H2 offtake)
    pv_actual = float(pv_kw[t])
    soc_vrfb = np.clip(soc_vrfb + eta_vrfb * DT_HR * p_applied[0],
                       soc_min_v, soc_max_v)
    soc_pv   = np.clip(soc_pv   + eta_pv   * DT_HR * (p_applied[1] + pv_actual),
                       soc_min_p, soc_max_p)
    h2_pem   = np.clip(h2_pem   + eta_pem  * DT_HR * p_applied[2] - pem_offtake,
                       h2_min_pe, h2_max_pe)
    h2_alk   = np.clip(h2_alk   + eta_alk  * DT_HR * p_applied[3] - alk_offtake,
                       h2_min_al, h2_max_al)

    log_states[:, t + 1] = [soc_vrfb, soc_pv, h2_pem, h2_alk]

    if (t + 1) % 288 == 0:
        day = (t + 1) // 288
        day_iters = log_iters[t - 287:t + 1]
        conv_pct  = log_conv[t - 287:t + 1].mean() * 100
        viol_up   = (log_agg[t - 287:t + 1] > L_MAX + 1.0).mean() * 100
        print(f"  Day {day}: avg_iters={day_iters.mean():.1f}, "
              f"conv={conv_pct:.0f}%, viol_upper={viol_up:.1f}%")

elapsed = time.time() - t_start
print(f"Done in {elapsed:.1f}s  ({N_STEPS} steps × {int(np.mean(log_iters))} avg iters)")
print(f"Convergence: {log_conv.mean() * 100:.1f}%  "
      f"Upper violations: {(log_agg > L_MAX + 1.0).mean() * 100:.1f}%  "
      f"Lower violations: {(log_agg < L_MIN - 1.0).mean() * 100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
#  Compute financials  (real settlement dollars only — not the optimization penalty)
# ─────────────────────────────────────────────────────────────────────────────
# RT energy settlement: λ_k [$/MWh] × Δt [hr] × p_k [kW] / 1000 [kW/MW]  → $
rt_energy_cost = np.array([
    (rtm_lmp / 1000.0 * DT_HR * log_p[i]).sum() for i in range(4)
])

# H2 offtake revenue for electrolyzers (negative = earning)
# = -π_H2 [$/kg] × η [kg/kWh] × Δt [hr] × p [kW]
h2_revenue = np.zeros(4)
h2_revenue[2] = -(PEM_CFG["h2_price_per_kg"] * eta_pem * DT_HR * log_p[2].sum())
h2_revenue[3] = -(ALK_CFG["h2_price_per_kg"] * eta_alk * DT_HR * log_p[3].sum())

net_cost = rt_energy_cost + h2_revenue

print("\nFinancials (7-day, real settlement $):")
print(f"  {'Agent':12s}  {'RT energy ($)':>15s}  {'H2 rev ($)':>12s}  {'Net ($)':>12s}")
for i, name in enumerate(AGENT_NAMES):
    print(f"  {name:12s}  {rt_energy_cost[i]:>15.2f}  {h2_revenue[i]:>12.2f}  "
          f"{net_cost[i]:>12.2f}")
print(f"  {'TOTAL':12s}  {rt_energy_cost.sum():>15.2f}  {h2_revenue.sum():>12.2f}  "
      f"{net_cost.sum():>12.2f}")

# ─────────────────────────────────────────────────────────────────────────────
#  FIGURES
# ─────────────────────────────────────────────────────────────────────────────

time_hr = np.arange(N_STEPS) * DT_HR
days    = time_hr / 24

def save(fig, name):
    path = os.path.join(FIG_DIR, f"simple_{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)

# ── Fig 1: Problem structure comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.set_title("Old Formulation", fontweight="bold")
categories = ["Decision\n$x_i$", "Per-agent\n$\\theta_i$", "Shared\n$n_p$"]
old_vals   = [18, 44, 44]
new_vals   = [6,  19, 42]
x_pos = np.arange(len(categories))
w = 0.35
bars_old = ax.bar(x_pos - w/2, old_vals, w, label="Old (z⁺/z⁻, deg.)", color="#E07B54", alpha=0.85)
bars_new = ax.bar(x_pos + w/2, new_vals, w, label="New (simple)", color="#4CAF82", alpha=0.85)
ax.set_xticks(x_pos); ax.set_xticklabels(categories)
ax.set_ylabel("Dimension")
ax.legend()
ax.set_ylim(0, 55)
for bar, v in zip(bars_old, old_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, str(v), ha="center", fontsize=8)
for bar, v in zip(bars_new, new_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, str(v), ha="center", fontsize=8)

ax2 = axes[1]
ax2.axis("off")
table_data = [
    ["Aspect",              "Old formulation",        "New (simple)"],
    ["Decision x_i",        "[p, z+, z-] in R^18",   "p in R^6"],
    ["Cost Hessian Q_i",    "block-diag + deg.",      "gamma*I_6  (diagonal)"],
    ["Imbalance penalty",   "linear gamma+z++gamma-z-","(1/2)*gamma*||p-pDA||^2"],
    ["Ramp constraints",    "yes",                    "no"],
    ["Degradation term",    "a_deg * p^2",            "no"],
    ["z-update dimension",  "2H = 12  (one-sided)",   "H = 6  (bilateral clip)"],
    ["Per-agent theta_i",   "n_p = 44",               "3H+1 = 19"],
    ["mpQP param total",    "62",                      "37"],
]
tbl = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.2, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#DDEBF7")
        cell.set_text_props(fontweight="bold")
    elif c == 2:
        cell.set_facecolor("#E8F5E9")
    elif c == 1:
        cell.set_facecolor("#FFF3E0")
ax2.set_title("Formulation Comparison", fontweight="bold")
fig.suptitle("Simple Game: Problem Structure vs.\ Old Formulation", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig1_structure")

# ── Fig 2: ADMM convergence (single step, detailed) ──────────────────────────
# Re-run one step with full residual logging
t_demo = 144  # noon Day 1
price_idx_d = np.arange(t_demo, t_demo + H) % N_STEPS
p_vec_demo = make_simple_param(
    layout=layout,
    state_inits=[VRFB_CFG["soc_init_kwh"], PV_CFG["soc_init_kwh"],
                 PEM_CFG["tank_init_kg"],   ALK_CFG["tank_init_kg"]],
    lmp_rt=rtm_lmp[price_idx_d],
    da_schedules=[da_schedules_5min[i][price_idx_d] for i in range(4)],
    l_max_kw=L_MAX, l_min_kw=L_MIN,
    pv_forecast=pv_kw[price_idx_d],
)
res_demo = admm_solve(game, p_vec_demo, rho=5.0, max_iter=300, tol=0.01, verbose=False)
pr_hist = res_demo.primal_res_hist if hasattr(res_demo, "primal_res_hist") else None
dr_hist = res_demo.dual_res_hist   if hasattr(res_demo, "dual_res_hist")   else None

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
iters_ax = np.arange(1, len(log_iters_hist) + 1)

ax = axes[0]
ax.hist(log_iters_hist, bins=30, color="#2196F3", edgecolor="white", alpha=0.85)
ax.axvline(np.mean(log_iters_hist), color="red", lw=1.5, linestyle="--",
           label=f"Mean = {np.mean(log_iters_hist):.1f}")
ax.set_xlabel("ADMM iterations per RTM step")
ax.set_ylabel("Count")
ax.set_title(f"Iteration Histogram ({N_STEPS} steps)")
ax.legend()

ax = axes[1]
iters_per_day = [log_iters[d*288:(d+1)*288].mean() for d in range(N_DAYS)]
ax.bar(np.arange(1, N_DAYS+1), iters_per_day, color=plt.cm.Blues(np.linspace(0.4, 0.9, N_DAYS)))
ax.set_xlabel("Day")
ax.set_ylabel("Avg iterations")
ax.set_title("Per-Day ADMM Performance")
ax.set_xticks(np.arange(1, N_DAYS+1))
ax.set_xticklabels([f"Day {d}" for d in range(1, N_DAYS+1)], rotation=20)
conv_pct_d = [log_conv[d*288:(d+1)*288].mean()*100 for d in range(N_DAYS)]
for i, (v, c) in enumerate(zip(iters_per_day, conv_pct_d)):
    ax.text(i+1, v+0.05, f"{v:.1f}\n({c:.0f}%)", ha="center", fontsize=7)

fig.suptitle("ADMM Convergence Statistics — 7-Day Rolling Horizon", fontweight="bold")
fig.tight_layout()
save(fig, "fig2_admm_convergence")

# ── Fig 3: Power trajectories (Day 1 & 2) ────────────────────────────────────
D = 2   # first two days
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
hrs_2d = time_hr[:D*288]

for i, (ax, col, name) in enumerate(zip(axes, AGENT_COLORS, AGENT_NAMES)):
    p_rt = log_p[i, :D*288]
    da   = da_schedules_5min[i][:D*288]
    ax.step(hrs_2d, p_rt, where="post", color=col,      lw=1.3, label="RTM (ADMM)")
    ax.step(hrs_2d, da,   where="post", color="gray",   lw=0.9, ls="--", label="DA schedule")
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_ylabel(f"{name}\n[kW]")
    ax.legend(loc="upper right", ncol=2)
    # shade nights
    for day in range(D):
        ax.axvspan(day*24 + 0, day*24 + 6,  alpha=0.05, color="navy")
        ax.axvspan(day*24 + 20, day*24 + 24, alpha=0.05, color="navy")

axes[-1].set_xlabel("Hour")
axes[-1].set_xlim(0, D*24)
axes[-1].xaxis.set_major_locator(plt.MultipleLocator(4))
fig.suptitle("Per-Agent RTM Power Decisions — First 2 Days", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig3_power_trajectories")

# ── Fig 4: Coupling constraint satisfaction ───────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax = axes[0]
ax.fill_between(days[:288], log_agg[:288], step="post", alpha=0.4, color="#2196F3")
ax.step(days[:288], log_agg[:288], where="post", color="#2196F3", lw=1.2, label="$\\Sigma p_i$")
ax.axhline(L_MAX, color="red", lw=1.5, ls="--", label=f"$L_{{max}}={L_MAX:.0f}$ kW")
ax.axhline(L_MIN if L_MIN > -5000 else 0, color="orange", lw=1.0, ls=":", label="$L_{min}$")
ax.set_ylabel("Aggregate power [kW]")
ax.set_title("Day 1 — Aggregate PCC Power")
ax.legend(ncol=3)

ax = axes[1]
agg_max_per_step = log_agg
margin = L_MAX - agg_max_per_step
# rolling hourly max violation (if any)
col_bar = np.where(agg_max_per_step > L_MAX, "#E53935", "#43A047")
ax.bar(days, agg_max_per_step, width=DT_HR/24, color=col_bar, alpha=0.7)
ax.axhline(L_MAX, color="red", lw=1.5, ls="--", label=f"$L_{{max}}$")
ax.set_xlabel("Day")
ax.set_ylabel("Aggregate power [kW]")
ax.set_title(f"7-Day PCC — violations: {(log_agg > L_MAX + 1.0).sum()} / {N_STEPS} steps "
             f"({(log_agg > L_MAX + 1.0).mean()*100:.1f}%)")
ax.legend()
ax.set_xlim(0, N_DAYS)

fig.suptitle("PCC Coupling Constraint Satisfaction", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig4_coupling")

# ── Fig 5: State trajectories ─────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

labels     = ["SoC [kWh]", "SoC [kWh]", "H₂ Inventory [kg]", "H₂ Inventory [kg]"]
mins_state = [soc_min_v, soc_min_p, h2_min_pe, h2_min_al]
maxs_state = [soc_max_v, soc_max_p, h2_max_pe, h2_max_al]

for i in range(4):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    state_vals = log_states[i, :-1]
    ax.plot(days, state_vals, color=AGENT_COLORS[i], lw=1.2)
    ax.axhline(mins_state[i], color="red",  lw=1.0, ls="--", label="min/max")
    ax.axhline(maxs_state[i], color="red",  lw=1.0, ls="--")
    ax.fill_between(days, mins_state[i], maxs_state[i], alpha=0.05, color=AGENT_COLORS[i])
    ax.set_title(AGENT_NAMES[i])
    ax.set_ylabel(labels[i])
    ax.set_xlabel("Day")
    ax.legend(fontsize=7)
    ax.set_xlim(0, N_DAYS)

fig.suptitle("State Trajectories — 7 Days (SoC and H₂ Inventory)", fontsize=11, fontweight="bold")
save(fig, "fig5_states")

# ── Fig 6: Economic response to price (scatter) ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for i, (ax, col, name) in enumerate(zip(axes.flat, AGENT_COLORS, AGENT_NAMES)):
    # downsample for clarity
    idx = np.arange(0, N_STEPS, 3)
    lmp_pts = rtm_lmp[idx]
    p_pts   = log_p[i, idx]
    sc = ax.scatter(lmp_pts, p_pts, c=col, s=4, alpha=0.4)
    # linear fit
    fit = np.polyfit(lmp_pts, p_pts, 1)
    xl  = np.linspace(lmp_pts.min(), lmp_pts.max(), 100)
    ax.plot(xl, np.polyval(fit, xl), "k--", lw=1.2,
            label=f"slope={fit[0]:.1f} kW/($/MWh)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("RTM LMP [$/MWh]")
    ax.set_ylabel("RTM power [kW]")
    ax.set_title(name)
    ax.legend(fontsize=7)

fig.suptitle("Economic Response: Power vs.\ RT LMP", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig6_price_response")

# ── Fig 7: Per-agent and aggregate cost summary ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
x_pos = np.arange(4)
w = 0.3
ax.bar(x_pos - w/2, rt_energy_cost, w, label="RT energy cost (\$)", color="#42A5F5")
ax.bar(x_pos + w/2, h2_revenue,     w, label="H2 revenue (\$)",     color="#66BB6A")
ax.set_xticks(x_pos); ax.set_xticklabels(AGENT_NAMES, rotation=15)
ax.set_ylabel("Cost / Revenue (\$)")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("7-Day Settlement Breakdown per Agent")
ax.legend()

ax = axes[1]
net_colors = ["#E53935" if v > 0 else "#43A047" for v in net_cost]
ax.bar(np.arange(4), net_cost, color=net_colors, alpha=0.85)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(AGENT_NAMES, rotation=15)
ax.set_ylabel("Net Cost (\$)  [negative = profit]")
ax.set_title(f"Net 7-Day Cost  (Total = \${net_cost.sum():.0f})")
for i, v in enumerate(net_cost):
    ax.text(i, v + (50 if v > 0 else -200), f"\${v:.0f}", ha="center", fontsize=8)

fig.suptitle("Simple Game — 7-Day Financial Summary", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig7_financials")

# ── Fig 8: Cumulative costs over time ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for i, (col, name) in enumerate(zip(AGENT_COLORS, AGENT_NAMES)):
    cum_rt  = np.cumsum(rtm_lmp / 1000 * DT_HR * log_p[i])
    ax.plot(days, cum_rt, color=col, lw=1.4, label=name)

ax.axhline(0, color="black", lw=0.7, ls=":")
ax.set_xlabel("Day")
ax.set_ylabel("Cumulative RT energy cost (\$)")
ax.set_title("Cumulative RT Energy Cost (negative = earning from export)")
ax.legend()
ax.set_xlim(0, N_DAYS)

fig.tight_layout()
save(fig, "fig8_cumulative_cost")

# ── Fig 9: GNE problem parameter map ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")

N_ag = layout.n_agents
rows = [
    ["Slot",        "Symbol",            "Size",        "Description"],
    ["[0 .. N-1]",  "SoC0 / Inv0",       f"N = {N_ag}", "Own initial state"],
    ["[N .. N+H-1]","lambda_RT (1..H)",   f"H = {H}",   "RT LMP forecast (shared)"],
    ["[N+H ..]",    "p_DA_i (1..H)",      f"N*H = {N_ag*H}", "DA schedules (all agents)"],
    ["[N+H+NH]",    "L_max",              "1",          "PCC upper limit (kW)"],
    ["[N+H+NH+1]",  "L_min",              "1",          "PCC lower limit (kW)"],
    ["[N+H+NH+2..]","g_PV (1..H)",        f"H = {H}",   "PV forecast (agent 1 only)"],
    ["TOTAL",        "",                  str(layout.n_p), ""],
]
tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
               cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#DDEBF7")
        cell.set_text_props(fontweight="bold")
    if r == len(rows) - 1 and r > 0:
        cell.set_facecolor("#E8F5E9")
        cell.set_text_props(fontweight="bold")

n_theta = 1 + H + H + (N_ag - 1) * H
ax.set_title(
    f"Shared parameter vector p: n_p = {layout.n_p}  (N={N_ag}, H={H}, with PV)\n"
    f"Per-agent mpQP theta_i: state(1) + prices({H}) + own DA({H}) + p_neg_i({(N_ag-1)*H}) = {n_theta}",
    fontsize=10,
)
fig.tight_layout()
save(fig, "fig9_param_map")

print(f"\nAll figures saved to {FIG_DIR}/simple_*.pdf")
print("\nKey results:")
print(f"  ADMM convergence: {log_conv.mean()*100:.1f}% of {N_STEPS} steps")
print(f"  Avg iterations:   {np.mean(log_iters):.1f}")
print(f"  PCC upper viol.:  {(log_agg > L_MAX+1.0).sum()} steps ({(log_agg > L_MAX+1.0).mean()*100:.2f}%)")
print(f"  PCC lower viol.:  {(log_agg < L_MIN-1.0).sum()} steps")
print(f"  7-day net cost:   \${net_cost.sum():.2f}  ({'profit' if net_cost.sum() < 0 else 'cost'})")
