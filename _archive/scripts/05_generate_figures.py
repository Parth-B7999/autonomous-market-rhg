"""
10_generate_figures.py — Generate all paper figures for FACET-mpGNE report.

Figures saved to results/figures/:
  fig_pjm_week_data.pdf
  fig_cr_parameter_bounds.pdf
  fig_communication_rounds.pdf
  fig_power_trajectories.pdf
  fig_state_trajectories.pdf
  fig_cost_comparison.pdf
  fig_h2_production.pdf

Run: python scripts/10_generate_figures.py
"""

from __future__ import annotations
import pickle, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open(ROOT / "results" / "sim_rhg_pjm.pkl",  "rb") as f: gne  = pickle.load(f)
with open(ROOT / "results" / "sim_admm_pjm.pkl", "rb") as f: admm = pickle.load(f)

cfg     = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text())
T       = gne["T"]           # 2016
H       = gne["H"]           # 6
DT_HR   = 5 / 60
steps   = np.arange(T)
hours   = steps * 5 / 60     # x-axis in hours

rtm_lmp = gne["rtm_lmp"]     # (2016,)
pv_5min = gne["pv_5min"]     # (2016,)
dam_lmp = None
try:
    from amrhg.data.pjm import load_pjm_data
    pjm = load_pjm_data(cfg)
    dam_lmp = pjm["dam_lmp"]   # (168,)
except Exception:
    pass

x_gne   = gne["x_traj"]     # (2016, 24)
x_admm  = admm["x_traj"]
s_gne   = gne["states_traj"]  # (2016, 4)
s_admm  = admm["states_traj"]
c_gne   = gne["costs_traj"]   # (2016, 4)
c_admm  = admm["costs_traj"]
iters   = admm["n_iter_traj"]  # (2016,)

AGENT_NAMES = ["VRFB", "PV+Batt", "PEM", "Alk"]
COLORS = {"gne": "#1f77b4", "admm": "#ff7f0e"}

# ── RC params ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "lines.linewidth": 1.0, "figure.dpi": 150,
})

def day_vlines(ax, n_days=7, **kwargs):
    for d in range(1, n_days):
        ax.axvline(d * 24, color="gray", lw=0.5, ls="--", alpha=0.5, **kwargs)

def save(name):
    p = FIG_DIR / name
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  saved → {p.name}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: PJM week data
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

# (a) RTM LMP
ax = axes[0]
ax.plot(hours, rtm_lmp, lw=0.5, color="steelblue", alpha=0.8, label="RTM LMP")
ax.axhline(10,  color="red",  lw=1.0, ls="--", alpha=0.7, label="CR bound [10, 220]")
ax.axhline(220, color="red",  lw=1.0, ls="--", alpha=0.7)
ax.fill_between(hours, 10, 220, alpha=0.06, color="green", label="97.7% of steps")
ax.set_ylabel("LMP [\\$/MWh]")
ax.set_title("(a) RTM 5-min LMP — PJM-RTO, 2024-07-08")
ax.legend(loc="upper right", ncol=2)
ax.set_ylim(bottom=0)
day_vlines(ax)

# (b) DAM LMP
ax = axes[1]
if dam_lmp is not None:
    dam_hours = np.arange(len(dam_lmp))
    ax.step(dam_hours, dam_lmp, where="post", color="darkorange", lw=1.0, label="DAM LMP")
    ax.axhline(10,  color="red", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(220, color="red", lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("LMP [\\$/MWh]")
    ax.set_title("(b) Hourly DAM LMP")
    ax.legend(loc="upper right")
    day_vlines(ax)
else:
    ax.text(0.5, 0.5, "DAM data not available", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_ylabel("LMP [\\$/MWh]")
    ax.set_title("(b) Hourly DAM LMP")

# (c) PV generation
ax = axes[2]
ax.fill_between(hours, 0, pv_5min, alpha=0.6, color="gold")
ax.plot(hours, pv_5min, lw=0.4, color="goldenrod")
ax.axhline(950, color="purple", lw=1.0, ls="--", alpha=0.7, label="CR bound 950 kW")
ax.set_ylabel("PV [kW]")
ax.set_xlabel("Time [hours from Mon 2024-07-08]")
ax.set_title("(c) 5-min PV Generation (1 MW capacity)")
ax.legend(loc="upper right")
ax.set_xlim(0, 168)
day_vlines(ax)
for d in range(7):
    ax.text(d * 24 + 1, 980, f"Day {d+1}", fontsize=6, color="gray")

plt.tight_layout()
save("fig_pjm_week_data.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: CR bounds & counts
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

# Left: bar chart old vs new CR counts
old_cr = [2810, 4516, 8025, 8427]
new_cr = [2810, 3895, 2662, 2667]
x = np.arange(4)
w = 0.35
ax = axes[0]
bars_old = ax.bar(x - w/2, old_cr, w, label="Old bounds", color="lightcoral", alpha=0.8)
bars_new = ax.bar(x + w/2, new_cr, w, label="Week-specific", color="steelblue", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(AGENT_NAMES)
ax.set_ylabel("Critical Regions (CRs)")
ax.set_title("CR Count: Old vs. Week-Specific Bounds")
ax.legend()
for bar, old, new in zip(bars_new, old_cr, new_cr):
    if new < old:
        pct = (old - new) / old * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"-{pct:.0f}%", ha="center", va="bottom", fontsize=7, color="darkgreen")
total_old = sum(old_cr); total_new = sum(new_cr)
ax.text(0.5, 0.97, f"Total: {total_old:,} → {total_new:,}  ({(total_old-total_new)/total_old*100:.0f}% reduction)",
        transform=ax.transAxes, ha="center", va="top", fontsize=7, color="darkred")

# Right: parameter bounds table as text
ax = axes[1]
ax.axis("off")
table_data = [
    ["Parameter", "Old bounds", "New bounds", "Ratio"],
    ["LMP [\\$/MWh]", "[20, 130]", "[10, 220]", "1.9×"],
    ["VRFB DA [kW]", "[-100, 100]", "[-20, 20]", "0.2×"],
    ["PV DA [kW]", "[-500, 0]", "[-480, 0]", "0.96×"],
    ["PEM DA [kW]", "[0, 1200]", "[750, 900]", "0.13×"],
    ["Alk DA [kW]", "[0, 1000]", "[620, 750]", "0.13×"],
    ["PV [kW]", "[0, 1000]", "[0, 950]", "0.95×"],
    ["Neighbor pairs", "130,196", "52,595", "0.40×"],
]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
tbl.scale(1, 1.4)
ax.set_title("Parameter Space Calibration", fontsize=9)

plt.tight_layout()
save("fig_cr_parameter_bounds.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Communication rounds
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

# Left: bar chart  GNE=0 vs ADMM=18 median
ax = axes[0]
methods = ["FACET-mpGNE", "ADMM\n(per step)"]
vals = [0, np.median(iters)]
colors = [COLORS["gne"], COLORS["admm"]]
bars = ax.bar(methods, vals, color=colors, width=0.5, alpha=0.9)
ax.bar(methods[1:], [np.percentile(iters, 95) - np.median(iters)],
       bottom=[np.median(iters)], color=COLORS["admm"], width=0.5, alpha=0.4,
       label=f"p95 = {np.percentile(iters,95):.0f}")
ax.set_ylabel("Iterations per 5-min step")
ax.set_title("Online Communication Rounds")
ax.text(0, 0.5, "0\n(explicit)", ha="center", va="bottom", fontsize=9, color="white",
        fontweight="bold")
ax.text(1, np.median(iters) + 0.5, f"median={np.median(iters):.0f}", ha="center",
        va="bottom", fontsize=8)
ax.legend()
ax.annotate(f"33,803 total rounds\neliminated over 7 days",
            xy=(0.5, 0.7), xycoords="axes fraction", ha="center",
            fontsize=8, color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# Right: histogram of ADMM iteration counts
ax = axes[1]
ax.hist(iters, bins=range(int(iters.min()), int(iters.max())+2), color=COLORS["admm"],
        alpha=0.8, edgecolor="white", linewidth=0.3)
ax.axvline(np.median(iters), color="red", lw=1.5, ls="--", label=f"median = {np.median(iters):.0f}")
ax.axvline(np.percentile(iters, 95), color="darkred", lw=1.5, ls=":",
           label=f"p95 = {np.percentile(iters,95):.0f}")
ax.set_xlabel("ADMM iterations per step")
ax.set_ylabel("Count (steps)")
ax.set_title("ADMM Iteration Distribution (T=2016)")
ax.legend()

plt.tight_layout()
save("fig_communication_rounds.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Power trajectories (first 2 days)
# ─────────────────────────────────────────────────────────────────────────────
T2 = 2 * 288   # 2 days
h2 = hours[:T2]

# First-step control action per agent
u_gne  = [x_gne[:T2,  i*H]   for i in range(4)]
u_admm = [x_admm[:T2, i*H]   for i in range(4)]

fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True)
ylabels = ["Power [kW]\n(+import)", "Power [kW]\n(+import)",
           "Power [kW]\n(load)", "Power [kW]\n(load)"]
for i in range(4):
    ax = axes[i]
    ax.plot(h2, u_gne[i],  lw=0.8, color=COLORS["gne"],  label="FACET-GNE", alpha=0.9)
    ax.plot(h2, u_admm[i], lw=0.8, color=COLORS["admm"], label="ADMM", alpha=0.7,
            linestyle="--")
    ax.set_ylabel(ylabels[i])
    ax.set_title(f"Agent {i}: {AGENT_NAMES[i]}")
    if i == 0:
        ax.legend(loc="upper right", ncol=2)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    day_vlines(ax)

# Add RT LMP on top axis
ax2 = axes[0].twinx()
ax2.plot(h2, rtm_lmp[:T2], color="gray", lw=0.5, alpha=0.5, label="LMP")
ax2.set_ylabel("LMP [\\$/MWh]", color="gray", fontsize=7)
ax2.tick_params(axis="y", labelcolor="gray", labelsize=6)

axes[-1].set_xlabel("Time [hours from Mon 2024-07-08]")
axes[-1].set_xlim(0, 48)
plt.tight_layout()
save("fig_power_trajectories.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: State trajectories
# ─────────────────────────────────────────────────────────────────────────────
vrfb_bounds = (cfg["agents"]["vrfb"]["soc_min_kwh"], cfg["agents"]["vrfb"]["soc_max_kwh"])
pv_bounds   = (cfg["agents"]["pv_battery"]["soc_min_kwh"], cfg["agents"]["pv_battery"]["soc_max_kwh"])
pem_bounds  = (cfg["agents"]["electrolyzer_pem"]["tank_min_kg"], cfg["agents"]["electrolyzer_pem"]["tank_max_kg"])
alk_bounds  = (cfg["agents"]["electrolyzer_alk"]["tank_min_kg"], cfg["agents"]["electrolyzer_alk"]["tank_max_kg"])

bounds_list = [vrfb_bounds, pv_bounds, pem_bounds, alk_bounds]
ylabels_s   = ["SoC [kWh]", "SoC [kWh]", "H₂ Inv [kg]", "H₂ Inv [kg]"]

fig, axes = plt.subplots(4, 1, figsize=(7, 8), sharex=True)
for i in range(4):
    ax = axes[i]
    ax.plot(hours, s_gne[:, i],  lw=0.9, color=COLORS["gne"],  label="FACET-GNE")
    ax.plot(hours, s_admm[:, i], lw=0.9, color=COLORS["admm"], label="ADMM",
            linestyle="--", alpha=0.7)
    ax.axhline(bounds_list[i][0], color="k", lw=1.0, ls=":", alpha=0.6, label="Bounds")
    ax.axhline(bounds_list[i][1], color="k", lw=1.0, ls=":", alpha=0.6)
    ax.set_ylabel(ylabels_s[i])
    ax.set_title(f"Agent {i}: {AGENT_NAMES[i]}")
    if i == 0:
        ax.legend(loc="upper right", ncol=3)
    day_vlines(ax)

axes[-1].set_xlabel("Time [hours from Mon 2024-07-08]")
axes[-1].set_xlim(0, 168)
plt.tight_layout()
save("fig_state_trajectories.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Cost comparison
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

# Left: per-agent 7-day total cost
total_gne  = c_gne.sum(axis=0)
total_admm = c_admm.sum(axis=0)
x = np.arange(4)
w = 0.35
ax = axes[0]
b1 = ax.bar(x - w/2, total_gne,  w, label="FACET-GNE", color=COLORS["gne"],  alpha=0.9)
b2 = ax.bar(x + w/2, total_admm, w, label="ADMM",      color=COLORS["admm"], alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(AGENT_NAMES)
ax.set_ylabel("7-day Net Cost [\\$]\n(positive = pay, negative = profit)")
ax.set_title("Per-Agent 7-Day Cost")
ax.legend()
ax.axhline(0, color="k", lw=0.8)
for xi, (gc, ac) in enumerate(zip(total_gne, total_admm)):
    delta = (gc - ac) / (abs(ac) + 1e-9) * 100
    ax.text(xi, max(gc, ac) + abs(max(gc, ac)) * 0.02, f"{delta:+.1f}%",
            ha="center", fontsize=6, color="darkgreen" if delta < 0 else "darkred")

# Right: cumulative total cluster cost
cum_gne  = c_gne.sum(axis=1).cumsum()
cum_admm = c_admm.sum(axis=1).cumsum()
ax = axes[1]
ax.plot(hours, cum_gne,  color=COLORS["gne"],  lw=1.2, label="FACET-GNE")
ax.plot(hours, cum_admm, color=COLORS["admm"], lw=1.2, label="ADMM", ls="--")
ax.fill_between(hours, cum_gne, cum_admm,
                where=cum_gne < cum_admm, alpha=0.2, color=COLORS["gne"],
                label="GNE saving")
ax.set_xlabel("Time [hours]")
ax.set_ylabel("Cumulative Cluster Cost [\\$]")
ax.set_title("Cumulative Cluster Cost")
ax.legend()
day_vlines(ax)
total_saving = float(cum_admm[-1] - cum_gne[-1])
ax.annotate(f"7-day saving:\n${total_saving:.0f} (−{total_saving/abs(cum_admm[-1])*100:.1f}%)",
            xy=(100, cum_gne[-1]),
            xytext=(60, cum_gne[-1] - abs(cum_admm[-1]) * 0.1),
            arrowprops=dict(arrowstyle="->", color="darkgreen"),
            fontsize=7, color="darkgreen")

plt.tight_layout()
save("fig_cost_comparison.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: H2 production
# ─────────────────────────────────────────────────────────────────────────────
pem_eta = cfg["agents"]["electrolyzer_pem"]["h2_production_kg_per_kwh"]
alk_eta = cfg["agents"]["electrolyzer_alk"]["h2_production_kg_per_kwh"]
pem_target_day = cfg["agents"]["electrolyzer_pem"]["h2_daily_target_kg"]
alk_target_day = cfg["agents"]["electrolyzer_alk"]["h2_daily_target_kg"]
n_days = gne["n_days"]

# Cumulative H2: u_i[t] * eta * dt_hr (kg per step)
h2_pem_gne  = (x_gne[:,  12] * pem_eta * DT_HR).cumsum()
h2_pem_admm = (x_admm[:, 12] * pem_eta * DT_HR).cumsum()
h2_alk_gne  = (x_gne[:,  18] * alk_eta * DT_HR).cumsum()
h2_alk_admm = (x_admm[:, 18] * alk_eta * DT_HR).cumsum()
target_pem  = pem_target_day / 288 * (steps + 1)   # linear target
target_alk  = alk_target_day / 288 * (steps + 1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(hours, h2_pem_gne,  color=COLORS["gne"],  lw=1.2, ls="-",  label="PEM GNE")
ax.plot(hours, h2_pem_admm, color=COLORS["gne"],  lw=1.2, ls="--", label="PEM ADMM", alpha=0.7)
ax.plot(hours, h2_alk_gne,  color=COLORS["admm"], lw=1.2, ls="-",  label="Alk GNE")
ax.plot(hours, h2_alk_admm, color=COLORS["admm"], lw=1.2, ls="--", label="Alk ADMM", alpha=0.7)
ax.plot(hours, target_pem,  color="navy",    lw=1.0, ls=":",  label=f"PEM target ({pem_target_day} kg/day)")
ax.plot(hours, target_alk,  color="sienna",  lw=1.0, ls=":",  label=f"Alk target ({alk_target_day} kg/day)")

# Annotate final values
for label, arr, target in [
    ("PEM GNE", h2_pem_gne,  pem_target_day * n_days),
    ("Alk GNE", h2_alk_gne,  alk_target_day * n_days),
]:
    pct = arr[-1] / target * 100
    ax.text(170, arr[-1], f"{arr[-1]:.0f} kg\n({pct:.1f}%)", fontsize=7)

ax.set_xlabel("Time [hours from Mon 2024-07-08]")
ax.set_ylabel("Cumulative H₂ Produced [kg]")
ax.set_title("Cumulative H₂ Production vs. Take-or-Pay Target")
ax.legend(ncol=2, fontsize=7)
ax.set_xlim(0, 168)
day_vlines(ax)
plt.tight_layout()
save("fig_h2_production.pdf")

print(f"\nAll 7 figures saved to {FIG_DIR}")