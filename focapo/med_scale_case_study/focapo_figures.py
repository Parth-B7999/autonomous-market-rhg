"""
focapo_figures.py — FOCAPO-CPC 2027 paper figures & table numbers.

Self-contained, cell-structured (#%%) script that loads the latest Individual
and GNE result runs and regenerates EVERY figure and table number used in the
manuscript (FOCAPO_Parth / FOCAPO_CPC_2027_Prism/main.tex).

Run cell-by-cell in VS Code / Spyder, or all at once:
    python focapo_figures.py

Outputs (PDF + PNG) are written to:
    results/comparison/focapo/         (working copies)
    FOCAPO_CPC_2027_Prism/figures/     (drop-in for the LaTeX source)

Figures produced
----------------
  fig1_weekly_aggregate   — whole-WEEK aggregate grid import, Individual vs GNE
                            (replaces the single-day Fig. 1)
  fig1b_aggregate_grid    — alternative 2x4 small-multiples (one panel per day)
  fig2_daily_curtailment  — daily fleet curtailment ONLY (right per-asset panel
                            removed; it lives in Table 2)
  fig3_mechanism          — NEW results figure: per-asset electrolyzer power
                            breakdown (renewable / battery / grid) for the four
                            renewable assets, Individual vs GNE, with curtailed
                            renewable shown as a red hatch — the "report Fig. 6"
                            style figure.

Table numbers (printed to stdout)
---------------------------------
  Table 1 — fleet parameters
  Table 2 — 7-day Individual vs GNE financials (LaTeX rows, ready to paste)
  Results-text key numbers (corrected Fig.1 caption values, %s, totals)
"""
# %% [0] ── Imports & global style ───────────────────────────────────────────
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

PROJ    = Path(__file__).resolve().parent
RESULTS = PROJ / "results"
OUT_DIRS = [RESULTS / "comparison" / "focapo",          # working copy
            PROJ / "Focapo Paper overleaf" / "figures"]  # paper figures
for d in OUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

# make local modules importable (agents_med, agents)
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
_PARENT = str(PROJ.parent / "admm_elz_market")
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from agents import DT_RT                                   # 5 min = 5/60 h
from agents_med import FLEET_DICTS, L_MIN, N_DAYS          # fleet + market params

# publication style — Times-like serif, vector text, tight
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         14,
    "axes.titlesize":    14,
    "axes.labelsize":    14,
    "legend.fontsize":   12.5,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
    "axes.linewidth":    0.8,
    "savefig.dpi":       300,
    "figure.dpi":        120,
    "pdf.fonttype":      42,   # editable text in the PDF
    "ps.fonttype":       42,
})

C_IND  = "#1f5f8b"   # Individual  (deep blue)
C_GNE  = "#e07b1a"   # GNE         (orange)
C_LMIN = "#c0392b"   # L_min line  (red)
DAYS   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
NSTEP  = 288         # 5-min steps per day


def _save(fig, name: str):
    """Save a figure as PDF into every output directory."""
    for d in OUT_DIRS:
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    print(f"  saved  {name}.pdf  ->  {', '.join(str(d) for d in OUT_DIRS)}")


# %% [1] ── Load latest Individual + GNE runs ────────────────────────────────
def _latest(prefix: str) -> Path:
    dirs = sorted(d for d in RESULTS.iterdir()
                  if d.is_dir() and d.name.startswith(prefix))
    if not dirs:
        raise FileNotFoundError(f"no '{prefix}*' run dir in {RESULTS}")
    return dirs[-1]


# Paper uses the July-8-2024 week. Pin those runs explicitly so the figures stay
# stable even when newer (e.g. stress-test) runs exist. Set PIN_* = None to fall
# back to auto-discovering the most recent runs.
PIN_IND = "individual_20260606_1133_Lmin100_7d"
PIN_GNE = "gne_20260606_1133_Lmin100_noLmax_7d"
IND_DIR = (RESULTS / PIN_IND) if PIN_IND else _latest("individual_")
GNE_DIR = (RESULTS / PIN_GNE) if PIN_GNE else _latest("gne_")
ind = dict(np.load(IND_DIR / "results.npz", allow_pickle=True))
gne = dict(np.load(GNE_DIR / "results.npz", allow_pickle=True))

AGENTS = [d["name"].replace("_M", "") for d in FLEET_DICTS]
N      = len(AGENTS)

print("Loaded runs:")
print(f"  Individual : {IND_DIR.name}")
print(f"  GNE        : {GNE_DIR.name}")
print(f"  agents     : {AGENTS}")


# %% [2] ── Financial / physical helpers ─────────────────────────────────────
def per_agent_financials(data: dict):
    """Return dicts of 7-day totals per agent: profit, cost, h2, curtail."""
    cost = np.array([float(data["cost_rt"][:, :, i].sum()) for i in range(N)])
    h2   = np.array([float(data["h2_daily"][:, i].sum())   for i in range(N)])
    rev  = np.array([float((data["p_elec"][:, :, i]
                            * FLEET_DICTS[i]["eta_kg_per_kwh"] * DT_RT
                            * FLEET_DICTS[i].get("r_h2", 3.0)).sum())
                     for i in range(N)])
    curt = np.array([float(data["curtail"][:, :, i].sum() * DT_RT) for i in range(N)])
    return dict(profit=rev - cost, cost=cost, rev=rev, h2=h2, curt=curt)


F_IND = per_agent_financials(ind)
F_GNE = per_agent_financials(gne)


# %% [3] ── FIGURE 1 — whole-WEEK aggregate grid import ───────────────────────
# Continuous 7-day strip: the recurring midday dips of the GNE coalition toward
# L_min are the story. Shaded band = import the coalition avoids vs. individual.
t_week = np.arange(N_DAYS * NSTEP) / NSTEP * 24          # hours 0..168
agg_ind = ind["agg"].reshape(-1)                         # (7*288,)
agg_gne = gne["agg"].reshape(-1)

# Figure 1 uses its own (larger) font sizes; other figures keep the global style.
with plt.rc_context({"axes.labelsize": 22, "xtick.labelsize": 20,
                     "ytick.labelsize": 20, "legend.fontsize": 20}):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(t_week, agg_gne, agg_ind, where=agg_ind >= agg_gne,
                    color="#9aa7b1", alpha=0.30, linewidth=0)
    ax.plot(t_week, agg_ind, color=C_IND, lw=0.7, alpha=0.9, label="Individual (Aggregated)")
    ax.plot(t_week, agg_gne, color=C_GNE, lw=0.7, alpha=0.95, label="GNE coalition")
    ax.axhline(L_MIN, color=C_LMIN, ls="--", lw=1.2, label=f"$L_{{\\min}}={L_MIN:.0f}$ kW")

    # day separators + labels
    for d in range(N_DAYS):
        if d:
            ax.axvline(d * 24, color="0.75", lw=0.6, ls=":")
        ax.text(d * 24 + 12, 1.02, DAYS[d], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=20, color="0.25")

    ax.set_xlim(0, N_DAYS * 24)
    ax.set_ylim(0, None)
    ax.set_xticks(np.arange(0, N_DAYS * 24 + 1, 12))
    ax.set_xticklabels([f"{h%24:02d}" for h in range(0, N_DAYS * 24 + 1, 12)])
    ax.set_xlabel("Hour of week", labelpad=8)
    ax.set_ylabel("Aggregate grid import [kW]")
    ax.margins(x=0)
    # legend placed clearly below the x-axis label (no overlap)
    fig.subplots_adjust(bottom=0.34)
    ax.legend(ncol=4, loc="upper center", frameon=True, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.30))
    # _save(fig, "fig1_weekly_aggregate")
    plt.show()


# %% [4] ── FIGURE 2 — daily fleet curtailment ONLY (right panel removed) ─────
curt_ind_day = ind["curtail"].sum(axis=(1, 2)) * DT_RT   # kWh/day
curt_gne_day = gne["curtail"].sum(axis=(1, 2)) * DT_RT

fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(N_DAYS)
b1 = ax.bar(x - 0.19, curt_ind_day, 0.38, color=C_IND,
            label="Individual", edgecolor="white", linewidth=0.5)
b2 = ax.bar(x + 0.19, curt_gne_day, 0.38, color=C_GNE,
            label="GNE coalition", edgecolor="white", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(DAYS)
ax.set_xlabel("Day")
ax.set_ylabel("Curtailed renewable energy [kWh]")
ax.grid(axis="x", alpha=0)
ax.legend(frameon=True, framealpha=0.9)
tot_i, tot_g = curt_ind_day.sum(), curt_gne_day.sum()
# ax.text(0.97, 0.95,
#         f"7-day total\nIndividual: {tot_i:,.0f} kWh\nGNE: {tot_g:,.0f} kWh"
#         f"  ($-${(1-tot_g/tot_i)*100:.0f}%)",
#         transform=ax.transAxes, ha="right", va="top", fontsize=12,
#         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))
fig.tight_layout()
# _save(fig, "fig2_daily_curtailment")
plt.show()


# %% [5] ── FIGURE 3 (NEW) — mechanism: per-asset power breakdown ─────────────
# "Report Fig. 6" style. Renewable assets only (where the mechanism acts).
# Where does each electrolyzer's power come from?  Stacked sources that sum
# EXACTLY to the converter load (black line):
#     renewable used (gold) + battery discharge (green) + grid import (gray)
# Curtailed renewable is drawn as a red hatch CAP above the load line (wasted
# generation).  Colours are identical in both rows; rows are labelled on the
# y-axis (Individual vs GNE), so the legend matches the plot exactly.
REN_AGENTS = [i for i, d in enumerate(FLEET_DICTS)
              if d.get("renewable_cap_kw", 0) > 0]            # PV/Wind assets
DAY = 0 # which day to plot:  0=Mon 1=Tue 2=Wed 3=Thu 4=Fri 5=Sat 6=Sun
t = np.arange(NSTEP) / NSTEP * 24

C_REN  = "#f2b705"   # renewable used   (bright gold — the message)
C_BAT  = "#2e9e4f"   # battery discharge (green, positive)
C_CHG  = "#6a4c93"   # battery charge    (purple, negative)
C_GRID = "#9aa7b1"   # grid import      (neutral gray — recedes)
C_CURT = "#e03131"   # curtailed / wasted renewable (red hatch)

# per-column y-limits (shared by both rows of an asset, for fair comparison)
ycol_top, ycol_bot = [], []
for i in REN_AGENTS:
    top = max(float((d["p_elec"][DAY, :, i] + d["curtail"][DAY, :, i]).max())
              for d in (ind, gne))
    chg = max(float(d["p_ch"][DAY, :, i].max()) for d in (ind, gne))
    ycol_top.append(max(top, FLEET_DICTS[i]["p_elec_max_kw"]) * 1.15)
    ycol_bot.append(-chg * 1.20 if chg > 1e-6 else -0.02 * ycol_top[-1])

fig, axes = plt.subplots(2, len(REN_AGENTS), figsize=(4.2 * len(REN_AGENTS), 7.0),
                         sharex=True)
for row, (label, data) in enumerate([("Individual", ind), ("GNE coalition", gne)]):
    for col, i in enumerate(REN_AGENTS):
        ax = axes[row, col]
        curt    = data["curtail"][DAY, :, i]
        bat_dis = data["p_dis"][DAY, :, i]
        bat_ch  = data["p_ch"][DAY, :, i]
        grid    = data["p_net"][DAY, :, i]
        load    = data["p_elec"][DAY, :, i]
        # renewable that directly fed the converter (energy balance, exact):
        ren_to_load = np.clip(load - grid - bat_dis, 0, None)
        pmax = FLEET_DICTS[i]["p_elec_max_kw"]

        l1 = ren_to_load
        l2 = l1 + bat_dis
        l3 = l2 + grid                                   # == load
        ax.fill_between(t, 0,  l1, color=C_REN,  lw=0, label="Renewable used")
        if FLEET_DICTS[i].get("bat_cap_kwh", 0) > 0:
            ax.fill_between(t, l1, l2, color=C_BAT, lw=0, label="Battery discharge")
            # battery charge drawn below the axis (power leaving the bus)
            ax.fill_between(t, 0, -bat_ch, color=C_CHG, lw=0, label="Battery charge")
        ax.fill_between(t, l2, l3, color=C_GRID, lw=0, label="Grid import")
        # curtailed renewable = red hatch cap sitting above the converter load
        ax.fill_between(t, load, load + curt, facecolor="none", edgecolor=C_CURT,
                        hatch="////", lw=0.0, label="Curtailed (wasted)")
        ax.plot(t, load, color="k", lw=1.1, label="Converter load")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axhline(pmax, color="0.35", ls=":", lw=1.0)

        ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylim(ycol_bot[col], ycol_top[col])
        ax.grid(axis="x", alpha=0)
        ax.tick_params(labelsize=16)
        if row == 0:
            _rt = (FLEET_DICTS[i].get("renewable_type") or "").upper()
            _rt = {"PV": "PV", "WIND": "Wind"}.get(_rt, _rt)
            _desc = _rt + (" + Battery" if FLEET_DICTS[i].get("bat_cap_kwh", 0) else "")
            ax.set_title(f"Agent {i+1} ({_desc})", fontsize=16, fontweight="bold")
        if row == 1:
            ax.set_xlabel("Hour of day", fontsize=18)
        if col == 0:
            ax.set_ylabel(f"{label}\nPower [kW]", fontsize=18, fontweight="bold")

# one shared legend — colours match the plot exactly
leg = [Patch(fc=C_REN,  label="Renewable used"),
       Patch(fc=C_BAT,  label="Battery discharge"),
       Patch(fc=C_CHG,  label="Battery charge ($-$)"),
       Patch(fc=C_GRID, label="Grid import"),
       Patch(fc="none", ec=C_CURT, hatch="////", label="Curtailed (wasted)"),
       Line2D([], [], color="k",    lw=1.1, label="Converter load"),
       Line2D([], [], color="0.35", ls=":", lw=1.0, label="$P^{\\max}$")]
# reserve top space first, then place the legend in it (no overlap with titles)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.legend(handles=leg, ncol=7, loc="upper center", bbox_to_anchor=(0.5, 0.995),
           frameon=True, framealpha=0.9, fontsize=14)
_save(fig, "fig3_mechanism")
plt.show()


# %% [6] ── TABLE 1 — fleet parameters (print) ───────────────────────────────
print("\n" + "=" * 72)
print("TABLE 1 — Fleet parameters")
print("=" * 72)
print(f"{'Asset':14s} {'Tech':5s} {'Pmax[kW]':>9s} {'eta[kg/kWh]':>12s} "
      f"{'Ren[kW]':>10s} {'Bat[kWh]':>9s}")
for d in FLEET_DICTS:
    tech = "ALK" if "ALK" in d["name"] else "PEM"
    rt   = (d.get("renewable_type") or "").upper()
    ren  = f"{d['renewable_cap_kw']:.0f} ({rt})" if d.get("renewable_cap_kw") else "—"
    bat  = f"{d.get('bat_cap_kwh', 0):.0f}" if d.get("bat_cap_kwh") else "—"
    print(f"{d['name'].replace('_M',''):14s} {tech:5s} {d['p_elec_max_kw']:>9.0f} "
          f"{d['eta_kg_per_kwh']:>12.3f} {ren:>10s} {bat:>9s}")


# %% [7] ── TABLE 2 — 7-day financials, LaTeX rows (print) ────────────────────
print("\n" + "=" * 72)
print("TABLE 2 — 7-day Individual vs GNE  (LaTeX body rows, paste into main.tex)")
print("=" * 72)


def _f(v):           # integer with thousands separator
    return f"{v:,.0f}"


for i in range(N):
    name = AGENTS[i].replace("_", r"\_")
    print(f"    {name:14s} & "
          f"{_f(F_IND['profit'][i]):>6s} & {_f(F_IND['cost'][i]):>6s} & "
          f"{_f(F_IND['h2'][i]):>5s} & {_f(F_IND['curt'][i]):>6s} & "
          f"{_f(F_GNE['profit'][i]):>6s} & {_f(F_GNE['cost'][i]):>6s} & "
          f"{_f(F_GNE['h2'][i]):>5s} & {_f(F_GNE['curt'][i]):>6s} \\\\")

tot = lambda F, k: F[k].sum()
print(r"    \midrule")
print(f"    \\textbf{{Fleet Total}} & "
      f"\\textbf{{{_f(tot(F_IND,'profit'))}}} & \\textbf{{{_f(tot(F_IND,'cost'))}}} & "
      f"\\textbf{{{_f(tot(F_IND,'h2'))}}} & \\textbf{{{_f(tot(F_IND,'curt'))}}} & "
      f"\\textbf{{{_f(tot(F_GNE,'profit'))}}} & \\textbf{{{_f(tot(F_GNE,'cost'))}}} & "
      f"\\textbf{{{_f(tot(F_GNE,'h2'))}}} & \\textbf{{{_f(tot(F_GNE,'curt'))}}} \\\\")


def _pct(a, b):      # % change b vs a
    return (b - a) / abs(a) * 100


print(f"    GNE vs. Ind.  ->  profit {_pct(tot(F_IND,'profit'), tot(F_GNE,'profit')):+.0f}%   "
      f"cost {_pct(tot(F_IND,'cost'), tot(F_GNE,'cost')):+.0f}%   "
      f"H2 {_pct(tot(F_IND,'h2'), tot(F_GNE,'h2')):+.0f}%   "
      f"curt {_pct(tot(F_IND,'curt'), tot(F_GNE,'curt')):+.0f}%")


# %% [8] ── Key numbers for the Results text (corrected) ──────────────────────
print("\n" + "=" * 72)
print("RESULTS-TEXT KEY NUMBERS  (use these to correct the manuscript)")
print("=" * 72)

# Fig.1 caption — Day-1 average aggregate import (CORRECTED)
a1_ind = float(ind["agg"][0].mean())
a1_gne = float(gne["agg"][0].mean())
print(f"[Fig.1] Day-1 avg aggregate import: "
      f"{a1_ind:.1f} kW -> {a1_gne:.1f} kW  ({_pct(a1_ind, a1_gne):+.0f}%)")
aw_ind = float(ind["agg"].mean()); aw_gne = float(gne["agg"].mean())
print(f"[Fig.1] WEEK avg aggregate import:  "
      f"{aw_ind:.1f} kW -> {aw_gne:.1f} kW  ({_pct(aw_ind, aw_gne):+.0f}%)")

# Fig.2 / curtailment
ci, cg = F_IND["curt"].sum(), F_GNE["curt"].sum()
print(f"[Fig.2] 7-day curtailment: {ci:,.0f} kWh -> {cg:,.0f} kWh "
      f"({_pct(ci, cg):+.0f}%)")
for i in REN_AGENTS:
    print(f"        {AGENTS[i]:14s}: {F_IND['curt'][i]:>7.0f} -> "
          f"{F_GNE['curt'][i]:>6.0f} kWh  ({_pct(max(F_IND['curt'][i],1e-9), F_GNE['curt'][i]):+.0f}%)")

# Economics
print(f"[Tab.2] Fleet profit: ${tot(F_IND,'profit'):,.0f} -> ${tot(F_GNE,'profit'):,.0f} "
      f"({_pct(tot(F_IND,'profit'), tot(F_GNE,'profit')):+.0f}%)")
print(f"[Tab.2] Fleet cost:   ${tot(F_IND,'cost'):,.0f} -> ${tot(F_GNE,'cost'):,.0f} "
      f"({_pct(tot(F_IND,'cost'), tot(F_GNE,'cost')):+.0f}%)")
print(f"[Tab.2] Fleet H2:     {tot(F_IND,'h2'):,.0f} -> {tot(F_GNE,'h2'):,.0f} kg "
      f"({_pct(tot(F_IND,'h2'), tot(F_GNE,'h2')):+.0f}%)")
print("        per-asset profit change [%]:")
for i in range(N):
    print(f"        {AGENTS[i]:14s}: {_pct(F_IND['profit'][i], F_GNE['profit'][i]):+.0f}%")
pc = [_pct(F_IND['profit'][i], F_GNE['profit'][i]) for i in range(N)]
print(f"        range: +{min(pc):.0f}% to +{max(pc):.0f}%  (all assets better off: "
      f"{all(p > 0 for p in pc)})")

print("\nDone.")

# %%
