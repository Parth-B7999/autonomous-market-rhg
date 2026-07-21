"""Generate FOCAPO-style result figures for the mp-GNE FACET coalition (both weeks)."""
import sys, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "simple_game")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import rhg_mpqp as R, rhg_online as O, dam as F
import rhg_week as W
from pathlib import Path

OUT = Path("results/figures/report"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "font.size": 12,
                     "axes.titlesize": 13, "savefig.dpi": 200, "axes.titleweight": "bold"})
N, H, DT = R.N, R.H, R.DT
names = [R.FLEET[i][0] for i in range(N)]
LMIN, LMAX = R.L_MIN, R.L_MAX
C_REN, C_GRID, C_CURT = "#f2b705", "#9aa7b1", "#e03131"
sols, game = O.load_and_prepare()
WEEKS = {"April 1-7":  ["2025-04-0%d"%d for d in range(1,8)],
         "July 7-13":  ["2025-07-%02d"%d for d in range(7,14)]}

# ── run both weeks (cache to pkl so figure tweaks don't re-run) ──────────────
import pickle
CACHE = Path("results/report_fig_data.pkl")
if CACHE.exists():
    data = pickle.load(open(CACHE, "rb")); print("loaded cached run data", flush=True)
else:
    data = {}
    for wk, days in WEEKS.items():
        rng = np.random.default_rng(F.SEED)
        rows = []
        for day in days:
            print(f"Running {wk} - {day}", flush=True)
            r = W.run_day(day, sols, game, rng, verbose=True)
            lam_da, lam_rt, cs_h, cw_h = F.load(day)
            r["cs_h"], r["cw_h"] = cs_h, cw_h
            rows.append(r)
        data[wk] = rows
        print(f"{wk}: done", flush=True)
    pickle.dump(data, open(CACHE, "wb"))

def gstep(r, i, t):
    ty, rc = R._typ(i), R._rcap(i)
    if ty == "grid": return 0.0
    h = t // 4
    return (r["cs_h"][h] if ty == "solar" else r["cw_h"][h]) * rc

# ── FIG 1: weekly aggregate grid import + DAM/RT prices, both weeks ───────────
fig, axes = plt.subplots(3, 2, figsize=(14, 8.5), sharex="col")
for c, (wk, rows) in enumerate(data.items()):
    agg = np.concatenate([rows[d]["P"].sum(1) for d in range(7)])
    lam_rt = np.concatenate([rows[d]["lam_rt"] for d in range(7)])
    lam_da = np.concatenate([rows[d]["lam_da"] for d in range(7)])
    hrs = np.arange(len(agg)) * 0.25
    hrs_da = np.arange(len(lam_da))
    
    axes[0, c].plot(hrs_da, lam_da, color="#9400D3", lw=1.2); axes[0, c].set_yscale("symlog")
    axes[0, c].axhline(60, color="r", ls=":", lw=0.8, label="H$_2$ break-even $60")
    axes[0, c].set_title(f"{wk}: DAM price"); axes[0, c].set_ylabel("$\\lambda_{DA}$ [\\$/MWh]")
    axes[0, c].legend(fontsize=9, loc="upper right")
    
    axes[1, c].plot(hrs, lam_rt, color="#333", lw=0.7); axes[1, c].set_yscale("symlog")
    axes[1, c].axhline(60, color="r", ls=":", lw=0.8, label="H$_2$ break-even $60")
    axes[1, c].set_title(f"{wk}: RT price"); axes[1, c].set_ylabel("$\\lambda_{RT}$ [\\$/MWh]")
    axes[1, c].legend(fontsize=9, loc="upper right")
    
    axes[2, c].plot(hrs, agg, color="#1f6feb", lw=0.8, label="Coalition $\\Sigma p$")
    axes[2, c].axhline(LMAX, color="purple", ls="--", lw=1, label=f"$L_{{max}}$={LMAX:.0f}")
    axes[2, c].axhline(LMIN, color="green", ls="--", lw=1, label=f"$L_{{min}}$={LMIN:.0f}")
    axes[2, c].set_title(f"{wk}: coalition aggregate grid import (FACET)")
    axes[2, c].set_ylabel("$\\Sigma_i p_i$ [kW]"); axes[2, c].set_xlabel("hour of week")
    axes[2, c].set_ylim(0, LMAX*1.08); axes[2, c].legend(fontsize=9, ncol=3, loc="lower center")
    
    for d in range(1, 7):
        axes[0, c].axvline(d*24, color="0.8", lw=0.5, ls=":")
        axes[1, c].axvline(d*24, color="0.8", lw=0.5, ls=":")
        axes[2, c].axvline(d*24, color="0.8", lw=0.5, ls=":")
fig.tight_layout(); fig.savefig(OUT/"fig1_weekly_aggregate.pdf"); fig.savefig(OUT/"fig1_weekly_aggregate.png"); plt.close(fig)

# ── FIG 2: per-agent weekly H2 produced vs target, both weeks ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for c, (wk, rows) in enumerate(data.items()):
    h2 = sum(rows[d]["h2"] for d in range(7)); tg = sum(rows[d]["d_day"] for d in range(7))
    x = np.arange(N)
    axes[c].bar(x-0.2, h2, 0.4, color="#1f6feb", label="produced", edgecolor="w")
    axes[c].bar(x+0.2, tg, 0.4, color="#c0c0c0", label="target (floor)", edgecolor="w")
    for xi in x: axes[c].text(xi-0.2, h2[xi], f"{100*h2[xi]/tg[xi]:.0f}%", ha="center", va="bottom", fontsize=8)
    axes[c].set_xticks(x); axes[c].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[c].set_ylabel("H$_2$ [kg/week]"); axes[c].set_title(f"{wk}: weekly H$_2$ per agent")
    axes[c].legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"fig2_h2_per_agent.pdf"); fig.savefig(OUT/"fig2_h2_per_agent.png"); plt.close(fig)

# ── FIG 3: daily curtailment + per-agent curtailment ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for c, (wk, rows) in enumerate(data.items()):
    daily = [rows[d]["curt"].sum() for d in range(7)]
    axes[0].bar(np.arange(7)+c*0.4-0.2, daily, 0.4, label=wk, edgecolor="w")
axes[0].set_xticks(range(7)); axes[0].set_xticklabels([f"D{d+1}" for d in range(7)])
axes[0].set_ylabel("curtailed renewable [kWh/day]"); axes[0].set_title("Daily fleet curtailment"); axes[0].legend(fontsize=9)
ren = [i for i in range(N) if R._typ(i) != "grid"]
for c, (wk, rows) in enumerate(data.items()):
    curt = sum(rows[d]["curt"] for d in range(7)); avail = sum(rows[d]["ren_avail"] for d in range(7))
    frac = [100*curt[i]/max(avail[i],1) for i in ren]
    axes[1].bar(np.arange(len(ren))+c*0.4-0.2, frac, 0.4, label=wk, edgecolor="w")
axes[1].set_xticks(range(len(ren))); axes[1].set_xticklabels([names[i] for i in ren], rotation=25, ha="right", fontsize=9)
axes[1].set_ylabel("curtailed / available [%]"); axes[1].set_title("Per-agent curtailment fraction"); axes[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"fig3_curtailment.pdf"); fig.savefig(OUT/"fig3_curtailment.png"); plt.close(fig)

# ── FIG 4: mechanism — per-agent power breakdown, representative day ──────────
# 04-06: coalition swings ceiling↔floor (λ −$1..$234), retreating to L_min when expensive
wk = "April 1-7"; rows = data[wk]; DAYIDX = 5; r = rows[DAYIDX]
t = np.arange(96) * 0.25
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))   # no sharex: mechanism panels keep hour ticks
order = [0, 1, 2, 4]  # PEM_Elec(grid), ALK(grid), PEM_PV, PEM_Wind (1 of each pair)
titles = ["PEM_Elec (grid)", "ALK (grid)", "PEM_PV (solar)", "PEM_Wind (wind)"]
panel = [(0,0),(0,1),(1,0),(1,1)]
for (rr,cc), i, ttl in zip(panel, order, titles):
    ax = axes[rr, cc]; pmax = R._pmax(i)
    grid = r["P"][:, i]
    g = np.array([gstep(r, i, tt) for tt in range(96)])
    pe = np.clip(grid + g, 0, pmax)
    ren_used = np.clip(pe - grid, 0, None)
    curt = np.maximum(0, grid + g - pmax)
    ax.fill_between(t, 0, ren_used, color=C_REN, lw=0, label="Renewable used")
    ax.fill_between(t, ren_used, ren_used+grid, color=C_GRID, lw=0, label="Grid import")
    ax.fill_between(t, pe, pe+curt, facecolor="none", edgecolor=C_CURT, hatch="////", lw=0, label="Curtailed (wasted)")
    ax.plot(t, pe, "k", lw=1.1, label="Converter load")
    ax.axhline(pmax, color="0.35", ls=":", lw=1)
    ax.set_title(ttl); ax.set_xlim(0, 24); ax.set_xticks([0,6,12,18,24]); ax.set_ylim(0, pmax*1.15)
    if cc == 0: ax.set_ylabel("power [kW]")
    if rr == 1: ax.set_xlabel("hour of day")
# aggregate panel (bottom-right spanning) + FACET accuracy (top-right)
axes[0,2].plot(t, r["P"].sum(1), color="#1f6feb", lw=1, label="$\\Sigma p$")
axes[0,2].axhline(LMAX, color="purple", ls="--", lw=1); axes[0,2].axhline(LMIN, color="green", ls="--", lw=1)
axes[0,2].set_title(f"Coalition $\\Sigma p$ ({WEEKS[wk][DAYIDX]})"); axes[0,2].set_ylim(0, LMAX*1.08); axes[0,2].set_ylabel("kW")
axes[0,2].set_xlim(0, 24); axes[0,2].set_xticks([0,6,12,18,24]); axes[0,2].set_xlabel("hour of day")
# FACET accuracy + fallbacks across both weeks
alld = [(wkn, dd, rows2[dd]) for wkn, rows2 in data.items() for dd in range(7)]
ec = [x[2]["e_cent"] for x in alld]; fb = [x[2]["comm"].get("fallback",0) for x in alld]
labels = [f"{'A' if 'Apr' in x[0] else 'J'}{x[1]+1}" for x in alld]
ax = axes[1,2]; ax.set_yscale("log")
ax.bar(range(len(ec)), np.maximum(ec, 1e-6), color="#1f6feb", label="map$=$cent [kW]")
ax.axhline(1e-3, color="green", ls="--", lw=0.8, label="1e-3 kW")
ax.set_xticks(range(len(ec))); ax.set_xticklabels(labels, fontsize=7, rotation=90)
ax.set_ylabel("map$=$centralized [kW]"); ax.set_title("FACET accuracy per day (A=Apr J=Jul)")
ax2 = ax.twinx(); ax2.plot(range(len(fb)), fb, "rs-", ms=4, lw=1, label="ADMM fallbacks"); ax2.set_ylabel("fallbacks /96", color="r"); ax2.grid(False)
h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2.get_legend_handles_labels(); ax.legend(h1+h2, l1+l2, fontsize=8, loc="upper left")
handles = [Patch(fc=C_REN,label="Renewable used"), Patch(fc=C_GRID,label="Grid import"),
           Patch(fc="none",ec=C_CURT,hatch="////",label="Curtailed"), Line2D([],[],color="k",lw=1.1,label="Converter load")]
fig.legend(handles=handles, ncol=4, loc="lower center", fontsize=10, bbox_to_anchor=(0.5,-0.02))
fig.suptitle(f"Per-agent power balance, coalition on {WEEKS[wk][DAYIDX]} "
             f"(evening retreat to $L_{{min}}$=100 kW) + FACET accuracy", y=1.0)
fig.tight_layout(rect=[0,0.03,1,1]); fig.savefig(OUT/"fig4_mechanism.pdf"); fig.savefig(OUT/"fig4_mechanism.png"); plt.close(fig)

# ── FIG 5: Prices (DAM & RTM) vs Forecasts ────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
wk = "April 1-7"; DAYIDX = 5; r = data[wk][DAYIDX]

ax1 = plt.subplot(2, 2, 1)
ax1.plot(np.arange(24), r["lam_da"], 'k-', label="Actual DAM Price", lw=1.5)
ax1.plot(np.arange(24), r["lam_da_fc"], 'r--', label="Forecast DAM Price", lw=1.5)
ax1.set_title(f"DAM Prices on {WEEKS[wk][DAYIDX]}")
ax1.set_xlabel("Hour of day")
ax1.set_ylabel("Price [$/MWh]")
ax1.legend()

ax2 = plt.subplot(2, 2, 2)
t_rtm = np.arange(96) * 0.25
ax2.plot(t_rtm, r["lam_rt"], 'k-', label="Actual RTM Price", lw=1.5)
ax2.plot(t_rtm, r["lam_rt_fc"], 'b--', label="Nowcast RTM Price (Step 0)", lw=1.5)
ax2.set_title(f"RTM Prices on {WEEKS[wk][DAYIDX]}")
ax2.set_xlabel("Hour of day")
ax2.set_ylabel("Price [$/MWh]")
ax2.legend()

ax3 = plt.subplot(2, 1, 2)
lam_da_week = np.concatenate([data[wk][d]["lam_da"] for d in range(7)])
lam_da_fc_week = np.concatenate([data[wk][d]["lam_da_fc"] for d in range(7)])
t_week = np.arange(7 * 24)
ax3.plot(t_week, lam_da_week, 'k-', label="Actual DAM Price", lw=1.5)
ax3.plot(t_week, lam_da_fc_week, 'r--', label="Forecast DAM Price", lw=1.5)
ax3.set_title(f"DAM Prices for {wk}")
ax3.set_xlabel("Hour of week")
ax3.set_ylabel("Price [$/MWh]")
ax3.legend()

fig.tight_layout(); fig.savefig(OUT/"fig5_prices.pdf"); fig.savefig(OUT/"fig5_prices.png"); plt.close(fig)

print("FIGURES →", OUT)
