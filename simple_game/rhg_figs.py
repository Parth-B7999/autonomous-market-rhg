"""rhg_figs.py — aggregates + figures from the receding-horizon week run."""
import sys, pickle
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
import rhg_mpqp as R
res = pickle.load(open(HERE.parent / "results" / "rhg_week_results.pkl", "rb"))
percru = [507, 507, 8359, 8359, 7335, 7335]
KN = float(np.prod([float(c) for c in percru]))
allc = Counter()
for r in res:
    for k, v in r["crhits"].items():
        allc[k] += v
h2 = sum(r["h2"] for r in res); tg = sum(r["d_day"] for r in res)
emax = max(r["e_cent"] for r in res)
tms = np.mean([1e3 * r["t_on"] / 96 for r in res])
print("=== WEEK AGGREGATES ===")
print(f"per-agent CRs: {percru}   K^N (combos possible) = {KN:.2e}")
print(f"distinct GNE combos used all week = {len(allc)}   (of {KN:.1e})")
print(f"map==centralized max over week = {emax:.2e} kW ; misses = {sum(r['miss'] for r in res)}")
print(f"H2 week = {h2.sum():.0f}/{tg.sum():.0f} kg = {100*h2.sum()/tg.sum():.0f}%")
print(f"online FACET mean = {tms:.1f} ms/step")
# fig 1: week price + aggregate dispatch
lam = np.concatenate([r["lam_rt"] for r in res]); agg = np.concatenate([r["P"].sum(1) for r in res])
hh = np.arange(len(lam)) * 0.25 / 24
fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
_span = f"{res[0]['day']} – {res[-1]['day']}" if res else ""
ax[0].plot(hh, lam, lw=0.7); ax[0].axhline(60, color="r", ls=":", lw=0.8); ax[0].set_yscale("symlog")
ax[0].set_ylabel("λ_RT [$/MWh]"); ax[0].set_title(f"ERCOT RTM price (symlog), {_span}")
ax[1].plot(hh, agg, lw=0.8, color="tab:blue"); ax[1].axhline(R.L_MIN, color="g", ls="--", lw=0.8, label="L_min=100")
ax[1].axhline(R.L_MAX, color="purple", ls="--", lw=0.8, label="L_max=900")
ax[1].set_ylabel("Σp [kW]"); ax[1].set_xlabel("day"); ax[1].legend(fontsize=8)
ax[1].set_title("Coalition aggregate grid import — H=4 receding-horizon mp-GNE (online FACET)")
fig.tight_layout(); fig.savefig(HERE.parent / "results" / "figures" / "rhg_week_dispatch.png", dpi=120); plt.close(fig)
# fig 2: combo recurrence
fig, ax = plt.subplots(figsize=(9, 4))
top = sorted(allc, key=lambda k: -allc[k])[:30]
ax.bar(range(len(top)), [allc[k] for k in top], color="teal")
ax.set_xlabel("distinct GNE combination (rank)"); ax.set_ylabel("5-min visits over the week")
ax.set_title(f"Combo recurrence — {len(allc)} distinct GNE combos used of K^N≈{KN:.0e} possible\n"
             f"(FACET explores a tiny recurring set; exhaustive enumeration is intractable)")
fig.tight_layout(); fig.savefig(HERE.parent / "results" / "figures" / "rhg_recurrence.png", dpi=120); plt.close(fig)
# fig 3: H2 per agent
fig, ax = plt.subplots(figsize=(9, 4)); xr = np.arange(6)
ax.bar(xr - 0.2, h2, 0.4, label="produced"); ax.bar(xr + 0.2, tg, 0.4, label="target")
ax.set_xticks(xr); ax.set_xticklabels([f[0] for f in R.FLEET], rotation=25, fontsize=8)
ax.set_ylabel("H2 [kg/week]"); ax.legend(); ax.set_title("Weekly H2: produced vs contract target (receding-horizon tracking)")
fig.tight_layout(); fig.savefig(HERE.parent / "results" / "figures" / "rhg_h2.png", dpi=120); plt.close(fig)
print("figures → results/figures/rhg_*.png")
