"""
year_report.py — figures (PDF) + LaTeX tables for the full-year (ERCOT 2025) run.

Reads the 362 per-day files in results/year2025/ and emits:
  results/figures/year_rounds.pdf     grouped bars: FACET vs ADMM median rounds per season
  results/figures/year_scatter.pdf    daily FACET reach vs binding fraction (+ trend, r)
  results/figures/year_binding.pdf    monthly mean binding steps/day
  results/year_monthly_table.tex      per-month ledger (\\input-able)
  results/year_seasonal_table.tex     per-season summary + renewable driver

Usage:  python simple_game/year_report.py
"""
from __future__ import annotations
import pickle, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
YDIR = HERE.parent / "results" / "year2025"
FIG = HERE.parent / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
RES = HERE.parent / "results"
SEASON = {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
SNAME = {"DJF":"Winter","MAM":"Spring","JJA":"Summer","SON":"Fall"}
MON = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
FACET, ADMM, BIND, HOT = "#2a5fa8", "#e0662f", "#5b8fcf", "#c0392b"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555", "axes.linewidth": 0.8, "figure.dpi": 150,
})


def load():
    daily = []
    for p in sorted(glob.glob(str(YDIR / "bench_day=*.pkl"))):
        day = re.search(r"(\d{4}-\d{2}-\d{2})", p).group(1); mon = int(day[5:7])
        d = pickle.load(open(p, "rb")); s, b = d["sim"], d["bench"]; warm = b["blocks"]["warm"]
        fac_r = np.array([r["facet"]["rounds"] for r in warm]); adm_r = np.array([r["admm"]["rounds"] for r in warm])
        fac_t = np.array([r["facet"]["t_wall"] for r in warm]) * 1e3; adm_t = np.array([r["admm"]["t_wall"] for r in warm]) * 1e3
        fac_ok = np.array([np.isfinite(r["facet"]["t_tol"]) for r in warm]); adm_ok = np.array([np.isfinite(r["admm"]["t_tol"]) for r in warm])
        daily.append(dict(day=day, mon=mon, season=SEASON[mon], binding=int(b["binding"].sum()),
            h2=float(100*s["h2"].sum()/s["d_day"].sum()), ren=float(s["ren_avail"].sum()),
            curt=float(s["curt"].sum()), fb=int(s["comm"].get("fallback", 0)),
            fac_reach=float(fac_ok.mean()), adm_reach=float(adm_ok.mean()),
            fac_ms=float(np.median(fac_t)), adm_ms=float(np.median(adm_t)),
            fac_rnd=float(np.median(fac_r)), adm_rnd=float(np.median(adm_r))))
    return daily


def grp(rs):
    return dict(days=len(rs), bind=np.mean([r["binding"] for r in rs]),
        h2=np.mean([r["h2"] for r in rs]), curt=sum(r["curt"] for r in rs)/1000,
        ren=np.mean([r["ren"] for r in rs]), fb=sum(r["fb"] for r in rs),
        fac_reach=100*np.mean([r["fac_reach"] for r in rs]), adm_reach=100*np.mean([r["adm_reach"] for r in rs]),
        fac_ms=np.median([r["fac_ms"] for r in rs]), adm_ms=np.median([r["adm_ms"] for r in rs]),
        fac_rnd=np.median([r["fac_rnd"] for r in rs]), adm_rnd=np.median([r["adm_rnd"] for r in rs]))


def main():
    daily = load()
    months = {m: grp([r for r in daily if r["mon"] == m]) for m in range(1, 13)}
    seasons = {s: grp([r for r in daily if r["season"] == s]) for s in ["DJF", "MAM", "JJA", "SON"]}
    year = grp(daily); tot_fb = sum(r["fb"] for r in daily); tot = len(daily)*96
    year["iterfree"] = 100*(1-tot_fb/tot)
    b = np.array([r["binding"] for r in daily]); fr = 100*np.array([r["fac_reach"] for r in daily])
    r_bf = np.corrcoef(b, fr)[0, 1]; r_rb = np.corrcoef([r["ren"] for r in daily], b)[0, 1]

    # ── Fig 1: rounds by season ──
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    sk = ["DJF", "MAM", "JJA", "SON"]; xs = np.arange(4); w = 0.38
    fv = [seasons[s]["fac_rnd"] for s in sk]; av = [seasons[s]["adm_rnd"] for s in sk]
    ax.bar(xs-w/2, fv, w, color=FACET, label="FACET-RHG")
    ax.bar(xs+w/2, av, w, color=ADMM, label="ADMM")
    for x, v in zip(xs-w/2, fv): ax.text(x, v+1.2, f"{v:.0f}", ha="center", fontsize=8)
    for x, v in zip(xs+w/2, av): ax.text(x, v+1.2, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([f"{SNAME[s]}\n{s}" for s in sk])
    ax.set_ylabel("median inter-agent rounds / step"); ax.set_ylim(0, 78)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG/"year_rounds.pdf"); plt.close(fig)

    # ── Fig 2: scatter binding vs reach ──
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    oth = [r for r in daily if r["mon"] != 9]; sep = [r for r in daily if r["mon"] == 9]
    ax.scatter([r["binding"] for r in oth], [100*r["fac_reach"] for r in oth], s=15, c=FACET, alpha=.35, edgecolors="none", label="each day")
    ax.scatter([r["binding"] for r in sep], [100*r["fac_reach"] for r in sep], s=22, c=HOT, alpha=.9, edgecolors="none", label="September")
    xx = np.array([0, b.max()]); m1, m0 = np.polyfit(b, fr, 1)
    ax.plot(xx, m0+m1*xx, "--", c="#333", lw=1.4)
    ax.text(.96, .93, f"$r={r_bf:.2f}$", transform=ax.transAxes, ha="right", fontsize=11)
    ax.set_xlabel("ceiling-binding steps per day (of 96)"); ax.set_ylabel("FACET reach (\\% steps at $10^{-4}$ kW)")
    ax.set_ylim(0, 100); ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG/"year_scatter.pdf"); plt.close(fig)

    # ── Fig 3: monthly binding ──
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    xs = np.arange(1, 13); cols = [HOT if m == 9 else BIND for m in xs]
    ax.bar(xs, [months[m]["bind"] for m in xs], color=cols, width=.68)
    ax.set_xticks(xs); ax.set_xticklabels([MON[m] for m in xs])
    ax.set_ylabel("mean binding steps / day"); ax.axhline(year["bind"], ls=":", c="#888", lw=1)
    ax.text(12.4, year["bind"], f" year {year['bind']:.0f}", va="center", fontsize=8, c="#666")
    fig.tight_layout(); fig.savefig(FIG/"year_binding.pdf"); plt.close(fig)

    # ── Table: monthly ledger ──
    L = [r"\begin{table}[H]\centering\small", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Month & Days & bind/96 & H$_2$\% & FACET & ADMM & FACET & ADMM & ADMM\\",
         r" & & & & reach & reach & [ms] & [ms] & fb\\", r"\midrule"]
    for m in range(1, 13):
        d = months[m]
        L.append(f"{MON[m]} & {d['days']} & {d['bind']:.1f} & {d['h2']:.0f} & "
                 f"{d['fac_reach']:.0f}\\% & {d['adm_reach']:.0f}\\% & {d['fac_ms']:.1f} & {d['adm_ms']:.0f} & {d['fb']}\\\\")
    L += [r"\midrule",
          f"\\textbf{{Year}} & {year['days']} & {year['bind']:.1f} & {year['h2']:.0f} & "
          f"{year['fac_reach']:.0f}\\% & {year['adm_reach']:.0f}\\% & {year['fac_ms']:.1f} & {year['adm_ms']:.0f} & {tot_fb}\\\\",
          r"\bottomrule", r"\end{tabular}",
          r"\caption{Full-year monthly ledger (ERCOT 2025, warm block). ``bind/96'' is mean ceiling-binding "
          r"steps per day; ``reach'' is the share of steps meeting the $10^{-4}$~kW oracle bar; median comms "
          f"rounds are 1 (FACET) / 66--68 (ADMM) in every month. Iteration-free rate {year['iterfree']:.1f}\\% "
          f"({tot-tot_fb}/{tot}).}}", r"\label{tab:year-monthly}", r"\end{table}"]
    (RES/"year_monthly_table.tex").write_text("\n".join(L)+"\n")

    # ── Table: seasonal summary + driver ──
    S = [r"\begin{table}[H]\centering\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Season & Days & bind/96 & Renew.\ & FACET & ADMM & Fall-\\", r" & & & [kWh/d] & reach & reach & backs\\", r"\midrule"]
    for s in ["DJF", "MAM", "JJA", "SON"]:
        d = seasons[s]
        S.append(f"{SNAME[s]} ({s}) & {d['days']} & {d['bind']:.1f} & {d['ren']:.0f} & "
                 f"{d['fac_reach']:.0f}\\% & {d['adm_reach']:.0f}\\% & {d['fb']}\\\\")
    S += [r"\bottomrule", r"\end{tabular}",
          f"\\caption{{Seasonal summary. Ceiling-binding is renewable-scarcity-driven "
          f"($r={r_rb:.2f}$ between daily renewable availability and binding fraction); FACET's accuracy tracks "
          f"it inversely ($r={r_bf:.2f}$ binding vs.\\ reach). Fall (SON) is the annual worst case.}}",
          r"\label{tab:year-seasonal}", r"\end{table}"]
    (RES/"year_seasonal_table.tex").write_text("\n".join(S)+"\n")

    print("figures → results/figures/year_{rounds,scatter,binding}.pdf")
    print("tables  → results/year_{monthly,seasonal}_table.tex")
    print(f"r(binding,reach)={r_bf:.3f}  r(renew,binding)={r_rb:.3f}  iter-free={year['iterfree']:.2f}%")


if __name__ == "__main__":
    main()
