"""
run_week.py — 7-day FOCAPO-faithful two-settlement run with the mp-GNE RTM map +
exact eq.(17) variational selection.  Produces all metrics + figures + a results
pickle for the LaTeX report.

Per day: real DAM solve (H2 floor+coupling) on forecast → hourly p_DA; ONE daily RTM
map (anchor=daily-mean p_DA); RTM closed loop with the exact eq.(17) selector,
executed vs realized; H2 tracking; map-vs-centralized (+ADMM subsample) certification.
"""
from __future__ import annotations
import sys, time, json, pickle
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import focapo_1day as F
import calibrate as C
from amrhg.solvers.mp_solver import solve_all_agents_mp
from amrhg.solvers.gne_combiner import build_gne_solution
from amrhg.solvers.vgne_select import VariationalSelector
from amrhg.solvers.admm_solver import admm_solve

DAYS = [f"2025-04-0{d}" for d in range(1, 8)]     # 04-01 .. 04-07 (incl. $3553 spike)
SEED = 42


def run_day(day, rng):
    lam_da, lam_rt, cs_h, cw_h = F.load(day)
    N = len(F.FLEET)
    d_day = np.array([F.H2_FRAC*pmax*eta*24 for (nm,typ,pmax,rcap,eta) in F.FLEET])
    # forecasts
    lam_da_fc = np.maximum(lam_da+rng.normal(0,16,24), -50)
    cs_fc = np.clip(cs_h+rng.normal(0,0.20,24),0,1); cw_fc = np.clip(cw_h+rng.normal(0,0.20,24),0,0.8)
    p_da, _ = F.solve_dam(lam_da_fc, cs_fc, cw_fc, d_day)
    game = F.build_rtm(p_da.mean(axis=1))
    sols = solve_all_agents_mp(game, verbose=False)
    gf = build_gne_solution(game, sols, verbose=False, equilibrium_select="potential")
    sel = VariationalSelector(game)
    ncr = gf.n_cr; combos = int(np.prod([s.n_cr for s in sols]))
    h2 = np.zeros(N); cost_mp = np.zeros(N); cost_dam = np.zeros(N)
    P = np.zeros((96, N)); crhits = Counter()
    errs = []; e_admm = 0.0; t_map = 0.0; t_admm = 0.0; n_admm = 0
    for t in range(96):
        h = t//4
        lam_fc = np.clip(lam_rt[t]+rng.normal(0,8), *F.LAM_BOX)
        cs = np.clip(cs_h[h]+rng.normal(0,0.05),0,1); cw = np.clip(cw_h[h]+rng.normal(0,0.08),0,0.8)
        th = np.array([lam_fc, cs, cw])
        s = time.perf_counter(); x = sel.evaluate(gf, th); t_map += time.perf_counter()-s
        for k in gf.locate_all(th): crhits[k]+=1
        if x is None: x = admm_solve(game, th, rho=0.5, max_iter=6000, tol=1e-8).x_stacked
        xc = C.cent(game, th); errs.append(float(np.max(np.abs(x-xc))))
        if t % 12 == 0:
            s=time.perf_counter(); xa=admm_solve(game,th,rho=0.5,max_iter=6000,tol=1e-8).x_stacked; t_admm+=time.perf_counter()-s; n_admm+=1
            e_admm=max(e_admm,float(np.max(np.abs(x-xa))))
        for i in range(N):
            p = F.x_p(game,x,i); P[t,i]=p
            (nm,typ,pmax,rcap,eta)=F.FLEET[i]
            g=(cs_h[h] if typ=="solar" else cw_h[h])*rcap if typ!="grid" else 0.0
            pe=min(max(0.0,p+g),pmax); h2[i]+=eta*pe*F.DT_RT
            cost_mp[i]+=F.DT_RT*(lam_rt[t]/1000)*p - F.R_H2*eta*pe*F.DT_RT
            # DAM-only baseline: commit p_DA[h], settle at realized RT
            pda=p_da[i,h]; peb=min(max(0.0,pda+g),pmax)
            cost_dam[i]+=F.DT_RT*(lam_rt[t]/1000)*pda - F.R_H2*eta*peb*F.DT_RT
    return dict(day=day, lam_rt=lam_rt, lam_da=lam_da, ncr=ncr, combos=combos,
                crhits=dict(crhits), h2=h2, d_day=d_day, P=P, p_da=p_da,
                cost_mp=cost_mp, cost_dam=cost_dam, err_med=float(np.median(errs)),
                err_max=float(np.max(errs)), e_admm=e_admm, n_admm=n_admm,
                t_map=t_map, t_admm=t_admm/max(n_admm,1),
                sumpda=(p_da.sum(0).min(), p_da.sum(0).max()))


def main():
    rng = np.random.default_rng(SEED)
    print("="*76); print("7-DAY RUN — FOCAPO-faithful two-settlement, mp-GNE RTM + eq.(17)"); print("="*76)
    res = []
    for day in DAYS:
        t0=time.perf_counter(); r = run_day(day, rng)
        print(f"  {day}: λ[{r['lam_rt'].min():.0f},{r['lam_rt'].max():.0f}]  "
              f"CRs={r['ncr']} visited={len(r['crhits'])}  errMed={r['err_med']:.1e} errMax={r['err_max']:.1e} "
              f"|map-ADMM|={r['e_admm']:.1e}  ({time.perf_counter()-t0:.0f}s)")
        res.append(r)
    # aggregate
    all_crhits = Counter()
    for r in res:
        for k,v in r['crhits'].items(): all_crhits[k]+=v
    combos = res[0]['combos']
    cost_mp = sum(r['cost_mp'].sum() for r in res); cost_dam = sum(r['cost_dam'].sum() for r in res)
    print("\n"+"="*76+"\nWEEK SUMMARY\n"+"="*76)
    print(f"  RTM map CRs/day (min-max)     : {min(r['ncr'] for r in res)}-{max(r['ncr'] for r in res)}")
    print(f"  combos possible (K^N)         : {combos}")
    print(f"  DISTINCT CRs used all week    : {len(all_crhits)}  (recurrence: {len(all_crhits)}/{combos} of combos)")
    print(f"  map vs centralized  median    : {np.median([r['err_med'] for r in res]):.2e} kW  max {max(r['err_max'] for r in res):.2e} kW")
    print(f"  map vs ADMM (subsample) max   : {max(r['e_admm'] for r in res):.2e} kW")
    print(f"  online selection  : {1e3*np.mean([r['t_map'] for r in res])/96:.2f} ms/step   ADMM {np.mean([r['t_admm'] for r in res]):.2f} s/step")
    print(f"  week cost  mp-GNE : {cost_mp:8.1f} $   DAM-only : {cost_dam:8.1f} $   Δ = {cost_dam-cost_mp:+.1f} $")
    tot_h2 = sum(r['h2'] for r in res); tot_tgt = sum(r['d_day'] for r in res)
    print(f"  H2 week produced/target       : {tot_h2.sum():.0f} / {tot_tgt.sum():.0f} kg ({100*tot_h2.sum()/tot_tgt.sum():.0f}%)")
    with open(HERE/"results"/"week_results.pkl","wb") as f: pickle.dump(res, f)
    _figs(res, all_crhits, combos)
    print(f"\n  results → results/week_results.pkl ; figures → results/figures/week_*.png")


def _figs(res, all_crhits, combos):
    out=HERE/"results"/"figures"; out.mkdir(parents=True,exist_ok=True)
    # 1. week price + aggregate dispatch
    lam=np.concatenate([r['lam_rt'] for r in res]); agg=np.concatenate([r['P'].sum(1) for r in res])
    hh=np.arange(len(lam))*0.25/24
    fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True)
    ax[0].plot(hh,lam,lw=0.7); ax[0].axhline(60,color="r",ls=":",lw=0.8); ax[0].set_ylabel("λ_RT [$/MWh]"); ax[0].set_yscale("symlog"); ax[0].set_title("Week RTM price (symlog)")
    ax[1].plot(hh,agg,lw=0.8,color="tab:blue"); ax[1].axhline(F.L_MIN,color="g",ls="--",lw=0.8,label="L_min"); ax[1].axhline(F.L_MAX,color="purple",ls="--",lw=0.8,label="L_max")
    ax[1].set_ylabel("Σp [kW]"); ax[1].set_xlabel("day"); ax[1].legend(fontsize=8); ax[1].set_title("Coalition aggregate grid import (mp-GNE map)")
    fig.tight_layout(); fig.savefig(out/"week_price_dispatch.png",dpi=120); plt.close(fig)
    # 2. combo recurrence
    fig,ax=plt.subplots(figsize=(9,4))
    top=sorted(all_crhits, key=lambda k: -all_crhits[k])[:40]
    ax.bar([str(k) for k in top],[all_crhits[k] for k in top],color="teal")
    ax.tick_params(axis="x",labelsize=5,rotation=90)
    ax.set_xlabel("GNE critical region id"); ax.set_ylabel("5-min visits over the week")
    ax.set_title(f"Combo recurrence — {len(all_crhits)} distinct CRs used of {combos} possible combinations")
    fig.tight_layout(); fig.savefig(out/"week_recurrence.png",dpi=120); plt.close(fig)
    # 3. H2 per agent
    fig,ax=plt.subplots(figsize=(9,4)); h2=sum(r['h2'] for r in res); tg=sum(r['d_day'] for r in res)
    xr=np.arange(len(F.FLEET)); ax.bar(xr-0.2,h2,0.4,label="produced"); ax.bar(xr+0.2,tg,0.4,label="target")
    ax.set_xticks(xr); ax.set_xticklabels([f[0] for f in F.FLEET],rotation=25,fontsize=8); ax.set_ylabel("H2 [kg/week]"); ax.legend(); ax.set_title("Weekly H2: produced vs contract target")
    fig.tight_layout(); fig.savefig(out/"week_h2.png",dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
