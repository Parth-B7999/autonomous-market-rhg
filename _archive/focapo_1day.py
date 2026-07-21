"""
focapo_1day.py — FOCAPO-faithful (core) one-day two-settlement run, mp-GNE RTM.

Pipeline (per PLAN_mp_focapo_rtm + FOCAPO forecast model):
  1. Load real ERCOT day: DAM hourly λ_DA, RTM 15-min λ_RT, solar/wind → CF.
  2. Forecasts (FOCAPO): DAM = real + N(0,16 $/MWh), renewable + N(0,20% cap);
     RTM nowcast = real + N(0,8 $/MWh), PV + N(0,5%), wind + N(0,8%).
  3. DAM solve — 24h centralized QP (social optimum = v-GNE) on DAM FORECAST, with
     hard daily H2 floor + per-hour coupling → real hourly anchor p_DA[i][h].
  4. RTM closed loop — per hour, build the explicit mp-GNE map with that hour's p_DA
     baked; each 15-min step evaluate at the NOWCAST θ, execute against the REALIZED
     price/renewable; track H2 inventory vs the daily target.
  5. Report DAM plan, RTM-vs-DA, forecast error, H2 tracking, map==ADMM, cost, CRs.

Scope note: H=1 single-step map (horizon-coupled H2 map deferred); no batteries, no
65-min binary gate — consistent with the pure-mpQP mp-GNE method.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from collections import Counter
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from amrhg.solvers.game import Agent, GNEGame
from amrhg.solvers.mp_solver import solve_all_agents_mp
from amrhg.solvers.gne_combiner import build_gne_solution, filter_variational_kkt
from amrhg.solvers.admm_solver import admm_solve

R_H2 = 3.0
DT_RT, DT_DA = 0.25, 1.0
EPS_CV = 1e-3
WIND_CAP_MW, SOLAR_CAP_MW = 40000, 32000
GAMMA = 5e-4
GAMMA_DA = 1e-4          # DAM subproblem strict-convexity regularizer on grid import p
L_MIN, L_MAX = 350.0, 600.0
LAM_BOX = (-50.0, 300.0)
DAY = sys.argv[1] if len(sys.argv) > 1 else "2025-03-29"
SEED = 42
N_P = 3; LAM, CF_S, CF_W = 0, 1, 2
# (name, type, p_max, ren_cap, eta)
FLEET = [
    ("PEM_Elec","grid",250.,  0.,0.020),("ALK","grid",200.,  0.,0.018),
    ("PEM_PV","solar",125.,125.,0.020),("PEM_PV_2","solar",125.,125.,0.020),
    ("PEM_Wind","wind",250.,250.,0.020),("PEM_Wind_2","wind",250.,250.,0.020),
]
H2_FRAC = 0.55          # daily H2 target = frac · max producible (keeps DAM feasible vs L_max)


# ── RTM single-step game with a given per-agent anchor p_DA ────────────────────
def build_rtm(p_da):
    agents = []
    for i,(nm,typ,pmax,rcap,eta) in enumerate(FLEET):
        a = R_H2*eta; g = GAMMA; pda = p_da[i]
        if typ == "grid":
            Q=np.array([[g]]); c=np.array([-DT_RT*a - g*pda]); F=np.zeros((1,N_P)); F[0,LAM]=DT_RT/1000
            A=np.array([[1.],[-1.]]); b=np.array([pmax,0.]); S=np.zeros((2,N_P)); C=np.array([[1.],[-1.]])
            agents.append(Agent(index=i,n_x=1,Q=Q,c=c,F=F,A_loc=A,b_loc=b,S_loc=S,C=C))
        else:
            cf = CF_S if typ=="solar" else CF_W
            Q=np.array([[g,0.],[0.,EPS_CV]]); c=np.array([-DT_RT*a-g*pda, DT_RT*a]); F=np.zeros((2,N_P)); F[0,LAM]=DT_RT/1000
            A=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[0.,1.],[1.,-1.],[-1.,1.]])
            b=np.array([pmax,rcap,0.,0.,0.,pmax,0.]); S=np.zeros((7,N_P))
            S[1,cf]=rcap; S[4,cf]=rcap; S[5,cf]=-rcap; S[6,cf]=rcap
            agents.append(Agent(index=i,n_x=2,Q=Q,c=c,F=F,A_loc=A,b_loc=b,S_loc=S,C=np.array([[1.,0.],[-1.,0.]])))
    return GNEGame(agents=agents,d=np.array([L_MAX,-L_MIN]),S_coup=np.zeros((2,N_P)),
                   d_lb=None,S_coup_lb=None,p_lb=np.array([LAM_BOX[0],0.,0.]),p_ub=np.array([LAM_BOX[1],1.,0.8]))


def x_p(game,x,i): return x[game.x_slice(i).start]


# ── DAM: 24h centralized QP (social optimum) on forecast, daily H2 floor ───────
# NOTE: kept only as the offline validation ORACLE for the distributed DAM below.
# The live pipeline uses solve_dam_admm (no agent shares its model).
def solve_dam_centralized(lam_da, cf_s_da, cf_w_da, d_day):
    N = len(FLEET); H = 24
    # decision: per agent per hour  p (grid), and cv for renewables. flatten.
    # variables: for each agent, H p-vars; renewables also H cv-vars.
    idx = []; off = 0
    for (nm,typ,pmax,rcap,eta) in FLEET:
        n = H*(2 if typ!="grid" else 1); idx.append((off,off+n,typ)); off += n
    nv = off
    def unpack(z):
        out=[]
        for (s,e,typ) in idx:
            blk=z[s:e]
            if typ=="grid": out.append((blk, None))
            else: out.append((blk[:H], blk[H:]))
        return out
    def pelec(i,p,cv):
        (nm,typ,pmax,rcap,eta)=FLEET[i]
        g = (cf_s_da if typ=="solar" else cf_w_da)*rcap if typ!="grid" else np.zeros(H)
        return p + g - (cv if cv is not None else 0.0)
    def obj(z):
        # Same objective the distributed agents solve (build_dam_game): linear
        # energy−H2 term + γ_DA strict-convexity on p (+ ε on renewable cv), so
        # this oracle is an apples-to-apples reference for the ADMM DAM gate.
        u=unpack(z); J=0.0
        for i,(p,cv) in enumerate(u):
            (nm,typ,pmax,rcap,eta)=FLEET[i]; a=R_H2*eta
            J += np.sum(DT_DA*(lam_da/1000 - a)*p) + 0.5*GAMMA_DA*np.sum(p*p)
            if cv is not None: J += np.sum(DT_DA*a*cv) + 0.5*EPS_CV*np.sum(cv*cv)
        return float(J)
    cons=[]; bnds=[]
    for i,(s,e,typ) in enumerate(idx):
        (nm,tp,pmax,rcap,eta)=FLEET[i]
        for h in range(H): bnds.append((0.,pmax))         # p
        if typ!="grid":
            for h in range(H): bnds.append((0.,rcap))     # cv
    # coupling per hour + H2 daily + renewable p_elec box
    def make(z):
        return unpack(z)
    for h in range(H):
        cons.append({'type':'ineq','fun':(lambda z,h=h: sum(make(z)[i][0][h] for i in range(N)) - L_MIN)})
        cons.append({'type':'ineq','fun':(lambda z,h=h: L_MAX - sum(make(z)[i][0][h] for i in range(N)))})
    for i in range(N):
        (nm,typ,pmax,rcap,eta)=FLEET[i]
        cons.append({'type':'ineq','fun':(lambda z,i=i: np.sum(pelec(i,*make(z)[i])*eta*DT_DA) - d_day[i])})
        if typ!="grid":
            for h in range(H):
                cons.append({'type':'ineq','fun':(lambda z,i=i,h=h: pmax - pelec(i,*make(z)[i])[h])})
                cons.append({'type':'ineq','fun':(lambda z,i=i,h=h: pelec(i,*make(z)[i])[h])})
    z0=np.zeros(nv)
    for (s,e,typ) in idx:
        z0[s:s+H]=L_MIN/N
    r=minimize(obj,z0,bounds=bnds,constraints=cons,method='SLSQP',options={'ftol':1e-7,'maxiter':500})
    u=unpack(r.x)
    p_da=np.array([u[i][0] for i in range(N)])            # (N,24)
    return p_da, r


# ── DAM (distributed): each agent's own 24h QP; only the per-hour aggregate is ──
# shared. No agent ever sees another's model. Coupling L_MIN ≤ Σ_i p_{i,h} ≤ L_MAX
# per hour is enforced by ADMM z/λ (bilateral clip). This is the live DAM.
def build_dam_game(lam_da, cf_s_da, cf_w_da, d_day):
    """24-hour DAM as a GNEGame (n_p=0: prices/CF baked in as constants)."""
    H = 24
    c_p = DT_DA * (lam_da / 1000.0)                        # energy cost coeff per hour
    agents = []
    for i, (nm, typ, pmax, rcap, eta) in enumerate(FLEET):
        a = R_H2 * eta
        F = np.zeros((0, 0))                              # placeholder, set per-branch below
        if typ == "grid":
            nx = H
            Q = GAMMA_DA * np.eye(H)
            c = c_p - DT_DA * a                           # (H,)  min Σ (λ/1000 − a)Δt p
            F = np.zeros((nx, 0))
            # local: 0 ≤ p_h ≤ pmax  (2H rows)  +  H2 floor  −ηΔt Σ p ≤ −D_day
            A = np.vstack([np.eye(H), -np.eye(H), -eta * DT_DA * np.ones((1, H))])
            b = np.concatenate([pmax * np.ones(H), np.zeros(H), [-d_day[i]]])
            S = np.zeros((A.shape[0], 0))
            C = np.eye(H)                                 # p_h → hour-h coupling
            agents.append(Agent(index=i, n_x=nx, Q=Q, c=c, F=F,
                                A_loc=A, b_loc=b, S_loc=S, C=C))
        else:
            nx = 2 * H                                    # [p_0..23, cv_0..23]
            g = (cf_s_da if typ == "solar" else cf_w_da) * rcap       # (H,) known
            Q = np.zeros((nx, nx))
            Q[:H, :H] = GAMMA_DA * np.eye(H)
            Q[H:, H:] = EPS_CV * np.eye(H)
            c = np.concatenate([c_p - DT_DA * a, DT_DA * a * np.ones(H)])  # p then cv
            F = np.zeros((nx, 0))
            Ip, Icv = np.eye(H), np.eye(H)
            Z = np.zeros((H, H))
            # rows: p box (2H) | cv box 0≤cv≤g (2H) | p_elec box 0≤p+g−cv≤pmax (2H)
            #       | H2 floor −ηΔt Σ(p−cv) ≤ −(D_day − ηΔt Σ g)
            A = np.vstack([
                np.hstack([Ip, Z]), np.hstack([-Ip, Z]),          # 0 ≤ p ≤ pmax
                np.hstack([Z, Icv]), np.hstack([Z, -Icv]),        # 0 ≤ cv ≤ g
                np.hstack([Ip, -Icv]), np.hstack([-Ip, Icv]),     # p−cv ≤ pmax−g ; −(p−cv) ≤ g
                np.hstack([-eta * DT_DA * np.ones((1, H)), eta * DT_DA * np.ones((1, H))]),
            ])
            b = np.concatenate([
                pmax * np.ones(H), np.zeros(H),
                g, np.zeros(H),
                pmax - g, g,
                [-(d_day[i] - eta * DT_DA * np.sum(g))],
            ])
            S = np.zeros((A.shape[0], 0))
            C = np.hstack([np.eye(H), np.zeros((H, H))])          # only p enters coupling
            agents.append(Agent(index=i, n_x=nx, Q=Q, c=c, F=F,
                                A_loc=A, b_loc=b, S_loc=S, C=C))
    N = len(FLEET)
    return GNEGame(agents=agents,
                   d=L_MAX * np.ones(H), S_coup=np.zeros((H, 0)),
                   d_lb=L_MIN * np.ones(H), S_coup_lb=np.zeros((H, 0)),
                   p_lb=np.array([]), p_ub=np.array([]))


def solve_dam_admm(lam_da, cf_s_da, cf_w_da, d_day, rho=0.1, max_iter=5000, tol=1e-3):
    """Distributed DAM. Returns (p_da (N,24), ADMMResult)."""
    game = build_dam_game(lam_da, cf_s_da, cf_w_da, d_day)
    res = admm_solve(game, np.array([]), rho=rho, max_iter=max_iter,
                     tol=tol, qp_solver="osqp")
    p_da = np.array([res.x_sol[i][:24] for i in range(len(FLEET))])       # (N,24)
    return p_da, res


def load(day):
    d=HERE/"data"/"ercot"
    da=pd.read_csv(d/"ercot_dam_lmp_2025.csv")
    da=da[(da.deliveryDate==day)&(da.settlementPoint=="HB_HUBAVG")].sort_values("hourEnding")
    lam_da=da.settlementPointPrice.to_numpy(float)[:24]
    rtm=pd.read_csv(d/"ercot_rtm_lmp_2025.csv")
    rtm=rtm[(rtm.deliveryDate==day)&(rtm.settlementPoint=="HB_HUBAVG")].sort_values(["deliveryHour","deliveryInterval"])
    lam_rt=rtm.settlementPointPrice.to_numpy(float)
    sol=pd.read_csv(d/"ercot_solar_production_2025.csv"); win=pd.read_csv(d/"ercot_wind_production_2025.csv")
    cs_h=sol[sol.deliveryDate==day].sort_values("hourEnding").genSystemWide.to_numpy(float)/SOLAR_CAP_MW
    cw_h=win[win.deliveryDate==day].sort_values("hourEnding").genSystemWide.to_numpy(float)/WIND_CAP_MW
    return lam_da[:24], lam_rt[:96], cs_h[:24], cw_h[:24]


def main():
    rng=np.random.default_rng(SEED)
    print("="*76); print(f"FOCAPO-faithful 1-day two-settlement | mp-GNE RTM | ERCOT {DAY}"); print("="*76)
    lam_da, lam_rt, cs_h, cw_h = load(DAY)
    N=len(FLEET)
    d_day=np.array([H2_FRAC*pmax*eta*24 for (nm,typ,pmax,rcap,eta) in FLEET])
    print(f"Fleet N={N}, daily H2 targets [kg] = {np.round(d_day,1)}")

    # ── forecasts ─────────────────────────────────────────────────────────────
    lam_da_fc=np.maximum(lam_da+rng.normal(0,16,24),-50)
    cs_da_fc=np.clip(cs_h+rng.normal(0,0.20,24),0,1); cw_da_fc=np.clip(cw_h+rng.normal(0,0.20,24),0,0.8)

    # ── DAM (distributed ADMM; each agent solves only its own 24h QP) ──────────
    print("\n[DAM] distributed ADMM — per-agent 24h QP, only per-hour aggregate shared ...")
    t0=time.perf_counter(); p_da, rd = solve_dam_admm(lam_da_fc, cs_da_fc, cw_da_fc, d_day)
    print(f"    DAM-ADMM ({time.perf_counter()-t0:.1f}s, converged={rd.converged}, "
          f"iters={rd.n_iter}); Σp_DA/hour range = [{p_da.sum(0).min():.0f},{p_da.sum(0).max():.0f}] kW")

    # ── validation gate: aggregate/cost/H2 vs centralized oracle (offline only) ─
    p_da_c, _ = solve_dam_centralized(lam_da_fc, cs_da_fc, cw_da_fc, d_day)
    def _dam_cost(P):
        return float(sum(np.sum(DT_DA*(lam_da_fc/1000 - R_H2*eta)*P[i])
                         for i,(nm,typ,pmax,rcap,eta) in enumerate(FLEET)))
    agg_err  = float(np.max(np.abs(p_da.sum(0) - p_da_c.sum(0))))
    Jc = _dam_cost(p_da_c); cost_err = abs(_dam_cost(p_da)-Jc)/max(abs(Jc),1e-9)*100
    h2_made = np.zeros(N)
    for i,(nm,typ,pmax,rcap,eta) in enumerate(FLEET):
        xi = rd.x_sol[i]
        if typ=="grid":
            pe = np.clip(xi[:24], 0, pmax)
        else:
            g = (cs_da_fc if typ=="solar" else cw_da_fc)*rcap
            pe = np.clip(xi[:24] + g - xi[24:48], 0, pmax)
        h2_made[i] = eta*DT_DA*np.sum(pe)
    h2_ok = bool(np.all(h2_made >= d_day - 1e-2))
    dam_gate = (agg_err < 1.0) and (cost_err < 0.5) and h2_ok
    print(f"    [DAM gate vs centralized] Σp/hr err={agg_err:.2f} kW, cost Δ={cost_err:.2f}%, "
          f"H2 floors {'met' if h2_ok else 'MISSED'} → {'PASS ✅' if dam_gate else 'FAIL ❌'}")

    # ── RTM: ONE daily map (daily-representative anchor = mean of DAM p_DA) ─────
    p_da_bar = p_da.mean(axis=1)
    print(f"\n[RTM] build ONE daily map (anchor = daily-mean p_DA, Σ={p_da_bar.sum():.0f}) ...")
    s=time.perf_counter(); game=build_rtm(p_da_bar)
    sols=solve_all_agents_mp(game,verbose=False)
    gf=build_gne_solution(game,sols,verbose=False,equilibrium_select="potential")
    gmap=filter_variational_kkt(gf,game,verbose=False)
    t_build=time.perf_counter()-s
    print(f"    per-agent CRs {[s_.n_cr for s_ in sols]} → {gf.n_cr} GNE → {gmap.n_cr} variational  ({t_build:.1f}s)")
    ncr_hours=[gmap.n_cr]
    h2_inv=np.zeros(N); P_exec=np.zeros((96,N)); P_da_step=np.zeros((96,N))
    cr_all=Counter(); e_admm=0.0; n_admm=0; t_map=0.0; t_admm=0.0; cost_rt=np.zeros(N)
    for t in range(96):
        h=t//4
        lam_fc=np.clip(lam_rt[t]+rng.normal(0,8),*LAM_BOX)
        cs_fc=np.clip(cs_h[h]+rng.normal(0,0.05),0,1); cw_fc=np.clip(cw_h[h]+rng.normal(0,0.08),0,0.8)
        th=np.array([lam_fc,cs_fc,cw_fc])
        s=time.perf_counter(); xm=gmap.evaluate(th); t_map+=time.perf_counter()-s
        k=gmap.locate(th); cr_all[k]+=1
        if xm is None:
            xm=admm_solve(game,th,rho=0.5,max_iter=6000,tol=1e-8).x_stacked
        for i in range(N):
            p=x_p(game,xm,i); P_exec[t,i]=p; P_da_step[t,i]=p_da[i,h]
            (nm,typ,pmax,rcap,eta)=FLEET[i]
            g=(cs_h[h] if typ=="solar" else cw_h[h])*rcap if typ!="grid" else 0.0
            pe=min(max(0.0,p+g),pmax)
            h2_inv[i]+=eta*pe*DT_RT
            cost_rt[i]+=DT_RT*(lam_rt[t]/1000)*p - R_H2*eta*pe*DT_RT
        if t%12==0:
            s=time.perf_counter(); xa=admm_solve(game,th,rho=0.5,max_iter=6000,tol=1e-8).x_stacked; t_admm+=time.perf_counter()-s; n_admm+=1
            e_admm=max(e_admm,float(np.max(np.abs(xm-xa))))

    # ── report ────────────────────────────────────────────────────────────────
    print("\n"+"="*76+"\nRESULTS\n"+"="*76)
    print(f"  DAM anchor Σp_DA/hour           : [{p_da.sum(0).min():.0f}, {p_da.sum(0).max():.0f}] kW  (coupling {L_MIN:.0f}-{L_MAX:.0f})")
    print(f"  RTM map CRs (one daily map)     : {ncr_hours[0]}")
    print(f"  distinct CRs visited over day   : {len([k for k in cr_all if k is not None])} / {ncr_hours[0]}")
    print(f"  max |map − ADMM| ({n_admm} checks)     : {e_admm:.2e} kW")
    print(f"  H2 produced vs target [kg]:")
    for i,(nm,typ,pmax,rcap,eta) in enumerate(FLEET):
        print(f"     {nm:12s}: {h2_inv[i]:6.1f} / {d_day[i]:6.1f}  ({100*h2_inv[i]/d_day[i]:5.1f}%)")
    print(f"  RTM cost per agent [$]          : {np.round(cost_rt,1)}")
    print(f"  offline map build (24 hrs)      : {t_build:.1f}s   online lookup {1e6*t_map/96:.1f} µs/step")
    print(f"  ADMM per step                   : {t_admm/max(n_admm,1):.2f}s")
    gate = e_admm < 1e-2
    print(f"\n  GATE (map == ADMM): {'PASS ✅' if gate else 'FAIL ❌'}")
    _plot(lam_da, lam_rt, cs_h, cw_h, p_da, P_exec, P_da_step, h2_inv, d_day)
    print(f"\n  Figure → {HERE/'results'/'figures'/'focapo_1day.png'}")


def _plot(lam_da, lam_rt, cs, cw, p_da, Pe, Pda, h2, d_day):
    out=HERE/"results"/"figures"; out.mkdir(parents=True,exist_ok=True)
    hr=np.arange(96)*0.25; hh=np.arange(24)
    fig,ax=plt.subplots(3,1,figsize=(11,9))
    ax[0].step(hh,lam_da,where="mid",label="λ_DA (hourly)",lw=1.4)
    ax[0].plot(hr,lam_rt,"k-",lw=0.8,alpha=0.7,label="λ_RT (15-min realized)")
    ax[0].axhline(60,color="r",ls=":",lw=0.8); ax[0].set_ylabel("$/MWh"); ax[0].legend(fontsize=8); ax[0].set_title(f"ERCOT {DAY} — DAM vs RTM price")
    for i,f in enumerate(FLEET):
        ax[1].plot(hr,Pe[:,i],"-",lw=1.3,label=f[0]); ax[1].step(hh,p_da[i],where="mid",ls="--",lw=0.7,alpha=0.6)
    ax[1].axhline(L_MIN,color="g",ls="--",lw=0.7); ax[1].axhline(L_MAX,color="purple",ls="--",lw=0.7)
    ax[1].set_ylabel("grid buy [kW]"); ax[1].legend(fontsize=7,ncol=3); ax[1].set_title("RTM executed (solid) vs DAM anchor (dashed)")
    ax[2].bar(range(len(FLEET)),h2,label="produced"); ax[2].bar(range(len(FLEET)),d_day,fill=False,edgecolor="r",label="target")
    ax[2].set_xticks(range(len(FLEET))); ax[2].set_xticklabels([f[0] for f in FLEET],rotation=30,fontsize=7)
    ax[2].set_ylabel("H2 [kg/day]"); ax[2].legend(fontsize=8); ax[2].set_title("Daily H2: produced vs target")
    fig.tight_layout(); fig.savefig(out/"focapo_1day.png",dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
