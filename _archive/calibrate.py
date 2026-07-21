"""
calibrate.py — find (gamma, anchors, L_min, L_max, lam_ub) so the coupling binds on
BOTH faces over a real ERCOT day and the map exercises many CRs.  Uses the fast
centralized QP (not ADMM) for the accuracy check so the sweep is quick.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from collections import Counter
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from amrhg.solvers.game import Agent, GNEGame
from amrhg.solvers.mp_solver import solve_all_agents_mp
from amrhg.solvers.gne_combiner import build_gne_solution, filter_variational_kkt

R_H2 = 3.0; DT = 0.25; EPS_CV = 1e-3
WIND_CAP_MW, SOLAR_CAP_MW = 40000, 32000
N_P = 3; LAM, CF_S, CF_W = 0, 1, 2
# (name, type, p_max, ren_cap, eta, anchor_frac_of_pmax)
BASE = [
    ("PEM_Elec","grid",250.,  0.,0.020,0.5),("ALK","grid",200.,  0.,0.018,0.5),
    ("PEM_PV","solar",125.,125.,0.020,0.4),("PEM_PV_2","solar",125.,125.,0.020,0.4),
    ("PEM_Wind","wind",250.,250.,0.020,0.4),("PEM_Wind_2","wind",250.,250.,0.020,0.4),
]


def build(gamma, anchor_scale, L_min, L_max, lam_box):
    agents = []
    for i,(nm,typ,pmax,rcap,eta,af) in enumerate(BASE):
        a = R_H2*eta; pda = af*anchor_scale*pmax; g = gamma
        if typ == "grid":
            Q=np.array([[g]]); c=np.array([-DT*a - g*pda]); F=np.zeros((1,N_P)); F[0,LAM]=DT/1000
            A=np.array([[1.],[-1.]]); b=np.array([pmax,0.]); S=np.zeros((2,N_P)); C=np.array([[1.],[-1.]])
            agents.append(Agent(index=i,n_x=1,Q=Q,c=c,F=F,A_loc=A,b_loc=b,S_loc=S,C=C))
        else:
            cf = CF_S if typ=="solar" else CF_W
            Q=np.array([[g,0.],[0.,EPS_CV]]); c=np.array([-DT*a-g*pda, DT*a]); F=np.zeros((2,N_P)); F[0,LAM]=DT/1000
            A=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[0.,1.],[1.,-1.],[-1.,1.]])
            b=np.array([pmax,rcap,0.,0.,0.,pmax,0.]); S=np.zeros((7,N_P))
            S[1,cf]=rcap; S[4,cf]=rcap; S[5,cf]=-rcap; S[6,cf]=rcap
            C=np.array([[1.,0.],[-1.,0.]])
            agents.append(Agent(index=i,n_x=2,Q=Q,c=c,F=F,A_loc=A,b_loc=b,S_loc=S,C=C))
    d=np.array([L_max,-L_min]); S_coup=np.zeros((2,N_P))
    p_lb=np.array([lam_box[0],0.,0.]); p_ub=np.array([lam_box[1],1.,0.8])
    return GNEGame(agents=agents,d=d,S_coup=S_coup,d_lb=None,S_coup_lb=None,p_lb=p_lb,p_ub=p_ub)


def cent(game, th):
    nx=game.n_x_total; Q=np.zeros((nx,nx));cc=np.zeros(nx);F=np.zeros((nx,N_P))
    for ag in game.agents: sl=game.x_slice(ag.index);Q[sl,sl]=ag.Q;cc[sl]=ag.c;F[sl]=ag.F
    Gr=[];wr=[];Wr=[]
    for ag in game.agents:
        sl=game.x_slice(ag.index)
        for r in range(ag.A_loc.shape[0]): row=np.zeros(nx);row[sl]=ag.A_loc[r];Gr.append(row);wr.append(ag.b_loc[r]);Wr.append(ag.S_loc[r])
    for r in range(game.n_coupling):
        row=np.zeros(nx)
        for ag in game.agents: row[game.x_slice(ag.index)]=ag.C[r]
        Gr.append(row);wr.append(game.d[r]);Wr.append(game.S_coup[r])
    G=np.array(Gr);w0=np.array(wr);W=np.array(Wr);q=cc+F@th;rhs=w0+W@th
    r=minimize(lambda x:0.5*x@Q@x+q@x,np.zeros(nx),constraints=[{'type':'ineq','fun':(lambda x,g=g,v=v:v-g@x)} for g,v in zip(G,rhs)],method='SLSQP',options={'ftol':1e-11,'maxiter':800})
    return r.x


def load(day):
    d=HERE/"data"/"ercot"; rtm=pd.read_csv(d/"ercot_rtm_lmp_2025.csv")
    rtm=rtm[(rtm.deliveryDate==day)&(rtm.settlementPoint=="HB_HUBAVG")].sort_values(["deliveryHour","deliveryInterval"])
    lam=rtm.settlementPointPrice.to_numpy(float)
    sol=pd.read_csv(d/"ercot_solar_production_2025.csv"); win=pd.read_csv(d/"ercot_wind_production_2025.csv")
    cs=np.repeat(sol[sol.deliveryDate==day].sort_values("hourEnding").genSystemWide.to_numpy(float)/SOLAR_CAP_MW,4)[:len(lam)]
    cw=np.repeat(win[win.deliveryDate==day].sort_values("hourEnding").genSystemWide.to_numpy(float)/WIND_CAP_MW,4)[:len(lam)]
    return lam,cs,cw


def evaluate(cfg, day="2025-03-29"):
    game=build(**cfg)
    sols=solve_all_agents_mp(game,verbose=False)
    gf=build_gne_solution(game,sols,verbose=False,equilibrium_select="potential")
    gmap=filter_variational_kkt(gf,game,verbose=False)
    lam,cs,cw=load(day); lb,ub=cfg["lam_box"]
    L_min,L_max=cfg["L_min"],cfg["L_max"]
    vis=Counter(); nfloor=nceil=0; emax=0.; miss=0
    for t in range(len(lam)):
        th=np.array([np.clip(lam[t],lb,ub),cs[t],cw[t]])
        x=gmap.evaluate(th); k=gmap.locate(th)
        if x is None: miss+=1; continue
        vis[k]+=1
        sp=sum(x[game.x_slice(i).start] for i in range(game.N))
        if abs(sp-L_min)<1: nfloor+=1
        if abs(sp-L_max)<1: nceil+=1
        emax=max(emax,float(np.max(np.abs(x-cent(game,th)))))
    return dict(ncr=gmap.n_cr, visited=len(vis), floor=nfloor, ceil=nceil, miss=miss, emax=emax, n=len(lam))


if __name__ == "__main__":
    configs = [
        dict(gamma=5e-3, anchor_scale=1.0, L_min=100, L_max=700, lam_box=(-50,150)),   # current
        dict(gamma=1e-3, anchor_scale=1.0, L_min=300, L_max=700, lam_box=(-50,250)),
        dict(gamma=5e-4, anchor_scale=1.0, L_min=400, L_max=650, lam_box=(-50,250)),
        dict(gamma=3e-4, anchor_scale=1.0, L_min=450, L_max=600, lam_box=(-50,300)),
        dict(gamma=5e-4, anchor_scale=0.8, L_min=350, L_max=600, lam_box=(-50,300)),
    ]
    print(f"{'gamma':>7} {'Lmin':>5} {'Lmax':>5} {'lamUB':>6} | {'CRs':>4} {'vis':>4} {'floor%':>7} {'ceil%':>6} {'miss':>5} {'err':>9}")
    for c in configs:
        r = evaluate(c)
        print(f"{c['gamma']:>7.0e} {c['L_min']:>5} {c['L_max']:>5} {c['lam_box'][1]:>6} | "
              f"{r['ncr']:>4} {r['visited']:>4} {100*r['floor']/r['n']:>6.0f}% {100*r['ceil']/r['n']:>5.0f}% "
              f"{r['miss']:>5} {r['emax']:>9.1e}")
