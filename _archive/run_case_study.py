"""
run_case_study.py — Iteration-free RTM explicit mp-GNE, med-scale H2 fleet, ERCOT 2025.

Med-scale FOCAPO fleet (batteries dropped → plain electrolyzers), two-sided coupling
L_min ≤ Σp ≤ L_max, DA anchor ½γ(p−p_DA)², buy-only.  Single-step RTM game per 15-min
ERCOT interval.  θ = [λ_RT, CF_solar, CF_wind] (p_DA baked per the daily-rebuild design).

Pipeline: per-agent mpQP → build_gne_solution → filter_variational_kkt → explicit map.
Closed loop over one ERCOT day: at each 15-min step evaluate the map at θ_t and compare
to iterative ADMM (ground truth) + timing.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from amrhg.solvers.game import Agent, GNEGame
from amrhg.solvers.mp_solver import solve_all_agents_mp
from amrhg.solvers.gne_combiner import build_gne_solution, filter_variational_kkt
from amrhg.solvers.admm_solver import admm_solve

R_H2 = 3.0
DT = 0.25                     # 15-min step [hr]
EPS_CV = 1e-3
WIND_CAP_MW, SOLAR_CAP_MW = 40000, 32000
L_MIN, L_MAX = 100.0, 700.0
LAM_BOX = (-50.0, 150.0)
DAY = sys.argv[1] if len(sys.argv) > 1 else "2025-07-15"

# Med-scale fleet, batteries removed. (name, type, p_max, ren_cap, eta, gamma, p_DA)
FLEET = [
    ("PEM_Elec", "grid",  250.0,   0.0, 0.020, 5e-3, 120.0),
    ("ALK",      "grid",  200.0,   0.0, 0.018, 5e-3, 100.0),
    ("PEM_PV",   "solar", 125.0, 125.0, 0.020, 5e-3,  60.0),
    ("PEM_PV_2", "solar", 125.0, 125.0, 0.020, 5e-3,  60.0),
    ("PEM_Wind", "wind",  250.0, 250.0, 0.020, 4e-3, 120.0),
    ("PEM_Wind_2","wind", 250.0, 250.0, 0.020, 4e-3, 120.0),
]
# θ layout: [0]=λ, [1]=CF_solar, [2]=CF_wind
N_P = 3
LAM, CF_S, CF_W = 0, 1, 2


def build_game():
    agents = []
    for i, (nm, typ, pmax, rcap, eta, gam, pda) in enumerate(FLEET):
        a = R_H2 * eta                                # $/kWh break-even
        if typ == "grid":
            Q = np.array([[gam]])
            c = np.array([-DT * a - gam * pda])       # −dt·rη·p − γ·p_DA·p (anchor)
            F = np.zeros((1, N_P)); F[0, LAM] = DT / 1000.0
            A = np.array([[1.0], [-1.0]]); b = np.array([pmax, 0.0]); S = np.zeros((2, N_P))
            C = np.array([[1.0], [-1.0]])             # p in both coupling rows
            agents.append(Agent(index=i, n_x=1, Q=Q, c=c, F=F, A_loc=A, b_loc=b, S_loc=S, C=C))
        else:
            cf = CF_S if typ == "solar" else CF_W
            Q = np.array([[gam, 0.0], [0.0, EPS_CV]])
            c = np.array([-DT * a - gam * pda, DT * a])   # p anchor + curtailment cost
            F = np.zeros((2, N_P)); F[0, LAM] = DT / 1000.0
            # rows: p≤pmax; cv≤g; -p≤0; -cv≤0; cv≤g; p-cv≤pmax-g; -p+cv≤g   (g=cf·rcap)
            A = np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[0.,1.],[1.,-1.],[-1.,1.]])
            b = np.array([pmax, rcap, 0., 0., 0., pmax, 0.])
            S = np.zeros((7, N_P))
            S[1, cf] = rcap        # cv ≤ cf·rcap
            S[4, cf] = rcap        # cv ≤ cf·rcap (dup guard)
            S[5, cf] = -rcap       # p−cv ≤ pmax − cf·rcap
            S[6, cf] = rcap        # −p+cv ≤ cf·rcap
            C = np.array([[1.,0.],[-1.,0.]])
            agents.append(Agent(index=i, n_x=2, Q=Q, c=c, F=F, A_loc=A, b_loc=b, S_loc=S, C=C))
    d = np.array([L_MAX, -L_MIN]); S_coup = np.zeros((2, N_P))
    p_lb = np.array([LAM_BOX[0], 0.0, 0.0]); p_ub = np.array([LAM_BOX[1], 1.0, 0.8])
    return GNEGame(agents=agents, d=d, S_coup=S_coup, d_lb=None, S_coup_lb=None,
                   p_lb=p_lb, p_ub=p_ub)


def load_ercot(day):
    d = HERE / "data" / "ercot"
    rtm = pd.read_csv(d / "ercot_rtm_lmp_2025.csv")
    rtm = rtm[(rtm.deliveryDate == day) & (rtm.settlementPoint == "HB_HUBAVG")]
    rtm = rtm.sort_values(["deliveryHour", "deliveryInterval"])
    lam = rtm.settlementPointPrice.to_numpy(float)                 # 15-min λ
    sol = pd.read_csv(d / "ercot_solar_production_2025.csv")
    win = pd.read_csv(d / "ercot_wind_production_2025.csv")
    cs = (sol[sol.deliveryDate == day].sort_values("hourEnding").genSystemWide.to_numpy(float) / SOLAR_CAP_MW)
    cw = (win[win.deliveryDate == day].sort_values("hourEnding").genSystemWide.to_numpy(float) / WIND_CAP_MW)
    cs = np.repeat(cs, 4)[:len(lam)]; cw = np.repeat(cw, 4)[:len(lam)]   # hourly → 15-min
    return lam, cs, cw


def main():
    print("=" * 74)
    print(f"CASE STUDY — iteration-free RTM mp-GNE | med-scale fleet | ERCOT {DAY}")
    print("=" * 74)
    game = build_game()
    print(f"Fleet N={game.N} (2 grid + 4 renewable), Σcap={sum(f[2] for f in FLEET):.0f} kW")
    print(f"Coupling {L_MIN:.0f} ≤ Σp ≤ {L_MAX:.0f} | θ=[λ,CF_solar,CF_wind] n_p={game.n_p}")

    t0 = time.perf_counter()
    sols = solve_all_agents_mp(game, verbose=False)
    gf = build_gne_solution(game, sols, verbose=False, equilibrium_select="potential")
    gmap = filter_variational_kkt(gf, game, verbose=False)
    t_build = time.perf_counter() - t0
    print(f"\nExplicit map: per-agent CRs {[s.n_cr for s in sols]} → {gf.n_cr} GNE → "
          f"{gmap.n_cr} variational CRs  (offline build {t_build:.2f}s)")

    lam, cs, cw = load_ercot(DAY)
    n = len(lam)
    ADMM_EVERY = 8                      # ADMM ground-truth check on a subsample (it's ~10 s/call)
    print(f"\nClosed loop over {n} × 15-min ERCOT steps "
          f"(λ ∈ [{lam.min():.0f}, {lam.max():.0f}] $/MWh) ; ADMM check every {ADMM_EVERY} steps ...")
    P_map = np.zeros((n, game.N)); P_admm = np.full((n, game.N), np.nan)
    e_max = 0.0; miss = 0; t_map = 0.0; t_admm = 0.0; n_admm = 0
    cr_visits = []
    for t in range(n):
        th = np.array([np.clip(lam[t], *LAM_BOX), cs[t], cw[t]])
        s = time.perf_counter(); xm = gmap.evaluate(th); t_map += time.perf_counter() - s
        k = gmap.locate(th); cr_visits.append(k)
        if xm is None:
            miss += 1; continue
        for i in range(game.N):
            P_map[t, i] = xm[game.x_slice(i).start]
        if t % ADMM_EVERY == 0:
            s = time.perf_counter()
            xa = admm_solve(game, th, rho=0.5, max_iter=6000, tol=1e-9).x_stacked
            t_admm += time.perf_counter() - s; n_admm += 1
            for i in range(game.N):
                P_admm[t, i] = xa[game.x_slice(i).start]
            e_max = max(e_max, float(np.max(np.abs(xm - xa))))

    print("\n" + "=" * 74 + "\nRESULTS\n" + "=" * 74)
    from collections import Counter
    vis = Counter(k for k in cr_visits if k is not None)
    print(f"  variational CRs in the map (total)   : {gmap.n_cr}")
    print(f"  distinct CRs VISITED this day        : {len(vis)} / {gmap.n_cr}")
    print(f"  CR visit counts (region: steps)      : {dict(sorted(vis.items()))}")
    print(f"  lookup misses                        : {miss} / {n}")
    print(f"  max |map − ADMM| ({n_admm} checks)          : {e_max:.2e} kW")
    print(f"  online map lookup   : {1e6*t_map/n:.1f} µs/step  ({n} steps, {t_map*1e3:.1f} ms total)")
    print(f"  ADMM ground truth   : {t_admm/max(n_admm,1):.2f} s/step  ({n_admm} steps)")
    print(f"  ⇒ per-step speedup {(t_admm/max(n_admm,1))/max(t_map/n,1e-12):.0e}× ; map = ZERO inter-agent iterations")
    gate = miss == 0 and e_max < 1e-2
    print(f"\n  GATE (map == ADMM, 0 misses): {'PASS ✅' if gate else 'FAIL ❌'}")

    _plot(lam, cs, cw, P_map, P_admm)
    print(f"\n  Figure → {HERE/'results'/'figures'/'case_study_day.png'}")


def _plot(lam, cs, cw, Pm, Pa):
    out = HERE / "results" / "figures"; out.mkdir(parents=True, exist_ok=True)
    h = np.arange(len(lam)) * 0.25
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(h, lam, "k-", lw=1.2); ax[0].axhline(60, color="r", ls=":", lw=0.8, label="break-even ~$60")
    ax[0].set_ylabel("λ_RT [$/MWh]"); ax[0].legend(fontsize=8); ax[0].set_title(f"ERCOT {DAY} — RTM price")
    ax[1].plot(h, cs, label="solar CF"); ax[1].plot(h, cw, label="wind CF")
    ax[1].set_ylabel("capacity factor"); ax[1].legend(fontsize=8)
    for i, f in enumerate(FLEET):
        ax[2].plot(h, Pm[:, i], "-", lw=1.6, label=f[0])
        ax[2].plot(h, Pa[:, i], "o", ms=2.5, mfc="none", alpha=0.6)
    ax[2].axhline(L_MIN, color="g", ls="--", lw=0.7); ax[2].axhline(L_MAX, color="purple", ls="--", lw=0.7)
    ax[2].set_ylabel("grid buy p [kW]"); ax[2].set_xlabel("hour of day"); ax[2].legend(fontsize=7, ncol=3)
    ax[2].set_title("Per-agent dispatch — lines = explicit map, ○ = ADMM")
    fig.tight_layout(); fig.savefig(out / "case_study_day.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
