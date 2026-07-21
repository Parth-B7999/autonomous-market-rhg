"""
dam.py — Distributed day-ahead market (ADMM) for the 6-agent mp-GNE case study.

Each agent solves ONLY its own 24-hour QP; the sole shared quantity is the per-hour
coalition import Σ_i p_{i,h}, projected onto the coupling band [L_MIN, L_MAX] by the ADMM
consensus step. No agent reveals its cost model — the same privacy guarantee as the RTM.

SINGLE SOURCE OF TRUTH: the fleet and the PCC coupling band (L_MIN=100, L_MAX=900) are
imported from rhg_mpqp (the locked spec, FORMULATION.md), so the day-ahead and real-time
stages always model the *same* physical system.

Provides
  load(day)               — ERCOT day (DAM hourly λ, RTM 15-min λ, solar/wind CF)
  solve_dam_admm(...)     — the LIVE distributed day-ahead solve (returns p_DA (N,24))
  solve_dam_centralized() — centralized solve of the SAME regularized problem; OFFLINE
                            validation oracle only (never in deployment)
  build_dam_game(...)     — the 24-hour GNEGame the ADMM runs on
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
import rhg_mpqp as R
from amrhg.solvers.game import Agent, GNEGame
from amrhg.solvers.admm_solver import admm_solve

# ── shared physical spec (single source of truth = rhg_mpqp / FORMULATION.md) ──
FLEET = R.FLEET                        # 7-tuple (name,type,pmax,rcap,eta,gamma,dmax)
N = R.N
L_MIN, L_MAX = R.L_MIN, R.L_MAX        # 100, 900 kW — SAME PCC band as the RTM
R_H2 = R.R_H2                          # $3/kg
EPS_CV = R.EPS_CV
LAM_BOX = R.LAM_BOX

# ── day-ahead-specific constants ──
DT_DA = 1.0                            # hourly day-ahead step [h]
GAMMA_DA = 1e-4                        # strict-convexity regularizer on grid import p
H2_FRAC = 0.55                         # daily H2 target = frac · max producible
SEED = 42
WIND_CAP_MW, SOLAR_CAP_MW = 40000, 32000   # ERCOT system nameplate (for capacity factors)


def _fleet5(i):
    """(name, type, p_max, ren_cap, eta) — the DAM only needs these five."""
    nm, typ, pmax, rcap, eta = R.FLEET[i][:5]
    return nm, typ, pmax, rcap, eta


# ── DAM (distributed): each agent's own 24h QP; only the per-hour aggregate is ──
# shared. Coupling L_MIN ≤ Σ_i p_{i,h} ≤ L_MAX per hour is enforced by ADMM z/λ.
def build_dam_game(lam_da, cf_s_da, cf_w_da, d_day):
    """24-hour DAM as a GNEGame (n_p=0: prices/CF baked in as constants)."""
    H = 24
    c_p = DT_DA * (lam_da / 1000.0)                        # energy cost coeff per hour
    agents = []
    for i in range(N):
        nm, typ, pmax, rcap, eta = _fleet5(i)
        a = R_H2 * eta
        if typ == "grid":
            nx = H
            Q = GAMMA_DA * np.eye(H)
            c = c_p - DT_DA * a                            # (H,)  min Σ (λ/1000 − a)Δt p
            F = np.zeros((nx, 0))
            # local: 0 ≤ p_h ≤ pmax  (2H rows)  +  H2 floor  −ηΔt Σ p ≤ −D_day
            A = np.vstack([np.eye(H), -np.eye(H), -eta * DT_DA * np.ones((1, H))])
            b = np.concatenate([pmax * np.ones(H), np.zeros(H), [-d_day[i]]])
            S = np.zeros((A.shape[0], 0))
            C = np.eye(H)                                  # p_h → hour-h coupling
            agents.append(Agent(index=i, n_x=nx, Q=Q, c=c, F=F,
                                A_loc=A, b_loc=b, S_loc=S, C=C))
        else:
            nx = 2 * H                                     # [p_0..23, cv_0..23]
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
    return GNEGame(agents=agents,
                   d=L_MAX * np.ones(H), S_coup=np.zeros((H, 0)),
                   d_lb=L_MIN * np.ones(H), S_coup_lb=np.zeros((H, 0)),
                   p_lb=np.array([]), p_ub=np.array([]))


def solve_dam_admm(lam_da, cf_s_da, cf_w_da, d_day, rho=0.1, max_iter=5000, tol=1e-3):
    """Distributed DAM. Returns (p_da (N,24), ADMMResult)."""
    game = build_dam_game(lam_da, cf_s_da, cf_w_da, d_day)
    res = admm_solve(game, np.array([]), rho=rho, max_iter=max_iter,
                     tol=tol, qp_solver="osqp")
    p_da = np.array([res.x_sol[i][:24] for i in range(N)])               # (N,24)
    return p_da, res


def solve_dam_centralized(lam_da, cf_s_da, cf_w_da, d_day):
    """Centralized solve of the SAME regularized DAM — OFFLINE validation oracle only."""
    H = 24
    idx = []; off = 0
    for i in range(N):
        nm, typ, pmax, rcap, eta = _fleet5(i)
        n = H * (2 if typ != "grid" else 1); idx.append((off, off + n, typ)); off += n
    nv = off
    def unpack(z):
        out = []
        for (s, e, typ) in idx:
            blk = z[s:e]
            out.append((blk, None) if typ == "grid" else (blk[:H], blk[H:]))
        return out
    def pelec(i, p, cv):
        nm, typ, pmax, rcap, eta = _fleet5(i)
        g = (cf_s_da if typ == "solar" else cf_w_da) * rcap if typ != "grid" else np.zeros(H)
        return p + g - (cv if cv is not None else 0.0)
    def obj(z):
        # Same objective the distributed agents solve: linear energy−H2 + γ_DA on p (+ ε on cv).
        u = unpack(z); J = 0.0
        for i, (p, cv) in enumerate(u):
            nm, typ, pmax, rcap, eta = _fleet5(i); a = R_H2 * eta
            J += np.sum(DT_DA * (lam_da / 1000 - a) * p) + 0.5 * GAMMA_DA * np.sum(p * p)
            if cv is not None: J += np.sum(DT_DA * a * cv) + 0.5 * EPS_CV * np.sum(cv * cv)
        return float(J)
    cons = []; bnds = []
    for i, (s, e, typ) in enumerate(idx):
        nm, tp, pmax, rcap, eta = _fleet5(i)
        for h in range(H): bnds.append((0., pmax))
        if typ != "grid":
            for h in range(H): bnds.append((0., rcap))
    def make(z): return unpack(z)
    for h in range(H):
        cons.append({'type': 'ineq', 'fun': (lambda z, h=h: sum(make(z)[i][0][h] for i in range(N)) - L_MIN)})
        cons.append({'type': 'ineq', 'fun': (lambda z, h=h: L_MAX - sum(make(z)[i][0][h] for i in range(N)))})
    for i in range(N):
        nm, typ, pmax, rcap, eta = _fleet5(i)
        cons.append({'type': 'ineq', 'fun': (lambda z, i=i, eta=eta: np.sum(pelec(i, *make(z)[i]) * eta * DT_DA) - d_day[i])})
        if typ != "grid":
            for h in range(H):
                cons.append({'type': 'ineq', 'fun': (lambda z, i=i, h=h, pmax=pmax: pmax - pelec(i, *make(z)[i])[h])})
                cons.append({'type': 'ineq', 'fun': (lambda z, i=i, h=h: pelec(i, *make(z)[i])[h])})
    z0 = np.zeros(nv)
    for (s, e, typ) in idx: z0[s:s + H] = L_MIN / N
    r = minimize(obj, z0, bounds=bnds, constraints=cons, method='SLSQP', options={'ftol': 1e-7, 'maxiter': 500})
    u = unpack(r.x)
    p_da = np.array([u[i][0] for i in range(N)])           # (N,24)
    return p_da, r


def load(day):
    d = HERE.parent / "data" / "ercot"
    da = pd.read_csv(d / "ercot_dam_lmp_2025.csv")
    da = da[(da.deliveryDate == day) & (da.settlementPoint == "HB_HUBAVG")].sort_values("hourEnding")
    lam_da = da.settlementPointPrice.to_numpy(float)[:24]
    rtm = pd.read_csv(d / "ercot_rtm_lmp_2025.csv")
    rtm = rtm[(rtm.deliveryDate == day) & (rtm.settlementPoint == "HB_HUBAVG")].sort_values(["deliveryHour", "deliveryInterval"])
    lam_rt = rtm.settlementPointPrice.to_numpy(float)
    sol = pd.read_csv(d / "ercot_solar_production_2025.csv"); win = pd.read_csv(d / "ercot_wind_production_2025.csv")
    cs_h = sol[sol.deliveryDate == day].sort_values("hourEnding").genSystemWide.to_numpy(float) / SOLAR_CAP_MW
    cw_h = win[win.deliveryDate == day].sort_values("hourEnding").genSystemWide.to_numpy(float) / WIND_CAP_MW
    return lam_da[:24], lam_rt[:96], cs_h[:24], cw_h[:24]


# ── standalone DAM check: python dam.py [YYYY-MM-DD] → convergence + gate vs oracle ──
def main():
    import time
    day = sys.argv[1] if len(sys.argv) > 1 else "2025-04-01"
    rng = np.random.default_rng(SEED)
    lam_da, lam_rt, cs_h, cw_h = load(day)
    d_day = np.array([H2_FRAC * _fleet5(i)[2] * _fleet5(i)[4] * 24 for i in range(N)])
    lam_da_fc = np.maximum(lam_da + rng.normal(0, 16, 24), -50)
    cs_fc = np.clip(cs_h + rng.normal(0, 0.20, 24), 0, 1)
    cw_fc = np.clip(cw_h + rng.normal(0, 0.20, 24), 0, 0.8)
    print(f"[DAM] distributed ADMM (band {L_MIN:.0f}-{L_MAX:.0f} kW) — ERCOT {day}")
    t0 = time.perf_counter(); p_da, res = solve_dam_admm(lam_da_fc, cs_fc, cw_fc, d_day)
    print(f"    solved ({time.perf_counter()-t0:.1f}s, converged={res.converged}, iters={res.n_iter}); "
          f"Σp_DA/hr = [{p_da.sum(0).min():.0f},{p_da.sum(0).max():.0f}] kW")
    p_c, _ = solve_dam_centralized(lam_da_fc, cs_fc, cw_fc, d_day)
    agg = float(np.max(np.abs(p_da.sum(0) - p_c.sum(0))))
    h2 = np.zeros(N)
    for i in range(N):
        nm, typ, pmax, rcap, eta = _fleet5(i); xi = res.x_sol[i]
        pe = np.clip(xi[:24], 0, pmax) if typ == "grid" else \
             np.clip(xi[:24] + (cs_fc if typ == "solar" else cw_fc) * rcap - xi[24:48], 0, pmax)
        h2[i] = eta * DT_DA * np.sum(pe)
    ok = bool(np.all(h2 >= d_day - 1e-2))
    print(f"    [gate vs centralized] Σp/hr err={agg:.2f} kW, H2 floors {'met' if ok else 'MISSED'} "
          f"→ {'PASS ✅' if (agg < 30 and ok) else 'CHECK'}")


if __name__ == "__main__":
    main()
