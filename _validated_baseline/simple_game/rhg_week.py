"""
rhg_week.py — receding-horizon H=4 closed loop over ERCOT, mp-GNE RTM via ONLINE FACET.

Per day: real DAM solve (H2 daily floor) on forecast → hourly p_DA_i + daily H2 target.
RTM receding loop (every 15 min): compute the paced window demand D_i(t) (receding H2
state), assemble public θ_t = [D(6), λ_RT nowcast(4), p_DA(6), g_solar(4), g_wind(4)],
solve the GNE by ONLINE FACET (warm-started), apply step-0, execute vs realized, track H2.
Also proves the horizon COUPLES (∂p_{i,0}/∂λ_1 ≠ 0 when H2 binds).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src")); sys.path.insert(0, str(HERE))
import rhg_mpqp as R
import rhg_online as O
import dam as F                   # distributed DAM solve, ERCOT load (band/fleet from rhg_mpqp)

DT = R.DT; H = R.H; N = R.N
D0, LAM0, PDA0, GS0, GW0 = R.D_START, R.LAM_START, R.PDA_START, R.GS_START, R.GW_START


def theta_pgne(D, lam4, p_da, cs4, cw4):
    """Assemble the 24-dim public parameter vector."""
    th = np.zeros(R.N_P_GNE)
    th[D0:D0 + N] = D
    th[LAM0:LAM0 + H] = np.clip(lam4, *R.LAM_BOX)
    th[PDA0:PDA0 + N] = p_da
    th[GS0:GS0 + H] = np.clip(cs4, 0, 1) * 125.0      # solar output = CF × PV cap
    th[GW0:GW0 + H] = np.clip(cw4, 0, 0.8) * 250.0    # wind output  = CF × wind cap
    return th


def run_day(day, sols, game, rng, verbose=True):
    lam_da, lam_rt, cs_h, cw_h = F.load(day)
    d_day = np.array([F.H2_FRAC * R._pmax(i) * R._eta(i) * 24 for i in range(N)])
    # DAM on forecast → hourly p_DA
    lam_da_fc = np.maximum(lam_da + rng.normal(0, 16, 24), -50)
    cs_fc = np.clip(cs_h + rng.normal(0, 0.20, 24), 0, 1)
    cw_fc = np.clip(cw_h + rng.normal(0, 0.20, 24), 0, 0.8)
    import time
    t_dam = time.perf_counter()
    p_da, _ = F.solve_dam_admm(lam_da_fc, cs_fc, cw_fc, d_day)     # (N,24) distributed DAM
    t_dam = time.perf_counter() - t_dam

    lam_rt_fc = np.zeros(96)
    h2 = np.zeros(N); P = np.zeros((96, N))
    curt = np.zeros(N); ren_avail = np.zeros(N)   # curtailed / available renewable [kWh]
    # t=0 seed = PUBLIC DAM aggregate (the cleared day-ahead position is public); each
    # agent's grid import initialized to its DAM award p_DA[i,0], cv=0.  The seed combo is
    # located from that public aggregate — NO centralized solve.
    prev_x = np.zeros(game.n_x_total)
    for i in range(N):
        st = game.x_slice(i).start
        prev_x[st:st + H] = p_da[i, 0]
    prev_combo = None
    e_cent = 0.0; n_cent = 0; miss = 0; t_on = 0.0; crhits = Counter(); binds = 0
    comm = {}          # data-transfer accounting (see rhg_online.solve_step docstring)
    for t in range(96):
        if t % 20 == 0:
            print(f"  Step {t}", flush=True)
        h = t // 4
        # paced receding window demand D_i(t): produce remaining evenly over remaining steps
        rem_steps = max(96 - t, H)          # 96 fifteen-min steps in a day (was 288 — bug)
        D = np.clip((d_day - h2) * H / rem_steps, 0, [R._dmax(i) for i in range(N)])
        if np.any(D > 1e-6):
            binds += 1
        # nowcast λ and renewable over the H-step lookahead
        idx = np.minimum(np.arange(t, t + H), 95)
        lam4 = lam_rt[idx] + rng.normal(0, 8, H)
        lam_rt_fc[t] = lam4[0]
        cs4 = np.full(H, cs_h[h]) + rng.normal(0, 0.05, H)
        cw4 = np.full(H, cw_h[h]) + rng.normal(0, 0.08, H)
        th = theta_pgne(D, lam4, p_da[:, h], cs4, cw4)
        s = time.perf_counter()
        # Point-location-first clearing (v-GNE solve) with neighbour walk + ADMM fallback;
        # always returns a decision (no skipped step). Per-step data transfer tracked in comm.
        x, prev_combo = O.solve_step(th, sols, game, prev_x, prev_combo, stats=comm, max_hops=3)
        prev_x = x
        t_on += time.perf_counter() - s
        crhits[tuple(prev_combo)] += 1
        # apply step-0, execute vs realized price/renewable
        for i in range(N):
            p0 = x[game.x_slice(i).start]; P[t, i] = p0
            ty, pmax, rc, eta = R._typ(i), R._pmax(i), R._rcap(i), R._eta(i)
            g = (cs_h[h] if ty == "solar" else cw_h[h]) * rc if ty != "grid" else 0.0
            pe = min(max(0.0, p0 + g), pmax)
            h2[i] += eta * pe * DT
            if ty != "grid":                      # curtailed renewable = can't be absorbed
                curt[i] += max(0.0, p0 + g - pmax) * DT
                ren_avail[i] += g * DT
        if t % 8 == 0:                                     # certify vs centralized
            xc = O.centralized(game, th)
            if xc is not None:
                e_cent = max(e_cent, float(np.max(np.abs(x - xc)))); n_cent += 1
    miss = comm.get('fallback', 0)      # "miss" of the pure map = a step that needed ADMM fallback
    return dict(day=day, h2=h2, d_day=d_day, P=P, e_cent=e_cent, n_cent=n_cent,
                miss=miss, t_on=t_on, t_dam=t_dam, crhits=crhits, binds=binds,
                lam_rt=lam_rt, p_da=p_da, comm=comm, curt=curt, ren_avail=ren_avail,
                lam_da=lam_da, lam_da_fc=lam_da_fc, lam_rt_fc=lam_rt_fc)


def coupling_proof(sols, game, rng):
    """Show ∂p_{i,0}/∂λ_1 ≠ 0 at an H2-binding θ (receding-horizon signature)."""
    # dear prices (agents want out) + H2 demand near max → H2 binds → horizon couples
    D = np.array([R._dmax(i) for i in range(N)])
    lam = np.full(H, 120.0)                     # above break-even → retreat, H2 forces buy
    cs4 = np.full(H, 0.3); cw4 = np.full(H, 0.3)
    p_da = np.array([0.4 * R._pmax(i) for i in range(N)])
    th0 = theta_pgne(D, lam.copy(), p_da, cs4, cw4)
    x0 = O.centralized(game, th0)
    lam2 = lam.copy(); lam2[1] += 40.0          # raise ONLY step-1 price
    th1 = theta_pgne(D, lam2, p_da, cs4, cw4)
    x1 = O.centralized(game, th1)
    if x0 is None or x1 is None:
        return None
    s0 = game.x_slice(0).start
    p_before = x0[s0:s0 + H]; p_after = x1[s0:s0 + H]
    dp = p_after - p_before                     # PEM horizon change (4 steps)
    return dp, float(p_before.sum()), float(p_after.sum())


def main():
    days = sys.argv[1].split(",") if len(sys.argv) > 1 else ["2025-04-06"]
    print("=" * 74); print("RECEDING-HORIZON H=4 mp-GNE (online FACET) over ERCOT"); print("=" * 74)
    sols, game = O.load_and_prepare()
    print(f"loaded: per-agent CRs {[s.n_cr for s in sols]}  n_p={game.n_p}")
    rng = np.random.default_rng(F.SEED)
    cp = coupling_proof(sols, game, rng)
    if cp:
        dp, sb, sa = cp
        coupled = np.max(np.abs(np.delete(dp, 1))) > 1e-3   # any step other than λ-perturbed step-1
        print(f"\n[cross-step coupling] raise λ_1 by 40 at H2-binding θ (PEM):")
        print(f"    Δp over horizon = {np.round(dp,2)} kW   Σp: {sb:.1f}→{sa:.1f} (H2-fixed)")
        print(f"    → step-1 change forces OTHER steps to move ⇒ horizon "
              f"{'COUPLES ✅' if coupled else 'decoupled ❌'} (single-step would be all-zero)")
    res = []
    for day in days:
        t = time.perf_counter(); r = run_day(day, sols, game, rng)
        h2pct = 100 * r["h2"].sum() / r["d_day"].sum()
        cm = r["comm"]
        tr = cm.get('transfers', [])
        n_map = cm.get('map_steps', 0); n_fb = cm.get('fallback', 0)
        print(f"\n{day}: λ[{r['lam_rt'].min():.0f},{r['lam_rt'].max():.0f}]  "
              f"map==cent max {r['e_cent']:.1e} ({r['n_cent']}chk)  ADMM-fallbacks {n_fb}/96  "
              f"H2 {h2pct:.0f}%  distinct combos {len(r['crhits'])}  online {1e3*r['t_on']/96:.1f}ms/step  "
              f"({time.perf_counter()-t:.0f}s)")
        print(f"    DATA TRANSFER: total {sum(tr)} rounds over {len(tr)} steps "
              f"(mean {sum(tr)/max(len(tr),1):.2f}/step); "
              f"map-walk steps {n_map} (1 round each), "
              f"fallback steps {n_fb} ({cm.get('fallback_rounds',0)} ADMM rounds); "
              f"BFS combos checked {cm.get('combos_checked',0)}")
        res.append(r)
    import pickle
    pickle.dump(res, open(HERE.parent / "results" / "rhg_week_results.pkl", "wb"))
    print("\ncached → results/rhg_week_results.pkl")


if __name__ == "__main__":
    main()
