"""
run_experiment.py — one closed-loop simulation run (FOCAPO-CPC 2027 talk experiments).

Defaults reproduce the paper configuration (7 days from 2024-07-08, seed 42,
L_min = 100 kW, no L_max, rho_rt = 0.05, rho_da = 1e-4, tol 1 kW).

Example:
    python run_experiment.py --scenario gne --out results/paper_gne

Outputs in --out:
    financials.csv   per agent + TOTAL: H2 kg, energy cost $, H2 revenue $,
                     net profit $, curtailment kWh, avg import kW
    results.npz      full time series (same arrays as case_study_*.py) + solver stats
    run_params.json  all arguments, fleet, versions, git commit
    admm_stats.json  iterations, convergence, wall-clock per step
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'med_scale_case_study'))   # med modules must win

import simulation_sc                                      # noqa: E402
from agents import ElzAgent, DT_RT                        # noqa: E402
import agents_med as P                                    # noqa: E402

# Options that later phases implement. Phase A accepts only the paper values.
_NOT_YET = {
    'scenario': ({'central'}, 'Phase B5'),
    'commit_mode': ({'admm'}, 'Phase B1'),
    'baseline': ({'fixed'}, 'Phase B2-B4'),
}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--scenario', choices=['individual', 'gne', 'central'], default='gne')
    ap.add_argument('--start-date', default=P.START_DATE)
    ap.add_argument('--n-days', type=int, default=P.N_DAYS)
    ap.add_argument('--seed', type=int, default=P.SEED)
    ap.add_argument('--noise-scale', type=float, default=1.0,
                    help='multiplies all forecast noise std devs (0 = perfect forecast)')
    ap.add_argument('--commit-mode', choices=['central', 'admm'], default='central',
                    help='GNE hourly commitment: centralized MIQP (paper) or ADMM')
    ap.add_argument('--baseline', choices=['paper', 'fixed'], default='paper')
    ap.add_argument('--agents', default='all',
                    help="agent subset as 0-based indices, e.g. '0,2,5' (default all 6)")
    ap.add_argument('--l-min', type=float, default=P.L_MIN)
    ap.add_argument('--rho-rt', type=float, default=P.RHO_RT)
    ap.add_argument('--rho-da', type=float, default=P.RHO_DA)
    ap.add_argument('--tol-rt', type=float, default=P.TOL_RT)
    ap.add_argument('--tol-da', type=float, default=P.TOL_DA)
    ap.add_argument('--out', default=None, help='output folder (default results/<auto name>)')
    args = ap.parse_args(argv)

    for key, (bad, phase) in _NOT_YET.items():
        if getattr(args, key) in bad:
            ap.error(f"--{key.replace('_', '-')} {getattr(args, key)} is not implemented yet ({phase})")
    if args.noise_scale != 1.0:
        ap.error('--noise-scale is not implemented yet (Phase C)')

    if args.agents == 'all':
        args.agent_idx = list(range(len(P.FLEET_DICTS)))
    else:
        args.agent_idx = sorted({int(x) for x in args.agents.replace(' ', ',').split(',') if x})
        if not args.agent_idx or not all(0 <= i < len(P.FLEET_DICTS) for i in args.agent_idx):
            ap.error(f'--agents must be indices in 0..{len(P.FLEET_DICTS) - 1}')
    if args.out is None:
        tag = '' if args.agents == 'all' else '_S' + ''.join(map(str, args.agent_idx))
        args.out = str(ROOT / 'results' /
                       f'{args.scenario}_{args.baseline}_{args.start_date}_s{args.seed}{tag}')
    return args


# ── solver-call recorder (timing + iterations, no change to the solver) ────────
class _Recorder:
    def __init__(self):
        self.calls = []   # (kind, n_iter, converged, wall_s)

    def wrap(self, fn, kind_of):
        def inner(*a, **k):
            t0 = time.perf_counter()
            out = fn(*a, **k)
            wall = time.perf_counter() - t0
            self.calls.append((kind_of(a, k), int(getattr(out, 'n_iter', 1)),
                               bool(getattr(out, 'converged', True)), wall))
            return out
        return inner

    def of(self, kind):
        c = [x for x in self.calls if x[0] == kind]
        return (np.array([x[1] for x in c], dtype=int),
                np.array([x[2] for x in c], dtype=bool),
                np.array([x[3] for x in c], dtype=float))


def _install_recorder() -> _Recorder:
    rec = _Recorder()
    mode = lambda a, k: k.get('mode', 'commit')          # noqa: E731
    simulation_sc.solve_gne = rec.wrap(simulation_sc.solve_gne, mode)
    simulation_sc.solve_individual = rec.wrap(simulation_sc.solve_individual, mode)
    simulation_sc.commit_u_gne = rec.wrap(simulation_sc.commit_u_gne, lambda a, k: 'commit')
    simulation_sc.commit_u_individual = rec.wrap(simulation_sc.commit_u_individual,
                                                 lambda a, k: 'commit')
    return rec


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(['git', 'diff', '--quiet', '--', '.'], cwd=ROOT,
                                stderr=subprocess.DEVNULL) != 0
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def main(argv=None) -> None:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dicts = [P.FLEET_DICTS[i] for i in args.agent_idx]
    fleet = [ElzAgent(h2_target_kg_day=0.0, **d) for d in dicts]
    rec = _install_recorder()

    print(f'scenario={args.scenario} baseline={args.baseline} commit={args.commit_mode} '
          f'start={args.start_date} days={args.n_days} seed={args.seed} '
          f'agents={[d["name"] for d in dicts]} -> {out}')
    t0 = time.perf_counter()
    if args.scenario == 'gne':
        res, agents = simulation_sc.run_gne(
            fleet, n_days=args.n_days, start_date=args.start_date, seed=args.seed,
            rho=args.rho_rt, rho_da=args.rho_da, tol_rt=args.tol_rt, tol_da=args.tol_da,
            l_min=args.l_min, l_max=float('inf'), h2_profile=P.H2_DEMAND_PROFILE)
    else:
        res, agents = simulation_sc.run_individual(
            fleet, n_days=args.n_days, start_date=args.start_date, seed=args.seed,
            l_min=args.l_min, h2_profile=P.H2_DEMAND_PROFILE)
    wall_total = time.perf_counter() - t0

    # ── financials.csv ────────────────────────────────────────────────────────
    rows = []
    for i, ag in enumerate(agents):
        ec = float(res.cost_rt[:, :, i].sum())
        rev = float((res.p_elec[:, :, i] * ag.eta_kg_per_kwh * DT_RT * ag.r_h2).sum())
        rows.append(dict(agent=ag.name, h2_kg=float(res.h2_daily[:, i].sum()),
                         energy_cost=ec, h2_revenue=rev, net_profit=rev - ec,
                         curtail_kwh=float(res.curtail[:, :, i].sum() * DT_RT),
                         avg_import_kw=float(res.p_net[:, :, i].mean())))
    tot = dict(agent='TOTAL', **{k: sum(r[k] for r in rows)
                                  for k in rows[0] if k != 'agent'})
    rows.append(tot)
    cols = ['agent', 'h2_kg', 'energy_cost', 'h2_revenue', 'net_profit',
            'curtail_kwh', 'avg_import_kw']
    with open(out / 'financials.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['agent', 'h2_kg', 'energy_cost_$', 'h2_revenue_$', 'net_profit_$',
                    'curtail_kwh', 'avg_import_kw'])
        for r in rows:
            w.writerow([r['agent']] + [f'{r[c]:.4f}' for c in cols[1:]])

    # ── solver stats ──────────────────────────────────────────────────────────
    rt_it, rt_cv, rt_wall = rec.of('rtm')
    da_it, da_cv, da_wall = rec.of('dam')
    _, _, cm_wall = rec.of('commit')
    stats = dict(
        wall_total_s=wall_total,
        n_rt_steps=int(res.rt_iters.size),
        rt_conv_pct=float(res.rt_conv.mean() * 100),
        rt_iter_mean=float(res.rt_iters.mean()),
        rt_iter_median=float(np.median(res.rt_iters)),
        rt_iter_max=int(res.rt_iters.max()),
        rt_wall_mean_s=float(rt_wall.mean()) if rt_wall.size else None,
        rt_wall_max_s=float(rt_wall.max()) if rt_wall.size else None,
        da_conv_pct=float(res.da_conv.mean() * 100),
        da_iter_mean=float(res.da_iters.mean()),
        da_wall_mean_s=float(da_wall.mean()) if da_wall.size else None,
        commit_calls=int(cm_wall.size),
        commit_wall_total_s=float(cm_wall.sum()),
        agg_below_lmin_steps=int(((res.agg > 1e-6) & (res.agg < args.l_min - 1e-6)).sum()),
        agg_min_positive_kw=float(res.agg[res.agg > 1e-6].min()) if (res.agg > 1e-6).any() else None,
    )
    with open(out / 'admm_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # ── results.npz ───────────────────────────────────────────────────────────
    np.savez_compressed(
        out / 'results.npz',
        p_net=res.p_net, p_elec=res.p_elec, p_ch=res.p_ch, p_dis=res.p_dis,
        bat_soc=res.bat_soc, ren=res.ren, h2_step=res.h2_step, h2_cum=res.h2_cum,
        h2_daily=res.h2_daily, h2_target=res.h2_target,
        lmp_rt=res.lmp_rt, lmp_da=res.lmp_da, cost_rt=res.cost_rt,
        p_da=res.p_da, agg=res.agg, lmin_bind=res.lmin_bind, lmax_bind=res.lmax_bind,
        curtail=res.curtail, participation=res.participation,
        agent_names=[ag.name for ag in agents],
        da_iters=res.da_iters, da_conv=res.da_conv,
        rt_iters=res.rt_iters, rt_conv=res.rt_conv,
        rt_wall_s=rt_wall, da_wall_s=da_wall, commit_wall_s=cm_wall,
    )

    # ── run_params.json ───────────────────────────────────────────────────────
    import gurobipy
    params = dict(
        args={k: v for k, v in vars(args).items()},
        l_max='inf', h2_price=P.H2_PRICE, h2_profile=P.H2_DEMAND_PROFILE,
        h2_rtm_margin=simulation_sc.H2_RTM_MARGIN,
        fleet=dicts,
        git_commit=_git_commit(),
        timestamp=datetime.now().isoformat(timespec='seconds'),
        python=platform.python_version(), numpy=np.__version__,
        gurobi='.'.join(map(str, gurobipy.gurobi.version())),
        host=platform.node(),
    )
    with open(out / 'run_params.json', 'w') as f:
        json.dump(params, f, indent=2, default=str)

    print(f"\nFleet net profit ${tot['net_profit']:.2f} | cost ${tot['energy_cost']:.2f} | "
          f"H2 {tot['h2_kg']:.1f} kg | curtail {tot['curtail_kwh']:.1f} kWh | "
          f"RT conv {stats['rt_conv_pct']:.1f}% mean-iter {stats['rt_iter_mean']:.2f} | "
          f"wall {wall_total:.0f}s")


if __name__ == '__main__':
    main()
