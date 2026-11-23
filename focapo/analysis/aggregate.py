"""
aggregate.py — collect the talk experiments and make tables, figures and SUMMARY.md.
Runs on the laptop from pulled results (no CHTC needed).

    python analysis/aggregate.py <results_root> [--out <dir>]

<results_root> may contain (tar.gz files are extracted in place):
    all_weeks/    {ind_paper,ind_fixed,gne_fixed,central_fixed}_<week>   (CHTC)
    e4/           coal_<agents>_<week>                                  (CHTC)
    stoch_weeks/  {ind,gne}_<week>   stochastic, S=10                   (laptop)
Outputs in --out (default <results_root>/analysis): tidy CSVs, fig*.png/.pdf, SUMMARY.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from scipy.optimize import linprog                                 # noqa: E402

# Fixed categorical order (entity → colour never changes between figures)
C = {'ind_fixed': '#2a78d6', 'gne': '#eb6834', 'central': '#1baf7a', 'ind_paper': '#eda100',
     'shapley': '#e87ba4', 'nash': '#008300', 'stoch': '#6250d6'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
PAPER_GAIN = 3969.94 - 2899.57          # paper single run (noise forecast, paper baseline)

plt.rcParams.update({
    'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 13, 'axes.edgecolor': INK2,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.spines.top': False,
    'axes.spines.right': False, 'legend.frameon': False, 'figure.dpi': 150,
    'savefig.bbox': 'tight', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
})


# ── loading ────────────────────────────────────────────────────────────────────
def _extract(folder: Path) -> None:
    for t in folder.glob('*.tar.gz'):
        if not (folder / t.name[:-7]).exists():
            with tarfile.open(t) as tf:
                tf.extractall(folder, filter='data')


def load_family(root: Path, family: str) -> pd.DataFrame:
    folder = root / family
    if not folder.exists():
        return pd.DataFrame()
    _extract(folder)
    rows = []
    for d in sorted(p for p in folder.iterdir() if p.is_dir() and (p / 'financials.csv').exists()):
        m = re.match(r'(.+)_(\d{4}-\d{2}-\d{2})$', d.name)
        if not m:
            continue
        kind, week = m.groups()
        st = json.load(open(d / 'admm_stats.json')) if (d / 'admm_stats.json').exists() else {}
        for r in csv.DictReader(open(d / 'financials.csv')):
            rows.append(dict(family=family, kind=kind, week=week, agent=r['agent'],
                             h2_kg=float(r['h2_kg']), energy_cost=float(r['energy_cost_$']),
                             h2_revenue=float(r['h2_revenue_$']), net_profit=float(r['net_profit_$']),
                             curtail_kwh=float(r['curtail_kwh']), avg_import_kw=float(r['avg_import_kw']),
                             rt_conv_pct=st.get('rt_conv_pct'), rt_iter_mean=st.get('rt_iter_mean'),
                             wall_s=st.get('wall_total_s'), agg_below_lmin=st.get('agg_below_lmin_steps')))
    return pd.DataFrame(rows)


def _q(s: pd.Series) -> dict:
    s = s.dropna()
    return dict(n=len(s), mean=s.mean(), std=s.std(), p5=s.quantile(.05), p50=s.median(),
                p95=s.quantile(.95), min=s.min(), max=s.max())


def _save(fig, out: Path, name: str) -> None:
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}')
    plt.close(fig)


# ── weekly study (replaces E1 + E2) ────────────────────────────────────────────
def weekly(df: pd.DataFrame, out: Path) -> dict:
    tot = df[df.agent == 'TOTAL'].pivot_table(index='week', columns='kind',
                                              values=['net_profit', 'curtail_kwh', 'h2_kg'])
    need = ['ind_fixed', 'gne_fixed']
    tot = tot[[c for c in tot.columns if c[1] in set(tot.columns.get_level_values(1))]]
    W = pd.DataFrame(index=tot.index)
    P = tot['net_profit']
    W['ind_paper'] = P.get('ind_paper'); W['ind_fixed'] = P['ind_fixed']
    W['gne'] = P['gne_fixed']; W['central'] = P.get('central_fixed')
    W['gain_fixed'] = W.gne - W.ind_fixed
    W['gain_paper'] = W.gne - W.ind_paper
    W['gain_fixed_pct'] = np.where(W.ind_fixed > 0, 100 * W.gain_fixed / W.ind_fixed, np.nan)
    W['curtail_reduction_kwh'] = tot['curtail_kwh']['ind_fixed'] - tot['curtail_kwh']['gne_fixed']
    W['h2_change_kg'] = tot['h2_kg']['gne_fixed'] - tot['h2_kg']['ind_fixed']
    W['gap_central'] = W.gne - W.central
    W['gap_central_pct'] = 100 * W.gap_central / W.central.abs()
    if 'gne_loose' in P:
        W['gne_loose'] = P['gne_loose']
        W['loose_minus_tight'] = W.gne_loose - W.gne
    W = W.dropna(subset=need[:1] + ['gne'])

    ag = df[(df.agent != 'TOTAL') & df.kind.isin(['ind_fixed', 'gne_fixed'])].pivot_table(
        index=['week', 'agent'], columns='kind', values='net_profit')
    ag['gain'] = ag['gne_fixed'] - ag['ind_fixed']
    per_agent = ag['gain'].groupby('agent').agg(['mean', 'std', 'min', 'max',
                                                 lambda s: (s >= -0.01).mean() * 100])
    per_agent.columns = ['mean_gain', 'std', 'min', 'max', 'pct_weeks_better_off']
    all_better = ag['gain'].groupby('week').apply(lambda s: (s >= -0.01).all())
    W['all_agents_better_off'] = all_better.reindex(W.index)

    W.to_csv(out / 'weekly_fleet.csv')
    ag.reset_index().to_csv(out / 'weekly_per_agent.csv', index=False)
    per_agent.to_csv(out / 'per_agent_gain.csv')
    stats = {k: _q(W[k]) for k in ('gain_fixed', 'gain_fixed_pct', 'gain_paper', 'curtail_reduction_kwh',
                                   'h2_change_kg', 'gap_central', 'gap_central_pct')}
    if 'loose_minus_tight' in W:
        stats['loose_minus_tight'] = _q(W.loose_minus_tight)
    stats['share_weeks_all_better'] = float(W.all_agents_better_off.mean() * 100)
    stats['share_weeks_gne_wins'] = float((W.gain_fixed > 0).mean() * 100)
    pd.DataFrame(stats).T.to_csv(out / 'weekly_stats.csv')

    # fig 1: distribution of weekly gains (fair vs paper baseline) and curtailment reduction
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axs[0]
    data = [W.gain_paper.dropna(), W.gain_fixed.dropna()]
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=INK2),
                    capprops=dict(color=INK2), flierprops=dict(marker='o', ms=5, mfc=INK2, mec='none', alpha=.5))
    for b, c in zip(bp['boxes'], (C['ind_paper'], C['ind_fixed'])):
        b.set(facecolor=c, alpha=.35, edgecolor=c, lw=2)
    ax.axhline(PAPER_GAIN, color=INK2, lw=1.5, ls='--')
    ax.text(1.5, PAPER_GAIN, f'paper week +${PAPER_GAIN:,.0f}', va='bottom', ha='center', fontsize=11, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks([1, 2], ['vs paper\nindividual', 'vs fair\nindividual'])
    ax.set_ylabel('GNE fleet profit gain per week ($)')
    ax.set_title(f'GNE gain over individual ({len(W)} weeks)', loc='left')
    ax = axs[1]
    bp = ax.boxplot([W.curtail_reduction_kwh / 1000], widths=0.45, patch_artist=True,
                    medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                    flierprops=dict(marker='o', ms=5, mfc=INK2, mec='none', alpha=.5))
    bp['boxes'][0].set(facecolor=C['gne'], alpha=.35, edgecolor=C['gne'], lw=2)
    ax.set_xticks([1], ['GNE vs fair\nindividual'])
    ax.set_ylabel('Curtailment avoided per week (MWh)')
    ax.set_title('Renewable curtailment avoided', loc='left')
    fig.tight_layout(); _save(fig, out, 'fig1_weekly_gain')

    # fig 2: gain over the calendar (season effect)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = pd.to_datetime(W.index)
    ax.bar(x, W.gain_fixed, width=5, color=C['gne'], alpha=.85)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel('GNE gain vs fair individual ($/week)')
    ax.set_title('Weekly GNE gain across seasons (Apr 2024 – Nov 2025)', loc='left')
    fig.tight_layout(); _save(fig, out, 'fig2_gain_by_week')

    # fig 5: GNE vs central gap
    if W.central.notna().any():
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.hist(W.gap_central_pct.dropna(), bins=20, color=C['central'], alpha=.8, edgecolor='#fcfcfb')
        med = W.gap_central_pct.median()
        ax.axvline(med, color=INK, lw=2); ax.text(med, ax.get_ylim()[1] * .92, f'  median {med:.1f}%', color=INK)
        ax.set_xlabel('GNE profit vs centralized optimum (%)'); ax.set_ylabel('weeks')
        ax.set_title('How close is the distributed GNE to the central planner?', loc='left')
        fig.tight_layout(); _save(fig, out, 'fig5_central_gap')
    return dict(W=W, stats=stats, per_agent=per_agent)


# ── E4: coalition values, Shapley, core, Nash bargaining ───────────────────────
def coalitions(df: pd.DataFrame, out: Path) -> dict:
    tot = df[df.agent == 'TOTAL'].copy()
    tot['S'] = tot.kind.str.replace('coal_', '', regex=False)
    v_week = tot.pivot_table(index='S', columns='week', values='net_profit')
    v = v_week.mean(axis=1)                                       # average over weeks
    n = 6
    key = lambda S: ''.join(map(str, sorted(S)))                  # noqa: E731
    N = tuple(range(n))
    grand = df[(df.kind == 'coal_012345') & (df.agent != 'TOTAL')]
    names = list(dict.fromkeys(grand.agent))
    x = grand.groupby('agent', sort=False).net_profit.mean().reindex(names).to_numpy()

    subsets = [S for r in range(1, n + 1) for S in combinations(N, r)]
    missing = [key(S) for S in subsets if key(S) not in v.index]
    if missing:
        print(f'  E4: {len(missing)} coalitions missing, e.g. {missing[:3]}')
    vv = {S: float(v.get(key(S), np.nan)) for S in subsets}

    # superadditivity: v(S ∪ T) >= v(S) + v(T) for disjoint S, T
    viol = []
    for S in subsets:
        for T in subsets:
            if S < T and not set(S) & set(T):
                U = tuple(sorted(S + T))
                viol.append(vv[U] - vv[S] - vv[T])
    viol = np.array(viol)
    # Shapley (exact)
    phi = np.zeros(n)
    for i in N:
        for S in [()] + [s for s in subsets if i not in s]:
            if len(S) == n:
                continue
            w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
            vS = vv[S] if S else 0.0
            phi[i] += w * (vv[tuple(sorted(S + (i,)))] - vS)
    # core check of GNE allocation x, and least core
    excess = {S: sum(x[i] for i in S) - vv[S] for S in subsets if len(S) < n}
    worst_S = min(excess, key=excess.get)
    A = [[-1.0 if i in S else 0.0 for i in N] + [-1.0] for S in subsets if len(S) < n]
    b = [-vv[S] for S in subsets if len(S) < n]
    lp = linprog(c=[0] * n + [1], A_ub=A, b_ub=b, A_eq=[[1] * n + [0]], b_eq=[vv[N]],
                 bounds=[(None, None)] * (n + 1), method='highs')
    eps = float(lp.x[-1]) if lp.success else float('nan')
    singles = np.array([vv[(i,)] for i in N])
    nash = singles + (vv[N] - singles.sum()) / n

    alloc = pd.DataFrame({'agent': names, 'alone_v_i': singles, 'gne': x, 'shapley': phi,
                          'nash_bargaining': nash, 'least_core': lp.x[:n] if lp.success else np.nan})
    alloc.to_csv(out / 'e4_allocations.csv', index=False)
    v_week.assign(mean=v).to_csv(out / 'e4_coalition_values.csv')
    res = dict(v_N=vv[N], sum_singles=float(singles.sum()),
               superadditive_pairs_pct=float((viol >= -1e-6).mean() * 100),
               worst_superadditivity=float(viol.min()),
               gne_in_core=bool(excess[worst_S] >= -1e-6),
               core_worst_violation=float(excess[worst_S]), core_worst_coalition=[names[i] for i in worst_S],
               least_core_eps=eps, weeks=list(v_week.columns))
    json.dump(res, open(out / 'e4_summary.json', 'w'), indent=2)

    # fig 4: per-agent allocation
    fig, ax = plt.subplots(figsize=(11, 4.6))
    xs = np.arange(n); w = 0.2
    for j, (col, lab, c) in enumerate([('alone_v_i', 'alone', C['ind_fixed']), ('gne', 'GNE', C['gne']),
                                       ('shapley', 'Shapley', C['shapley']), ('nash_bargaining', 'Nash bargaining', C['nash'])]):
        ax.bar(xs + (j - 1.5) * w, alloc[col], width=w * 0.92, color=c, label=lab)
    ax.set_xticks(xs, [a.replace('_M', '').replace('_', ' ') for a in names], fontsize=11)
    ax.set_ylabel('Net profit per week ($)')
    ax.set_title('Who gets what: GNE outcome vs fair-division benchmarks', loc='left')
    ax.legend(ncol=4, loc='upper left')
    ax.axhline(0, color=INK2, lw=1)
    fig.tight_layout(); _save(fig, out, 'fig4_fairness')
    return dict(alloc=alloc, res=res)


# ── stochastic vs deterministic (replaces E3) ──────────────────────────────────
def stochastic(det: pd.DataFrame, sto: pd.DataFrame, out: Path, tight: bool = False) -> pd.DataFrame | None:
    if sto.empty or det.empty:
        return None
    # compare like with like: tight det GNE (gne_fixed in tight mode) vs stochastic
    # gne_tight; default-tolerance det GNE vs default-tolerance stochastic gne
    d = det[(det.agent == 'TOTAL') & det.kind.isin(['ind_fixed', 'gne_fixed'])]
    d = d.assign(kind=d.kind.map({'ind_fixed': 'ind_det', 'gne_fixed': 'gne_det'}))
    s = sto[(sto.agent == 'TOTAL') & sto.kind.isin(['ind', 'gne_tight' if tight else 'gne'])]
    s = s.assign(kind=s.kind.map({'ind': 'ind_stoch', 'gne': 'gne_stoch', 'gne_tight': 'gne_stoch'}))
    T = pd.concat([d, s]).pivot_table(index='week', columns='kind', values=['net_profit', 'curtail_kwh'])
    T = T.loc[T.index.isin(s.week.unique())]
    T.to_csv(out / 'stoch_vs_det.csv')
    P = T['net_profit']
    fig, ax = plt.subplots(figsize=(11, 4.6))
    xs = np.arange(len(P)); w = 0.2
    spec = [('ind_det', 'individual, deterministic', C['ind_fixed'], None),
            ('ind_stoch', 'individual, stochastic', C['ind_fixed'], '//'),
            ('gne_det', 'GNE, deterministic', C['gne'], None),
            ('gne_stoch', 'GNE, stochastic', C['gne'], '//')]
    for j, (k, lab, c, h) in enumerate(spec):
        if k in P:
            ax.bar(xs + (j - 1.5) * w, P[k], width=w * .92, color=c if h is None else '#fcfcfb',
                   edgecolor=c, hatch=h, lw=1.5, label=lab)
    ax.set_xticks(xs, P.index)
    ax.set_ylabel('Fleet net profit per week ($)')
    ax.set_title('Deterministic vs stochastic (10 scenarios) planning', loc='left')
    ax.legend(ncol=2, loc='upper left', fontsize=11)
    ax.axhline(0, color=INK2, lw=1)
    fig.tight_layout(); _save(fig, out, 'fig3_stoch_vs_det')
    return T


# ── SUMMARY.md ─────────────────────────────────────────────────────────────────
def summary(out: Path, wk, e4, st) -> None:
    L = ['# FOCAPO-CPC 2027 talk — experiment summary', '']
    if wk:
        s = wk['stats']; W = wk['W']
        f = lambda d, u='$': f"{d['p50']:+,.0f}{'' if u == '$' else u} (mean {d['mean']:+,.0f}, 5–95%: {d['p5']:+,.0f} to {d['p95']:+,.0f})"  # noqa: E731
        L += [f'## All weeks ({len(W)} weeks, Apr 2024 – Nov 2025, real forecaster)', '',
              f"- GNE gain vs **fair** individual, $/week: median {f(s['gain_fixed'])}",
              f"- GNE gain vs fair individual, %: median {s['gain_fixed_pct']['p50']:+.1f}% "
              f"(5–95%: {s['gain_fixed_pct']['p5']:+.1f}% to {s['gain_fixed_pct']['p95']:+.1f}%)",
              f"- GNE gain vs **paper** individual, $/week: median {f(s['gain_paper'])}  (paper week: +${PAPER_GAIN:,.0f})",
              f"- GNE beats the fair individual in {s['share_weeks_gne_wins']:.0f}% of weeks; "
              f"every agent better off in {s['share_weeks_all_better']:.0f}% of weeks",
              f"- Curtailment avoided: median {s['curtail_reduction_kwh']['p50']:,.0f} kWh/week",
              f"- H2 change (GNE − individual): median {s['h2_change_kg']['p50']:+,.0f} kg/week",
              f"- GNE vs central planner: median {s['gap_central_pct']['p50']:+.1f}% "
              f"(5–95%: {s['gap_central_pct']['p5']:+.1f}% to {s['gap_central_pct']['p95']:+.1f}%)"]
        if 'loose_minus_tight' in s:
            L += [f"- GNE with the default ADMM tolerance (tol=1, stops early) minus tight GNE: "
                  f"median {f(s['loose_minus_tight'])} $/week"]
        L += ['',
              '![](fig1_weekly_gain.png)', '', '![](fig2_gain_by_week.png)', '', '![](fig5_central_gap.png)', '',
              '### Per-agent gain (GNE − fair individual), $/week', '',
              wk['per_agent'].round(1).to_markdown(), '']
    if st is not None:
        L += ['## Deterministic vs stochastic (10 scenarios)', '', st['net_profit'].round(0).to_markdown(), '',
              '![](fig3_stoch_vs_det.png)', '']
    if e4:
        r = e4['res']
        L += [f"## Fairness (all 63 coalitions, averaged over {len(r['weeks'])} weeks)", '',
              f"- v(N) = ${r['v_N']:,.0f}/week vs Σ v(i) = ${r['sum_singles']:,.0f}",
              f"- Superadditive for {r['superadditive_pairs_pct']:.0f}% of disjoint pairs (worst {r['worst_superadditivity']:+,.1f} $)",
              f"- GNE allocation in the core: **{'yes' if r['gne_in_core'] else 'no'}** "
              f"(worst Σx−v(S) = {r['core_worst_violation']:+,.1f} $ for {', '.join(r['core_worst_coalition'])})",
              f"- Least-core ε = {r['least_core_eps']:+,.1f} $ (≤ 0 means the core is non-empty)", '',
              e4['alloc'].round(1).to_markdown(index=False), '', '![](fig4_fairness.png)', '']
    (out / 'SUMMARY.md').write_text('\n'.join(L))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('root')
    ap.add_argument('--out', default=None)
    ap.add_argument('--gne', choices=['loose', 'tight'], default='loose',
                    help="'tight': use the GNE runs with ADMM tol 1e-3 (gne_tight/, e4_tight/); "
                         "the default-tolerance GNE stays in the tables as gne_loose")
    a = ap.parse_args(argv)
    root = Path(a.root)
    out = Path(a.out) if a.out else root / ('analysis' if a.gne == 'loose' else 'analysis_tight')
    out.mkdir(parents=True, exist_ok=True)
    det, e4df, sto = (load_family(root, f) for f in ('all_weeks', 'e4', 'stoch_weeks'))
    if a.gne == 'tight':
        gt, e4t = load_family(root, 'gne_tight'), load_family(root, 'e4_tight')
        det = pd.concat([det.assign(kind=det.kind.replace({'gne_fixed': 'gne_loose'})),
                         gt.assign(kind='gne_fixed')])
        e4df = e4t.assign(kind=e4t.kind.str.replace('tcoal_', 'coal_', regex=False))
    pd.concat([det, e4df, sto]).to_csv(out / 'runs_long.csv', index=False)
    wk = weekly(det, out) if not det.empty else None
    co = coalitions(e4df, out) if not e4df.empty else None
    st = stochastic(det, sto, out, tight=(a.gne == 'tight'))
    summary(out, wk, co, st)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
