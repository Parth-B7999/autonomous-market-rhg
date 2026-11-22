"""Phase B comparison table from run_experiment.py output folders.

    python tools/compare_runs.py results/phaseB  [> table.md]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ORDER = ['ind_paper', 'ind_b2', 'ind_fixed', 'gne_paper', 'gne_paper_cadmm', 'gne_b4',
         'gne_fixed_cadmm', 'central_paper', 'central_fixed']


def load(d: Path):
    fin = {r['agent']: r for r in csv.DictReader(open(d / 'financials.csv'))}
    st = json.load(open(d / 'admm_stats.json'))
    z = np.load(d / 'results.npz', allow_pickle=True)
    return fin, st, z


def main(root: str) -> None:
    root = Path(root)
    runs = [n for n in ORDER if (root / n / 'financials.csv').exists()]
    runs += sorted(p.name for p in root.iterdir()
                   if p.is_dir() and p.name not in runs and (p / 'financials.csv').exists())
    data = {n: load(root / n) for n in runs}
    agents = [a for a in next(iter(data.values()))[0] if a != 'TOTAL']

    def f(x, nd=1):
        return '—' if x is None else (f'{x:,.{nd}f}' if isinstance(x, float) else str(x))

    print('### Fleet (7 days, seed 42)\n')
    print('| run | net profit $ | energy cost $ | H2 kg | curtail kWh | in-mkt agent-h '
          '| DA zero-import h | agg<Lmin steps | B4 fixes | RT conv % | RT iters | wall s |')
    print('|' + '---|' * 12)
    for n in runs:
        fin, st, z = data[n]
        t = fin['TOTAL']
        da_zero = int((z['p_da'] <= 1.0).sum())
        print(f"| {n} | {float(t['net_profit_$']):,.2f} | {float(t['energy_cost_$']):,.2f} "
              f"| {float(t['h2_kg']):,.1f} | {float(t['curtail_kwh']):,.1f} "
              f"| {st['agent_hours_in_market']} | {da_zero} | {st['agg_below_lmin_steps']} "
              f"| {st['floor_fix_steps']} | {st['rt_conv_pct']:.1f} | {st['rt_iter_mean']:.2f} "
              f"| {st['wall_total_s']:.0f} |")

    print('\n### Net profit per agent ($)\n')
    print('| run | ' + ' | '.join(agents) + ' |')
    print('|' + '---|' * (len(agents) + 1))
    for n in runs:
        fin = data[n][0]
        print(f'| {n} | ' + ' | '.join(f"{float(fin[a]['net_profit_$']):,.2f}" for a in agents) + ' |')

    print('\n### Solver / timing\n')
    print('| run | RT serial agent s | RT parallel agent s | commit ADMM iters | commit ADMM conv % '
          '| commit agree w/ central % | central agent-dropped steps | central slack steps |')
    print('|' + '---|' * 8)
    for n in runs:
        st = data[n][1]
        print(f"| {n} | {f(st.get('rt_agent_serial_total_s'))} | {f(st.get('rt_agent_parallel_total_s'))} "
              f"| {f(st.get('commit_admm_iter_mean'), 2)} | {f(st.get('commit_admm_conv_pct'))} "
              f"| {f(st.get('commit_agree_with_central_pct'))} "
              f"| {st.get('central_steps_agent_dropped', '—')} | {st.get('central_slack_steps', '—')} |")

    # B1: final participation, admm vs central commitment (different trajectories)
    if {'gne_paper', 'gne_paper_cadmm'} <= set(data):
        a, b = data['gne_paper'][2]['participation'], data['gne_paper_cadmm'][2]['participation']
        print(f"\nB1: final participation u agrees in {(a == b).mean() * 100:.1f}% of "
              f"{a.size} agent-hours (gne_paper vs gne_paper_cadmm).")
    # GNE vs central gap
    for g, c in [('gne_paper', 'central_paper'), ('gne_fixed_cadmm', 'central_fixed')]:
        if {g, c} <= set(data):
            pg = float(data[g][0]['TOTAL']['net_profit_$'])
            pc = float(data[c][0]['TOTAL']['net_profit_$'])
            per = ', '.join(f"{a} {float(data[g][0][a]['net_profit_$']) - float(data[c][0][a]['net_profit_$']):+.0f}"
                            for a in agents)
            print(f"GNE vs central ({g} − {c}): {pg - pc:+,.2f} $ ({(pg - pc) / abs(pc) * 100:+.2f}%); per agent: {per}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/phaseB')
