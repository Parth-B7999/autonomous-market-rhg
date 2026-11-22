"""
compare.py — End-of-week comparison: Individual vs GNE market participation.

Discovers the most recent 'individual_*' and 'gne_*' result directories,
loads results.npz from each, and generates:
  • Side-by-side H₂ production comparison (daily + weekly)
  • Side-by-side financial comparison (daily + weekly)
  • Aggregate import overlay (individual vs GNE)
  • Per-agent import comparison
  • Hourly breakdown figures
  • comparison_summary.csv
  • All comparison figures saved to results/comparison/

Usage:
  python compare.py                          # auto-discover latest runs
  python compare.py <individual_dir> <gne_dir>   # explicit paths
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, date as _date, timedelta
import csv, sys

PROJ    = Path(__file__).parent
RESULTS = PROJ / "results"

_PARENT = str(PROJ.parent / 'admm_elz_market')
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PARENT not in sys.path:
    sys.path.append(_PARENT)  # parent is fallback only (avoid shadowing med-local modules)

from agents import DT_RT
from agents_med import FLEET_DICTS, L_MIN, N_DAYS, START_DATE, H2_DEMAND_PROFILE


def _find_latest(prefix: str) -> Path:
    """Find the most recent result directory matching prefix."""
    dirs = sorted([d for d in RESULTS.iterdir() if d.is_dir() and d.name.startswith(prefix)])
    if not dirs:
        raise FileNotFoundError(f"No result directories matching '{prefix}*' in {RESULTS}")
    return dirs[-1]


def _load_results(run_dir: Path) -> dict:
    """Load results.npz and return as dict."""
    npz = np.load(run_dir / "results.npz", allow_pickle=True)
    return dict(npz)


def _savefig(fig, name: str, out_dir: Path):
    """Save figure as PDF and PNG."""
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches='tight')
    fig.savefig(out_dir / f"{name}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)


def run_comparison(ind_dir: Path, gne_dir: Path):
    """Generate all comparison outputs."""
    out_dir = RESULTS / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    ind = _load_results(ind_dir)
    gne = _load_results(gne_dir)

    agent_names = [d['name'] for d in FLEET_DICTS]
    N = len(agent_names)
    DAYS_FULL = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    print(f"\n{'='*60}")
    print(f"  Comparison: Individual vs GNE")
    print(f"  Individual: {ind_dir.name}")
    print(f"  GNE:        {gne_dir.name}")
    print(f"  Output:     {out_dir}")
    print(f"{'='*60}\n")

    # ════════════════════════════════════════════════════════════════════════
    # 1. H₂ Production Comparison — Daily Breakdown
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Per-agent daily H₂
    x = np.arange(N_DAYS)
    width = 0.35
    for i in range(N):
        ax = axes[0] if i < 3 else axes[1]
        idx = i if i < 3 else i - 3
        offset = idx * (2 * width + 0.1)
        ax.bar(x + offset, ind['h2_daily'][:, i], width, alpha=0.7,
               label=f'{agent_names[i]} (Ind)' if i in [0, 3] else agent_names[i] + ' (Ind)')
        ax.bar(x + offset + width, gne['h2_daily'][:, i], width, alpha=0.7,
               label=f'{agent_names[i]} (GNE)' if i in [0, 3] else agent_names[i] + ' (GNE)')

    for ax_idx, ax in enumerate(axes):
        ax.set_xlabel('Day')
        ax.set_ylabel('H₂ Production [kg]')
        ax.set_xticks(x + width)
        ax.set_xticklabels([DAYS_FULL[d % 7] for d in range(N_DAYS)])
        ax.legend(ncol=3, fontsize=8)
        ax.set_title(f'Daily H₂: Agents {1+ax_idx*3}–{3+ax_idx*3}' )
    fig.suptitle('H₂ Production — Individual vs GNE (Daily)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _savefig(fig, 'compare_h2_daily', out_dir)

    # Fleet total daily H₂
    fig, ax = plt.subplots(figsize=(10, 5))
    tot_ind = ind['h2_daily'].sum(axis=1)
    tot_gne = gne['h2_daily'].sum(axis=1)
    tot_tgt = ind['h2_target'].sum(axis=1)
    ax.bar(x - width/2, tot_ind, width, label='Individual', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, tot_gne, width, label='GNE', color='darkorange', alpha=0.8)
    ax.step(x, tot_tgt, where='mid', color='red', linewidth=2, linestyle='--', label='Target')
    ax.set_xlabel('Day')
    ax.set_ylabel('Fleet H₂ [kg]')
    ax.set_xticks(x)
    ax.set_xticklabels([DAYS_FULL[d % 7] for d in range(N_DAYS)])
    ax.legend()
    ax.set_title('Fleet Total H₂ Production — Individual vs GNE', fontweight='bold')
    fig.tight_layout()
    _savefig(fig, 'compare_h2_fleet_total', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 2. Financial Comparison
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['Energy Cost [$]', 'H₂ Revenue [$]', 'Net Profit [$]']

    for m_idx, (ax, metric_name) in enumerate(zip(axes, metrics)):
        ind_vals = np.zeros(N)
        gne_vals = np.zeros(N)
        for i in range(N):
            ec_ind = float(ind['cost_rt'][:, :, i].sum())
            ec_gne = float(gne['cost_rt'][:, :, i].sum())
            eta = FLEET_DICTS[i]['eta_kg_per_kwh']
            r   = FLEET_DICTS[i].get('r_h2', 3.0)
            rev_ind = float((ind['p_elec'][:, :, i] * eta * DT_RT * r).sum())
            rev_gne = float((gne['p_elec'][:, :, i] * eta * DT_RT * r).sum())
            if m_idx == 0:
                ind_vals[i], gne_vals[i] = ec_ind, ec_gne
            elif m_idx == 1:
                ind_vals[i], gne_vals[i] = rev_ind, rev_gne
            else:
                ind_vals[i], gne_vals[i] = rev_ind - ec_ind, rev_gne - ec_gne

        x_ag = np.arange(N)
        ax.bar(x_ag - width/2, ind_vals, width, label='Individual', color='steelblue', alpha=0.8)
        ax.bar(x_ag + width/2, gne_vals, width, label='GNE', color='darkorange', alpha=0.8)
        ax.set_xticks(x_ag)
        ax.set_xticklabels([n.replace('_M', '') for n in agent_names], rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(metric_name)
        ax.legend(fontsize=8)
        ax.set_title(metric_name, fontweight='bold')

    fig.suptitle('7-Day Financial Comparison — Individual vs GNE', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _savefig(fig, 'compare_financials', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 3. Aggregate Import — Daily Overlay
    # ════════════════════════════════════════════════════════════════════════
    for d in range(N_DAYS):
        fig, ax = plt.subplots(figsize=(14, 4))
        t_hrs = np.arange(288) * 5 / 60

        ax.plot(t_hrs, ind['agg'][d], alpha=0.6, linewidth=0.8,
                color='steelblue', label='Individual (aggregate)')
        ax.plot(t_hrs, gne['agg'][d], alpha=0.6, linewidth=0.8,
                color='darkorange', label='GNE (aggregate)')
        ax.axhline(L_MIN, color='red', linestyle='--', linewidth=1.5, label=f'L_min={L_MIN} kW')

        ax.set_xlabel('Hour')
        ax.set_ylabel('Aggregate Import [kW]')
        ax.set_title(f'Day {d+1} ({DAYS_FULL[d%7]}) — Aggregate Grid Import',
                      fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(0, 24)
        fig.tight_layout()
        _savefig(fig, f'compare_aggregate_day{d+1}', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 4. Per-Agent Import Comparison — Day 1
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    t_hrs = np.arange(288) * 5 / 60
    for i, ax in enumerate(axes.flat):
        ax.plot(t_hrs, ind['p_net'][0, :, i], alpha=0.6, linewidth=0.7,
                color='steelblue', label='Individual')
        ax.plot(t_hrs, gne['p_net'][0, :, i], alpha=0.6, linewidth=0.7,
                color='darkorange', label='GNE')
        ax.axhline(L_MIN, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax.set_title(agent_names[i], fontweight='bold')
        ax.set_ylabel('Grid Import [kW]')
        if i >= 3:
            ax.set_xlabel('Hour')
        ax.set_xlim(0, 24)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle('Day 1 — Per-Agent Grid Import: Individual vs GNE', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _savefig(fig, 'compare_per_agent_day1', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 5. Hourly Breakdown — Energy Cost
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(N_DAYS, 1, figsize=(14, 3 * N_DAYS), sharex=True)
    for d in range(N_DAYS):
        ax = axes[d]
        # Sum cost across agents per step, then aggregate to hourly
        ind_cost_step = ind['cost_rt'][d].sum(axis=1)  # (288,)
        gne_cost_step = gne['cost_rt'][d].sum(axis=1)
        # Cumulative
        ax.plot(t_hrs, np.cumsum(ind_cost_step), color='steelblue', linewidth=1.2,
                label='Individual (cumulative)')
        ax.plot(t_hrs, np.cumsum(gne_cost_step), color='darkorange', linewidth=1.2,
                label='GNE (cumulative)')
        ax.set_ylabel('Cum. Cost [$]')
        ax.set_title(f'Day {d+1} ({DAYS_FULL[d%7]})', fontsize=10)
        if d == 0:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel('Hour')
    fig.suptitle('Cumulative Energy Cost — Individual vs GNE (Hourly)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _savefig(fig, 'compare_hourly_cost', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 6. Renewable Curtailment Comparison
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Per-day total curtailment
    ind_curtail_day = ind['curtail'].sum(axis=2).sum(axis=1) * DT_RT  # kWh per day
    gne_curtail_day = gne['curtail'].sum(axis=2).sum(axis=1) * DT_RT
    axes[0].bar(x - width/2, ind_curtail_day, width, label='Individual', color='steelblue', alpha=0.8)
    axes[0].bar(x + width/2, gne_curtail_day, width, label='GNE', color='darkorange', alpha=0.8)
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Curtailed Energy [kWh]')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([DAYS_FULL[d % 7] for d in range(N_DAYS)])
    axes[0].legend()
    axes[0].set_title('Daily Renewable Curtailment', fontweight='bold')

    # Per-agent total curtailment (7-day)
    ind_curtail_agent = ind['curtail'].sum(axis=(0,1)) * DT_RT
    gne_curtail_agent = gne['curtail'].sum(axis=(0,1)) * DT_RT
    x_ag = np.arange(N)
    axes[1].bar(x_ag - width/2, ind_curtail_agent, width, label='Individual', color='steelblue', alpha=0.8)
    axes[1].bar(x_ag + width/2, gne_curtail_agent, width, label='GNE', color='darkorange', alpha=0.8)
    axes[1].set_xticks(x_ag)
    axes[1].set_xticklabels([n.replace('_M', '') for n in agent_names], rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('Curtailed Energy [kWh]')
    axes[1].legend()
    axes[1].set_title('7-Day Agent Curtailment', fontweight='bold')

    fig.suptitle('Renewable Curtailment — Individual vs GNE', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _savefig(fig, 'compare_curtailment', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 7. Profit Improvement % per Agent
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    profits_ind = np.zeros(N)
    profits_gne = np.zeros(N)
    for i in range(N):
        ec_ind = float(ind['cost_rt'][:, :, i].sum())
        ec_gne = float(gne['cost_rt'][:, :, i].sum())
        eta = FLEET_DICTS[i]['eta_kg_per_kwh']
        r   = FLEET_DICTS[i].get('r_h2', 3.0)
        rev_ind = float((ind['p_elec'][:, :, i] * eta * DT_RT * r).sum())
        rev_gne = float((gne['p_elec'][:, :, i] * eta * DT_RT * r).sum())
        profits_ind[i] = rev_ind - ec_ind
        profits_gne[i] = rev_gne - ec_gne

    pct_change = np.where(
        np.abs(profits_ind) > 1e-3,
        (profits_gne - profits_ind) / np.abs(profits_ind) * 100,
        np.where(profits_gne > 0, np.full(N, 999.0), np.zeros(N))
    )
    short_names = [n.replace('_M', '') for n in agent_names]
    colors_bar  = ['#2ecc71' if v >= 0 else '#e74c3c' for v in pct_change]

    ax = axes[0]
    bars = ax.bar(np.arange(N), pct_change, color=colors_bar, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Net Profit Change [%]')
    ax.set_title('% Profit Improvement: GNE vs Individual', fontweight='bold')
    for bar, val in zip(bars, pct_change):
        label = f'+{val:.0f}%' if val >= 0 else f'{val:.0f}%'
        if abs(val) > 500:
            label = f'+{val:.0f}%'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2 = axes[1]
    abs_change = profits_gne - profits_ind
    colors_abs = ['#2ecc71' if v >= 0 else '#e74c3c' for v in abs_change]
    bars2 = ax2.bar(np.arange(N), abs_change, color=colors_abs, alpha=0.85, edgecolor='white')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(np.arange(N))
    ax2.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Absolute Profit Change [$]')
    ax2.set_title('Absolute Profit Change: GNE vs Individual', fontweight='bold')
    for bar, val in zip(bars2, abs_change):
        label = f'+${val:.0f}' if val >= 0 else f'-${abs(val):.0f}'
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (max(abs_change) * 0.02),
                 label, ha='center', va='bottom', fontsize=8, fontweight='bold')

    fleet_pct = (profits_gne.sum() - profits_ind.sum()) / abs(profits_ind.sum()) * 100
    fig.suptitle(
        f'Coalition Benefit per Agent  |  Fleet: ${profits_gne.sum() - profits_ind.sum():.0f} (+{fleet_pct:.0f}%) more profit with GNE',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    _savefig(fig, 'compare_profit_improvement', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 8. Value Decomposition — cost savings vs revenue change
    # ════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))

    ec_ind_fleet  = float(ind['cost_rt'].sum())
    ec_gne_fleet  = float(gne['cost_rt'].sum())
    rev_ind_fleet = sum(float((ind['p_elec'][:, :, i] * FLEET_DICTS[i]['eta_kg_per_kwh'] * DT_RT
                               * FLEET_DICTS[i].get('r_h2', 3.0)).sum()) for i in range(N))
    rev_gne_fleet = sum(float((gne['p_elec'][:, :, i] * FLEET_DICTS[i]['eta_kg_per_kwh'] * DT_RT
                               * FLEET_DICTS[i].get('r_h2', 3.0)).sum()) for i in range(N))

    cost_saving    = ec_ind_fleet - ec_gne_fleet        # positive = GNE cheaper
    revenue_change = rev_gne_fleet - rev_ind_fleet      # may be negative (less H2)

    categories  = ['Ind.\nProfit', '+ Cost\nSavings', '+ Revenue\nChange', '= GNE\nProfit']
    values      = [profits_ind.sum(), cost_saving, revenue_change, profits_gne.sum()]
    bottoms     = [0, profits_ind.sum(), profits_ind.sum() + cost_saving, 0]
    bar_colors  = ['#3498db', '#2ecc71', '#e74c3c' if revenue_change < 0 else '#2ecc71', '#e67e22']

    for idx, (cat, val, bot, col) in enumerate(zip(categories, values, bottoms, bar_colors)):
        ax.bar(idx, val, bottom=bot, color=col, alpha=0.85, edgecolor='white', width=0.5)
        y_label = bot + val / 2
        ax.text(idx, y_label, f'${val:+.0f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    ax.axhline(profits_ind.sum(), color='gray', linestyle=':', linewidth=1)
    ax.axhline(profits_gne.sum(), color='#e67e22', linestyle=':', linewidth=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Fleet Profit [$]')
    delta_profit = profits_gne.sum() - profits_ind.sum()
    ax.set_title(
        f'Value Decomposition: How GNE Creates ${delta_profit:.0f} Extra Profit\n'
        f'(Energy cost savings outweigh lower H₂ production)',
        fontweight='bold')
    fig.tight_layout()
    _savefig(fig, 'compare_value_decomposition', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 9. Renewable Utilization Rate
    # ════════════════════════════════════════════════════════════════════════
    renewable_agents = [i for i, d in enumerate(FLEET_DICTS) if d.get('renewable_cap_kw', 0) > 0]
    if renewable_agents:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ren_names = [agent_names[i].replace('_M', '') for i in renewable_agents]
        curt_ind  = np.array([float(ind['curtail'][:, :, i].sum() * DT_RT) for i in renewable_agents])
        curt_gne  = np.array([float(gne['curtail'][:, :, i].sum() * DT_RT) for i in renewable_agents])
        ren_avail = np.array([float(ind['ren'][:, :, i].sum() * DT_RT) for i in renewable_agents])

        util_ind = np.where(ren_avail > 0, (1 - curt_ind / ren_avail) * 100, 100.0)
        util_gne = np.where(ren_avail > 0, (1 - curt_gne / ren_avail) * 100, 100.0)

        x_r = np.arange(len(renewable_agents))
        axes[0].bar(x_r - width/2, util_ind, width, label='Individual', color='steelblue', alpha=0.8)
        axes[0].bar(x_r + width/2, util_gne, width, label='GNE',        color='darkorange', alpha=0.8)
        axes[0].set_xticks(x_r)
        axes[0].set_xticklabels(ren_names, rotation=20, ha='right', fontsize=9)
        axes[0].set_ylabel('Renewable Utilization [%]')
        axes[0].set_ylim(0, 110)
        axes[0].axhline(100, color='green', linestyle='--', linewidth=1, alpha=0.6)
        axes[0].legend()
        axes[0].set_title('Renewable Utilization Rate', fontweight='bold')
        for j, (yi, yg) in enumerate(zip(util_ind, util_gne)):
            axes[0].text(j - width/2, yi + 1, f'{yi:.1f}%', ha='center', va='bottom', fontsize=8)
            axes[0].text(j + width/2, yg + 1, f'{yg:.1f}%', ha='center', va='bottom', fontsize=8)

        axes[1].bar(x_r - width/2, curt_ind, width, label='Individual', color='steelblue', alpha=0.8)
        axes[1].bar(x_r + width/2, curt_gne, width, label='GNE',        color='darkorange', alpha=0.8)
        axes[1].set_xticks(x_r)
        axes[1].set_xticklabels(ren_names, rotation=20, ha='right', fontsize=9)
        axes[1].set_ylabel('Curtailed Energy [kWh]')
        axes[1].legend()
        axes[1].set_title('7-Day Curtailment (Renewable Agents)', fontweight='bold')
        for j, (ci, cg) in enumerate(zip(curt_ind, curt_gne)):
            axes[1].text(j - width/2, ci + 20, f'{ci:.0f}', ha='center', va='bottom', fontsize=8)
            axes[1].text(j + width/2, cg + 20, f'{cg:.0f}', ha='center', va='bottom', fontsize=8)

        fig.suptitle(
            f'Renewable Energy Utilization — Individual vs GNE\n'
            f'(Total curtailment: {curt_ind.sum():.0f} kWh → {curt_gne.sum():.0f} kWh, '
            f'{(1 - curt_gne.sum()/curt_ind.sum())*100:.0f}% reduction)',
            fontsize=12, fontweight='bold')
        fig.tight_layout()
        _savefig(fig, 'compare_renewable_utilization', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 10. Daily Energy Cost Savings
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    daily_cost_ind = np.array([float(ind['cost_rt'][d].sum()) for d in range(N_DAYS)])
    daily_cost_gne = np.array([float(gne['cost_rt'][d].sum()) for d in range(N_DAYS)])
    daily_savings  = daily_cost_ind - daily_cost_gne
    day_labels     = [DAYS_FULL[d % 7] for d in range(N_DAYS)]

    ax = axes[0]
    ax.bar(x - width/2, daily_cost_ind, width, label='Individual', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, daily_cost_gne, width, label='GNE',        color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels)
    ax.set_ylabel('Fleet Energy Cost [$]')
    ax.legend()
    ax.set_title('Daily Fleet Energy Cost: Individual vs GNE', fontweight='bold')

    ax2 = axes[1]
    saving_colors = ['#2ecc71' if s >= 0 else '#e74c3c' for s in daily_savings]
    bars = ax2.bar(x, daily_savings, color=saving_colors, alpha=0.85, edgecolor='white')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(day_labels)
    ax2.set_ylabel('Cost Savings [$] (Ind − GNE)')
    ax2.set_title(f'Daily Cost Savings by Switching to GNE  |  7-Day Total: ${daily_savings.sum():.0f}',
                   fontweight='bold')
    for bar, val in zip(bars, daily_savings):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'${val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.tight_layout()
    _savefig(fig, 'compare_energy_cost_savings', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 11. H₂ Efficiency — kg per $ spent on energy
    # ════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 5))

    h2_eff_ind = np.zeros(N)
    h2_eff_gne = np.zeros(N)
    for i in range(N):
        ec_ind_i = float(ind['cost_rt'][:, :, i].sum())
        ec_gne_i = float(gne['cost_rt'][:, :, i].sum())
        h2_ind_i = float(ind['h2_daily'][:, i].sum())
        h2_gne_i = float(gne['h2_daily'][:, i].sum())
        h2_eff_ind[i] = h2_ind_i / ec_ind_i if ec_ind_i > 0 else 0
        h2_eff_gne[i] = h2_gne_i / ec_gne_i if ec_gne_i > 0 else 0

    x_ag = np.arange(N)
    ax.bar(x_ag - width/2, h2_eff_ind, width, label='Individual', color='steelblue', alpha=0.8)
    ax.bar(x_ag + width/2, h2_eff_gne, width, label='GNE',        color='darkorange', alpha=0.8)
    ax.set_xticks(x_ag)
    ax.set_xticklabels(short_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('H₂ Efficiency [kg / $ energy cost]')
    ax.legend()

    fleet_eff_ind = float(ind['h2_daily'].sum()) / float(ind['cost_rt'].sum())
    fleet_eff_gne = float(gne['h2_daily'].sum()) / float(gne['cost_rt'].sum())
    ax.set_title(
        f'H₂ Production Efficiency (kg per $ energy cost)\n'
        f'Fleet avg: Individual={fleet_eff_ind:.3f}  GNE={fleet_eff_gne:.3f}  '
        f'(GNE {(fleet_eff_gne/fleet_eff_ind - 1)*100:.0f}% more efficient)',
        fontweight='bold')
    for j in range(N):
        ax.text(j - width/2, h2_eff_ind[j] + 0.002, f'{h2_eff_ind[j]:.3f}',
                ha='center', va='bottom', fontsize=7)
        ax.text(j + width/2, h2_eff_gne[j] + 0.002, f'{h2_eff_gne[j]:.3f}',
                ha='center', va='bottom', fontsize=7)

    fig.tight_layout()
    _savefig(fig, 'compare_h2_efficiency', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 12. Per-Agent Import — Day 3 (highest coupling activity, 28.5% binding)
    # ════════════════════════════════════════════════════════════════════════
    DAY_HIGH_COUPLING = 2  # Day 3 index (0-based)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    t_hrs = np.arange(288) * 5 / 60
    for i, ax in enumerate(axes.flat):
        ax.plot(t_hrs, ind['p_net'][DAY_HIGH_COUPLING, :, i], alpha=0.7, linewidth=0.9,
                color='steelblue', label='Individual')
        ax.plot(t_hrs, gne['p_net'][DAY_HIGH_COUPLING, :, i], alpha=0.7, linewidth=0.9,
                color='darkorange', label='GNE')
        ax.fill_between(t_hrs, ind['p_net'][DAY_HIGH_COUPLING, :, i],
                        gne['p_net'][DAY_HIGH_COUPLING, :, i],
                        alpha=0.15, color='purple', label='Difference')
        ax.axhline(L_MIN, color='red', linestyle=':', linewidth=1, alpha=0.5, label=f'L_min={L_MIN}kW')
        ax.set_title(agent_names[i], fontweight='bold')
        ax.set_ylabel('Grid Import [kW]')
        ax.set_xlim(0, 24)
        if i >= 3:
            ax.set_xlabel('Hour')
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle(
        f'Day 3 (Wed) — Per-Agent Grid Import: Individual vs GNE\n'
        f'(Highest coupling activity: 28.5% of RT steps at L_min)',
        fontsize=13, fontweight='bold')
    fig.tight_layout()
    _savefig(fig, 'compare_per_agent_day3', out_dir)

    # ════════════════════════════════════════════════════════════════════════
    # 13. All-Days Aggregate Import — 7-panel overview
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    axes_flat = axes.flat
    t_hrs = np.arange(288) * 5 / 60

    for d in range(N_DAYS):
        ax = next(axes_flat)
        ax.plot(t_hrs, ind['agg'][d], alpha=0.7, linewidth=0.9,
                color='steelblue', label='Individual')
        ax.plot(t_hrs, gne['agg'][d], alpha=0.7, linewidth=0.9,
                color='darkorange', label='GNE')
        ax.axhline(L_MIN, color='red', linestyle='--', linewidth=1.2,
                   alpha=0.8, label=f'L_min={L_MIN}kW')
        ax.fill_between(t_hrs, ind['agg'][d], gne['agg'][d],
                        alpha=0.12, color='gray')
        ax.set_title(f'Day {d+1} — {DAYS_FULL[d % 7]}', fontweight='bold')
        ax.set_ylabel('Agg. Import [kW]')
        ax.set_xlim(0, 24)
        if d == 0:
            ax.legend(fontsize=8)

    # Last panel: summary bar chart
    ax = next(axes_flat)
    daily_avg_ind = np.array([float(ind['agg'][d].mean()) for d in range(N_DAYS)])
    daily_avg_gne = np.array([float(gne['agg'][d].mean()) for d in range(N_DAYS)])
    ax.bar(x - width/2, daily_avg_ind, width, label='Individual', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, daily_avg_gne, width, label='GNE',        color='darkorange', alpha=0.8)
    ax.axhline(L_MIN, color='red', linestyle='--', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([DAYS_FULL[d % 7] for d in range(N_DAYS)])
    ax.set_ylabel('Avg. Aggregate Import [kW]')
    ax.legend()
    ax.set_title('Daily Average Aggregate Import', fontweight='bold')

    fig.suptitle('All-Days Aggregate Grid Import: Individual vs GNE', fontsize=14, fontweight='bold')
    fig.tight_layout()
    _savefig(fig, 'compare_aggregate_all7', out_dir)

    print("  New figures: compare_profit_improvement, compare_value_decomposition,")
    print("               compare_renewable_utilization, compare_energy_cost_savings,")
    print("               compare_h2_efficiency, compare_per_agent_day3, compare_aggregate_all7")

    # ════════════════════════════════════════════════════════════════════════
    # 14. Daily Electrolyzer Power Breakdown — Individual vs GNE (all 7 days)
    # ════════════════════════════════════════════════════════════════════════
    COLORS_AGENT = ['#4878d0','#ee854a','#6acc65','#d65f5f','#956cb4','#82c6e2']
    t_hrs_bd = np.arange(288) * 5 / 60

    for d in range(N_DAYS):
        fig, axs = plt.subplots(2, N, figsize=(4 * N, 9), sharex=True)
        row_labels = ['Individual', 'GNE']

        for row_idx, (label, data) in enumerate(zip(row_labels, [ind, gne])):
            for i in range(N):
                ax = axs[row_idx, i]
                pe = data['p_elec'][d, :, i]
                gr = data['ren'][d, :, i]
                pd_ = data['p_dis'][d, :, i]
                pc  = data['p_ch'][d, :, i]
                col = COLORS_AGENT[i]

                ax.fill_between(t_hrs_bd, 0, gr, alpha=0.45, color='gold',     label='Renewable')
                has_bat = FLEET_DICTS[i].get('bat_cap_kwh', 0) > 0
                if has_bat:
                    ax.fill_between(t_hrs_bd, gr, gr + pd_, alpha=0.40,
                                    color='limegreen', label='Bat discharge')
                    ax.fill_between(t_hrs_bd, 0, -pc, alpha=0.35,
                                    color='tomato',    label='Bat charge')
                ax.fill_between(t_hrs_bd, gr + pd_, pe, alpha=0.50,
                                color=col, label='Grid top-up')
                ax.plot(t_hrs_bd, pe, color=col, lw=1.2, label='Total elec.')

                p_max = FLEET_DICTS[i]['p_elec_max_kw']
                ax.axhline(p_max, c='k', ls=':', lw=0.8, label=f'$P^{{max}}$={p_max:.0f}kW')
                ax.grid(True, alpha=0.25)
                ax.set_xlim(0, 24)
                ax.set_xticks([0, 6, 12, 18, 24])
                ax.set_ylabel('Power [kW]')

                if row_idx == 0:
                    ren_cap = FLEET_DICTS[i].get('renewable_cap_kw', 0)
                    bat_cap = FLEET_DICTS[i].get('bat_cap_kwh', 0)
                    ren_t   = FLEET_DICTS[i].get('renewable_type') or ''
                    parts   = []
                    if ren_cap:
                        parts.append(f'{ren_t.upper()} {ren_cap:.0f}kW')
                    if bat_cap:
                        parts.append(f'Bat {bat_cap:.0f}kWh')
                    subtitle = ', '.join(parts) if parts else 'no ren/bat'
                    ax.set_title(f'{agent_names[i]}\n({subtitle})', fontsize=8, fontweight='bold')
                if row_idx == 1:
                    ax.set_xlabel('Hour of Day')
                if i == 0:
                    ax.legend(fontsize=6, loc='upper left')

            axs[row_idx, 0].set_ylabel(f'{label}\nPower [kW]', fontsize=9, fontweight='bold')

        fig.suptitle(
            f'Day {d+1} ({DAYS_FULL[d%7]}) — Electrolyzer Power Breakdown: Individual vs GNE\n'
            f'Gold=Renewable  ·  Green=Bat.Discharge  ·  Red(below)=Bat.Charge  ·  Colour fill=Grid top-up',
            fontsize=12, fontweight='bold')
        fig.tight_layout()
        _savefig(fig, f'compare_elz_breakdown_day{d+1}', out_dir)

    print("  Breakdown figures: compare_elz_breakdown_day1 … compare_elz_breakdown_day7")

    # ════════════════════════════════════════════════════════════════════════
    # 15. Hour-ahead participation commitment (semi-continuous {0}∪[100,Pmax])
    # ════════════════════════════════════════════════════════════════════════
    if ('participation' in ind and 'participation' in gne
            and ind['participation'] is not None and gne['participation'] is not None
            and np.asarray(ind['participation']).ndim == 3):
        u_ind = np.asarray(ind['participation'])   # (D, 24, N)
        u_gne = np.asarray(gne['participation'])

        # Panel 1: participation rate per agent (% of operating hours bidding ≥100)
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        rate_ind = u_ind.reshape(-1, N).mean(axis=0) * 100
        rate_gne = u_gne.reshape(-1, N).mean(axis=0) * 100
        x_ag = np.arange(N)
        axes[0].bar(x_ag - width/2, rate_ind, width, label='Individual', color='steelblue', alpha=0.85)
        axes[0].bar(x_ag + width/2, rate_gne, width, label='GNE',        color='darkorange', alpha=0.85)
        axes[0].set_xticks(x_ag)
        axes[0].set_xticklabels([n.replace('_M', '') for n in agent_names], rotation=30, ha='right', fontsize=9)
        axes[0].set_ylabel('Hours bidding in market [%]')
        axes[0].set_ylim(0, 105)
        axes[0].legend()
        axes[0].set_title('Market Participation Rate per Agent\n(semi-continuous: bid ≥100 kW or stay out)',
                          fontweight='bold')
        for j in range(N):
            axes[0].text(j - width/2, rate_ind[j] + 1, f'{rate_ind[j]:.0f}', ha='center', va='bottom', fontsize=8)
            axes[0].text(j + width/2, rate_gne[j] + 1, f'{rate_gne[j]:.0f}', ha='center', va='bottom', fontsize=8)

        # Panel 2: number of participating agents per hour, fleet-wide (GNE)
        # Aggregate the 7×24 commitment grid into hourly participation count.
        part_count_gne = u_gne.reshape(-1, N).sum(axis=1)   # (D*24,)
        part_count_ind = u_ind.reshape(-1, N).sum(axis=1)
        hrs = np.arange(part_count_gne.size)
        axes[1].plot(hrs, part_count_ind, color='steelblue', lw=1.0, alpha=0.8, label='Individual')
        axes[1].plot(hrs, part_count_gne, color='darkorange', lw=1.2, alpha=0.9, label='GNE')
        axes[1].set_xlabel('Operating hour (over 7 days)')
        axes[1].set_ylabel('# agents bidding in market')
        axes[1].set_ylim(-0.2, N + 0.2)
        axes[1].set_yticks(range(N + 1))
        axes[1].legend(fontsize=9)
        axes[1].set_title('Number of Market Participants per Hour', fontweight='bold')
        for dd in range(1, N_DAYS):
            axes[1].axvline(dd * 24, color='gray', ls=':', lw=0.6, alpha=0.5)

        fig.suptitle('Hour-Ahead Participation Commitment — Individual vs GNE', fontsize=13, fontweight='bold')
        fig.tight_layout()
        _savefig(fig, 'compare_participation', out_dir)
        print("  Participation figure: compare_participation")

    # ════════════════════════════════════════════════════════════════════════
    # 14. Summary CSV
    # ════════════════════════════════════════════════════════════════════════
    with open(out_dir / "comparison_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Agent", "Scenario",
                    "7d_H2_kg", "7d_energy_cost_$", "7d_H2_revenue_$", "7d_net_profit_$",
                    "7d_curtail_kWh", "avg_daily_import_kW"])

        for scenario_name, data in [("Individual", ind), ("GNE", gne)]:
            for i in range(N):
                h2_tot     = float(data['h2_daily'][:, i].sum())
                ec_tot     = float(data['cost_rt'][:, :, i].sum())
                eta        = FLEET_DICTS[i]['eta_kg_per_kwh']
                r          = FLEET_DICTS[i].get('r_h2', 3.0)
                rev_tot    = float((data['p_elec'][:, :, i] * eta * DT_RT * r).sum())
                curt_tot   = float(data['curtail'][:, :, i].sum() * DT_RT)
                avg_import = float(data['p_net'][:, :, i].mean())
                w.writerow([agent_names[i], scenario_name,
                            f"{h2_tot:.1f}", f"{ec_tot:.2f}", f"{rev_tot:.2f}",
                            f"{rev_tot - ec_tot:.2f}", f"{curt_tot:.1f}", f"{avg_import:.1f}"])

            # Fleet total
            h2_tot     = float(data['h2_daily'].sum())
            ec_tot     = sum(float(data['cost_rt'][:, :, i].sum()) for i in range(N))
            rev_tot    = sum(
                float((data['p_elec'][:, :, i]
                       * FLEET_DICTS[i]['eta_kg_per_kwh'] * DT_RT
                       * FLEET_DICTS[i].get('r_h2', 3.0)).sum())
                for i in range(N)
            )
            curt_tot   = float(data['curtail'].sum() * DT_RT)
            avg_import = float(data['p_net'].mean())
            w.writerow(["TOTAL", scenario_name,
                        f"{h2_tot:.1f}", f"{ec_tot:.2f}", f"{rev_tot:.2f}",
                        f"{rev_tot - ec_tot:.2f}", f"{curt_tot:.1f}", f"{avg_import:.1f}"])

    # ════════════════════════════════════════════════════════════════════════
    # 8. Daily Breakdown CSV
    # ════════════════════════════════════════════════════════════════════════
    with open(out_dir / "daily_breakdown.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Day", "DoW", "Scenario",
                    "fleet_H2_kg", "fleet_energy_cost_$", "fleet_H2_revenue_$",
                    "fleet_net_profit_$", "fleet_curtail_kWh",
                    "fleet_avg_import_kW"])
        for d in range(N_DAYS):
            for scenario_name, data in [("Individual", ind), ("GNE", gne)]:
                h2_d     = float(data['h2_daily'][d].sum())
                ec_d     = float(data['cost_rt'][d].sum())
                rev_d    = sum(
                    float((data['p_elec'][d, :, i]
                           * FLEET_DICTS[i]['eta_kg_per_kwh'] * DT_RT
                           * FLEET_DICTS[i].get('r_h2', 3.0)).sum())
                    for i in range(N)
                )
                curt_d   = float(data['curtail'][d].sum() * DT_RT)
                avg_imp  = float(data['p_net'][d].mean())
                w.writerow([d + 1, DAYS_FULL[d % 7], scenario_name,
                            f"{h2_d:.1f}", f"{ec_d:.2f}", f"{rev_d:.2f}",
                            f"{rev_d - ec_d:.2f}", f"{curt_d:.1f}", f"{avg_imp:.1f}"])

    print(f"\nComparison outputs saved to {out_dir}/")
    print("  Figures: compare_h2_daily, compare_h2_fleet_total, compare_financials,")
    print("           compare_aggregate_day*, compare_per_agent_day1,")
    print("           compare_hourly_cost, compare_curtailment")
    print("  CSVs:   comparison_summary.csv, daily_breakdown.csv")
    print("Done — Comparison.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) == 3:
        ind_dir = Path(sys.argv[1])
        gne_dir = Path(sys.argv[2])
    else:
        ind_dir = _find_latest("individual_")
        gne_dir = _find_latest("gne_")
        print(f"Auto-discovered:\n  Individual: {ind_dir}\n  GNE:        {gne_dir}")

    run_comparison(ind_dir, gne_dir)
