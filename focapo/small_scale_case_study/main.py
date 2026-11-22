"""
main.py — ADMM Green Hydrogen Market Study, entry point.

6-agent fleet over 7 days:
  PEM_Elec      — Electrolyzer only
  PEM_PV        — PV + Electrolyzer
  PEM_Wind      — Wind + Electrolyzer
  ALK_Bat       — Battery + Electrolyzer
  PEM_PV_Bat    — PV + Battery + Electrolyzer
  PEM_Wind_Bat  — Wind + Battery + Electrolyzer

RTM horizon: H=12 steps (1 hour at 5-min resolution)
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from agents import L_MIN_KW, L_MAX_KW, DT_RT, STEPS_PER_DAY
from simulation import run
from forecasting import load_pjm_week, get_day, dam_forecast, rt_forecast

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

COLORS  = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728', '#8c564b']
T_HR    = np.arange(STEPS_PER_DAY) * 5 / 60
H_TICKS = np.arange(0, 25, 4)

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'lines.linewidth': 1.3,
})


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ('pdf', 'png'):
        fig.savefig(RESULTS / f"{name}.{ext}", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved  {name}.pdf / .png")


# ── Figure 1: power profiles ──────────────────────────────────────────────────

def fig_power(res, agents, d: int = 0) -> None:
    fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    ax = axs[0]
    for i, ag in enumerate(agents):
        ax.plot(T_HR, res.p_net[d, :, i], color=COLORS[i], label=ag.name)
    ax.axhline(L_MIN_KW, c='r',  ls='--', lw=1.0, label=f'$L_{{\\min}}$ = {L_MIN_KW:.0f} kW')
    ax.axhline(L_MAX_KW, c='k',  ls='--', lw=0.9, label=f'$L_{{\\max}}$ = {L_MAX_KW:.0f} kW')
    ax.set_ylabel('Net Grid Import [kW]')
    ax.set_title(f'Day {d+1} — Per-Agent Net Grid Import  ($p_i \\geq 0$)')
    ax.legend(ncol=3); ax.grid(True, alpha=0.3); ax.set_xticks(H_TICKS)

    ax = axs[1]
    for i, ag in enumerate(agents):
        if ag.has_renewable:
            lbl = f'{ag.name} ({ag.renewable_type.upper()}, {ag.renewable_cap_kw:.0f} kW)'
            ax.plot(T_HR, res.ren[d, :, i], color=COLORS[i], lw=1.1, label=lbl)
    ax.set_ylabel('Renewable Generation [kW]')
    ax.set_title('Collocated Renewable Generation')
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)

    ax = axs[2]
    a  = res.agg[d]
    lb = L_MIN_KW
    ax.fill_between(T_HR, lb, np.maximum(a, lb), alpha=0.18, color='green',
                    label='Feasible band')
    ax.fill_between(T_HR, a, lb, where=a < lb, alpha=0.55, color='red',
                    label=f'$L_{{\\min}}$ binding ({res.lmin_bind[d].mean()*100:.1f}%)')
    ax.plot(T_HR, a, 'k', lw=1.5, label='Aggregate $\\sum_i p_i$')
    ax.axhline(L_MIN_KW, c='r', ls='--', lw=1.0, label=f'$L_{{\\min}}$ = {L_MIN_KW:.0f} kW')
    ax.axhline(L_MAX_KW, c='k', ls='--', lw=0.9, label=f'$L_{{\\max}}$ = {L_MAX_KW:.0f} kW')
    ax.fill_between(T_HR, a, L_MAX_KW, where=a > L_MAX_KW,
                    alpha=0.45, color='orange',
                    label=f'$L_{{\\max}}$ binding ({res.lmax_bind[d].mean()*100:.1f}%)')
    ax.set_ylabel('Aggregate Import [kW]')
    ax.set_xlabel('Hour of Day')
    ax.set_title('Aggregate Import vs Bilateral Coupling Bounds')
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f'fig1_power_day{d+1}')


# ── Figure 2: electrolyzer breakdown (2×3) ───────────────────────────────────

def fig_elz_breakdown(res, agents, d: int = 0) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)

    for i, (ax, ag) in enumerate(zip(axs.flat, agents)):
        pe  = res.p_elec[d, :, i]
        gr  = res.ren[d, :, i]
        pd_ = res.p_dis[d, :, i]
        pc  = res.p_ch[d, :, i]

        ax.fill_between(T_HR, 0, gr, alpha=0.45, color='gold', label='Renewable')
        if ag.has_battery:
            ax.fill_between(T_HR, gr, gr + pd_, alpha=0.40, color='limegreen',
                            label='Bat discharge')
            ax.fill_between(T_HR, 0, -pc, alpha=0.35, color='tomato',
                            label='Bat charge')
        ax.fill_between(T_HR, gr + pd_, pe, alpha=0.50, color=COLORS[i],
                        label='Grid top-up')
        ax.plot(T_HR, pe, color=COLORS[i], lw=1.3, label='Total electrolyzer')
        ax.axhline(ag.p_elec_max_kw, c='k',    ls=':', lw=0.8,
                   label=f'$P^{{max}}$ = {ag.p_elec_max_kw:.0f} kW')
        if ag.p_elec_min_kw > 0:
            ax.axhline(ag.p_elec_min_kw, c='gray', ls=':', lw=0.7,
                       label=f'$P^{{min}}$ = {ag.p_elec_min_kw:.0f} kW')

        parts = []
        if ag.has_renewable:
            parts.append(f'{ag.renewable_type.upper()} {ag.renewable_cap_kw:.0f} kW')
        if ag.has_battery:
            parts.append(f'Bat {ag.bat_cap_kwh:.0f} kWh')
        ax.set_title(f'{ag.name}  ({", ".join(parts) if parts else "no ren/bat"})')
        ax.set_ylabel('Power [kW]')
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        ax.set_xticks(H_TICKS)

    for ax in axs[1]:
        ax.set_xlabel('Hour of Day')
    fig.suptitle(f'Day {d+1} — Electrolyzer Power Breakdown')
    plt.tight_layout()
    _save(fig, f'fig2_elz_breakdown_day{d+1}')


# ── Figure 3: H2 production ───────────────────────────────────────────────────

def fig_h2(res, agents) -> None:
    nd   = res.h2_daily.shape[0]
    days = np.arange(1, nd + 1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    w   = 0.13
    off = np.linspace(-2.5, 2.5, len(agents)) * w
    ax  = axs[0]
    for i, ag in enumerate(agents):
        ax.bar(days + off[i], res.h2_daily[:, i], w,
               color=COLORS[i], alpha=0.82, label=ag.name)
        lbl_tgt = 'Target (dashed)' if i == 0 else '_nolegend_'
        ax.plot(days + off[i], res.h2_target[:, i], color=COLORS[i],
                ls='--', lw=1.1, marker='x', ms=5, label=lbl_tgt)
    ax.set_xlabel('Day'); ax.set_ylabel('H₂ Production [kg/day]')
    ax.set_title('Daily H₂ Production vs Take-or-Pay Target (dashed × = target)')
    ax.set_xticks(days); ax.legend(ncol=2); ax.grid(True, alpha=0.3, axis='y')

    ax = axs[1]
    for i, ag in enumerate(agents):
        cum    = np.cumsum(res.h2_daily[:, i])
        target = np.cumsum(res.h2_target[:, i])
        ax.plot(days, cum,    color=COLORS[i], lw=2.0,  label=f'{ag.name} actual')
        ax.plot(days, target, color=COLORS[i], lw=1.1,  ls='--',
                label=f'{ag.name} target' if i == 0 else '_nolegend_')
    # Single legend entry for "target" style
    from matplotlib.lines import Line2D
    extra = [Line2D([0],[0], color='gray', ls='--', lw=1.1, label='Target (dashed)')]
    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('Day'); ax.set_ylabel('Cumulative H₂ [kg]')
    ax.set_title('Cumulative H₂ vs Target (dashed)')
    ax.set_xticks(days)
    ax.legend(handles + extra, labels + ['Target (dashed)'], ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, 'fig3_h2_production')


# ── Figure 4: ADMM convergence ────────────────────────────────────────────────

def fig_admm_conv(res) -> None:
    nd  = res.rt_iters.shape[0]
    its = res.rt_iters.ravel()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    ax = axs[0]
    ax.hist(its, bins=40, color='steelblue', edgecolor='white', lw=0.5)
    ax.axvline(its.mean(),             c='r',      ls='--', lw=1.4,
               label=f'Mean={its.mean():.1f}')
    ax.axvline(np.percentile(its, 95), c='orange', ls='--', lw=1.4,
               label=f'P95={np.percentile(its,95):.0f}')
    ax.set_xlabel('ADMM Iterations'); ax.set_ylabel('Steps (count)')
    ax.set_title('RTM ADMM Iteration Distribution')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    ax = axs[1]
    conv = res.rt_conv.mean(axis=1) * 100
    ax.bar(np.arange(1, nd+1), conv, color='green', alpha=0.78)
    ax.axhline(conv.mean(), c='r', ls='--', lw=1.4, label=f'Mean={conv.mean():.1f}%')
    ax.set_ylim(0, 105); ax.set_xlabel('Day')
    ax.set_ylabel('Convergence Rate [%]')
    ax.set_title('RTM ADMM: % Steps Converged per Day')
    ax.set_xticks(np.arange(1, nd+1)); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    ax = axs[2]
    ax.bar(np.arange(1, nd+1), res.da_iters, color='coral', alpha=0.82,
           label='DAM iterations')
    ax.axhline(res.da_iters.mean(), c='r', ls='--', lw=1.4,
               label=f'Mean = {res.da_iters.mean():.0f}')
    ax.set_xlabel('Day'); ax.set_ylabel('Iterations')
    ax.set_title('DAM ADMM Iterations / Day')
    ax.set_xticks(np.arange(1, nd+1)); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save(fig, 'fig4_admm_convergence')


# ── Figure 5: DA schedule vs RT execution (2×3) ───────────────────────────────

def fig_da_vs_rt(res, agents, d: int = 0) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    hours = np.arange(24) + 0.5

    for i, (ax, ag) in enumerate(zip(axs.flat, agents)):
        da      = res.p_da[d, :, i]
        rt_full = res.p_net[d, :, i]
        rt_hrly = rt_full.reshape(24, -1).mean(axis=1)
        ax.step(hours, da, where='mid', color=COLORS[i], lw=2.0, label='DA schedule')
        ax.plot(T_HR, rt_full, color=COLORS[i], alpha=0.25, lw=0.7, label='RT 5-min')
        ax.step(hours, rt_hrly, where='mid', color=COLORS[i], ls='--', lw=1.4,
                label='RT hourly avg')
        ax.set_title(ag.name)
        ax.set_ylabel('Net Import [kW]')
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        ax.set_xticks(H_TICKS)

    for ax in axs[1]:
        ax.set_xlabel('Hour of Day')
    fig.suptitle(f'Day {d+1} — DA Schedule (step) vs RT Execution (line)')
    plt.tight_layout()
    _save(fig, f'fig5_da_vs_rt_day{d+1}')


# ── Figure 6: LMP and costs ───────────────────────────────────────────────────

def fig_lmp_cost(res, agents, d: int = 0) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axs[0]
    ax.plot(T_HR, res.lmp_rt[d], 'k', lw=1.3, label='RT LMP')
    ax.fill_between(T_HR, 0, res.lmp_rt[d], alpha=0.12, color='black')
    ax.set_ylabel('RT LMP [$/MWh]')
    ax.set_title(f'Day {d+1} — Real-Time LMP and Cumulative Net Cost')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(H_TICKS)

    ax = axs[1]
    for i, ag in enumerate(agents):
        energy = res.cost_rt[d, :, i]
        h2_rev = (res.p_elec[d, :, i] * ag.eta_kg_per_kwh * DT_RT * ag.r_h2)
        net_cum = np.cumsum(energy - h2_rev)
        ax.plot(T_HR, net_cum, color=COLORS[i], label=ag.name)
    ax.axhline(0, c='k', ls='--', lw=0.8, label='Breakeven (zero net cost)')
    ax.set_ylabel('Cumulative Net Cost [$]')
    ax.set_xlabel('Hour of Day')
    ax.legend(ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f'fig6_lmp_cost_day{d+1}')


# ── Figure 0: All 7 days aggregate (for paper) ───────────────────────────────

def fig_all_days_aggregate(res) -> None:
    """
    2×4 grid (7 panels + 1 legend panel): aggregate net import vs
    L_min/L_max bounds for each of the 7 simulation days.
    Shows bilateral coupling constraint activity across the full week.
    """
    nd = res.agg.shape[0]
    fig, axs = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)

    DAYS = ['Mon 07/08', 'Tue 07/09', 'Wed 07/10', 'Thu 07/11',
            'Fri 07/12', 'Sat 07/13', 'Sun 07/14']
    lb = L_MIN_KW
    ub = L_MAX_KW

    for d, ax in enumerate(axs.flat[:nd]):
        a = res.agg[d]
        # Shade L_max violation
        ax.fill_between(T_HR, a, ub, where=a > ub,
                        alpha=0.55, color='orange', label='$L_{\\max}$ binding')
        # Shade below L_min
        ax.fill_between(T_HR, a, lb, where=a < lb,
                        alpha=0.55, color='steelblue', label='$L_{\\min}$ binding')
        ax.fill_between(T_HR, lb, np.minimum(a, ub), alpha=0.15, color='green')
        ax.plot(T_HR, a, 'k', lw=1.3, label='Aggregate $\\sum_i p_i$')
        ax.axhline(lb, c='steelblue', ls='--', lw=1.0)
        ax.axhline(ub, c='orange',    ls='--', lw=1.0)
        ax.set_title(DAYS[d], fontsize=8)
        plot_ub = ub if ub != float('inf') else max(a.max(), lb * 2.0)
        ax.set_ylim(0, plot_ub * 1.08)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(True, alpha=0.3)
        # Annotate binding fraction
        pct = res.lmax_bind[d].mean() * 100
        ax.text(0.97, 0.95, f'$L_{{\\max}}$: {pct:.0f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=7,
                color='orange')

    # 8th panel: legend
    ax_leg = axs.flat[7]
    ax_leg.axis('off')
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='k', lw=1.3, label='Aggregate import'),
        Patch(facecolor='orange',    alpha=0.55, label=f'$L_{{\\max}}={L_MAX_KW:.0f}$\\,kW binding'),
        Patch(facecolor='steelblue', alpha=0.55, label=f'$L_{{\\min}}={L_MIN_KW:.0f}$\\,kW binding'),
        Patch(facecolor='green',     alpha=0.20, label='Feasible region'),
    ]
    ax_leg.legend(handles=handles, loc='center', fontsize=8, frameon=True)

    for ax in axs[1, :]:
        ax.set_xlabel('Hour of day', fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel('Aggregate import [kW]', fontsize=8)

    fig.suptitle(f'7-Day Aggregate Net Import vs Bilateral PCC Coupling Bounds  '
                 f'($L_{{\\min}}={L_MIN_KW:.0f}$\\,kW, $L_{{\\max}}={L_MAX_KW:.0f}$\\,kW)',
                 fontsize=9)
    plt.tight_layout()
    _save(fig, 'fig0_all_days_aggregate')


# ── Figure 7: Battery SoC ─────────────────────────────────────────────────────

def fig_bat_soc(res, agents, d: int = 0) -> None:
    """Battery SoC and charge/discharge profiles for battery agents."""
    bat_agents = [(i, ag) for i, ag in enumerate(agents) if ag.has_battery]
    if not bat_agents:
        return
    n_bat = len(bat_agents)
    fig, axs = plt.subplots(n_bat, 2, figsize=(14, 4 * n_bat), sharex=True)
    if n_bat == 1:
        axs = axs[np.newaxis, :]

    t_soc = np.arange(STEPS_PER_DAY + 1) * 5 / 60

    for row, (i, ag) in enumerate(bat_agents):
        # SoC
        ax = axs[row, 0]
        soc = res.bat_soc[d, :, i]
        ax.plot(t_soc, soc, color=COLORS[i], lw=1.6, label='SoC [kWh]')
        ax.axhline(ag.bat_cap_kwh, c='k', ls=':', lw=0.9,
                   label=f'Capacity {ag.bat_cap_kwh:.0f} kWh')
        ax.axhline(0,              c='gray', ls=':', lw=0.7)
        ax.set_ylabel('SoC [kWh]')
        ax.set_title(f'{ag.name} — Battery State of Charge')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Charge / discharge
        ax = axs[row, 1]
        ax.fill_between(T_HR,  0, res.p_ch[d, :, i],  alpha=0.65,
                        color='tomato',    label='Charging (grid draw)')
        ax.fill_between(T_HR, 0, res.p_dis[d, :, i], alpha=0.65,
                        color='limegreen', label='Discharging (to electrolyzer)')
        ax.set_ylabel('Power [kW]')
        ax.set_title(f'{ag.name} — Battery Charge / Discharge')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        if ag.has_renewable:
            ax2 = ax.twinx()
            ax2.plot(T_HR, res.ren[d, :, i], 'gold', lw=1.1, alpha=0.7,
                     label='Renewable [kW]')
            ax2.set_ylabel('Renewable [kW]', color='goldenrod')
            ax2.legend(fontsize=6, loc='upper right')

    for ax in axs[-1]:
        ax.set_xlabel('Hour of Day')
        ax.set_xticks(H_TICKS)
    plt.tight_layout()
    _save(fig, f'fig7_bat_soc_day{d+1}')


# ── Figure 8: DAM forecast vs actual ─────────────────────────────────────────

def fig_dam_forecast(week: dict, agents: list, rng, d: int = 0) -> None:
    hours = np.arange(24)
    fc    = dam_forecast(week, d, agents, rng)
    true  = get_day(week, d, agents)

    true_da = true['da_price']
    fc_da   = fc['da_price']
    err_da  = fc_da - true_da

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    ax = axs[0, 0]
    ax.plot(hours, true_da, 'k-', lw=2.0, label='True DA LMP')
    ax.plot(hours, fc_da,  'r--', lw=1.4, label='DA Forecast (σ=16 $/MWh)')
    ax.set_ylabel('Price [$/MWh]')
    ax.set_title(f'Day {d+1} — DA Price: Forecast vs True')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axs[0, 1]
    ax.bar(hours, err_da,
           color=np.where(err_da >= 0, 'steelblue', 'tomato'), alpha=0.8)
    ax.axhline(0, c='k', lw=0.8)
    ax.set_ylabel('Error [$/MWh]')
    ax.set_title('DA Price Forecast Error  (forecast − true)')
    ax.set_xlabel('Hour of Day'); ax.grid(True, alpha=0.3, axis='y')

    # Renewable: first PV agent and first Wind agent
    pv_idx   = next((i for i, ag in enumerate(agents) if ag.renewable_type == 'pv'), None)
    wind_idx = next((i for i, ag in enumerate(agents) if ag.renewable_type == 'wind'), None)

    ax = axs[1, 0]
    ax2 = axs[1, 1]
    for ag_idx, ls in [(pv_idx, '-'), (wind_idx, '--')]:
        if ag_idx is None:
            continue
        ag       = agents[ag_idx]
        true_ren = true['ren'][ag_idx, ::12]
        fc_ren   = fc['ren'][ag_idx]
        err_ren  = fc_ren - true_ren
        lbl = f'{ag.name} ({ag.renewable_type.upper()})'
        ax.plot(hours, true_ren, color=COLORS[ag_idx], ls=ls, lw=2.0,
                label=f'{lbl} true')
        ax.plot(hours, fc_ren,  color=COLORS[ag_idx], ls=':', lw=1.3,
                label=f'{lbl} fc')
        ax2.plot(hours, err_ren, color=COLORS[ag_idx], ls=ls, lw=1.3,
                 label=f'{lbl} (σ=20% cap)')

    ax.set_ylabel('Renewable Output [kW]')
    ax.set_title('Renewable Forecast vs True')
    ax.set_xlabel('Hour of Day')
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    ax2.axhline(0, c='k', lw=0.8)
    ax2.set_ylabel('Forecast Error [kW]')
    ax2.set_title('Renewable Forecast Error  (forecast − true)')
    ax2.set_xlabel('Hour of Day')
    ax2.legend(fontsize=6); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f'fig8_dam_forecast_day{d+1}')


# ── Figure 9: RTM forecast fan ────────────────────────────────────────────────

def fig_rtm_forecast(week: dict, agents: list, rng, d: int = 0,
                     t_start: int = 72, n_steps: int = 36) -> None:
    from agents import H_RT
    true  = get_day(week, d, agents)
    t_end = t_start + n_steps
    t_ax  = np.arange(t_start, t_end) * 5 / 60
    true_lmp = true['rt_price'][t_start:t_end]

    pv_idx = next((i for i, ag in enumerate(agents) if ag.renewable_type == 'pv'), None)

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axs[0]
    ax.plot(t_ax, true_lmp, 'k-', lw=2.0, label='True RT LMP', zorder=5)
    for t in range(t_start, t_end, 3):
        ag0 = agents[0]
        fc = rt_forecast(t, week, d, H_RT, ag0.renewable_type,
                         ag0.renewable_cap_kw, rng)
        t_fc = np.arange(t, min(t + H_RT, t_end)) * 5 / 60
        ax.plot(t_fc, fc['lam'][:len(t_fc)], 'r-', alpha=0.22, lw=0.8)
    ax.plot([], [], 'r-', alpha=0.6, lw=1.2, label='RT Forecast fan (σ=30 $/MWh)')
    ax.set_ylabel('RT LMP [$/MWh]')
    ax.set_title(f'Day {d+1} Hours {t_start*5//60:02d}:00–{t_end*5//60:02d}:00 — RTM Forecast vs True')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axs[1]
    if pv_idx is not None:
        ag = agents[pv_idx]
        true_ren = true['ren'][pv_idx, t_start:t_end]
        ax.plot(t_ax, true_ren, color=COLORS[pv_idx], lw=2.0,
                label=f'True PV ({ag.name})', zorder=5)
        for t in range(t_start, t_end, 3):
            fc = rt_forecast(t, week, d, H_RT, ag.renewable_type, ag.renewable_cap_kw, rng)
            t_fc = np.arange(t, min(t + H_RT, t_end)) * 5 / 60
            ax.plot(t_fc, fc['ren'][:len(t_fc)], color=COLORS[pv_idx], alpha=0.22, lw=0.8)
        ax.plot([], [], color=COLORS[pv_idx], alpha=0.6, lw=1.2,
                label='RT PV Forecast fan (σ=20% cap)')
    ax.set_ylabel('PV Output [kW]')
    ax.set_xlabel('Hour of Day')
    ax.set_title('RTM Renewable Forecast Fan')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f'fig9_rtm_forecast_day{d+1}')


# ── Summary statistics ────────────────────────────────────────────────────────

def print_summary(res, agents) -> None:
    nd  = res.h2_daily.shape[0]
    T   = STEPS_PER_DAY
    its = res.rt_iters.ravel()

    print()
    print("=" * 65)
    print("  ADMM ELECTROLYZER MARKET — 6-AGENT SIMULATION SUMMARY")
    print("=" * 65)
    print(f"  Days: {nd}  |  Steps/day: {T}  |  Total RT steps: {nd*T}")
    print(f"  L_min={L_MIN_KW:.0f} kW  |  L_max={L_MAX_KW:.0f} kW  |  RTM horizon H=12 (1 hr)")
    print()
    print("  RTM ADMM")
    print(f"    Convergence rate : {res.rt_conv.mean()*100:.1f}%")
    print(f"    Mean iterations  : {its.mean():.1f}")
    print(f"    P95  iterations  : {np.percentile(its,95):.0f}")
    print()
    print("  DAM ADMM")
    print(f"    Convergence rate : {res.da_conv.mean()*100:.1f}%")
    print(f"    Mean iterations  : {res.da_iters.mean():.1f}")
    print()
    print("  Coupling constraint")
    print(f"    L_min binding    : {res.lmin_bind.mean()*100:.1f}% of RT steps")
    print(f"    L_max binding    : {res.lmax_bind.mean()*100:.1f}% of RT steps")
    print()
    print("  H2 Production  (avg/day actual vs avg daily target, attainment %)")
    total_tgt = total_act = 0.0
    for i, ag in enumerate(agents):
        actual = float(res.h2_daily[:, i].mean())
        tgt    = float(res.h2_target[:, i].mean())
        total_tgt += tgt; total_act += actual
        bat_str = f'[Bat {ag.bat_cap_kwh:.0f}kWh]' if ag.has_battery else ''
        print(f"    {ag.name:16s} {bat_str:16s}  {actual:6.0f} kg  "
              f"(avg target {tgt:.0f} kg,  {actual/tgt*100:.0f}%)" if tgt > 0
              else f"    {ag.name:16s} {bat_str:16s}  {actual:6.0f} kg  (no target)")
    print(f"    {'TOTAL':34s}  {total_act:6.0f} kg  "
          f"(avg target {total_tgt:.0f} kg,  {total_act/total_tgt*100:.0f}%)")
    print()
    print("  Financials per agent (7-day totals)")
    print(f"    {'Agent':16s}  {'Energy cost':>12s}  {'H2 revenue':>10s}  {'Net profit':>10s}")
    print(f"    {'-'*52}")
    for i, ag in enumerate(agents):
        ec  = float(res.cost_rt[:, :, i].sum())
        rev = float((res.p_elec[:, :, i] * ag.eta_kg_per_kwh * DT_RT * ag.r_h2).sum())
        print(f"    {ag.name:16s}  ${ec:11,.0f}  ${rev:9,.0f}  ${rev-ec:9,.0f}")
    print("=" * 65)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("ADMM Green Hydrogen Market Simulation — 6-Agent Fleet")
    print("Fleet: PEM_Elec | PEM_PV | PEM_Wind | ALK_Bat | PEM_PV_Bat | PEM_Wind_Bat")
    print(f"RTM horizon H=12 (1 hr)  |  L_min={L_MIN_KW:.0f} kW  |  L_max={L_MAX_KW:.0f} kW")
    print()

    res, agents = run(n_days=7, seed=42)
    print_summary(res, agents)

    print("\nGenerating figures ...")
    week    = load_pjm_week(start_date='2024-07-08', n_days=7)
    fc_rng  = np.random.default_rng(99)

    for d in range(7):
        fig_power(res, agents, d)
        fig_elz_breakdown(res, agents, d)
        fig_da_vs_rt(res, agents, d)
        fig_lmp_cost(res, agents, d)
        fig_bat_soc(res, agents, d)
        fig_dam_forecast(week, agents, fc_rng, d)
        fig_rtm_forecast(week, agents, fc_rng, d)
    fig_h2(res, agents)
    fig_admm_conv(res)
    fig_all_days_aggregate(res)

    print(f"\nAll figures saved to  {RESULTS}/")
