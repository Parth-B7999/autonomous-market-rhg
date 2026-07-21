"""
analysis/plots.py — Paper figure generation.

All figures are generated from saved SimulationResult .pkl files so they
can be regenerated without re-running the simulation.
"""

from __future__ import annotations

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amrhg.simulation.logger import SimulationResult


def plot_daily_trajectory(
    result: SimulationResult,
    day_idx: int = 0,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot one day of DAM commitment vs RTM realisation for all agents.

    Parameters
    ----------
    result : SimulationResult
    day_idx : int
        Which day to plot (0-based).
    save_path : str or None
        If given, save figure to this path.
    """
    T_per_day = 288  # 5-min steps
    start = day_idx * T_per_day
    end = start + T_per_day
    minutes = np.arange(T_per_day) * 5 / 60  # hours

    N = result.p_net.shape[0]
    fig, axes = plt.subplots(N, 1, figsize=(12, 3 * N), sharex=True)
    if N == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        p_rt = result.p_net[i, start:end]
        # DA schedule is hourly — repeat to 5-min
        da_hourly = result.da_schedules[i, day_idx * 24:(day_idx + 1) * 24]
        da_5min = np.repeat(da_hourly, 12)
        ax.step(minutes, p_rt, where="mid", label="RTM", linewidth=1.2, color="#2196F3")
        ax.step(minutes, da_5min, where="mid", label="DAM", linewidth=1.0,
                linestyle="--", color="#FF9800")
        ax.set_ylabel(f"{result.agent_names[i]}\nPower (kW)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")

    axes[-1].set_xlabel("Hour of day")
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(f"Day {day_idx + 1}: DAM vs RTM", color="white", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor="#1a1a2e")
    return fig


def plot_cost_breakdown_comparison(
    results: dict[str, SimulationResult],
    settlement_cfg: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Bar chart comparing total cost per agent across methods.
    """
    from amrhg.analysis.metrics import compute_cost_breakdown

    methods = list(results.keys())
    N = results[methods[0]].p_net.shape[0]
    names = results[methods[0]].agent_names

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(N)
    width = 0.25

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for j, method in enumerate(methods):
        costs = []
        for i in range(N):
            bd = compute_cost_breakdown(results[method], i, settlement_cfg)
            costs.append(bd["total_cost"])
        ax.bar(x + j * width, costs, width, label=method.upper(), color=colors[j % len(colors)])

    ax.set_ylabel("Total Cost ($)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    fig.patch.set_facecolor("#1a1a2e")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, facecolor="#1a1a2e")
    return fig


def make_all_figures(results_dir: str, figures_dir: str) -> None:
    """
    Regenerate all paper figures from saved SimulationResult files.

    Parameters
    ----------
    results_dir : str
        Path containing sim_rhg.pkl, sim_open_loop.pkl, sim_admm.pkl.
    figures_dir : str
        Output directory for PDF/PNG figures.
    """
    os.makedirs(figures_dir, exist_ok=True)

    methods = ["rhg", "open_loop", "admm"]
    results: dict[str, SimulationResult] = {}
    for method in methods:
        path = os.path.join(results_dir, f"sim_{method}.pkl")
        if os.path.exists(path):
            results[method] = SimulationResult.load(path)

    if not results:
        print("No simulation results found — skipping figures.")
        return

    settlement_cfg = {"gamma_plus": 1.5, "gamma_minus": 2.0}

    # Fig 2: Daily trajectories
    for day in range(min(7, results[list(results.keys())[0]].n_steps // 288)):
        for method, result in results.items():
            plot_daily_trajectory(
                result, day_idx=day,
                save_path=os.path.join(figures_dir, f"fig2_day{day + 1}_{method}.png"),
            )

    # Fig 3: Cost breakdown comparison
    plot_cost_breakdown_comparison(
        results, settlement_cfg,
        save_path=os.path.join(figures_dir, "fig3_cost_comparison.png"),
    )

    plt.close("all")
    print(f"Figures saved to {figures_dir}/")
