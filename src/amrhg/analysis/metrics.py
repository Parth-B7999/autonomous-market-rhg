"""
analysis/metrics.py — Cost breakdown and performance metrics.
"""

from __future__ import annotations

import numpy as np

from amrhg.simulation.logger import SimulationResult


def compute_cost_breakdown(
    result: SimulationResult,
    agent_idx: int,
    settlement_cfg: dict,
) -> dict:
    """
    Compute per-agent cost breakdown over the full simulation.

    Returns
    -------
    dict with keys:
        energy_cost     : total DAM + RTM energy cost ($)
        imbalance_penalty : total z+/z- penalties ($)
        total_cost      : sum of all components ($)
    """
    gamma_plus = settlement_cfg["gamma_plus"]
    gamma_minus = settlement_cfg["gamma_minus"]
    T = result.n_steps

    p = result.p_net[agent_idx]         # (T,)
    rtm = result.rtm_prices             # (T,)
    da = result.da_schedules[agent_idx]  # (T_hr,)

    energy_cost = 0.0
    imbalance = 0.0

    for t in range(T):
        hour_idx = min(t // 12, len(da) - 1)
        da_kw = da[hour_idx]
        energy_cost += (rtm[t] / 1000.0) * (p[t] - da_kw)  # RTM imbalance cost in $

        imb = p[t] - da_kw
        if imb > 0:
            imbalance += (gamma_plus / 1000.0) * imb
        else:
            imbalance += (gamma_minus / 1000.0) * (-imb)

    return {
        "energy_cost": float(energy_cost),
        "imbalance_penalty": float(imbalance),
        "total_cost": float(energy_cost + imbalance),
    }


def compute_total_welfare(results: dict[str, SimulationResult]) -> dict[str, float]:
    """Sum of all agent total costs per method."""
    welfare = {}
    for method, result in results.items():
        welfare[method] = float(np.sum(result.total_cost))
    return welfare


def compute_coupling_violation_rate(
    result: SimulationResult,
    l_min: float,
    l_max: float,
) -> float:
    """Fraction of steps where coupling constraints are violated > 1 kW."""
    if result.coupling_violations is None:
        return 0.0
    violations = result.coupling_violations > 1.0
    return float(np.mean(violations))
