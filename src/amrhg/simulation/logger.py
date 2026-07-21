"""
simulation/logger.py — Simulation result storage and serialisation.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimulationResult:
    """
    Stores all trajectories from one closed-loop simulation run.

    Attributes
    ----------
    method : str
        Identifier: "rhg", "open_loop", or "admm".
    p_net : ndarray (N, T)
        Per-agent net power at each 5-min step (kW).
    soc : ndarray (N, T+1)
        Per-agent state (SoC or inventory) including initial condition.
    rtm_prices : ndarray (T,)
        RTM LMP at each step ($/MWh).
    dam_prices : ndarray (T_hr,)
        DAM LMP ($/MWh) for each hour.
    da_schedules : ndarray (N, T_hr)
        Day-ahead committed power for each agent per hour (kW).
    combo_history : list[int]
        GNE combo index per RTM step.
    total_cost : ndarray (N,)
        Cumulative cost per agent ($).
    coupling_violations : ndarray (T,)
        Max coupling constraint violation per step (kW).
    n_steps : int
        Total 5-min steps simulated.
    agent_names : list[str]
    """

    method: str
    p_net: np.ndarray
    soc: np.ndarray
    rtm_prices: np.ndarray
    dam_prices: np.ndarray
    da_schedules: np.ndarray
    combo_history: list[int] = field(default_factory=list)
    total_cost: np.ndarray | None = None
    coupling_violations: np.ndarray | None = None
    n_steps: int = 0
    agent_names: list[str] = field(default_factory=list)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> SimulationResult:
        with open(path, "rb") as f:
            return pickle.load(f)
