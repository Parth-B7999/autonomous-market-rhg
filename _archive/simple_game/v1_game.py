"""
v1_game.py — Horizon version of the market GNE (the CR-explosion test).

v1 = v0 extended to an H-step receding horizon (per PLAN_mp_focapo_rtm.md):
  • Same N grid-only, buy-only electrolyzers (constants γ_i, η_i, p_max_i from v0).
  • Decision per agent: x_i = [p_{i,0}, …, p_{i,H-1}] ∈ ℝ^H.
  • Parameter: the price FORECAST λ = [λ_0, …, λ_{H-1}] ∈ ℝ^H (full H dims — the
    deliberately generous choice; PCA compression is the recorded fallback).
  • Per-step cost, per-step box, per-step coupling L_min ≤ Σ_i p_{i,k} ≤ L_max.

KEY STRUCTURAL FACT (recorded in the plan): with NO ramp, NO storage, NO H₂
tracking there is NO intertemporal coupling — the cost is separable across k, the
box is per-step, and the coupling at step k involves only {p_{i,k}}.  So the H-step
game is exactly H INDEPENDENT COPIES of the v0 single-step game.  Consequences:

  • Naive full-horizon mpQP: each agent's H-step problem is a product of H
    independent 1-D problems → ~(single-step CR count)^H critical regions.
    This is the CR EXPLOSION the test demonstrates.
  • Exact fix (decoupling reduction): build the single-step map ONCE (v0, 3
    variational CRs over λ∈ℝ), evaluate it per step.  θ stays 1-D, no explosion.

Per-agent matrices (decision x_i ∈ ℝ^H, parameter λ ∈ ℝ^H):
  Q_i    = γ_i · I_H
  c_i    = −a_i · 1_H                         a_i = r_H2·η_i·1000
  F_i    = I_H                                (linear cost coef: (λ_k − a_i)·p_{i,k})
  A_loc  = [I_H; −I_H],  b_loc = [p_max·1_H; 0],  S_loc = 0     (per-step box)
  C_i    = [I_H; −I_H]                        (per-step coupling block, 2H×H)
  d      = [L_max·1_H; −L_min·1_H],  S_coup = 0
  p box  = λ ∈ [λ_lb, λ_ub]^H
"""

from __future__ import annotations
import numpy as np
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from amrhg.solvers.game import Agent, GNEGame
from v0_game import DEFAULT_FLEET, DEFAULT_L_MIN, DEFAULT_L_MAX, DEFAULT_LAM, V0AgentSpec


def build_v1_game(
    H: int,
    fleet: list[V0AgentSpec] | None = None,
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
    lam_lb: float = DEFAULT_LAM[0],
    lam_ub: float = DEFAULT_LAM[1],
) -> tuple[GNEGame, list[V0AgentSpec]]:
    """Build the H-step v1 GNEGame (n_x = H per agent, n_p = H = λ forecast)."""
    fleet = list(fleet) if fleet is not None else list(DEFAULT_FLEET)
    n_p = H
    I = np.eye(H)

    agents: list[Agent] = []
    for i, spec in enumerate(fleet):
        Q = spec.gamma * I
        c = -spec.a * np.ones(H)
        F = I.copy()                                  # (λ_k − a_i) p_{i,k}
        A_loc = np.vstack([I, -I])                    # (2H, H)
        b_loc = np.concatenate([spec.p_max * np.ones(H), np.zeros(H)])
        S_loc = np.zeros((2 * H, n_p))
        C = np.vstack([I, -I])                        # (2H, H): per-step Σ ≤ L_max, ≥ L_min
        agents.append(Agent(
            index=i, n_x=H, Q=Q, c=c, F=F,
            A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C,
        ))

    d = np.concatenate([l_max * np.ones(H), -l_min * np.ones(H)])
    S_coup = np.zeros((2 * H, n_p))
    game = GNEGame(
        agents=agents, d=d, S_coup=S_coup, d_lb=None, S_coup_lb=None,
        p_lb=lam_lb * np.ones(H), p_ub=lam_ub * np.ones(H),
    )
    return game, fleet
