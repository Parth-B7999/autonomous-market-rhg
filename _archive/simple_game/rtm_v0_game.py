"""
rtm_v0_game.py — v0 of the LOCKED ERCOT small-scale RTM formulation (FORMULATION.md).

Pipeline-validation rung.  What v0 must prove:
  1. The pipeline (per-agent mpQP → combiner → KKT filter → FACET → map) works with
     the NEW **L_min-only** coupling (single lower-bound row; L_max dropped per the
     lock).  The old v0–v2 used the two-sided L_min≤Σp≤L_max encoding — this is the
     first test of one-sided coupling through mp_solver / combiner / ADMM.
  2. The map == ADMM == centralized QP with the small-scale ERCOT constants and the
     **DA anchor** ½γ(p−p_DA)², which is what supplies strict convexity (the real
     γ=5e-3 alone is nearly degenerate; see note below).

Scope (simplest rung, matches the ladder):
  • N grid-only electrolyzer agents, BUY-ONLY (p ≥ 0).  No renewables, no H₂ demand
    (H=1 → the cumulative-H₂ row is vacuous), no batteries, no ramp.
  • H = 1 → single-shot game.
  • θ = [ λ , p_DA_0 , … , p_DA_{N-1} ]   (n_p = 1 + N).  p_DA is a PARAMETER, not
    baked (per the lock: nothing approximated).

Per-agent problem (single step, dt hr):
    min_{p_i}  ½ γ_i (p_i − p_DA_i)²  +  dt·(λ/1000 − r_H2·η_i)·p_i
    s.t.       0 ≤ p_i ≤ p_max_i

  Expanded to (Q, c, F) with x_i = [p_i], θ = [λ, p_DA_0, …]:
    Q_i = [[γ_i]]
    ½γ(p−p_DA)² = ½γp² − γ·p_DA·p + const  ⇒  linear coef −γ·p_DA·p
    c_i  = −dt·r_H2·η_i                        (constant part)
    F_i  = [ dt/1000 (λ) ,  −γ_i at p_DA_i col ,  0 elsewhere ]

Coupling — L_min ONLY (the market rule; L_max dropped):
    Σ_i p_i ≥ L_min      ⇔      −Σ_i p_i ≤ −L_min
    C_i = [[−1]]   (1 row),   d = [−L_min],   S_coup = 0
  (one-sided; d_lb left None — the two-row trick is unnecessary with a single bound.)

NOTE on constants (FORMULATION.md §5): γ_i = 5e-3 is small, so WITHOUT the anchor the
cost is ~linear and p collapses to ≈0.  The anchor ½γ(p−p_DA)² keeps p near the DA
award and makes the QP well-posed — that is why the anchor (and thus p_DA) is in v0.

L_min: the 2 real grid agents total 90 kW, so the full L_min=100 is INFEASIBLE grid-
only.  v0/v1 use L_min=60 (feasible, binds at dear λ); L_min=100 returns once the
renewable agents (v2) add grid-import capacity.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from amrhg.solvers.game import Agent, GNEGame

R_H2 = 3.0       # $/kg hydrogen price
DT_HR = 0.25     # 15-min RTM step [hr]


@dataclass(frozen=True)
class GridSpec:
    name:  str
    p_max: float   # kW capacity
    eta:   float   # kg/kWh electrolyzer efficiency
    gamma: float   # $/MWh per kW  DA-anchor curvature

    @property
    def a(self) -> float:
        """H₂ marginal revenue in $/MWh units (break-even price)."""
        return R_H2 * self.eta * 1000.0


# Small-scale grid-only fleet (from small_scale_case_study/agents.py; batteries N/A).
DEFAULT_GRID = [
    GridSpec("PEM_Elec", 50.0, 0.020, 5e-3),   # a = 60 $/MWh
    GridSpec("ALK",      40.0, 0.018, 5e-3),   # a = 54 $/MWh
]

DEFAULT_L_MIN = 60.0             # feasible for 90 kW grid cap; binds at dear λ
DEFAULT_LAM   = (-50.0, 150.0)   # ERCOT λ box ($/MWh), tails clamped


def build_rtm_v0_game(
    grid:   list[GridSpec] | None = None,
    l_min:  float = DEFAULT_L_MIN,
    lam_lb: float = DEFAULT_LAM[0],
    lam_ub: float = DEFAULT_LAM[1],
    dt:     float = DT_HR,
) -> tuple[GNEGame, list[GridSpec]]:
    """Build the v0 GNEGame.  θ = [λ, p_DA_0, …, p_DA_{N-1}], n_p = 1 + N.
    Returns (game, grid_specs)."""
    grid = list(grid) if grid is not None else list(DEFAULT_GRID)
    N = len(grid)
    n_p = 1 + N                       # [λ, p_DA_0, ..., p_DA_{N-1}]

    agents: list[Agent] = []
    for i, s in enumerate(grid):
        Q = np.array([[s.gamma]])
        c = np.array([-dt * R_H2 * s.eta])            # −dt·r·η   (constant linear part)
        F = np.zeros((1, n_p))
        F[0, 0]     = dt / 1000.0                     # +dt·λ/1000 · p
        F[0, 1 + i] = -s.gamma                        # −γ·p_DA_i · p   (anchor cross term)

        A_loc = np.array([[1.0], [-1.0]])             # 0 ≤ p ≤ p_max
        b_loc = np.array([s.p_max, 0.0])
        S_loc = np.zeros((2, n_p))

        C = np.array([[-1.0]])                        # −p_i  →  Σ −p_i ≤ −L_min
        agents.append(Agent(index=i, n_x=1, Q=Q, c=c, F=F,
                            A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C))

    d = np.array([-l_min])                            # Σp ≥ L_min
    S_coup = np.zeros((1, n_p))
    p_lb = np.concatenate([[lam_lb], np.zeros(N)])                    # p_DA ≥ 0
    p_ub = np.concatenate([[lam_ub], np.array([s.p_max for s in grid])])
    game = GNEGame(agents=agents, d=d, S_coup=S_coup, d_lb=None, S_coup_lb=None,
                   p_lb=p_lb, p_ub=p_ub)
    return game, grid


def centralized_gne(game: GNEGame, grid: list[GridSpec], theta: np.ndarray,
                    dt: float = DT_HR) -> np.ndarray:
    """
    Ground-truth variational GNE = argmin Σ_i J_i s.t. boxes + Σp ≥ L_min.
    Exact potential game (separable costs + one convex shared constraint) ⇒ the
    v-GNE minimises the potential Σ_i J_i.  Solved via SLSQP.  θ = [λ, p_DA_0, …].
    """
    from scipy.optimize import minimize
    N = game.N
    lam   = float(theta[0])
    p_da  = np.asarray(theta[1:1 + N], float)
    gamma = np.array([s.gamma for s in grid])
    a     = np.array([s.a for s in grid])
    pmax  = np.array([s.p_max for s in grid])
    l_min = -float(game.d[0])

    def obj(p):
        return float(np.sum(0.5 * gamma * (p - p_da) ** 2
                            + dt * (lam / 1000.0 - R_H2 * np.array([s.eta for s in grid])) * p))

    def grad(p):
        return gamma * (p - p_da) + dt * (lam / 1000.0 - R_H2 * np.array([s.eta for s in grid]))

    cons = [{"type": "ineq", "fun": lambda p: np.sum(p) - l_min}]   # Σp ≥ L_min
    bounds = [(0.0, pmax[i]) for i in range(N)]
    p0 = np.clip(np.full(N, l_min / N), 0.0, pmax)
    res = minimize(obj, p0, jac=grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    return res.x
