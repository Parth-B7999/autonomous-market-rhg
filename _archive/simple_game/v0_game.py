"""
v0_game.py — The simplest possible market GNE for the iteration-free RTM paper.

v0 (pipeline validation, per PLAN_mp_focapo_rtm.md):
  • N grid-only electrolyzer agents, BUY-ONLY (p_i >= 0, no export).
  • No states, no horizon, no batteries, no H2 tracking.  H = 1 → each 5-min step
    is an independent single-shot game.
  • The ONLY parameter is the real-time price λ (1 scalar).  θ_i = [x_{-i}; λ];
    after the equilibrium solve the PUBLIC map is over λ alone (1-D).

Per-agent problem (single step):

    min_{p_i}   0.5 * γ_i * p_i^2  +  (λ − a_i) * p_i
    s.t.        0 <= p_i <= p_max_i

    where a_i = r_H2 * η_i * 1000  is the H2 marginal revenue expressed in the same
    $/MWh units as λ, so the linear coefficient (λ − a_i) flips sign at λ = a_i.
    γ_i > 0 is a small operating-cost curvature that makes each agent's problem
    strictly convex (unique best response, PD Hessian for PPOPT).

Coupling (the market rule — the whole game):

    L_min <= Σ_i p_i <= L_max

  Encoded as a TWO-ROW upper-form coupling so that mp_solver, gne_combiner AND
  admm_solver (which all read only C / d, not d_lb) solve the *identical* polytope:

    C_i = [[+1], [-1]]        (2 rows, 1 col)
    d   = [L_max, −L_min]     Σp ≤ L_max  and  −Σp ≤ −L_min ⟺ Σp ≥ L_min
    S_coup = 0                (coupling RHS is constant, not parametric)

  d_lb is left None on purpose (its bilateral-clip path is ADMM-only and would be
  double counting here).

Three price regimes make the game non-trivial across λ ∈ [λ_min, λ_max]:
  • cheap  λ  → everyone wants max power → Σp hits L_max   (who gets headroom)
  • mid    λ  → interior unconstrained best responses       (coupling slack)
  • dear   λ  → everyone wants p=0 but L_min forces Σp≥L_min (who covers the floor)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Make amrhg importable whether or not the package is pip-installed.
import sys
from pathlib import Path
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from amrhg.solvers.game import Agent, GNEGame


R_H2 = 3.0  # $/kg hydrogen price (fixed)


@dataclass(frozen=True)
class V0AgentSpec:
    name:   str
    p_max:  float   # kW  capacity
    eta:    float   # kg/kWh electrolyzer efficiency
    gamma:  float   # $/MWh per kW  operating-cost curvature

    @property
    def a(self) -> float:
        """H2 marginal revenue in $/MWh units (matches λ)."""
        return R_H2 * self.eta * 1000.0


# Default 3-agent grid-only fleet (heterogeneous γ, η, p_max).
# Tuned so all three regimes (L_max / interior / L_min) appear for λ ∈ [0, 80].
DEFAULT_FLEET = [
    V0AgentSpec(name="PEM",  p_max=250.0, eta=0.020, gamma=0.24),  # a = 60
    V0AgentSpec(name="ALK",  p_max=200.0, eta=0.018, gamma=0.27),  # a = 54
    V0AgentSpec(name="PEM2", p_max=150.0, eta=0.019, gamma=0.40),  # a = 57
]

DEFAULT_L_MIN = 100.0
DEFAULT_L_MAX = 400.0
DEFAULT_LAM   = (0.0, 80.0)   # ($/MWh) parameter box on λ


def build_v0_game(
    fleet:  list[V0AgentSpec] | None = None,
    l_min:  float = DEFAULT_L_MIN,
    l_max:  float = DEFAULT_L_MAX,
    lam_lb: float = DEFAULT_LAM[0],
    lam_ub: float = DEFAULT_LAM[1],
) -> tuple[GNEGame, list[V0AgentSpec]]:
    """Build the v0 GNEGame (1 parameter = λ). Returns (game, fleet)."""
    fleet = list(fleet) if fleet is not None else list(DEFAULT_FLEET)
    N = len(fleet)
    n_p = 1  # λ only

    agents: list[Agent] = []
    for i, spec in enumerate(fleet):
        Q = np.array([[spec.gamma]])            # 0.5 γ p^2
        c = np.array([-spec.a])                 # constant linear part: −a_i
        F = np.array([[1.0]])                    # + λ · p_i   (coef on the single param)

        # local box: 0 <= p_i <= p_max  →  [[+1],[-1]] p <= [p_max, 0]
        A_loc = np.array([[1.0], [-1.0]])
        b_loc = np.array([spec.p_max, 0.0])
        S_loc = np.zeros((2, n_p))

        # coupling block: two rows enforce L_min <= Σp <= L_max
        C = np.array([[1.0], [-1.0]])

        agents.append(Agent(
            index=i, n_x=1, Q=Q, c=c, F=F,
            A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C,
        ))

    d       = np.array([l_max, -l_min])
    S_coup  = np.zeros((2, n_p))
    game = GNEGame(
        agents=agents,
        d=d, S_coup=S_coup, d_lb=None, S_coup_lb=None,
        p_lb=np.array([lam_lb]), p_ub=np.array([lam_ub]),
    )
    return game, fleet


def centralized_gne(game: GNEGame, fleet: list[V0AgentSpec], lam: float) -> np.ndarray:
    """
    Ground-truth variational GNE = argmin Σ_i J_i(p_i; λ) over the shared feasible
    set.  Valid because the game is an exact potential game (separable costs, one
    convex shared constraint): the v-GNE minimizes the potential Σ_i J_i.

    Solved as a tiny box-constrained QP with 2 linear coupling rows via SLSQP.
    """
    from scipy.optimize import minimize
    N = game.N
    gamma = np.array([s.gamma for s in fleet])
    a     = np.array([s.a for s in fleet])
    pmax  = np.array([s.p_max for s in fleet])
    l_max, l_min = game.d[0], -game.d[1]

    def obj(p):
        return float(np.sum(0.5 * gamma * p**2 + (lam - a) * p))

    def grad(p):
        return gamma * p + (lam - a)

    cons = [
        {"type": "ineq", "fun": lambda p: l_max - np.sum(p)},   # Σp ≤ L_max
        {"type": "ineq", "fun": lambda p: np.sum(p) - l_min},   # Σp ≥ L_min
    ]
    bounds = [(0.0, pmax[i]) for i in range(N)]
    p0 = np.clip(np.full(N, l_min / N), 0.0, pmax)
    res = minimize(obj, p0, jac=grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    return res.x
