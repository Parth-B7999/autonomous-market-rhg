"""
v2_game.py — Add renewable agents (electrolyzer + co-located PV/wind).

What makes the renewable forecast g a GENUINE parameter (not a constant offset):
a renewable agent's electrolyzer has finite capacity P_elec_max.  Its total load is
p_elec = p + g − cv (grid import + renewable − curtailment).  So

    0 ≤ p + g − cv ≤ P_elec_max,   0 ≤ cv ≤ g,   p ≥ 0,

meaning g EATS electrolyzer headroom: more renewable ⇒ less room for grid import p
(unless you curtail, which throws away free H₂ at cost a·cv).  Thus g shifts the
agent's grid bid p and, through the coupling L_min ≤ Σp ≤ L_max, the whole game.
The single-step public map is therefore 2-D in (λ, g) (with R renewable agents:
(λ, g_0, …, g_{R-1}), n_p = 1+R).

Buy-only preserved (p ≥ 0, no export).  Still NO ramp / storage / H₂ tracking ⇒ the
H-step game still DECOUPLES per step (validated in v2_test.py); the horizon becomes
real only at v3 (H₂ cumulative demand).

Single-step decision & cost:
  grid-only agent i:  x_i = [p_i],       J_i = ½γ_i p_i² + (λ − a_i) p_i
  renewable  agent r: x_r = [p_r, cv_r], J_r = ½γ_r p_r² + ½ε cv_r²
                                              + (λ − a_r) p_r + a_r cv_r
  a_i = r_H2·η_i·1000  ($/MWh).  ε: tiny curtailment curvature for a PD Hessian.
Coupling (grid imports only):  L_min ≤ Σ_i p_i ≤ L_max.
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

R_H2 = 3.0
EPS_CV = 1e-3   # curtailment curvature (PD Hessian for PPOPT)


@dataclass(frozen=True)
class GridSpec:
    name: str; p_max: float; eta: float; gamma: float
    @property
    def a(self): return R_H2 * self.eta * 1000.0


@dataclass(frozen=True)
class RenSpec:
    name: str; p_elec_max: float; eta: float; gamma: float; g_max: float
    @property
    def a(self): return R_H2 * self.eta * 1000.0


DEFAULT_GRID = [
    GridSpec("PEM", 250.0, 0.020, 0.24),   # a=60
    GridSpec("ALK", 200.0, 0.018, 0.27),   # a=54
]
DEFAULT_REN = [
    RenSpec("PEM_PV", 200.0, 0.019, 0.30, g_max=150.0),   # a=57, PV up to 150 kW
]
DEFAULT_L_MIN = 100.0
DEFAULT_L_MAX = 400.0
DEFAULT_LAM = (0.0, 80.0)


def build_v2_step_game(
    grid: list[GridSpec] | None = None,
    ren: list[RenSpec] | None = None,
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
    lam_lb: float = DEFAULT_LAM[0],
    lam_ub: float = DEFAULT_LAM[1],
) -> tuple[GNEGame, list, int]:
    """
    Single-step v2 game.  Parameter vector p_gne = [λ, g_0, …, g_{R-1}], n_p = 1+R.
    Returns (game, specs_in_order, n_ren).  Agent order: grid agents then renewable.
    """
    grid = list(grid) if grid is not None else list(DEFAULT_GRID)
    ren = list(ren) if ren is not None else list(DEFAULT_REN)
    R = len(ren)
    n_p = 1 + R                       # [λ, g_0, ..., g_{R-1}]
    specs = list(grid) + list(ren)
    agents: list[Agent] = []

    # ── grid-only agents: x=[p], couples via p ────────────────────────────────
    for idx, s in enumerate(grid):
        Q = np.array([[s.gamma]])
        c = np.array([-s.a])
        F = np.zeros((1, n_p)); F[0, 0] = 1.0            # +λ·p
        A_loc = np.array([[1.0], [-1.0]])                # 0 ≤ p ≤ p_max
        b_loc = np.array([s.p_max, 0.0])
        S_loc = np.zeros((2, n_p))
        C = np.array([[1.0], [-1.0]])                    # p in coupling
        agents.append(Agent(index=idx, n_x=1, Q=Q, c=c, F=F,
                            A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C))

    # ── renewable agents: x=[p, cv], param g_r = p_gne[1+r] ────────────────────
    for r, s in enumerate(ren):
        idx = len(grid) + r
        gcol = 1 + r                                     # index of g_r in p_gne
        Q = np.array([[s.gamma, 0.0], [0.0, EPS_CV]])
        c = np.array([-s.a, s.a])                        # (λ−a)p + a·cv
        F = np.zeros((2, n_p)); F[0, 0] = 1.0            # +λ·p (cv has no param-linear term)

        # local rows: box [+I;-I] first (for _extract_box_bounds), then parametric
        #  0: p ≤ p_elec_max            5:  cv ≤ g            → S_loc[·,gcol]=+1
        #  1: cv ≤ g_max               6:  p−cv ≤ P−g        → S_loc[·,gcol]=-1
        #  2: -p ≤ 0                    7: -p+cv ≤ g         → S_loc[·,gcol]=+1
        #  3: -cv ≤ 0
        A_loc = np.array([
            [ 1.0,  0.0],   # p ≤ p_elec_max
            [ 0.0,  1.0],   # cv ≤ g_max
            [-1.0,  0.0],   # p ≥ 0
            [ 0.0, -1.0],   # cv ≥ 0
            [ 0.0,  1.0],   # cv ≤ g
            [ 1.0, -1.0],   # p_elec = p+g-cv ≤ P_elec_max
            [-1.0,  1.0],   # p_elec = p+g-cv ≥ 0
        ])
        b_loc = np.array([s.p_elec_max, s.g_max, 0.0, 0.0, 0.0, s.p_elec_max, 0.0])
        S_loc = np.zeros((7, n_p))
        S_loc[4, gcol] = 1.0
        S_loc[5, gcol] = -1.0
        S_loc[6, gcol] = 1.0
        C = np.array([[1.0, 0.0], [-1.0, 0.0]])          # only p in coupling
        agents.append(Agent(index=idx, n_x=2, Q=Q, c=c, F=F,
                            A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C))

    d = np.array([l_max, -l_min])
    S_coup = np.zeros((2, n_p))
    p_lb = np.array([lam_lb] + [0.0] * R)
    p_ub = np.array([lam_ub] + [s.g_max for s in ren])
    game = GNEGame(agents=agents, d=d, S_coup=S_coup, d_lb=None, S_coup_lb=None,
                   p_lb=p_lb, p_ub=p_ub)
    return game, specs, R


def centralized_step(game: GNEGame, specs, n_ren: int, theta: np.ndarray) -> np.ndarray:
    """Ground-truth variational GNE for the single-step game = argmin Σ J_i s.t.
    boxes + coupling.  theta = [λ, g_0, …].  Returns stacked x (game ordering)."""
    from scipy.optimize import minimize
    lam = float(theta[0])
    l_max, l_min = float(game.d[0]), float(-game.d[1])

    slices, x0 = [], []
    off = 0
    for a in game.agents:
        slices.append(slice(off, off + a.n_x)); off += a.n_x
    n = off

    def split(x):
        return [x[s] for s in slices]

    def obj(x):
        xs = split(x); J = 0.0
        gi = 0
        for a, s in zip(game.agents, specs):
            xi = xs[a.index]
            if a.n_x == 1:
                J += 0.5 * s.gamma * xi[0]**2 + (lam - s.a) * xi[0]
            else:
                J += (0.5 * s.gamma * xi[0]**2 + 0.5 * EPS_CV * xi[1]**2
                      + (lam - s.a) * xi[0] + s.a * xi[1])
        return J

    cons = [
        {"type": "ineq", "fun": lambda x: l_max - sum(split(x)[a.index][0] for a in game.agents)},
        {"type": "ineq", "fun": lambda x: sum(split(x)[a.index][0] for a in game.agents) - l_min},
    ]
    bounds = []; gi = 0
    for a, s in zip(game.agents, specs):
        if a.n_x == 1:
            bounds.append((0.0, s.p_max))
        else:
            g = float(theta[1 + gi]); gi += 1
            bounds.append((0.0, s.p_elec_max))     # p
            bounds.append((0.0, g))                # cv ≤ g
            # p_elec box handled via constraints below
    # p_elec constraints for renewable agents
    gi = 0
    for a, s in zip(game.agents, specs):
        if a.n_x == 2:
            g = float(theta[1 + gi]); gi += 1
            sl = slices[a.index]
            cons.append({"type": "ineq",
                         "fun": (lambda x, sl=sl, g=g, P=s.p_elec_max:
                                 P - (x[sl][0] + g - x[sl][1]))})   # p+g-cv ≤ P
            cons.append({"type": "ineq",
                         "fun": (lambda x, sl=sl, g=g: (x[sl][0] + g - x[sl][1]))})  # ≥0
    x0 = np.array([0.5 * (b[0] + b[1]) for b in bounds])
    res = minimize(obj, x0, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-12, "maxiter": 800})
    return res.x
