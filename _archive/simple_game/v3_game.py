"""
v3_game.py — H₂ cumulative demand + DAM anchor: the horizon becomes REAL.

This is the first rung where the horizon genuinely COUPLES the steps.  In v0–v2 the
cost was step-separable, the box was per-step and the only coupling (L_min ≤ Σp ≤ L_max)
was per-step, so the H-step game was exactly H independent copies of the single-step
game (the decoupling trick, proved in v1).  v3 adds one thing that breaks that:

  H₂ CUMULATIVE DEMAND (per agent, over the horizon):
      Σ_k η_i · p_elec_{i,k}  ≥  D_i          (produce ≥ D_i kg of H₂ this window)

Because it is a SUM over k, the steps are no longer independent — H₂ banked in a cheap
step reduces what must be bought in an expensive step.  The receding-horizon "state" is
D_i = kg still to produce today; it shrinks as the coalition produces and is a genuine
parameter.  So the horizon is a real θ dimension now, not H copies of a 1-step map.

Decisions locked with Parth (2026-07-10):
  • PER-AGENT demand D_i (one θ per agent, N total).  Faithful to per-electrolyzer
    offtake contracts; the full D-vector is public (coupling makes each agent's best
    response depend on all D_j), updated slowly like a day-ahead state — NOT per-5-min
    RTM communication.
  • DAM anchor FOLDED IN: cost has ½γ_i (p_{i,k} − p_DA_{i,k})², p_DA a baked-in
    constant (0 θ; maps rebuilt per hour in the closed loop).  p_DA=0 recovers v2.
  • ERCOT 15-min RTM framing → H steps = (lookahead)/15min.  H=2 ⇒ 30-min window.
    Synthetic θ for validation (real ERCOT prices only needed for the closed-loop week).

Buy-only preserved (p ≥ 0).  Still NO ramp, NO storage, NO batteries.

Parameter vector (n_p = H + R·H + N):
    θ = [ λ_0..λ_{H-1} , g_{r,0}..g_{r,H-1} (per renewable r) , D_0..D_{N-1} ]
      λ:  price forecast over the horizon           cols  0 .. H-1
      g:  renewable forecast (per renewable agent)  cols  H + r·H + k
      D:  per-agent H₂ demand remaining             cols  H + R·H + i

Per-agent cost / constraints (a_i = r_H2·η_i·1000  $/MWh):
  grid agent i   (x_i = [p_0..p_{H-1}], n_x = H):
      J_i = Σ_k [ ½γ_i (p_k − p_DA_{i,k})² + (λ_k − a_i) p_k ]
      0 ≤ p_k ≤ p_max_i        (per step)
      Σ_k η_i p_k ≥ D_i        (H₂ demand — couples the steps)
  renewable agent r (x_r = [p_0..p_{H-1}, cv_0..cv_{H-1}], n_x = 2H):
      J_r = Σ_k [ ½γ_r (p_k − p_DA_{r,k})² + ½ε cv_k² + (λ_k − a_r) p_k + a_r cv_k ]
      0 ≤ p_k ≤ p_elec_max,  0 ≤ cv_k ≤ min(g_max, g_k),  0 ≤ p_k+g_k−cv_k ≤ p_elec_max
      Σ_k η_r (p_k + g_k − cv_k) ≥ D_r     (free renewable energy counts toward H₂)
Coupling (grid imports only, per step):  L_min ≤ Σ_i p_{i,k} ≤ L_max.
"""

from __future__ import annotations
from dataclasses import dataclass, field
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
    p_da: float          # day-ahead award (baked-in constant, per step)
    d_max: float         # upper end of the H₂-demand θ range (feasibility-safe)
    @property
    def a(self): return R_H2 * self.eta * 1000.0


@dataclass(frozen=True)
class RenSpec:
    name: str; p_elec_max: float; eta: float; gamma: float; g_max: float
    p_da: float
    d_max: float
    @property
    def a(self): return R_H2 * self.eta * 1000.0


# Demand ranges chosen feasibility-safe: even at the worst corner (all D at max, g=0)
# the required grid energy Σ_i D_i/η_i stays under L_max·H, and each D_i is below its
# agent's producible η·p_max·H.  Binding at high λ (agent would otherwise sit at the
# L_min floor but must buy more to meet H₂).
DEFAULT_GRID = [
    GridSpec("PEM", 250.0, 0.020, 0.24, p_da=100.0, d_max=5.0),   # a=60, producible=10
    GridSpec("ALK", 200.0, 0.018, 0.27, p_da=80.0,  d_max=4.0),   # a=54, producible=7.2
]
DEFAULT_REN = [
    RenSpec("PEM_PV", 200.0, 0.019, 0.30, g_max=150.0, p_da=60.0, d_max=4.0),  # a=57, prod≤7.6
]
DEFAULT_L_MIN = 100.0
DEFAULT_L_MAX = 400.0
DEFAULT_LAM = (0.0, 80.0)


@dataclass
class V3Meta:
    """θ layout + fleet bookkeeping so test/plot code can index θ unambiguously."""
    H: int
    R: int
    N: int
    n_p: int
    specs: list
    ren_indices: list          # agent indices that are renewable
    def lam_col(self, k):      return k
    def g_col(self, r, k):     return self.H + r * self.H + k
    def d_col(self, i):        return self.H + self.R * self.H + i


def build_v3_game(
    H: int = 2,
    grid: list[GridSpec] | None = None,
    ren: list[RenSpec] | None = None,
    l_min: float = DEFAULT_L_MIN,
    l_max: float = DEFAULT_L_MAX,
    lam_lb: float = DEFAULT_LAM[0],
    lam_ub: float = DEFAULT_LAM[1],
) -> tuple[GNEGame, V3Meta]:
    """Build the H-step v3 game with per-agent H₂ cumulative demand + DAM anchor."""
    grid = list(grid) if grid is not None else list(DEFAULT_GRID)
    ren = list(ren) if ren is not None else list(DEFAULT_REN)
    R = len(ren)
    N = len(grid) + R
    n_p = H + R * H + N
    specs = list(grid) + list(ren)
    ren_indices = list(range(len(grid), N))
    meta = V3Meta(H=H, R=R, N=N, n_p=n_p, specs=specs, ren_indices=ren_indices)

    I = np.eye(H)
    agents: list[Agent] = []

    # ── grid-only agents: x = [p_0..p_{H-1}] ─────────────────────────────────────
    for i, s in enumerate(grid):
        Q = s.gamma * I
        c = np.full(H, -s.a - s.gamma * s.p_da)          # (λ_k−a)p + ½γ(p−p_DA)² linear const
        F = np.zeros((H, n_p))
        for k in range(H):
            F[k, meta.lam_col(k)] = 1.0                  # +λ_k·p_k
        # local: per-step box [I;-I] (2H rows) + one H₂ cumulative row
        A_box = np.vstack([I, -I])                       # (2H, H)
        b_box = np.concatenate([s.p_max * np.ones(H), np.zeros(H)])
        S_box = np.zeros((2 * H, n_p))
        A_h2 = -s.eta * np.ones((1, H))                  # −η Σ p_k ≤ −D_i
        b_h2 = np.zeros(1)
        S_h2 = np.zeros((1, n_p)); S_h2[0, meta.d_col(i)] = -1.0
        A_loc = np.vstack([A_box, A_h2])
        b_loc = np.concatenate([b_box, b_h2])
        S_loc = np.vstack([S_box, S_h2])
        C = np.vstack([I, -I])                           # per-step coupling (2H, H)
        agents.append(Agent(index=i, n_x=H, Q=Q, c=c, F=F,
                            A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C))

    # ── renewable agents: x = [p_0..p_{H-1}, cv_0..cv_{H-1}] ─────────────────────
    for r, s in enumerate(ren):
        idx = len(grid) + r
        nx = 2 * H
        Q = np.zeros((nx, nx))
        Q[:H, :H] = s.gamma * I
        Q[H:, H:] = EPS_CV * I
        c = np.concatenate([np.full(H, -s.a - s.gamma * s.p_da), np.full(H, s.a)])
        F = np.zeros((nx, n_p))
        for k in range(H):
            F[k, meta.lam_col(k)] = 1.0                  # +λ_k·p_k (cv has no param-linear term)

        # per-step local rows (7 per step), then one H₂ cumulative row
        A_rows, b_rows, S_rows = [], [], []
        for k in range(H):
            pk, cvk = k, H + k
            gcol = meta.g_col(r, k)
            def row(coeffs):
                v = np.zeros(nx)
                for pos, val in coeffs: v[pos] = val
                return v
            # 1: p_k ≤ p_elec_max
            A_rows.append(row([(pk, 1.0)]));            b_rows.append(s.p_elec_max); S_rows.append(np.zeros(n_p))
            # 2: cv_k ≤ g_max
            A_rows.append(row([(cvk, 1.0)]));           b_rows.append(s.g_max);      S_rows.append(np.zeros(n_p))
            # 3: -p_k ≤ 0
            A_rows.append(row([(pk, -1.0)]));           b_rows.append(0.0);          S_rows.append(np.zeros(n_p))
            # 4: -cv_k ≤ 0
            A_rows.append(row([(cvk, -1.0)]));          b_rows.append(0.0);          S_rows.append(np.zeros(n_p))
            # 5: cv_k ≤ g_k              (S[gcol]=+1)
            sv = np.zeros(n_p); sv[gcol] = 1.0
            A_rows.append(row([(cvk, 1.0)]));           b_rows.append(0.0);          S_rows.append(sv)
            # 6: p_k − cv_k ≤ p_elec_max − g_k    (p+g-cv ≤ P; S[gcol]=-1)
            sv = np.zeros(n_p); sv[gcol] = -1.0
            A_rows.append(row([(pk, 1.0), (cvk, -1.0)])); b_rows.append(s.p_elec_max); S_rows.append(sv)
            # 7: −p_k + cv_k ≤ g_k       (p+g-cv ≥ 0; S[gcol]=+1)
            sv = np.zeros(n_p); sv[gcol] = 1.0
            A_rows.append(row([(pk, -1.0), (cvk, 1.0)])); b_rows.append(0.0);         S_rows.append(sv)
        # H₂: −η Σ_k (p_k+g_k−cv_k) ≤ −D_r  ⇔ −ηΣp_k + ηΣcv_k ≤ −D_r + ηΣg_k
        h2 = np.zeros(nx); h2[:H] = -s.eta; h2[H:] = s.eta
        s_h2 = np.zeros(n_p); s_h2[meta.d_col(idx)] = -1.0
        for k in range(H):
            s_h2[meta.g_col(r, k)] = s.eta
        A_rows.append(h2); b_rows.append(0.0); S_rows.append(s_h2)

        A_loc = np.vstack(A_rows)
        b_loc = np.array(b_rows)
        S_loc = np.vstack(S_rows)
        C = np.zeros((2 * H, nx))
        C[:H, :H] = I; C[H:, :H] = -I                    # only p in coupling
        agents.append(Agent(index=idx, n_x=nx, Q=Q, c=c, F=F,
                            A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C))

    d = np.concatenate([l_max * np.ones(H), -l_min * np.ones(H)])
    S_coup = np.zeros((2 * H, n_p))
    p_lb = np.concatenate([
        lam_lb * np.ones(H),                              # λ
        np.zeros(R * H),                                  # g
        np.zeros(N),                                      # D
    ])
    p_ub = np.concatenate([
        lam_ub * np.ones(H),
        np.concatenate([s.g_max * np.ones(H) for s in ren]) if R else np.zeros(0),
        np.array([s.d_max for s in specs]),
    ])
    game = GNEGame(agents=agents, d=d, S_coup=S_coup, d_lb=None, S_coup_lb=None,
                   p_lb=p_lb, p_ub=p_ub)
    return game, meta


def centralized_qp(game: GNEGame, theta: np.ndarray) -> np.ndarray | None:
    """
    Exact variational GNE = argmin ½xᵀQx+(c+Fθ)ᵀx s.t. Gx ≤ w0+Wθ, via Gurobi.
    This is the DEFINITION of the variational equilibrium (potential-minimising QP),
    so it is an independent, reliable reference (unlike SLSQP, which drifts at binding
    points).  Returns stacked x, or None if infeasible.
    """
    import gurobipy as gp
    from gurobipy import GRB
    from amrhg.solvers.gne_combiner import _stacked_cost, _centralized_constraints
    Q, c, F = _stacked_cost(game)
    G, w0, W = _centralized_constraints(game)
    theta = np.asarray(theta).ravel()
    q = c + F @ theta
    rhs = w0 + W @ theta
    n = Q.shape[0]
    m = gp.Model(); m.Params.OutputFlag = 0
    x = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    m.setObjective(0.5 * (x @ Q @ x) + q @ x, GRB.MINIMIZE)
    m.addConstr(G @ x <= rhs)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return None
    return np.array(x.X)
