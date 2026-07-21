"""
residual.py — the NATURAL RESIDUAL, the single stopping/quality metric shared by every
solver in the comparison (FACET, ADMM, Douglas–Rachford).

Why this metric and not each solver's own:
  ADMM stops on its primal/dual residual; DR stops on its fixed-point gap; FACET stops on
  combination self-consistency.  Those are three different scales and cannot be compared.
  The natural residual is defined on the PROBLEM, not on any algorithm, so all three can be
  held to the identical bar.  This mirrors Benenati & Belgioioso (arXiv:2512.07749), §IV,
  who benchmark every solver at r(u) < 1e-6.

Definition.  Our RTM clearing is the variational inequality VI(F, U) with
    F(x, θ) = Q x + c + F_θ θ          (affine pseudo-gradient; = ∇Φ, since this is a
                                        potential game — see report §2)
    U(θ)    = { x : G x ≤ w0 + W θ }   (the coalition's joint feasible set)
and the natural residual of a point x is
    r(x) = ‖ x − Proj_{U(θ)}( x − F(x, θ) ) ‖₂ .
r(x) = 0  ⟺  x solves the VI  ⟺  x is the variational GNE.

The projection is itself a small QP (40 vars, 142 rows here), solved with OSQP.

IMPORTANT (benchmark fairness): computing r() is INSTRUMENTATION, not solver work.  Callers
must exclude its wall time from any solver's clock.  See bench_solvers.py, which pauses the
timer around every residual evaluation.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_OSQP_CACHE: dict[int, tuple] = {}


def _problem_data(game):
    """(Q, c, F_θ, G, w0, W) for `game`, cached on the game object id."""
    from .gne_combiner import _stacked_cost, _centralized_constraints
    k = id(game)
    if k not in _OSQP_CACHE:
        Q, c, Fp = _stacked_cost(game)
        G, w0, W = _centralized_constraints(game)
        _OSQP_CACHE[k] = (np.asarray(Q, float), np.asarray(c, float), np.asarray(Fp, float),
                          np.asarray(G, float), np.asarray(w0, float), np.asarray(W, float))
    return _OSQP_CACHE[k]


def pseudo_gradient(game, x, theta):
    """F(x, θ) = Q x + c + F_θ θ."""
    Q, c, Fp, _, _, _ = _problem_data(game)
    return Q @ x + c + Fp @ theta


class Projector:
    """Euclidean projection onto U(θ) = {x : G x ≤ w0 + W θ}, via OSQP.

    The quadratic form of a Euclidean projection is the identity and therefore CONSTANT, so
    the OSQP factorization is set up once and only the linear term / RHS are updated per call.
    That keeps the projection cheap enough to use inside a DR iteration.
    """

    def __init__(self, game, eps=1e-10, max_iter=20000):
        import osqp
        Q, c, Fp, G, w0, W = _problem_data(game)
        self.n = Q.shape[0]
        self.G, self.w0, self.W = G, w0, W
        P = sp.identity(self.n, format="csc")               # ½‖x − v‖²  → P = I
        A = sp.csc_matrix(G)
        self._lo = np.full(G.shape[0], -np.inf)
        self.m = osqp.OSQP()
        self.m.setup(P=P, q=np.zeros(self.n), A=A, l=self._lo, u=np.zeros(G.shape[0]),
                     eps_abs=eps, eps_rel=eps, max_iter=max_iter,
                     polish=True, verbose=False)
        self._ok = True

    def __call__(self, v, theta):
        """argmin_x ½‖x − v‖² s.t. G x ≤ w0 + W θ."""
        u = self.w0 + self.W @ theta
        self.m.update(q=-np.asarray(v, float), u=u)
        res = self.m.solve()
        if res.info.status not in ("solved", "solved inaccurate"):
            self._ok = False
            return None
        return np.asarray(res.x, float)


_PROJ: dict[int, Projector] = {}


def get_projector(game) -> Projector:
    k = id(game)
    if k not in _PROJ:
        _PROJ[k] = Projector(game)
    return _PROJ[k]


def natural_residual(game, x, theta, proj: Projector | None = None) -> float:
    """r(x) = ‖x − Proj_{U(θ)}(x − F(x,θ))‖₂.   Returns np.inf if the projection fails."""
    p = proj if proj is not None else get_projector(game)
    x = np.asarray(x, float)
    y = p(x - pseudo_gradient(game, x, theta), theta)
    if y is None:
        return float("inf")
    return float(np.linalg.norm(x - y))


# ─────────────────────────────────────────────────────────────────────────────
#  Reference solution (the benchmark's ground truth)
# ─────────────────────────────────────────────────────────────────────────────
#
# MEASURED 2026-07-16 — why we do NOT use Benenati & Belgioioso's r < 1e-6 bar:
#   The natural residual has a numerical floor on THIS problem of ~1.7e-6 (worst case over
#   random θ; ~1.5e-7 typical), even when x is produced by Gurobi at OptimalityTol=1e-9.
#   Cause: our RTM cost is nearly LINEAR — ‖Q‖₂ = 5e-3, since the quadratic part is only a
#   small regularizer (γ_i ~ 4-5e-3, EPS_CV = 1e-3) on top of a price·power linear term. The
#   optimum therefore sits near a vertex, ‖x‖ ≈ 605, and r(x) is a difference of two vectors
#   of that magnitude. Asking for r < 1e-6 is asking for ~1.6e-9 RELATIVE accuracy, which is
#   below what the oracle itself achieves. Benchmarking against a bar the ground truth cannot
#   clear would produce meaningless "never converged" rows.
#
#   We therefore report TWO metrics:
#     PRIMARY   ‖x − x_ref‖_∞ < 1e-4 kW — physical (kW), and exactly the pipeline's own
#               STRICT_TOL (rhg_online.py:274). This is the bar that matters for dispatch.
#     SECONDARY r(x) < 1e-5 — the Benenati-style metric, at a tolerance one decade above the
#               measured floor. Reported for comparability with the cited literature.

_REF_TOL = 1e-9


def reference_solution(game, theta, tol: float = _REF_TOL):
    """High-accuracy variational GNE: argmin ½xᵀQx + (c+Fθ)ᵀx s.t. Gx ≤ w0+Wθ.

    This is `rhg_online.centralized` with the Gurobi tolerances tightened and NumericFocus
    raised — the pipeline's oracle runs at Gurobi defaults (~1e-6), which is not accurate
    enough to serve as benchmark ground truth. Returns None if not solved to optimality.
    """
    import gurobipy as gp
    from gurobipy import GRB
    Q, c, Fp, G, w0, W = _problem_data(game)
    q = c + Fp @ theta
    rhs = w0 + W @ theta
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.OptimalityTol = tol
    m.Params.FeasibilityTol = tol
    m.Params.BarConvTol = tol
    m.Params.NumericFocus = 3
    x = m.addMVar(Q.shape[0], lb=-GRB.INFINITY, ub=GRB.INFINITY)
    m.setObjective(0.5 * (x @ Q @ x) + q @ x, GRB.MINIMIZE)
    m.addConstr(G @ x <= rhs)
    m.optimize()
    return np.array(x.X) if m.Status == GRB.OPTIMAL else None
