"""
vgne_select.py — box-constrained variational-GNE selection (Hall-Bemporad eq. 17),
done exactly.  NEW module; does NOT modify filter_variational_kkt (v0-v2 use that).

Motivation: filter_variational_kkt builds an OFFLINE single-valued map by clipping
CRs, which drops coverage at degenerate box+coupling vertices (binding markets) →
gaps.  The reference select_v_gne solves the equal-multiplier condition UNCONSTRAINED
→ box-violating points.  The paper's eq. (17) is the CONSTRAINED v-GNE selection over
each critical region.  Here we implement it exactly:

For a query θ, among all GNE CRs covering θ (gf.locate_all), for each CR solve
    min_x  ½ xᵀQx + (c+Fθ)ᵀx           (the potential)
    s.t.   M_x x = M_p θ + M_1          (this CR's equilibrium manifold — pins the combo)
           G x ≤ w0 + W θ               (all boxes + shared coupling)
and return the feasible minimiser with the lowest potential across covering CRs.
For a potential game this is exactly the variational GNE (= centralized optimum),
respecting the boxes — gap-free and exact.

Uses Gurobi for the small per-CR QP (exact).  This is an OFFLINE coordinator step
(it has every agent's cost); the online agents still only read their own slice.
"""
from __future__ import annotations
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .game import GNEGame
from .gne_combiner import _stacked_cost, _centralized_constraints

_ENV = None
def _env():
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True); _ENV.setParam("OutputFlag", 0); _ENV.start()
    return _ENV


class VariationalSelector:
    """Exact eq.(17) variational-GNE selection over a combiner GNESolution."""

    def __init__(self, game: GNEGame):
        self.game = game
        self.Q, self.c, self.F = _stacked_cost(game)
        self.G, self.w0, self.W = _centralized_constraints(game)
        self.n = self.Q.shape[0]

    def _cr_solve(self, cr, theta, q, rhs):
        """min potential s.t. this CR's equilibrium manifold + global feasibility."""
        m = gp.Model(env=_env())
        x = m.addMVar(self.n, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        m.setObjective(0.5 * (x @ self.Q @ x) + q @ x, GRB.MINIMIZE)
        m.addConstr(self.G @ x <= rhs)
        # equilibrium manifold of this combo:  M_x x = M_p θ + M_1
        rhs_eq = np.atleast_1d(cr.Mp @ theta + cr.M1).ravel()
        m.addConstr(cr.Mx @ x == rhs_eq)
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            return None, np.inf
        xv = np.array(x.X)
        return xv, float(0.5 * xv @ self.Q @ xv + q @ xv)

    def evaluate(self, gf, theta, tol=1e-6):
        """Return the variational GNE x*(θ), or None if θ is outside all CRs."""
        theta = np.asarray(theta).ravel()
        ks = gf.locate_all(theta, tol=1e-7)
        if not ks:
            return None
        q = self.c + self.F @ theta
        rhs = self.w0 + self.W @ theta
        best, bJ = None, np.inf
        for k in ks:
            xv, J = self._cr_solve(gf[k], theta, q, rhs)
            if xv is not None and np.all(self.G @ xv <= rhs + 1e-5) and J < bJ - tol:
                bJ, best = J, xv
        return best
