"""
dr_solver.py — Douglas–Rachford operator splitting for the RTM variational inequality.

This is a BASELINE for the solver comparison, not part of the FACET pipeline. It is the
operator-splitting family Benenati & Belgioioso (arXiv:2512.07749) benchmark against
(their ref [34], Eckstein & Ferris, "Operator-Splitting Methods for Monotone Affine
Variational Inequalities").

Problem.  Find x solving VI(F, U):   0 ∈ F(x, θ) + N_{U(θ)}(x),   with
    F(x, θ) = Q x + c + F_θ θ        (affine, monotone: Q ≻ 0)
    U(θ)    = { x : G x ≤ w0 + W θ } (polyhedral)

Douglas–Rachford splits this into the two operators A = F (single-valued, affine) and
B = N_U (normal cone, whose resolvent is the Euclidean projection):

    x^k     = J_{γA}(z^k)  = (I + γQ)^{-1} (z^k − γ q)     ← ONE linear solve (pre-factored)
    y^k     = J_{γB}(2x^k − z^k) = Proj_{U(θ)}(2x^k − z^k) ← ONE small QP (OSQP)
    z^{k+1} = z^k + (y^k − x^k)

with q = c + F_θ θ. Since Q is constant, (I + γQ) is factored ONCE per γ and reused for
every θ and every iteration — the per-iteration cost is dominated by the projection.

═══════════════════════════════════════════════════════════════════════════════════════
HONESTY — DR IS A CENTRALIZED BASELINE, NOT A PRIVACY-PRESERVING ONE.
  Proj_{U(θ)} is a projection onto the COALITION's joint feasible set: it couples all six
  agents' variables in one QP, so evaluating it requires a party that knows every agent's
  local constraint rows. DR therefore does NOT satisfy this paper's privacy model, and it
  is not a distributed method. It is included because it is the operator-splitting baseline
  of the cited work and it bounds what a *centralized* iterative solver achieves.
  ADMM is the only baseline here that competes with FACET on model privacy.
═══════════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from .residual import _problem_data, get_projector


@dataclass
class DRResult:
    """Mirrors ADMMResult's vocabulary so the benchmark can treat them uniformly."""
    x_sol: np.ndarray
    n_iter: int
    converged: bool
    solve_time: float = 0.0
    # trace[k] = (cumulative solver seconds after iteration k, iterate x^k).
    # Instrumentation time (residual evaluation) is NEVER included — the caller pauses.
    trace: list = field(default_factory=list)

    @property
    def x_stacked(self):
        return self.x_sol


_FACT: dict[tuple, tuple] = {}


def _resolvent_factor(game, gamma: float):
    """Cholesky factor of (I + γQ), cached per (game, γ). Q is constant across θ."""
    key = (id(game), float(gamma))
    if key not in _FACT:
        Q, c, Fp, _, _, _ = _problem_data(game)
        n = Q.shape[0]
        M = np.eye(n) + gamma * Q
        _FACT[key] = (sla.cho_factor(M, lower=True), Q, c, Fp)
    return _FACT[key]


def dr_solve(game, theta, gamma: float = 1.0, max_iter: int = 5000,
             tol: float = 1e-10, x_init=None, trace: bool = True, stop_fn=None) -> DRResult:
    """Douglas–Rachford for VI(F, U(θ)).

    Args:
      gamma:    DR step size. Tuned in the Phase-1 smoke test (see bench_solvers.py).
      max_iter: hard cap on iterations.
      tol:      fixed-point tolerance ‖y^k − x^k‖. Deliberately TIGHT by default: the
                benchmark judges convergence by the SHARED metric (distance to the
                reference solution / natural residual), computed post-hoc from `trace`,
                not by this internal test. Set tight so the solver keeps improving and the
                trace covers the whole accuracy range.
      x_init:   warm start (stacked). None → cold start at 0.
      trace:    record (t_k, x_k) each iteration. The clock is paused around bookkeeping.

    Returns DRResult. `converged` refers only to the internal fixed-point test.
    """
    cho, Q, c, Fp = _resolvent_factor(game, gamma)
    proj = get_projector(game)
    q = c + Fp @ np.asarray(theta, float)
    gq = gamma * q
    n = Q.shape[0]

    z = np.zeros(n) if x_init is None else np.asarray(x_init, float).copy()
    tr = []
    elapsed = 0.0
    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        t0 = time.perf_counter()
        x = sla.cho_solve(cho, z - gq)              # J_{γF}(z)
        y = proj(2.0 * x - z, theta)                # Proj_U(2x − z)
        if y is None:                               # projection failed → abort honestly
            elapsed += time.perf_counter() - t0
            return DRResult(x_sol=x, n_iter=k, converged=False, solve_time=elapsed, trace=tr)
        z = z + (y - x)
        gap = float(np.linalg.norm(y - x))
        elapsed += time.perf_counter() - t0         # ── clock stops here ──
        if trace:
            tr.append((elapsed, y.copy()))          # bookkeeping is NOT timed
            if stop_fn is not None and stop_fn(y):  # shared-metric early stop, not timed
                break
        if gap <= tol:
            converged = True
            break
    return DRResult(x_sol=y, n_iter=k, converged=converged, solve_time=elapsed, trace=tr)
