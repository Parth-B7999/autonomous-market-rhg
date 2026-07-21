from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

from amrhg.solvers.cr_store import AgentCR, AgentSolution, agent_solution_from_ppopt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simple_game"))
import rhg_online as O


def test_ppopt_multiplier_metadata_is_retained():
    pcr = SimpleNamespace(
        E=np.array([[1.0, 0.0]]), f=np.array([[2.0]]),
        A=np.eye(2), b=np.zeros((2, 1)),
        C=np.array([[3.0, 4.0]]), d=np.array([[5.0]]), active_set=[7],
    )
    sol = SimpleNamespace(critical_regions=[pcr])
    cr = agent_solution_from_ppopt(sol, agent_index=0, n_x_i=2, n_theta_i=2,
                                   n_constraints=10).regions[0]
    assert np.array_equal(cr.lambda_A, pcr.C)
    assert np.array_equal(cr.lambda_b, pcr.d.ravel())
    assert cr.active_set == [7]
    assert cr.n_constraints == 10


def test_hall_selector_uses_stored_coupling_multiplier_maps():
    """A singular two-agent best-response system is made unique by Hall Eq. (16)."""
    h, n_p = O.H, 1

    class Game:
        N = 2
        n_x_total = 2 * h
        n_coupling = h
        n_p = 1
        agents = [SimpleNamespace(index=0, n_x=h), SimpleNamespace(index=1, n_x=h)]

        @staticmethod
        def x_slice(i):
            return slice(i * h, (i + 1) * h)

    # x_0 = -x_1 and x_1 = -x_0: M_x has a four-dimensional nullspace.
    # lambda_0 = x_1 and lambda_1 = x_0; equal multipliers select x_0=x_1=0.
    def cr(index):
        A = np.hstack([-np.eye(h), np.zeros((h, n_p))])
        L = np.hstack([np.eye(h), np.zeros((h, n_p))])
        return AgentCR(
            E=np.zeros((1, h + n_p)), f=np.ones(1), A=A, b=np.zeros(h),
            lambda_A=L, lambda_b=np.zeros(h), active_set=list(range(h)),
            n_constraints=h, index=index,
        )

    sols = [AgentSolution(0, h, h + n_p, [cr(0)]),
            AgentSolution(1, h, h + n_p, [cr(0)])]
    x = O._solve_combo_hall_vgne((0, 0), np.zeros(n_p), sols, Game())
    assert x is not None
    assert np.allclose(x, 0.0, atol=1e-10)
