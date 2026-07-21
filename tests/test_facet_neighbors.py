import numpy as np

from amrhg.solvers.cr_store import AgentCR, AgentSolution
from amrhg.solvers.facet_gne import (
    _chebyshev_center,
    _facet_lp_test_fast,
    _find_shared_hyperplane,
    find_facet_neighbors_by_crossing,
)


def _cr(E, f, index):
    return AgentCR(
        E=np.asarray(E, dtype=float),
        f=np.asarray(f, dtype=float),
        A=np.zeros((1, 2)),
        b=np.zeros(1),
        index=index,
    )


def test_skew_facet_uses_relative_interior_and_builds_reciprocal_edge():
    # A long, thin box split near its upper-right corner by x+y=10.1.
    # Projecting the whole-region Chebyshev center orthogonally onto that
    # hyperplane lands above y=1, outside the source polytope.
    box_E = np.array([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ])
    box_f = np.array([10.0, 0.0, 1.0, 0.0])
    split = np.array([1.0, 1.0])

    cr0 = _cr(np.vstack([box_E, split]), np.r_[box_f, 10.1], 0)
    cr1 = _cr(np.vstack([box_E, -split]), np.r_[box_f, -10.1], 1)
    sol = AgentSolution(agent_index=0, n_x_i=1, n_theta_i=2,
                        regions=[cr0, cr1])

    centre = _chebyshev_center(cr0.E, cr0.f)
    normal = split / np.linalg.norm(split)
    projected = centre + normal * ((10.1 - split @ centre) / np.linalg.norm(split))
    assert np.max(cr0.E @ projected - cr0.f) > 1e-3

    find_facet_neighbors_by_crossing([sol], verbose=False)

    assert cr0.facet_neighbors == [1]
    assert cr1.facet_neighbors == [0]


def test_every_stored_edge_is_symmetric_and_lp_certified():
    # Three vertical strips. Only consecutive strips share a one-dimensional
    # facet; CR0 and CR2 are separated.
    strips = []
    for i, (lo, hi) in enumerate(((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))):
        E = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])
        f = np.array([hi, -lo, 1.0, 0.0])
        strips.append(_cr(E, f, i))
    sol = AgentSolution(agent_index=0, n_x_i=1, n_theta_i=2,
                        regions=strips)

    find_facet_neighbors_by_crossing([sol], verbose=False)

    assert strips[0].facet_neighbors == [1]
    assert strips[1].facet_neighbors == [0, 2]
    assert strips[2].facet_neighbors == [1]

    for v, cr_v in enumerate(strips):
        for w in cr_v.facet_neighbors:
            assert v in strips[w].facet_neighbors
            j = _find_shared_hyperplane(
                cr_v.E, cr_v.f, strips[w].E, strips[w].f)
            assert j is not None
            assert _facet_lp_test_fast(
                cr_v.E, cr_v.f, strips[w].E, strips[w].f, j)
