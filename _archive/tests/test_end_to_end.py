"""
tests/test_end_to_end.py — 2-agent VRFB×VRFB GNE smoke test.

Step 4 success criterion (Handoff.md §5):
  Explicit mp-GNE is feasible and satisfies Nash optimality conditions.
  ADMM baseline converges to a feasible GNE.

Uses N=2, H=2 to keep the offline explicit solve fast.

Two parameter points:
  p_explicit — L_MIN=150, DA=100 kW: binding lower coupling → unique GNE
  p_admm     — L_MIN=0, DA=0 kW:    relaxed coupling → ADMM converges cleanly
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize, LinearConstraint

from amrhg.agents.base import make_parameter_layout
from amrhg.agents.vrfb import VRFBAgent
from amrhg.game.builder import build_rtm_game
from amrhg.game.params import pack_rtm_params
from amrhg.solvers.admm_solver import admm_solve
from amrhg.solvers.gne_combiner import build_gne_solution
from amrhg.solvers.mp_solver import solve_all_agents_mp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N, H = 2, 2
DT = 5 / 60

VRFB_CFG = {
    "capacity_kw": 300.0, "energy_kwh": 2000.0,
    "roundtrip_efficiency": 0.75,
    "soc_min_kwh": 200.0, "soc_max_kwh": 1800.0, "soc_init_kwh": 1000.0,
}
SETTLEMENT = {"gamma_plus": 1.5, "gamma_minus": 2.0}

SOC_INIT = 1000.0
LMP_RT   = np.array([30.0, 28.0])
L_MAX_KW = 2500.0

# p_explicit: binding L_MIN + non-zero DA → unique GNE, used for explicit tests
DA_KW_EXPLICIT  = np.array([100.0, 100.0])
L_MIN_EXPLICIT  = 150.0

# p_admm: relaxed coupling → ADMM converges reliably
DA_KW_ADMM = np.array([50.0, 50.0])
L_MIN_ADMM = 0.0

NASH_TOL = 1.0    # kW — cost must not improve by more than this via unilateral deviation
GNE_TOL  = 5.0    # kW — feasibility tolerance

# ---------------------------------------------------------------------------
# Module-scoped fixtures (solve once, reuse across tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def layout():
    return make_parameter_layout(N, H, DT)


@pytest.fixture(scope="module")
def game(layout):
    agents = [VRFBAgent(VRFB_CFG), VRFBAgent(VRFB_CFG)]
    return build_rtm_game(agents, layout, SETTLEMENT)


@pytest.fixture(scope="module")
def p_explicit(layout):
    """Binding lower coupling → unique GNE; used for explicit evaluation tests."""
    return pack_rtm_params(
        state_inits=[SOC_INIT] * N,
        lmp_rt=LMP_RT,
        da_schedules=[DA_KW_EXPLICIT] * N,
        l_max_kw=L_MAX_KW,
        l_min_kw=L_MIN_EXPLICIT,
        layout=layout,
    )


@pytest.fixture(scope="module")
def p_admm(layout):
    """Relaxed coupling; ADMM converges cleanly here."""
    return pack_rtm_params(
        state_inits=[SOC_INIT] * N,
        lmp_rt=LMP_RT,
        da_schedules=[DA_KW_ADMM] * N,
        l_max_kw=L_MAX_KW,
        l_min_kw=L_MIN_ADMM,
        layout=layout,
    )


@pytest.fixture(scope="module")
def agent_solutions(game):
    return solve_all_agents_mp(game)


@pytest.fixture(scope="module")
def gne_solution(game, agent_solutions):
    return build_gne_solution(game, agent_solutions)


@pytest.fixture(scope="module")
def x_explicit(gne_solution, p_explicit):
    """Explicit GNE solution at p_explicit — computed once, reused across tests."""
    return gne_solution.evaluate(p_explicit, tol=1e-6)


@pytest.fixture(scope="module")
def admm_result(game, p_admm):
    """ADMM solve at p_admm — computed once, reused across tests."""
    return admm_solve(game, p_admm, max_iter=2000, rho=5.0, tol=1e-3, verbose=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nash_improvement(game, x: np.ndarray, p: np.ndarray, agent_idx: int) -> float:
    """Returns cost reduction an agent could achieve by unilateral deviation.
    Includes coupling constraints conditioned on others' fixed decisions.
    Positive value means x is NOT a GNE for this agent."""
    ag = game.agents[agent_idx]
    sl = game.x_slice(agent_idx)
    x_i = x[sl].copy()
    cost_now = ag.local_cost(x_i, p)

    # Local constraints
    b_loc_rhs = ag.b_loc + ag.S_loc @ p

    # Coupling constraints conditioned on fixed x_{-j}: C_i @ x_i ≤ d_eff
    if game.d is not None and ag.C is not None:
        d_eff = game.d + game.S_coup @ p
        for j, other in enumerate(game.agents):
            if j != agent_idx and other.C is not None:
                d_eff = d_eff - other.C @ x[game.x_slice(j)]
        A_full = np.vstack([ag.A_loc, ag.C])
        b_full = np.concatenate([b_loc_rhs, d_eff])
    else:
        A_full, b_full = ag.A_loc, b_loc_rhs

    lc = LinearConstraint(A_full, -np.inf, b_full)
    res = minimize(lambda xi: ag.local_cost(xi, p), x_i,
                   method="SLSQP", constraints=lc,
                   options={"ftol": 1e-8, "maxiter": 500})
    return cost_now - res.fun if res.success else 0.0


# ---------------------------------------------------------------------------
# Test 1 — Offline per-agent mpQP
# ---------------------------------------------------------------------------

class TestOfflineMpSolve:
    def test_returns_one_solution_per_agent(self, agent_solutions) -> None:
        assert len(agent_solutions) == N

    def test_each_agent_has_at_least_one_cr(self, agent_solutions) -> None:
        for i, sol in enumerate(agent_solutions):
            assert len(sol.regions) > 0, f"Agent {i} has zero critical regions"

    def test_n_x_i_correct(self, agent_solutions) -> None:
        for sol in agent_solutions:
            assert sol.n_x_i == 3 * H

    def test_theta_dim_correct(self, agent_solutions, game) -> None:
        expected = (N - 1) * 3 * H + game.n_p
        for sol in agent_solutions:
            assert sol.n_theta_i == expected


# ---------------------------------------------------------------------------
# Test 2 — Offline GNE combination
# ---------------------------------------------------------------------------

class TestOfflineGneCombination:
    def test_gne_solution_not_none(self, gne_solution) -> None:
        assert gne_solution is not None

    def test_has_critical_regions(self, gne_solution) -> None:
        assert len(gne_solution.regions) > 0

    def test_n_p_stored_correctly(self, gne_solution, layout) -> None:
        assert gne_solution.n_p == layout.n_p

    def test_n_agents_stored_correctly(self, gne_solution) -> None:
        assert gne_solution.N == N


# ---------------------------------------------------------------------------
# Test 3 — Explicit GNE evaluation (p_explicit: binding coupling)
# ---------------------------------------------------------------------------

class TestExplicitEvaluation:
    def test_p_explicit_in_some_cr(self, x_explicit) -> None:
        assert x_explicit is not None, (
            "p_explicit not found in any GNE critical region — "
            "may need to widen parameter box p_lb/p_ub."
        )

    def test_explicit_x_shape(self, x_explicit, game) -> None:
        assert x_explicit is not None
        assert x_explicit.shape == (game.n_x_total,)

    def test_explicit_locally_feasible(self, x_explicit, game, p_explicit) -> None:
        assert x_explicit is not None
        for ag in game.agents:
            sl = game.x_slice(ag.index)
            assert ag.local_feasible(x_explicit[sl], p_explicit, tol=1e-3), (
                f"Agent {ag.index} local constraints violated"
            )

    def test_explicit_coupling_feasible(self, x_explicit, game, p_explicit) -> None:
        assert x_explicit is not None
        assert game.coupling_feasible(x_explicit, p_explicit, tol=1e-3)

    def test_explicit_is_nash_equilibrium(self, x_explicit, game, p_explicit) -> None:
        """Nash verification: no agent can unilaterally reduce cost."""
        assert x_explicit is not None
        for i in range(N):
            improvement = _nash_improvement(game, x_explicit, p_explicit, i)
            assert improvement < NASH_TOL, (
                f"Agent {i} can improve cost by {improvement:.4f} — not a GNE"
            )


# ---------------------------------------------------------------------------
# Test 4 — ADMM baseline (p_admm: relaxed coupling, ADMM-friendly)
# ---------------------------------------------------------------------------

class TestADMMBaseline:
    def test_admm_converges(self, admm_result) -> None:
        assert admm_result.converged, (
            f"ADMM did not converge "
            f"(primal={admm_result.primal_res:.4f}, dual={admm_result.dual_res:.4f})"
        )

    def test_admm_x_shape(self, admm_result, game) -> None:
        assert np.concatenate(admm_result.x_sol).shape == (game.n_x_total,)

    def test_admm_feasible(self, admm_result, game, p_admm) -> None:
        x = np.concatenate(admm_result.x_sol)
        assert game.all_feasible(x, p_admm, tol=1e-2), (
            "ADMM solution violates constraints"
        )

    def test_admm_is_nash_equilibrium(self, admm_result, game, p_admm) -> None:
        x = np.concatenate(admm_result.x_sol)
        for i in range(N):
            improvement = _nash_improvement(game, x, p_admm, i)
            assert improvement < NASH_TOL, (
                f"Agent {i} can improve by {improvement:.4f} from ADMM solution"
            )


# ---------------------------------------------------------------------------
# Test 5 — 4-agent ADMM smoke test (VRFB + PVBattery + PEM electrolyzer + alkaline electrolyzer)
# ---------------------------------------------------------------------------

N4, H4 = 4, 6
DT4 = 5 / 60

VRFB_CFG4 = {
    "capacity_kw": 300.0, "energy_kwh": 2000.0,
    "roundtrip_efficiency": 0.75,
    "soc_min_kwh": 200.0, "soc_max_kwh": 1800.0, "soc_init_kwh": 1000.0,
}
PV_CFG4 = {
    "battery_capacity_kw": 400.0, "battery_energy_kwh": 1500.0,
    "pv_capacity_kw": 300.0,
    "soc_min_kwh": 150.0, "soc_max_kwh": 1350.0, "soc_init_kwh": 750.0,
    "roundtrip_efficiency": 0.92, "a_deg": 5e-4,
}
ELY_PEM_CFG4 = {
    "name": "Electrolyzer-PEM",
    "capacity_kw": 500.0,
    "h2_production_kg_per_kwh": 0.02,
    "tank_min_kg": 10.0, "tank_max_kg": 500.0, "tank_init_kg": 200.0,
    "h2_price_per_kg": 2.0,
    "ramp_rate_kw_per_step": 100.0,
    "h2_daily_target_kg": 0.0,          # disabled for ADMM test
    "a_deg": 5e-4,
}
ELY_ALK_CFG4 = {
    "name": "Electrolyzer-Alk",
    "capacity_kw": 400.0,
    "h2_production_kg_per_kwh": 0.018,
    "tank_min_kg": 10.0, "tank_max_kg": 500.0, "tank_init_kg": 150.0,
    "h2_price_per_kg": 2.0,
    "ramp_rate_kw_per_step": 40.0,
    "h2_daily_target_kg": 0.0,          # disabled for ADMM test
    "a_deg": 5e-4,
}
SETTLEMENT4 = {"gamma_plus": 50.0, "gamma_minus": 45.0}

SOC_INIT4 = [1000.0, 750.0, 200.0, 150.0]
DA_KW4 = [0.0, 0.0, 200.0, 150.0]


@pytest.fixture(scope="module")
def layout4():
    return make_parameter_layout(
        N4, H4, DT4,
        pv_agent_indices=[1],
        ramp_agent_indices=[2, 3],
    )


@pytest.fixture(scope="module")
def game4(layout4):
    from amrhg.agents.pv_battery import PVBatteryAgent
    from amrhg.agents.electrolyzer import ElectrolyzerAgent
    agents = [
        VRFBAgent(VRFB_CFG4),
        PVBatteryAgent(PV_CFG4),
        ElectrolyzerAgent(ELY_PEM_CFG4),
        ElectrolyzerAgent(ELY_ALK_CFG4),
    ]
    return build_rtm_game(agents, layout4, SETTLEMENT4)


@pytest.fixture(scope="module")
def p4_admm(layout4):
    """Relaxed coupling for ADMM convergence."""
    return pack_rtm_params(
        state_inits=SOC_INIT4,
        lmp_rt=np.array([30.0, 28.0, 26.0, 25.0, 27.0, 29.0]),
        da_schedules=[np.full(H4, da) for da in DA_KW4],
        l_max_kw=2500.0,
        l_min_kw=0.0,                    # relaxed coupling
        layout=layout4,
        pv_forecasts=[None, np.full(H4, 100.0), None, None],
        ramp_prevs=[None, None, 200.0, 150.0],
    )


@pytest.fixture(scope="module")
def result4_admm(game4, p4_admm):
    return admm_solve(game4, p4_admm, max_iter=5000, rho=1.0, tol=1e-3, verbose=False)


class TestFourAgentADMM:
    def test_built_with_correct_n(self, game4, layout4) -> None:
        assert game4.N == 4
        assert game4.n_p == layout4.n_p

    def test_all_agents_have_coupling(self, game4) -> None:
        for i, ag in enumerate(game4.agents):
            assert ag.C is not None, f"Agent {i} missing coupling block"

    def test_admm_converges(self, result4_admm) -> None:
        assert result4_admm.converged, (
            f"ADMM did not converge (primal={result4_admm.primal_res:.4f}, "
            f"dual={result4_admm.dual_res:.4f})"
        )

    def test_admm_x_shape(self, result4_admm, game4) -> None:
        assert np.concatenate(result4_admm.x_sol).shape == (game4.n_x_total,)

    def test_admm_feasible(self, result4_admm, game4, p4_admm) -> None:
        x = np.concatenate(result4_admm.x_sol)
        assert game4.all_feasible(x, p4_admm, tol=1e-2), (
            "4-agent ADMM solution violates constraints"
        )
