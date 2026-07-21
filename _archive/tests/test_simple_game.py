"""
tests/test_simple_game.py — Tests for the minimal GNE game structure.

Covers:
  1. SimpleParameterLayout  — correct indices and n_p
  2. build_simple_vrfb      — shapes, Q PD, constraint feasibility
  3. Single-agent QP solve  — uncoupled VRFB finds optimum
  4. Two-agent ADMM          — coupling satisfied at convergence, tol=1 kW
"""
from __future__ import annotations

import numpy as np
import pytest

from amrhg.game.simple_game import (
    SimpleParameterLayout,
    make_simple_layout,
    make_simple_param,
    build_simple_vrfb,
    build_simple_pv_battery,
    build_simple_electrolyzer,
    build_4agent_simple_game,
    build_simple_game,
    _condensed_dynamics,
)
from amrhg.solvers.admm_solver import admm_solve

# ─────────────────────────────────────────────────────────────────────────────
#  Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

N = 2
H = 6
DT_HR = 5 / 60

# VRFB parameters (scaled-down for fast tests)
VRFB_SOC_MIN = 200.0    # kWh
VRFB_SOC_MAX = 1800.0   # kWh
VRFB_P_MIN   = -300.0   # kW (discharging → exporting)
VRFB_P_MAX   =  300.0   # kW (charging → importing)
VRFB_ETA     = 0.866    # ≈ sqrt(0.75) roundtrip
VRFB_GAMMA   = 1.0      # imbalance penalty $/kW²

L_MAX = 400.0   # kW  — PCC upper limit
L_MIN = -200.0  # kW  — cluster can export up to 200 kW


@pytest.fixture
def layout() -> SimpleParameterLayout:
    return make_simple_layout(N, H, DT_HR)


@pytest.fixture
def vrfb0(layout: SimpleParameterLayout):
    return build_simple_vrfb(
        index=0, layout=layout,
        soc_min_kwh=VRFB_SOC_MIN, soc_max_kwh=VRFB_SOC_MAX,
        p_min_kw=VRFB_P_MIN, p_max_kw=VRFB_P_MAX,
        roundtrip_eta=VRFB_ETA, gamma_imb=VRFB_GAMMA,
    )


@pytest.fixture
def vrfb1(layout: SimpleParameterLayout):
    return build_simple_vrfb(
        index=1, layout=layout,
        soc_min_kwh=VRFB_SOC_MIN, soc_max_kwh=VRFB_SOC_MAX,
        p_min_kw=VRFB_P_MIN, p_max_kw=VRFB_P_MAX,
        roundtrip_eta=VRFB_ETA, gamma_imb=VRFB_GAMMA,
    )


@pytest.fixture
def p_test(layout: SimpleParameterLayout) -> np.ndarray:
    lmp = np.array([50.0, 52.0, 55.0, 60.0, 58.0, 53.0])
    da  = np.array([100.0] * H)
    return make_simple_param(
        layout=layout,
        state_inits=[1000.0, 1000.0],
        lmp_rt=lmp,
        da_schedules=[da, da],
        l_max_kw=L_MAX,
        l_min_kw=L_MIN,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  1. Parameter layout
# ─────────────────────────────────────────────────────────────────────────────

class TestSimpleParameterLayout:
    def test_n_p(self, layout: SimpleParameterLayout):
        # N + H + N*H + 2  =  2 + 6 + 12 + 2 = 22
        assert layout.n_p == N + H + N * H + 2

    def test_price_slice_length(self, layout: SimpleParameterLayout):
        assert layout.price_slice.stop - layout.price_slice.start == H

    def test_da_slices_non_overlapping(self, layout: SimpleParameterLayout):
        s0 = set(range(*layout.da_slice(0).indices(layout.n_p)))
        s1 = set(range(*layout.da_slice(1).indices(layout.n_p)))
        assert s0.isdisjoint(s1)

    def test_l_max_l_min_adjacent(self, layout: SimpleParameterLayout):
        assert layout.l_min_idx == layout.l_max_idx + 1

    def test_make_simple_param_values(self, layout: SimpleParameterLayout, p_test: np.ndarray):
        assert p_test[layout.l_max_idx] == L_MAX
        assert p_test[layout.l_min_idx] == L_MIN
        np.testing.assert_array_equal(
            p_test[layout.price_slice],
            [50.0, 52.0, 55.0, 60.0, 58.0, 53.0],
        )


# ─────────────────────────────────────────────────────────────────────────────
#  2. VRFB agent structure
# ─────────────────────────────────────────────────────────────────────────────

class TestSimpleVRFBStructure:
    def test_n_x_is_H(self, vrfb0):
        assert vrfb0.n_x == H

    def test_Q_is_diagonal_gamma(self, vrfb0):
        np.testing.assert_allclose(vrfb0.Q, VRFB_GAMMA * np.eye(H))

    def test_Q_positive_definite(self, vrfb0):
        eigvals = np.linalg.eigvalsh(vrfb0.Q)
        assert np.all(eigvals > 0)

    def test_c_is_zero(self, vrfb0):
        np.testing.assert_array_equal(vrfb0.c, np.zeros(H))

    def test_C_is_identity(self, vrfb0):
        np.testing.assert_array_equal(vrfb0.C, np.eye(H))

    def test_A_loc_shape(self, vrfb0):
        # 4H rows: SoC_ub(H) + SoC_lb(H) + power_ub(H) + power_lb(H)
        assert vrfb0.A_loc.shape == (4 * H, H)

    def test_b_loc_shape(self, vrfb0):
        assert vrfb0.b_loc.shape == (4 * H,)

    def test_S_loc_shape(self, layout: SimpleParameterLayout, vrfb0):
        assert vrfb0.S_loc.shape == (4 * H, layout.n_p)

    def test_F_shape(self, layout: SimpleParameterLayout, vrfb0):
        assert vrfb0.F.shape == (H, layout.n_p)

    def test_F_price_contribution(self, layout: SimpleParameterLayout, vrfb0):
        # F[k, price_start+k] should be +dt  (higher price → agent reduces import)
        ps = layout.price_start
        for k in range(H):
            assert vrfb0.F[k, ps + k] == pytest.approx(+DT_HR)

    def test_F_da_contribution(self, layout: SimpleParameterLayout, vrfb0):
        # F[k, da_start+k] should be -gamma
        ds = layout.da_starts[0]
        for k in range(H):
            assert vrfb0.F[k, ds + k] == pytest.approx(-VRFB_GAMMA)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Local feasibility checks
# ─────────────────────────────────────────────────────────────────────────────

class TestSimpleVRFBFeasibility:
    def test_zero_power_feasible(self, vrfb0, layout, p_test):
        x = np.zeros(H)
        assert vrfb0.local_feasible(x, p_test), "Zero power should be feasible at mid SoC"

    def test_full_charge_infeasible_near_full(self, vrfb0, layout: SimpleParameterLayout):
        # SoC_0 = 1780, full charge (300 kW) → SoC violates max
        p = make_simple_param(
            layout, state_inits=[1780.0, 1000.0],
            lmp_rt=np.ones(H) * 50.0, da_schedules=[np.ones(H) * 300.0, np.zeros(H)],
            l_max_kw=L_MAX, l_min_kw=L_MIN,
        )
        x = np.full(H, 300.0)
        assert not vrfb0.local_feasible(x, p), "Full charge near-full SoC should be infeasible"

    def test_power_exceeds_max_infeasible(self, vrfb0, p_test):
        x = np.full(H, 500.0)   # 500 > p_max=300
        assert not vrfb0.local_feasible(x, p_test)

    def test_soc_dynamics_consistent(self):
        """Verify condensed dynamics match recursive formula."""
        alpha, eta, dt = 1.0, VRFB_ETA, DT_HR
        Gamma, alpha_vec = _condensed_dynamics(H, dt, alpha, eta)
        p_seq = np.array([100.0, -50.0, 80.0, 0.0, -100.0, 60.0])
        soc_0 = 1000.0
        soc_condensed = alpha_vec * soc_0 + Gamma @ p_seq

        # Verify against recursive simulation
        soc = soc_0
        soc_recursive = []
        for k in range(H):
            soc = alpha * soc + eta * dt * p_seq[k]
            soc_recursive.append(soc)

        np.testing.assert_allclose(soc_condensed, soc_recursive, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
#  4. Single-agent uncoupled solve
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleAgentSolve:
    """Single VRFB, no coupling — ADMM reduces to a single QP solve."""

    def test_converges_quickly(self, vrfb0, layout: SimpleParameterLayout, p_test: np.ndarray):
        from amrhg.solvers.game import GNEGame
        game_1 = GNEGame(
            agents=[vrfb0],
            d=np.zeros(H),
            S_coup=np.zeros((H, layout.n_p)),
            d_lb=np.zeros(H),
            S_coup_lb=np.zeros((H, layout.n_p)),
            p_lb=layout.p_lb(),
            p_ub=layout.p_ub(),
        )
        res = admm_solve(game_1, p_test, rho=1.0, max_iter=200, tol=1.0)
        assert res.converged, f"Single VRFB did not converge: r={res.primal_res:.3e}"

    def test_solution_locally_feasible(self, vrfb0, layout: SimpleParameterLayout, p_test: np.ndarray):
        from amrhg.solvers.game import GNEGame
        game_1 = GNEGame(
            agents=[vrfb0],
            d=np.zeros(H),
            S_coup=np.zeros((H, layout.n_p)),
            d_lb=np.zeros(H),
            S_coup_lb=np.zeros((H, layout.n_p)),
            p_lb=layout.p_lb(),
            p_ub=layout.p_ub(),
        )
        res = admm_solve(game_1, p_test, rho=1.0, max_iter=500, tol=0.1)
        p_sol = res.x_sol[0]
        assert vrfb0.local_feasible(p_sol, p_test, tol=1.0), (
            f"Solution violates local constraints: {p_sol}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  5. Two-agent ADMM — coupling must be satisfied at convergence
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoAgentADMM:

    @pytest.fixture
    def game(self, vrfb0, vrfb1, layout):
        return build_simple_game([vrfb0, vrfb1], layout)

    def test_converges(self, game, p_test):
        res = admm_solve(game, p_test, rho=2.0, max_iter=1000, tol=1.0)
        assert res.converged, (
            f"2-agent ADMM did not converge in 1000 iter. "
            f"primal={res.primal_res:.2e}, dual={res.dual_res:.2e}"
        )

    def test_coupling_upper_satisfied(self, game, p_test):
        res = admm_solve(game, p_test, rho=2.0, max_iter=1000, tol=1.0)
        p_sum = res.x_sol[0] + res.x_sol[1]
        violations = p_sum - L_MAX
        assert np.all(violations <= 1.0), (
            f"PCC upper violated by up to {violations.max():.1f} kW"
        )

    def test_coupling_lower_satisfied(self, game, p_test):
        res = admm_solve(game, p_test, rho=2.0, max_iter=1000, tol=1.0)
        p_sum = res.x_sol[0] + res.x_sol[1]
        violations = L_MIN - p_sum
        assert np.all(violations <= 1.0), (
            f"PCC lower violated by up to {violations.max():.1f} kW"
        )

    def test_both_locally_feasible(self, game, vrfb0, vrfb1, p_test):
        res = admm_solve(game, p_test, rho=2.0, max_iter=1000, tol=1.0)
        assert vrfb0.local_feasible(res.x_sol[0], p_test, tol=1.0)
        assert vrfb1.local_feasible(res.x_sol[1], p_test, tol=1.0)

    def test_primal_residual_smaller_than_tol(self, game, p_test):
        res = admm_solve(game, p_test, rho=2.0, max_iter=1000, tol=1.0)
        assert res.primal_res < 1.0, f"primal_res={res.primal_res:.3e} ≥ tol=1.0"

    def test_z_sol_dimension(self, game, p_test):
        # With bilateral coupling, z is H-dimensional per agent (not 2H)
        res = admm_solve(game, p_test, rho=2.0, max_iter=500, tol=5.0)
        for zi in res.z_sol:
            assert zi.shape == (H,), f"z_i should be ({H},), got {zi.shape}"

    def test_unconstrained_optimum_direction(self, layout: SimpleParameterLayout):
        """
        When λ is HIGH, the battery should export more (p < p_DA).
        Unconstrained optimum: p* = p_DA - dt·λ/γ
        """
        da = np.ones(H) * 100.0
        lmp_high = np.ones(H) * 200.0   # very high price
        p_high = make_simple_param(
            layout, state_inits=[1000.0, 1000.0],
            lmp_rt=lmp_high, da_schedules=[da, da],
            l_max_kw=2000.0, l_min_kw=-2000.0,   # wide coupling — not binding
        )
        from amrhg.solvers.game import GNEGame
        vrfb_wide = build_simple_vrfb(
            index=0, layout=layout,
            soc_min_kwh=VRFB_SOC_MIN, soc_max_kwh=VRFB_SOC_MAX,
            p_min_kw=-1e6, p_max_kw=1e6,   # no power limit — unconstrained
            roundtrip_eta=VRFB_ETA, gamma_imb=VRFB_GAMMA,
        )
        game_1 = GNEGame(
            agents=[vrfb_wide],
            d=np.zeros(H), S_coup=np.zeros((H, layout.n_p)),
            d_lb=np.zeros(H), S_coup_lb=np.zeros((H, layout.n_p)),
            p_lb=layout.p_lb(), p_ub=layout.p_ub(),
        )
        res = admm_solve(game_1, p_high, rho=1.0, max_iter=500, tol=0.01)
        p_sol = res.x_sol[0]
        # High price → agent should want to export (p* < p_DA = 100)
        assert np.all(p_sol < da), (
            f"High-price: expected p < p_DA=100, got {p_sol.round(1)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  6. PV+Battery agent
# ─────────────────────────────────────────────────────────────────────────────

PV_SOC_MIN   = 400.0
PV_SOC_MAX   = 3800.0
PV_BATT_CAP  = 1000.0
PV_ETA       = 0.92

@pytest.fixture
def pv_layout():
    return make_simple_layout(N, H, DT_HR, pv_agent_idx=1)


@pytest.fixture
def pv_agent(pv_layout):
    return build_simple_pv_battery(
        index=1, layout=pv_layout,
        soc_min_kwh=PV_SOC_MIN, soc_max_kwh=PV_SOC_MAX,
        battery_capacity_kw=PV_BATT_CAP,
        roundtrip_eta=PV_ETA, gamma_imb=VRFB_GAMMA,
    )


class TestPVBatteryStructure:
    def test_n_x_is_H(self, pv_agent):
        assert pv_agent.n_x == H

    def test_Q_diagonal_gamma(self, pv_agent):
        np.testing.assert_allclose(pv_agent.Q, VRFB_GAMMA * np.eye(H))

    def test_Q_positive_definite(self, pv_agent):
        assert np.all(np.linalg.eigvalsh(pv_agent.Q) > 0)

    def test_c_is_zero(self, pv_agent):
        np.testing.assert_array_equal(pv_agent.c, np.zeros(H))

    def test_A_loc_shape(self, pv_agent):
        # 4H rows: SoC_ub(H) + SoC_lb(H) + batt_pw_ub(H) + batt_pw_lb(H)
        assert pv_agent.A_loc.shape == (4 * H, H)

    def test_C_is_identity(self, pv_agent):
        np.testing.assert_array_equal(pv_agent.C, np.eye(H))

    def test_pv_in_S_loc(self, pv_layout, pv_agent):
        pv_sl = pv_layout.pv_slice
        assert pv_sl is not None
        # PV appears in SoC constraint rows (first 2H) and power bound rows (last 2H)
        assert not np.all(pv_agent.S_loc[:2*H, pv_sl] == 0), \
            "PV forecast should affect SoC constraint RHS"
        assert not np.all(pv_agent.S_loc[2*H:, pv_sl] == 0), \
            "PV forecast should affect battery power bound RHS"

    def test_zero_pv_matches_vrfb_constraints(self, pv_layout):
        """With zero PV, PV+Battery constraints reduce to plain battery form."""
        pv = build_simple_pv_battery(
            index=1, layout=pv_layout,
            soc_min_kwh=PV_SOC_MIN, soc_max_kwh=PV_SOC_MAX,
            battery_capacity_kw=PV_BATT_CAP, roundtrip_eta=PV_ETA,
        )
        lmp = np.ones(H) * 50.0
        p = make_simple_param(
            pv_layout, state_inits=[1000.0, 2000.0],
            lmp_rt=lmp, da_schedules=[np.zeros(H), np.zeros(H)],
            l_max_kw=2000.0, l_min_kw=-200.0,
            pv_forecast=np.zeros(H),   # zero PV
        )
        x_zero = np.zeros(H)
        assert pv.local_feasible(x_zero, p), "Zero power with zero PV should be feasible"

    def test_high_pv_tightens_upper_bound(self, pv_layout, pv_agent):
        """High PV charges battery — less room to import from grid."""
        lmp = np.ones(H) * 50.0
        p_nopv = make_simple_param(
            pv_layout, state_inits=[1000.0, 2000.0],
            lmp_rt=lmp, da_schedules=[np.zeros(H), np.zeros(H)],
            l_max_kw=2000.0, l_min_kw=-200.0,
            pv_forecast=np.zeros(H),
        )
        p_highpv = make_simple_param(
            pv_layout, state_inits=[1000.0, 2000.0],
            lmp_rt=lmp, da_schedules=[np.zeros(H), np.zeros(H)],
            l_max_kw=2000.0, l_min_kw=-200.0,
            pv_forecast=np.ones(H) * 800.0,  # 800 kW PV
        )
        # Importing 500 kW should be feasible with no PV but not with 800 kW PV
        # (battery_cap=1000, SoC_0=2000, PV already charging → battery fills up)
        x_import = np.full(H, 500.0)
        # No PV: battery sees 500 kW charging → SoC rises from 2000
        # High PV: battery sees 500+800=1300 kW > cap=1000 → battery power violated
        assert not pv_agent.local_feasible(x_import, p_highpv), \
            "Importing 500 kW with 800 kW PV should violate battery power cap of 1000 kW"


# ─────────────────────────────────────────────────────────────────────────────
#  7. Electrolyzer agent
# ─────────────────────────────────────────────────────────────────────────────

ELY_TANK_MIN  = 0.0
ELY_TANK_MAX  = 800.0
ELY_P_MAX     = 2000.0
ELY_ETA_PROD  = 0.02     # kg H2 / kWh
ELY_H2_PRICE  = 4.0      # $/kg
ELY_TANK_INIT = 200.0

@pytest.fixture
def ely_layout():
    return make_simple_layout(4, H, DT_HR)


@pytest.fixture
def ely_agent(ely_layout):
    return build_simple_electrolyzer(
        index=2, layout=ely_layout,
        tank_min_kg=ELY_TANK_MIN, tank_max_kg=ELY_TANK_MAX,
        p_max_kw=ELY_P_MAX,
        eta_prod_kg_per_kwh=ELY_ETA_PROD,
        h2_price_per_kg=ELY_H2_PRICE,
        gamma_imb=VRFB_GAMMA,
    )


class TestElectrolyzerStructure:
    def test_n_x_is_H(self, ely_agent):
        assert ely_agent.n_x == H

    def test_Q_diagonal_gamma(self, ely_agent):
        np.testing.assert_allclose(ely_agent.Q, VRFB_GAMMA * np.eye(H))

    def test_c_is_h2_revenue(self, ely_agent):
        expected = -(ELY_H2_PRICE * ELY_ETA_PROD * DT_HR)
        np.testing.assert_allclose(ely_agent.c, expected * np.ones(H))

    def test_C_is_identity(self, ely_agent):
        np.testing.assert_array_equal(ely_agent.C, np.eye(H))

    def test_A_loc_shape(self, ely_agent):
        # 4H rows: inv_ub(H) + inv_lb(H) + pw_ub(H) + pw_lb(H)
        assert ely_agent.A_loc.shape == (4 * H, H)

    def test_unidirectional_lower_bound_zero(self, ely_agent):
        """Electrolyzer can't export: lower power bound rows give b=0."""
        # Power bound rows: A_pw = [I; -I], b_pw = [p_max; 0]
        # Lower half of b: b[3H:4H] should be zeros (p ≥ 0)
        b = ely_agent.b_loc
        np.testing.assert_array_equal(b[3*H:4*H], np.zeros(H))

    def test_negative_power_infeasible(self, ely_layout, ely_agent):
        p = make_simple_param(
            ely_layout, state_inits=[0.0, 0.0, ELY_TANK_INIT, 0.0],
            lmp_rt=np.ones(H) * 50.0, da_schedules=[np.zeros(H)] * 4,
            l_max_kw=5000.0, l_min_kw=-1000.0,
        )
        x_neg = np.full(H, -100.0)
        assert not ely_agent.local_feasible(x_neg, p), \
            "Negative power (export) should be infeasible for electrolyzer"

    def test_zero_power_feasible(self, ely_layout, ely_agent):
        p = make_simple_param(
            ely_layout, state_inits=[0.0, 0.0, ELY_TANK_INIT, 0.0],
            lmp_rt=np.ones(H) * 50.0, da_schedules=[np.zeros(H)] * 4,
            l_max_kw=5000.0, l_min_kw=-1000.0,
        )
        assert ely_agent.local_feasible(np.zeros(H), p)

    def test_h2_revenue_encourages_consumption(self, ely_layout):
        """H2 revenue in c makes high-consumption cheaper — c < 0."""
        ely = build_simple_electrolyzer(
            index=2, layout=ely_layout,
            tank_min_kg=0.0, tank_max_kg=1e6,
            p_max_kw=ELY_P_MAX,
            eta_prod_kg_per_kwh=ELY_ETA_PROD,
            h2_price_per_kg=ELY_H2_PRICE,
        )
        assert np.all(ely.c < 0), f"H2 revenue should make c negative, got {ely.c}"


# ─────────────────────────────────────────────────────────────────────────────
#  8. 4-agent ADMM (full game from base.yaml params)
# ─────────────────────────────────────────────────────────────────────────────

VRFB_CFG = {
    "capacity_kw": 1000.0, "roundtrip_efficiency": 0.75,
    "soc_min_kwh": 700.0, "soc_max_kwh": 6300.0,
}
PV_CFG = {
    "battery_capacity_kw": 1000.0, "roundtrip_efficiency": 0.92,
    "soc_min_kwh": 400.0, "soc_max_kwh": 3800.0,
}
PEM_CFG = {
    "capacity_kw": 2000.0, "h2_production_kg_per_kwh": 0.02,
    "tank_min_kg": 0.0, "tank_max_kg": 800.0, "h2_price_per_kg": 4.0,
}
ALK_CFG = {
    "capacity_kw": 1500.0, "h2_production_kg_per_kwh": 0.018,
    "tank_min_kg": 0.0, "tank_max_kg": 1200.0, "h2_price_per_kg": 4.0,
}

L_MAX_4 = 1500.0   # kW  (PCC import cap — binding with 2 electrolyzers)
L_MIN_4 = -200.0   # kW  (can export up to 200 kW)


@pytest.fixture
def four_agent_setup():
    agents, layout, game = build_4agent_simple_game(
        VRFB_CFG, PV_CFG, PEM_CFG, ALK_CFG,
    )
    pv_fcast = np.array([150.0, 140.0, 120.0, 90.0, 60.0, 30.0])
    lmp = np.array([45.0, 48.0, 52.0, 55.0, 60.0, 65.0])
    p = make_simple_param(
        layout,
        state_inits=[3500.0, 2000.0, 200.0, 320.0],
        lmp_rt=lmp,
        da_schedules=[
            np.full(H, 100.0),    # VRFB DA
            np.full(H, -50.0),    # PV+Bat DA (slight export)
            np.full(H, 800.0),    # PEM DA
            np.full(H, 600.0),    # Alk DA
        ],
        l_max_kw=L_MAX_4,
        l_min_kw=L_MIN_4,
        pv_forecast=pv_fcast,
    )
    return agents, layout, game, p


class TestFourAgentADMM:
    def test_n_p(self, four_agent_setup):
        _, layout, _, _ = four_agent_setup
        # 4 + 6 + 4*6 + 2 + 6 = 42
        assert layout.n_p == 4 + H + 4 * H + 2 + H

    def test_all_agent_n_x(self, four_agent_setup):
        agents, _, _, _ = four_agent_setup
        for ag in agents:
            assert ag.n_x == H

    def test_all_Q_positive_definite(self, four_agent_setup):
        agents, _, _, _ = four_agent_setup
        for ag in agents:
            assert np.all(np.linalg.eigvalsh(ag.Q) > 0)

    def test_admm_converges(self, four_agent_setup):
        _, _, game, p = four_agent_setup
        res = admm_solve(game, p, rho=5.0, max_iter=2000, tol=1.0)
        assert res.converged, (
            f"4-agent ADMM did not converge. "
            f"primal={res.primal_res:.2e}, dual={res.dual_res:.2e}, iter={res.n_iter}"
        )

    def test_coupling_upper_satisfied(self, four_agent_setup):
        _, _, game, p = four_agent_setup
        res = admm_solve(game, p, rho=5.0, max_iter=2000, tol=1.0)
        p_sum = sum(res.x_sol[i] for i in range(4))
        assert np.all(p_sum <= L_MAX_4 + 1.0), \
            f"PCC upper violated: max excess = {(p_sum - L_MAX_4).max():.1f} kW"

    def test_coupling_lower_satisfied(self, four_agent_setup):
        _, _, game, p = four_agent_setup
        res = admm_solve(game, p, rho=5.0, max_iter=2000, tol=1.0)
        p_sum = sum(res.x_sol[i] for i in range(4))
        assert np.all(p_sum >= L_MIN_4 - 1.0), \
            f"PCC lower violated: max deficit = {(L_MIN_4 - p_sum).max():.1f} kW"

    def test_electrolyzer_non_negative(self, four_agent_setup):
        """Electrolyzers (agents 2 and 3) must only consume, never export."""
        _, _, game, p = four_agent_setup
        res = admm_solve(game, p, rho=5.0, max_iter=2000, tol=1.0)
        for i in [2, 3]:
            assert np.all(res.x_sol[i] >= -1.0), \
                f"Electrolyzer {i} exported: {res.x_sol[i].min():.1f} kW"

    def test_dual_z_dimension(self, four_agent_setup):
        """z and λ should be H-dimensional per agent (bilateral coupling)."""
        _, _, game, p = four_agent_setup
        res = admm_solve(game, p, rho=5.0, max_iter=500, tol=10.0)
        for zi in res.z_sol:
            assert zi.shape == (H,)
        for li in res.lambda_sol:
            assert li.shape == (H,)