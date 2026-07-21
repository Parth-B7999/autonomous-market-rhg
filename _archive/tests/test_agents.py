"""
tests/test_agents.py — Unit tests for agent builders.

Step 3 success criterion (from Handoff.md §5):
  pytest tests/test_agents.py::test_vrfb passes
"""

from __future__ import annotations

import numpy as np
import pytest

from amrhg.agents.base import make_parameter_layout, ParameterLayout
from amrhg.agents.vrfb import VRFBAgent
from amrhg.agents.pv_battery import PVBatteryAgent
from amrhg.agents.electrolyzer import ElectrolyzerAgent
from amrhg.solvers.game import Agent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_AGENTS = 4
H = 6
DT_HR = 5 / 60          # 5-min steps

SETTLEMENT_CFG = {"gamma_plus": 1.5, "gamma_minus": 2.0}

VRFB_CFG = {
    "capacity_kw": 300.0,
    "energy_kwh": 2000.0,
    "roundtrip_efficiency": 0.75,
    "soc_min_kwh": 200.0,
    "soc_max_kwh": 1800.0,
    "soc_init_kwh": 1000.0,
}


@pytest.fixture
def layout() -> ParameterLayout:
    return make_parameter_layout(N_AGENTS, H, DT_HR)


@pytest.fixture
def vrfb_agent() -> VRFBAgent:
    return VRFBAgent(VRFB_CFG)


@pytest.fixture
def vrfb_solver_agent(vrfb_agent: VRFBAgent, layout: ParameterLayout) -> Agent:
    return vrfb_agent.build_rtm_agent(
        index=1,
        layout=layout,
        settlement_cfg=SETTLEMENT_CFG,
    )


# ---------------------------------------------------------------------------
# ParameterLayout tests
# ---------------------------------------------------------------------------

class TestParameterLayout:
    def test_n_p_formula(self, layout: ParameterLayout) -> None:
        expected = N_AGENTS * (H + 1) + H + 2
        assert layout.n_p == expected, f"n_p={layout.n_p}, expected={expected}"

    def test_price_slice_bounds(self, layout: ParameterLayout) -> None:
        s = layout.price_slice
        assert s.stop - s.start == H

    def test_da_slices_non_overlapping(self, layout: ParameterLayout) -> None:
        slices = [layout.da_slice(i) for i in range(N_AGENTS)]
        indices = [set(range(s.start, s.stop)) for s in slices]
        for a in range(N_AGENTS):
            for b in range(a + 1, N_AGENTS):
                assert indices[a].isdisjoint(indices[b]), (
                    f"DA slices for agents {a} and {b} overlap"
                )

    def test_l_max_l_min_at_end(self, layout: ParameterLayout) -> None:
        assert layout.l_max_idx == layout.n_p - 2
        assert layout.l_min_idx == layout.n_p - 1

    def test_p_lb_ub_shapes(self, layout: ParameterLayout) -> None:
        lb = layout.p_lb()
        ub = layout.p_ub()
        assert lb.shape == (layout.n_p,)
        assert ub.shape == (layout.n_p,)
        assert np.all(lb <= ub)


# ---------------------------------------------------------------------------
# VRFBAgent construction tests
# ---------------------------------------------------------------------------

class TestVRFBAgent:
    def test_repr(self, vrfb_agent: VRFBAgent) -> None:
        r = repr(vrfb_agent)
        assert "VRFB" in r

    def test_build_returns_agent(self, vrfb_solver_agent: Agent) -> None:
        assert isinstance(vrfb_solver_agent, Agent)

    def test_index_stored(self, vrfb_solver_agent: Agent) -> None:
        assert vrfb_solver_agent.index == 1

    def test_n_x(self, vrfb_solver_agent: Agent) -> None:
        assert vrfb_solver_agent.n_x == 3 * H

    def test_n_p(self, vrfb_solver_agent: Agent, layout: ParameterLayout) -> None:
        assert vrfb_solver_agent.n_p == layout.n_p


# ---------------------------------------------------------------------------
# Q must be positive definite (Hall-Bemporad requirement)
# ---------------------------------------------------------------------------

class TestQPositiveDefinite:
    def test_Q_shape(self, vrfb_solver_agent: Agent) -> None:
        n = vrfb_solver_agent.n_x
        assert vrfb_solver_agent.Q.shape == (n, n)

    def test_Q_symmetric(self, vrfb_solver_agent: Agent) -> None:
        Q = vrfb_solver_agent.Q
        np.testing.assert_allclose(Q, Q.T, atol=1e-12, err_msg="Q is not symmetric")

    def test_Q_positive_definite(self, vrfb_solver_agent: Agent) -> None:
        eigvals = np.linalg.eigvalsh(vrfb_solver_agent.Q)
        assert np.all(eigvals > 0), (
            f"Q has non-positive eigenvalues: min={eigvals.min():.3e}"
        )


# ---------------------------------------------------------------------------
# Constraint shape consistency
# ---------------------------------------------------------------------------

class TestConstraintShapes:
    def test_A_loc_shape(self, vrfb_solver_agent: Agent, layout: ParameterLayout) -> None:
        A = vrfb_solver_agent.A_loc
        assert A.shape[1] == vrfb_solver_agent.n_x, "A_loc column mismatch"

    def test_b_loc_shape(self, vrfb_solver_agent: Agent) -> None:
        A = vrfb_solver_agent.A_loc
        b = vrfb_solver_agent.b_loc
        assert b.shape == (A.shape[0],), "b_loc length mismatch"

    def test_S_loc_shape(self, vrfb_solver_agent: Agent, layout: ParameterLayout) -> None:
        S = vrfb_solver_agent.S_loc
        A = vrfb_solver_agent.A_loc
        assert S.shape == (A.shape[0], layout.n_p), "S_loc shape mismatch"

    def test_F_shape(self, vrfb_solver_agent: Agent, layout: ParameterLayout) -> None:
        F = vrfb_solver_agent.F
        assert F.shape == (vrfb_solver_agent.n_x, layout.n_p), "F shape mismatch"

    def test_C_shape(self, vrfb_solver_agent: Agent) -> None:
        C = vrfb_solver_agent.C
        assert C is not None, "C (coupling block) is None"
        assert C.shape == (2 * H, vrfb_solver_agent.n_x), "C shape mismatch"
        # Power block has +I / -I; slack blocks are zero
        np.testing.assert_array_equal(C[:H, :H],  np.eye(H),  err_msg="C upper != +I")
        np.testing.assert_array_equal(C[H:, :H], -np.eye(H),  err_msg="C lower != -I")
        np.testing.assert_array_equal(C[:, H:],   np.zeros((2*H, 2*H)),
                                      err_msg="C slack columns non-zero")


# ---------------------------------------------------------------------------
# Feasibility + SoC dynamics simulation
# ---------------------------------------------------------------------------

def _build_nominal_p(layout: ParameterLayout, soc_init: float = 1000.0,
                     lmp_rt: float = 30.0, da_kw: float = 0.0,
                     l_max_kw: float = 2500.0, l_min_kw: float = 1000.0) -> np.ndarray:
    """Build a plausible parameter vector for feasibility checks."""
    p = np.zeros(layout.n_p)
    for idx in layout.state_init_indices:
        p[idx] = soc_init
    p[layout.price_slice] = lmp_rt
    for i in range(layout.n_agents):
        p[layout.da_slice(i)] = da_kw
    p[layout.l_max_idx] = l_max_kw
    p[layout.l_min_idx] = l_min_kw
    return p


# ---------------------------------------------------------------------------
# Multi-agent layout (for PV + ramp agents)
# ---------------------------------------------------------------------------

FULL_LAYOUT = make_parameter_layout(
    n_agents=4, H=H, dt_hr=DT_HR,
    pv_agent_indices=[1],          # agent 1 = PVBattery
    ramp_agent_indices=[2, 3],     # agents 2,3 = PEM & alkaline electrolyzers
)


@pytest.fixture
def full_layout() -> ParameterLayout:
    return FULL_LAYOUT


# ---------------------------------------------------------------------------
# PVBattery config and fixtures
# ---------------------------------------------------------------------------

PV_CFG = {
    "battery_capacity_kw": 400.0,
    "battery_energy_kwh": 1500.0,
    "roundtrip_efficiency": 0.92,
    "soc_min_kwh": 150.0,
    "soc_max_kwh": 1350.0,
    "soc_init_kwh": 750.0,
    "pv_capacity_kw": 300.0,
}


@pytest.fixture
def pv_agent() -> PVBatteryAgent:
    return PVBatteryAgent(PV_CFG)


@pytest.fixture
def pv_solver_agent(pv_agent: PVBatteryAgent, full_layout: ParameterLayout) -> Agent:
    return pv_agent.build_rtm_agent(index=1, layout=full_layout, settlement_cfg=SETTLEMENT_CFG)


# ---------------------------------------------------------------------------
# Electrolyzer config and fixtures
# ---------------------------------------------------------------------------

ELY_CFG = {
    "capacity_kw": 500.0,
    "h2_production_kg_per_kwh": 0.02,
    "tank_min_kg": 10.0,
    "tank_max_kg": 500.0,
    "tank_init_kg": 200.0,
    "h2_price_per_kg": 2.0,
    "ramp_rate_kw_per_step": 100.0,
    "h2_daily_target_kg": 1.0,   # achievable in 0.5 h (6×0.02×0.0833×100 = 1.0 kg at 100 kW)
}


@pytest.fixture
def ely_agent() -> ElectrolyzerAgent:
    return ElectrolyzerAgent(ELY_CFG)


@pytest.fixture
def ely_solver_agent(ely_agent: ElectrolyzerAgent, full_layout: ParameterLayout) -> Agent:
    return ely_agent.build_rtm_agent(index=2, layout=full_layout, settlement_cfg=SETTLEMENT_CFG)


# ---------------------------------------------------------------------------
# Second electrolyzer (alkaline) config and fixtures
# ---------------------------------------------------------------------------

ELY_ALK_CFG = {
    "name": "Electrolyzer-Alk",
    "capacity_kw": 400.0,
    "h2_production_kg_per_kwh": 0.018,
    "tank_min_kg": 10.0,
    "tank_max_kg": 300.0,
    "tank_init_kg": 80.0,
    "h2_price_per_kg": 2.0,
    "ramp_rate_kw_per_step": 40.0,
    "h2_daily_target_kg": 0.0,
}


@pytest.fixture
def ely_alk_agent() -> ElectrolyzerAgent:
    return ElectrolyzerAgent(ELY_ALK_CFG)


@pytest.fixture
def ely_alk_solver_agent(ely_alk_agent: ElectrolyzerAgent, full_layout: ParameterLayout) -> Agent:
    return ely_alk_agent.build_rtm_agent(index=3, layout=full_layout, settlement_cfg=SETTLEMENT_CFG)


# ---------------------------------------------------------------------------
# ParameterLayout — extended tests
# ---------------------------------------------------------------------------

class TestFullParameterLayout:
    def test_n_p_with_pv_and_ramp(self, full_layout: ParameterLayout) -> None:
        # n_p = N*(H+1) + H + 2 + H_pv + N_ramp = 4*7 + 6 + 2 + 6 + 2 = 44
        assert full_layout.n_p == 44

    def test_pv_slice_correct(self, full_layout: ParameterLayout) -> None:
        sl = full_layout.pv_slice(1)
        assert sl is not None
        assert sl.stop - sl.start == H

    def test_pv_slice_none_for_non_pv(self, full_layout: ParameterLayout) -> None:
        assert full_layout.pv_slice(0) is None
        assert full_layout.pv_slice(2) is None

    def test_ramp_idx_correct(self, full_layout: ParameterLayout) -> None:
        assert full_layout.ramp_idx(2) is not None
        assert full_layout.ramp_idx(3) is not None

    def test_ramp_idx_none_for_non_ramp(self, full_layout: ParameterLayout) -> None:
        assert full_layout.ramp_idx(0) is None
        assert full_layout.ramp_idx(1) is None

    def test_full_p_lb_ub_shapes(self, full_layout: ParameterLayout) -> None:
        lb = full_layout.p_lb()
        ub = full_layout.p_ub()
        assert lb.shape == (full_layout.n_p,)
        assert ub.shape == (full_layout.n_p,)
        assert np.all(lb <= ub)

    def test_pv_bounds(self, full_layout: ParameterLayout) -> None:
        pv_sl = full_layout.pv_slice(1)
        assert pv_sl is not None
        assert np.all(full_layout.p_lb()[pv_sl] >= 0.0)

    def test_ramp_bounds(self, full_layout: ParameterLayout) -> None:
        ri = full_layout.ramp_idx(2)
        assert ri is not None
        assert full_layout.p_lb()[ri] <= full_layout.p_ub()[ri]


# ---------------------------------------------------------------------------
# PVBatteryAgent tests
# ---------------------------------------------------------------------------

class TestPVBatteryAgent:
    def test_repr(self, pv_agent: PVBatteryAgent) -> None:
        r = repr(pv_agent)
        assert "PV+Battery" in r

    def test_build_returns_agent(self, pv_solver_agent: Agent) -> None:
        assert isinstance(pv_solver_agent, Agent)

    def test_index_stored(self, pv_solver_agent: Agent) -> None:
        assert pv_solver_agent.index == 1

    def test_n_x(self, pv_solver_agent: Agent) -> None:
        assert pv_solver_agent.n_x == 3 * H

    def test_n_p(self, pv_solver_agent: Agent, full_layout: ParameterLayout) -> None:
        assert pv_solver_agent.n_p == full_layout.n_p


class TestPVBatteryQPD:
    def test_Q_positive_definite(self, pv_solver_agent: Agent) -> None:
        eigvals = np.linalg.eigvalsh(pv_solver_agent.Q)
        assert np.all(eigvals > 0)

    def test_Q_symmetric(self, pv_solver_agent: Agent) -> None:
        np.testing.assert_allclose(pv_solver_agent.Q, pv_solver_agent.Q.T, atol=1e-12)


class TestPVBatteryConstraints:
    def test_C_shape(self, pv_solver_agent: Agent) -> None:
        C = pv_solver_agent.C
        assert C is not None
        assert C.shape == (2 * H, 3 * H)

    def test_A_loc_b_loc_S_loc_shapes(self, pv_solver_agent: Agent, full_layout: ParameterLayout) -> None:
        A = pv_solver_agent.A_loc
        assert A.shape[1] == 3 * H
        assert pv_solver_agent.b_loc.shape == (A.shape[0],)
        assert pv_solver_agent.S_loc.shape == (A.shape[0], full_layout.n_p)

    def test_F_has_pv_entries(self, pv_solver_agent: Agent, full_layout: ParameterLayout) -> None:
        pv_sl = full_layout.pv_slice(1)
        assert pv_sl is not None
        # F[:H, pv_slice] should be nonzero (degradation cross-term)
        F_pv_block = pv_solver_agent.F[:H, pv_sl]
        assert not np.allclose(F_pv_block, 0.0), "F should have PV cross-term entries"


class TestPVBatteryFeasibility:
    def test_zero_power_feasible(
        self, pv_solver_agent: Agent, full_layout: ParameterLayout
    ) -> None:
        """Zero net power with mid SoC and zero PV should be feasible."""
        x = np.zeros(pv_solver_agent.n_x)
        p = _build_nominal_p(full_layout, soc_init=750.0, da_kw=0.0)
        assert pv_solver_agent.local_feasible(x, p)

    def test_full_charge_infeasible_near_full_soc(
        self, pv_solver_agent: Agent, full_layout: ParameterLayout
    ) -> None:
        """Charging at full battery rate from near-full SoC violates SoC_max."""
        x = np.zeros(pv_solver_agent.n_x)
        x[:H] = 400.0
        p = _build_nominal_p(full_layout, soc_init=1300.0, da_kw=400.0)
        assert not pv_solver_agent.local_feasible(x, p)

    def test_pv_charges_battery(
        self, pv_solver_agent: Agent, full_layout: ParameterLayout
    ) -> None:
        """PV generation charges battery: p_net=0 with PV=200 → p_batt=+200 (charging)."""
        x = np.zeros(pv_solver_agent.n_x)
        p = _build_nominal_p(full_layout, soc_init=500.0, da_kw=0.0)
        pv_sl = full_layout.pv_slice(1)
        assert pv_sl is not None
        p[pv_sl] = 200.0
        assert pv_solver_agent.local_feasible(x, p), (
            "Zero net power with 200kW PV charging should be feasible at mid SoC"
        )

    def test_c_vector_structure(self, pv_solver_agent: Agent) -> None:
        c = pv_solver_agent.c
        np.testing.assert_array_equal(c[:H], np.zeros(H))
        np.testing.assert_allclose(c[H:2*H], SETTLEMENT_CFG["gamma_plus"] * np.ones(H))
        np.testing.assert_allclose(c[2*H:3*H], SETTLEMENT_CFG["gamma_minus"] * np.ones(H))


# ---------------------------------------------------------------------------
# ElectrolyzerAgent tests
# ---------------------------------------------------------------------------

class TestElectrolyzerAgent:
    def test_repr(self, ely_agent: ElectrolyzerAgent) -> None:
        assert "Electrolyzer" in repr(ely_agent)

    def test_build_returns_agent(self, ely_solver_agent: Agent) -> None:
        assert isinstance(ely_solver_agent, Agent)

    def test_unidirectional_bounds(self, ely_agent: ElectrolyzerAgent) -> None:
        assert ely_agent.p_min_kw == 0.0
        assert ely_agent.p_max_kw == 500.0

    def test_n_x(self, ely_solver_agent: Agent) -> None:
        assert ely_solver_agent.n_x == 3 * H


class TestElectrolyzerQPD:
    def test_Q_positive_definite(self, ely_solver_agent: Agent) -> None:
        eigvals = np.linalg.eigvalsh(ely_solver_agent.Q)
        assert np.all(eigvals > 0)


class TestElectrolyzerConstraints:
    def test_C_shape(self, ely_solver_agent: Agent) -> None:
        assert ely_solver_agent.C is not None
        assert ely_solver_agent.C.shape == (2 * H, 3 * H)

    def test_ramp_constraints_present(self, ely_solver_agent: Agent) -> None:
        b = ely_solver_agent.b_loc
        # Layout: inv_upper(H) + inv_lower(H) + power_bounds(2H) = 4H, then ramp(2H)
        ramp_b = b[4*H:6*H]
        assert np.allclose(ramp_b, 100.0), f"Ramp b values should be 100, got {ramp_b[:4]}"

    def test_power_lower_bound_zero(self, ely_solver_agent: Agent) -> None:
        """Unidirectional: -p_k ≤ 0 means lower power bound at 0."""
        b = ely_solver_agent.b_loc
        # power bounds rows: inv_upper(H) + inv_lower(H) = 2H, then power_bounds (2H rows)
        pw_start = 2 * H
        # Lower half of power bounds: -p_k ≤ 0
        pw_lower_b = b[pw_start + H:pw_start + 2*H]
        assert np.allclose(pw_lower_b, 0.0), f"Lower power bounds should be 0, got {pw_lower_b}"


class TestElectrolyzerFeasibility:
    def test_negative_power_infeasible(
        self, ely_solver_agent: Agent, full_layout: ParameterLayout
    ) -> None:
        """Electrolyzer cannot export (p < 0 is infeasible)."""
        x = np.zeros(ely_solver_agent.n_x)
        x[:H] = -100.0
        p = _build_nominal_p(full_layout, soc_init=200.0, da_kw=-100.0,
                             l_min_kw=0.0)
        assert not ely_solver_agent.local_feasible(x, p), "Negative power should be infeasible"

    def test_ramp_constraint_works(
        self, ely_solver_agent: Agent, full_layout: ParameterLayout
    ) -> None:
        """A step change > ramp_max should violate ramp constraints."""
        x = np.zeros(ely_solver_agent.n_x)
        x[0] = 0.0
        x[1] = 600.0   # jump of 600 > ramp_max=100
        p = _build_nominal_p(full_layout, soc_init=200.0, da_kw=200.0,
                             l_min_kw=0.0)
        assert not ely_solver_agent.local_feasible(x, p), (
            "Step change of 600 kW should violate 100 kW ramp limit"
        )

    def test_ramp_prev_allows_first_step(
        self, ely_solver_agent: Agent, full_layout: ParameterLayout
    ) -> None:
        """p_prev influences ramp for k=0."""
        x = np.zeros(ely_solver_agent.n_x)
        x[:H] = 200.0  # constant 200 kW
        p = _build_nominal_p(full_layout, soc_init=200.0, da_kw=200.0,
                             l_min_kw=0.0)
        ri = full_layout.ramp_idx(2)
        assert ri is not None

        p[ri] = 150.0  # ramp_prev = 150, p_0=200 → step=50 ≤ 100
        assert ely_solver_agent.local_feasible(x, p), "Ramp from 150→200 should be feasible"

        p[ri] = 0.0    # ramp_prev = 0, p_0=200 → step=200 > 100
        assert not ely_solver_agent.local_feasible(x, p), "Ramp from 0→200 should be infeasible"

    def test_c_has_negative_revenue(self, ely_solver_agent: Agent) -> None:
        """H₂ revenue makes c[:H] negative (reduces cost)."""
        c = ely_solver_agent.c
        assert np.all(c[:H] < 0), f"Power block of c should be negative (revenue), got {c[:H]}"


# ---------------------------------------------------------------------------
# Second-electrolyzer (alkaline) tests — confirms a second ElectrolyzerAgent
# instance at index 3 with its own ramp & inventory params builds correctly.
# ---------------------------------------------------------------------------

class TestSecondElectrolyzerAgent:
    def test_repr(self, ely_alk_agent: ElectrolyzerAgent) -> None:
        assert "Electrolyzer-Alk" in repr(ely_alk_agent)

    def test_build_returns_agent(self, ely_alk_solver_agent: Agent) -> None:
        assert isinstance(ely_alk_solver_agent, Agent)

    def test_unidirectional_bounds(self, ely_alk_agent: ElectrolyzerAgent) -> None:
        assert ely_alk_agent.p_min_kw == 0.0
        assert ely_alk_agent.p_max_kw == 400.0

    def test_index_stored(self, ely_alk_solver_agent: Agent) -> None:
        assert ely_alk_solver_agent.index == 3


class TestSecondElectrolyzerQPD:
    def test_Q_positive_definite(self, ely_alk_solver_agent: Agent) -> None:
        eigvals = np.linalg.eigvalsh(ely_alk_solver_agent.Q)
        assert np.all(eigvals > 0)


class TestSecondElectrolyzerFeasibility:
    def test_moderate_power_feasible(
        self, ely_alk_solver_agent: Agent, full_layout: ParameterLayout
    ) -> None:
        """Moderate consumption within the slow ramp should be feasible."""
        x = np.zeros(ely_alk_solver_agent.n_x)
        x[:H] = 200.0
        p = _build_nominal_p(full_layout, soc_init=200.0, da_kw=200.0, l_min_kw=0.0)
        ri = full_layout.ramp_idx(3)
        assert ri is not None
        p[ri] = 180.0  # ramp_prev=180, p_0=200 → step=20 ≤ 40
        assert ely_alk_solver_agent.local_feasible(x, p)

    def test_ramp_violation(
        self, ely_alk_solver_agent: Agent, full_layout: ParameterLayout
    ) -> None:
        """Alkaline ramp (40 kW/step) is much tighter than PEM (100)."""
        x = np.zeros(ely_alk_solver_agent.n_x)
        x[0] = 0.0
        x[1] = 100.0   # step of 100 > ramp_max=40
        p = _build_nominal_p(full_layout, soc_init=200.0, da_kw=50.0, l_min_kw=0.0)
        assert not ely_alk_solver_agent.local_feasible(x, p)


class TestVRFBFeasibility:
    def test_zero_power_feasible(
        self, vrfb_solver_agent: Agent, layout: ParameterLayout
    ) -> None:
        """x=0 with DA=0 and SoC_init mid-range should be locally feasible."""
        x = np.zeros(vrfb_solver_agent.n_x)
        p = _build_nominal_p(layout, soc_init=1000.0, da_kw=0.0)
        assert vrfb_solver_agent.local_feasible(x, p), (
            "Zero power decision should be feasible with DA=0"
        )

    def test_max_charge_infeasible_near_full_soc(
        self, vrfb_solver_agent: Agent, layout: ParameterLayout
    ) -> None:
        """Charging at full rate from near-full SoC should violate SoC_max."""
        x = np.zeros(vrfb_solver_agent.n_x)
        x[:H] = 300.0          # full charging power for all steps
        # DA=300 so imbalance constraint is met
        p = _build_nominal_p(layout, soc_init=1790.0, da_kw=300.0)
        # Near-full SoC + full charging should breach soc_max
        assert not vrfb_solver_agent.local_feasible(x, p), (
            "Full charge from near-full SoC should be infeasible"
        )

    def test_soc_dynamics_within_bounds(self, vrfb_agent: VRFBAgent) -> None:
        """Simulate SoC trajectory for a small charge sequence; check bounds."""
        H_test = H
        dt = DT_HR
        p_seq = np.array([100.0] * H_test)   # gentle charging

        Gamma, alpha_vec = vrfb_agent._condensed_dynamics(H_test, dt)
        soc_0 = 1000.0
        soc_vec = alpha_vec * soc_0 + Gamma @ p_seq

        assert np.all(soc_vec >= vrfb_agent.soc_min_kwh - 1.0), (
            f"SoC dips below min: {soc_vec.min():.1f} kWh"
        )
        assert np.all(soc_vec <= vrfb_agent.soc_max_kwh + 1.0), (
            f"SoC exceeds max: {soc_vec.max():.1f} kWh"
        )

    def test_c_vector_structure(self, vrfb_solver_agent: Agent) -> None:
        """First H entries of c should be 0; next H = γ+; last H = γ-."""
        c = vrfb_solver_agent.c
        np.testing.assert_array_equal(c[:H], np.zeros(H), err_msg="c power block nonzero")
        np.testing.assert_allclose(c[H:2*H],  SETTLEMENT_CFG["gamma_plus"]  * np.ones(H))
        np.testing.assert_allclose(c[2*H:3*H], SETTLEMENT_CFG["gamma_minus"] * np.ones(H))

    def test_F_maps_prices_to_power_block_only(
        self, vrfb_solver_agent: Agent, layout: ParameterLayout
    ) -> None:
        """F should be nonzero only in rows 0..H-1 (power block) × price columns."""
        F = vrfb_solver_agent.F
        price_s = layout.price_slice
        # Power block × price columns should be identity
        np.testing.assert_array_equal(
            F[:H, price_s], np.eye(H), err_msg="F price mapping not identity"
        )
        # Everything else in F should be zero
        F_other = F.copy()
        F_other[:H, price_s] = 0.0
        assert np.all(F_other == 0.0), "F has unexpected nonzero entries"
