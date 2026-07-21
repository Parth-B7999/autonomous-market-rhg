"""Tests for simulation driver and logger."""

import numpy as np
import pytest

from amrhg.simulation.logger import SimulationResult
from amrhg.simulation.driver import SimulationState, _step_dynamics

# Import agent classes for step_dynamics test
from amrhg.agents.vrfb import VRFBAgent
from amrhg.agents.pv_battery import PVBatteryAgent
from amrhg.agents.electrolyzer import ElectrolyzerAgent


# ---- SimulationResult ----

class TestSimulationResult:
    def test_construction_minimal(self):
        sr = SimulationResult(
            method="test",
            p_net=np.zeros((2, 10)),
            soc=np.zeros((2, 11)),
            rtm_prices=np.zeros(10),
            dam_prices=np.zeros(5),
            da_schedules=np.zeros((2, 5)),
        )
        assert sr.method == "test"
        assert sr.p_net.shape == (2, 10)
        assert sr.soc.shape == (2, 11)
        assert sr.total_cost is None
        assert sr.coupling_violations is None

    def test_construction_full(self):
        sr = SimulationResult(
            method="rhg",
            p_net=np.ones((2, 10)),
            soc=np.ones((2, 11)),
            rtm_prices=np.ones(10),
            dam_prices=np.ones(5),
            da_schedules=np.ones((2, 5)),
            combo_history=[0, 1, 2],
            total_cost=np.array([100.0, 200.0]),
            coupling_violations=np.zeros(10),
            n_steps=10,
            agent_names=["A", "B"],
        )
        assert len(sr.combo_history) == 3
        assert sr.total_cost[1] == 200.0
        assert sr.n_steps == 10

    def test_save_load_roundtrip(self, tmp_path):
        sr = SimulationResult(
            method="rhg",
            p_net=np.array([[1.0, 2.0], [3.0, 4.0]]),
            soc=np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 4.0]]),
            rtm_prices=np.array([50.0, 60.0]),
            dam_prices=np.array([45.0]),
            da_schedules=np.array([[40.0], [45.0]]),
            total_cost=np.array([10.0, 20.0]),
            coupling_violations=np.array([0.0, 5.0]),
            n_steps=2,
            agent_names=["A", "B"],
        )
        path = tmp_path / "test.pkl"
        sr.save(str(path))
        loaded = SimulationResult.load(str(path))
        assert loaded.method == "rhg"
        assert np.array_equal(loaded.p_net, sr.p_net)
        assert np.array_equal(loaded.soc, sr.soc)
        assert np.array_equal(loaded.rtm_prices, sr.rtm_prices)
        assert np.array_equal(loaded.total_cost, sr.total_cost)
        assert loaded.n_steps == 2
        assert loaded.agent_names == ["A", "B"]


# ---- SimulationState ----

class TestSimulationState:
    def test_construction(self):
        state = SimulationState(soc=[100.0, 200.0], prev_power=[0.0, 0.0])
        assert state.soc == [100.0, 200.0]
        assert state.prev_power == [0.0, 0.0]
        assert state.step == 0

    def test_copy_is_deep(self):
        state = SimulationState(soc=[100.0, 200.0], prev_power=[10.0, 20.0], step=5)
        copy = state.copy()
        copy.soc[0] = 999.0
        copy.prev_power[0] = 888.0
        assert state.soc[0] == 100.0
        assert state.prev_power[0] == 10.0
        assert copy.step == 5


# ---- _step_dynamics ----

VRFB_CFG = {
    "capacity_kw": 300.0,
    "soc_min_kwh": 100.0, "soc_max_kwh": 1900.0, "soc_init_kwh": 1000.0,
    "roundtrip_efficiency": 0.75, "a_deg": 5e-4,
}
PV_CFG = {
    "battery_capacity_kw": 400.0, "battery_energy_kwh": 1500.0,
    "pv_capacity_kw": 300.0,
    "soc_min_kwh": 150.0, "soc_max_kwh": 1350.0, "soc_init_kwh": 750.0,
    "roundtrip_efficiency": 0.92, "a_deg": 5e-4,
}
ELY_PEM_CFG = {
    "name": "Electrolyzer-PEM",
    "capacity_kw": 500.0,
    "h2_production_kg_per_kwh": 0.02,
    "tank_min_kg": 5.0, "tank_max_kg": 200.0, "tank_init_kg": 50.0,
    "h2_price_per_kg": 3.0, "ramp_rate_kw_per_min": 25.0, "a_deg": 5e-4,
    "h2_daily_target_kg": 0.0,
}
ELY_ALK_CFG = {
    "name": "Electrolyzer-Alk",
    "capacity_kw": 400.0,
    "h2_production_kg_per_kwh": 0.018,
    "tank_min_kg": 5.0, "tank_max_kg": 300.0, "tank_init_kg": 80.0,
    "h2_price_per_kg": 3.0, "ramp_rate_kw_per_min": 8.0, "a_deg": 5e-4,
    "h2_daily_target_kg": 0.0,
}


class TestStepDynamics:
    @pytest.fixture
    def agents(self):
        return [
            VRFBAgent(VRFB_CFG),
            PVBatteryAgent(PV_CFG),
            ElectrolyzerAgent(ELY_PEM_CFG),
            ElectrolyzerAgent(ELY_ALK_CFG),
        ]

    def test_steps_all_agents(self, agents):
        state = SimulationState(
            soc=[1000.0, 750.0, 50.0, 80.0],
            prev_power=[0.0, 0.0, 0.0, 0.0],
        )
        p_rt = np.array([150.0, -200.0, 300.0, 200.0])
        _step_dynamics(agents, state, p_rt, dt_hr=5.0 / 60.0, pv_actual=0.0)
        assert state.step == 1
        assert state.prev_power == [150.0, -200.0, 300.0, 200.0]
        # All SoCs should have changed from initial
        assert state.soc[0] != 1000.0  # VRFB charging
        assert state.soc[1] != 750.0   # PV battery discharging
        assert state.soc[2] != 50.0    # PEM electrolyzer producing
        assert state.soc[3] != 80.0    # Alkaline electrolyzer producing

    def test_soc_clamped_to_bounds(self, agents):
        state = SimulationState(
            soc=[1900.0, 750.0, 50.0, 80.0],  # VRFB near max
            prev_power=[0.0, 0.0, 0.0, 0.0],
        )
        p_rt = np.array([300.0, 0.0, 0.0, 0.0])  # Charge VRFB: should clamp
        _step_dynamics(agents, state, p_rt, dt_hr=5.0 / 60.0, pv_actual=0.0)
        assert state.soc[0] <= 1900.0  # Clamped to SoC max

    def test_pv_actual_affects_battery(self, agents):
        state = SimulationState(
            soc=[1000.0, 750.0, 50.0, 80.0],
            prev_power=[0.0, 0.0, 0.0, 0.0],
        )
        # PV battery: p_net=0, pv_actual=300kW -> battery charges at 300kW
        p_rt = np.array([0.0, 0.0, 0.0, 0.0])
        soc_before = state.soc[1]
        _step_dynamics(agents, state, p_rt, dt_hr=5.0 / 60.0, pv_actual=300.0)
        # p_batt = p_net + p_pv = 0 + 300 = 300 kW charging
        assert state.soc[1] > soc_before

    def test_electrolyzer_inventory_increases(self, agents):
        state = SimulationState(
            soc=[1000.0, 750.0, 50.0, 80.0],
            prev_power=[0.0, 0.0, 0.0, 0.0],
        )
        p_rt = np.array([0.0, 0.0, 500.0, 0.0])
        soc_before = state.soc[2]
        _step_dynamics(agents, state, p_rt, dt_hr=5.0 / 60.0, pv_actual=0.0)
        assert state.soc[2] > soc_before  # Production increased inventory
