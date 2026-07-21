"""Tests for analysis metrics module."""

import numpy as np
import pytest

from amrhg.simulation.logger import SimulationResult
from amrhg.analysis.metrics import (
    compute_cost_breakdown,
    compute_total_welfare,
    compute_coupling_violation_rate,
)


@pytest.fixture
def simple_result():
    """3 steps, 2 agents, trivial prices for deterministic cost calc."""
    T = 3
    return SimulationResult(
        method="test",
        p_net=np.array([[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]]),
        soc=np.zeros((2, T + 1)),
        rtm_prices=np.array([50000.0, 50000.0, 50000.0]),  # $/MWh
        dam_prices=np.array([40000.0]),
        da_schedules=np.array([[120.0], [180.0]]),
        total_cost=np.array([10.0, 20.0]),
        coupling_violations=np.array([0.0, 5.0, 50.0]),
        n_steps=T,
        agent_names=["A", "B"],
    )


@pytest.fixture
def settlement_cfg():
    return {"gamma_plus": 1.5, "gamma_minus": 2.0}


class TestCostBreakdown:
    def test_returns_expected_keys(self, simple_result, settlement_cfg):
        bd = compute_cost_breakdown(simple_result, 0, settlement_cfg)
        assert set(bd.keys()) == {"energy_cost", "imbalance_penalty", "total_cost"}

    def test_total_cost_matches_sum_of_components(self, simple_result, settlement_cfg):
        bd = compute_cost_breakdown(simple_result, 0, settlement_cfg)
        assert bd["total_cost"] == pytest.approx(
            bd["energy_cost"] + bd["imbalance_penalty"]
        )

    def test_zero_imbalance_yields_zero_penalty(self, settlement_cfg):
        """When RTM = DA exactly, imbalance penalty is zero."""
        T = 3
        sr = SimulationResult(
            method="test",
            p_net=np.array([[100.0, 100.0, 100.0], [200.0, 200.0, 200.0]]),
            soc=np.zeros((2, T + 1)),
            rtm_prices=np.array([50000.0, 50000.0, 50000.0]),
            dam_prices=np.array([40000.0]),
            da_schedules=np.array([[100.0], [200.0]]),  # exactly matches p_net
            total_cost=np.array([0.0, 0.0]),
            n_steps=T,
            agent_names=["A", "B"],
        )
        bd = compute_cost_breakdown(sr, 0, settlement_cfg)
        assert bd["imbalance_penalty"] == pytest.approx(0.0)

    def test_energy_cost_is_rtm_price_times_imbalance(self, simple_result, settlement_cfg):
        """energy = Σ (rtm[t]/1000) * (p[t] - da[t]). rtm=50000, da=120, p=100 → -20*50"""
        bd = compute_cost_breakdown(simple_result, 0, settlement_cfg)
        # Agent 0: p=100, da=120, so p-da = -20 at each step
        # energy_cost = 3 * (50000/1000) * (-20) = 3 * 50 * (-20) = -3000
        assert bd["energy_cost"] == pytest.approx(-3000.0)

    def test_over_delivery_uses_gamma_plus(self, simple_result, settlement_cfg):
        """Agent 1: p=200, da=180, imb=+20, gamma_plus=1.5 → penalty = 3*(1.5/1000)*20"""
        bd = compute_cost_breakdown(simple_result, 1, settlement_cfg)
        expected_penalty = 3 * (1.5 / 1000.0) * 20.0  # 0.09
        assert bd["imbalance_penalty"] == pytest.approx(expected_penalty)


class TestTotalWelfare:
    def test_sums_all_agent_costs(self):
        sr1 = SimulationResult(
            method="rhg", p_net=np.zeros((2, 10)), soc=np.zeros((2, 11)),
            rtm_prices=np.zeros(10), dam_prices=np.zeros(5),
            da_schedules=np.zeros((2, 5)), total_cost=np.array([100.0, 200.0]),
        )
        sr2 = SimulationResult(
            method="open_loop", p_net=np.zeros((2, 10)), soc=np.zeros((2, 11)),
            rtm_prices=np.zeros(10), dam_prices=np.zeros(5),
            da_schedules=np.zeros((2, 5)), total_cost=np.array([150.0, 250.0]),
        )
        welfare = compute_total_welfare({"rhg": sr1, "open_loop": sr2})
        assert welfare["rhg"] == pytest.approx(300.0)
        assert welfare["open_loop"] == pytest.approx(400.0)


class TestCouplingViolationRate:
    def test_no_violations(self):
        sr = SimulationResult(
            method="test", p_net=np.zeros((2, 10)), soc=np.zeros((2, 11)),
            rtm_prices=np.zeros(10), dam_prices=np.zeros(5),
            da_schedules=np.zeros((2, 5)),
            coupling_violations=np.zeros(10),
        )
        rate = compute_coupling_violation_rate(sr, 100.0, 1000.0)
        assert rate == 0.0

    def test_all_violations(self):
        sr = SimulationResult(
            method="test", p_net=np.zeros((2, 10)), soc=np.zeros((2, 11)),
            rtm_prices=np.zeros(10), dam_prices=np.zeros(5),
            da_schedules=np.zeros((2, 5)),
            coupling_violations=np.full(10, 100.0),
        )
        rate = compute_coupling_violation_rate(sr, 100.0, 1000.0)
        assert rate == 1.0

    def test_none_violations_returns_zero(self):
        sr = SimulationResult(
            method="test", p_net=np.zeros((2, 10)), soc=np.zeros((2, 11)),
            rtm_prices=np.zeros(10), dam_prices=np.zeros(5),
            da_schedules=np.zeros((2, 5)),
            coupling_violations=None,
        )
        rate = compute_coupling_violation_rate(sr, 100.0, 1000.0)
        assert rate == 0.0

    def test_partial_violations(self):
        sr = SimulationResult(
            method="test", p_net=np.zeros((2, 10)), soc=np.zeros((2, 11)),
            rtm_prices=np.zeros(10), dam_prices=np.zeros(5),
            da_schedules=np.zeros((2, 5)),
            coupling_violations=np.array([0.0, 0.0, 5.0, 10.0, 0.0]),
        )
        rate = compute_coupling_violation_rate(sr, 100.0, 1000.0)
        assert rate == 0.4  # 2 out of 5
