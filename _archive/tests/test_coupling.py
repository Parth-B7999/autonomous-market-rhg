"""tests/test_coupling.py — Coupling and game-builder unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from amrhg.agents.base import make_parameter_layout, ParameterLayout
from amrhg.agents.vrfb import VRFBAgent
from amrhg.game.builder import build_rtm_game, validate_game
from amrhg.game.coupling import make_coupling_rhs, check_coupling_feasible
from amrhg.game.params import pack_rtm_params, unpack_rtm_params, default_rtm_params
from amrhg.solvers.game import GNEGame

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N, H = 2, 2          # minimal for fast tests; n_p = 2*(2+1)+2+2 = 10
DT = 5 / 60

VRFB_CFG = {
    "capacity_kw": 300.0, "energy_kwh": 2000.0,
    "roundtrip_efficiency": 0.75,
    "soc_min_kwh": 200.0, "soc_max_kwh": 1800.0, "soc_init_kwh": 1000.0,
}
SETTLEMENT = {"gamma_plus": 1.5, "gamma_minus": 2.0}


@pytest.fixture
def layout() -> ParameterLayout:
    return make_parameter_layout(N, H, DT)


@pytest.fixture
def game(layout: ParameterLayout) -> GNEGame:
    agents = [VRFBAgent(VRFB_CFG), VRFBAgent(VRFB_CFG)]
    return build_rtm_game(agents, layout, SETTLEMENT)


# ---------------------------------------------------------------------------
# make_coupling_rhs
# ---------------------------------------------------------------------------

class TestMakeCouplingRhs:
    def test_d_shape_and_zeros(self, layout: ParameterLayout) -> None:
        d, _ = make_coupling_rhs(layout)
        assert d.shape == (2 * H,)
        np.testing.assert_array_equal(d, np.zeros(2 * H))

    def test_S_coup_shape(self, layout: ParameterLayout) -> None:
        _, S = make_coupling_rhs(layout)
        assert S.shape == (2 * H, layout.n_p)

    def test_upper_rows_at_l_max_idx(self, layout: ParameterLayout) -> None:
        _, S = make_coupling_rhs(layout)
        np.testing.assert_array_equal(S[:H, layout.l_max_idx], np.ones(H))

    def test_lower_rows_at_l_min_idx(self, layout: ParameterLayout) -> None:
        _, S = make_coupling_rhs(layout)
        np.testing.assert_array_equal(S[H:, layout.l_min_idx], -np.ones(H))

    def test_no_other_nonzeros(self, layout: ParameterLayout) -> None:
        _, S = make_coupling_rhs(layout)
        S_copy = S.copy()
        S_copy[:H, layout.l_max_idx] = 0.0
        S_copy[H:, layout.l_min_idx] = 0.0
        assert np.all(S_copy == 0.0), "S_coup has unexpected nonzero entries"


# ---------------------------------------------------------------------------
# check_coupling_feasible
# ---------------------------------------------------------------------------

class TestCheckCouplingFeasible:
    def test_zero_power_feasible_above_l_min(self, layout: ParameterLayout) -> None:
        # p = 0 for both agents → aggregate = 0 → violates L_min = 1000 kW
        x_all = [np.zeros(3 * H), np.zeros(3 * H)]
        C_list = [ag.C for ag in
                  [VRFBAgent(VRFB_CFG).build_rtm_agent(i, layout, SETTLEMENT)
                   for i in range(N)]]
        p = default_rtm_params(layout, l_min_kw=0.0)   # L_min = 0, so zero power OK
        ok, lhs = check_coupling_feasible(x_all, C_list, p, layout)
        assert ok, f"Zero power with L_min=0 should be feasible, lhs={lhs}"

    def test_over_l_max_infeasible(self, layout: ParameterLayout) -> None:
        # Both agents at 300 kW → aggregate = 600 > L_max = 500
        x_all = [np.zeros(3 * H), np.zeros(3 * H)]
        for x in x_all:
            x[:H] = 300.0                   # power block
        C_list = [ag.C for ag in
                  [VRFBAgent(VRFB_CFG).build_rtm_agent(i, layout, SETTLEMENT)
                   for i in range(N)]]
        p = default_rtm_params(layout, l_max_kw=500.0, l_min_kw=0.0)
        ok, lhs = check_coupling_feasible(x_all, C_list, p, layout)
        assert not ok, f"Aggregate 600 kW should violate L_max=500, lhs={lhs}"


# ---------------------------------------------------------------------------
# build_rtm_game / validate_game
# ---------------------------------------------------------------------------

class TestBuildRtmGame:
    def test_returns_gne_game(self, game: GNEGame) -> None:
        assert isinstance(game, GNEGame)

    def test_n_agents(self, game: GNEGame) -> None:
        assert game.N == N

    def test_n_coupling(self, game: GNEGame) -> None:
        assert game.n_coupling == 2 * H

    def test_n_x_total(self, game: GNEGame) -> None:
        assert game.n_x_total == N * 3 * H

    def test_validate_clean(self, game: GNEGame, layout: ParameterLayout) -> None:
        errors = validate_game(game, layout)
        assert errors == [], f"validate_game found errors: {errors}"

    def test_coupling_feasible_at_zero(
        self, game: GNEGame, layout: ParameterLayout
    ) -> None:
        import numpy as np
        x = np.zeros(game.n_x_total)
        p = default_rtm_params(layout, l_min_kw=0.0)
        assert game.coupling_feasible(x, p), "Zero-power should be coupling-feasible with L_min=0"

    def test_wrong_agent_count_raises(self, layout: ParameterLayout) -> None:
        with pytest.raises(ValueError, match="n_agents"):
            build_rtm_game([VRFBAgent(VRFB_CFG)], layout, SETTLEMENT)  # 1 ≠ 2


# ---------------------------------------------------------------------------
# pack / unpack params
# ---------------------------------------------------------------------------

class TestPackUnpackParams:
    def test_round_trip(self, layout: ParameterLayout) -> None:
        state_inits = [1000.0, 800.0]
        lmp = np.array([25.0, 35.0])
        da = [np.array([50.0, 60.0]), np.array([30.0, 40.0])]
        p = pack_rtm_params(state_inits, lmp, da, 2500.0, 1000.0, layout)
        d = unpack_rtm_params(p, layout)
        assert d["state_inits"] == state_inits
        np.testing.assert_array_equal(d["lmp_rt"], lmp)
        np.testing.assert_array_equal(d["da_schedules"][0], da[0])
        np.testing.assert_array_equal(d["da_schedules"][1], da[1])
        assert d["l_max_kw"] == 2500.0
        assert d["l_min_kw"] == 1000.0

    def test_wrong_n_agents_raises(self, layout: ParameterLayout) -> None:
        with pytest.raises(ValueError):
            pack_rtm_params([1.0], np.ones(H), [np.ones(H)], 2500.0, 1000.0, layout)

    def test_wrong_lmp_shape_raises(self, layout: ParameterLayout) -> None:
        with pytest.raises(ValueError):
            pack_rtm_params([1.0]*N, np.ones(H+1), [np.ones(H)]*N, 2500.0, 1000.0, layout)

    def test_default_rtm_params_shape(self, layout: ParameterLayout) -> None:
        p = default_rtm_params(layout)
        assert p.shape == (layout.n_p,)
