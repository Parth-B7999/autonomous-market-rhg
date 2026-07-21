"""Tests for synthetic ERCOT data generation."""

import numpy as np
import pytest

from amrhg.data.ercot import generate_synthetic_week, load_or_generate_ercot_data


class TestGenerateSyntheticWeek:
    def test_returns_dict_with_expected_keys(self):
        data = generate_synthetic_week(start_date="2023-07-10", n_days=1, seed=42)
        assert set(data.keys()) == {
            "timestamps", "dam_lmp", "rtm_lmp",
            "pv_capacity_factor", "dam_timestamps",
        }

    def test_correct_shapes_one_day(self):
        data = generate_synthetic_week(n_days=1, seed=42)
        assert len(data["timestamps"]) == 288
        assert data["dam_lmp"].shape == (24,)
        assert data["rtm_lmp"].shape == (288,)
        assert data["pv_capacity_factor"].shape == (288,)
        assert len(data["dam_timestamps"]) == 24

    def test_correct_shapes_seven_days(self):
        data = generate_synthetic_week(n_days=7, seed=42)
        assert len(data["timestamps"]) == 2016
        assert data["dam_lmp"].shape == (168,)
        assert data["rtm_lmp"].shape == (2016,)
        assert data["pv_capacity_factor"].shape == (2016,)

    def test_dam_lmp_in_realistic_range(self):
        data = generate_synthetic_week(n_days=7, seed=42)
        assert np.all(data["dam_lmp"] >= 15.0)
        assert np.all(data["dam_lmp"] <= 120.0)
        assert 20 < np.min(data["dam_lmp"]) < 45
        assert 50 < np.max(data["dam_lmp"]) < 80

    def test_rtm_lmp_in_realistic_range(self):
        data = generate_synthetic_week(n_days=7, seed=42)
        assert np.all(data["rtm_lmp"] >= 0.0)
        assert np.all(data["rtm_lmp"] <= 200.0)
        # RTM should have wider range than DAM
        assert np.max(data["rtm_lmp"]) > np.max(data["dam_lmp"])
        assert np.min(data["rtm_lmp"]) < np.min(data["dam_lmp"])

    def test_pv_cf_in_valid_range(self):
        data = generate_synthetic_week(n_days=7, seed=42)
        assert np.all(data["pv_capacity_factor"] >= 0.0)
        assert np.all(data["pv_capacity_factor"] <= 1.0)

    def test_pv_zero_at_night(self):
        data = generate_synthetic_week(n_days=1, seed=42)
        # First ~72 steps (6 hours) should be near-zero PV
        cf_night = data["pv_capacity_factor"][:60]
        assert np.max(cf_night) < 0.05

    def test_pv_peaks_near_noon(self):
        data = generate_synthetic_week(n_days=1, seed=42)
        # Steps 155-185 = 12:55-15:25 ~ solar maximum
        cf_midday = data["pv_capacity_factor"][155:185]
        assert np.max(cf_midday) > 0.5

    def test_reproducible_with_seed(self):
        d1 = generate_synthetic_week(seed=42)
        d2 = generate_synthetic_week(seed=42)
        assert np.array_equal(d1["dam_lmp"], d2["dam_lmp"])
        assert np.array_equal(d1["rtm_lmp"], d2["rtm_lmp"])
        assert np.array_equal(d1["pv_capacity_factor"], d2["pv_capacity_factor"])

    def test_different_seeds_produce_different_data(self):
        d1 = generate_synthetic_week(seed=42)
        d2 = generate_synthetic_week(seed=99)
        # Different RNG streams produce different data
        assert not np.array_equal(d1["rtm_lmp"], d2["rtm_lmp"])
        assert not np.array_equal(d1["pv_capacity_factor"], d2["pv_capacity_factor"])


class TestLoadOrGenerate:
    def test_returns_dict_with_expected_keys(self):
        cfg = {"simulation": {"start_date": "2023-07-10", "n_days": 1, "seed": 42}}
        data = load_or_generate_ercot_data(cfg)
        assert set(data.keys()) == {
            "timestamps", "dam_lmp", "rtm_lmp",
            "pv_capacity_factor", "dam_timestamps",
        }

    def test_timestamps_match_n_days(self):
        cfg = {"simulation": {"start_date": "2023-07-10", "n_days": 2, "seed": 42}}
        data = load_or_generate_ercot_data(cfg)
        # Cached file is always a full week; the function returns all of it.
        # The datetime list is regenerated to match n_days in length.
        assert len(data["timestamps"]) == data["rtm_lmp"].shape[0]
        assert len(data["dam_timestamps"]) == data["dam_lmp"].shape[0]
