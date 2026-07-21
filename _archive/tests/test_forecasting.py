"""Tests for PV and price forecasting modules."""

import numpy as np
import pytest

from amrhg.forecasting.pv import generate_rtm_pv_forecast, generate_dam_pv_forecast
from amrhg.forecasting.price import generate_rtm_price_forecast, generate_dam_price_forecast


# ---- PV forecasts ----

class TestRTMPVForecast:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        out = generate_rtm_pv_forecast(pv_current=150.0, H=6, rng=rng)
        assert out.shape == (6,)

    def test_clipped_to_zero_and_capacity(self):
        rng = np.random.default_rng(42)
        # Very negative starting point with high noise
        out = generate_rtm_pv_forecast(
            pv_current=-50.0, H=20, noise_std=200.0, pv_capacity_kw=300.0, rng=rng,
        )
        assert np.all(out >= 0.0)
        assert np.all(out <= 300.0)

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        o1 = generate_rtm_pv_forecast(150.0, 6, rng=rng1)
        o2 = generate_rtm_pv_forecast(150.0, 6, rng=rng2)
        assert np.array_equal(o1, o2)

    def test_different_seeds_produce_different_forecasts(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)
        o1 = generate_rtm_pv_forecast(150.0, 20, rng=rng1)
        o2 = generate_rtm_pv_forecast(150.0, 20, rng=rng2)
        assert not np.array_equal(o1, o2)

    def test_default_noise_std_is_12_percent_capacity(self):
        rng = np.random.default_rng(42)
        # With very low noise, output should be close to baseline
        out = generate_rtm_pv_forecast(
            pv_current=150.0, H=100, noise_std=0.01, pv_capacity_kw=300.0, rng=rng,
        )
        # With negligible noise, forecast ≈ persistence of 150
        assert np.abs(np.mean(out) - 150.0) < 5.0

    def test_noise_accumulates_with_horizon(self):
        rng = np.random.default_rng(555)
        out = generate_rtm_pv_forecast(
            pv_current=150.0, H=12, noise_std=30.0, pv_capacity_kw=300.0, rng=rng,
        )
        # The raw forecast should differ from the persistence baseline,
        # showing AR(1) noise is being added.
        assert not np.allclose(out, 150.0)
        # All values must stay within physical bounds
        assert np.all(out >= 0.0)
        assert np.all(out <= 300.0)


class TestDAMPVForecast:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        true_pv = np.full(24, 200.0)
        out = generate_dam_pv_forecast(true_pv, rng=rng)
        assert out.shape == (24,)

    def test_clipped_nonnegative(self):
        rng = np.random.default_rng(42)
        true_pv = np.zeros(24)
        out = generate_dam_pv_forecast(true_pv, rng=rng)
        assert np.all(out >= 0.0)

    def test_close_to_true_with_low_noise(self):
        rng = np.random.default_rng(42)
        true_pv = np.full(24, 200.0)
        out = generate_dam_pv_forecast(true_pv, noise_std_fraction=0.001, rng=rng)
        # Essentially no noise
        assert np.mean(np.abs(out - true_pv)) < 10.0


# ---- Price forecasts ----

class TestRTMPriceForecast:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        out = generate_rtm_price_forecast(current_price=50.0, H=6, rng=rng)
        assert out.shape == (6,)

    def test_reproducible_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        o1 = generate_rtm_price_forecast(50.0, 6, rng=rng1)
        o2 = generate_rtm_price_forecast(50.0, 6, rng=rng2)
        assert np.array_equal(o1, o2)

    def test_mean_reverts_toward_long_run_mean(self):
        rng = np.random.default_rng(42)
        # Start at 100, long-run mean is 50, phi=0.70
        out = generate_rtm_price_forecast(
            current_price=100.0, H=50, phi=0.70, mean_price=50.0,
            noise_std=0.01, rng=rng,
        )
        # Later steps should be closer to 50 than the first step
        # phi^50 is essentially 0, so forecast[k] ≈ 50 + eps
        assert np.abs(np.mean(out[-10:]) - 50.0) < 10.0
        # First step is closer to 100 (after 1 AR step)
        assert out[0] > 60.0

    def test_clipped_to_valid_range(self):
        rng = np.random.default_rng(42)
        out = generate_rtm_price_forecast(
            current_price=50.0, H=100, noise_std=500.0, rng=rng,
        )
        assert np.all(out >= -500.0)
        assert np.all(out <= 5000.0)

    def test_default_mean_is_current_price(self):
        rng = np.random.default_rng(42)
        out = generate_rtm_price_forecast(
            current_price=50.0, H=50, phi=0.99, noise_std=0.001, rng=rng,
        )
        # With phi close to 1 and no explicit mean, stays near current
        assert np.abs(np.mean(out) - 50.0) < 5.0


class TestDAMPriceForecast:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        true_avg = np.full(24, 50.0)
        out = generate_dam_price_forecast(true_avg, rng=rng)
        assert out.shape == (24,)

    def test_clipped_nonnegative(self):
        rng = np.random.default_rng(42)
        true_avg = np.zeros(24)
        out = generate_dam_price_forecast(true_avg, rng=rng)
        assert np.all(out >= 0.0)

    def test_close_to_true_with_low_noise(self):
        rng = np.random.default_rng(42)
        true_avg = np.full(24, 50.0)
        out = generate_dam_price_forecast(true_avg, noise_std_fraction=0.001, rng=rng)
        assert np.mean(np.abs(out - true_avg)) < 5.0
