"""
forecasting/price.py — RTM and DAM price forecast models.

Per Handoff.md §3.7:
  - RTM price: AR(1) around current observation, phi=0.70
  - DAM forecast: realized RTM average + small Gaussian noise
"""

from __future__ import annotations

import numpy as np


def generate_rtm_price_forecast(
    current_price: float,
    H: int,
    phi: float = 0.70,
    mean_price: float | None = None,
    noise_std: float = 5.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate H-step RTM price forecast as AR(1) around current observation.

    price_forecast[k] = mean + phi^{k+1} * (current - mean) + noise

    Parameters
    ----------
    current_price : float
        Most recent RTM LMP ($/MWh).
    H : int
        Forecast horizon.
    phi : float
        AR(1) persistence parameter (0.70 per Handoff.md).
    mean_price : float or None
        Long-run mean. If None, uses current_price (random walk component).
    noise_std : float
        Per-step noise standard deviation ($/MWh, 5-min step).
    rng : np.random.Generator or None

    Returns
    -------
    price_forecast : ndarray (H,)
        Forecast RTM prices over horizon, clipped to [0, 5000].
    """
    if rng is None:
        rng = np.random.default_rng()
    if mean_price is None:
        mean_price = current_price

    forecast = np.zeros(H)
    eps = rng.normal(0, noise_std, H)

    # AR(1) with mean reversion
    running = current_price
    for k in range(H):
        running = mean_price + phi * (running - mean_price) + eps[k]
        forecast[k] = running

    return np.clip(forecast, -500.0, 5000.0)


def generate_dam_price_forecast(
    rtm_true_24h_avg: np.ndarray,
    noise_std_fraction: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate DAM (24-hour) price forecast.

    The DAM forecast is the expected RTM price for each hour, modeled as the
    true hourly average plus small Gaussian noise.

    Parameters
    ----------
    rtm_true_24h_avg : ndarray (24,)
        True hourly average RTM LMP ($/MWh).
    noise_std_fraction : float
        Noise as fraction of mean price level.
    rng : np.random.Generator or None

    Returns
    -------
    dam_price_forecast : ndarray (24,)
    """
    if rng is None:
        rng = np.random.default_rng()
    mean_level = np.mean(rtm_true_24h_avg)
    noise = rng.normal(0, noise_std_fraction * max(mean_level, 10.0), 24)
    return np.clip(rtm_true_24h_avg + noise, 0.0, 5000.0)
