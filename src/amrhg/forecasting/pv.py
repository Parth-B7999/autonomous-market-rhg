"""
forecasting/pv.py — PV generation forecast models.

Per Handoff.md §3.7:
  - RTM nowcast: persistence + AR(1) noise, phi=0.85, sigma=12% of capacity
  - DAM forecast: realized profile + Gaussian noise
"""

from __future__ import annotations

import numpy as np


def generate_rtm_pv_forecast(
    pv_current: float,
    H: int,
    phi: float = 0.85,
    noise_std: float | None = None,
    pv_capacity_kw: float = 300.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate H-step PV forecast using persistence + AR(1) noise.

    pv_forecast[k] = pv_current + sigma * sum_{j=0}^{k} phi^{k-j} * eps_j

    The AR(1) component models forecast error growing with horizon.

    Parameters
    ----------
    pv_current : float
        Current observed PV generation (kW).
    H : int
        Forecast horizon (number of 5-min steps).
    phi : float
        AR(1) persistence parameter.
    noise_std : float or None
        Per-step noise standard deviation (kW). If None, defaults to 12% of capacity.
    pv_capacity_kw : float
        PV capacity used to compute default noise_std.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    pv_forecast : ndarray (H,)
        Forecast PV generation at each step [kW], clipped to [0, pv_capacity].
    """
    if rng is None:
        rng = np.random.default_rng()
    if noise_std is None:
        noise_std = 0.12 * pv_capacity_kw  # 12% of capacity RMSE

    # Persistence baseline
    base = np.full(H, pv_current)

    # AR(1) noise accumulation
    eps = rng.normal(0, noise_std, H)
    ar_component = np.zeros(H)
    running = 0.0
    for k in range(H):
        running = phi * running + eps[k]
        ar_component[k] = running

    pv_forecast = base + ar_component
    return np.clip(pv_forecast, 0.0, pv_capacity_kw)


def generate_dam_pv_forecast(
    pv_true_24h: np.ndarray,
    noise_std_fraction: float = 0.12,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate DAM (24-hour) PV forecast as realized + Gaussian noise.

    Parameters
    ----------
    pv_true_24h : ndarray (24,)
        True hourly PV generation (average kW over each hour).
    noise_std_fraction : float
        Standard deviation as fraction of PV capacity.
    rng : np.random.Generator or None

    Returns
    -------
    pv_forecast_24h : ndarray (24,)
    """
    if rng is None:
        rng = np.random.default_rng()
    cap = np.max(pv_true_24h) / 0.85 if np.max(pv_true_24h) > 0 else 300.0
    noise = rng.normal(0, noise_std_fraction * cap, 24)
    return np.clip(pv_true_24h + noise, 0.0, cap * 1.1)
