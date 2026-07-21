"""
data/ercot.py — Synthetic ERCOT LZ_SOUTH data generator.

Produces realistic synthetic DAM/RTM LMPs and PV capacity factors for
a representative summer week. Cached to data/processed/ so the full
pipeline is reproducible without external data dependencies.

Real ERCOT LZ_SOUTH characteristics (summer):
  - Solar-rich South Texas: PV capacity factor peaks ~0.85 at noon
  - DAM LMPs: diurnal with solar noon dip (~$20/MWh) and evening peak (~$80/MWh)
  - RTM LMPs: 5-min, mean-reverting around DAM, higher volatility
  - Morning ramp 06:00-08:00, evening ramp 16:00-20:00
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np


def generate_synthetic_week(
    start_date: str = "2023-07-10",
    n_days: int = 7,
    seed: int = 42,
) -> dict:
    """
    Generate 1 week of synthetic ERCOT LZ_SOUTH data at 5-min resolution.

    Returns
    -------
    dict with keys:
        timestamps      : list[datetime]  — length 2016 (n_days × 288 five-min intervals)
        dam_lmp         : ndarray (n_days×24,)   — hourly DAM prices ($/MWh)
        rtm_lmp         : ndarray (2016,)         — 5-min RTM prices ($/MWh)
        pv_capacity_factor : ndarray (2016,)      — 0..1, 5-min PV capacity factor
        dam_timestamps  : list[datetime]  — length n_days×24
    """
    rng = np.random.default_rng(seed)
    n_rtm = n_days * 288  # 5-min intervals per day

    base = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [base + timedelta(minutes=5 * t) for t in range(n_rtm)]
    dam_ts = [base + timedelta(hours=h) for h in range(n_days * 24)]

    # ---- DAM LMP: sinusoidal diurnal pattern ----
    # Low at solar noon (hour 12-14), high at evening peak (hour 18-20)
    hours = np.arange(n_days * 24)
    hour_of_day = hours % 24
    # Base: $50/MWh mean, $20 amplitude
    dam_base = 50.0 - 15.0 * np.cos(2 * np.pi * (hour_of_day - 14) / 24)
    # Add day-to-day variation
    day_offsets = rng.normal(0, 5.0, n_days)
    dam_lmp = dam_base + np.repeat(day_offsets, 24)
    # Ensure realistic bounds
    dam_lmp = np.clip(dam_lmp, 15.0, 120.0)

    # ---- RTM LMP: interpolate DAM + AR(1) mean-reverting noise ----
    # Interpolate hourly DAM to 5-min
    dam_interp = np.repeat(dam_lmp, 12)  # 12 five-min steps per hour
    # AR(1) noise: x_{t+1} = phi * x_t + eps_t
    phi_price = 0.70
    noise_std_price = 8.0  # $/MWh per 5-min step
    ar_noise = np.zeros(n_rtm)
    ar_noise[0] = rng.normal(0, noise_std_price)
    for t in range(1, n_rtm):
        ar_noise[t] = phi_price * ar_noise[t - 1] + rng.normal(0, noise_std_price)
    rtm_lmp = dam_interp + ar_noise
    rtm_lmp = np.clip(rtm_lmp, 0.0, 200.0)

    # ---- PV capacity factor: clear-sky cosine + cloud noise ----
    # Sunrise ~06:00, sunset ~20:00 in July (14 hours daylight)
    minutes_of_day = np.array([(ts.hour * 60 + ts.minute) for ts in timestamps])
    sunrise_min = 360   # 06:00
    sunset_min = 1200   # 20:00
    daylight = np.clip((minutes_of_day - sunrise_min) / (sunset_min - sunrise_min), 0, 1)
    # Clear-sky: sin(pi * fraction) peaks at 0.85
    clear_sky = 0.85 * np.sin(np.pi * daylight)
    clear_sky[daylight <= 0] = 0.0
    clear_sky[daylight >= 1] = 0.0
    # Cloud noise: AR(1) with negative skew (clouds reduce, not increase)
    phi_cloud = 0.85
    cloud_noise = np.zeros(n_rtm)
    cloud_noise[0] = rng.normal(0, 0.08)
    for t in range(1, n_rtm):
        cloud_noise[t] = phi_cloud * cloud_noise[t - 1] + rng.normal(0, 0.06)
    pv_cf = np.clip(clear_sky + cloud_noise, 0.0, 1.0)
    pv_cf[daylight <= 0.01] = 0.0  # no PV at night

    return {
        "timestamps": timestamps,
        "dam_lmp": dam_lmp,
        "rtm_lmp": rtm_lmp,
        "pv_capacity_factor": pv_cf,
        "dam_timestamps": dam_ts,
    }


def load_or_generate_ercot_data(config: dict) -> dict:
    """
    Load cached synthetic data or generate fresh.

    Parameters
    ----------
    config : dict
        Must contain simulation.start_date, simulation.n_days, simulation.seed.

    Returns
    -------
    dict with the same keys as generate_synthetic_week().
    """
    sim_cfg = config["simulation"]
    start_date = sim_cfg["start_date"]
    n_days = sim_cfg["n_days"]
    seed = sim_cfg["seed"]

    # Determine cache path relative to project root
    cache_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "data", "processed"
    )
    cache_path = os.path.join(cache_dir, "synthetic_week.npz")

    if os.path.exists(cache_path):
        loaded = np.load(cache_path, allow_pickle=True)
        dam = loaded["dam_lmp"]
        rtm = loaded["rtm_lmp"]
        pv = loaded["pv_capacity_factor"]
        n_rtm = len(rtm)
        n_hr = len(dam)
        base = datetime.strptime(start_date, "%Y-%m-%d")
        return {
            "timestamps": [base + timedelta(minutes=5 * t) for t in range(n_rtm)],
            "dam_lmp": dam,
            "rtm_lmp": rtm,
            "pv_capacity_factor": pv,
            "dam_timestamps": [base + timedelta(hours=h) for h in range(n_hr)],
        }

    data = generate_synthetic_week(start_date, n_days, seed)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        cache_path,
        dam_lmp=data["dam_lmp"],
        rtm_lmp=data["rtm_lmp"],
        pv_capacity_factor=data["pv_capacity_factor"],
    )
    return data
