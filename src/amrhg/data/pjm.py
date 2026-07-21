"""
data/pjm.py — Real PJM data loader for market simulation.

Loads real historical PJM data from data/raw/:
  - da_hrl_lmps_{year}.csv    : hourly DAM settlement point prices (PJM-RTO)
  - rt_fivemin_mnt_lmps_{year}.csv : 5-min RTM settlement point prices (PJM-RTO)
  - gen_by_fuel_{year}.csv    : hourly generation by fuel type (Solar MW)

Historical std deviations are computed from the full year and used to generate
agent price/PV forecasts as: forecast = true_value + N(0, sigma^2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _raw_dir() -> Path:
    here = Path(__file__).resolve()
    # pjm.py → data/ → amrhg/ → src/ → project_root/
    return here.parents[3] / "data" / "raw"


def _load_da(year: int) -> pd.DataFrame:
    path = _raw_dir() / f"da_hrl_lmps_{year}.csv"
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["datetime_beginning_ept"],
                               format="%m/%d/%Y %I:%M:%S %p")
    return df.sort_values("dt").reset_index(drop=True)


def _load_rt(year: int) -> pd.DataFrame:
    path = _raw_dir() / f"rt_fivemin_mnt_lmps_{year}.csv"
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["datetime_beginning_ept"],
                               format="%m/%d/%Y %I:%M:%S %p")
    return df.sort_values("dt").reset_index(drop=True)


def _load_solar(year: int) -> pd.DataFrame:
    path = _raw_dir() / f"gen_by_fuel_{year}.csv"
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["datetime_beginning_ept"],
                               format="%m/%d/%Y %I:%M:%S %p")
    solar = df[df["fuel_type"] == "Solar"].sort_values("dt").reset_index(drop=True)
    return solar


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pjm_data(cfg: dict) -> dict:
    """
    Load real PJM data for the simulation week specified in cfg.

    Historical std deviations are computed from the full year so that
    agent forecasts are statistically calibrated:
        forecast[h] = true_cleared_price[h] + N(0, sigma_h^2)

    Parameters
    ----------
    cfg : dict
        Full config (configs/base.yaml).  Uses:
          simulation.start_date  (str, e.g. "2024-07-08")
          simulation.n_days      (int)
          simulation.seed        (int)
          agents.pv_battery.pv_capacity_kw  (float)

    Returns
    -------
    dict with keys:
        dam_lmp          : (n_days*24,)   true cleared DA LMPs [$/MWh]
        dam_lmp_std      : (24,)          historical hourly std dev [$/MWh]
        rtm_lmp          : (n_days*288,)  true 5-min RT LMPs [$/MWh]
        rtm_lmp_std      : (288,)         historical 5-min-interval std dev [$/MWh]
        pv_kw_hourly     : (n_days*24,)   true hourly PV generation [kW]
        pv_kw_5min       : (n_days*288,)  true 5-min PV generation [kW] (held from hourly)
        pv_cf_hourly     : (n_days*24,)   PV capacity factor [0, 1]
        pv_std_hourly    : (24,)          historical hourly PV std dev [kW]
        peak_solar_mw    : float          PJM system solar peak used for CF scaling
        pv_capacity_kw   : float          agent PV capacity from config
    """
    sim_cfg = cfg["simulation"]
    start_str = sim_cfg["start_date"]
    n_days = sim_cfg["n_days"]
    pv_cap_kw = cfg["agents"]["pv_battery"]["pv_capacity_kw"]

    start = pd.Timestamp(start_str)
    end = start + pd.Timedelta(days=n_days)
    year = start.year

    # ── DA prices ────────────────────────────────────────────────────────────
    da = _load_da(year)
    da["hour"] = da["dt"].dt.hour

    dam_std_by_hour = (
        da.groupby("hour")["total_lmp_da"].std().sort_index().values
    )  # (24,)

    da_week = da[(da["dt"] >= start) & (da["dt"] < end)].copy()
    if len(da_week) != n_days * 24:
        raise ValueError(
            f"Expected {n_days*24} DA rows for {start_str} + {n_days} days, "
            f"got {len(da_week)}. Check that year {year} data covers this range."
        )
    dam_lmp = da_week["total_lmp_da"].values.astype(float)

    # ── RT prices ─────────────────────────────────────────────────────────────
    rt = _load_rt(year)
    rt["interval_of_day"] = rt["dt"].dt.hour * 12 + rt["dt"].dt.minute // 5

    rtm_std_by_interval = (
        rt.groupby("interval_of_day")["total_lmp_rt"].std().sort_index().values
    )  # (288,)

    rt_week = rt[(rt["dt"] >= start) & (rt["dt"] < end)].copy()
    if len(rt_week) != n_days * 288:
        raise ValueError(
            f"Expected {n_days*288} RT rows, got {len(rt_week)}."
        )
    rtm_lmp = rt_week["total_lmp_rt"].values.astype(float)

    # ── Solar / PV ────────────────────────────────────────────────────────────
    solar = _load_solar(year)
    solar["hour"] = solar["dt"].dt.hour

    # 99.9th percentile as peak to avoid extreme outliers
    peak_solar_mw = float(solar["mw"].quantile(0.999))
    solar["cf"] = (solar["mw"] / peak_solar_mw).clip(0.0, 1.0)

    # Hourly std dev of PV generation (in kW at our agent's capacity)
    pv_std_by_hour = (
        solar.groupby("hour")["cf"].std().sort_index().values * pv_cap_kw
    )  # (24,)

    sol_week = solar[(solar["dt"] >= start) & (solar["dt"] < end)].copy()
    if len(sol_week) != n_days * 24:
        raise ValueError(
            f"Expected {n_days*24} solar rows, got {len(sol_week)}."
        )

    pv_cf_hourly = sol_week["cf"].values.astype(float)
    pv_kw_hourly = pv_cf_hourly * pv_cap_kw
    # Expand hourly → 5-min (each hourly value held constant for 12 steps)
    pv_kw_5min = np.repeat(pv_kw_hourly, 12)

    return {
        "dam_lmp": dam_lmp,
        "dam_lmp_std": dam_std_by_hour,
        "rtm_lmp": rtm_lmp,
        "rtm_lmp_std": rtm_std_by_interval,
        "pv_kw_hourly": pv_kw_hourly,
        "pv_kw_5min": pv_kw_5min,
        "pv_cf_hourly": pv_cf_hourly,
        "pv_std_hourly": pv_std_by_hour,
        "peak_solar_mw": peak_solar_mw,
        "pv_capacity_kw": pv_cap_kw,
    }


def sample_dam_forecast(
    dam_lmp_true: np.ndarray,
    dam_lmp_std: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a noisy DA price forecast for one day (24 hours).

    forecast[h] = true_cleared_price[h] + N(0, sigma_h^2)

    Parameters
    ----------
    dam_lmp_true : (24,)  true cleared DA price [$/MWh]
    dam_lmp_std  : (24,)  historical std dev per hour [$/MWh]
    rng          : numpy Generator

    Returns
    -------
    forecast : (24,) [$/MWh], clipped to [0, 5000]
    """
    noise = rng.normal(0.0, dam_lmp_std)
    return np.clip(dam_lmp_true + noise, 0.0, 5000.0)


def sample_pv_forecast(
    pv_kw_true: np.ndarray,
    pv_std_hourly: np.ndarray,
    pv_capacity_kw: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a noisy PV forecast for one day (24 hours).

    forecast[h] = true_pv[h] + N(0, sigma_h^2), clipped to [0, pv_capacity_kw]

    Parameters
    ----------
    pv_kw_true      : (24,)  true hourly PV generation [kW]
    pv_std_hourly   : (24,)  historical std dev per hour [kW]
    pv_capacity_kw  : float
    rng             : numpy Generator

    Returns
    -------
    forecast : (24,) [kW]
    """
    noise = rng.normal(0.0, pv_std_hourly)
    return np.clip(pv_kw_true + noise, 0.0, pv_capacity_kw)
