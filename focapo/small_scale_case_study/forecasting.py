"""
forecasting.py — Real PJM data, scaled to agent nameplate capacity.

How renewable scaling works
---------------------------
PJM Solar generation on July 8 noon  : 6000 MW  (system-wide)
PJM Solar annual peak                 : 7953 MW  (system-wide)
Agent PV nameplate                    : 250 kW

Agent PV output at that hour = (6000 / 7953) * 250 kW = 188 kW

Same logic for wind.  Agents with renewable_cap_kw = 0 always get g = 0.

Forecast noise
--------------
RT price   : real future 5-min price  + N(0, 30 $/MWh)    -- intentionally bad
DA price   : real future hourly price + N(0, 16 $/MWh)
Renewable  : real future hourly value + N(0, 20% of cap)
             (0 for agents with no renewable)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

_HERE    = Path(__file__).parent
DATA_DIR = (_HERE.parent / "autonomous_market_rhg" / "data" / "raw").resolve()

STEPS_PER_DAY  = 288
STEPS_PER_HOUR = 12


# ── Load one week of real PJM data ────────────────────────────────────────────

def load_pjm_week(start_date: str = '2024-07-08', n_days: int = 7) -> dict:
    """
    Returns
    -------
    'rt_lmp'     : (n_days*288,)  real 5-min RT LMP  [$/MWh]
    'da_lmp'     : (n_days*24,)   real hourly DA LMP [$/MWh]
    'solar_mw'   : (n_days*24,)   real hourly PJM Solar generation [MW]
    'wind_mw'    : (n_days*24,)   real hourly PJM Wind  generation [MW]
    'solar_peak' : float          annual PJM Solar peak [MW]
    'wind_peak'  : float          annual PJM Wind  peak [MW]
    'n_days'     : int
    """
    fmt   = '%m/%d/%Y %I:%M:%S %p'
    start = pd.Timestamp(start_date)
    end   = start + pd.Timedelta(days=n_days)
    yr    = start.year   # pick the matching annual CSV (2024 / 2025 available)

    # ── RT LMP (5-min) ────────────────────────────────────────────────────────
    rt = (pd.read_csv(DATA_DIR / f'rt_fivemin_mnt_lmps_{yr}.csv')
            .rename(columns={'datetime_beginning_ept': 'dt'}))
    rt['dt'] = pd.to_datetime(rt['dt'], format=fmt)
    rt = rt.sort_values('dt').drop_duplicates('dt').set_index('dt')
    rt_lmp = (rt.loc[start : end - pd.Timedelta(minutes=5), 'total_lmp_rt']
                .values[:n_days * STEPS_PER_DAY].astype(float).clip(0))

    # ── DA LMP (hourly) ───────────────────────────────────────────────────────
    da = (pd.read_csv(DATA_DIR / f'da_hrl_lmps_{yr}.csv')
            .rename(columns={'datetime_beginning_ept': 'dt'}))
    da['dt'] = pd.to_datetime(da['dt'], format=fmt)
    da = da.sort_values('dt').drop_duplicates('dt').set_index('dt')
    da_lmp = (da.loc[start : end - pd.Timedelta(hours=1), 'total_lmp_da']
                .values[:n_days * 24].astype(float).clip(0))

    # ── Solar and Wind MW (hourly) ────────────────────────────────────────────
    gen = pd.read_csv(DATA_DIR / f'gen_by_fuel_{yr}.csv')
    gen['dt'] = pd.to_datetime(gen['datetime_beginning_ept'], format=fmt)
    gen = gen.set_index('dt')

    solar_all = gen[gen['fuel_type'] == 'Solar']['mw'].groupby(level=0).sum().sort_index()
    wind_all  = gen[gen['fuel_type'] == 'Wind' ]['mw'].groupby(level=0).sum().sort_index()

    solar_peak = float(solar_all.max())
    wind_peak  = float(wind_all.max())

    solar_mw = (solar_all.loc[start : end - pd.Timedelta(hours=1)]
                          .values[:n_days * 24].astype(float).clip(0))
    wind_mw  = (wind_all .loc[start : end - pd.Timedelta(hours=1)]
                          .values[:n_days * 24].astype(float).clip(0))

    return {
        'rt_lmp':     rt_lmp,
        'da_lmp':     da_lmp,
        'solar_mw':   solar_mw,
        'wind_mw':    wind_mw,
        'solar_peak': solar_peak,
        'wind_peak':  wind_peak,
        'n_days':     n_days,
    }


def _agent_ren(week: dict, ag_type: str | None, cap_kw: float,
               hour_start: int, n_hours: int) -> np.ndarray:
    """
    Agent renewable output [kW] for n_hours hours starting at hour_start.
    Returns zeros if cap_kw <= 0 or ag_type is None.
    """
    if cap_kw <= 0.0 or ag_type is None:
        return np.zeros(n_hours)
    if ag_type == 'pv':
        raw  = week['solar_mw'][hour_start : hour_start + n_hours]
        peak = week['solar_peak']
    else:
        raw  = week['wind_mw'][hour_start : hour_start + n_hours]
        peak = week['wind_peak']
    return (raw / peak * cap_kw).clip(0, cap_kw)


# ── True values for one day ───────────────────────────────────────────────────

def get_day(week: dict, d: int, agents: list) -> dict:
    """
    True executed values for day d (no noise).

    Solar/Wind output is hourly, held constant for 12 five-minute steps.
    Agents with no renewable get g = 0.
    """
    s5 = d * STEPS_PER_DAY
    s1 = d * 24

    rt_price = week['rt_lmp'][s5 : s5 + STEPS_PER_DAY]
    da_price = week['da_lmp'][s1 : s1 + 24]

    N   = len(agents)
    ren = np.zeros((N, STEPS_PER_DAY))
    for i, ag in enumerate(agents):
        ren_hourly = _agent_ren(week, ag.renewable_type, ag.renewable_cap_kw, s1, 24)
        ren[i] = np.repeat(ren_hourly, STEPS_PER_HOUR)

    return {'rt_price': rt_price, 'da_price': da_price, 'ren': ren}


# ── DAM forecast (24-hour noisy) ──────────────────────────────────────────────

def dam_forecast(week: dict, d: int, agents: list,
                 rng: np.random.Generator) -> dict:
    """Noisy 24-hour forecast used by the DAM ADMM solve."""
    s1     = d * 24
    da_fc  = np.maximum(
        week['da_lmp'][s1 : s1 + 24] + rng.normal(0, 16.0, 24), 0.0
    )
    N      = len(agents)
    ren_fc = np.zeros((N, 24))
    for i, ag in enumerate(agents):
        true_h = _agent_ren(week, ag.renewable_type, ag.renewable_cap_kw, s1, 24)
        if ag.renewable_cap_kw > 0.0:
            ren_fc[i] = np.clip(
                true_h + rng.normal(0, 0.20 * ag.renewable_cap_kw, 24),
                0.0, ag.renewable_cap_kw
            )
        # else: ren_fc[i] stays 0
    return {'da_price': da_fc, 'ren': ren_fc}


# ── RTM forecast (H-step noisy) ───────────────────────────────────────────────

def rt_forecast(t_day: int, week: dict, d: int, H: int,
                ag_type: str | None, cap_kw: float,
                rng: np.random.Generator) -> dict:
    """
    Noisy H-step forecast at 5-min step t_day within day d.

    RT price  : real future 5-min + N(0, 30 $/MWh)
    Renewable : real future hourly value + N(0, 20% of cap)
                Returns zeros for agents with cap_kw <= 0.
    """
    n_rt = week['n_days'] * STEPS_PER_DAY
    n_hr = week['n_days'] * 24

    # RT price: 1-hour-ahead forecast — much tighter than DA (σ=8 vs σ=16 $/MWh)
    g0   = d * STEPS_PER_DAY + t_day
    idxs = np.minimum(np.arange(g0, g0 + H), n_rt - 1)
    lam  = np.maximum(week['rt_lmp'][idxs] + rng.normal(0, 8.0, H), 0.0)

    # Renewable: 1-hour-ahead forecasts are significantly more accurate than DA.
    # PV is nearly deterministic 1h ahead (cloud cover); wind has more residual error.
    if cap_kw <= 0.0 or ag_type is None:
        ren = np.zeros(H)
    else:
        hr_idxs = np.minimum(idxs // STEPS_PER_HOUR, n_hr - 1)
        if ag_type == 'pv':
            raw_mw = week['solar_mw'][hr_idxs];  peak = week['solar_peak']
            sigma  = 0.05 * cap_kw   # 5% cap for PV (near-deterministic 1h ahead)
        else:
            raw_mw = week['wind_mw'][hr_idxs];   peak = week['wind_peak']
            sigma  = 0.08 * cap_kw   # 8% cap for wind (tighter but more variable)
        ren = np.clip(
            raw_mw / peak * cap_kw + rng.normal(0, sigma, H),
            0.0, cap_kw
        )
    return {'lam': lam, 'ren': ren}
