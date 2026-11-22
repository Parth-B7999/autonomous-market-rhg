"""PJM data loading for the forecaster (hourly table + 5-min RT prices).

Times are EPT wall-clock, like the simulation (duplicates at the autumn DST
change are dropped, keep first; the missing spring hour is interpolated).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

FMT = '%m/%d/%Y %I:%M:%S %p'
YEARS = (2024, 2025)


def data_dir() -> Path:
    env = os.environ.get('FOCAPO_DATA_DIR')
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / 'data' / 'raw').resolve()


def _read(stem: str, years, cols) -> pd.DataFrame:
    frames = []
    for y in years:
        f = data_dir() / f'{stem}_{y}.csv'
        if f.exists():
            frames.append(pd.read_csv(f, usecols=['datetime_beginning_ept'] + cols))
    df = pd.concat(frames, ignore_index=True)
    df['ts'] = pd.to_datetime(df['datetime_beginning_ept'], format=FMT)
    return df.drop(columns='datetime_beginning_ept')


def _regular(s: pd.Series, freq: str) -> pd.Series:
    s = s.sort_index()
    s = s[~s.index.duplicated(keep='first')]
    full = pd.date_range(s.index.min(), s.index.max(), freq=freq)
    return s.reindex(full).interpolate(limit=24).ffill().bfill()


@lru_cache(maxsize=2)
def load_rt5(years=YEARS) -> pd.Series:
    df = _read('rt_fivemin_mnt_lmps', years, ['total_lmp_rt'])
    s = df.set_index('ts')['total_lmp_rt'].astype(float)
    return _regular(s, '5min').rename('rt5')


@lru_cache(maxsize=2)
def load_hourly(years=YEARS) -> pd.DataFrame:
    da = _read('da_hrl_lmps', years, ['total_lmp_da']).set_index('ts')['total_lmp_da']
    da = _regular(da.astype(float), 'h')
    load = _read('hrl_load_metered', years, ['mw']).groupby('ts')['mw'].sum()
    load = _regular(load.astype(float), 'h')
    gen = _read('gen_by_fuel', years, ['fuel_type', 'mw'])
    solar = _regular(gen[gen.fuel_type == 'Solar'].groupby('ts')['mw'].sum().astype(float), 'h')
    wind = _regular(gen[gen.fuel_type == 'Wind'].groupby('ts')['mw'].sum().astype(float), 'h')
    rt = load_rt5(years).resample('h').mean()
    df = pd.concat({'da': da, 'rt': rt, 'load': load, 'solar': solar, 'wind': wind}, axis=1)
    end = min(df[c].last_valid_index() for c in df)
    return df.loc[:end].interpolate(limit=24).ffill()


def annual_peaks(year: int) -> tuple[float, float]:
    """Solar / wind annual peak [MW], as in forecasting.load_pjm_week."""
    gen = _read('gen_by_fuel', (year,), ['fuel_type', 'mw'])
    solar = gen[gen.fuel_type == 'Solar'].groupby('ts')['mw'].sum()
    wind = gen[gen.fuel_type == 'Wind'].groupby('ts')['mw'].sum()
    return float(solar.max()), float(wind.max())
