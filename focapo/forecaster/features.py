"""Features that only use information available when the forecast is made.

Day-ahead stage (issued on day d-1 for every hour of day d):
    calendar of the target hour; DA price 24 h and 168 h earlier (published
    the day before); mean DA price of day d-1; RT price, load, solar, wind
    48 h and 168 h earlier (day d-2 is fully metered).  No same-day actuals.

Real-time stage (5-min price, anchor a = last realised step, target a+k):
    last 3 realised 5-min prices, mean of the last hour, the DA price of the
    target hour (known), calendar of the target, horizon k = 1..K_RT.

Real-time renewables (hourly, anchor j = last metered hour, target j+m):
    value at j and j-1, value 24 h before the target, calendar, horizon m.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_RT = 26          # 5-min steps ahead (covers RTM horizon 12 and 65-min commitment)
M_REN = 4          # hours ahead for RT renewables

DA_FEATURES = ['hour', 'dow', 'month', 'weekend',
               'da_l24', 'da_l168', 'da_prevday_mean', 'rt_l48', 'rt_l168',
               'load_l48', 'load_l168', 'solar_l48', 'solar_l168', 'wind_l48', 'wind_l168']
RT_FEATURES = ['p1', 'p2', 'p3', 'p_mean12', 'da_target', 'hour', 'minute', 'dow', 'month', 'k']
REN_RT_FEATURES = ['v0', 'v1', 'v_l24', 'hour', 'month', 'm']


def da_features(h: pd.DataFrame) -> pd.DataFrame:
    """One row per hourly timestamp of h (target hour)."""
    X = pd.DataFrame(index=h.index)
    X['hour'] = h.index.hour
    X['dow'] = h.index.dayofweek
    X['month'] = h.index.month
    X['weekend'] = (h.index.dayofweek >= 5).astype(int)
    X['da_l24'] = h['da'].shift(24)
    X['da_l168'] = h['da'].shift(168)
    day_mean = h['da'].resample('D').mean()
    X['da_prevday_mean'] = (day_mean.shift(1)
                            .reindex(h.index.normalize()).to_numpy())
    for c in ('rt', 'load', 'solar', 'wind'):
        X[f'{c}_l48'] = h[c].shift(48)
        X[f'{c}_l168'] = h[c].shift(168)
    return X[DA_FEATURES]


def rt_rows(rt5: pd.Series, da_hourly: pd.Series, anchors: np.ndarray,
            ks: np.ndarray) -> pd.DataFrame:
    """Rows for (anchor index, horizon k) pairs; indices are positions in rt5."""
    v = rt5.to_numpy()
    idx = rt5.index
    A = np.repeat(anchors, len(ks))
    K = np.tile(ks, len(anchors))
    T = A + K
    csum = np.concatenate([[0.0], np.cumsum(v)])
    mean12 = (csum[A + 1] - csum[np.maximum(A - 11, 0)]) / np.minimum(A + 1, 12)
    t_ts = idx[np.minimum(T, len(idx) - 1)]
    da_t = da_hourly.reindex(t_ts.floor('h')).to_numpy()
    X = pd.DataFrame({
        'p1': v[A], 'p2': v[A - 1], 'p3': v[A - 2], 'p_mean12': mean12,
        'da_target': da_t, 'hour': t_ts.hour, 'minute': t_ts.minute,
        'dow': t_ts.dayofweek, 'month': t_ts.month, 'k': K,
    })
    return X[RT_FEATURES], T


def ren_rt_rows(s: pd.Series, anchors: np.ndarray, ms: np.ndarray):
    v = s.to_numpy()
    idx = s.index
    A = np.repeat(anchors, len(ms))
    M = np.tile(ms, len(anchors))
    T = A + M
    t_ts = idx[np.minimum(T, len(idx) - 1)]
    X = pd.DataFrame({'v0': v[A], 'v1': v[A - 1], 'v_l24': v[np.maximum(T - 24, 0)],
                      'hour': t_ts.hour, 'month': t_ts.month, 'm': M})
    return X[REN_RT_FEATURES], T
