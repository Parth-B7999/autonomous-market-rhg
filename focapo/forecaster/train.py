"""
Train the XGBoost forecaster for one test week, using ONLY data before it.

    python forecaster/train.py --week 2024-07-08 [--week 2025-06-23 ...]

For each week W (cutoff = W 00:00):
  * final models are fitted on all data with target time < W
  * residual banks for scenario generation come from "validation" models
    fitted on data < W - 28 d and evaluated on [W - 28 d, W)   (out-of-sample)
  * accuracy on the test week itself is written to accuracy.json, next to the
    accuracy the current Gaussian-noise forecast would have (0.798·sigma).
Outputs: forecaster/models/<W>/{da,solar,wind,rt,solar_rt,wind_rt}.json,
         residuals.npz, accuracy.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_hourly, load_rt5, annual_peaks          # noqa: E402
from features import (da_features, rt_rows, ren_rt_rows,     # noqa: E402
                      K_RT, M_REN)

MODEL_DIR = Path(__file__).resolve().parent / 'models'
VAL_DAYS = 28
PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, tree_method='hist', n_jobs=4, random_state=0)
RT_ANCHOR_STRIDE = 3          # subsample anchors for training (rows ~ 1/3)


# Models learn the CHANGE from a naive forecast (yesterday's DA price, last RT
# price, last metered renewable).  Trees cannot extrapolate price levels they
# never saw (e.g. a July heat wave after a winter-only history); a naive anchor
# keeps the forecast at least as good as persistence.
BASE = {'da': 'da_l24', 'rt': 'p1', 'solar_rt': 'v0', 'wind_rt': 'v0'}


class Model:
    def __init__(self, booster: xgb.Booster, base: str | None):
        self.booster, self.base = booster, base

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.booster.inplace_predict(X.to_numpy(dtype=float))
        return p + X[self.base].to_numpy(dtype=float) if self.base else p

    def save(self, path: Path) -> None:
        self.booster.save_model(str(path))
        json.dump({'base': self.base}, open(str(path) + '.meta', 'w'))

    @classmethod
    def load(cls, path: Path) -> 'Model':
        b = xgb.Booster()
        b.load_model(str(path))
        return cls(b, json.load(open(str(path) + '.meta'))['base'])


def _fit(X: pd.DataFrame, y: np.ndarray, base: str | None = None) -> Model:
    y = np.asarray(y, dtype=float)
    ok = X.notna().all(axis=1).to_numpy() & np.isfinite(y)
    if ok.sum() < 500:
        raise ValueError(f'only {ok.sum()} training rows before the cutoff: not enough history')
    target = y - X[base].to_numpy(dtype=float) if base else y
    # prices: robust (pseudo-Huber) loss — PJM spikes up to ~$2,400/MWh in the
    # training data otherwise dominate a squared loss.
    obj = 'reg:pseudohubererror' if base in ('da_l24', 'p1') else 'reg:squarederror'
    m = xgb.XGBRegressor(objective=obj, **PARAMS)
    m.fit(X[ok], target[ok])
    return Model(m.get_booster(), base)


# ── day-ahead models (price, solar MW, wind MW) ────────────────────────────────
def fit_da(h: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    X = da_features(h)
    train = X.index < cutoff
    return {c: _fit(X[train], h[c].to_numpy()[train], BASE.get(c))
            for c in ('da', 'solar', 'wind')}


def predict_da(models: dict, h: pd.DataFrame, start, end) -> pd.DataFrame:
    X = da_features(h).loc[start:end - pd.Timedelta(hours=1)]
    out = pd.DataFrame(index=X.index)
    for c, m in models.items():
        p = m.predict(X)
        out[c] = np.maximum(p, 0.0) if c != 'da' else p
    return out


# ── RT price model (5-min, anchor + horizon k) ─────────────────────────────────
def _pos(index: pd.DatetimeIndex, ts) -> int:
    return int(index.searchsorted(pd.Timestamp(ts)))


def fit_rt(rt5: pd.Series, da: pd.Series, cutoff, stride=RT_ANCHOR_STRIDE):
    lo, hi = 12, _pos(rt5.index, cutoff) - K_RT - 1
    anchors = np.arange(lo, hi, stride)
    X, T = rt_rows(rt5, da, anchors, np.arange(1, K_RT + 1))
    return _fit(X, rt5.to_numpy()[T], BASE['rt'])


def fit_ren_rt(s: pd.Series, cutoff, base='v0'):
    lo, hi = 24, _pos(s.index, cutoff) - M_REN - 1
    X, T = ren_rt_rows(s, np.arange(lo, hi), np.arange(1, M_REN + 1))
    return _fit(X, s.to_numpy()[T], base)


def fit_all(h, rt5, cutoff) -> dict:
    mods = fit_da(h, cutoff)
    mods['rt'] = fit_rt(rt5, h['da'], cutoff)
    mods['solar_rt'] = fit_ren_rt(h['solar'], cutoff)
    mods['wind_rt'] = fit_ren_rt(h['wind'], cutoff)
    return mods


# ── residuals on a window (out-of-sample errors) ───────────────────────────────
def residuals(mods, h, rt5, start, end) -> dict:
    """Errors (actual − forecast) on [start, end) for every model."""
    out = {}
    da_fc = predict_da({k: mods[k] for k in ('da', 'solar', 'wind')}, h, start, end)
    n_days = len(da_fc) // 24
    for c in ('da', 'solar', 'wind'):
        err = h[c].reindex(da_fc.index).to_numpy() - da_fc[c].to_numpy()
        out[f'{c}_da'] = err[:n_days * 24].reshape(n_days, 24)          # (days, 24)
    a0, a1 = _pos(rt5.index, start) - 1, _pos(rt5.index, end) - K_RT - 1
    anchors = np.arange(a0, a1)
    X, T = rt_rows(rt5, h['da'], anchors, np.arange(1, K_RT + 1))
    out['rt'] = (rt5.to_numpy()[T] - mods['rt'].predict(X)).reshape(len(anchors), K_RT)
    out['rt_anchor_ts'] = rt5.index[anchors].astype('int64')
    for c in ('solar', 'wind'):
        s = h[c]
        b0, b1 = _pos(s.index, start) - 1, _pos(s.index, end) - M_REN - 1
        an = np.arange(b0, b1)
        Xr, Tr = ren_rt_rows(s, an, np.arange(1, M_REN + 1))
        pr = np.maximum(mods[f'{c}_rt'].predict(Xr), 0.0)
        out[f'{c}_rt'] = (s.to_numpy()[Tr] - pr).reshape(len(an), M_REN)
    return out


def _mae(e):
    e = np.asarray(e, dtype=float)
    return float(np.nanmean(np.abs(e)))


def _persist(rt5, W, n_days, k):
    w = rt5.loc[W - pd.Timedelta(minutes=5 * k):W + pd.Timedelta(days=n_days) - pd.Timedelta(minutes=5)]
    v = w.to_numpy()
    return v[k:] - v[:-k]


def train_week(week: str, n_days: int = 7) -> dict:
    t0 = time.time()
    W = pd.Timestamp(week)
    h, rt5 = load_hourly(), load_rt5()
    hist_days = (W - h.index[0]).days
    out_dir = MODEL_DIR / week
    out_dir.mkdir(parents=True, exist_ok=True)

    # validation models → residual bank for scenarios
    v_cut = W - pd.Timedelta(days=VAL_DAYS)
    val = fit_all(h, rt5, v_cut)
    bank = residuals(val, h, rt5, v_cut, W)
    np.savez_compressed(out_dir / 'residuals.npz', **bank)

    # final models
    mods = fit_all(h, rt5, W)
    for k, m in mods.items():
        m.save(out_dir / f'{k}.json')

    # accuracy on the test week (forecast vs truth), and the noise baseline
    test = residuals(mods, h, rt5, W, W + pd.Timedelta(days=n_days))
    sp, wp = annual_peaks(W.year)
    acc = dict(
        week=week, history_days=hist_days,
        da_price_mae=_mae(test['da_da']), da_price_mae_noise=0.798 * 16.0,
        rt_price_mae_k1=_mae(test['rt'][:, 0]), rt_price_mae_k12=_mae(test['rt'][:, 11]),
        rt_price_mae_noise=0.798 * 8.0,
        solar_da_mae_pct_peak=100 * _mae(test['solar_da']) / sp,
        wind_da_mae_pct_peak=100 * _mae(test['wind_da']) / wp,
        ren_da_mae_pct_noise=100 * 0.798 * 0.20,
        solar_rt_mae_pct_peak=100 * _mae(test['solar_rt'][:, 0]) / sp,
        wind_rt_mae_pct_peak=100 * _mae(test['wind_rt'][:, 0]) / wp,
        solar_rt_mae_pct_noise=100 * 0.798 * 0.05, wind_rt_mae_pct_noise=100 * 0.798 * 0.08,
        da_price_mae_naive_lag24=_mae((h['da'] - h['da'].shift(24)).loc[W:W + pd.Timedelta(hours=24 * n_days - 1)]),
        rt_price_mae_persist_k1=_mae(_persist(rt5, W, n_days, 1)),
        rt_price_mae_persist_k12=_mae(_persist(rt5, W, n_days, 12)),
        bank_days=int(bank['da_da'].shape[0]),
        train_seconds=round(time.time() - t0, 1),
    )
    json.dump(acc, open(out_dir / 'accuracy.json', 'w'), indent=2)
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--week', action='append', required=True)
    ap.add_argument('--n-days', type=int, default=7)
    a = ap.parse_args(argv)
    for w in a.week:
        acc = train_week(w, a.n_days)
        print(json.dumps(acc))


if __name__ == '__main__':
    main()
