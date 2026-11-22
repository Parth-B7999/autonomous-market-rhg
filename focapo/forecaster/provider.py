"""
XGBForecaster — forecasts for the closed-loop simulation (--forecaster xgb).

Same interface role as small_scale_case_study/forecasting.py (dam_forecast,
rt_forecast) but with real forecasts trained on data before the test week
(forecaster/train.py).  Point forecasts are deterministic (no random draws);
`scenarios_*` sample past out-of-sample errors for stochastic optimisation.

Information timing in the simulation (step = 5 min, relative to week start):
  DAM for day d   : forecast of day d issued before it (features lagged >= 24 h)
  RTM at step t   : last realised RT price = step t-1; last metered hour = hour(t)-1
  commitment at t : same information as the RTM step t, target hour two hours on
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from data import load_hourly, load_rt5, annual_peaks          # noqa: E402
from features import rt_rows, ren_rt_rows, da_features, K_RT, M_REN  # noqa: E402
from train import Model, MODEL_DIR, train_week, _pos         # noqa: E402

STEPS_PER_DAY = 288


class XGBForecaster:
    def __init__(self, start_date: str, n_days: int, model_dir: Path | None = None):
        W = pd.Timestamp(start_date)
        mdir = Path(model_dir) if model_dir else MODEL_DIR / start_date
        if not (mdir / 'rt.json').exists():
            print(f'  forecaster: no models for {start_date}, training (data before {start_date}) …')
            train_week(start_date, n_days)
        M = {k: Model.load(mdir / f'{k}.json')
             for k in ('da', 'solar', 'wind', 'rt', 'solar_rt', 'wind_rt')}
        self.bank = dict(np.load(mdir / 'residuals.npz'))
        h, rt5 = load_hourly(), load_rt5()
        self.n_days, self.n_steps, self.n_hours = n_days, n_days * STEPS_PER_DAY, n_days * 24
        self.solar_peak, self.wind_peak = annual_peaks(W.year)

        # day-ahead forecasts (price, solar MW, wind MW) for every hour
        X = da_features(h).loc[W:W + pd.Timedelta(hours=self.n_hours - 1)]
        self.da = {}
        for c in ('da', 'solar', 'wind'):
            p = M[c].predict(X)
            self.da[c] = (p if c == 'da' else np.maximum(p, 0.0)).reshape(n_days, 24)

        # RT price: anchors a = -1 .. n_steps-2 (relative), horizons 1..K_RT
        p0 = _pos(rt5.index, W)
        anchors = p0 + np.arange(-1, self.n_steps - 1)
        Xr, _ = rt_rows(rt5, h['da'], anchors, np.arange(1, K_RT + 1))
        self.rt_price = M["rt"].predict(Xr).reshape(len(anchors), K_RT)

        # RT renewables (hourly MW): anchors j = -1 .. n_hours-2, horizons 1..M_REN
        self.ren_rt = {}
        for c in ('solar', 'wind'):
            s = h[c]
            q0 = _pos(s.index, W)
            an = q0 + np.arange(-1, self.n_hours - 1)
            Xs, _ = ren_rt_rows(s, an, np.arange(1, M_REN + 1))
            self.ren_rt[c] = np.maximum(M[f'{c}_rt'].predict(Xs), 0.0).reshape(len(an), M_REN)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _scale(self, mw: np.ndarray, ag_type, cap_kw: float) -> np.ndarray:
        if cap_kw <= 0.0 or ag_type is None:
            return np.zeros_like(mw, dtype=float)
        peak = self.solar_peak if ag_type == 'pv' else self.wind_peak
        return np.clip(mw / peak * cap_kw, 0.0, cap_kw)

    # ── interface used by simulation_sc ───────────────────────────────────────
    def dam(self, d: int, agents) -> dict:
        ren = np.zeros((len(agents), 24))
        for i, ag in enumerate(agents):
            if ag.renewable_cap_kw > 0.0 and ag.renewable_type is not None:
                mw = self.da['solar' if ag.renewable_type == 'pv' else 'wind'][d]
                ren[i] = self._scale(mw, ag.renewable_type, ag.renewable_cap_kw)
        return {'da_price': np.maximum(self.da['da'][d], 0.0), 'ren': ren}

    def rt(self, d: int, t_start: int, t_info: int, H: int, ag_type, cap_kw: float) -> dict:
        """Forecast for steps t_start..t_start+H-1 of day d, made at step t_info."""
        a = d * STEPS_PER_DAY + t_info - 1                    # last realised step (rel)
        steps = np.minimum(d * STEPS_PER_DAY + t_start + np.arange(H), self.n_steps - 1)
        k = np.clip(steps - a, 1, K_RT)
        lam = np.maximum(self.rt_price[a + 1, k - 1], 0.0)
        if cap_kw <= 0.0 or ag_type is None:
            ren = np.zeros(H)
        else:
            j = (d * STEPS_PER_DAY + t_info) // 12 - 1        # last metered hour (rel)
            m = np.clip(steps // 12 - j, 1, M_REN)
            mw = self.ren_rt['solar' if ag_type == 'pv' else 'wind'][j + 1, m - 1]
            ren = self._scale(mw, ag_type, cap_kw)
        return {'lam': lam, 'ren': ren}

    # ── scenarios from past out-of-sample errors (for stochastic optimisation) ─
    def dam_scenarios(self, d: int, agents, S: int, rng: np.random.Generator) -> dict:
        """S day-ahead scenarios: point forecast + errors of S randomly drawn
        past days (same day for price, solar and wind, keeping correlation).
        Returns da_price (S,24), ren (S,N,24), prob (S,)."""
        base = self.dam(d, agents)
        days = rng.integers(0, self.bank['da_da'].shape[0], S)
        price = np.maximum(base['da_price'][None, :] + self.bank['da_da'][days], 0.0)
        ren = np.zeros((S, len(agents), 24))
        for i, ag in enumerate(agents):
            if ag.renewable_cap_kw > 0.0 and ag.renewable_type is not None:
                c = 'solar' if ag.renewable_type == 'pv' else 'wind'
                mw = np.maximum(self.da[c][d][None, :] + self.bank[f'{c}_da'][days], 0.0)
                ren[:, i] = self._scale(mw, ag.renewable_type, ag.renewable_cap_kw)
        return {'da_price': price, 'ren': ren, 'prob': np.full(S, 1.0 / S)}

    def rt_scenarios(self, d: int, t_start: int, t_info: int, H: int, agents,
                     S: int, rng: np.random.Generator) -> dict:
        """S real-time scenarios for steps t_start..t_start+H-1 (made at t_info).
        Price errors come from a drawn past anchor (whole horizon, keeps the
        error path); renewable errors from a drawn past hour.
        Returns lam (S,H), ren (S,N,H), prob (S,)."""
        a = d * STEPS_PER_DAY + t_info - 1
        steps = np.minimum(d * STEPS_PER_DAY + t_start + np.arange(H), self.n_steps - 1)
        k = np.clip(steps - a, 1, K_RT)
        rows = rng.integers(0, self.bank['rt'].shape[0], S)
        lam_pt = self.rt_price[a + 1, k - 1]
        lam = np.maximum(lam_pt[None, :] + self.bank['rt'][rows][:, k - 1], 0.0)
        ren = np.zeros((S, len(agents), H))
        j = (d * STEPS_PER_DAY + t_info) // 12 - 1
        m = np.clip(steps // 12 - j, 1, M_REN)
        for i, ag in enumerate(agents):
            if ag.renewable_cap_kw > 0.0 and ag.renewable_type is not None:
                c = 'solar' if ag.renewable_type == 'pv' else 'wind'
                r = rng.integers(0, self.bank[f'{c}_rt'].shape[0], S)
                mw = np.maximum(self.ren_rt[c][j + 1, m - 1][None, :]
                                + self.bank[f'{c}_rt'][r][:, m - 1], 0.0)
                ren[:, i] = self._scale(mw, ag.renewable_type, ag.renewable_cap_kw)
        return {'lam': lam, 'ren': ren, 'prob': np.full(S, 1.0 / S)}
