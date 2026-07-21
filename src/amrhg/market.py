"""
market.py — MarketConfig registry so ONE codebase runs the case study on multiple
ISOs / RTM timescales without forking.

Each config freezes everything that differs between markets: the RTM granularity
(Δt, steps/day), the receding horizon H, the PCC band, the θ price box, and the fleet.
The rest of the pipeline (rhg_mpqp/dam/rhg_online/rhg_week) reads the ACTIVE config via
`rhg_mpqp.set_market(cfg)` — ERCOT is the default so the validated ERCOT path is unchanged.

Markets
-------
ERCOT : 15-min RTM, H=4 (1-h lookahead), 96 steps/day.  The validated baseline; these
        numbers reproduce the frozen `_validated_baseline/` spec exactly.
PJM   : 5-min RTM, H=6 (30-min lookahead), 288 steps/day, PJM-RTO zone.  The 5-min timescale
        is the scalability point (Δt=1/12 h, 288 RTM steps/day); H was set to 6 after the H=12
        offline solve proved intractable even with the parallel graph solver (per-agent mpQP
        n_x=24, ~109 constraints).  Renewable farms are down-sized vs ERCOT to hold the offline
        critical-region count down (see `fleet` note + the dry-run).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketConfig:
    name: str                    # "ercot" | "pjm" — also the offline-cache / data key
    # ── RTM timescale ──
    dt: float                    # hours per RTM step (0.25 ercot, 1/12 pjm)
    H: int                       # receding-horizon length in RTM steps
    steps_per_day: int           # RTM steps in a day (96 ercot, 288 pjm)
    intervals_per_hour: int      # RTM steps per clock hour (4 ercot, 12 pjm); h = t // this
    # ── economics / coupling (shared physical PCC) ──
    l_min: float                 # PCC import floor [kW]
    l_max: float                 # PCC import ceiling [kW]
    r_h2: float                  # H2 price [$/kg]
    eps_cv: float                # curtailment regularizer
    lam_box: tuple               # (lo, hi) θ-box on RTM λ [$/MWh]
    # ── fleet: tuple of (name, type, p_max, ren_cap, eta, gamma, d_max) ──
    fleet: tuple
    # ── data adapter ──
    data_kind: str               # "ercot" (data/ercot CSVs) | "pjm" (data/raw CSVs)
    solar_cap_mw: float          # system solar nameplate for CF normalisation (ercot only;
    wind_cap_mw: float           # pjm derives peak from gen_by_fuel — see data/pjm.py)


# ── ERCOT — the validated baseline (must match _validated_baseline / current rhg_mpqp) ──
_ERCOT_FLEET = (
    ("PEM_Elec",  "grid",  250.0,   0.0, 0.0200, 5e-3, 3.0),
    ("ALK",       "grid",  200.0,   0.0, 0.0180, 5e-3, 2.5),
    ("PEM_PV",    "solar", 150.0, 125.0, 0.0210, 5e-3, 1.9),
    ("PEM_PV_2",  "solar", 100.0, 125.0, 0.0190, 5e-3, 1.3),
    ("PEM_Wind",  "wind",  275.0, 250.0, 0.0195, 4e-3, 3.3),
    ("PEM_Wind_2","wind",  225.0, 250.0, 0.0185, 4e-3, 2.7),
)

ERCOT = MarketConfig(
    name="ercot", dt=0.25, H=4, steps_per_day=96, intervals_per_hour=4,
    l_min=100.0, l_max=900.0, r_h2=3.0, eps_cv=1e-3, lam_box=(-50.0, 300.0),
    fleet=_ERCOT_FLEET, data_kind="ercot", solar_cap_mw=32000.0, wind_cap_mw=40000.0,
)


# ── PJM — 5-min RTM, H=12.  Same 2-grid/2-solar/2-wind structure and PCC band as ERCOT;
#    renewable farms START smaller than ERCOT to hold the H=12 CR count down (curtailment
#    binds less when g is small).  ren_cap here is a DRY-RUN starting point — tuned against
#    the measured per-agent CR table before the full offline solve.  P_max/eta kept distinct
#    (6 distinct solves; see rhg_mpqp FLEET note).  Sum(P_max) held at 1200 kW vs L_max=900.
_PJM_FLEET = (
    ("PEM_Elec",  "grid",  250.0,   0.0, 0.0200, 5e-3, 3.0),
    ("ALK",       "grid",  200.0,   0.0, 0.0180, 5e-3, 2.5),
    ("PEM_PV",    "solar", 150.0,  90.0, 0.0210, 5e-3, 1.9),
    ("PEM_PV_2",  "solar", 100.0,  90.0, 0.0190, 5e-3, 1.3),
    ("PEM_Wind",  "wind",  275.0, 180.0, 0.0195, 4e-3, 3.3),
    ("PEM_Wind_2","wind",  225.0, 180.0, 0.0185, 4e-3, 2.7),
)

PJM = MarketConfig(
    name="pjm", dt=1.0 / 12.0, H=6, steps_per_day=288, intervals_per_hour=12,
    l_min=100.0, l_max=900.0, r_h2=3.0, eps_cv=1e-3, lam_box=(-50.0, 300.0),
    fleet=_PJM_FLEET, data_kind="pjm", solar_cap_mw=0.0, wind_cap_mw=0.0,
)


MARKETS: dict[str, MarketConfig] = {"ercot": ERCOT, "pjm": PJM}


def get_market(name: str) -> MarketConfig:
    try:
        return MARKETS[name.lower()]
    except KeyError:
        raise KeyError(f"unknown market '{name}'; known: {sorted(MARKETS)}")
