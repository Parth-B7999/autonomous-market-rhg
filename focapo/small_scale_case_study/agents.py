"""
agents.py — Electrolyzer agent definitions for the ADMM green hydrogen market study.

Fleet: 6 agents across four configuration types:
  1. Electrolyzer only              (PEM_Elec)
  2. Renewable + Electrolyzer       (PEM_PV, PEM_Wind)
  3. Battery + Electrolyzer         (ALK_Bat)
  4. Battery + Renewable + Elec     (PEM_PV_Bat, PEM_Wind_Bat)

No selling: net grid import p_i,k >= 0 at every time step.
Battery cannot sell to grid (enforced by p_i,k >= 0 no-sell constraint).

Notation
--------
  p_i,k       : net grid import [kW], decision variable
  g_i,k       : renewable output [kW], parameter (forecast)
  p_ch_i,k    : battery charging power [kW], decision variable (0 if no battery)
  p_dis_i,k   : battery discharging power [kW], decision variable (0 if no battery)
  p_elec_i,k  : p_i,k + g_i,k + p_dis_i,k - p_ch_i,k  [kW], total electrolyzer power

Local constraints (per step k, given renewable forecast g_i,k):
  Non-battery agents:
    lb_i,k = max(0, p_elec_min_i - g_i,k)   [no-sell + min loading]
    ub_i,k = max(0, p_elec_max_i - g_i,k)   [capacity headroom]
  Battery agents (loose — Gurobi enforces tighter via coupled constraints):
    lb_i,k = 0
    ub_i,k = max(0, p_elec_max_i + bat_p_ch_max_i - g_i,k)

Coupling constraint (bilateral, per step k):
  L_MIN <= sum_i p_i,k <= L_MAX

Battery SoC dynamics:
  E_{k+1} = E_k + bat_eta_ch * p_ch_k * dt - p_dis_k / bat_eta_dis * dt
  0 <= E_k <= bat_cap_kwh
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

# ── Market constants ──────────────────────────────────────────────────────────
L_MAX_KW: float     = 200.0    # PCC interconnection capacity [kW]
L_MIN_KW: float     = 100.0    # minimum aggregate import commitment [kW]
H_RT: int           = 12       # RT lookahead steps (1 hr at 5-min resolution)
H_DA: int           = 24       # DA horizon (hours)
DT_RT: float        = 5 / 60   # RT step size [hr]
DT_DA: float        = 1.0      # DA step size [hr]
H2_PRICE: float     = 3.0      # H2 revenue [$/kg]
STEPS_PER_DAY: int  = 288      # 5-min steps per day


@dataclass
class ElzAgent:
    """
    Electrolyzer agent, optionally with collocated renewable and/or battery.

    Mandatory
    ---------
    name, p_elec_max_kw, p_elec_min_kw, eta_kg_per_kwh, h2_target_kg_day, gamma

    Optional renewable (set renewable_cap_kw=0 for no renewable)
    -------------------
    renewable_type : 'pv' or 'wind'  (ignored if renewable_cap_kw == 0)
    renewable_cap_kw : nameplate capacity [kW]

    Optional battery (set bat_cap_kwh=0 for no battery)
    ----------------
    bat_cap_kwh      : energy capacity [kWh]
    bat_p_ch_max_kw  : max charge rate [kW]
    bat_p_dis_max_kw : max discharge rate [kW]
    bat_eta_ch       : round-trip charge efficiency
    bat_eta_dis      : round-trip discharge efficiency
    bat_soc_init_frac: initial SoC as fraction of capacity
    """
    name:               str
    renewable_type:     Optional[Literal['pv', 'wind']]
    p_elec_max_kw:      float
    p_elec_min_kw:      float
    renewable_cap_kw:   float
    eta_kg_per_kwh:     float
    h2_target_kg_day:   float
    gamma:              float
    r_h2:               float = field(default=H2_PRICE)

    # Battery (optional)
    bat_cap_kwh:        float = field(default=0.0)
    bat_p_ch_max_kw:    float = field(default=0.0)
    bat_p_dis_max_kw:   float = field(default=0.0)
    bat_eta_ch:         float = field(default=0.95)
    bat_eta_dis:        float = field(default=0.95)
    bat_soc_init_frac:  float = field(default=0.50)

    # Physical constraints
    c_startup:          float = field(default=40.0)
    ramp_rate_kw_per_hr:float = field(default=-1.0) # -1 means uninitialized

    # Runtime state — reset at the start of each simulated day
    h2_inv_kg:   float = field(default=0.0, init=False, repr=False)
    bat_soc_kwh: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        if self.ramp_rate_kw_per_hr < 0:
            # Default to 50% of P_max per 5-minute step -> 6 * P_max per hour
            self.ramp_rate_kw_per_hr = 6.0 * self.p_elec_max_kw

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def has_battery(self) -> bool:
        return self.bat_cap_kwh > 0.0

    @property
    def has_renewable(self) -> bool:
        return self.renewable_cap_kw > 0.0

    def reset(self) -> None:
        """Reset per-day state."""
        self.h2_inv_kg = 0.0
        self.bat_soc_kwh = self.bat_soc_init_frac * self.bat_cap_kwh

    # ── Constraint helpers ────────────────────────────────────────────────────

    def bounds(self, g_ren: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-step box bounds for net import p_i given renewable output g_ren.

        Non-battery agents:
          lb_k = max(0, p_elec_min - g_ren_k)   exact tighter bounds
          ub_k = max(0, p_elec_max - g_ren_k)

        Battery agents:
          lb_k = 0                               Gurobi enforces electrolyzer floor
          ub_k = max(0, p_elec_max + p_ch_max - g_ren_k)  loose upper bound
        """
        if self.has_battery:
            lb = np.zeros(len(g_ren))
            ub = np.maximum(0.0, self.p_elec_max_kw + self.bat_p_ch_max_kw - g_ren)
        else:
            lb = np.maximum(0.0, self.p_elec_min_kw - g_ren)
            ub = np.maximum(0.0, self.p_elec_max_kw - g_ren)
        return lb, ub

    # ── Cost helpers ──────────────────────────────────────────────────────────

    def linear_cost(self, lam: np.ndarray, dt: float) -> np.ndarray:
        """
        Per-step linear cost coefficient c_k [$/kW] for net grid import.

        c_k = (lambda_k / 1000  -  r_h2 * eta) * dt

        Interpretation: cost to buy 1 kW for dt hours minus H2 revenue from that kW.
        c_k < 0 when H2 revenue exceeds electricity cost (agent profitable).

        For battery agents this is only the portion attributable to grid import.
        Battery H2 revenue (from discharge) is added in the Gurobi objective.
        """
        return (lam / 1000.0 - self.r_h2 * self.eta_kg_per_kwh) * dt

    def h2_production(self, p_net: np.ndarray, g_ren: np.ndarray,
                      p_dis: np.ndarray, p_ch: np.ndarray, dt: float) -> np.ndarray:
        """H2 produced per step [kg].  p_elec = p_net + g_ren + p_dis - p_ch."""
        p_elec = p_net + g_ren + p_dis - p_ch
        return np.maximum(p_elec, 0.0) * self.eta_kg_per_kwh * dt


# ── Default 6-agent fleet ─────────────────────────────────────────────────────

def build_fleet() -> list[ElzAgent]:
    """
    Six-agent electrolyzer fleet covering all four configuration types:

    Type A — Electrolyzer only    : PEM_Elec
    Type B — Renewable + Elec     : PEM_PV, PEM_Wind
    Type C — Battery + Elec       : ALK_Bat
    Type D — Bat + Renewable + El : PEM_PV_Bat, PEM_Wind_Bat

    H2 targets at 75% of maximum possible daily production.
    Battery initial SoC = 50% of capacity.
    Battery can charge from grid + renewables (unrestricted).
    """
    return [
        # ── Type A: Electrolyzer only (larger elec — no renewable capex) ────────
        ElzAgent(
            name              = "PEM_Elec",
            renewable_type    = None,
            p_elec_max_kw     = 50.0,
            p_elec_min_kw     = 0.0,
            renewable_cap_kw  = 0.0,
            eta_kg_per_kwh    = 0.020,
            h2_target_kg_day  = 0.0,
            gamma             = 5e-3,
        ),
        # ── Type B: Renewable + Electrolyzer (elec ≈ renewable nameplate) ────
        ElzAgent(
            name              = "PEM_PV",
            renewable_type    = "pv",
            p_elec_max_kw     = 25.0,
            p_elec_min_kw     = 0.0,
            renewable_cap_kw  = 25.0,
            eta_kg_per_kwh    = 0.020,
            h2_target_kg_day  = 0.0,
            gamma             = 5e-3,
        ),
        ElzAgent(
            name              = "PEM_Wind",
            renewable_type    = "wind",
            p_elec_max_kw     = 50.0,
            p_elec_min_kw     = 0.0,
            renewable_cap_kw  = 50.0,
            eta_kg_per_kwh    = 0.020,
            h2_target_kg_day  = 0.0,
            gamma             = 4e-3,
        ),
        # ── Type C: Battery + Electrolyzer (larger elec — no renewable capex) ─
        ElzAgent(
            name              = "ALK_Bat",
            renewable_type    = None,
            p_elec_max_kw     = 40.0,
            p_elec_min_kw     = 0.0,
            renewable_cap_kw  = 0.0,
            eta_kg_per_kwh    = 0.018,
            h2_target_kg_day  = 0.0,
            gamma             = 5e-3,
            bat_cap_kwh       = 30.0,
            bat_p_ch_max_kw   =  7.5,
            bat_p_dis_max_kw  =  7.5,
        ),
        # ── Type D: Bat + Renewable + Electrolyzer (elec ≈ renewable) ─────────
        ElzAgent(
            name              = "PEM_PV_Bat",
            renewable_type    = "pv",
            p_elec_max_kw     = 25.0,
            p_elec_min_kw     = 0.0,
            renewable_cap_kw  = 25.0,
            eta_kg_per_kwh    = 0.020,
            h2_target_kg_day  = 0.0,
            gamma             = 5e-3,
            bat_cap_kwh       = 15.0,
            bat_p_ch_max_kw   =  7.5,
            bat_p_dis_max_kw  =  7.5,
        ),
        ElzAgent(
            name              = "PEM_Wind_Bat",
            renewable_type    = "wind",
            p_elec_max_kw     = 50.0,
            p_elec_min_kw     = 0.0,
            renewable_cap_kw  = 50.0,
            eta_kg_per_kwh    = 0.020,
            h2_target_kg_day  = 0.0,
            gamma             = 4e-3,
            bat_cap_kwh       = 30.0,
            bat_p_ch_max_kw   = 15.0,
            bat_p_dis_max_kw  = 15.0,
        ),
    ]
