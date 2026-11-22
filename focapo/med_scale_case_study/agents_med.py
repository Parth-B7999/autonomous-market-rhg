"""
agents_med.py — Medium-scale (125–250 kW) electrolyzer fleet definitions.

5× electrolyzer/renewable scaling from the original admm_elz_market fleet.
Battery sizing uses industry-standard C-rates and durations (reviewed, not
blind 5× scaling).

Key property: every agent's P_max ≥ 125 kW > L_min = 100 kW, so each CAN
participate in the wholesale market individually.

No L_max constraint in this case study (removed per advisor feedback).
L_min = 100 kW applies to both scenarios:
  - Individual: p_i ≥ L_min at every step (each agent alone)
  - GNE:        Σ p_i ≥ L_min at every step (aggregate)
"""
from __future__ import annotations
import sys
from pathlib import Path

# Import ElzAgent from the original project.  IMPORTANT: keep THIS folder ahead
# of the parent on sys.path — the parent also defines simulation_sc.py and
# miqp_admm.py, and inserting the parent at position 0 would silently shadow the
# med-local versions (the case study would run the parent's code instead).
_HERE   = str(Path(__file__).resolve().parent)
_PARENT = str(Path(__file__).resolve().parent.parent / 'admm_elz_market')
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from agents import ElzAgent  # noqa: E402

# ── Market parameters ─────────────────────────────────────────────────────────
L_MIN     = 100.0    # [kW]  minimum market participation (per-agent or aggregate)
H2_PRICE  = 3.0      # [$/kg] hydrogen wholesale price

RHO_RT    = 0.05     # ADMM penalty — real-time market
RHO_DA    = 1e-4     # ADMM penalty — day-ahead market
TOL_RT    = 1.0      # [kW]  RTM convergence tolerance
TOL_DA    = 1.0      # [kW]  DAM convergence tolerance

N_DAYS     = 7
START_DATE = '2024-07-08'
SEED       = 42

# Weekday/weekend H₂ demand multiplier (Mon–Sun)
H2_DEMAND_PROFILE = [1.00, 1.00, 1.00, 1.00, 0.90, 0.65, 0.45]


# ── Fleet parameters — medium-scale agents ────────────────────────────────────
# Electrolyzer & renewable: 5× scaling from original fleet.
# Battery: reviewed sizing with industry-standard durations.
#
# Original → Medium scaling:
#   PEM_Elec:      50 kW  →  250 kW
#   PEM_PV:        25 kW  →  125 kW (PV: 25→125 kW)
#   PEM_Wind:      50 kW  →  250 kW (Wind: 50→250 kW)
#   ALK_Bat:       40 kW  →  200 kW (Bat: 30 kWh/7.5 kW → 200 kWh/50 kW, 4h duration)
#   PEM_PV_Bat:    25 kW  →  125 kW (Bat: 15 kWh/7.5 kW → 100 kWh/50 kW, 2h duration)
#   PEM_Wind_Bat:  50 kW  →  250 kW (Bat: 30 kWh/15 kW → 200 kWh/75 kW, 2.7h duration)

A1 = dict(
    name              = "PEM_Elec_M",
    renewable_type    = None,
    p_elec_max_kw     = 250.0,
    p_elec_min_kw     = 0.0,    # min-loading removed: electrolyzer modulates continuously 0..P_max
    renewable_cap_kw  = 0.0,
    eta_kg_per_kwh    = 0.020,
    gamma             = 5e-3,
    r_h2              = H2_PRICE,
)

A2 = dict(
    name              = "PEM_PV_M",
    renewable_type    = "pv",
    p_elec_max_kw     = 125.0,
    p_elec_min_kw     = 0.0,    # min-loading removed: electrolyzer modulates continuously 0..P_max
    renewable_cap_kw  = 125.0,
    eta_kg_per_kwh    = 0.020,
    gamma             = 5e-3,
    r_h2              = H2_PRICE,
)

A3 = dict(
    name              = "PEM_Wind_M",
    renewable_type    = "wind",
    p_elec_max_kw     = 250.0,
    p_elec_min_kw     = 0.0,    # min-loading removed: electrolyzer modulates continuously 0..P_max
    renewable_cap_kw  = 250.0,
    eta_kg_per_kwh    = 0.020,
    gamma             = 4e-3,
    r_h2              = H2_PRICE,
)

A4 = dict(
    name              = "ALK_Bat_M",
    renewable_type    = None,
    p_elec_max_kw     = 200.0,
    p_elec_min_kw     = 0.0,    # min-loading removed: electrolyzer modulates continuously 0..P_max
    renewable_cap_kw  = 0.0,
    eta_kg_per_kwh    = 0.018,
    gamma             = 5e-3,
    r_h2              = H2_PRICE,
    bat_cap_kwh       = 200.0,       # 4h duration (200 kWh / 50 kW)
    bat_p_ch_max_kw   = 50.0,        # 0.25C rate
    bat_p_dis_max_kw  = 50.0,
)

A5 = dict(
    name              = "PEM_PV_Bat_M",
    renewable_type    = "pv",
    p_elec_max_kw     = 125.0,
    p_elec_min_kw     = 0.0,    # min-loading removed: electrolyzer modulates continuously 0..P_max
    renewable_cap_kw  = 125.0,
    eta_kg_per_kwh    = 0.020,
    gamma             = 5e-3,
    r_h2              = H2_PRICE,
    bat_cap_kwh       = 100.0,       # 2h duration (100 kWh / 50 kW)
    bat_p_ch_max_kw   = 50.0,        # 0.5C rate
    bat_p_dis_max_kw  = 50.0,
)

A6 = dict(
    name              = "PEM_Wind_Bat_M",
    renewable_type    = "wind",
    p_elec_max_kw     = 250.0,
    p_elec_min_kw     = 0.0,    # min-loading removed: electrolyzer modulates continuously 0..P_max
    renewable_cap_kw  = 250.0,
    eta_kg_per_kwh    = 0.020,
    gamma             = 4e-3,
    r_h2              = H2_PRICE,
    bat_cap_kwh       = 200.0,       # 2.67h duration (200 kWh / 75 kW)
    bat_p_ch_max_kw   = 75.0,        # 0.375C rate
    bat_p_dis_max_kw  = 75.0,
)

FLEET_DICTS = [A1, A2, A3, A4, A5, A6]


def build_med_fleet() -> list[ElzAgent]:
    """Build the medium-scale 6-agent fleet."""
    return [ElzAgent(h2_target_kg_day=0.0, **d) for d in FLEET_DICTS]
