# FORMULATION — iteration-free RTM GNE, ERCOT case study (CURRENT)

> Updated 2026-07-12 to match the shipped code (`simple_game/rhg_*.py`, `focapo_1day.py`)
> and `report/rhg_detailed_report.tex`. This supersedes the earlier small-scale / L_min-only
> draft (archived). Single source of truth for the H=4 receding-horizon design.

## 1. Thesis
The coalition GNE for real-time market bidding is precomputed **offline** as a per-agent
explicit multiparametric map. **Online RTM operation is iteration-free**: exactly **one
decision broadcast per agent per step**, resolved by a single local self-consistency solve on
the precomputed maps — no ADMM, no inter-agent negotiation rounds.

## 2. Fleet (medium-scale, N = 6) — `rhg_mpqp.FLEET`

| # | Agent | type | P_max [kW] | renew cap | η [kg/kWh] | γ | D_max [kg/window] |
|---|---|---|---|---|---|---|---|
| 0 | PEM_Elec | grid | 250 | – | 0.020 | 5e-3 | 3.0 |
| 1 | ALK | grid | 200 | – | 0.018 | 5e-3 | 2.5 |
| 2 | PEM_PV | PV | 125 | 125 | 0.020 | 5e-3 | 1.6 |
| 3 | PEM_PV_2 | PV | 125 | 125 | 0.020 | 5e-3 | 1.6 |
| 4 | PEM_Wind | wind | 250 | 250 | 0.020 | 4e-3 | 3.0 |
| 5 | PEM_Wind_2 | wind | 250 | 250 | 0.020 | 4e-3 | 3.0 |

Total nameplate 1200 kW. r_H2 = $3/kg, H = 4 (15-min steps), Δt = 0.25 h, ε_cv = 1e-3.
Agents 2–3 and 4–5 are identical pairs → **4 distinct offline solves**.

## 3. Coupling (both faces, per 15-min step k)
`L_min = 100 kW ≤ Σ_i p_{i,k} ≤ L_max = 900 kW`.
L_min = ISO minimum offer; L_max = 75% of 1200 kW interconnection limit.
(Both faces are active — the earlier L_min-only draft is superseded.)

## 4. Per-agent RTM game — strictly-convex mpQP (`rhg_mpqp.build_agent_matrices`)
Agent i, horizon k = 0…3:
```
min Σ_k [ ½ γ_i (p_k − p_DA_i)²  + (λ_k/1000 − r·η_i)·Δt·p_k ]   (+ ½ε cv_k² + r·η_i·Δt·cv_k for renewables)
s.t. 0 ≤ p_k ≤ P_max_i                                          (capacity box)
     renewables: p_elec_k = p_k + g_k − cv_k,  0 ≤ cv_k ≤ g_k,  0 ≤ p_elec_k ≤ P_max_i
     Σ_k η_i·p_elec_k·Δt ≥ D_i                                   (cumulative H₂ — couples the horizon)
     L_min ≤ p_k + s_k ≤ L_max,  s_k = Σ_{j≠i} p_{j,k}           (coupling; grid import p only)
```
The `−γ p_DA p` cross term is kept (anchor); the constant `½γ p_DA²` drops from the argmin.

## 5. θ — parameters (LOCKED)
Private θ_i = [ `s`(H) | `D_i`, `λ_RT`(H), `p_DA_i`, (`g`(H)) ], where `s = sum_x_neg` is the
coupling aggregate, **eliminated online via M_x⁻¹** (does not appear in the final map).

| Agent type | private p | + s(4) | **θ_i (PPOPT)** |
|---|---|---|---|
| grid (0,1) | 6 (D+λ+p_DA) | 4 | **10** |
| renewable (2–5) | 10 (D+λ+p_DA+g) | 4 | **14** |

Public union p_gne = [D(6) | λ(4) | p_DA(6) | g_solar(4) | g_wind(4)] = **24** (never solved
directly; the K^N ≈ 9.7e20 combinations are never enumerated — FACET walks a neighbor graph).

## 6. Two-settlement closed loop (`rhg_week.py`, DAM in `focapo_1day.py`)
- **DAM (offline, once/day).** Centralized 24-h social-optimum **LP** (SLSQP, γ=0): linear
  energy−H₂ objective, hard **daily H₂ floor** `Σ_h η_i p_elec_{i,h} Δt ≥ D_day_i` with
  `D_day_i = 0.55·P_max_i·η_i·24`, hourly coupling band, renewable balance → hourly anchor
  `p_DA_{i,h}`. **Not an explicit map, not ADMM.**
- **RTM (every 15 min, receding).** Paced window demand
  `D_i(t) = clip((D_day_i − h2_inv_i)·H/(steps_left_in_day), 0, D_max_i)` with
  `steps_left_in_day = max(96 − t, H)` (96 fifteen-min steps/day). Assemble nowcast θ_t;
  solve by **online FACET one-shot** (1 broadcast/step); apply step-0; execute vs realized.

## 7. Solver pipeline
```
per-agent mpQP in private θ_i (PPOPT combinatorial_parallel)   simple_game/rhg_mpqp.py, rhg_offline.py
  → expand to public p_gne slots
  → online: locate CR per agent at [s, θ] → block-linear equilibrium solve (eliminates s)
  → 1-hop variational refinement (min-potential feasible = variational GNE = social optimum)
  → FACET neighbor-graph point-location (never enumerates K^N)  src/amrhg/solvers/facet_gne.py
```
Centralized QP (`rhg_online.centralized`, Gurobi) = **offline accuracy oracle only**.

## 8. Acceptance gate (met)
`max |p_map − p_centralized| < 1e-3 kW`, zero lookup misses, single-valued. **Validated:**
1–7 Apr and 7–13 Jul 2025 — ≤ 2.9e-4 kW, 0 misses, H₂ 119–127%, exactly 1 broadcast/step.

## 9. Scope (out)
No batteries, no H₂ storage tanks, no ramp limits, no `{0}`-branch, no export — kept out to
remain a pure convex mpQP. DAM is an LP anchor generator, not an explicit map.
