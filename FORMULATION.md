# FORMULATION — iteration-free RTM GNE, ERCOT case study (CURRENT)

> Updated 2026-07-18 to match the shipped code (`simple_game/rhg_*.py`, `dam.py`) and
> `report/rhg_detailed_report.tex` — now the **heterogeneous 6-distinct** fleet (all agents
> distinct in P_max/η). This supersedes the earlier small-scale / L_min-only and
> identical-pairs drafts (archived). Single source of truth for the H=4 receding-horizon design.

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
| 2 | PEM_PV | solar | 150 | 125 | 0.0210 | 5e-3 | 1.9 |
| 3 | PEM_PV_2 | solar | 100 | 125 | 0.0190 | 5e-3 | 1.3 |
| 4 | PEM_Wind | wind | 275 | 250 | 0.0195 | 4e-3 | 3.3 |
| 5 | PEM_Wind_2 | wind | 225 | 250 | 0.0185 | 4e-3 | 2.7 |

Total nameplate 1200 kW. r_H2 = $3/kg, H = 4 (15-min steps), Δt = 0.25 h, ε_cv = 1e-3.
All six agents are distinct in (P_max, η) → **6 distinct offline solves**. Distinct η strictly
orders a_i = r·η_i, breaking the permutation symmetry the earlier identical PV/wind pairs created
at the PCC ceiling (which "backs off first" now has a unique answer → no tied-equilibrium continuum).

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
- **DAM (once/day).** **Distributed ADMM** (`dam.solve_dam_admm`, ρ=0.1): each agent solves only its
  own 24-h QP (linear energy−H₂ objective + tiny γ_DA=1e-4 regulariser) with a hard **daily H₂ floor**
  `Σ_h η_i p_elec_{i,h} Δt ≥ D_day_i`, `D_day_i = 0.55·P_max_i·η_i·24`, and renewable balance. The
  **only** exchanged quantity is the per-hour coalition import (already public at the PCC meter); the
  band is enforced by a closed-form clip in the z-update. Output: hourly anchor `p_DA_{i,h}`.
  A centralized solve of the *same* regularised problem exists as an offline accuracy oracle only.
- **RTM (every 15 min, receding).** Paced window demand
  `D_i(t) = clip((D_day_i − h2_inv_i)·H/(steps_left_in_day), 0, D_max_i)` with
  `steps_left_in_day = max(96 − t, H)` (96 fifteen-min steps/day). Assemble nowcast θ_t;
  solve by **online FACET one-shot** (1 broadcast/step); apply step-0; execute vs realized.

## 7. Solver pipeline
```
per-agent mpQP in private θ_i (PPOPT combinatorial_parallel)   simple_game/rhg_mpqp.py, rhg_offline.py
  → expand to public p_gne slots
  → online: locate CR per agent at [s, θ] → block-linear equilibrium solve (eliminates s),
            selecting the min-potential point on the manifold when M_x is rank-deficient
            (= variational GNE = social optimum; gne_combiner._solve_equilibrium select="potential")
  → self-consistency iteration on the combination; facet-neighbour BFS if it fails
  → FACET neighbor-graph point-location (never enumerates K^N)  src/amrhg/solvers/facet_gne.py
```
**This is pure FACET.** Two selection add-ons were evaluated and are OFF by default, kept in
the tree behind env switches with their measurements recorded beside them:
- `AMRHG_MAX_NBR` (default 0) — 1-hop min-potential refinement. Measured with Gurobi at every
  one of 480 April steps, `max_nbr` the only variable: worst case **9.052 kW either way**,
  helps 11/480 steps, hurts 0; over two weeks it costs p95 **12.2 → 200 ms**. Not worth it.
- `AMRHG_HALL_EQ16` (default 0) — Hall & Bemporad Eq. (16) equal-multiplier certificate.
  Fires on ~15% of steps and selects a different manifold point than the potential
  minimiser; measured 2–4× further from centralized.

The gap to centralized is closed by the min-potential selection **inside the base solve**,
not by either add-on.
Centralized QP (`rhg_online.centralized`, Gurobi) = **offline accuracy oracle only**.

## 8. Acceptance gate (met)
`max |p_map − p_centralized| < 1e-3 kW` in the interior, zero lookup misses, single-valued.
**Validated 2026-07-21 (pure FACET, both add-ons off):** 1–7 Apr and 7–13 Jul 2025 (1344 RTM
steps) — interior match ~1e-4–1e-6 kW, 1.12 inter-agent rounds/step, **99.6% iteration-free**
(6/1344 warm-started ADMM fallbacks), H₂ 140% Apr / 141% Jul, **median step 3.9 ms, p95 12.2 ms**.

**The 9.05 kW worst case is equilibrium multiplicity, not solver error — proven, not asserted.**
At the worst step (2025-04-06, k=88) the coalition aggregate is pinned at L_min across all four
horizon steps, so M_x is maximally rank-deficient and the GNE set is a continuum. A direct
best-response test (each agent re-optimising against the others' fixed decisions) certifies that
**both** our point and the centralized point are exact GNEs: largest profitable unilateral
deviation 8e-15 (ours) and 6e-12 (centralized), max|dev| 0.000 kW for all six agents. They differ
in coalition potential (−28.264 vs −28.536), i.e. the oracle selects a different member of the
same equilibrium set.

No 1-hop selection rule can reach the oracle's member: min-potential minimises Φ over the null
space of ONE combination's manifold, and the centralized GNE lies in a different combination
(agents 1, 3, 5 in other CRs, none in the facet-neighbour shell). Closing this residual would
require a *global* min-Φ over combinations — precisely the K^N enumeration FACET exists to avoid.
It is therefore a **documented limit of the approach**, not an open defect.

## 9. Scope (out)
No batteries, no H₂ storage tanks, no ramp limits, no `{0}`-branch, no export — kept out to
remain a pure convex mpQP. DAM is an LP anchor generator, not an explicit map.
