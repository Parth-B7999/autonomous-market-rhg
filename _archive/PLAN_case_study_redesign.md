# mp-GNE Case Study Redesign Plan

> Drafted 2026-07-07, reframed same day (discussion with Claude). Working doc for the
> mp-GNE + FACET market-participation paper.

## 0. FRAMING — this paper is STANDALONE (Parth, 2026-07-07)

This is a **methodology paper**, NOT a follow-up to the FOCAPO paper. Do NOT frame the
case study as a comparison to FOCAPO results/fleet/economics.
- FOCAPO asks: "is coalition participation worth it?" (economics)
- This paper asks: "can the coalition GNE be computed with ZERO online iterations?" (method)
- Analogy: **explicit MPC : MPC :: this paper : market bidding.** Offline mpQP → CRs →
  FACET-combined GNE map → online PWA lookup.
- ADMM's ONLY role: internal iterative reference to (a) certify the explicit map returns
  the true equilibrium, (b) quantify what iteration-freeness buys. Not "FOCAPO results."
- The market setting just needs to be a credible, parametrically RICH instance whose
  θ(t) variation drives CR transitions all week.

## 1. Goal

Show the **effectiveness of multiparametric programming to make market participation
iteration-free**, and that **FACET makes the explicit GNE computation scale**. Utility
claims: zero online communication, µs bounded-time evaluation, determinism/
certifiability, on-board (embedded) strategy, offline scalability via FACET.

## 2. Decisions locked (2026-07-07)

| Decision | Choice |
|---|---|
| Data | **ERCOT** (LZ_WEST proposed — negative prices + scarcity spikes in one week) |
| Discreteness | **Always-in-market**: `{0}` branch eliminated by construction, no binaries, one map |
| Money figure | **Communication degradation** (packet loss / wall-clock budget on ADMM vs. flat mp-GNE) |
| RT problem class | Pure mpQP per agent (battery δ via exact convex relaxation, verified ex-post) |
| PJM increment rule (Eq. 11 FOCAPO) | Dropped — ERCOT SCED has no such rule (verify vs. ADER pilot terms) |

## 3. How L_min / L_max enter WITHOUT binaries

The FOCAPO binary `u` exists only for the `{0}` branch of `Σp ∈ {0} ∪ [L_min, L_max]`.
Three-step elimination:

1. **Rule binds the aggregate only** (coalition advantage, FOCAPO Eq. 10).
   Individuals keep `0 ≤ p_i ≤ p_i_max` — continuous.
2. **Kill the `{0}` branch by fleet design, not assumption.**
   Include ≥1 **grid-only committed agent** (no co-located renewable): its no-sell
   lower bound `lb_i = p_elec_min_i > 0` whenever committed, so `Σp = 0` is
   *infeasible given gate-locked commitments*. The DA stage (H₂ daily demand floor
   with penalized slack) already guarantees commitment — verify once per day.
   Design condition (check + report):

       0 < Σ_i lb_{i,k} < L_min      for all k

   Left inequality ⇒ exit branch vacuous (disjunction collapses to a box, exactly).
   Right inequality ⇒ L_min remains a genuine coupling constraint (the game:
   who covers the floor during renewable peaks). Knob: grid-only floors ≈ 40–60 kW
   vs. L_min = 100 kW.
3. **Coupling enters each agent's mpQP as parametric bounds** (affine in θ):

       L_min − σ_{-i,k}  ≤  p_{i,k}  ≤  L_max − σ_{-i,k},   σ_{-i,k} = Σ_{j≠i} p_{j,k}

   Rows of `A x ≤ b + F θ` — native mpQP form. Binaries → active-set combinatorics,
   which is exactly what FACET searches. (Machinery already exists:
   `make_pgne_bounds` / `pext_game` in `case_study.py`.)

**Honesty checks to report:** (a) ex-post `min_k Σp_k ≥ L_min` and `Σlb > 0` held;
(b) ex-post rationality — full exit would never have been collectively profitable
(H₂ shortfall penalty dominates even at spike prices); report any exceptions.

## 4. Case study concept: "West Texas hydrogen park"

Industrial H₂ park behind a single PCC, co-located wind/PV, daily H₂ offtake
contract. Motivation hook: **ERCOT ADER pilot** (aggregated small DERs; replaces
PJM's 100 kW rule as the participation-threshold narrative). Autonomy story: each
asset owner keeps cost structure private and carries its strategy on-board as a
precomputed PWA map — no central operator, no comm at dispatch time.

**Fleet (base case):** heterogeneous, incentive conflict flips with price:

| Agent | Role in the game |
|---|---|
| PEM electrolyzer (fast ramp, grid-only) | Grabs cheap windows; provides the L_min floor; first to retreat at spikes |
| ALK electrolyzer (slow ramp, baseload, grid-only) | Ramp limit forces anticipatory behavior → rich intertemporal CRs |
| Li-ion battery + PV | Arbitrage; competes for PCC headroom at negative prices |
| VRFB (long duration) | Slow arbitrage; SoC state shifts equilibrium across hours |

**Why LZ_WEST:** negative LMPs (wind curtailment) → everyone imports max → **L_max
binds** (congestion game). Scarcity spikes (ORDC adders) → everyone wants out but
floor + H₂ contract force `Σp ≥ L_min` → **L_min binds** (reverse congestion game).
Both faces of the coupling in one real week.

**Week selection:** volatility is needed so θ(t) actually crosses CR boundaries all
week (a flat-price week sits in one region → map/recurrence figures are boring).
One volatile LZ_WEST week may suffice; a calm week optional as contrast.

## 5. Result blocks (method-centric — the "explicit MPC evidence set")

- **A. The equilibrium map figure** — 2D slice of θ-space (e.g., LMP level × battery
  SoC) showing GNE region partition with the week's actual θ(t) trajectory overlaid
  walking through regions. The signature mp figure — no iterative method can draw it.
- **B. Combo recurrence** — over ~2,000 closed-loop 5-min solves, count DISTINCT GNE
  active-set combos vs. the combinatorial total. If a few dozen recur, that's the
  empirical heart: market operation lives on a small recurring set of equilibrium
  structures — exactly what FACET's neighbor graph exploits.
- **C. Timing, determinism, footprint** — bounded worst-case PWA lookup vs. ADMM's
  variable iteration count (distribution over the week); map storage in MB → strategy
  runs on-board, no comm link at dispatch. Comm-degradation sweep as the money figure
  (ADMM feasibility collapses with packet loss; mp-GNE flat). Optional: warm-started
  ADMM iterations spike with LMP volatility (verify mechanism first, §6.2).
- **D. FACET necessity (internal)** — Mode 1 exhaustive K^N combiner vs. Mode 3 FACET
  on the SAME instance: combos checked, offline time, N = 4 → 8 → 12 → 16. Scalability
  shown within the mp framework, no external baseline needed.
- **E. Equivalence certificate** — one table: mp-GNE vs. converged ADMM closed-loop
  trajectories, gap < ε. ADMM's only appearance as ground truth.
- **F. Honesty metrics** — θ-space coverage, fallback frequency, offline compute.

## 6. Known risks / de-risk first (week-0 spike, 2–3 days)

1. **θ-dimensionality with L_min re-added.** Current code DROPS L_min ("CR explosion",
   see `src/amrhg/solvers/simple_mpqp.py` header). Re-adding H coupling rows grows the
   active-set lattice. Mitigations to test on ONE battery agent first:
   - **Forecast-basis compression**: λ̂ = λ̄ + B·w with 2–3 PCA factors from historical
     ERCOT 5-min windows (level/slope/spike). Target θ ≈ 10–14 dims.
   - Map horizon H = 6 vs. 12: closed-loop profit loss check.
   - Record: CR count, PPOPT solve time at full vs. compressed θ.
2. **Warm-start-degradation mechanism** (Result B-i) — verify empirically in existing
   ADMM code that iteration counts actually spike with volatility BEFORE betting the
   money figure on it.
3. **ERCOT ADER pilot terms** — one-pass check that aggregation narrative + no-increment-
   rule claim hold.

## 7. Problems with the CURRENT case study (re-evaluated under §0 standalone framing)

1. **L_min dropped entirely** (`simple_mpqp.py`: "L_min is DROPPED — non-binding and
   causes CR explosion"). Method-internal reasons it must come back:
   (a) title claims *market participation* — L_min IS the market rule; with only
   Σp ≤ L_max a reviewer reads distributed MPC with a shared resource;
   (b) one-sided coupling → impoverished active-set structure; the L_min-binding
   regime is what enriches the CR map + combo-recurrence figures (Results A/B).
   Its CR cost is exactly what the week-0 spike (§6.1) must resolve. CENTRAL RISK.
2. **H₂ demand floor missing — now LOAD-BEARING.** The §3 no-binary argument requires
   the demand floor to force DA commitment of grid-only agents (kills the {0} branch).
   Without it, always-in-market degrades to an assumption. Energy-cost − H₂-revenue
   objective needed for instance credibility (current objective is pure imbalance
   tracking, p* = p_DA − dt·λ/γ).
3. **θ-tube narrowing — STRONGER problem under methodology framing** (coverage IS the
   claim), with one correction: the daily CR rebuild with DA fixed is a *defensible*
   design (post-gate-closure DA = known data, legitimate dimension reduction in the
   spirit of §6.1 compression). Real issue: tube on remaining params (prices, states)
   must cover realistic excursions; daily rebuild cost reported in offline budget.
4. **PJM data in `case_study.py`** despite README claiming ERCOT LZ_SOUTH; neither
   matches chosen LZ_WEST. Housekeeping.
5. **Export/selling — DEMOTED to open decision** (§8.5). "Inconsistent with FOCAPO
   no-sell" is not a valid objection under §0. Export does NOT break the no-binary
   construction (grid-only committed agent keeps p_i ≥ p_min > 0 regardless), and
   exporters dragging the aggregate down make L_min bind MORE often → richer game.
   Decide from ERCOT/ADER rules.

## 8. Open questions (discuss tomorrow)

1. Confirm warm-start-degradation check result (risk 2) before locking Result B-i.
2. Fleet size base case: 4 archetypes (current repo) vs. 6 (FOCAPO structure)?
3. L_max value so negative-price congestion story binds (med_scale dropped L_max — we
   need it). Tune vs. fleet capacity.
4. Which ERCOT weeks exactly (calm + scarcity)? Candidate: summer 2023 heat-wave week.
5. No-sell (p ≥ 0) — keep or allow export under ADER rules?
