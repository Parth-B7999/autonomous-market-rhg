# DIAGNOSIS — the mp-GNE reference method under-computes the variational GNE on our L_min game

> Parth + Claude, 2026-07-11. Coding paused pending method decision. NO solver code
> was changed. Findings come from running the **reference `mpgne_ppopt/mpgne`** pipeline
> unmodified on the locked v0 formulation. Ground truth = central potential-QP mpQP.

## TL;DR

On the locked small-scale ERCOT v0 game (2 grid electrolyzers, DA anchor, **L_min-only
coupling**, n_p=3), the reference method (`solve_all_agents_mp` → `build_gne_solution`
→ `select_v_gne`) **does not return the variational GNE at ~15% of the θ box**. Two
independent, confirmed causes:

1. **The combiner under-enumerates GNE.** At some θ the true variational equilibrium is
   a valid Nash point that **no critical region covers** — it is never formed.
2. **v-GNE selection ignores the boxes.** On genuine infinite-equilibria CRs,
   `select_v_gne` solves the equal-multiplier condition *unconstrained* and returns
   points **outside** the agents' capacity boxes.

The central potential-QP built as one mpQP gives the correct answer everywhere (7 CRs,
0 gaps, exact to 1.4e-14 over 400 θ). So the *answers* are known; the reference
*combiner+selector* just doesn't produce them for this geometry.

## The reproducer (game)

`simple_game/rtm_v0_game.py` — 2 grid-only electrolyzers, buy-only, H=1:

```
min_{p_i}  ½ γ_i (p_i − p_DA_i)²  +  dt·(λ/1000 − r_H2·η_i)·p_i
s.t.       0 ≤ p_i ≤ p_max_i
coupling:  Σ_i p_i ≥ L_min = 60          (L_min-only; L_max dropped, per the lock)
θ = [λ, p_DA_0, p_DA_1],  n_p = 3
PEM: p_max 50, η 0.020, γ 5e-3, a=60 $/MWh ; ALK: p_max 40, η 0.018, γ 5e-3, a=54
dt = 0.25, r_H2 = 3, λ ∈ [−50,150], p_DA_i ∈ [0, p_max_i]
```

Coupling encoded for the reference (single upper-form row): `C_i = [[−1]]`, `d = [−60]`.
Run with the REFERENCE package: `sys.path.insert(0, mpgne_ppopt)`; import from `mpgne.*`.

Reference output: per-agent CRs = **[3, 3]** → combiner **9 GNE CRs (8 unique + 1 infinite)**.

## Failure 1 — combiner misses the variational GNE (under-enumeration)

θ = **[137, 40.8, 0.1]** (λ dear, anchors p_DA=[40.8, 0.1]).

- Reference `locate_all(θ)` → **only CR6** (unique, combo (2,0)) → x = **[36.95, 23.05]**.
- I verified **[36.95, 23.05] is a valid Nash** (each agent best-responds; both pinned on
  their own coupling bound — a *corner* GNE).
- The **variational** GNE is **[50, 10]** (PEM at its box max 50, ALK pinned at 10; lower
  potential; = centralized). I verified **[50,10] is also a valid Nash**.
- **No CR in the reference solution covers [50,10].** The combiner never formed it.

Cause: the variational point sits at a **degenerate vertex** where PEM has *both* its box
(p0 = 50) *and* the coupling (Σp = 60) active. The per-agent mpQP produces only 3 CRs and
does not separate this box+coupling vertex into its own active-set combo, so the combiner
cannot assemble it.

## Failure 2 — v-GNE selection violates the boxes (unconstrained eq. 17)

θ = **[131.3, 0.7, 24.1]**. Covering CRs = [3 (infinite), 5 (unique)].

- CR3 is the genuine infinite-equilibria segment Σp = 60. `select_v_gne` → **[18.45, 41.55]**
  — but **p1 = 41.55 > ALK box 40**. Infeasible.
- True v-GNE = **[20, 40]** = that segment's potential-min **clipped to the box**.

Cause: `gne_selector._v_gne_y2` solves the equal-multiplier system with **unconstrained
`np.linalg.lstsq`** (mpgne_ppopt/mpgne/gne_selector.py:261). The paper's eq. (17) is a
**constrained** mpQP over CR_k (subject to `C_{-i}x + D_i p ≤ e`, i.e. the boxes). The
reference implements only the unconstrained special case, correct only when the
equal-multiplier point is interior.

## Ground truth (for reference, NOT proposed as the method)

Central potential-QP as one mpQP (`min ½xᵀQx+(c+Fθ)ᵀx s.t. Gx≤w0+Wθ`, PPOPT): **7 CRs,
0 misses over 400 θ, max err 1.4e-14 kW** vs SLSQP centralized. Confirms [50,10] and
[20,40] are the correct v-GNE. (This is the *definition* of the v-GNE for a potential
game — it does not scale, which is why FACET/combiner exists; it is used here only to
certify the true answers.)

## Why v0–v2 (amrhg) "passed" before

Those used **two-sided L_min≤Σp≤L_max** coupling and fleets tuned so the binding geometry
never produced a degenerate box+coupling vertex or a box-clipped infinite segment. The
reference itself was validated on the battery-charging and two-mass-spring games (paper §III),
which likewise avoid this geometry. Our **L_min-only + tight electrolyzer boxes** is the
first instance that hits it.

## The decision (for Parth)

The reference method needs work in **two** places to be correct here — the selector fix
alone is insufficient because the combiner has nothing to select from at Failure-1 θ:

- **A. Fix completeness at the source.** Make the per-agent mpQP separate the degenerate
  box+coupling active set (so the variational combo gets formed), then implement the
  box-constrained eq. (17) selection. Most faithful to the paper; unknown effort until we
  see why the per-agent enumeration collapses the vertex.
- **B. Combiner infinite-CR handling.** Check whether the missing [50,10] should surface as
  a rank-deficient/infinite combo that `build_gne_solution` drops in its non-empty/rank
  test, and recover it there + box-constrained selection.
- **C. Escalate.** Treat this as a genuine gap in the published method for lower-bound
  coupling with tight boxes; raise with the authors / reconsider the map construction.

## What exists right now

- `simple_game/rtm_v0_game.py`, `rtm_v0_test.py` — the v0 game + test (map correct where
  covered: == centralized 2.8e-14, == ADMM 1.8e-5; fails only on the coverage gaps above).
- No reference or amrhg solver code was modified.
- `FORMULATION.md` — the locked formulation (unchanged; still valid).
