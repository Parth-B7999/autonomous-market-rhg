# Status & Handoff — DAM/RTM coupling band unification & boundary well-posedness
_Session 2026-07-14. Everything done this session, current code state, findings, and open decisions._

---

## 0. TL;DR

- **Goal:** unify the day-ahead (DAM) and real-time (RTM) PCC coupling band to the same
  physical `[L_min, L_max] = [100, 900]` kW, and keep the real-time explicit-map GNE
  well-posed when the aggregate import binds the ceiling.
- **What was blocking it:** the DAM used a *tighter* band `[350, 600]` (leftover from an old
  demo). Unifying to `[100, 900]` made the DAM buy up to the 900 ceiling, which drove **every**
  RTM step onto the coupling boundary — where the equilibrium is **degenerate / non-unique** and
  the online map + naive solver collapsed (whole days H₂=0%, 96/96 fallbacks).
- **What we fixed:** rewrote the RTM online clearing to be **point-location-first + variational-GNE
  (potential-min) solve + membership test + warm-started ADMM fallback**. The pipeline now runs
  the full `[100,900]` band on all 14 days, meets H₂, and clears ~99% of steps iteration-free.
- **What remains (one real limitation):** at the ceiling the game has **multiple GNE**; the online
  map sometimes lands on a *corner* GNE a **few kW (<1% of 900)** from the variational/social
  optimum, and ~**0.9%** of steps still need the (fast, warm-started) ADMM fallback. Driving the
  boundary steps to exact `1e-4` accuracy needs the paper's **common-multiplier v-GNE selection
  applied online** — a real (non-tuning) task.

---

## 1. Context / the two questions that started this

1. **Distributed DAM:** the DAM was a *centralized* solve (`focapo_1day.solve_dam`), which
   contradicts the paper's premise (agents never share models). We made it a **distributed ADMM**
   (each agent solves its own 24-h QP; only the per-hour aggregate is shared). Done & validated
   earlier; see `report/rhg_detailed_report.tex` (updated) and `simple_game/dam.py`.
2. **Band mismatch (this doc's focus):** DAM band `[350,600]` vs RTM band `[100,900]`.

---

## 2. Files changed / added this session

| File | Change |
|---|---|
| `simple_game/dam.py` | **NEW.** Distributed DAM (ADMM). Imports `FLEET`, `L_MIN=100`, `L_MAX=900`, `R_H2`, `EPS_CV`, `LAM_BOX` from `rhg_mpqp` (single source of truth). `build_dam_game`, `solve_dam_admm` (rho=0.1, max_iter=5000), `solve_dam_centralized` (oracle, same γ_DA-regularized objective), `load`. γ_DA=1e-4 strict-convexity regularizer on grid import. Standalone `python simple_game/dam.py <day>` prints DAM convergence + gate. |
| `simple_game/rhg_online.py` | **Core rewrite of the online clearing.** New `_solve_combo_vgne` (v-GNE potential-min via `gne_combiner._solve_equilibrium(select="potential")`), `_membership_ok`, `_feas_resid`, `_equilibrium_x`, and a rewritten **`solve_step`** (point-location-first). Added imports: `deque`, `admm_solve`, `_solve_equilibrium`. |
| `simple_game/rhg_week.py` | Imports `dam as F` (was `focapo_1day`); fleet params via `R._pmax/_eta/_typ/_rcap`. `run_day` uses new `solve_step(th, sols, game, prev_x, prev_combo, ...)`, threads `prev_x`, never skips a step, tracks per-step **data transfer** in `comm`. `main()` prints a `DATA TRANSFER` line. |
| `simple_game/rhg_figs.py` | Removed hardcoded "1–7 Apr 2025" title → dynamic from `res`. |
| `report/coupling_band_issue.tex` (+ `.pdf`) | **NEW memo** documenting the band/degeneracy issue for the PI (written before the fix; describes the *problem* and options A–D). |
| `_archive/focapo_1day.py` | **Moved** (was the H=1 single-step demo; its DAM logic now lives in `dam.py`). |
| `_archive/report/agent_formulations.*` | **Moved** (stale 4-agent report). |

**Single source of truth now:** `rhg_mpqp.py` holds `FLEET` (7-tuple), `L_MIN=100`, `L_MAX=900`,
`R_H2`, `EPS_CV`, `LAM_BOX`, `H=4`, `DT=0.25`. `dam.py` imports them. No more duplication.

---

## 3. The root-cause chain (what we learned, in order)

1. **Band mismatch confirmed:** `focapo_1day.L_MIN/L_MAX = 350/600`, `rhg_mpqp.L_MIN/L_MAX = 100/900`.
   `rhg_week`'s DAM inherited 350/600; RTM ran 100/900.
2. **Unifying to 100/900 broke the RTM:** April 322 misses/week, H₂ 71%; July whole days H₂=0%;
   map-vs-centralized up to 91 kW. Root: the 100/900 DAM front-loads buying to the **900 ceiling**
   (cheap power + \$3/kg H₂ = buy max; H₂ target is a *floor*, so it over-produces to 156–182%).
   Every RTM step then sits at the coupling boundary.
3. **Why the boundary breaks the map (Hall & Bemporad 2025, `papers/Hall_mp_GNE.pdf`):**
   - Their **Def. 2 / §II-C**: when the shared coupling is active for **≥2 agents** (i.e. Σp = L_max),
     the equilibrium matrix `M_x` is **rank-deficient** → **infinitely many GNE**, and critical
     regions **overlap** in θ-space.
   - `rhg_online._solve_combo` used a naive `np.linalg.solve(M_x,·)` → **fails on singular M_x**.
   - The self-consistency check `located_combo == combo` **fails on overlapping CRs**.
4. **The paper's fix already exists in the code:** `gne_combiner._solve_equilibrium(select="potential",
   cost=(Q,c,F))` picks the point on the solution manifold that **minimises the game potential
   Σ J_i = the variational GNE = social optimum** (matches centralized). This is the paper's
   common-multiplier v-GNE (their §II-C-3, eqs. 16–17). The H=1 archived demo used it; the H=4
   `rhg_online` path did not.
5. **The actual failure was warm-start / reachability (user's diagnosis was right):** debugging one
   fallback step showed the correct combo was **full-rank, solvable, feasible, membership-OK**, and
   `_equilibrium_x` accepted it — but the neighbor-graph BFS from `prev_combo` **never reached it**
   (consecutive steps' combos are graph-far apart near the boundary). **Point-location from the warm
   `x` returns the correct combo directly (0 hops).** After an ADMM fallback (which didn't converge),
   seeding `prev_combo` from the garbage `x` poisoned the next step → cascade → whole day dead.

---

## 4. The current RTM clearing algorithm (`rhg_online.solve_step`)

Per 15-min step, given the warm solution `prev_x` and `prev_combo`:

1. **Tier 1 — point-location self-consistency (the reliable primitive):** from `prev_x`, locate each
   agent's CR at `[Σ_{j≠i} p_j ; θ]`, **v-GNE-solve** that combo (`_solve_combo_vgne`), re-locate from
   the new `x`, iterate up to `max_rounds=4`. Accept (STRICT, `tol=1e-4`) if feasible + membership +
   self-consistent.
2. **Tier 1b — BFS neighbour walk** (1st/2nd/3rd neighbour, `max_hops=3`) from the located combo, STRICT.
3. **1-hop min-potential refinement** → locks the variational (lowest-potential feasible) combo, STRICT.
4. **Tier 2 — ADMM fallback** (`admm_solve`, **warm-started** from the best v-GNE iterate `loc_x`), used
   only if all the above fail. The ONLY iteration; a non-converged result is not propagated as a warm start.

**Two tolerances** (`rhg_online.py`): `STRICT_TOL=1e-4` (walk + refinement, keeps accuracy) and
`TOL_ACCEPT=1e-1` (only used to warm-start ADMM now — the earlier "bounded-residual accept" Tier 1c was
**removed** because feasibility ≠ optimality let it accept a corner GNE a few kW off).

**Data-transfer accounting** (in `comm`, printed by `rhg_week`): `transfers` = one entry per step = **1**
for a map step (single broadcast; walk/refinement is local) or **n_iter** for an ADMM-fallback step.
Plus `map_steps`, `fallback`, `fallback_rounds`, `combos_checked`.

---

## 5. Current results — full both weeks, band 100/900

| Day | λ range [$/MWh] | map==cent max | ADMM fallbacks /96 | H₂ |
|---|---|---|---|---|
| 04-01 | 10–73 | **2.6e-4** | 0 | 167% |
| 04-02 | 17–282 | 2.1 kW | 1 | 122% |
| 04-03 | 11–31 | **6.9e-6** | 0 | 134% |
| 04-04 | 9–60 | **6.4e-6** | 1 | 139% |
| 04-05 | 11–23 | 5.9 kW | 0 | 163% |
| 04-06 | −1–234 | 5.5 kW | 0 | 131% |
| 04-07 | 0–3553 | 3.4 kW | 1 | 126% |
| 07-07 | 22–95 | 1.9 kW | 1 | 135% |
| 07-08 | 21–151 | 4.2 kW | 1 | 132% |
| 07-09 | 17–98 | 4.9 kW | 1 | 148% |
| 07-10 | 11–79 | 2.6 kW | 4 | 156% |
| 07-11 | 12–1754 | 1.6 kW | 1 | 163% |
| 07-12 | 26–312 | 3.2e-1 | 1 | 132% |
| 07-13 | 16–82 | 3.2 kW | 0 | 129% |

**Totals:** ~**12 fallbacks / 1344 steps (0.9%)**; H₂ met every day (122–167%); data transfer ≈ 1 round/step
except the rare fallback steps (each ≈ tens–hundreds of ADMM rounds now, warm-started).

**Interpretation of the few-kW `map==cent`:** NOT a bug and NOT (only) the fallback. At the ceiling the
game has multiple GNE; the online map sometimes lands on a **corner GNE within a few kW (<1% of 900) of the
variational/social-optimum** the centralized computes. The 1-hop refinement doesn't always reach the
variational combo at the boundary.

**Baseline for comparison — old 350/600 DAM (interior anchor):** 0 fallbacks, `map==cent ≤ 5.4e-4`, H₂
119–125%, but never uses the top 300 kW of the wire. (That tighter ceiling was *load-bearing* — it kept RT
dispatch interior where the GNE is unique.)

---

## 6. Open decision (where we stopped)

Three paths (user leaning toward 2 = the principled Option B):

1. **Accept & document the current 100/900 result** — "iteration-free on ~99% of steps to ~1e-4 kW; at the
   rare multi-equilibrium ceiling vertex, within <1% of the social optimum or a fast warm-started ADMM
   fallback." Then finalize `report/rhg_detailed_report.tex` with these numbers.
2. **Implement the online variational-GNE selection** — apply the paper's **common-multiplier consensus**
   (eqs. 16–17, i.e. `filter_variational_kkt`'s KKT condition) **locally at boundary combos** inside
   `solve_step`, so the online map selects the variational GNE (not a corner one) and boundary steps hit
   exact 1e-4. This removes both the few-kW gap and most fallbacks. **Non-trivial but focused.**
3. **Revert to Option A** (conservative DAM ceiling, e.g. `L_max^DA ≈ 600`, `L_min=100` shared) — exact
   accuracy, 0 fallbacks, documented as a deliberate conservative day-ahead commitment; does not use the
   full interconnection. (This is the `report/coupling_band_issue.tex` recommendation.)

### Concrete next step for Option 2 (if chosen)
In `solve_step`, when the located combo has rank-deficient `M_x` (coupling active for ≥2 agents), instead of
relying on `select="potential"` + 1-hop refinement, **augment the equilibrium system with the paper's
multiplier-consensus equations** `λ*_{c,i} = λ*_{c,i1} ∀i` (eq. 16) and solve for the v-GNE affine law over
that CR (eq. 17). The building blocks exist: `gne_combiner.filter_variational_kkt` (offline, per-combo KKT +
CR clip) and `_solve_equilibrium`. The work is to apply the per-combo v-GNE (with multiplier consensus)
**online at the single located combo**, not by enumerating K^N.

---

## 7. How to reproduce

```bash
cd ".../autonomous_market_rhg"
# DAM only (per-day convergence + gate vs centralized oracle):
python simple_game/dam.py 2025-04-01
# One RTM day / week (prints map==cent, ADMM-fallbacks, H2, DATA TRANSFER line):
python simple_game/rhg_week.py 2025-04-01
python simple_game/rhg_week.py 2025-04-01,2025-04-02,...   # comma-separated week
# Aggregates + figures from the cached week pkl:
python simple_game/rhg_figs.py
```
Band is set in `rhg_mpqp.py` (`L_MIN, L_MAX`). Offline maps are cached in
`simple_game/out/rhg_agent_sols.pkl` (rebuild via `rhg_offline.py` if the per-agent mpQP changes).

---

## 8. Key file:line references

- Coupling band (single source): `simple_game/rhg_mpqp.py:40` (`L_MIN, L_MAX = 100.0, 900.0`)
- DAM (distributed ADMM): `simple_game/dam.py` (`build_dam_game`, `solve_dam_admm`, `solve_dam_centralized`)
- RTM online clearing: `simple_game/rhg_online.py::solve_step` (Tier 1 point-location → 1b BFS →
  refinement → Tier 2 warm-started ADMM); `_solve_combo_vgne`, `_membership_ok`, `_feas_resid`.
- v-GNE potential-min solver: `src/amrhg/solvers/gne_combiner.py::_solve_equilibrium` (`select="potential"`,
  lines ~178–188 docstring; SVD null-space potential-min).
- Offline v-GNE map (KKT / common multiplier, for Option 2): `gne_combiner.filter_variational_kkt`.
- Week loop + data-transfer tracking: `simple_game/rhg_week.py::run_day`, `main()`.
- Paper: `papers/Hall_mp_GNE.pdf` — §II-B (unique), §II-C (infinitely-many / rank-deficient),
  §II-C-3 (variational GNE via homogeneous multipliers), Algorithm 1.

---

## 9. Not yet done (report follow-ups)
- `report/rhg_detailed_report.tex` was updated earlier for the **distributed DAM** but its result tables
  reflect the **350/600** run. If we keep 100/900 (Option 1/2), the DAM section, band statements, and the
  week tables must be re-synced to the numbers in §5 above (and the `map==cent`/fallback story updated).
- `report/coupling_band_issue.tex` documents the problem; if Option 2 succeeds, add a resolution section.
