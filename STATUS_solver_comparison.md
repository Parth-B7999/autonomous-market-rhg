# Status & Handoff — Solver comparison (FACET vs ADMM) + RTM speed/communication cleanup

_Session 2026-07-16 → 07-18. What was done, current code state, findings, and open items._

---

## 0. TL;DR

- **Goal:** a defensible, apples-to-apples comparison of the online FACET clearing against ADMM,
  after the PI's request to benchmark; then act on what the benchmark exposed.
- **Headline result:** the old *"~35× faster than ADMM"* claim was **false** (measured on a biased
  sample — ADMM timed only on FACET's fallback steps, and at a mistuned ρ). Fixed, and after two
  local-compute cleanups the honest picture is now **better** than the old claim:
  - **Communication (the robust claim): 1 inter-agent round/step vs a tuned ADMM's 67.** FACET uses
    **1.8%** of ADMM's total communication over both weeks. Implementation- and latency-independent.
  - **Speed (secondary, now favourable): median 3.9 ms vs ADMM 104 ms (~34×),** and FACET now wins
    on the binding steps too. RTM median dropped **125 ms → 3.9 ms** via the refinement cut below.
  - **Accuracy (where FACET pays):** reaches 1e-4 kW on 1078/1344 steps vs ADMM 1277/1344; the
    misses are the ~20% binding-ceiling steps where FACET returns a **certified but non-variational**
    GNE, bounded to ~2% of the 900 kW band (worst ~19 kW).
- **99.3% iteration-free (9/1344) and accuracy are UNCHANGED** through all the cleanups.

---

## 1. Code changes (all in `simple_game/rhg_online.py` unless noted)

1. **Refinement cut ("skip-all").** `solve_step` used to run a 1-hop min-potential product over facet
   neighbours (up to 3⁶ = 729 SVD-solves/step) to chase the variational GNE. Measured: it is a NO-OP
   on ~80% (full-rank, unique GNE) steps and changes only the p90 of binding steps by ≤0.05 kW while
   costing ~177 ms each — and NEVER reaches the true variational GNE at the ceiling (>1 hop away). Now
   accepts the located self-consistent GNE (`found_x`) directly. **Byte-identical dispatch on the
   verification (max |Δx| = 0.000); RTM median 125 → 3.9 ms.** Facet neighbours remain load-bearing
   via the point-location **hint** (measured 8× speedup, 0.89 vs 7.15 ms) and the Tier-1b walk — so
   the abstract's "point-location over its facet-neighbour graph" is still accurate.
2. **Fallback ρ fix** (`rhg_online.py:446`). Tier-2 ADMM fallback used ρ=0.5, which never converged
   (hit max_iter=500 → each of 9 fallbacks cost 500 rounds = 77% of all FACET communication). Tuned
   on the exact production fallback steps: **ρ=0.002 → median 27 rounds** (23–43), 275× more accurate.
   Total FACET comms 5835 → 1591 rounds (6.5% → 1.8% of always-iterating).
3. **Walk budget: TESTED then REVERTED.** A fallback step's ~5 s wall time is NOT the ADMM (~50 ms) —
   it is the Tier-1b BFS walk churning ~15k combos when it fails. A combo budget makes failed walks
   ~60 ms BUT the fallback count is chaotically trajectory-sensitive (budget 200→15 fb, 5000→13,
   uncapped→9) and any tight cap turns 4–6 rescues into fallbacks (99.3%→99.0%). Reverted to preserve
   the 99.3% headline. If revisited: cap by HOPS (max_hops=2), not combo-count. **Do not re-try a
   small combo budget** — measured, it regresses the rate.

## 2. New files

- `src/amrhg/solvers/residual.py` — the shared stopping metric (natural residual) + high-accuracy
  reference oracle (Gurobi tol=1e-9). NB: Benenati's r<1e-6 bar is UNUSABLE here (floor ~5e-6, cost
  is nearly linear ‖Q‖=5e-3); primary metric is ‖x−x_ref‖∞ < 1e-4 kW (= pipeline STRICT_TOL).
- `src/amrhg/solvers/dr_solver.py` — Douglas–Rachford baseline. **Implemented, run, then dropped**:
  its projection is onto the coalition's joint feasible set → centralized → violates the privacy
  model. Retained but unused (FYI it ran ~5 ms; that is the price of the privacy constraint).
- `simple_game/bench_solvers.py` — the benchmark driver. Warm + cold blocks, identical init within a
  block, shared metric, instrumentation excluded from the clock, ADMM tuned on binding steps. Reads
  the exact θ FACET faced (new `th_step` recorded in `rhg_week.run_day`).
- `simple_game/report_bench.py` — text + LaTeX tables from `results/solver_bench.pkl`. Writes
  `results/bench_{warm,cold}_table.tex` (the report `\input`s them).
- `admm_solver.py` — added opt-in `trace=` (per-iteration x) and `stop_fn=` (shared-metric early
  stop); verified bit-identical when `trace=False`. `solve_time` is solver-only when tracing.

## 3. Report (`report/rhg_detailed_report.tex`, 33 pp, compiles clean)

- **§7 timing table (`tab:timing`)** — updated to the new numbers (map-resolved median 3.9 ms).
- **§8.4 (`sec:exp2`)** — rewritten from "proposed experiment" to **results**: `\input`s
  `bench_warm_table.tex`, keybox framing communication-first / speed-secondary / accuracy-cost.
- **Abstract** — replaced the biased 35× with communication (1 vs 67) + honest 34× median.
- **§8 title/intro** — "two proposed experiments" → the ADMM comparison is done; centralized-mpQP
  (Exp 1) still flagged as proposed.

## 4. Regenerate everything (from repo root)

```bash
python simple_game/report_figs.py        # → results/report_fig_data.pkl (re-runs both weeks, ~5 min)
python simple_game/report_numbers.py     # timing + iteration-free + LaTeX rows
python simple_game/bench_solvers.py --weeks   # → results/solver_bench.pkl (~70 min)
python simple_game/report_bench.py       # → results/bench_{warm,cold}_table.tex + console tables
cd report && pdflatex rhg_detailed_report.tex   # ×2 for refs
```
Cross-check invariant: `report_bench.py` asserts the bench harness FACET median matches the pipeline
`t_step` median within 25% — if it diverges, the harness is wrong, not the pipeline.

Backups (pre-change pickles): `results/{report_fig_data,solver_bench}.{PRE_V1,V1,SKIPALL,FINAL}.pkl`.

## 5. Open items (not done)

- **Report claim reconciliation (partial).** The timing table, comparison table, and abstract are
  updated. Not yet swept for every remaining mention of the old numbers (e.g. §5.1 still describes the
  refinement as active in places; §6 keybox step 4/5 still lists the refinement). A full pass over
  §5.1/§6 to state "refinement dropped; located GNE accepted directly" is the main remaining edit.
- **Experiment 1 (centralized big mpQP)** — still just proposed. Would settle whether a single joint
  explicit map builds at n_θ=24 (Benenati Table I says it fails at n_x=8). Matrices are already
  assembled in `centralized`; hand `(Q,c,F,G,w0,W)` to PPOPT under a wall-clock budget.
- **Binding-step accuracy** — the ~19 kW worst case is fundamental to local methods (variational GNE
  >1 hop away). Only Hall & Bemporad eqs 16–17 (dual-multiplier consensus, needs each CR's λ-law which
  `cr_store` currently discards) would close it. Documented in `STATUS_dam_rtm_boundary.md` §open.

See memory `project_mpgne_benchmark.md` for the full measured trail (traps, tuning, decisions).
