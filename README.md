# Iteration-Free Multi-Agent Receding-Horizon Games for Autonomous Market Participation

> Parth Brahmbhatt — UW-Madison

## Overview
Six independently-owned green-hydrogen electrolyzers behind one shared point-of-common-coupling
(PCC) bid collectively into the ERCOT two-settlement market. Real-time dispatch is a
**receding-horizon (H = 4, four 15-min steps) parametric generalized Nash equilibrium (GNE)**,
coupled across the horizon by a cumulative-H₂ demand constraint. Each agent's best response is
precomputed **offline** as an explicit multiparametric map in its own private parameters; the
only shared quantity is the aggregate grid import. Online operation is **iteration-free**:
exactly **one decision broadcast per agent per step**, resolved by a single local solve on the
precomputed maps (FACET neighbor-graph point-location) — 1 inter-agent round per step, with a
rare (0.7%) warm-started ADMM fallback at the degenerate coupling ceiling.

See `FORMULATION.md` for the locked spec and `report/rhg_detailed_report.pdf` for the full
case study.

## How to run
```bash
pip install -e .            # or: pip install -r requirements.txt

# 1. One-time OFFLINE solve (6 distinct per-agent mpQPs via PPOPT + facet_crossing graph, ~40 min)
python simple_game/rhg_offline.py                 # → simple_game/out/rhg_agent_sols.pkl

# 2. MAIN: DAM + H=4 receding RTM closed loop over ERCOT days
python simple_game/rhg_week.py 2025-04-01,2025-04-02,2025-04-03,2025-04-04,2025-04-05,2025-04-06,2025-04-07

# 3. Figures
python simple_game/rhg_figs.py                    # → results/figures/rhg_*.png
```

## Pipeline files (`simple_game/`)
| File | Role |
|---|---|
| `rhg_mpqp.py` | Per-agent H=4 private-θ mpQP builder (Q, H, c, G, b, F) + p_gne expansion |
| `rhg_offline.py` | One-time PPOPT solve of the **6 distinct** agents + `facet_crossing` neighbour graph → cached AgentSolutions |
| `rhg_online.py` | Online FACET GNE clearing (point-location + ADMM fallback) + centralized oracle + comms accounting |
| `rhg_week.py` | **Main**: per-day DAM anchor + receding RTM closed loop + metrics |
| `bench_solvers.py` | FACET vs ADMM head-to-head (warm/cold blocks, shared metric) → `results/solver_bench.pkl` |
| `report_bench.py` / `report_numbers.py` | LaTeX tables + figures straight from the result pickles (no hand-transcription) |
| `dam.py` | DAM **distributed ADMM** (`solve_dam_admm`), ERCOT `load()`, fleet — imported by `rhg_week` |

Core library: `src/amrhg/solvers/{game, cr_store, facet_gne, gne_combiner, mp_solver, admm_solver, dr_solver, residual}`.
Data: `data/ercot/` (2025 DAM/RTM LMP + solar/wind production CSVs).

## Method
- **Offline:** each of the **6 distinct** agents' strictly-convex mpQP solved in its private θ_i
  (10-dim grid, 14-dim renewable); expanded to the 24-dim public union. K^N ≈ 1.1×10²¹ region
  combinations are never enumerated. A `facet_crossing` neighbour graph is built (exact, 0 orphan CRs).
- **Online (per 15-min step):** locate each agent's critical region at `[aggregate, θ_t]` (warm-started
  by the previous step's CR + its facet neighbours — an ~8× speedup over a full scan), solve the
  block-linear equilibrium via the min-potential (variational) selection, and iterate to a
  self-consistent combination (a certified GNE). A 1-hop min-potential refinement then runs **only on
  steps where the coupling binds** — where `M_x` is rank-deficient the GNE is non-unique and the
  selection matters; on interior steps it is full rank, the GNE is unique, and the refinement is a
  provable no-op (so it is skipped). **Rare fallback** (0.4% of steps, at the degenerate ceiling): a
  warm-started distributed ADMM (ρ=0.002, ~27 rounds). Deployed: **1 broadcast + 1 local solve per step**.
- **Validation oracle:** centralized Gurobi QP (`rhg_online.centralized`), offline only.

## Results (measured 2026-07-21; see `report/rhg_detailed_report.pdf`)
Two ERCOT weeks (1–7 Apr, 7–13 Jul 2025), 1344 real-time steps:
- **99.6% iteration-free** (6/1344 fallbacks); **1.12 inter-agent rounds/step** vs a tuned ADMM's
  **67 rounds** for the same equilibrium.
- **Median RTM step 4.1 ms** against the 900 s dispatch interval (mean 168 ms, p95 200 ms — the tail
  is the binding-heavy days 04-07 and 07-07, where the refinement runs on most steps).
- Matches the centralized equilibrium to ~10⁻⁴–10⁻⁶ kW in the interior; at the non-unique coupling
  ceiling returns a certified GNE bounded to **9.1 kW worst case** (~1% of the 900 kW band).
- Weekly H₂ target 140–141%; renewable curtailment 5.8–6.8%.

> **Note on the refinement.** It was cut entirely on 2026-07-18 (see
> `docs/history/STATUS_solver_comparison.md`) after being measured as worth ≤0.05 kW — but that was on
> the *incomplete* facet-neighbour graph. On the corrected `facet_crossing` graph (~15 neighbours/CR vs
> ~7) it is worth ~14 kW → 9.1 kW worst case, so it was reinstated, gated on a binding coupling to keep
> the median at ~4 ms. **The report's tables are not yet reconciled with this** — its Week-1/2 tables
> predate the cut and its timing table postdates it.

See `bench_solvers.py`/`report_bench.py` for the head-to-head vs ADMM (Douglas–Rachford implemented
but dropped: it is centralized and violates the privacy model).

## `_validated_baseline/` — do not delete
A frozen 2026-07-16 snapshot (code + report + results) of the identical-pair-fleet build. It is the
only surviving copy of the pre-2026-07-20 online solver and was the reference used to recover the
variational selection after it was replaced. Keep it until the report is reconciled.

## Session history
`docs/history/` holds the dated handoff notes (`STATUS_*.md`). They are a record of what was done and
why, **not** current status — where they disagree with this README or the code, they are stale.

## License
MIT © 2026 Parth Brahmbhatt
