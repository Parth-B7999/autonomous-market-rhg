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
  self-consistent combination (a certified GNE). The located self-consistent GNE is accepted directly;
  a neighbour-based refinement was measured to change the dispatch by ≤0.05 kW at ~177 ms/ceiling-step,
  so it is omitted. **Rare fallback** (0.7% of steps, at the degenerate ceiling): a warm-started
  distributed ADMM (ρ=0.002, ~27 rounds). Deployed: **1 broadcast + 1 local solve per step**.
- **Validation oracle:** centralized Gurobi QP (`rhg_online.centralized`), offline only.

## Results (deployed one-shot; see `report/rhg_detailed_report.pdf`)
Two ERCOT weeks (1–7 Apr, 7–13 Jul 2025), 1344 real-time steps:
- **99.3% iteration-free** (9/1344 fallbacks); **1 broadcast/step** vs a tuned ADMM's **67 rounds**
  for the same equilibrium (FACET uses 1.8% of ADMM's total communication).
- **Median RTM step 3.9 ms** against the 900 s dispatch interval; ~34× faster than ADMM per step.
- Matches the centralized equilibrium to ~10⁻⁴ kW in the interior; at the non-unique coupling ceiling
  returns a certified but non-variational GNE, bounded to ~2% of the 900 kW band (worst ~19 kW).
- Weekly H₂ target 140–141%; renewable curtailment 5.8–6.8%.

See `bench_solvers.py`/`report_bench.py` for the head-to-head vs ADMM (Douglas–Rachford implemented
but dropped: it is centralized and violates the privacy model).

## Note
Earlier design generations (4-agent battery/VRFB framework, H=1 single-step runs, iterative
version experiments, and their tests/scripts) are in `_archive/` — not part of the current
pipeline.

## License
MIT © 2026 Parth Brahmbhatt
