# Validated baseline — homogeneous-pair fleet (snapshot 2026-07-16)

This is a frozen copy of the **validated** mp-GNE / FACET case study, taken immediately
before the fleet was changed to 6 fully-distinct agents at the PI's request.

**Do not edit anything in this folder.** It exists so the heterogeneous-fleet results can
always be compared against a known-good reference, and so no future patch/refactor can
destroy the validated numbers again (see `_archive/patch_scripts/` for why this matters).

## Fleet in this baseline

| # | name | type | p_max | ren_cap | eta | d_max |
|---|------|------|-------|---------|-----|-------|
| 0 | PEM_Elec   | grid  | 250 | 0   | 0.020 | 3.0 |
| 1 | ALK        | grid  | 200 | 0   | 0.018 | 2.5 |
| 2 | PEM_PV     | solar | 125 | 125 | 0.020 | 1.6 |
| 3 | PEM_PV_2   | solar | 125 | 125 | 0.020 | 1.6 |
| 4 | PEM_Wind   | wind  | 250 | 250 | 0.020 | 3.0 |
| 5 | PEM_Wind_2 | wind  | 250 | 250 | 0.020 | 3.0 |

Agents 2-3 and 4-5 are identical pairs -> only **4 distinct** offline mpQP solves
(`rhg_offline.DISTINCT = {0:[0], 1:[1], 2:[2,3], 4:[4,5]}`).

Sum of p_max = 1200 kW against L_MAX = 900 kW, so the PCC ceiling binds. This is the
regime that produces the opt-out / retreat mechanism shown in the figures.

## Validated results reproduced by this snapshot

Offline (one-time, `combinatorial_parallel`, 12-core): **829 s**, 73,740 facet-neighbour pairs.

| agent | 0 | 1 | 2 | 3 | 4 | 5 | total |
|-------|---|---|---|---|---|---|-------|
| CRs   | 507 | 507 | 8359 | 8359 | 7335 | 7335 | 32,402 |

| week | fallbacks | H2 | curtailment | max abs(map - centralized) |
|------|-----------|-----|-------------|------------------------|
| April (2025-04-01, 7 d) | 3 / 672 | 140% | 3383 / 43380 = 7.8% | 5.88 kW |
| July  (2025-07-01, 7 d) | 8 / 672 | 142% | 2390 / 34540 = 6.9% | 4.92 kW |

Combined iteration-free rate: **11/1344 fallbacks = 99.2%** map-resolved
(report quotes 99.1% from the 3+9 run; the 1-step July difference is solver-order
noise, not a regression — the CR sets are identical).

## Reproducing this baseline from scratch

The 317 MB `simple_game/out/rhg_agent_sols.pkl` is **intentionally not stored here** —
it is large, OneDrive-synced, and deterministically reproducible. To rebuild:

```bash
cp _validated_baseline/simple_game/*.py simple_game/
rm -f simple_game/out/rhg_agent_sols.pkl results/report_fig_data.pkl
python simple_game/rhg_offline.py        # ~14 min, must print 507/507/8359/8359/7335/7335
python simple_game/report_figs.py        # re-runs both weeks + regenerates fig1-fig5
cd report && latexmk -pdf rhg_detailed_report.tex
```

The offline solve **must** use `ALGO = mpqp_algorithm.combinatorial_parallel`. It is
exhaustive. `geometric_parallel_exp` is not, and silently produces incomplete critical-region
coverage, which makes online point-location fail and collapses the iteration-free rate
(observed: 319/672 fallbacks). Do not change it.
