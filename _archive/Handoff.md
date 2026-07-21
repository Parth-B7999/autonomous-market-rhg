# autonomous_market_rhg — Project Handoff

> **Iteration-Free Multi-Agent Receding Horizon Games for Autonomous Market Participation**
> Case study for the ERCOT two-settlement (DAM + RTM) market using explicit multiparametric GNE
> (mp-GNE) for the real-time stage and FACET-GNE for the day-ahead stage.

---

## 1. Project Overview

This project applies the **explicit multiparametric GNE** framework (Hall & Bemporad, arXiv
2512.05505, Dec 2025) and the FACET-style neighbor exploration developed in `mpgne_ppopt` to a
new application: **a cluster of small DERs behind a shared point-of-common-coupling (PCC),
each independently owned, coordinating only through the physical interconnection limit while
bidding collectively into the ERCOT day-ahead (DAM) and real-time (RTM) energy markets**.

Two distinct forces motivate the GNE formulation:

1. **Primary — physical coupling.** A single PCC / transformer caps the collective net
   power: `L_min ≤ Σ_i p_i ≤ L_max`. This constraint is engineering, not regulatory; it is
   what makes the agents' subproblems coupled and forces a game rather than N independent QPs.
2. **Secondary — market access.** Each asset is individually below the ISO minimum-bid
   threshold (≈ 1 MW in ERCOT). The collective bid is the only market-legal interface.

We use a GNE rather than central dispatch because each asset is **independently owned**: each
agent runs its own cost function (degradation, H₂ revenue, SoC preferences) and is unwilling
to surrender state to a coordinator. A Nash solution is self-enforcing — no side payments,
no information leak.

The headline claim is **iteration-free online operation**:

- Each agent evaluates a precomputed piecewise-affine GNE policy.
- No inter-agent communication rounds at runtime.
- No iterative consensus / ADMM at the 5-min RTM cycle.
- Privacy preserved — agents share parameters (initial state, forecasts) only, never trajectories
  or duals.

This contrasts with the baseline ADMM-based GNE seeking which requires many iterations + full
trajectory sharing per RTM cycle.

### Key narrative split

| Stage | Method | Why |
|---|---|---|
| DAM (24×1h, once/day) | **FACET-GNE** | Parameter dim ≈ 50, latency tolerant (minutes), full explicit enumeration intractable. |
| RTM (5-min, H=6) | **Explicit mp-GNE + FACET fallback** | Parameter dim ≈ 40, latency critical (<5 min), explicit lookup feasible; FACET extends the partition on-the-fly if runtime parameter exits the precomputed envelope. |

The paper earns "iteration-free" honestly: online there is no inter-agent iteration, only
local PWA evaluation + (rarely) local FACET neighbor expansion in each agent's own slice.

---

## 2. Relationship to sibling projects

`autonomous_market_rhg/` is a **standalone new project**. It does NOT depend on
`mpgne_ppopt/` or `dimpc_ppopt/` as installed packages. Solver code from those projects
will be **copied or re-implemented** here as needed, so this repo is self-contained and can
be released independently with the paper.

| Sibling | Will we copy code from it? | Notes |
|---|---|---|
| `mpgne_ppopt/` | Yes — explicit mp-GNE solver, FACET-GNE online solver, ADMM baseline, plant/game data structures. | Copy `mp_solver.py`, `cr_store.py`, `facet_gne.py`, `admm_solver.py`, `game.py`, `gne_combiner.py` (with adaptations). |
| `dimpc_ppopt/` | Possibly — FACET neighbor-finder logic if not already mirrored in `mpgne_ppopt/facet_gne.py`. | Confirm overlap before copying. |
| `GNE_PPOPT/` | No — older prototype. |

**Action item:** before any coding, do a one-pass audit of `mpgne_ppopt/mpgne/*.py` to decide
the exact copy set and identify what (if anything) needs re-implementation vs. straight copy.

---

## 3. Locked design decisions (from scoping brainstorm)

### 3.1 Agents — 4-agent base case

All agents are **independently owned** (no shared tanks / no shared infrastructure beyond the
PCC). Each is below the 1 MW market-participation threshold individually. They share one
transformer / feeder, which is what couples their dispatch decisions.

| # | Agent | Capacity | Internal state | Notes |
|---|---|---|---|---|
| 1 | PV + Li-ion battery | 300 kW PV + 400 kW / 1.5 MWh battery | SoC (kWh) | Stochastic generation from PV; fast storage. PV forecast error is the principal RTM driver. |
| 2 | Vanadium redox flow battery (VRFB) | 300 kW / 2 MWh | SoC (kWh) | Long-duration arbitrage. Lower round-trip eff (~75%) than Li-ion. |
| 3 | PEM electrolyzer + H₂ tank | 500 kW (~10 kg H₂/hr) | H₂ inventory (kg) | Fast-ramp (~25 kW/min). H₂ offtake at fixed contract price → revenue couples to electricity cost. |
| 4 | Alkaline electrolyzer + H₂ tank | 400 kW (~7 kg H₂/hr) | H₂ inventory (kg) | Slow-ramp (~8 kW/min), lower efficiency than PEM. Heterogeneity with agent 3 makes the coupling constraint binding. |

Per-agent peak: all < 1 MW. Aggregate peak load capacity ≈ 1.3–1.6 MW. The mix is deliberate
— a fast battery + slow alkaline + PV-coupled storage are what create genuinely *different*
preferred trajectories under the same coupling envelope, justifying the GNE rather than a
trivial sum.

### 3.2 Market structure — two-settlement

- **DAM:** single-shot, 24×1h horizon, gate closure once per day. Output = 24-element collective
  power schedule `Σ_i p_{i,k}^DA` for k=1..24. Internally each agent's `p_{i,k}^DA` is decided
  by the DAM-GNE.
- **RTM:** 5-min cadence, H=6 step (30 min) lookahead receding horizon. At each 5-min cycle,
  each agent's `p_{i,k}^RT` is read off its explicit PWA policy with current parameters.
- **Settlement:** two-settlement.
  - DAM revenue: `Σ_k λ_{DA,k} · p_{i,k}^DA`
  - RTM imbalance: `Σ_k λ_{RT,k} · (p_{i,k}^RT − p_{i,k}^DA) + γ⁺·z⁺_{i,k} + γ⁻·z⁻_{i,k}`
  - **Linear asymmetric imbalance penalty** via aux slack variables `z⁺, z⁻ ≥ 0` with
    `z⁺ − z⁻ = p^RT − p^DA`. Different penalty rates for over- vs under-generation (ERCOT-style).

### 3.3 Coupling constraints (per timestep)

- **Lower (participation threshold):** `Σ_i p_{i,k} ≥ L_min = 1 MW`
- **Upper (PCC / transformer limit):** `Σ_i p_{i,k} ≤ L_max = 2.5 MW`

Two coupling constraints → two coupling Lagrange multipliers → richer v-GNE consensus story.

### 3.4 Local constraints (per agent)

- SoC / inventory bounds (`x_min ≤ x_i ≤ x_max`)
- Charge/discharge or production rate limits
- Ramp limits (both electrolyzers; PEM faster, alkaline slower)
- Daily product budget (each electrolyzer must produce ≥ its target kg H₂/day)

### 3.5 Cost structure (each agent)

```
J_i = energy_cost + imbalance_penalty + asset_cost + (revenue if applicable)

energy_cost     = Σ_k λ_{DA,k} · p_{i,k}^DA + λ_{RT,k} · (p^RT − p^DA)         [LINEAR in p, parametric in λ]
imbalance       = Σ_k γ⁺·z⁺_{i,k} + γ⁻·z⁻_{i,k}                                [LINEAR in slacks]
asset_cost      = Σ_k a_i (Δu_{i,k})² + b_i (x_{i,k} − x_ref)²                  [QUADRATIC — gives Q_i ≻ 0]
revenue         = -π_H2 · production_i      (each electrolyzer, PEM and alkaline)  [LINEAR]
```

The quadratic asset terms ensure `Q_i ≻ 0` required by Hall–Bemporad mp-GNE; everything
parametric enters via the linear cost coefficient or constraint RHS, satisfying the framework's
assumptions.

### 3.6 Welfare / selection criterion

For critical regions with infinitely many GNEs, partition into subregions using:

- **Welfare GNE** (utilitarian sum, eq. 15 in Hall–Bemporad) — primary criterion for paper
- **v-GNE** (homogeneous coupling multipliers across agents) — comparison plot
- **Minimum-norm GNE** — comparison plot

### 3.7 Data + simulation

- **ISO data:** ERCOT historical, **South Load Zone (LZ_SOUTH)** — solar-rich, moderate
  volatility. Public via ERCOT MIS.
  - DAM LMP: hourly λ_DA
  - RTM LMP (5-min SCED): λ_RT
  - PV generation: solar generation profiles aligned to ERCOT solar fleet (or single-site if
    available; otherwise scale public NSRDB irradiance)
- **Forecast:**
  - DAM forecast: realized + Gaussian noise (the "good but imperfect" forecast)
  - RTM nowcast: persistence + AR(1) noise for PV and λ_RT over the 30-min horizon
- **Simulation length:** 1 representative day (headline figures) + 1 full week (statistics)
- **Baselines (2):**
  1. **DAM-only open-loop** (commit & stick — no RTM re-optimization)
  2. **ADMM iterative distributed GNE seeking** (Belgioioso & Grammatico 2018 style)

### 3.8 Sensitivity / scalability study

- Horizon sweep: H ∈ {3, 6, 12} at fixed N=4
- Agent count sweep: N ∈ {2, 3, 4, 6, 8} at fixed H=6 (clone agent templates or vary mix)
- Report: # critical regions, offline solve time, online lookup time, memory footprint,
  vs. ADMM inter-agent communication rounds + wallclock per RTM cycle.

---

## 4. Proposed repository structure

```
autonomous_market_rhg/
├── Handoff.md                    # this file
├── README.md                     # abstract, install, quickstart, figure index
├── LICENSE                       # MIT (match sibling projects)
├── .gitignore                    # python, OS, data/, results/, *.pkl, __pycache__
├── pyproject.toml                # package metadata, deps
├── requirements.txt              # pinned versions for reproducibility
│
├── src/amrhg/                    # the python package (name: amrhg)
│   ├── __init__.py
│   │
│   ├── agents/                   # per-agent specs: Q_i, c_i, F_i, A_i, b_i, S_i, dynamics
│   │   ├── __init__.py
│   │   ├── base.py               # Agent ABC: build_local_qp(), build_dynamics(), …
│   │   ├── pv_battery.py         # PV + Li-ion (300 kW PV, 400 kW/1.5 MWh battery)
│   │   ├── vrfb.py               # VRFB (300 kW / 2 MWh)
│   │   ├── electrolyzer.py       # PEM 500 kW + H₂ tank
│   │   # NOTE: alkaline electrolyzer reuses ElectrolyzerAgent with a second config block
│   │
│   ├── markets/
│   │   ├── __init__.py
│   │   ├── dam.py                # DAM 24×1h GNE assembly
│   │   ├── rtm.py                # RTM 5-min, H=6 RHG assembly
│   │   └── settlement.py         # Two-settlement; aux-var asymmetric linear imbalance
│   │
│   ├── game/                     # GNE problem composition
│   │   ├── __init__.py
│   │   ├── builder.py            # Stack agent QPs → block Q, c, F, A, b, S for mp-GNE
│   │   ├── coupling.py           # L_min ≤ Σ p ≤ L_max coupling constraints
│   │   └── params.py             # Parameter vector packing/unpacking
│   │
│   ├── solvers/                  # COPIED + adapted from mpgne_ppopt
│   │   ├── __init__.py
│   │   ├── mp_solver.py          # Per-agent mpQP via PPOPT (← mpgne_ppopt/mp_solver.py)
│   │   ├── cr_store.py           # CR / GNESolution data structures
│   │   ├── gne_combiner.py       # Combination enumeration + projection
│   │   ├── explicit_mpgne.py     # Top-level offline explicit GNE
│   │   ├── facet_gne.py          # Online FACET neighbor expansion (V2 from MATLAB)
│   │   ├── admm_baseline.py      # ADMM iterative baseline
│   │   └── dam_only.py           # Day-ahead-only open-loop baseline
│   │
│   ├── forecasting/
│   │   ├── __init__.py
│   │   ├── pv.py                 # Persistence + AR(1) noise PV nowcast
│   │   └── price.py              # AR(1) λ_RT forecast
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── ercot.py              # LZ_SOUTH λ_DA, λ_RT, PV loaders + alignment
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── driver.py             # Week-long loop: DAM @ midnight, RTM every 5 min
│   │   └── logger.py             # Results capture
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── metrics.py            # Cost, deviation, welfare vs v-GNE split, etc.
│       └── plots.py              # All figure scripts
│
├── configs/                      # YAML per-experiment configs
│   ├── base.yaml                 # 4-agent locked spec
│   ├── scalability.yaml          # N ∈ {2,3,4,6,8}
│   └── horizon_sweep.yaml        # H ∈ {3, 6, 12}
│
├── scripts/                      # one-file entry points; produce all paper artifacts
│   ├── 00_fetch_ercot.py         # Download/clean raw CSVs
│   ├── 01_solve_dam.py           # Offline FACET-GNE for DAM
│   ├── 02_solve_rtm.py           # Offline explicit mp-GNE for RTM
│   ├── 03_run_closed_loop.py     # 1-week sim, all methods
│   ├── 04_scalability.py         # N/H sweeps
│   └── 05_make_figures.py        # Regenerate paper figs
│
├── tests/                        # pytest
│   ├── __init__.py
│   ├── test_agents.py            # Q_i ≻ 0, dynamics sanity per agent
│   ├── test_coupling.py
│   ├── test_settlement.py        # Imbalance aux-var equivalence to |·|
│   ├── test_solvers.py           # Wrappers' smoke tests
│   └── test_end_to_end.py        # 2-agent toy sanity case
│
├── notebooks/                    # exploratory only; not paper figs
│   ├── 01_explore_ercot.ipynb
│   └── 02_inspect_crs.ipynb
│
├── data/                         # gitignored
│   ├── raw/                      # raw ERCOT downloads
│   └── processed/                # aligned weekly slices
│
└── results/                      # gitignored
    ├── figures/                  # paper PDFs
    └── tables/                   # CSVs for paper tables
```

---

## 5. Implementation order (when we start coding)

Each step ends with a runnable end-to-end check so we never accumulate untested code.

| # | Step | Done when… |
|---|---|---|
| 1 | Skeleton: empty folder + `pyproject.toml` + `.gitignore` + `README.md` stub + import smoke test | `python -c "import amrhg"` succeeds |
| 2 | Copy + adapt solver core from `mpgne_ppopt`: `mp_solver.py`, `cr_store.py`, `gne_combiner.py`, `facet_gne.py`, `admm_solver.py` into `src/amrhg/solvers/` | Existing `mpgne_ppopt` tests for those modules pass under new namespace |
| 3 | `agents/base.py` + `agents/vrfb.py` (simplest dynamics) + unit test (`Q_i ≻ 0`, dynamics step) | `pytest tests/test_agents.py::test_vrfb` passes |
| 4 | `game/builder.py` + `game/coupling.py` for 2-agent toy (VRFB × VRFB) → solve via `explicit_mpgne` → smoke | 2-agent toy GNE matches centralized SLSQP within tol |
| 5 | Remaining agents one at a time: `pv_battery`, then two `ElectrolyzerAgent` instances (PEM and alkaline configs), each with unit tests | All 4 agent unit tests pass |
| 6 | `markets/settlement.py` — linear asymmetric imbalance with aux vars; unit test against `\|·\|` reference | `pytest tests/test_settlement.py` passes |
| 7 | `markets/dam.py` + `scripts/01_solve_dam.py` (FACET-GNE) | Offline DAM solve for one day produces a valid PWA |
| 8 | `forecasting/` + `markets/rtm.py` + `scripts/02_solve_rtm.py` (explicit mp-GNE) | Offline RTM solve produces a CR partition |
| 9 | `simulation/driver.py` + `scripts/03_run_closed_loop.py` (RHG + DAM-only baseline) | 1-day closed-loop runs end-to-end |
| 10 | ADMM iterative baseline wrapper | ADMM run on same instance reaches same GNE within ε |
| 11 | `analysis/plots.py` + `scripts/05_make_figures.py` | All headline figures regenerate from saved results |
| 12 | Scalability sweep `scripts/04_scalability.py` + sensitivity tables | Tables 1–N populated |

---

## 6. Open items / questions before coding starts

1. **Audit `mpgne_ppopt/mpgne/*.py` overlap with `dimpc_ppopt/dimpc/*.py`** to confirm the exact
   copy set. Specifically: is `mpgne_ppopt/mpgne/facet_gne.py` the complete neighbor-expansion
   implementation, or does it depend on anything in `dimpc_ppopt/dimpc/facet_finder.py`?
2. **ERCOT data fetch:** confirm whether LZ_SOUTH 5-min RTM LMPs and aligned PV are available
   from a single source, or whether we need to align ERCOT LMP CSVs with NSRDB irradiance CSVs.
3. **Electrolyzer ramp constraints:** literature values vary widely. Current defaults: PEM
   ~25 kW/min, alkaline ~8 kW/min. Pin tighter defensible values before paper submission.
4. **Daily product budget enforcement:** equality at end-of-day (`Σ_k production_k = target`)
   or inequality (`≥ target`)? Equality is cleaner mathematically; inequality is more realistic.
5. **DAM gate-closure timing:** ERCOT DAM gate closes at 10:00 prevailing time the day before.
   Decide whether the simulation models this gate-closure lead time explicitly or assumes
   day-of midnight closure for simplicity.
6. **Random seed strategy:** all stochastic forecast noise paths reproducible — fix seeds at
   the `driver.py` level and log to results.

---

## 7. Style + conventions

Match conventions used in `mpgne_ppopt`:

- Python 3.11+
- `numpy`, `scipy`, `cvxpy` (or direct `gurobipy`), `ppopt` for mpQP
- Type hints throughout; `from __future__ import annotations`
- Unit tests with `pytest`, parametrized where useful
- Configs as YAML loaded via `pyyaml`
- Results checkpointed as `.pkl` or `.npz` under `results/` to enable re-plotting without
  re-solving
- All paper figures regenerable from a single `python scripts/05_make_figures.py` call
- License: MIT (match sibling projects)

---

## 8. Paper figure / table targets

Provisional list — finalize once results land.

| Artifact | Purpose |
|---|---|
| Fig. 1 | System diagram: 4 agents, PCC, DAM/RTM data flow, no aggregator |
| Fig. 2 | One representative day: DAM commitment vs. RTM realization, all 4 agents |
| Fig. 3 | Cost breakdown per agent: DAM revenue, RTM imbalance, asset cost, product revenue (RHG vs DAM-only vs ADMM) |
| Fig. 4 | Welfare GNE vs v-GNE vs min-norm subregion partition (one critical region) |
| Fig. 5 | Closed-loop response to PV forecast bust (contingency scenario, optional) |
| Tab. 1 | Per-method total cost over 1 week (RHG, DAM-only, ADMM, oracle) |
| Tab. 2 | Online compute + comm: RHG (algebraic, 0 rounds) vs ADMM (k rounds, ms wallclock) |
| Tab. 3 | Scalability: # CRs, offline time, online lookup time vs N and H |
| Tab. 4 | Privacy: what each agent transmits per RTM cycle in each method |

---

## 9. Next action

When ready to start coding: **execute Step 1** of §5 — create the skeleton (folder layout,
`pyproject.toml`, `.gitignore`, `README.md` stub, empty `__init__.py`s) and confirm `import
amrhg` works. Do not touch `mpgne_ppopt/` or `dimpc_ppopt/` while doing this.

Then **Step 2** is the audit + copy of solver modules. Coding proper begins at Step 3.
