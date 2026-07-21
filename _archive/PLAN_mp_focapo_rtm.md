# PLAN — mp-GNE version of the FOCAPO-type case study (iteration-free RTM)

> Approved by Parth 2026-07-09 (session with Claude). Supersedes the ERCOT choice in
> PLAN_case_study_redesign.md §2 (→ **PJM, FOCAPO-type fleet**, per PI direction);
> the standalone methodology framing and result blocks from that plan carry over.
>
> Hard requirements (Parth, 2026-07-09):
> 1. **ZERO iterations and ZERO communication in the RTM** (DAM may iterate — offline).
> 2. mpQP only, no MIQP.
> 3. θ lean, physics that matters kept: **horizon and renewables IN; H₂
>    storage/tracking, batteries, ramp limits OUT.**
> 4. Buy-only like FOCAPO (p_i ≥ 0, no export). **No battery agents ever** (they need
>    charge/discharge exclusivity — binary/nonconvex).

## The market rules, in plain words (no code names)

- **L_min (100 kW):** PJM minimum offer size — coalition total `Σp ≥ L_min`.
- **L_max:** cap on coalition total (PCC/interconnection limit). med_scale dropped it;
  **we keep it** — one constant row per step, and it makes the game two-sided: cheap
  prices → everyone wants max → L_max binds; expensive prices → everyone wants out →
  L_min binds. Value tuned to fleet capacity.
- **"0 or in" rule** (`Σp ∈ {0} ∪ [L_min, L_max]`): the 0-branch needed the MIQP
  binary. Eliminated by assumption: **the coalition is always in the market**
  (standing commitment — it runs an H₂ business). Stated once; no binary anywhere.
- **"Rule B"** (name from our own miqp_admm.py, NOT official PJM language): if RT
  consumption exceeds the DA award, the extra is itself an RT offer and must be
  ≥ 100 kW (no small slivers on top of the award). **Not modeled** — scope
  limitation; revisit at the very end only if the PI wants full PJM fidelity.

## Architecture

Real time is a pure piecewise-affine lookup of a precomputed explicit GNE map.

| Stage | When | What | Iter/comm |
|---|---|---|---|
| S1 DAM | day D−1 (v3 only) | existing FOCAPO DAM produces hourly p_DA | allowed (offline) |
| S2 map build | offline (per hour/day in v3 — p_DA baked in) | per-agent mpQPs (PPOPT) → combiner/FACET → explicit GNE map over public θ | offline, budget reported |
| S3 RTM | every 5 min | evaluate map at public θ_t (λ, g forecasts); each agent applies its own block | **0 / 0** |

Zero communication: the combined GNE map depends only on **public** parameters —
σ_{-i} is eliminated by the equilibrium solve. Every agent carries the same map
onboard (explicit-MPC style) and evaluates it at the same θ_t. Fallback-to-ADMM
count is a reported honesty metric (target 0).

## The game

**Per-agent RTM problem (horizon H, 5-min steps):**

    min Σ_k [ (γ_i/2)(p_{i,k} − p_DA_i)² + (λ_k − r_H2·η_i)·p_{i,k} ]
    s.t. 0 ≤ p_{i,k} ≤ p_max_i
         p_elec_{i,k} = p_{i,k} + g_{i,k}                (renewable agents)
         L_min − σ_{-i,k} ≤ p_{i,k} ≤ L_max − σ_{-i,k}   (coupling, per step)

Agents heterogeneous only in constants γ_i, η_i, p_max_i (values from
med_scale_case_study/agents_med.py).

### θ decisions (asked & answered by Parth, 2026-07-09)

| Parameter | Decision | θ cost | Recorded fallback if CRs explode |
|---|---|---|---|
| λ forecast over horizon | **Full H dims** | H | PCA 2–3 factors (level/slope/spike from historical PJM 5-min data); last resort 1 scalar (current λ + frozen shape) |
| Renewable forecast g_i | **Full H dims** per renewable agent | +H (own θ_i only) | 1 scale × baked daily shape (+1); or baked-in constants w/ daily rebuild (0) |
| DA anchor p_DA | **Keep, baked in hourly** (constant at map build; hourly/daily rebuilds, cost in offline budget) | 0 | promote to +1 param/agent if rebuilds too costly |
| Ramp / intertemporal link | **None** — horizon via forecasts only | 0 | — |
| H₂ storage/tracking | **OUT** | 0 | — |
| σ_{-i} (others' aggregate) | sum-compressed (already in simple_mpqp.py) | H | — |
| L_min, L_max | constants in rows | 0 | — |

θ_i budget: grid-only agent = 2H; renewable agent = 3H. At H = 6: 12–18 dims.
Deliberately generous — **CR count is the known risk**; the fallback column is the
pre-agreed retreat path, in order.

**Structural note (recorded):** with no ramp and no storage the H-step problem
decouples across steps — each k is an independent single-step game in
(λ_k, g_k, σ_{-i,k}). If CRs explode, one single-step map (θ ≈ 2–3 dims) evaluated
H times is an EXACT reduction — the cheapest escape hatch, before PCA.

## Version ladder (measure CR count + PPOPT time before every climb)

- **v0 — pipeline validation.** N = 3 grid-only agents, H = 1, θ_i = 2, public map
  1-dim (λ). Full pipeline PPOPT → combiner → FACET → lookup, validated against ADMM
  + centralized QP on a real PJM week. First map figure (p_i* vs λ, CR boundaries,
  week's λ(t) trace).
- **v1 — full horizon.** H = 6 (then 12), λ full H dims, coupling per step,
  θ_i = 2H. **The CR-explosion test.** If explosion: decoupling trick, then PCA.
- **v2 — renewable agents.** Add PV/wind agents, g full H dims each (θ_i = 3H).
- **v3 — market structure + paper results.** DAM (existing FOCAPO code) → hourly
  p_DA baked into map rebuilds; week-long closed loop; scale N = 3 → 6 → 12; Mode 1
  (exhaustive) vs Mode 3 (FACET) offline cost; ADMM iteration/comm counts vs mp's
  zero; result blocks: θ-map figure, combo recurrence, timing/comm-degradation,
  equivalence certificate, honesty metrics.

## Do NOT trust the old machinery blindly (Parth, 2026-07-09)

- Run existing test suites (`autonomous_market_rhg/tests/`, `mpgne_ppopt/tests/`).
- Cross-check gne_combiner/facet_gne on the v0 game vs a centralized Gurobi QP
  (v0 equilibrium is derivable analytically — use it).
- Cross-check the ADMM reference (miqp_admm.py / admm_solver.py) against the same
  centralized solution before using it as ground truth.
- simple_mpqp.py deliberately DROPPED L_min and the repo's README/data claims were
  inconsistent (PJM data despite ERCOT claims) — assume drift, verify everything,
  including that `forecasting.load_pjm_week` returns what it claims.

## Key files

- Reuse (after validation): `src/amrhg/solvers/{simple_mpqp,gne_combiner,facet_gne}.py`,
  `mpgne_ppopt/mpgne/`, `med_scale_case_study/{miqp_admm,agents_med}.py`.
- New: `autonomous_market_rhg/simple_game/` — game definition + pipeline script.

## Acceptance gate (every version)

max |p_map − p_ADMM| < 1e-3 kW over all test θ; zero map-lookup misses inside the
calibrated θ box; CR counts + PPOPT times reported.

## v0 BUILT & RUN (2026-07-09) — result + one blocking decision

Code: `simple_game/v0_game.py` (game) + `simple_game/v0_pipeline.py` (cross-check).
Game: 3 grid-only agents, H=1, θ=λ (1-D public map). Runs end-to-end.

**Works:** mpQP [4,4,4] CRs/agent → combiner 24 GNE CRs (0.05s); explicit map has
0 lookup misses over λ∈[0,80]; equilibrium residual ~4e-13. In the **interior**
regime (coupling slack, e.g. λ=30) map == ADMM == centralized EXACTLY.

**Two old-machinery bugs found (facet_gne.py multiprocessing):**
1. `_facet_lp_refine_worker` unpacks a 7-tuple but the task builder sends 6 →
   `facet_adjacency` neighbor path crashes workers and the call HANGS forever.
2. The PARALLEL SEED SEARCH in `build_gne_solution_facet` (spawn workers) returns
   nothing → reports 0 GNE CRs. Workaround used in v0: `hyperplane_adjacency`
   neighbors + pass an explicit seed (fork-based BFS works: 23 CRs).
   These block FACET scaling (v3/v4) and must be fixed then; not v0-critical.

**PRIVACY CONSTRAINT (Parth, 2026-07-09):** agents can NEVER share their cost
models — only their buying decisions (power p_i / the aggregate). ⇒ centralized
option (B) is OUT. Online dispatch shares only decisions (zero model exchange).
Offline map-building is a one-time coordinator step that already has each agent's
mpQP (explicit-MPC analogy); it does not weaken the online privacy claim.

**DECISION (2026-07-09): option A — variational GNE. IMPLEMENTED & v0 GATE PASSES.**
The variational GNE = social optimum = what ADMM converges to (Facchinei-Kanzow:
potential game ⇒ v-GNE = joint optimum). Non-uniqueness shows up TWO ways, both fixed:
1. WITHIN a binding combo (rank-deficient M_x): min-norm ‖x‖ gave the wrong (equal)
   split. FIX: `gne_combiner._solve_equilibrium(select="potential", cost=...)` picks
   the potential-minimising point on the manifold — affine in p, closed form
   z*(p)=−(NᵀQN)⁻¹Nᵀ(Q x_p+c+Fp). New helper `_stacked_cost(game)`. Guarded by flag
   (default "min_norm" so mpgne_ppopt is unchanged). `build_gne_solution(...,
   equilibrium_select="potential")`.
2. ACROSS combos: the combiner enumerates ALL GNE, so many "corner" GNE CRs OVERLAP in
   p-space (e.g. at λ=57.49, 7 CRs contain it, potentials 482→1634). `locate()` returns
   the first, not the variational one. The variational GNE is ALWAYS the min-potential
   member. v0 resolves this with `eval_variational()` (min-ΣJ among overlapping CRs).
RESULT: max|map−ADMM| = 2.8e-6 kW, max|map−centralized| = 3.2e-6, 0 misses, residual
1e-13, over all 3 regimes. Map figure `simple_game/out/v0_map.png` (clean PWA in λ).

**T1 DONE (2026-07-10) — single-valued variational map, offline, privacy-preserving.**
`gne_combiner.filter_variational(gne_sol, game)` keeps a CR iff at its Chebyshev centre
it is the min-ΣJ (variational) CR among all CRs covering that centre; drops the
overlapping corner GNE OFFLINE. `build_variational_gne_solution(game, sols)` = combiner
(select="potential") → filter. v0: 24 GNE CRs → **3 single-valued variational CRs**
(one per regime), 0 conflicting-overlap points; **online = plain `locate()`+affine eval,
NO costs, NO exchange** (agents share only decisions). PLAIN-lookup gate: max|map−ADMM|
= 2.8e-6, 0 misses. Caveat recorded in code: centre test is exact for 1-D θ / strictly
convex potential; for higher-D θ (v1+) re-verify single-valuedness or escalate to a
per-combo centralized-KKT (common-μ) filter.

**T2 DONE (2026-07-10) — FACET fixed + variational, Mode 3 == Mode 1 EXACTLY.**
Bug 1: the `facet_adjacency` Phase-2 builder in `find_all_agent_cr_neighbors` sent a
6-tuple to `_facet_lp_refine_worker` (expects 7) → workers crashed, pool hung. Fixed by
delegating to `refine_neighbors_with_lp`. Bug 2: parallel seed search raced (enqueued all
combos, terminated spawn workers before results) → 0 CRs. Replaced with a serial seed
scan (early exit; BFS stays parallel/fork). Threaded `equilibrium_select="potential"`
(+`_stacked_cost`) through `_process_combo_kernel`, `build_gne_solution_facet`, BFS
globals. v0: FACET (facet_adjacency) → 24 GNE CRs → filter → 3 variational,
**max|Mode1−Mode3| = 0.0**. All amrhg-only edits; mpgne_ppopt copy untouched.

## v1 DONE (2026-07-10) — CR-explosion test + exact decoupling. GATE PASSES.

Code: `simple_game/v1_game.py` (H-step game) + `simple_game/v1_horizon_test.py`.
Figure: `simple_game/out/v1_cr_explosion.png`.

- **Naive full-horizon mpQP explodes EXACTLY as 4^H** per agent (H=1→4, 2→16, 3→64,
  4→256, 5→1024; PPOPT 0.09→7.3s). θ_i=(N−1)H+H.
- **Exact decoupling reduction:** no ramp/storage/H₂ ⇒ the H-step game is H independent
  copies of v0 ⇒ build the single-step variational map ONCE (3 CRs, θ=1), apply per
  step. FLAT in H. Validated: decoupled == naive full-horizon variational map (H=2, to
  1e-13) == ADMM on the full coupled 6-step game (to 1e-6). GATE PASS.

**STRATEGIC INSIGHT (load-bearing for the ladder):** v0–v2 have NO intertemporal
coupling, so the RTM DECOUPLES per step and the explicit map is single-step. Adding
renewables (v2) only makes the single-step map 2-D (λ_k, g_k) — still decoupled across
steps. **The horizon becomes a genuine (non-decoupling) dimension only when H₂
cumulative demand is added (v3).** So the "receding-horizon game" story has teeth only
at v3; v1/v2 establish the per-step map + the decoupling reduction as the tool.

## v2 DONE (2026-07-10) — renewables, 2-D (λ,g) map. CONTENT GATE PASSES.

Code: `simple_game/v2_game.py` (2 grid-only + 1 renewable; renewable x=[p,cv], param g)
+ `simple_game/v2_test.py`. Figure: `simple_game/out/v2_map.png` (Σp, renewable grid
import, renewable electrolyzer load over the (λ,g) plane).

- Renewable g is a GENUINE parameter: finite electrolyzer cap ⇒ g eats grid-import
  headroom (p ≤ P_elec_max − g + cv; curtail cv loses H₂ at cost a). Single-step map
  is 2-D in (λ,g). per-agent CRs [4,4,6] → 5 variational CRs.
- **Content validated: variational map == ADMM to 1.98e-6** over the (λ,g) grid, 0
  misses; FACET Mode 3 == Mode 1 (5 CRs). Decoupling across horizon holds structurally
  (same as v1) → 2-D single-step map applied per step.

## n_p≥2 SINGLE-VALUED MAP — SOLVED (2026-07-10). The real method piece.

`gne_combiner.filter_variational_kkt(gne_sol, game)` — rigorous, works for ANY n_p
(replaces the Chebyshev `filter_variational`, which overlapped for n_p≥2).
`build_variational_gne_solution` now uses it. Method:
- The variational GNE = argmin_x Φ=½xᵀQx+(c+Fθ)ᵀx s.t. Gx≤w0+Wθ (potential QP; G/w0/W
  from `_centralized_constraints` = all boxes + shared coupling). A combo is variational
  iff centralized KKT holds: ∃μ(θ)≥0 with QxА+c+Fθ+G_Aᵀμ_A=0 on active set A.
- Per combo: active set at CR Chebyshev centre → solve affine μ_A(θ)=Mθ+m0 → check
  stationarity residual ≈0 (else drop). Then **CLIP** the CR by the LINEAR cuts
  μ_{A,j}(θ)≥0 (⇔ −M_jθ≤m0_j) instead of dropping whole — a combo variational on only
  part of its region keeps exactly that part. Result tiles θ-space single-valued, NO
  gaps. O(n_cr), per-combo → scalable; works via combiner (Mode 1) AND FACET (Mode 3).
- VALIDATED (`simple_game/kkt_filter_test.py` + `v2_test.py`): v0 (n_p=1) 3 CRs and v2
  (n_p=2) 5 CRs — 0 conflicting overlaps, 0 misses, **plain locate()+affine == ADMM to
  ~1e-6**, no costs online (privacy preserved). Both Mode 1 & Mode 3.

## Next session starts at

**v3 — H₂ cumulative demand + DAM stage** (where the horizon finally couples the steps
→ genuinely H-dim θ map, not H copies of a 1-step map). The single-valued KKT filter now
handles any n_p, so higher-dim θ is unblocked. Reuse simple_game/ structure; DAM produces
hourly p_DA (baked in per plan §θ-table). Measure CR count + PPOPT time. NOTE for v3:
`filter_variational_kkt` reconstructs the active set at the CR centre — for large n_p /
degenerate active sets, re-verify residual tol and single-valuedness on the v3 θ grid.
(`v2_game.centralized_step` SLSQP ref unreliable at binding pts — ADMM is ground truth.)
