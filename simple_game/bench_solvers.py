"""
bench_solvers.py — apples-to-apples comparison: FACET-RHG vs ADMM.

WHY THIS EXISTS
  The report's original "~35x faster than ADMM" was measured on a BIASED sample: ADMM was
  timed only on the 0.7% of steps where FACET fell back — exactly the degenerate ceiling
  steps where ADMM is slowest — and at rho=0.5, which we measure to be badly mistuned. This
  harness replaces that with a measurement, following Benenati & Belgioioso
  (arXiv:2512.07749, Figs. 3-4): run the baseline on EVERY step, and report both
  time-to-tolerance and accuracy-under-a-fixed-budget.

WHY ADMM IS THE ONLY BASELINE
  ADMM is the only iterative method here that satisfies this paper's privacy model: its
  x-update is per-agent (each agent solves its own QP with its own Q_i, c_i, A_loc_i) and the
  only quantity crossing between agents is the aggregate z. Douglas-Rachford was implemented
  and benchmarked (see src/amrhg/solvers/dr_solver.py, retained but unused) and then dropped:
  its resolvent is a projection onto the COALITION's joint feasible set, which requires one
  party to hold every agent's constraint rows. It is a centralized method and therefore not a
  competitor to FACET — comparing against it would be comparing against a method that solves
  a different problem than the one this paper poses.

THE FAIRNESS RULES (each was a real bias found and removed)
  1. IDENTICAL INITIALIZATION WITHIN A BLOCK. `warm` (all methods from the previous step's
     answer — deployment-realistic) and `cold` (all methods from nothing — Benenati's
     protocol). Warm-starting FACET while cold-starting ADMM would hand us the
     temporal-coherence advantage and deny it to the competitor.
  2. ONE SHARED STOPPING METRIC. The solvers' native tests are on different scales and cannot
     be compared. Both are judged by ||x - x_ref||_inf against a high-accuracy oracle
     (Gurobi at OptimalityTol=1e-9), with their own tolerances disabled (tol -> 1e-12).
  3. INSTRUMENTATION IS NOT BILLED TO THE SOLVER. Traces record (cumulative SOLVER seconds,
     x^k) with the clock paused around bookkeeping. Audited: the leak was 0.2% and is fixed
     (admm_solver.py reports t_acc, not wall, when tracing).
  4. TUNED BASELINE, ON BINDING STEPS. rho was swept on the steps where the coupling is
     ACTIVE. A sweep over all steps said rho ~ 1e-4 — invalid, because the coupling binds on
     only ~20% of steps and the slack majority decouples the game, so any rho converges
     trivially there.
  5. REGIME SPLIT. Binding vs slack reported separately: the coupling ceiling is the subject
     of this paper and it is where both methods are hard.

MEASURED CONFOUNDS THAT THE WALL-CLOCK NUMBERS CANNOT REMOVE — read before quoting a speedup:
  (a) ADMM's t_tol is ORACLE-ASSISTED and therefore OPTIMISTIC. We stop it at the first
      iterate within 1e-4 kW of the reference. A deployed ADMM cannot know when it has got
      there and must run to a conservative native tolerance — i.e. longer. This FAVOURS ADMM.
  (b) SERIAL MEASUREMENT IGNORES NETWORK LATENCY. Both methods are timed as pure compute on
      one machine. In a real deployment FACET costs 1 communication round-trip per step and
      ADMM costs ~65; at even 10 ms/round-trip that is +10 ms vs +650 ms. This UNDERSTATES
      FACET's advantage.
  (c) IMPLEMENTATION ASYMMETRY. FACET's refinement is interpreted NumPy; ADMM's inner
      x-updates are compiled OSQP (C). Some of the wall-clock gap is language, not method.
  Because of (a)-(c), the ROBUST, implementation-independent claim is the COMMUNICATION count
  (1 broadcast vs ~65 rounds), not the wall-clock ratio. Report it that way.

METRICS (per step, per method, per block)
  t_tol   : seconds to reach ||x - x_ref||_inf < TOL_KW (1e-4 kW = the pipeline's STRICT_TOL)
  err_at  : ||x - x_ref||_inf attained within each wall-clock budget in BUDGETS
  r_final : natural residual of the returned point (Benenati-style, secondary metric)
  rounds  : inter-agent communication rounds (FACET: 1 on a map step, n_iter on a fallback;
            ADMM: n_iter)

USAGE
  python simple_game/bench_solvers.py --day 2025-04-06        # Phase 1 smoke test (gates)
  python simple_game/bench_solvers.py --weeks                 # Phase 2 full 2-week run
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import rhg_mpqp as R          # noqa: E402
import rhg_online as O        # noqa: E402
import rhg_week as W          # noqa: E402
from amrhg.solvers.admm_solver import admm_solve            # noqa: E402
from amrhg.solvers.residual import (                        # noqa: E402
    natural_residual, get_projector, reference_solution,
)

# ── tuned baseline parameters (swept on BINDING steps — see module docstring rule 4) ──
# ADMM rho — swept on BINDING steps (median iters to ||x-x_ref||inf < 1e-4 kW):
#     0.001 -> 106 | 0.01 -> 43 | 0.05 -> 136 | 0.2 -> 311 | 0.5 -> 1/12 steps only
# Bracketed on both sides, so 0.01 is the optimum. NOTE two things this sweep exposed:
#   (a) a sweep over ALL steps instead said rho ~ 1e-4 (5 iters) — invalid, because the
#       coupling binds on only ~17% of steps and the slack majority decouples the game;
#   (b) the PIPELINE's Tier-2 fallback uses rho=0.5 (rhg_online.py:433), which reaches
#       1e-4 kW on only 1 of 12 binding steps within 1500 iterations. The fallback is
#       mistuned — it stops on its native primal/dual test at ~2.5e-2 kW. Physically
#       harmless (0.025 kW), but it must be reported, not hidden.
ADMM_RHO = 0.01
MAX_ITER = 3000

TOL_KW = 1e-4                         # = rhg_online.STRICT_TOL, the pipeline's own bar
BUDGETS = (0.010, 0.100, 1.000)       # seconds; 0.1 s brackets FACET's median and Benenati's
OUT = HERE.parent / "results" / "solver_bench.pkl"

WEEKS = {
    "April 1-7": [f"2025-04-0{d}" for d in range(1, 8)],
    "July 7-13": [f"2025-07-{d:02d}" for d in range(7, 14)],
}


def _err_curve(trace, x_ref):
    """(times, errs) from a [(t_k, x_k)] trace against the reference solution."""
    if not trace:
        return np.array([]), np.array([])
    ts = np.array([t for t, _ in trace], float)
    es = np.array([float(np.max(np.abs(x - x_ref))) for _, x in trace], float)
    return ts, es


def _summarize(ts, es):
    """time-to-TOL_KW and err-at-each-budget from an error curve."""
    out = {"t_tol": np.nan, "err_at": {b: np.nan for b in BUDGETS}, "err_final": np.nan}
    if ts.size == 0:
        return out
    out["err_final"] = float(es[-1])
    hit = np.where(es < TOL_KW)[0]
    if hit.size:
        out["t_tol"] = float(ts[hit[0]])
    for b in BUDGETS:
        within = np.where(ts <= b)[0]
        out["err_at"][b] = float(es[within[-1]]) if within.size else float("inf")
    return out


def bench_step(theta, x_ref, sols, game, proj, warm_x, warm_combo, mode):
    """Run FACET and ADMM on one theta. `mode` in {'warm','cold'}."""
    rec = {}
    cold = (mode == "cold")

    # ── FACET-RHG ────────────────────────────────────────────────────────────
    # Timed end-to-end via solve_step, so a Tier-2 ADMM fallback IS billed to FACET.
    px = None if cold else warm_x
    pc = None if cold else warm_combo
    st = {}
    t0 = time.perf_counter()
    xf, combo = O.solve_step(theta, sols, game, px, pc, stats=st, max_hops=3)
    tf = time.perf_counter() - t0
    fb = st.get("fallback", 0) > 0
    rec["facet"] = {
        "t_tol": tf if np.max(np.abs(xf - x_ref)) < TOL_KW else np.nan,
        "t_wall": tf,
        "err_final": float(np.max(np.abs(xf - x_ref))),
        "err_at": {b: (float(np.max(np.abs(xf - x_ref))) if tf <= b else float("inf"))
                   for b in BUDGETS},
        "r_final": natural_residual(game, xf, theta, proj),
        "rounds": int(st.get("transfers", [1])[-1]),
        "fallback": bool(fb),
    }

    # ── ADMM (model-private competitor) ──────────────────────────────────────
    # Early-stop on the SHARED metric (a bit past TOL_KW so the curve is resolved there),
    # not on ADMM's native primal/dual test. Without this every run would grind through
    # MAX_ITER (~9 s) even after converging.
    stop = lambda xk: float(np.max(np.abs(xk - x_ref))) < TOL_KW * 0.1
    res = admm_solve(game, theta, rho=ADMM_RHO, max_iter=MAX_ITER, tol=1e-12,
                     x_init=(None if cold else warm_x), qp_solver="osqp", trace=True,
                     stop_fn=stop)
    ts, es = _err_curve(res.x_trace, x_ref)
    s = _summarize(ts, es)
    s.update(t_wall=res.solve_time, rounds=int(res.n_iter),
             r_final=natural_residual(game, res.x_stacked, theta, proj), fallback=False)
    rec["admm"] = s


    return rec, xf, combo


def bench_day(day, sols, game, seed_rng, verbose=True):
    """Run the real pipeline for `day` (to get its exact theta + FACET's own timings),
    then replay those theta to every method in both blocks."""
    if verbose:
        print(f"\n{'='*72}\n{day}\n{'='*72}", flush=True)
    res = W.run_day(day, sols, game, seed_rng, verbose=False)
    thetas = res["th_step"]
    proj = get_projector(game)

    refs, binding = [], []
    for t in range(96):
        xr = reference_solution(game, thetas[t])
        refs.append(xr)
        sp = sum(xr[game.x_slice(i).start] for i in range(game.N))
        binding.append(bool(abs(sp - R.L_MAX) < 1e-3 or abs(sp - R.L_MIN) < 1e-3))
    if verbose:
        print(f"  reference solutions built | binding steps: {sum(binding)}/96", flush=True)

    # Warm-up: touch every timed code path once, untimed, so the first measured step is not
    # inflated by first-touch costs (OSQP setup, BLAS init, lazy caches). Audited as ~0 on
    # this machine (128.1 ms first vs 127.2 ms median), but it costs one step to be sure.
    bench_step(thetas[0], refs[0], sols, game, proj, refs[0], None, "warm")

    out = {"day": day, "binding": np.array(binding), "theta": thetas,
           "facet_pipeline_t_step": res["t_step"], "facet_pipeline_fb": res["fb_step"],
           "blocks": {}}
    for mode in ("warm", "cold"):
        recs = []
        wx, wc = None, None
        for t in range(96):
            if mode == "warm" and t > 0:
                wx = refs[t - 1]          # identical, best-possible warm start for ALL methods
            rec, xf, combo = bench_step(thetas[t], refs[t], sols, game, proj, wx, wc, mode)
            if mode == "warm":
                wc = combo
            recs.append(rec)
            if verbose and t % 24 == 0:
                print(f"  [{mode}] step {t:2d}  facet {rec['facet']['t_wall']*1e3:7.1f} ms | "
                      f"admm {rec['admm']['t_wall']*1e3:8.1f} ms", flush=True)
        out["blocks"][mode] = recs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None, help="single day (Phase 1 smoke test)")
    ap.add_argument("--weeks", action="store_true", help="both full weeks (Phase 2)")
    a = ap.parse_args()

    sols, game = O.load_and_prepare()
    import dam as F
    days = ([a.day] if a.day else
            [d for ds in WEEKS.values() for d in ds] if a.weeks else ["2025-04-06"])

    all_out = []
    t0 = time.perf_counter()
    for d in days:
        all_out.append(bench_day(d, sols, game, np.random.default_rng(F.SEED)))
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump({"days": all_out, "tol_kw": TOL_KW, "budgets": BUDGETS,
                     "admm_rho": ADMM_RHO, "max_iter": MAX_ITER}, f)
    print(f"\ntotal {time.perf_counter()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
