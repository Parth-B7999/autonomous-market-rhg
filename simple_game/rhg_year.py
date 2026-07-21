"""
rhg_year.py — full-year (ERCOT 2025) closed-loop run + WARM-block ADMM benchmark.

Runs the H=4 receding-horizon mp-GNE case study over every 2025 day and, on the SAME theta
sequence, the FACET-vs-ADMM head-to-head (warm block only — the deployment-realistic regime;
the cold-start block is already reported on the 2-week sample, see STATUS_solver_comparison.md).

Per day it SAVES BOTH:
  * "sim"   — the full run_day economic result (dispatch P, H2, curtailment, prices, comms,
              crhits, e_cent, ...) — the substrate for daily / weekly / seasonal analysis.
  * "bench" — the warm FACET-vs-ADMM per-step metrics (t_wall, t_tol, err_at, rounds,
              fallback) + the binding mask + pipeline timings, exactly as bench_solvers builds
              them (its validated bench_step is reused, so the fairness rules are unchanged).

Design for a long (~15 h) background run:
  * DST / data-gap days skipped (load() uses bare [:96]/[:24] slices that break otherwise).
  * Per-day DETERMINISTIC seed (from the date) → each day reproducible / re-runnable alone.
  * CHECKPOINT + RESUME: one pickle per day, written atomically; a day already on disk is
    skipped, so a crash costs at most the day in flight.
  * manifest.json rewritten each day (provenance + progress + any failures).
  * At the end, a report_bench-compatible consolidated results/solver_bench_year.pkl
    (WARM block only — see note in _write_consolidated).

Usage:
  python simple_game/rhg_year.py                 # full year (resumes)
  python simple_game/rhg_year.py --smoke 2       # first 2 runnable days only (validation)
  python simple_game/rhg_year.py --days 2025-04-06,2025-07-11
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import rhg_mpqp as R          # noqa: E402
import rhg_online as O        # noqa: E402
import rhg_week as W          # noqa: E402
import bench_solvers as B     # noqa: E402  (reuse the validated bench_step + tuned params)
import dam as F               # noqa: E402
from amrhg.solvers.residual import get_projector, reference_solution  # noqa: E402

YEAR = 2025
# Non-96-interval days: DST spring-forward (RTM 92 / solar 23 h), fall-back (RTM 100), and a
# lone RTM gap on 12-04 (95). load()'s [:96]/[:24] slices produce short arrays on these and
# crash theta assembly, so they are excluded from the year run (verified against the CSVs).
SKIP_DAYS = {"2025-03-09", "2025-11-02", "2025-12-04"}

OUTDIR = HERE.parent / "results" / "year2025"
MANIFEST = OUTDIR / "manifest.json"
CONSOLIDATED = HERE.parent / "results" / "solver_bench_year.pkl"


def all_days():
    """Every 2025 calendar day (YYYY-MM-DD), DST/gap days removed."""
    d0, d1 = date(YEAR, 1, 1), date(YEAR, 12, 31)
    out, d = [], d0
    while d <= d1:
        s = d.isoformat()
        if s not in SKIP_DAYS:
            out.append(s)
        d += timedelta(days=1)
    return out


def seed_for(day: str) -> int:
    """Deterministic per-day seed: the date as an int (2025-04-06 -> 20250406)."""
    return int(day.replace("-", ""))


def bench_day_warm(day, sols, game):
    """run_day (full economics) + the WARM FACET-vs-ADMM block on the identical theta.

    Mirrors bench_solvers.bench_day but (a) keeps the ENTIRE run_day result and (b) runs the
    warm block only. Uses B.bench_step verbatim so the measured comparison is byte-for-byte the
    validated harness."""
    rng = np.random.default_rng(seed_for(day))
    sim = W.run_day(day, sols, game, rng, verbose=False)      # full economic dict — kept whole
    thetas = sim["th_step"]
    proj = get_projector(game)

    refs, binding = [], []
    for t in range(96):
        xr = reference_solution(game, thetas[t])              # Gurobi oracle @ 1e-9
        refs.append(xr)
        sp = sum(xr[game.x_slice(i).start] for i in range(game.N))
        binding.append(bool(abs(sp - R.L_MAX) < 1e-3 or abs(sp - R.L_MIN) < 1e-3))

    # Warm-up one step untimed (first-touch: OSQP setup / BLAS init), as in bench_day.
    B.bench_step(thetas[0], refs[0], sols, game, proj, refs[0], None, "warm")

    recs, wx, wc = [], None, None
    for t in range(96):
        if t > 0:
            wx = refs[t - 1]                                  # identical best-possible warm start
        rec, _xf, combo = B.bench_step(thetas[t], refs[t], sols, game, proj, wx, wc, "warm")
        wc = combo
        recs.append(rec)

    bench = {"day": day, "binding": np.array(binding),
             "facet_pipeline_t_step": sim["t_step"], "facet_pipeline_fb": sim["fb_step"],
             "blocks": {"warm": recs}}
    return {"day": day, "sim": sim, "bench": bench}


def _atomic_dump(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)                                     # atomic on POSIX


def _write_manifest(days_all, done, failed):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "market": "ercot", "band_kW": [R.L_MIN, R.L_MAX], "H": R.H, "N": R.N,
        "seed_policy": "per-day: int(YYYYMMDD)",
        "block": "warm-only (cold reported on 2-week sample; see STATUS_solver_comparison.md)",
        "admm_rho": B.ADMM_RHO, "max_iter": B.MAX_ITER, "tol_kw": B.TOL_KW, "budgets": list(B.BUDGETS),
        "skip_days": sorted(SKIP_DAYS),
        "days_total": len(days_all), "days_done": len(done), "days_failed": failed,
        "done": done,
    }, indent=2))


def _write_consolidated(days_all):
    """Assemble a report_bench-shaped pkl from the per-day 'bench' dicts.

    WARM BLOCK ONLY: each day carries blocks={'warm': [...]}. report_bench.py as written
    loops for mode in ('warm','cold') and will KeyError/empty-crash on the missing 'cold';
    point it here only after guarding that loop to skip a mode absent from blocks. The
    validated 2-week results/solver_bench.pkl is left untouched."""
    days = []
    for day in days_all:
        p = OUTDIR / f"bench_day={day}.pkl"
        if p.exists():
            days.append(pickle.load(open(p, "rb"))["bench"])
    _atomic_dump({"days": days, "tol_kw": B.TOL_KW, "budgets": B.BUDGETS,
                  "admm_rho": B.ADMM_RHO, "max_iter": B.MAX_ITER}, CONSOLIDATED)
    return len(days)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default=None, help="comma-separated subset (default: full year)")
    ap.add_argument("--smoke", type=int, default=0, help="run only the first N runnable days")
    ap.add_argument("--no-consolidate", action="store_true", help="skip the final consolidated pkl")
    a = ap.parse_args()

    days = a.days.split(",") if a.days else all_days()
    if a.smoke:
        days = days[:a.smoke]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"YEAR RUN — ERCOT {YEAR}, {len(days)} days (warm ADMM benchmark) → {OUTDIR}")
    print("=" * 78, flush=True)
    t0 = time.perf_counter()
    print("loading offline map + neighbour graphs ...", flush=True)
    sols, game = O.load_and_prepare()
    print(f"  ready ({time.perf_counter()-t0:.0f}s)  CRs={[s.n_cr for s in sols]}  n_p={game.n_p}",
          flush=True)

    done, failed = [], {}
    for k, day in enumerate(days):
        path = OUTDIR / f"bench_day={day}.pkl"
        if path.exists():                                    # resume: already computed
            done.append(day)
            print(f"[{k+1}/{len(days)}] {day}  — skip (on disk)", flush=True)
            continue
        td = time.perf_counter()
        try:
            out = bench_day_warm(day, sols, game)
            _atomic_dump(out, path)
            done.append(day)
            s = out["sim"]
            h2pct = 100 * s["h2"].sum() / s["d_day"].sum()
            nfb = s["comm"].get("fallback", 0)
            print(f"[{k+1}/{len(days)}] {day}  H2 {h2pct:3.0f}%  fallbacks {nfb}/96  "
                  f"binding {int(out['bench']['binding'].sum())}/96  ({time.perf_counter()-td:.0f}s)",
                  flush=True)
        except Exception as e:                               # never let one day kill the year
            failed[day] = repr(e)
            print(f"[{k+1}/{len(days)}] {day}  — FAILED: {e!r}", flush=True)
            traceback.print_exc()
        _write_manifest(days, done, failed)                  # progress survivable at all times

    dt = time.perf_counter() - t0
    print(f"\nyear loop done: {len(done)}/{len(days)} ok, {len(failed)} failed  ({dt/3600:.1f} h)",
          flush=True)
    if not a.no_consolidate:
        n = _write_consolidated(days)
        print(f"consolidated {n} days → {CONSOLIDATED}", flush=True)
    if failed:
        print(f"FAILED days: {failed}", flush=True)


if __name__ == "__main__":
    main()
