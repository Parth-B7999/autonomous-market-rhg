"""
time_admm.py — Wall-clock instrumentation for the medium-scale GNE case study.

Standalone: imports nothing from this folder that it modifies, and writes only
results/timing_summary.txt.  No existing source file is touched.

Method
------
Monkey-patch the *names bound on the simulation_sc module object* (simulation_sc
does `from miqp_admm import solve_gne, ...` at line 35, so the call sites resolve
through simulation_sc's globals, not miqp_admm's).  Each wrapper records
time.perf_counter() around the original call, then delegates.

Three populations are separated:
  • solve_gne(mode='dam')   — 1/day day-ahead ADMM solve
  • solve_gne(mode='rtm')   — 1/5-min real-time ADMM dispatch
  • commit_u_gne(...)       — 1/hour 65-min-gate centralized MIQP

miqp_admm._solve_agent is also wrapped (a counter + timer only) to measure the
per-agent MIQP inside one ADMM iteration and to show, by construction, that the
six agent subproblems run one after another in a Python for-loop.

Run:  python3 time_admm.py            (full 7-day case study)
      python3 time_admm.py --days 1   (smoke test)
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── sys.path repair ───────────────────────────────────────────────────────────
# agents_med.py / simulation_sc.py / miqp_admm.py all append
# `../admm_elz_market`, which no longer exists — that folder is now
# `../small_scale_case_study`.  Append (never insert at 0) the real location so
# the shared modules (agents, forecasting) resolve while the med-local
# simulation_sc.py / miqp_admm.py keep priority.
_HERE   = Path(__file__).resolve().parent
_SHARED = _HERE.parent / 'small_scale_case_study'
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if _SHARED.is_dir() and str(_SHARED) not in sys.path:
    sys.path.append(str(_SHARED))

import simulation_sc                                                  # noqa: E402
import miqp_admm                                                      # noqa: E402
from agents import ElzAgent                                           # noqa: E402
from agents_med import (                                              # noqa: E402
    FLEET_DICTS, L_MIN, RHO_RT, RHO_DA, TOL_RT, TOL_DA,
    N_DAYS, START_DATE, SEED, H2_DEMAND_PROFILE,
)

L_MAX = float('inf')   # case_study_gne.py line 49


# ── Timing records ────────────────────────────────────────────────────────────
REC_DAM: list[dict] = []   # {t, n_iter, converged, n_sub}
REC_RTM: list[dict] = []
REC_CMT: list[dict] = []   # {t}
REC_SUB: list[float] = []  # per single-agent MIQP wall-clock [s]

_SUB_COUNT = [0]           # calls to _solve_agent, reset per solve_gne call


def _install_patches() -> None:
    """Wrap the solver names bound on the simulation_sc module object."""
    orig_solve_gne   = simulation_sc.solve_gne
    orig_commit_u    = simulation_sc.commit_u_gne
    orig_solve_agent = miqp_admm._solve_agent

    def timed_solve_agent(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_solve_agent(*a, **kw)
        finally:
            REC_SUB.append(time.perf_counter() - t0)
            _SUB_COUNT[0] += 1

    def timed_solve_gne(*a, **kw):
        # mode is passed as a keyword at both call sites (simulation_sc.py:221,
        # :359); fall back to the positional slot (index 13) defensively.
        mode = kw.get('mode', a[13] if len(a) > 13 else None)
        n0 = _SUB_COUNT[0]
        t0 = time.perf_counter()
        res = orig_solve_gne(*a, **kw)
        dt = time.perf_counter() - t0
        rec = {
            't':         dt,
            'n_iter':    int(getattr(res, 'n_iter', 1)),
            'converged': bool(getattr(res, 'converged', True)),
            'n_sub':     _SUB_COUNT[0] - n0,
        }
        (REC_DAM if mode == 'dam' else REC_RTM).append(rec)
        return res

    def timed_commit_u_gne(*a, **kw):
        t0 = time.perf_counter()
        res = orig_commit_u(*a, **kw)
        REC_CMT.append({'t': time.perf_counter() - t0})
        return res

    miqp_admm._solve_agent      = timed_solve_agent
    simulation_sc.solve_gne     = timed_solve_gne
    simulation_sc.commit_u_gne  = timed_commit_u_gne


# ── Environment probe ─────────────────────────────────────────────────────────
def _sysctl(key: str) -> str:
    try:
        return subprocess.check_output(['sysctl', '-n', key], text=True).strip()
    except Exception:
        return 'unknown'


def environment_lines() -> list[str]:
    import gurobipy as gp

    if platform.system() == 'Darwin':
        cpu   = _sysctl('machdep.cpu.brand_string')
        phys  = _sysctl('hw.physicalcpu')
        logi  = _sysctl('hw.logicalcpu')
        mem_b = _sysctl('hw.memsize')
        mem   = f"{int(mem_b)/2**30:.0f} GiB" if mem_b.isdigit() else 'unknown'
    else:
        cpu, phys, logi, mem = platform.processor(), 'unknown', 'unknown', 'unknown'

    gv = '.'.join(str(x) for x in gp.gurobi.version())
    try:
        from importlib.metadata import version as _v
        gpy = _v('gurobipy')
    except Exception:
        gpy = gv

    # Gurobi thread cap set by miqp_admm._env()
    return [
        f"CPU                 : {cpu}",
        f"Physical cores      : {phys}   (logical: {logi})",
        f"Memory              : {mem}",
        f"OS                  : {platform.system()} {platform.release()} ({platform.machine()})",
        f"Python              : {platform.python_version()} ({sys.executable})",
        f"gurobipy            : {gpy}   (Gurobi engine {gv})",
        f"Gurobi Threads param: 2  (set in miqp_admm._env(), per MIQP)",
        f"numpy               : {np.__version__}",
    ]


# ── Statistics ────────────────────────────────────────────────────────────────
def stats_ms(times_s: list[float]) -> dict:
    a = np.asarray(times_s, dtype=float) * 1e3
    return {
        'n':      a.size,
        'mean':   float(a.mean()),
        'median': float(np.median(a)),
        'p95':    float(np.percentile(a, 95)),
        'max':    float(a.max()),
        'min':    float(a.min()),
        'total':  float(a.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=N_DAYS,
                    help=f'days to simulate (default {N_DAYS}, the case-study value)')
    args = ap.parse_args()

    env_lines = environment_lines()
    print('\n'.join(env_lines))

    _install_patches()

    fleet = [ElzAgent(h2_target_kg_day=0.0, **d) for d in FLEET_DICTS]

    print(f"\nGNE case study — seed={SEED}  start={START_DATE}  days={args.days}  "
          f"L_min={L_MIN:.0f} kW  L_max=inf")
    print("Timing every solver call ...\n")

    t_wall0 = time.perf_counter()
    res, agents = simulation_sc.run_gne(
        agents      = fleet,
        n_days      = args.days,
        start_date  = START_DATE,
        seed        = SEED,
        rho         = RHO_RT,
        rho_da      = RHO_DA,
        tol_rt      = TOL_RT,
        tol_da      = TOL_DA,
        l_min       = L_MIN,
        l_max       = L_MAX,
        h2_profile  = H2_DEMAND_PROFILE,
    )
    wall_total = time.perf_counter() - t_wall0

    report = build_report(args.days, wall_total, res, env_lines)
    print(report)

    out = _HERE / 'results' / 'timing_summary.txt'
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(f"\nWritten to {out}")

    raw = _HERE / 'results' / 'timing_raw.npz'
    np.savez_compressed(
        raw,
        rtm_time_s   = np.array([r['t'] for r in REC_RTM]),
        rtm_n_iter   = np.array([r['n_iter'] for r in REC_RTM]),
        rtm_converged= np.array([r['converged'] for r in REC_RTM]),
        dam_time_s   = np.array([r['t'] for r in REC_DAM]),
        dam_n_iter   = np.array([r['n_iter'] for r in REC_DAM]),
        commit_time_s= np.array([r['t'] for r in REC_CMT]),
        subproblem_time_s = np.array(REC_SUB),
    )
    print(f"Raw per-call timings -> {raw}")


def build_report(n_days: int, wall_total: float, res, env_lines: list[str]) -> str:
    rt  = stats_ms([r['t'] for r in REC_RTM])
    cm  = stats_ms([r['t'] for r in REC_CMT])
    dam = [r['t'] for r in REC_DAM]
    sub = stats_ms(REC_SUB)

    rt_iters = np.array([r['n_iter'] for r in REC_RTM], dtype=float)
    rt_conv  = np.array([r['converged'] for r in REC_RTM], dtype=bool)
    rt_subs  = np.array([r['n_sub'] for r in REC_RTM], dtype=float)

    total_rt_iters = float(rt_iters.sum())
    ms_per_iter    = rt['total'] / total_rt_iters if total_rt_iters else float('nan')

    worst_p95_p95  = rt['p95']  + cm['p95']
    worst_p95_mean = rt['p95']  + cm['mean']
    worst_max_max  = rt['max']  + cm['max']
    INTERVAL_MS    = 5 * 60 * 1000.0

    N = len(res.rt_conv[0]) and res.p_net.shape[2]
    L = []
    a = L.append
    a("=" * 78)
    a("WALL-CLOCK TIMING — medium-scale GNE case study (MIQP-ADMM)")
    a("=" * 78)
    a(f"Generated : {datetime.now().isoformat(timespec='seconds')}")
    a(f"Script    : time_admm.py  (monkey-patched timers; no source file modified)")
    a(f"Config    : {n_days} days x 288 steps, {N} agents, seed={SEED}, "
      f"start={START_DATE}, L_min={L_MIN:.0f} kW, L_max=inf")
    a(f"ADMM      : rho_rt={RHO_RT}, tol_rt={TOL_RT} kW, max_iter=300  |  "
      f"rho_da={RHO_DA}, tol_da={TOL_DA} kW, max_iter=2000")
    a(f"Total run : {wall_total:.1f} s ({wall_total/60:.1f} min) end-to-end")
    a("")
    a("-" * 78)
    a("ENVIRONMENT")
    a("-" * 78)
    L.extend(env_lines)
    a("")
    a("-" * 78)
    a("CALL COUNTS")
    a("-" * 78)
    a(f"Day-ahead  solve_gne(mode='dam') : {len(dam):6d}   (expected {n_days})")
    a(f"Real-time  solve_gne(mode='rtm') : {rt['n']:6d}   (expected {n_days*288})")
    a(f"Hourly     commit_u_gne(...)     : {cm['n']:6d}   (expected {n_days*24})")
    a(f"Single-agent MIQPs (_solve_agent): {sub['n']:6d}")
    a("")
    a("-" * 78)
    a("1. REAL-TIME solve_gne  (mode='rtm', every 5 min, H=12)")
    a("-" * 78)
    a(f"  mean                  : {rt['mean']:9.2f} ms")
    a(f"  median                : {rt['median']:9.2f} ms")
    a(f"  95th percentile       : {rt['p95']:9.2f} ms")
    a(f"  max                   : {rt['max']:9.2f} ms")
    a(f"  min                   : {rt['min']:9.2f} ms")
    a(f"  total wall-clock      : {rt['total']/1e3:9.2f} s   over {rt['n']} calls")
    a(f"  total ADMM iterations : {total_rt_iters:9.0f}")
    a(f"  mean time per ADMM iteration : {ms_per_iter:.2f} ms"
      f"   (= total time / total iterations)")
    a(f"  -> per agent-MIQP within an iteration : ~{ms_per_iter/6:.2f} ms "
      f"(6 agents, solved in series)")
    a("")
    a("  Tail breakdown — the mean sits ABOVE the p95, so it is not a typical")
    a("  cost.  A small number of steps exhaust max_iter=300 without meeting")
    a("  tol_rt and dominate the total:")
    rt_t = np.asarray([r['t'] for r in REC_RTM]) * 1e3
    capped = rt_iters >= 300
    conv_m = rt_conv
    for label, mask in (("converged (tol_rt met)", conv_m),
                        ("NOT converged", ~conv_m),
                        ("hit max_iter=300 cap", capped)):
        if mask.sum() == 0:
            a(f"    {label:24s}: none")
            continue
        sl = rt_t[mask]
        a(f"    {label:24s}: n={int(mask.sum()):5d} "
          f"({mask.mean()*100:5.2f} %)  mean {sl.mean():8.2f} ms  "
          f"median {np.median(sl):7.2f} ms  p95 {np.percentile(sl,95):8.2f} ms  "
          f"max {sl.max():8.2f} ms  total {sl.sum()/1e3:6.2f} s "
          f"({sl.sum()/rt_t.sum()*100:5.2f} % of all RT time)")
    a(f"    iterations in capped steps : {rt_iters[capped].sum():.0f} of "
      f"{total_rt_iters:.0f} ({rt_iters[capped].sum()/total_rt_iters*100:.1f} %)")
    a("    => quote the MEDIAN and p95 as the operating cost; quote the mean")
    a("       only alongside the cap, or it misrepresents the typical step.")
    a("")
    a("-" * 78)
    a("2. HOURLY commit_u_gne  (65-min gate, one centralized MIQP over all agents)")
    a("-" * 78)
    a(f"  mean                  : {cm['mean']:9.2f} ms")
    a(f"  median                : {cm['median']:9.2f} ms")
    a(f"  95th percentile       : {cm['p95']:9.2f} ms")
    a(f"  max                   : {cm['max']:9.2f} ms")
    a(f"  total wall-clock      : {cm['total']/1e3:9.2f} s   over {cm['n']} calls")
    a("")
    a("-" * 78)
    a("3. DAY-AHEAD solve_gne  (mode='dam', once per day, H=24)")
    a("-" * 78)
    for i, t in enumerate(dam):
        r = REC_DAM[i]
        a(f"  day {i+1}: {t:8.3f} s   ({r['n_iter']} ADMM iters, "
          f"{'converged' if r['converged'] else 'NOT converged'}, "
          f"{r['n_sub']} agent MIQPs)")
    if dam:
        d = np.asarray(dam)
        a(f"  mean  : {d.mean():8.3f} s")
        a(f"  median: {np.median(d):8.3f} s")
        a(f"  max   : {d.max():8.3f} s")
        a(f"  total : {d.sum():8.3f} s")
    a("")
    a("-" * 78)
    a("4. COMBINED WORST CASE PER 5-MINUTE INTERVAL")
    a("-" * 78)
    a("   The 65-min commitment MIQP lands inside one of the twelve 5-min")
    a("   intervals of each hour, so that interval carries both solves.")
    a(f"   p95 real-time + p95 commitment : {worst_p95_p95:9.2f} ms   "
      f"({worst_p95_p95/INTERVAL_MS*100:.3f} % of the 300 s interval)")
    a(f"   p95 real-time + mean commitment: {worst_p95_mean:9.2f} ms   "
      f"({worst_p95_mean/INTERVAL_MS*100:.3f} %)")
    a(f"   max real-time + max commitment : {worst_max_max:9.2f} ms   "
      f"({worst_max_max/INTERVAL_MS*100:.3f} %)  [absolute observed worst]")
    a(f"   Real-time budget per interval  : {INTERVAL_MS/1e3:9.1f} s")
    a(f"   Worst-case margin (p95+p95)    : {INTERVAL_MS/worst_p95_p95:9.0f}x "
      f"faster than real time")
    a("")
    a("-" * 78)
    a("5. SANITY CHECK vs the existing run")
    a("-" * 78)
    a(f"  real-time calls with converged == True : "
      f"{rt_conv.mean()*100:.2f} %  ({int(rt_conv.sum())}/{rt_conv.size})")
    a(f"  median n_iter (real-time)              : {np.median(rt_iters):.1f}")
    a(f"  mean   n_iter (real-time)              : {rt_iters.mean():.2f}")
    a(f"  max    n_iter (real-time)              : {rt_iters.max():.0f}")
    a("  cross-check against SimResults arrays returned by run_gne:")
    a(f"    res.rt_conv.mean()  = {res.rt_conv.mean()*100:.2f} %")
    a(f"    median(res.rt_iters)= {np.median(res.rt_iters):.1f}")
    a(f"    res.da_iters        = {list(res.da_iters)}")
    a(f"    res.da_conv         = {list(map(bool, res.da_conv))}")
    a("")
    a("-" * 78)
    a("6. ARE THE SIX PER-AGENT MIQPs IN ONE ADMM ITERATION PARALLEL?")
    a("-" * 78)
    a("  NO — they are SEQUENTIAL, in a plain Python for-loop.")
    a("  miqp_admm.solve_gne, x-update (miqp_admm.py:448-461):")
    a("      for i, ag in enumerate(agents):")
    a("          sol = _solve_agent(ag, H, dt, ...)")
    a("  One gurobipy Model is built, optimized and disposed per agent per")
    a("  iteration.  There is no multiprocessing, threading, joblib or async")
    a("  anywhere in the module.  (The solve_gne docstring line 399 says")
    a("  '(in parallel)' — that describes the ALGORITHM's data dependence, i.e.")
    a("  the x-updates are independent given z and mu and COULD be run in")
    a("  parallel, not what this implementation does.)")
    a("  The only parallelism is inside each individual MIQP: Gurobi is given")
    a("  Threads=2 (miqp_admm._env(), miqp_admm.py:80).")
    a("")
    a("  Measured evidence:")
    a(f"    _solve_agent calls per real-time solve_gne : "
      f"mean {rt_subs.mean():.2f}, "
      f"= 6 x n_iter for {int((rt_subs == 6*rt_iters).sum())}/{rt_subs.size} calls")
    a(f"    mean single-agent MIQP                     : {sub['mean']:.3f} ms")
    a(f"    median single-agent MIQP                   : {sub['median']:.3f} ms")
    a(f"    6 x mean single-agent MIQP                 : {6*sub['mean']:.3f} ms")
    a(f"    measured mean per ADMM iteration           : {ms_per_iter:.3f} ms")
    a(f"    ratio (measured iteration / 6x single)     : "
      f"{ms_per_iter/(6*sub['mean']):.3f}"
      "   (~1.0 => strictly serial; ~1/6 would indicate 6-way parallelism)")
    a("")
    a("  For the paper: claim SEQUENTIAL. The reported per-step times are the")
    a("  cost of solving the six agent subproblems one after another on a")
    a("  single machine; a genuinely distributed deployment (one solver per")
    a("  agent) would be bounded by the slowest agent, not their sum.")
    a("=" * 78)
    return "\n".join(L) + "\n"


if __name__ == '__main__':
    main()
