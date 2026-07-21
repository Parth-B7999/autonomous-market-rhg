"""
report_bench.py — turn results/solver_bench.pkl into the report's comparison table.

Every number the report quotes about FACET vs ADMM is produced here, directly from the
raw traces. Nothing is hand-transcribed (same discipline as report_numbers.py).

Usage:  python simple_game/report_bench.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "results" / "solver_bench.pkl"
METHODS = [("facet", "FACET-RHG"), ("admm", "ADMM")]


def stats(a, scale=1e3):
    a = np.asarray([v for v in a if np.isfinite(v)], float) * scale
    if a.size == 0:
        return None
    return dict(n=a.size, mean=a.mean(), med=np.median(a),
                p95=np.percentile(a, 95), hi=a.max())


def collect(days, mode, key, regime=None):
    """Gather `key` across all steps of all days for one method-agnostic view."""
    out = {m: [] for m, _ in METHODS}
    for d in days:
        bind = d["binding"]
        for t, rec in enumerate(d["blocks"][mode]):
            if regime == "bind" and not bind[t]:
                continue
            if regime == "slack" and bind[t]:
                continue
            for m, _ in METHODS:
                out[m].append(rec[m][key])
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(BENCH), help="bench pickle (default: 2-week solver_bench.pkl)")
    ap.add_argument("--suffix", default="", help="suffix for the emitted .tex files (e.g. _year)")
    a = ap.parse_args()
    bench_path = Path(a.bench)
    if not bench_path.exists():
        sys.exit(f"missing {bench_path} — run: python simple_game/bench_solvers.py --weeks")
    D = pickle.load(open(bench_path, "rb"))
    days, TOL, BUD = D["days"], D["tol_kw"], D["budgets"]
    # only tabulate blocks actually present on every day (year runs are warm-only)
    modes = [m for m in ("warm", "cold") if all(m in d["blocks"] for d in days)]
    nb = sum(int(d["binding"].sum()) for d in days)
    nt = sum(len(d["binding"]) for d in days)

    print("=" * 96)
    print(f"SOLVER COMPARISON — {len(days)} day(s), {nt} steps ({nb} binding, {nt-nb} slack)")
    print(f"  tuned: ADMM rho={D['admm_rho']} (swept on BINDING steps)  max_iter={D['max_iter']}")
    print(f"  shared metric: ||x - x_ref||_inf < {TOL:g} kW   (= pipeline STRICT_TOL)")
    print("=" * 96)

    for mode in modes:
        print(f"\n\n{'#'*96}\n### {mode.upper()} BLOCK — every method initialized identically\n{'#'*96}")
        for regime, label in [(None, "ALL STEPS"), ("bind", "BINDING (coupling active)"),
                              ("slack", "SLACK (interior)")]:
            # TIMING is t_wall over ALL steps — the full per-step wall clock a deployment pays,
            # INCLUDING FACET's ADMM fallback on the steps that fall back (t_tol excluded those,
            # understating FACET's cost). "reached tol" (from t_tol) stays as the accuracy column.
            tw = collect(days, mode, "t_wall", regime)
            tt = collect(days, mode, "t_tol", regime)
            rounds = collect(days, mode, "rounds", regime)
            n_tot = len(tw["facet"])
            if n_tot == 0:
                continue
            print(f"\n--- {label}  (n={n_tot}) ---   [timing = full solve wall time, fallback included]")
            print(f"{'method':<20} {'reached tol':>12} {'mean ms':>9} {'median ms':>10} "
                  f"{'p95 ms':>9} {'max ms':>9} {'med rounds':>11}")
            for m, name in METHODS:
                s = stats(tw[m])
                reach = int(np.sum(np.isfinite(tt[m])))
                r = np.median([v for v in rounds[m] if np.isfinite(v)])
                if s is None:
                    print(f"{name:<20} {reach:>6}/{n_tot:<5} {'—':>9}")
                    continue
                print(f"{name:<20} {reach:>6}/{n_tot:<5} {s['mean']:9.1f} {s['med']:10.1f} "
                      f"{s['p95']:9.1f} {s['hi']:9.1f} {r:11.0f}")

        # accuracy under a fixed budget (Benenati Fig. 4 style)
        print(f"\n--- accuracy under a fixed wall-clock budget: median ||x-x_ref||_inf [kW] ---")
        print(f"{'method':<20}" + "".join(f"{int(b*1000):>14d} ms" for b in BUD))
        for m, name in METHODS:
            row = f"{name:<20}"
            for b in BUD:
                vals = []
                for d in days:
                    for rec in d["blocks"][mode]:
                        vals.append(rec[m]["err_at"][b])
                fin = [v for v in vals if np.isfinite(v)]
                med = np.median(vals) if fin and len(fin) > len(vals) / 2 else np.inf
                row += f"{med:>17.3e}" if np.isfinite(med) else f"{'not reached':>17}"
            print(row)

    # ── the claim that actually matters: communication ────────────────────────
    print(f"\n\n{'#'*96}\n### COMMUNICATION (the model-private comparison)\n{'#'*96}")
    for mode in modes:
        r = collect(days, mode, "rounds", None)
        print(f"\n  [{mode}] inter-agent rounds per step (median / mean / max):")
        for m, name in METHODS:
            v = np.asarray([x for x in r[m] if np.isfinite(x)], float)
            tag = ""
            print(f"    {name:<20} {np.median(v):8.0f} / {v.mean():8.1f} / {v.max():6.0f}{tag}")

    # ── cross-check against the production pipeline ───────────────────────────
    tp = np.concatenate([d["facet_pipeline_t_step"] for d in days]) * 1e3
    tb = np.asarray([rec["facet"]["t_wall"] for d in days for rec in d["blocks"]["warm"]]) * 1e3
    print(f"\n\n{'#'*96}\n### CROSS-CHECK: bench harness vs production pipeline (FACET)\n{'#'*96}")
    print(f"  pipeline t_step : mean {tp.mean():7.1f} ms  median {np.median(tp):7.1f} ms")
    print(f"  bench    t_wall : mean {tb.mean():7.1f} ms  median {np.median(tb):7.1f} ms")
    ok = abs(np.median(tp) - np.median(tb)) / np.median(tp) < 0.25
    print(f"  medians within 25%? {'YES — harness agrees with the pipeline' if ok else 'NO — HARNESS SUSPECT'}")

    emit_latex(days, TOL, BUD, D["admm_rho"], D["max_iter"], modes, a.suffix)


# ─────────────────────────────────────────────────────────────────────────────
#  LaTeX emission — the report copies these rows verbatim; nothing hand-typed.
# ─────────────────────────────────────────────────────────────────────────────

def emit_latex(days, TOL, BUD, rho, max_iter, modes=("warm", "cold"), suffix=""):
    print("\n\n" + "%" * 96)
    print("% LaTeX — paste into report/rhg_detailed_report.tex (report_bench.py output)")
    print("%" * 96)

    def block(mode, regime, label):
        # timing = t_wall over ALL steps (full per-step wall clock, FACET fallback INCLUDED);
        # "reached tol" = accuracy, from finite t_tol.
        tw = collect(days, mode, "t_wall", regime)
        tt = collect(days, mode, "t_tol", regime)
        rd = collect(days, mode, "rounds", regime)
        n = len(tw["facet"])
        rows = []
        for m, name in METHODS:
            st = stats(tw[m])
            reach = int(np.sum(np.isfinite(tt[m])))
            r = np.median([v for v in rd[m] if np.isfinite(v)])
            if st is None:
                rows.append(f"{name} & {reach}/{n} & --- & --- & --- & --- & {r:.0f}\\\\")
            else:
                rows.append(f"{name} & {reach}/{n} & {st['mean']:.1f} & {st['med']:.1f} & "
                            f"{st['p95']:.1f} & {st['hi']:.1f} & {r:.0f}\\\\")
        return n, rows

    outdir = HERE.parent / "results"
    for mode in modes:
        L = ["\\begin{table}[H]\\centering\\small",
             "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
             "\\begin{tabular}{llcccccc}", "\\toprule",
             "regime & method & reached $10^{-4}$~kW & mean [ms] & median [ms] & p95 [ms] & "
             "max [ms] & median rounds\\\\", "\\midrule"]
        for regime, label in [(None, "all steps"), ("bind", "binding"), ("slack", "slack")]:
            n, rows = block(mode, regime, label)
            for j, r in enumerate(rows):
                pre = f"\\multirow{{{len(rows)}}}{{*}}{{{label} ($n{{=}}{n}$)}}" if j == 0 else ""
                L.append(f"{pre} & {r}")
            L.append("\\midrule")
        L += ["\\bottomrule", "\\end{tabular}}",
              f"\\caption{{{mode.capitalize()}-start block: FACET-RHG vs.\\ ADMM, every method "
              f"initialised identically. \\textbf{{Timing is the full per-step wall clock over all "
              f"steps --- FACET's figure INCLUDES its ADMM fallback on the steps that fall back.}} "
              f"``Reached $10^{{-4}}$~kW'' is accuracy vs.\\ the oracle. ADMM tuned at $\\rho={rho}$ "
              f"(swept on binding steps); its stop is oracle-assisted (favouring ADMM). Communication "
              f"(median rounds) is the implementation-independent claim.}}",
              f"\\label{{tab:bench-{mode}}}", "\\end{table}"]
        block_tex = "\n".join(L) + "\n"
        (outdir / f"bench_{mode}{suffix}_table.tex").write_text(block_tex)
        print(f"\n%% ---- {mode.upper()} block  (written to results/bench_{mode}{suffix}_table.tex) ----")
        print(block_tex)

    # the robust claim
    r = collect(days, "warm", "rounds", None)
    print("\n%% ---- communication (the implementation-independent claim) ----")
    print("\\begin{center}\\small\\begin{tabular}{lccc}")
    print("\\toprule")
    print("method & median & mean & max\\\\")
    print("\\midrule")
    for m, name in METHODS:
        v = np.asarray([x for x in r[m] if np.isfinite(x)], float)
        print(f"{name} & {np.median(v):.0f} & {v.mean():.1f} & {v.max():.0f}\\\\")
    print("\\bottomrule\\end{tabular}\\end{center}")


if __name__ == "__main__":
    main()
