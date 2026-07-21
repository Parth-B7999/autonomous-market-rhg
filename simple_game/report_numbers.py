"""
report_numbers.py — dump EVERY number the LaTeX report quotes, straight from the cached run.

Run AFTER report_figs.py has populated results/report_fig_data.pkl. Prints ready-to-paste
LaTeX rows so the report tables are never hand-transcribed.

    python simple_game/report_numbers.py
"""
import sys, pickle
from pathlib import Path
from collections import Counter

sys.path.insert(0, "src"); sys.path.insert(0, "simple_game")
import numpy as np
import rhg_mpqp as R

CACHE = Path("results/report_fig_data.pkl")
SOLS = Path("simple_game/out/rhg_agent_sols.pkl")
N = R.N
names = [R.FLEET[i][0] for i in range(N)]


def tex(s):
    return s.replace("_", r"\_")


def main():
    if not CACHE.exists():
        sys.exit(f"missing {CACHE} — run `python simple_game/report_figs.py` first")
    data = pickle.load(open(CACHE, "rb"))

    # ── fleet table ──────────────────────────────────────────────────────────
    print("=" * 78)
    print("FLEET TABLE")
    print("=" * 78)
    for i, f in enumerate(R.FLEET):
        rc = "--" if f[1] == "grid" else f"{f[3]:.0f}"
        ty = {"grid": "grid", "solar": "PV", "wind": "wind"}[f[1]]
        gam = f"${f[5]*1e3:.0f}\\!\\times\\!10^{{-3}}$"
        print(f"{i} & {tex(f[0]):14s} & {ty:4s} & {f[2]:.0f} & {rc:3s} & {f[4]:.4f} & "
              f"{gam} & {R._a(i)*1e3:.1f}\\\\")
    print(f"% sum p_max = {sum(R._pmax(i) for i in range(N)):.0f} kW  vs L_MAX = {R.L_MAX:.0f}")

    # ── CR table ─────────────────────────────────────────────────────────────
    if SOLS.exists():
        sols = pickle.load(open(SOLS, "rb"))
        ncr = [len(s.regions) for s in sols]
        print()
        print("=" * 78)
        print("CR TABLE  (tab:cr)")
        print("=" * 78)
        for i in range(N):
            nt = R._priv_layout(i)["n_theta"]; nx = R._priv_layout(i)["nx"]
            ty = "grid" if not R._is_ren(i) else "renewable"
            print(f"{tex(names[i]):20s} & {ty:10s} & {nt} & $\\mathbb R^{{{nx}}}$ & {ncr[i]}\\\\")
        print(f"\\textbf{{Total}} & & & & \\textbf{{{sum(ncr):,}}} ({N} distinct solves)\\\\")
        kn = float(np.prod([float(c) for c in ncr]))
        print(f"% K^N = prod(ncr) = {kn:.3g}")
    else:
        ncr, kn = None, None
        print("\n(!) offline map pkl missing — CR table skipped")

    # ── per-week day tables ──────────────────────────────────────────────────
    tot_fb = 0; tot_steps = 0
    for wk, rows in data.items():
        print()
        print("=" * 78)
        print(f"WEEK TABLE — {wk}")
        print("=" * 78)
        wk_curt = 0.0; wk_avail = 0.0; wk_h2 = 0.0; wk_tgt = 0.0
        fbs = 0; ecs = []
        for r in rows:
            fb = r["comm"].get("fallback", 0)
            lam = r["lam_rt"]
            curt = float(np.sum(r["curt"])); avail = float(np.sum(r["ren_avail"]))
            h2 = float(np.sum(r["h2"])); tgt = float(np.sum(r["d_day"]))
            wk_curt += curt; wk_avail += avail; wk_h2 += h2; wk_tgt += tgt
            fbs += fb; ecs.append(r["e_cent"])
            day = r["day"][5:]
            print(f"{day} & ${lam.min():.0f}$--${lam.max():.0f}$ & ${r['e_cent']:.3g}$ & "
                  f"{fb} & {100*h2/tgt:.0f}\\% & {curt:.0f}\\\\")
        tot_fb += fbs; tot_steps += 96 * len(rows)
        print(f"\\textbf{{{wk}}} & & \\textbf{{$\\le{max(ecs):.2g}$}} & "
              f"\\textbf{{{fbs}/{96*len(rows)}}} & \\textbf{{{100*wk_h2/wk_tgt:.0f}\\%}} & "
              f"\\textbf{{{wk_curt:.0f}}}\\\\")
        print(f"% iteration-free = {100*(1-fbs/(96*len(rows))):.1f}%   "
              f"curtail {wk_curt:.0f}/{wk_avail:.0f} = {100*wk_curt/wk_avail:.1f}%   "
              f"max map=cent {max(ecs):.3g} kW")

    print()
    print("=" * 78)
    print("HEADLINE (both weeks)")
    print("=" * 78)
    print(f"fallbacks {tot_fb}/{tot_steps}  ->  iteration-free "
          f"{100*(1-tot_fb/tot_steps):.1f}%")

    # ── per-agent table ──────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("PER-AGENT TABLE  (all six distinct — no more 'each' rows)")
    print("=" * 78)
    wks = list(data.keys())
    per = {}
    for wk, rows in data.items():
        h2 = np.sum([r["h2"] for r in rows], axis=0)
        tgt = np.sum([r["d_day"] for r in rows], axis=0)
        cu = np.sum([r["curt"] for r in rows], axis=0)
        av = np.sum([r["ren_avail"] for r in rows], axis=0)
        per[wk] = (h2, tgt, cu, av)
    for i in range(N):
        cells = []
        for wk in wks:
            h2, tgt, cu, av = per[wk]
            c = "--- (grid)" if not R._is_ren(i) else f"${cu[i]:.0f}/{av[i]:.0f}$"
            cells.append(f"${h2[i]:.0f}/{tgt[i]:.0f}$ ({100*h2[i]/tgt[i]:.0f}\\%) & {c}")
        print(f"{tex(names[i]):14s} & " + " & ".join(cells) + "\\\\")

    # ── solve times ──────────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("SOLVE TIMES  (tab:timing)")
    print("=" * 78)

    def stats(a, unit="ms", scale=1e3):
        a = np.asarray(a, float) * scale
        if a.size == 0:
            return None
        return dict(n=a.size, mean=a.mean(), med=np.median(a), p95=np.percentile(a, 95),
                    lo=a.min(), hi=a.max(), tot=a.sum())

    have_t = all("t_step" in r for rows in data.values() for r in rows)
    if not have_t:
        print("(!) cached run predates per-step timing — delete results/report_fig_data.pkl "
              "and re-run report_figs.py")
        return

    rows_tex = []
    for wk, rows in data.items():
        dam_t = np.array([r["t_dam"] for r in rows])
        dam_it = np.array([r["dam_iters"] for r in rows])
        ts = np.concatenate([r["t_step"] for r in rows])
        fb = np.concatenate([r["fb_step"] for r in rows])
        d = stats(dam_t, scale=1.0)   # DAM in seconds
        m = stats(ts[~fb]); f = stats(ts[fb]); a = stats(ts)
        print(f"\n--- {wk} ---")
        print(f"DAM (1/day, {d['n']} solves): mean {d['mean']:.2f}s  median {d['med']:.2f}s  "
              f"range {d['lo']:.2f}-{d['hi']:.2f}s   ADMM iters mean {dam_it.mean():.0f} "
              f"median {np.median(dam_it):.0f}  converged {sum(r['dam_conv'] for r in rows)}/{len(rows)}")
        print(f"RTM all steps      (n={a['n']:4d}): mean {a['mean']:7.2f} ms  median {a['med']:7.2f} ms  "
              f"p95 {a['p95']:8.2f}  max {a['hi']:9.2f}")
        print(f"RTM map-resolved   (n={m['n']:4d}): mean {m['mean']:7.2f} ms  median {m['med']:7.2f} ms  "
              f"p95 {m['p95']:8.2f}  max {m['hi']:9.2f}")
        if f:
            print(f"RTM ADMM fallback  (n={f['n']:4d}): mean {f['mean']:7.2f} ms  median {f['med']:7.2f} ms  "
                  f"p95 {f['p95']:8.2f}  max {f['hi']:9.2f}   "
                  f"[{f['mean']/m['mean']:.0f}x slower than map]")
        else:
            print("RTM ADMM fallback  (n=   0): none")
        rows_tex.append((wk, d, m, f, a, dam_it))

    # both weeks pooled
    dam_t_all = np.array([r["t_dam"] for rows in data.values() for r in rows])
    dam_it_all = np.array([r["dam_iters"] for rows in data.values() for r in rows])
    ts_all = np.concatenate([r["t_step"] for rows in data.values() for r in rows])
    fb_all = np.concatenate([r["fb_step"] for rows in data.values() for r in rows])
    D = stats(dam_t_all, scale=1.0); M = stats(ts_all[~fb_all]); F_ = stats(ts_all[fb_all]); A = stats(ts_all)
    print(f"\n--- BOTH WEEKS ---")
    print(f"DAM   n={D['n']:4d}  mean {D['mean']:.2f}s  median {D['med']:.2f}s  "
          f"iters mean {dam_it_all.mean():.0f} median {np.median(dam_it_all):.0f}")
    print(f"RTM   n={A['n']:4d}  mean {A['mean']:.2f} ms  median {A['med']:.2f} ms")
    print(f"  map n={M['n']:4d}  mean {M['mean']:.2f} ms  median {M['med']:.2f} ms")
    if F_:
        print(f"  fb  n={F_['n']:4d}  mean {F_['mean']:.2f} ms  median {F_['med']:.2f} ms")
    print(f"\nRTM total compute per day = {A['mean']*96/1000:.2f}s vs 15-min (900s) real-time budget "
          f"-> {900/(A['mean']/1000):.0f}x headroom per step")

    print("\n" + "-" * 78)
    print("LaTeX rows for tab:timing")
    print("-" * 78)
    print(r"% Stage & n & mean & median & p95 & max \\")
    print(f"DAM (distributed ADMM, 1/day) & {D['n']} & {D['mean']:.2f}~s & {D['med']:.2f}~s & "
          f"{D['p95']:.2f}~s & {D['hi']:.2f}~s\\\\")
    print(f"\\quad ADMM iterations & {D['n']} & {dam_it_all.mean():.0f} & {np.median(dam_it_all):.0f} & "
          f"{np.percentile(dam_it_all,95):.0f} & {dam_it_all.max():.0f}\\\\")
    print(f"RTM FACET, map-resolved & {M['n']} & {M['mean']:.2f}~ms & {M['med']:.2f}~ms & "
          f"{M['p95']:.2f}~ms & {M['hi']:.2f}~ms\\\\")
    if F_:
        print(f"RTM ADMM fallback & {F_['n']} & {F_['mean']:.1f}~ms & {F_['med']:.1f}~ms & "
              f"{F_['p95']:.1f}~ms & {F_['hi']:.1f}~ms\\\\")
    print(f"RTM all steps & {A['n']} & {A['mean']:.2f}~ms & {A['med']:.2f}~ms & "
          f"{A['p95']:.2f}~ms & {A['hi']:.2f}~ms\\\\")

    # ── combination usage ────────────────────────────────────────────────────
    print()
    print("=" * 78)
    print("COMBINATION USAGE")
    print("=" * 78)
    allc = Counter(); perwk = {}
    for wk, rows in data.items():
        c = Counter(); daily = []
        for r in rows:
            d = Counter(r["crhits"])
            daily.append(len(d)); c.update(d)
        perwk[wk] = (c, daily)
        allc.update(c)
        print(f"{wk}: distinct/day = {min(daily)}--{max(daily)}, distinct over week = {len(c)}")
    print(f"BOTH WEEKS: {len(allc)} distinct combinations used")
    if allc:
        top, cnt = allc.most_common(1)[0]
        print(f"  most-used combination selected on {cnt} of {tot_steps} steps")
    if kn:
        print(f"  fraction of K^N = {len(allc)/kn:.2g}")


if __name__ == "__main__":
    main()
