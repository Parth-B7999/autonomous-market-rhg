"""
rhg_offline.py — one-time OFFLINE solve of the 6-agent H=4 game.

Solves one private mpQP per agent, expands each to its agent slot in p_gne, and caches the
expanded AgentSolutions.  This is the offline artifact the online FACET step uses.

All six agents are distinct (capacity + efficiency), so DISTINCT is the identity map and
there are 6 solves.  The mapping is kept explicit rather than hardcoded to identity so that
re-introducing shared/cloned agents only requires editing this dict.
"""
from __future__ import annotations
import sys, time, pickle, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import rhg_mpqp as R
from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm

# MUST stay combinatorial_parallel: it is exhaustive.  geometric_parallel_exp is not, and
# silently yields incomplete CR coverage -> online point-location misses -> ADMM fallbacks.
ALGO = mpqp_algorithm.combinatorial_parallel
# distinct private solve → list of agent slots that share it
DISTINCT = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5]}


def _check_distinct():
    """Guard: every agent covered exactly once, and slots sharing a solve really are identical."""
    slots = [j for v in DISTINCT.values() for j in v]
    assert sorted(slots) == list(range(R.N)), (
        f"DISTINCT covers {sorted(slots)}, expected all {R.N} agent slots"
    )
    for src, group in DISTINCT.items():
        for j in group:
            assert R.FLEET[j][1:] == R.FLEET[src][1:], (
                f"agent {j} ({R.FLEET[j][0]}) shares agent {src}'s ({R.FLEET[src][0]}) solve "
                f"but their FLEET params differ — fix DISTINCT in rhg_offline.py"
            )


def main():
    _check_distinct()
    n_solves = len(DISTINCT)
    print("=" * 70)
    print(f"OFFLINE solve — {R.N}-agent H=4 game ({n_solves} distinct private mpQPs)")
    print("=" * 70)
    priv = {}
    t_all = time.perf_counter()
    for i in DISTINCT:
        t = time.perf_counter()
        s, n = R.solve_agent_private(i, algorithm=ALGO, verbose=True)
        priv[i] = s
        print(f"    → {n} CRs  ({time.perf_counter()-t:.0f}s)", flush=True)
    # expand to all 6 slots
    sols = [None] * R.N
    for src, slots in DISTINCT.items():
        for j in slots:
            sols[j] = R.expand_for(priv[src], j)
    # ── Build facet-neighbour graph (CRITICAL for FACET online walk) ──
    # "facet_crossing" (not "hyperplane_adjacency"): the hash method requires an EXACT
    # match between two CRs' normalized facet coefficients and misses shared boundaries
    # represented with a different row scaling / redundant row. Measured 2026-07-16:
    # it recovered only ~12% of the true neighbour count on this fleet's maps, leaving
    # CRs with far too few recorded neighbours for the online walk to be useful (every
    # full-dimensional CR geometrically MUST have >=1 neighbour — CRs tile the space).
    # facet_crossing is exact (Chebyshev-centre-and-cross-the-facet, O(N·F)) and found
    # 0 zero-neighbour CRs on a smoke test where hyperplane_adjacency found ~40%.
    from amrhg.solvers.facet_gne import find_all_agent_cr_neighbors
    print("\nBuilding facet-neighbour graph (facet_crossing) ...")
    t_nb = time.perf_counter()
    find_all_agent_cr_neighbors(sols, method="facet_crossing", verbose=True)
    total_nb = sum(len(cr.facet_neighbors) for s in sols for cr in s.regions)
    print(f"  Facet neighbours built: {total_nb} total pairs  ({time.perf_counter()-t_nb:.0f}s)")

    print(f"\ntotal offline solve {time.perf_counter()-t_all:.0f}s")
    out = HERE / "out" / "rhg_agent_sols.pkl"
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix(".hall.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(sols, f)
    os.replace(tmp, out)
    print(f"cached → {out}")


if __name__ == "__main__":
    main()

