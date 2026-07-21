"""
rhg_offline.py — one-time OFFLINE solve of the 6-agent H=4 game.

Solves 4 DISTINCT private mpQPs (PEM grid, ALK grid, PV, wind), expands each to its
agent slot(s) in p_gne (2 PV share one solve, 2 wind share one), and caches the expanded
AgentSolutions.  This is the offline artifact the online FACET step uses.
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import rhg_mpqp as R
from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm

ALGO = mpqp_algorithm.combinatorial_parallel
# distinct private solve → list of agent slots that share it
DISTINCT = {0: [0], 1: [1], 2: [2, 3], 4: [4, 5]}


def main():
    print("=" * 70)
    print("OFFLINE solve — 6-agent H=4 game (4 distinct private mpQPs)")
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
    from amrhg.solvers.facet_gne import find_all_agent_cr_neighbors
    print("\nBuilding facet-neighbour graph (hyperplane_adjacency) ...")
    t_nb = time.perf_counter()
    find_all_agent_cr_neighbors(sols, method="hyperplane_adjacency", verbose=True)
    total_nb = sum(len(cr.facet_neighbors) for s in sols for cr in s.regions)
    print(f"  Facet neighbours built: {total_nb} total pairs  ({time.perf_counter()-t_nb:.0f}s)")

    print(f"\ntotal offline solve {time.perf_counter()-t_all:.0f}s")
    out = HERE / "out" / "rhg_agent_sols.pkl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(sols, f)
    print(f"cached → {out}")


if __name__ == "__main__":
    main()


