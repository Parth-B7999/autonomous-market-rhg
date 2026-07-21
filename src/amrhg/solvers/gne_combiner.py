"""
gne_combiner.py — Algorithm 1 (Steps 7-13) from Hall & Bemporad (2025).

Given each agent's explicit best-response mpQP solution (AgentSolution),
this module finds all valid GNE critical regions in p-space.

Analogy to dimpc IF-mpDiMPC
────────────────────────────
IF-mpDiMPC (_assemble_linear_system):
    Online, for a given x(k), search CR combinations → solve L U = R x(k) + d.

GNE combiner (here):
    Offline, parametric in p, for each CR combination → assemble M_x x* = M_p p + M_1
    then project the CR constraints to p-space and check for a non-empty polyhedron.

The key difference is that GNE combiner produces a PWA map  x*(p)  valid for ALL p,
not just for a single operating point.

Three-stage pipeline per combination C_k = (j_1, ..., j_N)
─────────────────────────────────────────────────────────────
1. _assemble_equilibrium_system   Build M_x, M_p, M_1  (paper Eq. 6)
2. _solve_equilibrium             Invert M_x (unique) or pseudoinverse (infinite)
3. _project_crs_to_p_space        Substitute x*(p) into each agent's CR → D p ≤ e
4. _cr_nonempty                   Chebyshev-center LP: is {p : D p ≤ e} non-empty?
5. build_gne_solution             Outer loop over all combinations → GNESolution

Notation (paper Eq. 6 → code)
──────────────────────────────
M_x x* = M_p p + M_1
  M_x ∈ R^{n_x × n_x}   : LHS matrix  (I on diagonal, -A_i_x blocks off-diagonal)
  M_p ∈ R^{n_x × n_p}   : p-coefficient from each agent's affine law
  M_1 ∈ R^{n_x}         : constant term from each agent's affine law

Unique case  rank(M_x) = n_x:
    x*(p) = H_x p + h_x   where H_x = M_x^{-1} M_p,  h_x = M_x^{-1} M_1

Infinite case  rank(M_x) < n_x  (min-norm selection):
    x*(p) = M_x^+ (M_p p + M_1)        (pseudoinverse, Eq. 9a in paper)
    solvability: U_2^T M_p = 0 and U_2^T M_1 = 0  (Eq. 10)
"""

from __future__ import annotations
import itertools
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.linalg import svd

from .game import GNEGame
from .cr_store import AgentSolution, GNECriticalRegion, GNESolution


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — build equilibrium linear system  (paper Eq. 6)
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_equilibrium_system(
    combo: tuple[int, ...],
    agent_solutions: list[AgentSolution],
    game: GNEGame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble  M_x x* = M_p p + M_1  from the combination C_k = (j_1,...,j_N).

    For each agent i with CR j_i, the affine best-response is:
        x_i* = A_i θ_i + b_i
             = A_i_x x_{-i}* + A_i_p p + b_i

    where θ_i = [x_{-i}; p], so:
        A_i_x = A_i[:, :n_x_neg_i]   (n_x_i, n_x_neg_i)
        A_i_p = A_i[:, n_x_neg_i:]   (n_x_i, n_p)

    Rearranged for all agents simultaneously:
        I x_i* - sum_{j≠i} A_i_x_j x_j* = A_i_p p + b_i

    In block matrix form → M_x x* = M_p p + M_1.

    Parameters
    ----------
    combo            : (j_0, j_1, ..., j_{N-1}) — one CR index per agent
    agent_solutions  : list of AgentSolution, one per agent
    game             : GNEGame

    Returns
    -------
    Mx : (n_x_total, n_x_total)
    Mp : (n_x_total, n_p)
    M1 : (n_x_total,)
    """
    N         = game.N
    n_x_total = game.n_x_total
    n_p       = game.n_p

    Mx = np.zeros((n_x_total, n_x_total))
    Mp = np.zeros((n_x_total, n_p))
    M1 = np.zeros(n_x_total)

    for i, j_i in enumerate(combo):
        cr      = agent_solutions[i][j_i]
        n_x_i   = game.agents[i].n_x
        # Auto-detect coupling mode from CR shape:
        #   individual mode: n_x_neg = n_x_total - n_x_i  (legacy 18-dim)
        #   sum mode:        n_x_neg = H < n_x_total - n_x_i  (6-dim)
        n_x_neg = cr.A.shape[1] - n_p
        use_sum = n_x_neg < (n_x_total - n_x_i)

        # Split affine law: A_i = [A_i_x | A_i_p]
        A_i_x = cr.A[:, :n_x_neg]   # (n_x_i, n_x_neg)
        A_i_p = cr.A[:, n_x_neg:]   # (n_x_i, n_p)

        row_s = game.x_slice(i).start
        row_e = game.x_slice(i).stop

        # Diagonal block: identity (x_i* on LHS)
        Mx[row_s:row_e, row_s:row_e] = np.eye(n_x_i)

        if use_sum:
            # Sum mode: A_i_x ∈ ℝ^{n_x_i × H} is the coefficient for sum_x_neg.
            # At equilibrium: sum_x_neg = Σ_{j≠i} x_j* → same block for all j≠i.
            for j in range(N):
                if j == i:
                    continue
                col_s = game.x_slice(j).start
                col_e = game.x_slice(j).stop
                Mx[row_s:row_e, col_s:col_e] = -A_i_x
        else:
            # Individual mode (legacy): x_{-i} ordering agents in index order, skip i
            col = 0
            for j in range(N):
                if j == i:
                    continue
                n_x_j = game.agents[j].n_x
                col_s  = game.x_slice(j).start
                col_e  = game.x_slice(j).stop
                Mx[row_s:row_e, col_s:col_e] = -A_i_x[:, col:col + n_x_j]
                col += n_x_j

        # RHS blocks
        Mp[row_s:row_e, :] = A_i_p
        M1[row_s:row_e]    = cr.b

    return Mx, Mp, M1


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 — solve M_x x* = M_p p + M_1
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _EquilibriumResult:
    H_x:       np.ndarray   # (n_x, n_p) — x*(p) = H_x p + h_x
    h_x:       np.ndarray   # (n_x,)
    is_unique: bool
    solvable:  bool          # False → no GNE exists for any p (rank drop + U_2 condition)


def _solve_equilibrium(
    Mx: np.ndarray,
    Mp: np.ndarray,
    M1: np.ndarray,
    tol_rank: float = 1e-8,
    select: str = "min_norm",
    cost: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> _EquilibriumResult:
    """
    Solve M_x x* = M_p p + M_1 for x*(p) = H_x p + h_x.

    Unique case (rank(M_x) == n_x):
        H_x = M_x^{-1} M_p,  h_x = M_x^{-1} M_1   (paper Eq. 7a)

    Infinite case (rank(M_x) < n_x) — a WHOLE MANIFOLD of GNE exists; pick one:
        Solvability check: U_2^T M_p ≈ 0 and U_2^T M_1 ≈ 0  (paper Eq. 10).
        General solution:  x(p) = x_p(p) + N z,  x_p = M_x^+ (M_p p + M_1),
        with N spanning null(M_x).

        select="min_norm" (default, legacy — DiMPC/mpgne behaviour):
            z = 0  →  x = x_p (least-‖x‖ point).  Arbitrary GNE.

        select="potential" (variational GNE — requires `cost`):
            Choose z to minimise the game potential Σ_i J_i(x_i, p) over the
            manifold, where (Q_full, c_full, F_full) = `cost` stack the agents'
            costs (J = ½xᵀQx + (c+Fp)ᵀx).  For a potential game with a shared
            convex coupling this selects the VARIATIONAL GNE = social optimum
            (Facchinei & Kanzow 2010), matching converged ADMM.  Closed form:
                z*(p) = -(NᵀQN)⁻¹ Nᵀ (Q x_p(p) + c + F p)   — affine in p,
            so x(p) stays piecewise-affine.  NᵀQN ≻ 0 because each Q_i ≻ 0.
    """
    n_x = Mx.shape[0]
    n_p = Mp.shape[1]

    rank = np.linalg.matrix_rank(Mx, tol=tol_rank)

    if rank == n_x:
        # Full rank → unique GNE (variational == min-norm, no freedom to select)
        try:
            Mx_inv = np.linalg.inv(Mx)
        except np.linalg.LinAlgError:
            return _EquilibriumResult(
                H_x=np.zeros((n_x, n_p)), h_x=np.zeros(n_x),
                is_unique=False, solvable=False,
            )
        return _EquilibriumResult(
            H_x=Mx_inv @ Mp,
            h_x=Mx_inv @ M1,
            is_unique=True,
            solvable=True,
        )

    # Rank-deficient → SVD for min-norm / infinite solution (paper Eq. 8-9)
    # gesdd (the default divide-and-conquer driver) occasionally raises "SVD did not
    # converge" on a particular Mx; retry once with the slower but robust gesvd driver.
    # Result-preserving: identical output whenever gesdd converges (the common case).
    try:
        U, sigma, Vt = svd(Mx, full_matrices=True)
    except np.linalg.LinAlgError:
        U, sigma, Vt = svd(Mx, full_matrices=True, lapack_driver="gesvd")
    n_M = rank                           # number of non-zero singular values
    sigma1 = sigma[:n_M]
    U1 = U[:, :n_M]
    U2 = U[:, n_M:]                      # null-space of M_x^T

    # Solvability: U_2^T (M_p p + M_1) = 0 must hold ∀ p  (paper Eq. 10)
    # i.e. U_2^T M_p = 0 and U_2^T M_1 = 0
    if (np.linalg.norm(U2.T @ Mp) > tol_rank * 10
            or np.linalg.norm(U2.T @ M1) > tol_rank * 10):
        return _EquilibriumResult(
            H_x=np.zeros((n_x, n_p)), h_x=np.zeros(n_x),
            is_unique=False, solvable=False,
        )

    # Particular (min-norm) affine law: x_p(p) = Hp p + hp
    V1 = Vt[:n_M, :].T                  # (n_x, n_M)
    Sigma1_inv = np.diag(1.0 / sigma1)
    Mx_pinv = V1 @ Sigma1_inv @ U1.T   # (n_x, n_x)
    Hp = Mx_pinv @ Mp
    hp = Mx_pinv @ M1

    if select == "potential" and cost is not None:
        # Right null space of M_x:  columns of V beyond the rank  (M_x @ N = 0)
        N = Vt[n_M:, :].T              # (n_x, n_x - n_M)
        Q_full, c_full, F_full = cost
        NtQN = N.T @ Q_full @ N
        try:
            NtQN_inv = np.linalg.inv(NtQN)
        except np.linalg.LinAlgError:
            NtQN_inv = np.linalg.pinv(NtQN)
        # z*(p) = A_z p + b_z  (minimiser of the potential over the manifold)
        A_z = -NtQN_inv @ N.T @ (Q_full @ Hp + F_full)
        b_z = -NtQN_inv @ N.T @ (Q_full @ hp + c_full)
        H_x = Hp + N @ A_z
        h_x = hp + N @ b_z
    else:
        H_x = Hp
        h_x = hp

    return _EquilibriumResult(
        H_x=H_x,
        h_x=h_x,
        is_unique=False,
        solvable=True,
    )


def _stacked_cost(
    game: GNEGame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack the agents' costs J_i = ½ x_iᵀ Q_i x_i + (c_i + F_i p)ᵀ x_i into the
    game potential Σ_i J_i = ½ xᵀ Q_full x + (c_full + F_full p)ᵀ x.

    Q_full : (n_x_total, n_x_total)  block-diagonal of Q_i
    c_full : (n_x_total,)            stacked c_i
    F_full : (n_x_total, n_p)        stacked F_i

    Used only for select="potential" (variational GNE selection).  This uses the
    same per-agent cost data the combiner already holds — no extra sharing.
    """
    n_x = game.n_x_total
    n_p = game.n_p
    Q_full = np.zeros((n_x, n_x))
    c_full = np.zeros(n_x)
    F_full = np.zeros((n_x, n_p))
    for a in game.agents:
        sl = game.x_slice(a.index)
        Q_full[sl, sl] = a.Q
        c_full[sl]     = a.c
        F_full[sl, :]  = a.F
    return Q_full, c_full, F_full


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3 — project agent CRs to p-space  (paper Eq. 7b)
# ─────────────────────────────────────────────────────────────────────────────

def _project_crs_to_p_space(
    combo: tuple[int, ...],
    agent_solutions: list[AgentSolution],
    game: GNEGame,
    H_x: np.ndarray,
    h_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Substitute x*(p) = H_x p + h_x into each agent's CR constraint,
    yielding a polyhedron purely in p-space.

    For agent i with CR j_i:
        E_i θ_i ≤ f_i   with θ_i = [x_{-i}*(p); p]
        = [E_i_x | E_i_p] [x_{-i}*(p); p] ≤ f_i
        = E_i_x (H_x_neg p + h_x_neg) + E_i_p p ≤ f_i
        = (E_i_x H_x_neg + E_i_p) p ≤ f_i - E_i_x h_x_neg
        =: D_i p ≤ e_i

    Returns
    -------
    D : (total_ineq, n_p)  — stacked D_i for all agents
    e : (total_ineq,)      — stacked e_i for all agents
    """
    n_p       = game.n_p
    D_blocks, e_blocks = [], []

    for i, j_i in enumerate(combo):
        cr      = agent_solutions[i][j_i]
        n_x_i   = game.agents[i].n_x
        n_x_neg = cr.E.shape[1] - n_p
        use_sum = n_x_neg < (game.n_x_total - n_x_i)

        E_i_x = cr.E[:, :n_x_neg]   # (n_ineq, n_x_neg)
        E_i_p = cr.E[:, n_x_neg:]   # (n_ineq, n_p)

        others = [j for j in range(game.N) if j != i]

        if use_sum:
            # Sum mode: H_x_neg = Σ_{j≠i} H_x[slice_j],  h_x_neg = Σ_{j≠i} h_x[slice_j]
            H_x_neg = np.zeros((n_x_neg, n_p))
            h_x_neg = np.zeros(n_x_neg)
            for j in others:
                H_x_neg += H_x[game.x_slice(j)]
                h_x_neg += h_x[game.x_slice(j)]
        else:
            # Individual mode (legacy)
            H_x_neg = np.vstack([H_x[game.x_slice(j)] for j in others])
            h_x_neg = np.concatenate([h_x[game.x_slice(j)] for j in others])

        D_i = E_i_x @ H_x_neg + E_i_p   # (n_ineq, n_p)
        e_i = cr.f - E_i_x @ h_x_neg    # (n_ineq,)

        D_blocks.append(D_i)
        e_blocks.append(e_i)

    return np.vstack(D_blocks), np.concatenate(e_blocks)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 4 — Chebyshev-center LP: is the p-space CR non-empty?
# ─────────────────────────────────────────────────────────────────────────────

def _cr_nonempty(
    D: np.ndarray,
    e: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """
    Check if { p : D p ≤ e } is non-empty using the Chebyshev-center LP.

    Variables: z = [p (n_p); r (1)]
    Maximize r  s.t.  D_i p + r ||D_i||_2 ≤ e_i

    Returns True iff the optimal r* > -tol  (polyhedron has interior).

    This is stronger than mere feasibility: r* > 0 means the polyhedron
    contains a ball of positive radius (full-dimensional interior).
    We use r* > -tol to also accept boundary-only intersections.
    """
    n_p  = D.shape[1]
    n_c  = D.shape[0]
    nrms = np.linalg.norm(D, axis=1, keepdims=True)   # (n_c, 1)

    # Build augmented constraint: [D | nrms] [p; r] ≤ e
    A_aug = np.hstack([D, nrms])                       # (n_c, n_p+1)
    b_aug = e

    # Objective: min -r  (maximise r)
    c_obj = np.zeros(n_p + 1)
    c_obj[-1] = -1.0

    res = linprog(
        c_obj, A_ub=A_aug, b_ub=b_aug,
        bounds=[(None, None)] * n_p + [(None, None)],
        method='highs',
        options={'disp': False},
    )

    if res.status == 0:
        return float(res.x[-1]) > -tol
    if res.status == 3:
        return True   # unbounded → feasible (no bound on r, D is degenerate)
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — build the full GNE solution
# ─────────────────────────────────────────────────────────────────────────────

def build_gne_solution(
    game: GNEGame,
    agent_solutions: list[AgentSolution],
    tol_rank: float = 1e-8,
    tol_nonempty: float = 1e-6,
    verbose: bool = True,
    equilibrium_select: str = "min_norm",
) -> GNESolution:
    """
    Algorithm 1 (Steps 7-13): enumerate all CR combinations and collect
    valid GNE critical regions in p-space.

    For each combination C_k = (j_1, ..., j_N):
      1. Assemble M_x, M_p, M_1  (Eq. 6)
      2. Solve for x*(p) = H_x p + h_x  (Eq. 7a or min-norm via SVD)
      3. Project CRs to p-space: D p ≤ e  (Eq. 7b)
      4. Add parameter box: p_lb ≤ p ≤ p_ub
      5. Check non-empty via Chebyshev LP → store GNECriticalRegion

    Parameters
    ----------
    game             : GNEGame
    agent_solutions  : list[AgentSolution] — one per agent (from mp_solver)
    tol_rank         : threshold for rank detection in M_x
    tol_nonempty     : Chebyshev radius threshold for non-empty CR check
    verbose          : print progress

    Returns
    -------
    GNESolution with all valid GNECriticalRegion objects
    """
    N         = game.N
    n_p       = game.n_p

    # Build iteration order over all combinations
    cr_index_lists = [list(range(agent_solutions[i].n_cr)) for i in range(N)]
    n_total = 1
    for idx in cr_index_lists:
        n_total *= len(idx)

    if verbose:
        print(f"\n[gne_combiner] N={N} agents, n_p={n_p}")
        cr_counts = [agent_solutions[i].n_cr for i in range(N)]
        print(f"  CRs per agent: {cr_counts}  →  {n_total} combinations to check")

    # Parameter space box rows (added to every CR in p-space)
    D_box = np.vstack([ np.eye(n_p), -np.eye(n_p)])   # (2*n_p, n_p)
    e_box = np.concatenate([game.p_ub, -game.p_lb])   # (2*n_p,)

    # Cost stack for variational (potential-minimising) equilibrium selection
    cost = _stacked_cost(game) if equilibrium_select == "potential" else None

    gne_regions: list[GNECriticalRegion] = []
    n_singular  = 0   # M_x rank-deficient combos
    n_insolvable = 0  # infinite case but U_2^T condition fails
    n_empty     = 0   # non-empty check failed

    t0 = time.perf_counter()

    for combo in itertools.product(*cr_index_lists):

        # ── Stage 1: equilibrium system ───────────────────────────────────────
        Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_solutions, game)

        # ── Stage 2: solve for x*(p) ──────────────────────────────────────────
        eq = _solve_equilibrium(Mx, Mp, M1, tol_rank=tol_rank,
                                select=equilibrium_select, cost=cost)

        if not eq.solvable:
            n_insolvable += 1
            continue

        if not eq.is_unique:
            n_singular += 1

        # ── Stage 3: project CRs to p-space ──────────────────────────────────
        D_crs, e_crs = _project_crs_to_p_space(
            combo, agent_solutions, game, eq.H_x, eq.h_x
        )

        # Merge with parameter box
        D_full = np.vstack([D_crs, D_box])
        e_full = np.concatenate([e_crs, e_box])

        # ── Stage 4: non-empty check ──────────────────────────────────────────
        if not _cr_nonempty(D_full, e_full, tol=tol_nonempty):
            n_empty += 1
            continue

        # ── Store valid GNECriticalRegion ─────────────────────────────────────
        gne_regions.append(GNECriticalRegion(
            combination=tuple(combo),
            D=D_full,
            e=e_full,
            H_x=eq.H_x,
            h_x=eq.h_x,
            Mx=Mx,
            Mp=Mp,
            M1=M1,
            is_unique=eq.is_unique,
        ))

    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"  Combinations: {n_total} total  |  "
              f"{len(gne_regions)} valid GNE CRs  |  "
              f"{n_insolvable} insolvable  |  "
              f"{n_singular} rank-deficient  |  "
              f"{n_empty} empty")
        print(f"  Elapsed: {elapsed:.2f}s")

    sol = GNESolution(regions=gne_regions, n_p=n_p, N=N)
    if verbose:
        print(f"  {sol.summary()}")
    return sol


# ─────────────────────────────────────────────────────────────────────────────
#  Offline variational filter — collapse overlapping GNE CRs to a single-valued map
# ─────────────────────────────────────────────────────────────────────────────

def _chebyshev_center_p(D: np.ndarray, e: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Return (interior point, radius) of {p : D p ≤ e} via the Chebyshev-center LP."""
    n_p  = D.shape[1]
    nrms = np.linalg.norm(D, axis=1, keepdims=True)
    A_aug = np.hstack([D, nrms])
    c_obj = np.zeros(n_p + 1)
    c_obj[-1] = -1.0
    res = linprog(c_obj, A_ub=A_aug, b_ub=e,
                  bounds=[(None, None)] * (n_p + 1), method="highs")
    if res.status == 0:
        return res.x[:n_p], float(res.x[-1])
    return None, -np.inf


def filter_variational(
    gne_sol: GNESolution,
    game: GNEGame,
    tol_center: float = 1e-7,
    tol_pot: float = 1e-6,
    verbose: bool = True,
) -> GNESolution:
    """
    Collapse the enumerate-ALL-GNE solution to a SINGLE-VALUED variational map.

    `build_gne_solution` returns every GNE CR; where the shared constraint binds these
    OVERLAP in p-space (many "corner" equilibria + the variational one).  The
    variational GNE = social optimum is always the MINIMUM-potential member among the
    CRs covering a point (Facchinei & Kanzow 2010).

    This does the min-potential selection ONCE, OFFLINE: a CR is kept iff at its own
    Chebyshev centre it is (within tol) the minimum-potential CR among all CRs covering
    that centre.  Dominated (never-variational) CRs are dropped.  The survivors tile
    p-space single-valued, so ONLINE lookup is `locate()` + affine eval — no costs, no
    inter-agent exchange (privacy preserved: agents share only decisions).

    NOTE (validated for 1-D θ): the centre test assumes a CR that is variational
    anywhere is variational at its Chebyshev centre — exact for interval CRs and the
    strictly-convex-potential mpQP structure.  For higher-dim θ (v1+) re-verify the
    pruned map is single-valued + matches ADMM; escalate to a per-combo centralized-KKT
    (common-multiplier) test if a CR is wrongly dropped.
    """
    Q_full, c_full, F_full = _stacked_cost(game)

    def _pot(x, p):
        return float(0.5 * x @ Q_full @ x + (c_full + F_full @ p) @ x)

    kept: list[GNECriticalRegion] = []
    n_drop = 0
    for cr in gne_sol.regions:
        pc, r = _chebyshev_center_p(cr.D, cr.e)
        if pc is None or r <= tol_center:
            n_drop += 1
            continue
        J_k = _pot(cr.evaluate(pc), pc)
        J_min = min(
            _pot(m.evaluate(pc), pc)
            for m in gne_sol.regions if m.contains(pc, tol=tol_center)
        )
        if J_k <= J_min + tol_pot:
            kept.append(cr)
        else:
            n_drop += 1

    out = GNESolution(regions=kept, n_p=gne_sol.n_p, N=gne_sol.N)
    if verbose:
        print(f"[filter_variational] kept {out.n_cr} / {gne_sol.n_cr} GNE CRs "
              f"({n_drop} non-variational dropped)")
    return out


def build_variational_gne_solution(
    game: GNEGame,
    agent_solutions: list[AgentSolution],
    tol_rank: float = 1e-8,
    tol_nonempty: float = 1e-6,
    verbose: bool = True,
) -> GNESolution:
    """
    Convenience: build the single-valued VARIATIONAL GNE map end-to-end.

    = build_gne_solution(..., equilibrium_select="potential")  [fixes within-combo
      degenerate selection]  →  filter_variational(...)  [drops overlapping
      non-variational corner GNE].  The result is a privacy-preserving explicit map:
      online is `locate()` + affine eval, no cost sharing.
    """
    full = build_gne_solution(
        game, agent_solutions, tol_rank=tol_rank, tol_nonempty=tol_nonempty,
        verbose=verbose, equilibrium_select="potential",
    )
    return filter_variational_kkt(full, game, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
#  Rigorous single-valued variational filter — per-combo centralized-KKT test
# ─────────────────────────────────────────────────────────────────────────────

def _centralized_constraints(game: GNEGame):
    """
    Stack every agent's local box + the shared coupling into one central system
        G x ≤ w0 + W θ          (x = [x_0; …; x_{N-1}])
    matching the potential-minimisation QP whose solution is the variational GNE.
    Returns (G, w0, W).
    """
    n_x = game.n_x_total
    n_p = game.n_p
    G_rows, w0_rows, W_rows = [], [], []
    # local box constraints (block-diagonal placement)
    for a in game.agents:
        sl = game.x_slice(a.index)
        for r in range(a.A_loc.shape[0]):
            row = np.zeros(n_x); row[sl] = a.A_loc[r]
            G_rows.append(row); w0_rows.append(a.b_loc[r])
            W_rows.append(a.S_loc[r] if a.S_loc is not None else np.zeros(n_p))
    # shared coupling  Σ_i C_i x_i ≤ d + S_coup θ
    if game.n_coupling > 0:
        for r in range(game.n_coupling):
            row = np.zeros(n_x)
            for a in game.agents:
                if a.C is not None:
                    row[game.x_slice(a.index)] = a.C[r]
            G_rows.append(row); w0_rows.append(game.d[r])
            W_rows.append(game.S_coup[r] if game.S_coup is not None else np.zeros(n_p))
    return np.array(G_rows), np.array(w0_rows), np.array(W_rows)


def _min_over_cr(a_lin: np.ndarray, a_const: float, D: np.ndarray, e: np.ndarray) -> float:
    """min over {θ: Dθ ≤ e} of  a_lin·θ + a_const.  Returns -inf if unbounded below."""
    res = linprog(a_lin, A_ub=D, b_ub=e, bounds=[(None, None)] * D.shape[1], method="highs")
    if res.status == 0:
        return float(res.fun) + a_const
    if res.status == 3:      # unbounded below
        return -np.inf
    return np.inf            # infeasible → treat as vacuously satisfied


def filter_variational_kkt(
    gne_sol: GNESolution,
    game: GNEGame,
    tol_active: float = 1e-6,
    tol_resid: float = 1e-6,
    tol_mult: float = 1e-7,
    verbose: bool = True,
) -> GNESolution:
    """
    Rigorous single-valued variational map (works for any n_p, unlike the Chebyshev-
    centre `filter_variational`).

    The variational GNE = argmin_x Φ(x,θ)=½xᵀQx+(c+Fθ)ᵀx  s.t.  G x ≤ w0+Wθ  (the
    potential-minimisation QP).  A combo's affine law x(θ)=H_xθ+h_x is on the
    variational map iff it satisfies the CENTRALISED KKT conditions on its CR:
        ∃ μ(θ) ≥ 0 :  Q x(θ)+c+Fθ + G_Aᵀ μ_A(θ) = 0   (A = active rows),  μ_A ≥ 0.
    Non-variational corner GNE (each agent best-responding with its OWN coupling
    multiplier) violate the common-multiplier / sign condition throughout their CR and
    are dropped ENTIRELY — the surviving combos tile θ-space single-valued (the
    variational solution's own polyhedral partition).  Online: `locate()`+affine, no
    costs, no exchange.

    Per combo: reconstruct the active set at the CR Chebyshev centre, solve stationarity
    for the affine multipliers μ_A(θ) (affine in θ), then — because μ_A(θ) ≥ 0 is a set
    of LINEAR cuts — CLIP the CR to its variational sub-region {θ∈CR : μ_A(θ) ≥ 0}
    rather than dropping it whole.  A combo variational on only part of its region keeps
    exactly that part; the clipped regions tile θ-space single-valued with NO gaps.
    Combos whose stationarity residual ≠ 0 (active set inconsistent) are dropped.
    O(n_cr) — scalable.
    """
    Q, c, F = _stacked_cost(game)
    G, w0, W = _centralized_constraints(game)
    n_p = game.n_p

    kept, n_drop, n_clip = [], 0, 0
    for cr in gne_sol.regions:
        pc, r = _chebyshev_center_p(cr.D, cr.e)
        if pc is None or r <= tol_active:
            n_drop += 1; continue
        Hx, hx = cr.H_x, cr.h_x
        Gg = Q @ Hx + F                 # ∇Φ θ-coefficient   (n_x, n_p)
        g0 = Q @ hx + c                 # ∇Φ constant        (n_x,)
        x_c = Hx @ pc + hx
        slack = G @ x_c - (w0 + W @ pc)          # ≤ 0; ≈0 ⇒ active
        A = np.where(np.abs(slack) <= tol_active * (1.0 + np.abs(w0 + W @ pc)))[0]

        if A.size == 0:
            if np.linalg.norm(Gg) <= tol_resid and np.linalg.norm(g0) <= tol_resid:
                kept.append(cr)
            else:
                n_drop += 1
            continue

        GA_T = G[A].T                    # (n_x, |A|)
        pinv = np.linalg.pinv(GA_T)
        M = -pinv @ Gg                   # μ_A(θ) = M θ + m0
        m0 = -pinv @ g0
        # stationarity must hold exactly for a genuine KKT point
        if (np.linalg.norm(GA_T @ M + Gg) > tol_resid * (1 + np.linalg.norm(Gg))
                or np.linalg.norm(GA_T @ m0 + g0) > tol_resid * (1 + np.linalg.norm(g0))):
            n_drop += 1; continue

        # CLIP: add μ_{A,j}(θ) ≥ 0  ⇔  −M_j·θ ≤ m0_j  for every active row
        D_new = np.vstack([cr.D, -M])
        e_new = np.concatenate([cr.e, m0])
        pc2, r2 = _chebyshev_center_p(D_new, e_new)
        if pc2 is None or r2 <= tol_mult:
            n_drop += 1; continue        # variational sub-region empty
        if D_new.shape[0] > cr.D.shape[0] and r2 < r - tol_mult:
            n_clip += 1
        kept.append(GNECriticalRegion(
            combination=cr.combination, D=D_new, e=e_new,
            H_x=cr.H_x, h_x=cr.h_x, Mx=cr.Mx, Mp=cr.Mp, M1=cr.M1,
            is_unique=cr.is_unique))

    out = GNESolution(regions=kept, n_p=n_p, N=gne_sol.N)
    if verbose:
        print(f"[filter_variational_kkt] kept {out.n_cr} / {gne_sol.n_cr} GNE CRs "
              f"({n_clip} clipped, {n_drop} dropped)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostic: verify a GNE solution at a specific p
# ─────────────────────────────────────────────────────────────────────────────

def verify_gne_at_p(
    p: np.ndarray,
    gne_sol: GNESolution,
    game: GNEGame,
    agent_solutions: list[AgentSolution],
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    For a given p, find the matching GNE CR and verify x*(p) is a valid GNE.

    Checks:
    1. p is inside a GNE CR
    2. Equilibrium residual ||M_x x* - M_p p - M_1|| is small
    3. x*(p) satisfies all game constraints
    4. Each agent's x_i* is inside its own AgentCR (in θ_i = [x_{-i}*; p] space)

    Returns dict with keys: 'found', 'residual', 'feasible', 'cr_valid', 'k'
    """
    p = np.asarray(p).ravel()
    k = gne_sol.locate(p)

    if k is None:
        if verbose:
            print(f"[verify] p not in any GNE CR")
        return {'found': False}

    cr_k = gne_sol[k]
    x_star = cr_k.evaluate(p)

    # 1. Equilibrium residual
    residual = cr_k.residual(p)

    # 2. Global feasibility
    feasible = game.all_feasible(x_star, p, tol=tol)

    # 3. Each agent's x_i* inside its CR (in θ_i space)
    cr_combo = cr_k.combination
    cr_valid = True
    for i, j_i in enumerate(cr_combo):
        cr_i    = agent_solutions[i][j_i]
        n_x_i   = game.agents[i].n_x
        n_x_neg = cr_i.E.shape[1] - game.n_p
        others  = [j for j in range(game.N) if j != i]
        if n_x_neg < (game.n_x_total - n_x_i):  # sum mode
            x_neg = np.zeros(n_x_neg)
            for j in others:
                x_neg += x_star[game.x_slice(j)]
        else:
            x_neg = np.concatenate([x_star[game.x_slice(j)] for j in others])
        theta_i = np.concatenate([x_neg, p])
        cr = agent_solutions[i][j_i]
        if not cr.contains(theta_i, tol=tol):
            cr_valid = False
            break

    if verbose:
        print(f"[verify] p={p}  →  GNE CR k={k}  combo={cr_combo}")
        print(f"  x*(p) = {x_star}")
        print(f"  equilibrium residual = {residual:.2e}")
        print(f"  feasible = {feasible}  |  CR valid = {cr_valid}")

    return {
        'found': True,
        'k': k,
        'x_star': x_star,
        'residual': residual,
        'feasible': feasible,
        'cr_valid': cr_valid,
    }
