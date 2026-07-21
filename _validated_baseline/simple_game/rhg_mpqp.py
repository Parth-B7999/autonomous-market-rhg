"""
rhg_mpqp.py — per-agent mpQP for the H=4 receding-horizon H2 GNE, in PRIVATE θ_i with
the SUM-COMPRESSED coupling variable sum_x_neg (per FORMULATION.md and Parth's correction:
each agent's PPOPT problem sees only its own params + the coupling variable sum_x_neg).

Locked spec:
  6 agents (2 grid + 2 PV + 2 wind), med-scale caps. H = 4 (15-min steps, Δt=0.25 h).
  Coupling per step k:  L_min=100 ≤ Σ_i p_{i,k} ≤ L_max=900.
  Per-agent H2 cumulative (couples the horizon):  Σ_k η_i·p_elec_{i,k}·Δt ≥ D_i.

Private θ_i  (sum_x_neg = Σ_{j≠i} p_{j} per step, H=4 dims — the ONLY shared quantity):
  grid i     x_i=[p_0..3]        θ_i = [sum_x_neg(4) | D_i(1), λ_RT(4), p_DA_i(1)]      → 10
  renewable  x_i=[p_0..3,cv_0..3] θ_i = [sum_x_neg(4) | D_i(1), λ_RT(4), p_DA_i(1), g(4)] → 14

Public p_gne (union, for the combiner after sum_x_neg elimination), n_p_gne = 24:
  [D_0..5 (6) | λ_0..3 (4) | p_DA_0..5 (6) | g_solar_0..3 (4) | g_wind_0..3 (4)]

Each agent's CRs are solved in private θ_i then EXPANDED to [sum_x_neg(4) | p_gne(24)]
= 28 dims so gne_combiner/facet auto-detect sum mode via (E.shape[1] - n_p_gne == H).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ppopt.mpqp_program import MPQP_Program
from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm
from amrhg.solvers.game import Agent, GNEGame
from amrhg.solvers.cr_store import AgentSolution, agent_solution_from_ppopt

R_H2 = 3.0
DT = 0.25
EPS_CV = 1e-3
H = 4
L_MIN, L_MAX = 100.0, 900.0
LAM_BOX = (-50.0, 300.0)
DEFAULT_ALGO = mpqp_algorithm.combinatorial

# (name, type, p_max, ren_cap, eta, gamma, d_max)  — med-scale FOCAPO caps
# LOCKED spec per FORMULATION.md: agents 2-3 (PV) and 4-5 (wind) are identical pairs
# -> only 4 distinct offline mpQP solves are needed (validated report numbers depend on this).
FLEET = [
    ("PEM_Elec",  "grid",  250.0,   0.0, 0.020, 5e-3, 3.0),
    ("ALK",       "grid",  200.0,   0.0, 0.018, 5e-3, 2.5),
    ("PEM_PV",    "solar", 125.0, 125.0, 0.020, 5e-3, 1.6),
    ("PEM_PV_2",  "solar", 125.0, 125.0, 0.020, 5e-3, 1.6),
    ("PEM_Wind",  "wind",  250.0, 250.0, 0.020, 4e-3, 3.0),
    ("PEM_Wind_2","wind",  250.0, 250.0, 0.020, 4e-3, 3.0),
]

N = len(FLEET)
NREN = sum(1 for f in FLEET if f[1] != "grid")

# Which agents share the exact same parameters and type?
DISTINCT = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5]}

# ── p_gne (public union) layout ───────────────────────────────────────────────
D_START   = 0            # D_0..D_5
LAM_START = N            # λ_0..λ_3        (6..9)
PDA_START = N + H        # p_DA_0..5       (10..15)
GS_START  = N + H + N    # g_solar_0..3    (16..19)
GW_START  = N + H + N + H # g_wind_0..3    (20..23)
N_P_GNE   = N + H + N + 2 * H   # 24


def _pmax(i): return FLEET[i][2]
def _rcap(i): return FLEET[i][3]
def _eta(i):  return FLEET[i][4]
def _gam(i):  return FLEET[i][5]
def _dmax(i): return FLEET[i][6]
def _typ(i):  return FLEET[i][1]
def _a(i):    return R_H2 * _eta(i)          # $/kWh break-even coefficient
def _is_ren(i): return _typ(i) != "grid"


def _priv_layout(i):
    """Index layout inside private θ_i = [sum_x_neg(H) | D, λ(H), p_DA, (g(H))]."""
    ren = _is_ren(i)
    nx = 2 * H if ren else H
    sneg = slice(0, H)
    d_th = H                      # D_i
    lam_th = slice(H + 1, H + 1 + H)
    pda_th = H + 1 + H            # p_DA_i
    g_th = slice(H + 2 + H, H + 2 + 2 * H) if ren else None
    n_theta = (H + 2 + 2 * H) if ren else (H + 2 + H)   # 14 or 10
    return dict(nx=nx, ren=ren, sneg=sneg, d_th=d_th, lam_th=lam_th, pda_th=pda_th,
                g_th=g_th, n_theta=n_theta)


def build_agent_matrices(i):
    """PPOPT (Q, H_pp, c, G, b, F) for agent i in private θ_i.
       obj: ½xᵀQx+(H_pp θ+c)ᵀx ,  G x ≤ b + F θ."""
    L = _priv_layout(i); nx = L["nx"]; nt = L["n_theta"]
    eta, gam, pmax, a, pda_col = _eta(i), _gam(i), _pmax(i), _a(i), L["pda_th"]
    I = np.eye(H)

    # ── cost ──────────────────────────────────────────────────────────────────
    Q = np.zeros((nx, nx)); Q[:H, :H] = gam * I
    if L["ren"]:
        Q[H:, H:] = EPS_CV * I
    c = np.zeros(nx); c[:H] = -a * DT               # −r·η·Δt on p_k
    if L["ren"]:
        c[H:] = a * DT                              # +r·η·Δt on cv_k (curtail opportunity)
    H_pp = np.zeros((nx, nt))
    for k in range(H):
        H_pp[k, L["lam_th"].start + k] = DT / 1000.0   # +λ_k·Δt/1000 · p_k
    H_pp[:H, pda_col] = -gam                            # −γ·p_DA · p_k (anchor cross term)

    # ── local rows ──────────────────────────────────────────────────────────────
    G_rows, b_rows, F_rows = [], [], []
    def add(g, b, fmap=None):
        G_rows.append(g); b_rows.append(b)
        f = np.zeros(nt)
        if fmap:
            for col, val in fmap: f[col] = val
        F_rows.append(f)

    if not L["ren"]:
        for k in range(H):                              # 0 ≤ p_k ≤ p_max
            r = np.zeros(nx); r[k] = 1.0;  add(r, pmax)
            r = np.zeros(nx); r[k] = -1.0; add(r, 0.0)
        # H2 cumulative:  −η·dt Σ p_k ≤ −D_i   (Σ_k η·p_k·dt ≥ D_i)
        r = np.zeros(nx); r[:H] = -eta * DT
        add(r, 0.0, [(L["d_th"], -1.0)])
    else:
        gcol = L["g_th"].start
        for k in range(H):
            pk, cvk, gk = k, H + k, gcol + k
            def row(coeffs):
                v = np.zeros(nx)
                for pos, val in coeffs: v[pos] = val
                return v
            add(row([(pk, 1.0)]), pmax)                       # p ≤ p_max
            add(row([(cvk, 1.0)]), _rcap(i))                  # cv ≤ g_max(=cap)
            add(row([(pk, -1.0)]), 0.0)                       # p ≥ 0
            add(row([(cvk, -1.0)]), 0.0)                      # cv ≥ 0
            add(row([(cvk, 1.0)]), 0.0, [(gk, 1.0)])          # cv ≤ g_k
            add(row([(pk, 1.0), (cvk, -1.0)]), pmax, [(gk, -1.0)])  # p+g−cv ≤ p_max
            add(row([(pk, -1.0), (cvk, 1.0)]), 0.0, [(gk, 1.0)])    # p+g−cv ≥ 0
        # H2 cumulative:  −ηΣp + ηΣcv ≤ −D + ηΣg
        r = np.zeros(nx); r[:H] = -eta * DT; r[H:] = eta * DT
        fmap = [(L["d_th"], -1.0)] + [(gcol + k, eta * DT) for k in range(H)]
        add(r, 0.0, fmap)

    # ── coupling per step:  L_min ≤ p_k + sum_x_neg_k ≤ L_max ────────────────────
    for k in range(H):
        r = np.zeros(nx); r[k] = 1.0                          # p_k + sneg_k ≤ L_max
        add(r, L_MAX, [(L["sneg"].start + k, -1.0)])
        r = np.zeros(nx); r[k] = -1.0                         # −(p_k+sneg_k) ≤ −L_min
        add(r, -L_MIN, [(L["sneg"].start + k, 1.0)])

    G = np.array(G_rows); b = np.array(b_rows); F = np.vstack(F_rows)
    return Q, H_pp, c, G, b, F


def build_theta_box(i):
    """Box A_t θ ≤ b_t on private θ_i."""
    L = _priv_layout(i); nt = L["n_theta"]
    lo = np.zeros(nt); hi = np.zeros(nt)
    # sum_x_neg per step: sum of other agents' p ranges [0, Σ_{j≠i} p_max_j]
    sneg_ub = sum(_pmax(j) for j in range(N) if j != i)
    lo[L["sneg"]] = 0.0; hi[L["sneg"]] = sneg_ub
    lo[L["d_th"]] = 0.0; hi[L["d_th"]] = _dmax(i)
    lo[L["lam_th"]] = LAM_BOX[0]; hi[L["lam_th"]] = LAM_BOX[1]
    lo[L["pda_th"]] = 0.0; hi[L["pda_th"]] = _pmax(i)
    if L["ren"]:
        lo[L["g_th"]] = 0.0; hi[L["g_th"]] = _rcap(i)
    A_t = np.vstack([np.eye(nt), -np.eye(nt)])
    b_t = np.concatenate([hi, -lo]).reshape(-1, 1)
    return A_t, b_t


def _embed(i):
    """Map private p_priv positions → p_gne indices (order: D_i, λ(4), p_DA_i, g(4))."""
    em = [D_START + i] + list(range(LAM_START, LAM_START + H)) + [PDA_START + i]
    if _is_ren(i):
        base = GS_START if _typ(i) == "solar" else GW_START
        em += list(range(base, base + H))
    return em


def _expand_solution(sol: AgentSolution, i: int) -> AgentSolution:
    """Expand CRs from private θ_i = [sneg(H)|p_priv] to [sneg(H)|p_gne(24)]."""
    from amrhg.solvers.cr_store import AgentCR
    embed = _embed(i); n_full = H + N_P_GNE
    regs = []
    for cr in sol.regions:
        E = np.zeros((cr.E.shape[0], n_full)); A = np.zeros((cr.A.shape[0], n_full))
        E[:, :H] = cr.E[:, :H]; A[:, :H] = cr.A[:, :H]
        for kk, gidx in enumerate(embed):
            E[:, H + gidx] += cr.E[:, H + kk]
            A[:, H + gidx] += cr.A[:, H + kk]
        regs.append(AgentCR(E=E, f=cr.f.copy(), A=A, b=cr.b.copy(), index=cr.index))
    return AgentSolution(agent_index=sol.agent_index, n_x_i=sol.n_x_i,
                         n_theta_i=n_full, regions=regs)


def solve_agent_private(i, algorithm=DEFAULT_ALGO, verbose=False):
    """Solve agent i's mpQP in PRIVATE θ_i (no expand). Returns (private AgentSolution, n_cr).
    Identical-spec agents (2 PV, 2 wind) share the same private solve."""
    L = _priv_layout(i)
    Q, H_pp, c, G, b, F = build_agent_matrices(i)
    A_t, b_t = build_theta_box(i)
    prog = MPQP_Program(G, b.reshape(-1, 1), c.reshape(-1, 1), H_pp, Q, A_t, b_t, F)
    sol = solve_mpqp(prog, algorithm=algorithm)
    n_cr = len(sol.critical_regions)
    sol_priv = agent_solution_from_ppopt(sol, agent_index=i, n_x_i=L["nx"],
                                         n_theta_i=L["n_theta"])
    if verbose:
        print(f"  agent {i} ({FLEET[i][0]:11s}, {'ren' if L['ren'] else 'grid'}): "
              f"n_x={L['nx']} θ_i={L['n_theta']} → {n_cr} CRs")
    return sol_priv, n_cr


def expand_for(sol_priv: AgentSolution, i: int) -> AgentSolution:
    """Expand a private solve into agent i's slot (its D_i/p_DA_i/g columns in p_gne)."""
    from dataclasses import replace
    s = AgentSolution(agent_index=i, n_x_i=sol_priv.n_x_i,
                      n_theta_i=sol_priv.n_theta_i, regions=sol_priv.regions)
    return _expand_solution(s, i)


def solve_agent(i, algorithm=DEFAULT_ALGO, verbose=False):
    sol_priv, n_cr = solve_agent_private(i, algorithm=algorithm, verbose=verbose)
    return expand_for(sol_priv, i), n_cr


def solve_all(algorithm=DEFAULT_ALGO, verbose=True):
    sols = []; ncrs = []
    for i in range(N):
        s, n = solve_agent(i, algorithm=algorithm, verbose=verbose)
        sols.append(s); ncrs.append(n)
    if verbose:
        print(f"  per-agent private CRs = {ncrs}  →  K^N = {int(np.prod(ncrs))} combos")
    return sols, ncrs


# ── public game object (for combiner: coupling in p_gne space) ─────────────────
def build_pgne_game() -> GNEGame:
    """GNEGame whose agents carry the p_gne-space coupling C/d and n_p=24, used by the
    combiner/selector.  Decisions x_i as in the private problem; coupling per step."""
    agents = []
    for i in range(N):
        L = _priv_layout(i); nx = L["nx"]
        Q = np.zeros((nx, nx)); Q[:H, :H] = _gam(i) * np.eye(H)
        if L["ren"]: Q[H:, H:] = EPS_CV * np.eye(H)
        c = np.zeros(nx); c[:H] = -_a(i) * DT
        if L["ren"]: c[H:] = _a(i) * DT
        F = np.zeros((nx, N_P_GNE))
        for k in range(H):
            F[k, LAM_START + k] = DT / 1000.0
        F[:H, PDA_START + i] = -_gam(i)
        # local + H2 rows in p_gne space (reuse builder logic, remap F cols)
        A_loc, b_loc, S_loc = _local_pgne(i)
        C = np.zeros((2 * H, nx))
        C[:H, :H] = np.eye(H); C[H:, :H] = -np.eye(H)     # p in both coupling sides
        agents.append(Agent(index=i, n_x=nx, Q=Q, c=c, F=F,
                            A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C))
    d = np.concatenate([L_MAX * np.ones(H), -L_MIN * np.ones(H)])
    S_coup = np.zeros((2 * H, N_P_GNE))
    p_lb = np.zeros(N_P_GNE); p_ub = np.zeros(N_P_GNE)
    p_lb[LAM_START:LAM_START + H] = LAM_BOX[0]; p_ub[LAM_START:LAM_START + H] = LAM_BOX[1]
    for i in range(N):
        p_ub[D_START + i] = _dmax(i)
        p_ub[PDA_START + i] = _pmax(i)
    # g range = the renewable cap of that type (PV cap vs wind cap)
    pv_cap = next(_rcap(i) for i in range(N) if _typ(i) == "solar")
    wind_cap = next(_rcap(i) for i in range(N) if _typ(i) == "wind")
    p_ub[GS_START:GS_START + H] = pv_cap      # solar CF × PV cap
    p_ub[GW_START:GW_START + H] = wind_cap    # wind CF × wind cap
    game = GNEGame(agents=agents, d=d, S_coup=S_coup, d_lb=None, S_coup_lb=None,
                   p_lb=p_lb, p_ub=p_ub)
    return game


def _local_pgne(i):
    """Local (box + renewable balance + H2) rows in p_gne space for agent i."""
    L = _priv_layout(i); nx = L["nx"]; eta = _eta(i); pmax = _pmax(i)
    Gr, br, Sr = [], [], []
    def add(g, b, smap=None):
        Gr.append(g); br.append(b); s = np.zeros(N_P_GNE)
        if smap:
            for col, val in smap: s[col] = val
        Sr.append(s)
    if not L["ren"]:
        for k in range(H):
            r = np.zeros(nx); r[k] = 1.0;  add(r, pmax)
            r = np.zeros(nx); r[k] = -1.0; add(r, 0.0)
        r = np.zeros(nx); r[:H] = -eta * DT; add(r, 0.0, [(D_START + i, -1.0)])
    else:
        base = GS_START if _typ(i) == "solar" else GW_START
        for k in range(H):
            pk, cvk, gk = k, H + k, base + k
            def row(cs):
                v = np.zeros(nx)
                for p, val in cs: v[p] = val
                return v
            add(row([(pk, 1.0)]), pmax)
            add(row([(cvk, 1.0)]), _rcap(i))
            add(row([(pk, -1.0)]), 0.0)
            add(row([(cvk, -1.0)]), 0.0)
            add(row([(cvk, 1.0)]), 0.0, [(gk, 1.0)])
            add(row([(pk, 1.0), (cvk, -1.0)]), pmax, [(gk, -1.0)])
            add(row([(pk, -1.0), (cvk, 1.0)]), 0.0, [(gk, 1.0)])
        r = np.zeros(nx); r[:H] = -eta * DT; r[H:] = eta * DT
        smap = [(D_START + i, -1.0)] + [(base + k, eta * DT) for k in range(H)]
        add(r, 0.0, smap)
    return np.array(Gr), np.array(br), np.vstack(Sr)
