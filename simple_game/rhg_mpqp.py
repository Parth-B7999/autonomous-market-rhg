"""
rhg_mpqp.py — per-agent mpQP for the H=4 receding-horizon H2 GNE, in PRIVATE θ_i with
the SUM-COMPRESSED coupling variable sum_x_neg (per FORMULATION.md and Parth's correction:
each agent's PPOPT problem sees only its own params + the coupling variable sum_x_neg).

Locked spec:
  6 agents (2 grid + 2 PV + 2 wind), all distinct, med-scale caps. H = 4 (15-min steps, Δt=0.25 h).
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
from amrhg.market import ERCOT, get_market

DEFAULT_ALGO = mpqp_algorithm.combinatorial

# ── Active-market parameters (assigned by set_market; default = ERCOT at import) ──────────
# These module globals are the single source the builders read.  set_market() swaps them so
# the WHOLE pipeline (dam/rhg_online/rhg_week reading R.*) retargets ISO / RTM timescale
# without threading a config through every call.  See amrhg/market.py.
#
# Fleet convention (name, type, p_max, ren_cap, eta, gamma, d_max):
#   p_max   = ELECTROLYZER power rating (private to the agent; free to differ)
#   ren_cap = co-located FARM size — drives g = CF*ren_cap, a SHARED public θ slot per TYPE
#             (see _embed/GS_START/GW_START), so it MUST be identical within a type
#             (enforced by _check_shared_caps).
# All agents distinct in (p_max, eta) → 6 distinct offline solves; distinct eta strictly
# orders a_i = R_H2*eta_i, breaking the PCC-ceiling permutation symmetry (no tied-equilibrium
# continuum).  Σ(p_max)=1200 kW held vs L_max=900 so the ceiling still binds.
ACTIVE_MARKET = None
R_H2 = DT = EPS_CV = H = None
L_MIN = L_MAX = None
LAM_BOX = None
FLEET = None
N = NREN = None
D_START = LAM_START = PDA_START = GS_START = GW_START = N_P_GNE = None


def set_market(cfg):
    """Point rhg_mpqp (and every module reading R.*) at market `cfg`.  Idempotent.
    Rebuilds the p_gne union layout for the fleet size N and horizon H of `cfg`."""
    global ACTIVE_MARKET, R_H2, DT, EPS_CV, H, L_MIN, L_MAX, LAM_BOX, FLEET, N, NREN
    global D_START, LAM_START, PDA_START, GS_START, GW_START, N_P_GNE
    ACTIVE_MARKET = cfg
    R_H2, DT, EPS_CV, H = cfg.r_h2, cfg.dt, cfg.eps_cv, cfg.H
    L_MIN, L_MAX = cfg.l_min, cfg.l_max
    LAM_BOX = tuple(cfg.lam_box)
    FLEET = [tuple(f) for f in cfg.fleet]
    N = len(FLEET)
    NREN = sum(1 for f in FLEET if f[1] != "grid")
    # public union p_gne layout: [D(N) | λ(H) | p_DA(N) | g_solar(H) | g_wind(H)]
    D_START = 0
    LAM_START = N
    PDA_START = N + H
    GS_START = N + H + N
    GW_START = N + H + N + H
    N_P_GNE = N + H + N + 2 * H
    _check_shared_caps()   # fail loudly on a bad fleet, not 14 min into an offline solve
    return cfg


def type_rcap(typ):
    """The farm size shared by every agent of `typ`. Single source of truth for g = CF*rcap."""
    caps = {FLEET[i][3] for i in range(len(FLEET)) if FLEET[i][1] == typ}
    if len(caps) != 1:
        raise ValueError(
            f"agents of type '{typ}' have differing ren_cap {sorted(caps)}. The public θ layout "
            f"gives each TYPE one shared g slot (GS_START/GW_START), so ren_cap must be identical "
            f"within a type. Vary p_max/eta instead, or move g to capacity-factor units and "
            f"rescale by _rcap(i) in _expand_solution/_local_pgne."
        )
    return caps.pop()


def _check_shared_caps():
    for t in ("solar", "wind"):
        type_rcap(t)


set_market(ERCOT)   # default market → validated ERCOT path unchanged for existing callers


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
        lambda_A = (None if cr.lambda_A is None
                    else np.zeros((cr.lambda_A.shape[0], n_full)))
        E[:, :H] = cr.E[:, :H]; A[:, :H] = cr.A[:, :H]
        if lambda_A is not None:
            lambda_A[:, :H] = cr.lambda_A[:, :H]
        for kk, gidx in enumerate(embed):
            E[:, H + gidx] += cr.E[:, H + kk]
            A[:, H + gidx] += cr.A[:, H + kk]
            if lambda_A is not None:
                lambda_A[:, H + gidx] += cr.lambda_A[:, H + kk]
        regs.append(AgentCR(
            E=E, f=cr.f.copy(), A=A, b=cr.b.copy(),
            lambda_A=lambda_A,
            lambda_b=(None if cr.lambda_b is None else cr.lambda_b.copy()),
            active_set=list(cr.active_set), n_constraints=cr.n_constraints,
            index=cr.index,
        ))
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
                                         n_theta_i=L["n_theta"],
                                         n_constraints=G.shape[0])
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
    # g range = the farm size of that type (shared within type; see type_rcap)
    p_ub[GS_START:GS_START + H] = type_rcap("solar")   # solar CF × PV farm
    p_ub[GW_START:GW_START + H] = type_rcap("wind")    # wind CF × wind farm
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
