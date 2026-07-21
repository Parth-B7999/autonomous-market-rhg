"""
simple_game.py — Minimal GNE game structure for RTM participation.

Per-agent decision: p_i ∈ ℝ^H  (net power at PCC, kW; +ve = importing)
Cost:  ½·γᵢ·‖p_i - p̄_i‖²   where p̄_i,k = p^DA_i,k + λ_k·dt/γᵢ
  → Q_i = γᵢ·I_H  (diagonal, trivially PD)
  → No z+/z-, no degradation, no ramp constraints

Local constraints (parametric in SoC_0 only):
  SoC bounds :  ±Γ·p_i ≤ ±SoC_{max/min} ∓ α_vec·SoC_0
  Power box  :  p_min ≤ p_i,k ≤ p_max

Coupling (upper only, H-dimensional per timestep):
  Σ_i p_i,k ≤ L_max
  → C_i = I_H  (trivial, p_i IS the coupling variable)
  L_min dropped — non-binding, causes CR explosion in mpQP

Parameter vector (shared, size n_p):
  [SoC_0^0, ..., SoC_0^{N-1},      N initial states
   λ_0, ..., λ_{H-1},               H RTM prices
   p^DA_0, ..., p^DA_{N-1},         N×H DA schedules
   L_max,                            1 PCC upper limit
   g^PV_0, ..., g^PV_{H-1}]         H PV forecast (optional)
  Total: N + H + N·H + 1 (+ H if PV)

For mpGNE each agent i's private θ_i will be:
  [SoC_0^i (1), λ (H), p^DA_i (H), p_{-i} (H)]  → 3H+1 = 19 for H=6
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from amrhg.solvers.game import Agent, GNEGame


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter layout
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SimpleParameterLayout:
    """
    Index map into the shared RTM parameter vector p.

    Layout (0-indexed):
      [0 .. N-1]          : initial states
      [N .. N+H-1]        : RTM LMP prices
      [N+H .. N+H+N-1]   : DA schedules — ONE scalar per agent (current hourly value)
      [N+H+N]             : L_max  (PCC upper limit, kW)
      (optional) [end..end+H] : PV forecast (H values, one agent only)
    """
    n_agents: int
    H: int
    dt_hr: float
    state_init_indices: tuple[int, ...]
    price_start: int
    da_starts: tuple[int, ...]
    l_max_idx: int
    pv_start: int | None = field(default=None)

    @property
    def n_p(self) -> int:
        base = self.l_max_idx + 1
        return base + (self.H if self.pv_start is not None else 0)

    @property
    def price_slice(self) -> slice:
        return slice(self.price_start, self.price_start + self.H)

    def da_idx(self, i: int) -> int:
        return self.da_starts[i]

    @property
    def pv_slice(self) -> slice | None:
        if self.pv_start is None:
            return None
        return slice(self.pv_start, self.pv_start + self.H)

    def p_lb(self) -> np.ndarray:
        lb = np.full(self.n_p, -np.inf)
        lb[list(self.state_init_indices)] = 0.0
        lb[self.l_max_idx] = 0.0
        if self.pv_start is not None:
            lb[self.pv_start:self.pv_start + self.H] = 0.0
        return lb

    def p_ub(self) -> np.ndarray:
        return np.full(self.n_p, np.inf)


def make_simple_layout(
    n_agents: int,
    H: int,
    dt_hr: float,
    pv_agent_idx: int | None = None,
) -> SimpleParameterLayout:
    """
    Build a SimpleParameterLayout for N agents, horizon H.

    Parameters
    ----------
    pv_agent_idx : int or None
        If set, appends H slots for that agent's PV forecast.
    """
    state_init_indices = tuple(range(n_agents))
    price_start = n_agents
    da_starts = tuple(n_agents + H + i for i in range(n_agents))  # 1 scalar per agent
    l_max_idx = n_agents + H + n_agents
    pv_start = (l_max_idx + 1) if pv_agent_idx is not None else None

    return SimpleParameterLayout(
        n_agents=n_agents,
        H=H,
        dt_hr=dt_hr,
        state_init_indices=state_init_indices,
        price_start=price_start,
        da_starts=da_starts,
        l_max_idx=l_max_idx,
        pv_start=pv_start,
    )


def make_simple_param(
    layout: SimpleParameterLayout,
    state_inits: list[float],
    lmp_rt: np.ndarray,
    da_schedules: list[np.ndarray],
    l_max_kw: float,
    pv_forecast: np.ndarray | None = None,
) -> np.ndarray:
    """Pack all RTM inputs into the shared parameter vector."""
    p = np.zeros(layout.n_p)
    for i, s in enumerate(state_inits):
        p[layout.state_init_indices[i]] = s
    p[layout.price_slice] = lmp_rt
    for i, da in enumerate(da_schedules):
        p[layout.da_idx(i)] = float(da) if np.ndim(da) == 0 else float(da[0])
    p[layout.l_max_idx] = l_max_kw
    if pv_forecast is not None and layout.pv_start is not None:
        p[layout.pv_slice] = pv_forecast
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Shared dynamics helper
# ─────────────────────────────────────────────────────────────────────────────

def _condensed_dynamics(
    H: int, dt: float, alpha: float, eta: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Condensed LTI dynamics:  state_k = alpha_vec[k]·state_0 + Gamma[k,:]·p

    Gamma[k, j] = eta·dt·alpha^{k-j}  for j ≤ k, else 0  (lower-triangular).
    alpha_vec[k] = alpha^{k+1}.
    """
    idx = np.arange(H)
    exp_mat = np.subtract.outer(idx, idx).astype(float)
    Gamma = np.tril(eta * dt * (alpha ** exp_mat))
    alpha_vec = alpha ** (idx + 1)
    return Gamma, alpha_vec


# ─────────────────────────────────────────────────────────────────────────────
#  VRFB agent builder
# ─────────────────────────────────────────────────────────────────────────────

def build_simple_vrfb(
    index: int,
    layout: SimpleParameterLayout,
    soc_min_kwh: float,
    soc_max_kwh: float,
    p_min_kw: float,
    p_max_kw: float,
    roundtrip_eta: float,
    gamma_imb: float = 1.0,
    alpha: float = 1.0,
) -> Agent:
    """
    Build a simple VRFB agent for the RTM GNE game.

    Decision: p_i ∈ ℝ^H  (net power, kW; +ve = charging from grid)
    Cost:     ½·γ·‖p_i‖² + (-dt·λ - γ·p^DA_i)ᵀ·p_i
    Dynamics: SoC_{k+1} = α·SoC_k + η·dt·p_k  (single η, linear approximation)

    Parameters
    ----------
    gamma_imb : float
        Imbalance penalty weight ($/kW²). Replaces z+/z- linear penalty.
        Default 1.0; tune so that γ ≈ Δt/δp where δp is acceptable deviation.
    alpha : float
        Self-discharge factor per step (use 1.0 for VRFB, negligible leakage).
    """
    H = layout.H
    dt = layout.dt_hr
    n_p = layout.n_p
    soc_idx = layout.state_init_indices[index]

    Gamma, alpha_vec = _condensed_dynamics(H, dt, alpha, roundtrip_eta)

    # ── Cost ──────────────────────────────────────────────────────────────────
    Q = gamma_imb * np.eye(H)   # diagonal, trivially PD
    c = np.zeros(H)

    # F maps the parameter vector θ to the linear cost term (c + F·θ)ᵀ·p.
    # Energy cost:      +λ_k·dt·p_k  (agent pays to import, earns to export)
    #   → F[k, price_k] = +dt   (higher λ → higher linear term → agent imports less)
    # Imbalance:        -γ·p_DA_k·p_k  (shifting cost minimum to p_DA)
    #   → F[k, da_k]   = -γ
    # Unconstrained optimum: p* = p_DA - dt·λ/γ  (export more when price is high) ✓
    F = np.zeros((H, n_p))
    ps = layout.price_start
    ds = layout.da_idx(index)
    F[np.arange(H), ps + np.arange(H)] = +dt    # energy cost
    F[np.arange(H), ds]                = -gamma_imb  # DA penalty — all H steps same column

    # ── Local constraints ─────────────────────────────────────────────────────
    # SoC upper: Γ·p ≤ SoC_max·1 - α_vec·SoC_0
    A_soc_ub = Gamma
    b_soc_ub = soc_max_kwh * np.ones(H)
    S_soc_ub = np.zeros((H, n_p))
    S_soc_ub[:, soc_idx] = -alpha_vec

    # SoC lower: -Γ·p ≤ -SoC_min·1 + α_vec·SoC_0
    A_soc_lb = -Gamma
    b_soc_lb = -soc_min_kwh * np.ones(H)
    S_soc_lb = np.zeros((H, n_p))
    S_soc_lb[:, soc_idx] = alpha_vec

    # Power box: p_min ≤ p_k ≤ p_max
    A_pw = np.vstack([np.eye(H), -np.eye(H)])
    b_pw = np.concatenate([p_max_kw * np.ones(H), -p_min_kw * np.ones(H)])
    S_pw = np.zeros((2 * H, n_p))

    A_loc = np.vstack([A_soc_ub, A_soc_lb, A_pw])       # (4H, H)
    b_loc = np.concatenate([b_soc_ub, b_soc_lb, b_pw])  # (4H,)
    S_loc = np.vstack([S_soc_ub, S_soc_lb, S_pw])        # (4H, n_p)

    # ── Coupling ──────────────────────────────────────────────────────────────
    # C = I_H: p_i IS the coupling variable, trivial selection
    C = np.eye(H)

    return Agent(
        index=index, n_x=H, Q=Q, c=c, F=F,
        A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PV + Battery agent builder
# ─────────────────────────────────────────────────────────────────────────────

def build_simple_pv_battery(
    index: int,
    layout: SimpleParameterLayout,
    soc_min_kwh: float,
    soc_max_kwh: float,
    battery_capacity_kw: float,
    roundtrip_eta: float,
    gamma_imb: float = 1.0,
    alpha: float = 1.0,
) -> Agent:
    """
    Simple PV + Li-ion battery agent.

    Decision: p_net ∈ ℝ^H  (net PCC power; +ve = importing from grid)
    PV generation g^PV enters as a disturbance parameter:
        SoC_{k+1} = α·SoC_k + η·dt·(p_net_k + g^PV_k)

    Constraints:
        SoC bounds:    parametric in (SoC_0, g^PV)
        Battery power: -P_batt ≤ p_net + g^PV ≤ P_batt  (parametric in g^PV)

    layout.pv_slice must not be None — set pv_agent_idx in make_simple_layout.
    """
    H = layout.H
    dt = layout.dt_hr
    n_p = layout.n_p
    soc_idx = layout.state_init_indices[index]
    pv_sl = layout.pv_slice
    if pv_sl is None:
        raise ValueError(
            "PV+Battery agent requires pv_agent_idx to be set in make_simple_layout"
        )

    Gamma, alpha_vec = _condensed_dynamics(H, dt, alpha, roundtrip_eta)

    # ── Cost ──────────────────────────────────────────────────────────────────
    # Same as VRFB: PV generation is free, no additional cost term
    Q = gamma_imb * np.eye(H)
    c = np.zeros(H)
    F = np.zeros((H, n_p))
    ps = layout.price_start
    ds = layout.da_idx(index)
    F[np.arange(H), ps + np.arange(H)] = +dt
    F[np.arange(H), ds]                = -gamma_imb  # all H steps same DA column

    # ── Local constraints ─────────────────────────────────────────────────────
    # SoC dynamics include PV:  SoC_k = α_vec·SoC_0 + Γ·(p_net + g^PV)
    # → Γ·p_net ≤ SoC_max - α_vec·SoC_0 - Γ·g^PV
    A_soc_ub = Gamma
    b_soc_ub = soc_max_kwh * np.ones(H)
    S_soc_ub = np.zeros((H, n_p))
    S_soc_ub[:, soc_idx] = -alpha_vec
    S_soc_ub[:, pv_sl]   = -Gamma       # PV charges battery → tightens upper bound

    # → -Γ·p_net ≤ -SoC_min + α_vec·SoC_0 + Γ·g^PV
    A_soc_lb = -Gamma
    b_soc_lb = -soc_min_kwh * np.ones(H)
    S_soc_lb = np.zeros((H, n_p))
    S_soc_lb[:, soc_idx] = alpha_vec
    S_soc_lb[:, pv_sl]   = Gamma        # PV relaxes lower bound

    # Battery power bounds: -P_batt ≤ p_net + g^PV ≤ P_batt
    # Upper: p_net ≤ P_batt - g^PV
    # Lower: -p_net ≤ P_batt + g^PV  (i.e. p_net ≥ -P_batt - g^PV)
    cap = battery_capacity_kw
    A_pw = np.vstack([np.eye(H), -np.eye(H)])
    b_pw = cap * np.ones(2 * H)
    S_pw = np.zeros((2 * H, n_p))
    S_pw[:H, pv_sl] = -np.eye(H)   # PV tightens upper: p_net ceiling drops
    S_pw[H:, pv_sl] =  np.eye(H)   # PV relaxes lower: p_net floor drops further

    A_loc = np.vstack([A_soc_ub, A_soc_lb, A_pw])
    b_loc = np.concatenate([b_soc_ub, b_soc_lb, b_pw])
    S_loc = np.vstack([S_soc_ub, S_soc_lb, S_pw])

    C = np.eye(H)

    return Agent(
        index=index, n_x=H, Q=Q, c=c, F=F,
        A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Electrolyzer agent builder  (PEM or alkaline — same math, different params)
# ─────────────────────────────────────────────────────────────────────────────

def build_simple_electrolyzer(
    index: int,
    layout: SimpleParameterLayout,
    tank_min_kg: float,
    tank_max_kg: float,
    p_max_kw: float,
    eta_prod_kg_per_kwh: float,
    h2_price_per_kg: float,
    gamma_imb: float = 1.0,
    offtake_per_step: float = 0.0,
) -> Agent:
    """
    Simple electrolyzer agent (PEM or alkaline — same QP structure).

    Decision: p_i ∈ [0, P_max]^H  (power consumed; +ve = consuming = making H2)

    Dynamics with constant H2 offtake q (kg/step):
        H2_{k+1} = H2_0 + Γ[k,:]·p - (k+1)·q
    where Γ is the lower-triangular condensed map (η_prod·dt).

    offtake_per_step : float
        Constant H2 withdrawal per 5-min step (kg).
        Typical: h2_daily_target_kg / (steps_per_day=288).
        Shifts b_inv_ub up (relaxes upper bound) and b_inv_lb down
        (enforces minimum production rate to keep tank from emptying).
        Default 0 → original formulation (no offtake, tank fills and shuts down).
    """
    H = layout.H
    dt = layout.dt_hr
    n_p = layout.n_p
    inv_idx = layout.state_init_indices[index]

    # Condensed dynamics for H2 inventory (α=1, no self-discharge)
    Gamma, alpha_vec = _condensed_dynamics(H, dt, alpha=1.0, eta=eta_prod_kg_per_kwh)

    # Cumulative offtake over the horizon: [1·q, 2·q, ..., H·q]
    q_cumul = offtake_per_step * np.arange(1, H + 1)   # (H,) kg

    # ── Cost ──────────────────────────────────────────────────────────────────
    Q = gamma_imb * np.eye(H)
    c = -(h2_price_per_kg * eta_prod_kg_per_kwh * dt) * np.ones(H)
    F = np.zeros((H, n_p))
    ps = layout.price_start
    ds = layout.da_idx(index)
    F[np.arange(H), ps + np.arange(H)] = +dt
    F[np.arange(H), ds]                = -gamma_imb  # all H steps same DA column

    # ── Local constraints ─────────────────────────────────────────────────────
    # H2_{k+1} = H2_0 + Γ[k,:]·p - q_cumul[k]
    # Upper:  Γ·p ≤ tank_max + q_cumul - α_vec·H2_0
    A_inv_ub = Gamma
    b_inv_ub = tank_max_kg * np.ones(H) + q_cumul
    S_inv_ub = np.zeros((H, n_p))
    S_inv_ub[:, inv_idx] = -alpha_vec

    # Lower:  -Γ·p ≤ -tank_min - q_cumul + α_vec·H2_0
    A_inv_lb = -Gamma
    b_inv_lb = -tank_min_kg * np.ones(H) - q_cumul
    S_inv_lb = np.zeros((H, n_p))
    S_inv_lb[:, inv_idx] = alpha_vec

    # Power bounds: 0 ≤ p_k ≤ P_max
    A_pw = np.vstack([np.eye(H), -np.eye(H)])
    b_pw = np.concatenate([p_max_kw * np.ones(H), np.zeros(H)])
    S_pw = np.zeros((2 * H, n_p))

    A_loc = np.vstack([A_inv_ub, A_inv_lb, A_pw])
    b_loc = np.concatenate([b_inv_ub, b_inv_lb, b_pw])
    S_loc = np.vstack([S_inv_ub, S_inv_lb, S_pw])

    C = np.eye(H)

    return Agent(
        index=index, n_x=H, Q=Q, c=c, F=F,
        A_loc=A_loc, b_loc=b_loc, S_loc=S_loc, C=C,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  4-agent game builder (convenience for the paper case study)
# ─────────────────────────────────────────────────────────────────────────────

def build_4agent_simple_game(
    vrfb_cfg: dict,
    pv_cfg: dict,
    pem_cfg: dict,
    alk_cfg: dict,
    H: int = 6,
    dt_hr: float = 5 / 60,
    gamma_imb: float = 1.0,
    steps_per_day: int = 288,
) -> tuple[list[Agent], SimpleParameterLayout, GNEGame]:
    """
    Build the full 4-agent RTM game using configs from base.yaml sub-dicts.

    Agents  (index):
      0 — VRFB          (vrfb_cfg)
      1 — PV+Battery    (pv_cfg)       ← has PV forecast parameter
      2 — Electrolyzer PEM  (pem_cfg)
      3 — Electrolyzer Alk  (alk_cfg)

    H2 offtake rates are derived from each electrolyzer's h2_daily_target_kg
    divided by steps_per_day (288 for 5-min intervals).  This keeps the
    inventory steady-state rather than filling and shutting down.

    Returns (agents, layout, game).
    """
    layout = make_simple_layout(n_agents=4, H=H, dt_hr=dt_hr, pv_agent_idx=1)

    vrfb = build_simple_vrfb(
        index=0, layout=layout,
        soc_min_kwh=vrfb_cfg["soc_min_kwh"],
        soc_max_kwh=vrfb_cfg["soc_max_kwh"],
        p_min_kw=-vrfb_cfg["capacity_kw"],
        p_max_kw=+vrfb_cfg["capacity_kw"],
        roundtrip_eta=vrfb_cfg["roundtrip_efficiency"],
        gamma_imb=gamma_imb,
    )

    pv = build_simple_pv_battery(
        index=1, layout=layout,
        soc_min_kwh=pv_cfg["soc_min_kwh"],
        soc_max_kwh=pv_cfg["soc_max_kwh"],
        battery_capacity_kw=pv_cfg["battery_capacity_kw"],
        roundtrip_eta=pv_cfg["roundtrip_efficiency"],
        gamma_imb=gamma_imb,
    )

    pem_offtake = pem_cfg.get("h2_daily_target_kg", 0.0) / steps_per_day
    pem = build_simple_electrolyzer(
        index=2, layout=layout,
        tank_min_kg=pem_cfg["tank_min_kg"],
        tank_max_kg=pem_cfg["tank_max_kg"],
        p_max_kw=pem_cfg["capacity_kw"],
        eta_prod_kg_per_kwh=pem_cfg["h2_production_kg_per_kwh"],
        h2_price_per_kg=pem_cfg["h2_price_per_kg"],
        gamma_imb=gamma_imb,
        offtake_per_step=pem_offtake,
    )

    alk_offtake = alk_cfg.get("h2_daily_target_kg", 0.0) / steps_per_day
    alk = build_simple_electrolyzer(
        index=3, layout=layout,
        tank_min_kg=alk_cfg["tank_min_kg"],
        tank_max_kg=alk_cfg["tank_max_kg"],
        p_max_kw=alk_cfg["capacity_kw"],
        eta_prod_kg_per_kwh=alk_cfg["h2_production_kg_per_kwh"],
        h2_price_per_kg=alk_cfg["h2_price_per_kg"],
        gamma_imb=gamma_imb,
        offtake_per_step=alk_offtake,
    )

    agents = [vrfb, pv, pem, alk]
    game = build_simple_game(agents, layout)
    return agents, layout, game


# ─────────────────────────────────────────────────────────────────────────────
#  Game builder
# ─────────────────────────────────────────────────────────────────────────────

def build_simple_game(
    agents: list[Agent],
    layout: SimpleParameterLayout,
) -> GNEGame:
    """
    Assemble a GNEGame with one-sided PCC coupling (upper bound only).

    Upper: Σᵢ pᵢ,k ≤ L_max   → d=0, S_coup[:, l_max_idx] = 1
    L_min dropped — non-binding, causes CR explosion in mpQP.
    """
    H = layout.H
    n_p = layout.n_p

    d = np.zeros(H)
    S_coup = np.zeros((H, n_p))
    S_coup[:, layout.l_max_idx] = 1.0

    return GNEGame(
        agents=agents,
        d=d,
        S_coup=S_coup,
        p_lb=layout.p_lb(),
        p_ub=layout.p_ub(),
    )