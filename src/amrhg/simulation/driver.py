"""
simulation/driver.py — Closed-loop RHG simulation driver.

Runs a 1-week, 5-min resolution receding-horizon simulation with:
  - RHG (explicit mp-GNE policy)
  - DAM-only (commit & stick, no RTM re-optimisation)
  - ADMM iterative baseline

DA schedules are provided externally (fixed for now; swappable for FACET-GNE).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from amrhg.agents.base import MarketAgent, ParameterLayout
from amrhg.game.params import pack_rtm_params
from amrhg.forecasting.pv import generate_rtm_pv_forecast
from amrhg.forecasting.price import generate_rtm_price_forecast
from amrhg.solvers.admm_solver import admm_solve
from amrhg.solvers.game import GNEGame
from amrhg.simulation.logger import SimulationResult


@dataclass
class SimulationState:
    """Mutable state tracked across RTM steps."""

    soc: list[float]          # current SoC / inventory per agent
    prev_power: list[float]   # p_{t-1} per agent (for ramp)
    step: int = 0

    def copy(self) -> SimulationState:
        return SimulationState(
            soc=list(self.soc),
            prev_power=list(self.prev_power),
            step=self.step,
        )


# ---------------------------------------------------------------------------
# DA schedule helpers
# ---------------------------------------------------------------------------

def generate_fixed_da_schedules(
    agents: list[MarketAgent],
    dam_prices: np.ndarray,   # (T_hr,) hourly DAM LMPs
    pv_profile_hourly: np.ndarray | None = None,  # (T_hr,) average PV kW per hour
) -> list[np.ndarray]:
    """
    Heuristic DA schedules: batteries charge at night (low price), discharge at
    peak; electrolyzers run at constant rate when price is below their H₂
    revenue threshold.

    Returns list of N arrays each shape (T_hr,).
    """
    N = len(agents)
    T_hr = len(dam_prices)
    schedules = []
    cheap = dam_prices < np.median(dam_prices)

    for i, ag in enumerate(agents):
        da = np.zeros(T_hr)
        name = ag.name.lower()

        if "vrfb" in name:
            da[cheap] = ag.p_max_kw * 0.6
            da[~cheap] = ag.p_min_kw * 0.6
        elif "pv+battery" in name or "pv" in name:
            if pv_profile_hourly is not None and len(pv_profile_hourly) == T_hr:
                solar_hours = pv_profile_hourly > 50
                da[solar_hours] = -ag.battery_capacity_kw * 0.5
                da[~solar_hours] = ag.battery_capacity_kw * 0.3
            else:
                da[:] = 50.0
        elif "electrolyzer" in name:
            da[cheap] = ag.p_max_kw * 0.7
            da[~cheap] = ag.p_max_kw * 0.3

        schedules.append(da)

    return schedules


# ---------------------------------------------------------------------------
# Agent dynamics stepping
# ---------------------------------------------------------------------------

def _step_dynamics(
    agents: list[MarketAgent],
    state: SimulationState,
    p_rt: np.ndarray,  # (N,) first-step power decisions
    dt_hr: float,
    pv_actual: float = 0.0,
) -> None:
    """Apply first-step RTM power to agent states in-place."""
    for i, ag in enumerate(agents):
        p = p_rt[i]
        name = ag.name.lower()

        if "vrfb" in name:
            # SoC_{t+1} = alpha * SoC_t + eta * dt * p_t
            eta = getattr(ag, "eta", 0.75)
            alpha = getattr(ag, "alpha", 1.0 - 1e-5)
            state.soc[i] = alpha * state.soc[i] + eta * dt_hr * p

        elif "pv+battery" in name or "pv" in name:
            eta = getattr(ag, "eta", 0.92)
            alpha = getattr(ag, "alpha", 1.0 - 1e-5)
            # p_batt = p_net + p_pv  (see pv_battery.py power balance)
            p_batt = p + pv_actual
            state.soc[i] = alpha * state.soc[i] + eta * dt_hr * p_batt

        elif "electrolyzer" in name:
            eta_prod = getattr(ag, "eta_prod", 0.02)
            state.soc[i] = state.soc[i] + eta_prod * dt_hr * p

        # Clamp to physical bounds
        soc_min = getattr(ag, "soc_min_kwh", getattr(ag, "tank_min_kg", None))
        soc_max = getattr(ag, "soc_max_kwh", getattr(ag, "tank_max_kg", None))
        if soc_min is not None:
            state.soc[i] = max(soc_min, state.soc[i])
        if soc_max is not None:
            state.soc[i] = min(soc_max, state.soc[i])

    state.prev_power = list(p_rt)
    state.step += 1


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_closed_loop(
    cfg: dict,
    agents: list[MarketAgent],
    layout: ParameterLayout,
    game: GNEGame,
    gne_solution,   # GNESolution (optional — can be None for ADMM-only)
    ercot_data: dict,
    rng: np.random.Generator | None = None,
) -> dict[str, SimulationResult]:
    """
    Run 1-week closed-loop simulation for all methods.

    Parameters
    ----------
    cfg : dict
        Full config (configs/base.yaml structure).
    agents : list[MarketAgent]
        Length N. Must match layout.n_agents.
    layout : ParameterLayout
    game : GNEGame
        Pre-built 4-agent GNE game.
    gne_solution : GNESolution or None
        Explicit mp-GNE solution. If None, skips RHG method.
    ercot_data : dict
        From load_or_generate_ercot_data(). Must contain rtm_lmp, dam_lmp,
        pv_capacity_factor.
    rng : np.random.Generator or None

    Returns
    -------
    dict[str, SimulationResult] keyed by method: "rhg", "open_loop", "admm".
    """
    if rng is None:
        rng = np.random.default_rng(cfg["simulation"]["seed"])

    sim_cfg = cfg["simulation"]
    settlement_cfg = cfg["settlement"]
    n_days = sim_cfg["n_days"]
    H = layout.H
    dt_hr = layout.dt_hr
    N = layout.n_agents
    T = n_days * 288  # 5-min steps per week

    rtm_lmp = ercot_data["rtm_lmp"]        # (T,)
    dam_lmp = ercot_data["dam_lmp"]         # (n_days*24,)
    pv_cf = ercot_data["pv_capacity_factor"]  # (T,)

    # Collapse PV capacity factor to hourly average for DAM
    pv_hourly = np.mean(pv_cf.reshape(-1, 12), axis=1)  # (n_days*24,)
    pv_capacity = getattr(agents[1], "pv_capacity_kw", 300.0) if N > 1 else 300.0

    # Generate DA schedules once
    da_schedules = generate_fixed_da_schedules(agents, dam_lmp, pv_hourly * pv_capacity)
    da_schedules_arr = np.array(da_schedules)  # (N, n_days*24)

    # Initial state from config
    agent_cfgs = cfg["agents"]
    soc_init = [
        agent_cfgs.get("vrfb", {}).get("soc_init_kwh", 1000.0),
        agent_cfgs.get("pv_battery", {}).get("soc_init_kwh", 750.0),
        agent_cfgs.get("electrolyzer_pem", {}).get("tank_init_kg", 50.0),
        agent_cfgs.get("electrolyzer_alk", {}).get("tank_init_kg", 80.0),
    ][:N]
    prev_power_init = [0.0] * N

    agent_names = [ag.name for ag in agents]

    results = {}

    # ---- RHG (explicit mp-GNE) ----
    if gne_solution is not None:
        state = SimulationState(soc=list(soc_init), prev_power=list(prev_power_init))
        p_net_rhg = np.zeros((N, T))
        soc_rhg = np.zeros((N, T + 1))
        soc_rhg[:, 0] = soc_init
        combo_hist = []
        cost_rhg = np.zeros(N)
        viol_rhg = np.zeros(T)

        for t in range(T):
            hour_idx = t // 12  # which hour in the week

            # Current DA schedule for the next H steps
            da_h = np.zeros((N, H))
            for i in range(N):
                for k in range(H):
                    h = (t + k) // 12
                    if h < da_schedules_arr.shape[1]:
                        da_h[i, k] = da_schedules_arr[i, h]

            # PV forecast
            pv_actual = pv_cf[t] * pv_capacity
            pv_fcast = generate_rtm_pv_forecast(
                pv_actual, H, phi=0.85,
                noise_std=0.12 * pv_capacity,
                pv_capacity_kw=pv_capacity, rng=rng,
            )

            # Price forecast
            price_forecast = generate_rtm_price_forecast(
                float(rtm_lmp[t]), H, phi=0.70,
                mean_price=float(np.mean(rtm_lmp[max(0, t - 288):t + 1])),
                rng=rng,
            )

            # Pack parameters
            pv_forecasts = [None] * N
            ramp_prevs = [None] * N
            for i in range(N):
                if layout.pv_slice(i) is not None:
                    pv_forecasts[i] = pv_fcast.copy()
                if layout.ramp_idx(i) is not None:
                    ramp_prevs[i] = float(state.prev_power[i])

            p_vec = pack_rtm_params(
                state_inits=list(state.soc),
                lmp_rt=price_forecast,
                da_schedules=[da_h[i] for i in range(N)],
                l_max_kw=cfg["market"]["l_max_mw"] * 1000,
                l_min_kw=cfg["market"]["l_min_mw"] * 1000,
                layout=layout,
                pv_forecasts=pv_forecasts,
                ramp_prevs=ramp_prevs,
            )

            # Evaluate explicit GNE
            try:
                x_sol = gne_solution.evaluate(p_vec, tol=1e-4)
                if x_sol is not None:
                    p_rt = np.array([x_sol[game.x_slice(i)][0] for i in range(N)])
                    if hasattr(gne_solution, 'last_combo'):
                        combo_hist.append(gne_solution.last_combo)
                else:
                    # Fallback: use DA schedule
                    p_rt = da_h[:, 0]
                    combo_hist.append(-1)
            except Exception:
                p_rt = da_h[:, 0]
                combo_hist.append(-1)

            p_net_rhg[:, t] = p_rt

            # Compute cost for this step
            for i in range(N):
                ag = game.agents[i]
                xi = np.zeros(ag.n_x)
                xi[0] = p_rt[i]
                # Imbalance: xi - DA = z+ - z-
                imb = p_rt[i] - da_h[i, 0]
                xi[H] = max(imb, 0)
                xi[2 * H] = max(-imb, 0)
                cost_rhg[i] += float(ag.local_cost(xi, p_vec))

            # Coupling violation
            total_p = np.sum(p_rt)
            l_min = cfg["market"]["l_min_mw"] * 1000
            l_max = cfg["market"]["l_max_mw"] * 1000
            viol_rhg[t] = max(0, l_min - total_p, total_p - l_max)

            # Step dynamics
            _step_dynamics(agents, state, p_rt, dt_hr, pv_actual)
            soc_rhg[:, t + 1] = state.soc

        results["rhg"] = SimulationResult(
            method="rhg",
            p_net=p_net_rhg,
            soc=soc_rhg,
            rtm_prices=rtm_lmp,
            dam_prices=dam_lmp,
            da_schedules=da_schedules_arr,
            combo_history=combo_hist,
            total_cost=cost_rhg,
            coupling_violations=viol_rhg,
            n_steps=T,
            agent_names=agent_names,
        )

    # ---- DAM-only (commit & stick) ----
    state = SimulationState(soc=list(soc_init), prev_power=list(prev_power_init))
    p_net_dam = np.zeros((N, T))
    soc_dam = np.zeros((N, T + 1))
    soc_dam[:, 0] = soc_init
    cost_dam = np.zeros(N)
    viol_dam = np.zeros(T)

    for t in range(T):
        hour_idx = t // 12
        p_rt = np.array([da_schedules_arr[i, min(hour_idx, da_schedules_arr.shape[1] - 1)]
                         for i in range(N)])
        p_net_dam[:, t] = p_rt
        pv_actual = pv_cf[t] * pv_capacity

        total_p = np.sum(p_rt)
        l_min = cfg["market"]["l_min_mw"] * 1000
        l_max = cfg["market"]["l_max_mw"] * 1000
        viol_dam[t] = max(0, l_min - total_p, total_p - l_max)

        for i, ag in enumerate(game.agents):
            xi = np.zeros(ag.n_x)
            xi[0] = p_rt[i]
            imb = p_rt[i]
            xi[H] = max(imb, 0)
            xi[2 * H] = max(-imb, 0)
            price_vec = np.zeros(layout.n_p)
            price_vec[layout.price_slice] = rtm_lmp[t]
            cost_dam[i] += float(ag.local_cost(xi, price_vec))

        _step_dynamics(agents, state, p_rt, dt_hr, pv_actual)
        soc_dam[:, t + 1] = state.soc

    results["open_loop"] = SimulationResult(
        method="open_loop",
        p_net=p_net_dam,
        soc=soc_dam,
        rtm_prices=rtm_lmp,
        dam_prices=dam_lmp,
        da_schedules=da_schedules_arr,
        total_cost=cost_dam,
        coupling_violations=viol_dam,
        n_steps=T,
        agent_names=agent_names,
    )

    # ---- ADMM baseline (sub-sampled: every 15 min to keep runtime manageable) ----
    sample_period = 3  # every 3rd step = 15 min
    T_admm = T // sample_period
    state = SimulationState(soc=list(soc_init), prev_power=list(prev_power_init))
    p_net_admm = np.zeros((N, T))
    soc_admm = np.zeros((N, T + 1))
    soc_admm[:, 0] = soc_init
    cost_admm = np.zeros(N)
    viol_admm = np.zeros(T)

    for t_admm in range(T_admm):
        t = t_admm * sample_period
        pv_actual = pv_cf[t] * pv_capacity

        # Build DA block for this step
        da_h = np.zeros((N, H))
        for i in range(N):
            for k in range(H):
                h = (t + k) // 12
                if h < da_schedules_arr.shape[1]:
                    da_h[i, k] = da_schedules_arr[i, h]

        pv_fcast = generate_rtm_pv_forecast(
            pv_actual, H, phi=0.85,
            noise_std=0.12 * pv_capacity,
            pv_capacity_kw=pv_capacity, rng=rng,
        )
        price_forecast = generate_rtm_price_forecast(
            float(rtm_lmp[t]), H, phi=0.70,
            mean_price=float(np.mean(rtm_lmp[max(0, t - 288):t + 1])),
            rng=rng,
        )

        pv_forecasts = [None] * N
        ramp_prevs = [None] * N
        for i in range(N):
            if layout.pv_slice(i) is not None:
                pv_forecasts[i] = pv_fcast.copy()
            if layout.ramp_idx(i) is not None:
                ramp_prevs[i] = float(state.prev_power[i])

        p_vec = pack_rtm_params(
            state_inits=list(state.soc),
            lmp_rt=price_forecast,
            da_schedules=[da_h[i] for i in range(N)],
            l_max_kw=cfg["market"]["l_max_mw"] * 1000,
            l_min_kw=cfg["market"]["l_min_mw"] * 1000,
            layout=layout,
            pv_forecasts=pv_forecasts,
            ramp_prevs=ramp_prevs,
        )

        result = admm_solve(game, p_vec, max_iter=1000, rho=5.0, tol=1e-2, verbose=False)
        x_admm = np.concatenate(result.x_sol)
        p_rt = np.array([x_admm[game.x_slice(i)][0] for i in range(N)])

        # Fill in the sub-sampled steps
        for s in range(sample_period):
            ts = t + s
            if ts < T:
                p_net_admm[:, ts] = p_rt

        total_p = np.sum(p_rt)
        l_min = cfg["market"]["l_min_mw"] * 1000
        l_max = cfg["market"]["l_max_mw"] * 1000
        viol_admm[t] = max(0, l_min - total_p, total_p - l_max)

        for i, ag in enumerate(game.agents):
            xi = np.zeros(ag.n_x)
            xi[0] = p_rt[i]
            imb = p_rt[i] - da_h[i, 0]
            xi[H] = max(imb, 0)
            xi[2 * H] = max(-imb, 0)
            cost_admm[i] += float(ag.local_cost(xi, p_vec))

        _step_dynamics(agents, state, p_rt, dt_hr, pv_actual)
        soc_admm[:, t + sample_period] = state.soc

    results["admm"] = SimulationResult(
        method="admm",
        p_net=p_net_admm,
        soc=soc_admm,
        rtm_prices=rtm_lmp,
        dam_prices=dam_lmp,
        da_schedules=da_schedules_arr,
        total_cost=cost_admm,
        coupling_violations=viol_admm,
        n_steps=T,
        agent_names=agent_names,
    )

    return results
