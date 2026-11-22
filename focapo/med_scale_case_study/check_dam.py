import numpy as np
from agents_med import build_med_fleet, H2_DEMAND_PROFILE
from simulation_sc import load_pjm_week, dam_forecast, solve_individual, H_DA, DT_DA

agents = build_med_fleet()
week = load_pjm_week('2024-07-08', 7)
rng = np.random.default_rng(42)
dam_fc = dam_forecast(week, 0, agents, rng)
bat_soc_dam = np.array([ag.bat_soc_init_frac * ag.bat_cap_kwh for ag in agents])

h2_base = [0.75 * ag.p_elec_max_kw * ag.eta_kg_per_kwh * 24 for ag in agents]
prof = H2_DEMAND_PROFILE[0]
for i, ag in enumerate(agents):
    ag.h2_target_kg_day = h2_base[i] * prof

dam_res = solve_individual(
    agents, dam_fc['da_price'], np.zeros(6), dam_fc['ren'],
    H_DA, DT_DA, bat_soc_dam,
    l_min=100.0, u_fixed=None, mode='dam',
    p_elec_init=np.zeros(6),
)

e_dt = agents[3].eta_kg_per_kwh * DT_DA
print("DAM scheduled H2 for ALK_Bat_M:", (dam_res.p_elec[3, :] * e_dt).sum())
