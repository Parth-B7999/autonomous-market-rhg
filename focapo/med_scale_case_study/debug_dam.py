import numpy as np
ind = np.load('results/individual_20260605_1132_Lmin100_7d/results.npz', allow_pickle=True)
h2_cum = ind['h2_cum'][0, :, 3]

# Need to rerun dam_forecast to get h2_dam_cum!
# Let's just do it directly.
from agents_med import build_med_fleet
from simulation_sc import load_pjm_week, dam_forecast, solve_individual, H_DA, DT_DA
rng = np.random.default_rng(42)
agents = build_med_fleet()
week = load_pjm_week('2024-07-08', 7)
dam_fc = dam_forecast(week, 0, agents, rng)
bat_soc_dam = np.array([ag.bat_soc_init_frac * ag.bat_cap_kwh for ag in agents])

dam_res = solve_individual(
    agents, dam_fc['da_price'], np.zeros(6), dam_fc['ren'],
    H_DA, DT_DA, bat_soc_dam,
    l_min=100.0, u_fixed=None, mode='dam',
    p_elec_init=np.zeros(6),
)

h2_dam_cum = np.zeros(289)
e_dt = agents[3].eta_kg_per_kwh * DT_DA / 12.0
cum = 0.0
for h in range(24):
    step_val = dam_res.p_elec[3, h] * e_dt
    for step in range(12):
        cum += step_val
        h2_dam_cum[12*h + step + 1] = cum

print("At Hr 13 (t=156):")
print("h2_dam_cum:", h2_dam_cum[156])
print("h2_acc:", h2_cum[156])
print("Total DAM H2:", h2_dam_cum[288])
