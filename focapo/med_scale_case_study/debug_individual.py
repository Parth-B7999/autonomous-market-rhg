import numpy as np
ind = np.load('results/individual_20260605_1132_Lmin100_7d/results.npz', allow_pickle=True)
p_elec = ind['p_elec'][0, :, 3]  # Day 1, ALK_Bat_M (agent 3)
for h in range(24):
    print(f"Hr {h:02d}: p_elec = {p_elec[h*12:(h+1)*12].mean():.1f} kW")
