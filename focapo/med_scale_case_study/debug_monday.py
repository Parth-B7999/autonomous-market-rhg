import numpy as np
ind = np.load('results/individual_20260605_1108_Lmin100_7d/results.npz', allow_pickle=True)
p_da = ind['p_da_all'][0, :, 5]
h2_da = p_da.sum() * (1/12) * 0.02
print("DAM scheduled H2:", h2_da)
