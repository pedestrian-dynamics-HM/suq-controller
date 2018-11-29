#!/usr/bin/env python3 

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

import numpy as np
import matplotlib.pyplot as plt

from suqc.two_density_data import load_data, FILE_ACCUM
from suqc.two_density_dmap import DMAPWrapper
from suqc.two_density_koopman import compute_koopman, compute_func_coeff, solve_koopman_system

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


df = load_data(FILE_ACCUM)


def compute_residual(kdata, ecol):

    all_idx = np.arange(0, df.shape[0])
    len_traj = len(np.unique(df.index.get_level_values(1).values))
    idx_old = all_idx[~(df.index.get_level_values(1) == len_traj)]

    edata = df.iloc[idx_old, ecol].values  # exact data
    edata = edata.reshape([50, 300])
    diff_mat_abs = np.abs(edata - kdata)

    resid = np.linalg.norm(diff_mat_abs, 'fro')
    return resid

# plot range of eps_values:


#eps_range = np.linspace(130)
eps_range = [130]
res = np.zeros_like(eps_range)

for i, eps in enumerate(eps_range):
    print(f"{i}/{len(eps_range)}")
    dmap = DMAPWrapper(df=df, eps=eps, num_eigenpairs=200, mode="fixed")
    phi_old, evK, psiKleft, psiKright = compute_koopman(df, dmap, plot=False)
    func_coeffs = compute_func_coeff(df, dmap)

    res_tmp = 0
    for j in range(df.shape[1]):
        kdata = solve_koopman_system(phi_old, evK, psiKleft, psiKright, func_coeffs[j])
        kdata = kdata[np.arange(0, 15000, 300), :]
        res_tmp += compute_residual(kdata, ecol=j)
    res[i] = res_tmp

plt.plot(eps_range, res, '-*')
plt.show()