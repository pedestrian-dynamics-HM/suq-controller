#!/usr/bin/env python3 

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

import numpy as np

from suqc.two_density_dmap import DMAPWrapper
from suqc.two_density_data import load_data, FILE_ACCUM

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


df = load_data(FILE_ACCUM)

def coefs_orig(basis, func_vals):
    # basis * coeff = func_vals
    return np.linalg.lstsq(basis.T, func_vals, rcond=None)[0]


dmap = DMAPWrapper.load_pickle()

phi = dmap.eigenvectors  # TODO: orthogonalize this...
phi, _, _ = np.linalg.svd(phi.T, full_matrices=False)


idx_old = list(set(np.arange(7525)).difference(set(np.arange(301, 7525, 301))))
idx_new = list(set(np.arange(7525)).difference(set(np.arange(0, 7525+1, 301))))

phi_old = phi[idx_old, :][:-1]  # columns are time, rows are space
phi_new = phi[idx_new, :]

K = np.linalg.pinv(phi_old, rcond=1E-13) @ phi_new

evK, psiK = np.linalg.eig(K)


func_coeffs = []

for i in range(df.shape[1]):
    ck = np.linalg.lstsq(phi_old, df.iloc[idx_old[:-1], i], rcond=1E-14)[0]
    func_coeffs.append(ck)
#    func_coeffs.append(coefs_orig(basis=phi_old, func_vals=df.iloc[:, i]))


# write basis in terms of Koopman basis
#E = np.zeros_like(phi_old)
# for j in range(E.shape[1]):
#     E[:, j] = coefs_orig(basis=psiK, func_vals=phi_old[:, j])


# solve system:
# init_vals = df.iloc[0, :]   # simply the re-do the first trajectory # corresponds to user input or sampling...

NT = 200

cg0 = func_coeffs[0]
ci = np.zeros([cg0.shape[0], NT])

g0t = np.zeros([phi_old.shape[0], NT])

for t in range(NT):
    ci[:, t] = np.linalg.matrix_power(K, t) @ cg0
    g0t[:, t] = phi_old @ ci[:, t]


import matplotlib.pyplot as plt
plt.figure()
for i in range(NT):
    plt.plot(g0t[::301, i])

f = plt.figure()

ax = f.add_subplot(211)
ax.imshow(g0t[::301, :], vmin=0, vmax=10)

ax = f.add_subplot(212)

d = df.iloc[idx_old[:-1], 0].values.reshape([25, 300])[:, :200]

ax.imshow(d, vmin=0, vmax=10)

plt.show()






