#!/usr/bin/env python3 

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

from suqc.two_density_data import FILE_ACCUM, load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import diffusion_maps

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


df = pd.read_csv("/home/daniel/REPOS/vadere/vadere/VadereModelTests/TestOSM/output/corrupt/two_density_2018-12-01_13-47-27.767/out.txt", header=[0], index_col=[0,1], delimiter=" ")
df.index = df.index.swaplevel()
df = df.sort_index()

idx = pd.IndexSlice

traj_samples = list()

shift = 2
len_traj = 15

id_ = 0

for i in np.unique(df.index.get_level_values(0)):
    traj = df.loc[idx[i, :], :]

    start = 0

    while start + len_traj < traj.shape[0]:
        df_traj = traj.iloc[start:start + len_traj, :].copy(deep=True)
        df_traj.loc[:, "id"] = id_

        traj_samples.append(df_traj)
        start = start+shift
        id_ += 1

df_samples = pd.concat(traj_samples, axis=0)
df_samples = df_samples.reset_index()
df_samples = df_samples.set_index("id")


corr_time = df_samples.loc[:, "timeStep"][::len_traj]

x_vals = df_samples.loc[:, "x"].values
y_vals = df_samples.loc[:, "y"].values

nr_traj = df_samples.index[-1]+1

x_vals = x_vals.reshape([nr_traj, len_traj])
y_vals = y_vals.reshape([nr_traj, len_traj])

x_vals_cen = x_vals - np.mean(x_vals, axis=1)[:, np.newaxis]
y_vals_cen = y_vals - np.mean(y_vals, axis=1)[:, np.newaxis]

all_vals_cen = np.hstack((x_vals_cen, y_vals_cen))
all_vals = np.hstack((x_vals, y_vals))

print(all_vals_cen.shape)

from scipy.spatial.distance import pdist, squareform

dist = squareform(pdist(all_vals_cen))

eps = 5E-2 * np.median(dist**2)
print(eps)

dmap = diffusion_maps.SparseDiffusionMaps(points=all_vals_cen, epsilon=eps, cut_off=np.inf, num_eigenpairs=10)

#diffusion_maps.plot_diffusion_maps(data=all_vals, dmaps=dmap)

fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1)
    ax.scatter(all_vals[:, 0], all_vals[:, len_traj], c=dmap.eigenvectors[i+1, :], cmap="seismic")
    ax.set_aspect("equal")

fig = plt.figure()
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1)
    ax.scatter(dmap.eigenvectors[1, :], dmap.eigenvectors[i+2, :])
    ax.set_aspect("equal")

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(dmap.eigenvectors[1, :], dmap.eigenvectors[2, :], dmap.eigenvectors[3, :])

bool_idx = corr_time == 1

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(dmap.eigenvectors[1, :], dmap.eigenvectors[2, :], c="black")
reds = ax.scatter(dmap.eigenvectors[1, bool_idx], dmap.eigenvectors[2, bool_idx], c="red")

def update(t):
    bool_idx = corr_time == np.mod(t*2+1, np.max(corr_time))

    ax.clear()
    ax.scatter(dmap.eigenvectors[1, :], dmap.eigenvectors[2, :], c="black")
    ax.scatter(dmap.eigenvectors[1, bool_idx], dmap.eigenvectors[2, bool_idx], c="red")

    #reds.update(dmap.eigenvectors[1, bool_idx], dmap.eigenvectors[2, bool_idx])
    return fig,

import matplotlib.animation

ani = matplotlib.animation.FuncAnimation(fig, update, frames=np.arange(np.max(corr_time)), repeat=True, interval=20)

plt.show()

