#!/usr/bin/env python3 

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.distance import pdist, squareform

import diffusion_maps
from suqc.two_density_data import load_data, FILE_ACCUM

import importlib.util
spec = importlib.util.spec_from_file_location("VarBwDMAP", "/home/daniel/LRZ Sync+Share/JHU_Work/variable_bandwidth_dmap/vbwdmap.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


class DMAPWrapper(object):

    def __init__(self, df, eps, num_eigenpairs, mode, **kwargs):
        self.epsilon = eps

        if mode == "fixed":
            self.dmap = diffusion_maps.SparseDiffusionMaps(points=df.values, epsilon=eps, num_eigenpairs=num_eigenpairs,
                                                      cut_off=np.inf, **kwargs)
            self.eigenvalues, self.eigenvectors = self.dmap.eigenvalues, self.dmap.eigenvectors.T
        elif mode == "variable":
            self.dmap = foo.VarBwDMAP(data=df.values, eps=eps, d=3, beta=-0.5, k=df.values.shape[0])
            self.eigenvalues, self.eigenvectors = self.dmap.dmap_eigenpairs(num_eigenpairs=num_eigenpairs)
            self.eigenvectors = self.eigenvectors.T

            self.eigenvalues = self.eigenvalues[::-1]
            self.eigenvectors = self.eigenvectors[::-1, :]

        elif mode == "id":
            self.eigenvectors = df.values  #
        else:
            raise ValueError

    @staticmethod
    def load_pickle():
        with open("dmap_file_BACKUP_best.p", "rb") as f:
            dmap = pickle.load(f)
        return dmap


    def save_pickle(self):
        with open("dmap_file_BACKUP_best.p", "wb") as f:
            pickle.dump(self, f)


def plot_eigenvalues(dmap):
    idx = np.arange(0, dmap.eigenvalues.shape[0])
    plt.plot(idx, dmap.eigenvalues, "-*")


def plot_single_trajectories(df, par_ids):
    idx = pd.IndexSlice
    traj = df.loc[idx[par_ids, :], :]


    f = plt.figure()
    xval = traj.loc[par_ids[0], :].index.values # get time from first traj and assume the same for all other

    for i in range(4):
        ax = f.add_subplot(2, 2, i + 1)
        ax.set_title(f"{par_ids[i]}")
        for j in range(traj.shape[1]):
            ax.plot(xval, traj.loc[par_ids[i], :].iloc[:, j], '-*', label=traj.columns.get_level_values(1)[j])
        ax.legend()

def plot_eigenfunction_pairs(dmap, color):
    cmap, c = color

    fix = 1
    var = [2, 3, 4, 5, 6, 7]
    evfix = dmap.eigenvectors[fix, :]
    evvar = dmap.eigenvectors[var, :]

    f = plt.figure()
    f.suptitle(f"eps={dmap.epsilon}")

    for i in range(6):
        ax = f.add_subplot(2, 3, i+1)
        im = ax.scatter(evfix, evvar[i, :], c=c, cmap=cmap)
        ax.set_title(f"eig.val. of idx {var[i]} = {dmap.eigenvalues[i]}")
        ax.set_xlabel(f"$\Psi_{{{fix}}}$"), ax.set_ylabel(f"$\Psi_{{{var[i]}}}$")

    f.subplots_adjust(right=0.76)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)


def plot3d_eigenfunctions(dmap, color):
    pairs = [[1, 2, 3], [1, 3, 4], [1, 2, 4], [1, 2, 5], [1, 4, 5], [1, 3, 4]]

    cmap, c = color

    f = plt.figure()
    f.suptitle(f"eps={dmap.epsilon}")

    for i, p in enumerate(pairs):
        ax = f.add_subplot(2, 3, i+1, projection="3d")
        ax.scatter(dmap.eigenvectors[p[0], :], dmap.eigenvectors[p[1], :], dmap.eigenvectors[p[2], :], c=c, cmap=cmap)
        ax.set_xlabel(f"$\Psi_{{{p[0]}}}$"), ax.set_ylabel(f"$\Psi_{{{p[1]}}}$"), ax.set_zlabel(f"$\Psi_{{{p[2]}}}$")


def get_color(df, mode):
    if mode == "time":
        return plt.get_cmap("plasma"), df.index.get_level_values(1).values
    elif mode == "traj":
        return plt.get_cmap("tab20"), df.index.get_level_values(0).values
    else:
        raise ValueError


if __name__ == "__main__":
    df = load_data(FILE_ACCUM)
    print(df)

    # df = df.iloc[:, :3]  # test how much effect the conservation" parameter q3 has...

    compute_dmap = True

    if compute_dmap:
        dm = DMAPWrapper(df=df, eps=130, num_eigenpairs=200, mode="fixed", **{"normalize_kernel": False})
        dm.save_pickle()
    else:
        ep = np.sqrt(np.median(squareform(pdist(df.values, metric="sqeuclidean"))))

        factors = [1, 0.7, 0.5, 0.4, 0.3]
        factors = []

        get_color(df, mode="traj")

        #plot_single_trajectories(df, par_ids=[0, 10, 15, 24])

        for f in factors:

            ep_val = ep * f

            dmap = DMAPWrapper(df=df, eps=ep_val, num_eigenpairs=10, mode="fixed")

            #plot_eigenvalues(dmap)
            plot_eigenfunction_pairs(dmap, color=get_color(df, mode="traj"))
            #plot3d_eigenfunctions(dmap, color=get_color(df, mode="traj"))

        plt.show()




