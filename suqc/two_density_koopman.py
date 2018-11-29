#!/usr/bin/env python3 

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colorbar

import scipy.linalg as linalg

from suqc.two_density_dmap import DMAPWrapper
from suqc.two_density_data import load_data, FILE_ACCUM

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg, Felix Dietrich"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


class Koopman(object):

    def __init__(self, dmap, data, mode: str="k"):

        assert mode in ["k", "fp"]

        self._mode = mode
        self._dmap = dmap

        self._v_data = data
        self._v_basis = None

    def _compute_shift_indices(self):
        all_idx = np.arange(0, self._v_data.shape[0])

        # TODO: check if all are the same length, later possibly generalize to different lengths
        len_traj = len(np.unique(self._v_data.index.get_level_values(1).values))

        # remove last position of trajectory
        idx_old = all_idx[~(self._v_data.index.get_level_values(1) == len_traj)]

        # remove first position of trajectory (note starts at 1 not 0)
        idx_new = all_idx[~(self._v_data.index.get_level_values(1) == 1)]
        return idx_old, idx_new

    def realdata_basis(self):
        idx_old, _ = self._compute_shift_indices()
        return self._v_data.iloc[idx_old, :]

    def _compute_shift_matrices(self):
        idx_old, idx_new = self._compute_shift_indices()

        phi = self._dmap.eigenvectors.T  # TODO: this is transposed, because the DMAP from Juan is assumed

        # TODO: orthogonalization methods -- seem not to have much impact currently.
        #phi, sval, _ = np.linalg.svd(phi.T, full_matrices=False)
        #phi, _ = np.linalg.qr(phi)

        phi_old = phi[idx_old, :]  # columns are time, rows are space
        phi_new = phi[idx_new, :]
        return phi_old, phi_new

    def _compute_basis(self):
        return self._compute_shift_matrices()[0]  # phi_old

    def _compute_func_coeff(self):
        func_coeffs = []

        self._compute_bump()

        phi_old, _ = self._compute_shift_matrices()
        df_basis = self.realdata_basis()

        for i in range(self._v_data.shape[1]):
            ck, res = np.linalg.lstsq(phi_old, df_basis.iloc[:, i], rcond=1E-14)[:2]
            print(f"Sums of residuals; squared Euclidean 2-norm least-square parameter {i}: {res}")
            func_coeffs.append(ck)

        return func_coeffs

    def _compute_eigdecomp_K(self, plot):
        phi_old, phi_new = self._compute_shift_matrices()

        # K = np.linalg.pinv(phi_old, rcond=1E-13) @ phi_new
        if self._mode == "k":
            K = np.linalg.lstsq(phi_old, phi_new, rcond=1E-14)[0]
        else: # mode == "fp"
            K = np.linalg.lstsq(phi_old, phi_new, rcond=1E-14)[0].T

        # TODO: more stable lstsq, but there is almost no difference...
        # Qold, Rold = np.linalg.qr(phi_old, mode="reduced")
        # K = np.linalg.solve(Rold, Qold.T@phi_new)

        evK, psiKright = np.linalg.eig(K)
        B = psiKright @ np.diag(evK)  # TODO: Can speed up by multiplying elementwise with row vector
        psiKleft = np.linalg.solve(B, K)
        del B

        print(f"Residual Euclidean 2-norm eigendecomposition: {np.linalg.norm(psiKright @ np.diag(evK) @ psiKleft - K)}")

        t = f"There are {np.sum(np.abs(evK) > 1)}/{len(evK)} eigenvals larger than 1, max abs. value = {np.max(np.abs(evK))}"
        print(t)

        if plot:
            plt.figure()
            plt.plot(np.real(evK), np.imag(evK), '*')
            plt.plot(np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200)))
            plt.title(t)
            plt.axis("equal")

        return phi_old, evK, psiKleft, psiKright

    def _compute_bump(self):
        from scipy.stats import multivariate_normal
        point = self._v_data.iloc[400 * 25, :]  # starting value of T1
        vals = multivariate_normal(point, 20).pdf(self._v_data.values) * 100000
        idx = pd.IndexSlice
        self._v_data.loc[:, idx["QoI_voronoiDensity_scalar", "bump"]] = vals

    def solve_kooman_system(self):

        func_coeffs = self._compute_func_coeff()
        phi_old, evK, psiKleft, psiKright = self._compute_eigdecomp_K(plot=False)

        nr_func = len(func_coeffs)

        NT = 300
        res = list()

        for i in range(nr_func):
            cg0 = func_coeffs[i]
            ci = np.zeros([cg0.shape[0], NT])

            g0t = np.zeros([phi_old.shape[0], NT])

            # cur_K = np.eye(K.shape[0])
            # np.linalg.matrix_power(K, t)

            for t in range(NT):
                # ci[:, t] = np.linalg.matrix_power(K, t) @ cg0

                # according to equation
                # ci[:, t] = np.real(psiKright @ np.diag(evK**t) @ psiKleft @ cg0)  # TODO: add check how large imag part is

                # speed up (make n**2 instead of n**3):
                vec = psiKleft @ cg0
                vec = evK**t * vec
                ci[:, t] = np.real(psiKright @ vec)
                g0t[:, t] = phi_old @ ci[:, t]
            res.append(g0t)
        return res

    def compute_relative_residual(self):

        kdata_list = self.solve_kooman_system()
        df_basis = self.realdata_basis()

        res = 0
        for i, kdata in enumerate(kdata_list):

            kdata = kdata[np.arange(0, 15000, 300), :]
            edata = df_basis.iloc[:, i].values
            edata = edata.reshape([50, 300])

            norm_factor = np.max(np.abs(edata))

            diff_vals = np.abs(edata - kdata) / norm_factor
            resid = np.sum(diff_vals)

            res += resid
        return res

    @staticmethod
    def plot_res_eps_range(df, eps_range, num_eigenpars: int=100):
        res = np.zeros_like(eps_range)

        for i, eps in enumerate(eps_range):
            print(f"{i}/{len(eps_range)}")
            dmap = DMAPWrapper(df=df, eps=eps, num_eigenpairs=num_eigenpars, mode="fixed")
            kpm = Koopman(dmap=dmap, data=df, mode="k")
            res[i] = kpm.compute_relative_residual()

        plt.plot(eps_range, res, '-*')
        plt.show()



def plot_blocks(m):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.00)

    t_factor = 30
    vmin, vmax = np.min(m[:, 5]), np.max(m[:, 5])

    for i in range(5):
        #vals = m[i * 300:(i + 1) * 300, :]

        t = t_factor * i
        col = m[:, t]
        col = np.reshape(col, [50, 300])

        ax = fig.add_subplot(5, 1, i+1)
        ax.set_title(f"time={t}")

        ax.imshow(col, vmin=vmin, vmax=vmax)


if __name__ == "__main__":

    df = load_data(FILE_ACCUM)


    Koopman.plot_res_eps_range(df=df, eps_range=[130])
    exit()

    dmap = DMAPWrapper.load_pickle()
    kpm = Koopman(dmap=dmap, data=df, mode="k")

    kdata_list = kpm.solve_kooman_system()
    idx_old, _ = kpm._compute_shift_indices()

    df_basis = kpm.realdata_basis()

    for i, kdata in enumerate(kdata_list):
        kdata = kdata[np.arange(0, 15000, 300), :]

        edata = df_basis.iloc[:, i].values  # the entire observation function (as from VADERE)
        edata = edata.reshape([50, 300])

        vmin = np.min([np.min(kdata), np.min(edata)])
        vmax = np.max([np.max(kdata), np.max(edata)])

        f = plt.figure()

        f.suptitle(f"Parameter $q_{{{i}}}$")

        ax = f.add_subplot(311)
        ax.set_title("Koopman")
        im = ax.imshow(kdata, vmin=vmin, vmax=vmax)
        cax = matplotlib.colorbar.make_axes(parents=ax, location="right")
        f.colorbar(im, cax=cax[0])

        ax = f.add_subplot(312)
        ax.set_title("exact")
        im = ax.imshow(edata, vmin=vmin, vmax=vmax)
        cax = matplotlib.colorbar.make_axes(parents=ax, location="right")
        f.colorbar(im, cax=cax[0])

        ax = f.add_subplot(313)
        diff_mat_abs = np.abs(edata - kdata)
        diff_mat_abs_rel = diff_mat_abs / np.max(np.abs(edata))

        ax.set_title(fr"relative error (norm factor={np.max(np.abs(edata)):.2f}), "
                     fr"$ \vert \vert E \vert \vert_{{Fro}} $={np.linalg.norm(edata-kdata, 'fro'):.2f}, "
                     fr"norm factor, abs error max={np.max(diff_mat_abs):.2f} ({np.max(diff_mat_abs)*100/np.max(np.abs(edata)):.2f}%)")
        im = ax.imshow(diff_mat_abs_rel, vmin=0, vmax=1, cmap=plt.cm.get_cmap("Reds"))

        f.colorbar(im)

    #animate_function(solve_koopman_system(4))

    plot_blocks(kdata_list[4])

    plt.show()

