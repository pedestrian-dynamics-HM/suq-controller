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


class Operator(object):

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

        phi = self._dmap_coordinates()

        # TODO: orthogonalization methods -- seem not to have much impact currently.
        #phi, sval, _ = np.linalg.svd(phi.T, full_matrices=False)
        #phi, _ = np.linalg.qr(phi)

        phi_old = phi[idx_old, :]  # columns are time, rows are space
        phi_new = phi[idx_new, :]
        return phi_old, phi_new

    def _compute_operator_basis(self):
        return self._compute_shift_matrices()[0]  # phi_old

    def _dmap_coordinates(self):
        return self._dmap.eigenvectors

    def _compute_func_coeff(self):
        func_coeffs = []

        self._set_bump()

        if self._mode == "fp":
            self._set_density()

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

        t = f"There are {np.sum(np.abs(evK) > 1)}/{len(evK)} eigenvals larger than 1, " \
            f"max abs. value = {np.max(np.abs(evK))} min abs. value = {np.min(np.abs(evK))}"
        print(t)

        if plot:
            plt.figure()
            plt.plot(np.real(evK), np.imag(evK), '*')
            plt.plot(np.cos(np.linspace(0, 2 * np.pi, 200)), np.sin(np.linspace(0, 2 * np.pi, 200)))
            plt.title(t)
            plt.axis("equal")

        return phi_old, evK, psiKleft, psiKright

    def _set_bump(self):
        from scipy.stats import multivariate_normal
        point = self._v_data.iloc[400 * 25, :]  # starting value of T1
        vals = multivariate_normal(point, 20).pdf(self._v_data.values) * 100000
        idx = pd.IndexSlice
        self._v_data.loc[:, idx["QoI_voronoiDensity_scalar", "bump"]] = vals

    def _set_density(self):

        idx_old, _ = self._compute_shift_indices()

        nr_first_points = 1

        time_idx = self._v_data.index.get_level_values(1).values

        idx_dens = time_idx[idx_old[:nr_first_points]]

        density = np.zeros(self._v_data.shape[0])
        density[np.isin(time_idx, idx_dens)] = 1

        idx = pd.IndexSlice
        self._v_data[idx["QoI_voronoiDensity_scalar", "density"]] = density

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
            resid = np.sqrt(np.mean(diff_vals**2))  # RMSE on relative errors

            res += resid
        return res


    def _setup_nystroem_interp(self, values):
        import diffusion_maps

        # points, values and epsilon are None because the diffusion maps are already computed and handed
        # via the diffusion_maps keyword
        interp = diffusion_maps.GeometricHarmonicsInterpolator(points=self._v_data.values,
                                                               values=values,
                                                               epsilon=None,
                                                               diffusion_maps=self._dmap.dmap)
        return interp

    def nystroem_extension(self, points, values):
        """points are the values at which to evaluate, values are the function values at the known points"""
        interp = self._setup_nystroem_interp(values)

        # TODO: get coefficients for the initial functions from Nyström extension
        # The Nyströms equation is:
        # \sum "<f, \phi> 1/\lambda_j K(x,y)" \phi(y)   --> the part in "" would then define the coefficients.
        # NOTE: the DMAP is computed with the entire dataset, whereas we are interested to have the basis \Phi_old,
        # For the Phi_old the eigenvalues may have changed... -- or can we use the Phi basis?

        # TODO: compute residual of interpolation w.r.t. to the functions

        return interp(points)

    def interp_basis(self):
        phi = self._dmap_coordinates()  # = basis functions (evaluated at known data)
        # interpolate the coordinates itself
        interp_phi = self.nystroem_extension(self._v_data.values, values=phi)

        print(f"residual interp_basis with Nyström extension: {np.sqrt(np.mean((phi - interp_phi)**2))}")

        idx_old, _ = self._compute_shift_indices()

        return interp_phi[idx_old, :]

    def interp_gradient(self, points, values):
        interp = self._setup_nystroem_interp(values)
        return interp.gradient(points)

    def interp_gradient_trajectories(self, idx, points):

        phi = self._dmap_coordinates()  # = basis functions (evaluated at known data)

        # TODO: in Juans Code only one function can be set to get the gradient, the Jacobian is not possible (the entire mapping!)
        interp = self._setup_nystroem_interp(values=self._v_data.iloc[:, idx].values)

        grad = interp.gradient(points)  # instead of points...

        return grad

    def interp_jacobian_basis(self):

        nr_grad, nr_func = self._v_data.shape

        # all points, the jacobian matrix is in each row and needs to be reshaped to be a matrix
        jac_matrix_full = np.zeros([nr_grad, nr_func**2])

        nr_blocks = 50
        blocksize = np.ceil(nr_grad / nr_blocks).astype(np.int)

        for i in range(nr_func):
            print(f"func {i + 1} of {nr_func}")

            for b in range(nr_blocks):

                print(f"---block {str(b).zfill(2)} of {nr_blocks}")

                start = b*blocksize
                end = np.min([(b+1)*blocksize, jac_matrix_full.shape[0]])

                jac_matrix_full[start:end, i*nr_func:i*nr_func+nr_func] = \
                    self.interp_gradient_trajectories(idx=i, points=self._v_data.iloc[start:end, :].values)

        return jac_matrix_full, nr_func

    def interp_det_basis(self, plot):

        jac_matrix, nr_func = self.interp_jacobian_basis()  # jacobians are row_wise

        idx_old, _ = self._compute_shift_indices()
        jac_matrix = jac_matrix[idx_old, :]  # only need the points of basis

        dets = np.zeros(jac_matrix.shape[0])

        for i in range(jac_matrix.shape[0]):
            dets[i] = np.linalg.det( jac_matrix[i, :].reshape([nr_func, nr_func]) )

        if plot:
            plt.figure(), plt.imshow(dets.reshape([50, 300]))


    @staticmethod
    def plot_res_eps_range(df, eps_range, num_eigenpars: int=100):
        res = np.zeros_like(eps_range)

        for i, eps in enumerate(eps_range):
            print(f"{i}/{len(eps_range)}")
            dmap = DMAPWrapper(df=df, eps=eps, num_eigenpairs=num_eigenpars, mode="fixed",
                               **{"normalize_kernel": False})

            kpm = Operator(dmap=dmap, data=df, mode="k")

            res[i] = kpm.compute_relative_residual()

        plt.plot(eps_range, res, '-*')
        print(f"residuals: {res}")
        plt.show()


def plot_blocks(df, m):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.00)

    steps = 5

    t_factor = 299//4
    vmin, vmax = np.min(m[:, 0:t_factor*(steps-1)]), np.max(m[:, :t_factor*(steps-1)])

    for i in range(steps):
        #vals = m[i * 300:(i + 1) * 300, :]

        t = t_factor * i
        col = m[:, t]
        col = np.reshape(col, [50, 300])

        ax = fig.add_subplot(5, 1, i+1)
        ax.set_title(f"time={t}")

        print(f"sum at time {t} : {np.sum(col)}")

        #ax.imshow(col, vmin=vmin, vmax=vmax)
        ax.imshow(col)


    # Do for the deterministic case:
    fig = plt.figure()
    fig.subplots_adjust(left=0.02, bottom=0.01, right=0.98, top=1, wspace=0.13, hspace=0)

    for i in range(steps):

        for j in range(np.min([df.shape[1], 4])):

            ax = fig.add_subplot(5, 4, (i*4)+j+1)

            if i == 0:
                ax.set_title(f"q{j}")

            qj = df.iloc[:, j]
            qj_mat = qj.values.reshape([50, 300])

            t = t_factor * i
            weight = np.zeros([50, 300])
            weight[:, t] = 1

            ax.set_xlabel(f"time = {t}")

            density_qj = qj_mat * weight
            ax.imshow(density_qj)

    # Do for FP approximation
    fig = plt.figure()
    fig.subplots_adjust(left=0.02, bottom=0.01, right=0.98, top=1, wspace=0.13, hspace=0)

    for i in range(steps):

        density_qj = list()
        for j in range(np.min([df.shape[1], 4])):
            qj = df.iloc[:, j]
            qj_mat = qj.values.reshape([50, 300])

            t = t_factor * i
            weight = m[:, t].reshape([50, 300])

            density_qj.append(qj_mat * weight)

        vmin = np.min([np.min(d) for d in density_qj])
        vmax = np.max([np.max(d) for d in density_qj])

        for j in range(np.min([df.shape[1], 4])):

            ax = fig.add_subplot(5, 4, (i*4)+j+1)

            if i == 0:
                ax.set_title(f"q{j}")
            ax.set_xlabel(f"time = {t}")


            ax.imshow(density_qj[j], vmin=vmin, vmax=vmax)


if __name__ == "__main__":

    MODE = 2

    df = load_data(FILE_ACCUM)
    #df = df.iloc[:, :3]

    if MODE == 1:
        Operator.plot_res_eps_range(df=df, eps_range=[130], num_eigenpars=100)
    elif MODE == 2:
        dmap = DMAPWrapper.load_pickle()

        mode = "k"
        kpm = Operator(dmap=dmap, data=df, mode=mode)

        #kpm.interp_gradient_trajectories(points=None)
        kpm.interp_det_basis(plot=True)
        plt.show()
        exit()

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
                         fr"norm factor, abs error max={np.max(diff_mat_abs):.2f} "
                         fr"({np.max(diff_mat_abs)*100/np.max(np.abs(edata)):.2f}%)")
            im = ax.imshow(diff_mat_abs_rel, vmin=0, vmax=1, cmap=plt.cm.get_cmap("Reds"))

            f.colorbar(im)

            #animate_function(solve_koopman_system(4))
        if mode == "fp":
            plot_blocks(df=kpm.realdata_basis(), m=kdata_list[-2])
        plt.show()

