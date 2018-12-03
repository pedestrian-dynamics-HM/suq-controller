#!/usr/bin/env python3 

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colorbar

from suqc.two_density_dmap import DMAPWrapper
from suqc.two_density_data import load_data, FILE_ACCUM

import multiprocessing

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

    def _func_coeff_lstsq(self, basis, values):
        ck, res = np.linalg.lstsq(basis, values, rcond=1E-14)[:2]

        return ck, res

    def _func_coeff_nystroem(self, basis, values):
        # TODO: there are two options: take K(x,y) = I or compute the kernel matrix
        # kernel_matrix = np.eye(values.shape[0])

        print("WARNING: Use with caution, this is still experimental, also check the TODOs")

        import scipy.spatial

        #self.nystroem_extension(points=self._v_data.iloc[:, :-1].values, values=self._v_data.iloc[:, 0].values)

        kdtree = scipy.spatial.cKDTree(self._v_data.iloc[:, :-1].values)
        dmaps_kdtree = self._dmap.dmap._kdtree

        distance_matrix = kdtree.sparse_distance_matrix(dmaps_kdtree, np.inf, output_type='coo_matrix')

        kernel_matrix = self._dmap.dmap.compute_kernel_matrix(distance_matrix)

        new_basis = kernel_matrix @ basis  # TODO: would require to use this new_basis for back transformation

        # compute coefficients:
        coeff = values @ basis / self._dmap.eigenvalues

        #import diffusion_maps
        #diffusion_maps.GeometricHarmonicsInterpolator

        res = np.linalg.norm(new_basis @ coeff - values)
        print(f"residual norm (different to lstq probably so don't compare!) {res}")

        return coeff, res

    def _compute_func_coeff(self):
        func_coeffs = []

        idx = pd.IndexSlice
        self._v_data.loc[:, idx["QoI_voronoiDensity_scalar", "bump"]] = self._set_bump()

        if self._mode == "fp":
            self._v_data[idx["QoI_voronoiDensity_scalar", "density"]] = self._set_density()

        self._set_p1p2_index()

        phi_old, _ = self._compute_shift_matrices()
        df_basis = self.realdata_basis()

        phi = self._dmap_coordinates()

        MODE = 1

        for i in range(self._v_data.shape[1]):

            if MODE == 1:
                ck, res = self._func_coeff_lstsq(basis=phi_old, values=df_basis.iloc[:, i])
                print(f"Sums of residuals; squared Euclidean 2-norm least-square parameter {i}: {res}")
            elif MODE == 2:
                ck, res = self._func_coeff_nystroem(basis=phi, values=df.iloc[:, i])
            else:
                raise ValueError

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
        return vals

    def _set_p1p2_index(self):
        idx = pd.IndexSlice
        index = self._v_data.index
        self._v_data.loc[:, idx["QoI_voronoiDensity_scalar", "p1_time"]] = index.get_level_values(1)
        self._v_data.loc[:, idx["QoI_voronoiDensity_scalar", "p2_traj"]] = index.get_level_values(0)

    def _set_density(self):

        idx_old, _ = self._compute_shift_indices()

        nr_first_points = 1

        time_idx = self._v_data.index.get_level_values(1).values

        idx_dens = time_idx[idx_old[:nr_first_points]]

        density = np.zeros(self._v_data.shape[0])
        density[np.isin(time_idx, idx_dens)] = 1

        return density

    def solve_koopman_system(self):

        func_coeffs = self._compute_func_coeff()
        phi_old, evK, psiKleft, psiKright = self._compute_eigdecomp_K(plot=False)

        nr_func = len(func_coeffs)

        NT = 300
        res = list()  # TODO: make dict to see which QoI is computed!

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

        kdata_list = self.solve_koopman_system()
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

    def interp_id_func(self, idx):
        return self._setup_nystroem_interp(values=self._v_data.values[:, idx])

    def interp_basis(self):
        phi = self._dmap_coordinates()  # = basis functions (evaluated at known data)
        # interpolate the coordinates itself
        interp_phi = self.nystroem_extension(points=self._v_data.values, values=phi)

        print(f"residual interp_basis with Nyström extension: {np.sqrt(np.mean((phi - interp_phi)**2))}")

        idx_old, _ = self._compute_shift_indices()

        return interp_phi[idx_old, :]

    def interp_gradient(self, points, values):
        interp = self._setup_nystroem_interp(values)
        return interp.gradient(points)

    def interp_gradient_trajectories(self, idx, points):

        #phi = self._dmap_coordinates()  # = basis functions (evaluated at known data)

        # TODO: in Juans Code only one function can be set to get the gradient, the Jacobian is not possible (the entire mapping!)
        interp = self._setup_nystroem_interp(values=self._v_data.iloc[:, idx].values)
        grad = interp.gradient(points)

        return grad

    def get_values_for_time(self, t):
        time_idx = self._v_data.index.get_level_values(1)
        bool_idx = time_idx == t

        if np.sum(bool_idx) <= 0:
            raise ValueError

        return self._v_data.loc[bool_idx, :]

    def fd_func(self, interp, point, idx_dir):

        h = 0.3  # TODO: at the moment all directions the same...

        eval_points = np.tile(point, [2, 1])
        eval_points[0, idx_dir] = eval_points[0, idx_dir]-h
        eval_points[1, idx_dir] = eval_points[1, idx_dir]+h

        eval_func = interp(eval_points)

        return (eval_func[1] - eval_func[0]) / (2*h)

    def fd_jacobi_det(self, points):

        if points.ndim == 1:
            points = points[np.newaxis, :]


        dets = np.zeros(points.shape[0])

        interp_funcs = [self.interp_id_func(j) for j in range(4)]

        for p in range(points.shape[0]):
            jacobi = np.zeros([4, 4])
            for j in range(4):
                grad = np.zeros(4)
                for i in range(4):
                    grad[i] = self.fd_func(interp_funcs[j], points[p, :], idx_dir=i)
                jacobi[j, :] = grad
            dets[p] = np.linalg.det(jacobi)
        return dets

    def _comp_blockjacobian_mp(self, kwargs):
        """parallelized version of jacobian computation"""

        i = kwargs["i"]
        nr_blocks = kwargs["nr_blocks"]
        blocksize = kwargs["blocksize"]
        last_idx = kwargs["last_idx"]

        for b in range(nr_blocks):
            print(f"---block {str(b).zfill(2)} of {nr_blocks}")
            start = b * blocksize
            end = np.min([(b + 1) * blocksize, last_idx])
            # TODO: for testing
            return self.interp_gradient_trajectories(idx=i, points=self._v_data.iloc[start:end, :].values)

    def test(self, arg):
        import time
        print("Hallo")
        time.sleep(20)
        return 2

    def interp_jacobian_basis(self):

        nr_grad, nr_func = self._v_data.shape

        # all points, the jacobian matrix is in each row and needs to be reshaped to be a matrix
        jac_matrix_full = np.zeros([nr_grad, nr_func**2])

        nr_blocks = 50
        blocksize = np.ceil(nr_grad / nr_blocks).astype(np.int)

        MODE = 1

        if MODE == 0:
            # single process
            for i in range(nr_func):
                print(f"func {i + 1} of {nr_func}")

                for b in range(nr_blocks):

                    print(f"---block {str(b).zfill(2)} of {nr_blocks}")

                    start = b*blocksize
                    end = np.min([(b+1)*blocksize, jac_matrix_full.shape[0]])

                    jac_matrix_full[start:end, i*nr_func:i*nr_func+nr_func] = \
                        self.interp_gradient_trajectories(idx=i, points=self._v_data.iloc[start:end, :].values)
        elif MODE == 1:

            pool = multiprocessing.Pool(processes=2)

            inp = lambda i: {"i": i, "nr_blocks": nr_blocks, "blocksize": blocksize, "last_idx": jac_matrix_full.shape[0]}
            inp_list = list(map(inp, np.arange(nr_blocks)))

            self._comp_blockjacobian_mp(inp_list[0])

            #results = pool.map(self.test, inp_list)

        return jac_matrix_full, nr_func

    def interp_det_basis(self, plot):

        jac_matrix, nr_func = self.interp_jacobian_basis()  # jacobians are row_wise

        idx_old, _ = self._compute_shift_indices()
        jac_matrix = jac_matrix[idx_old, :]  # only need the points of basis

        dets = np.zeros(jac_matrix.shape[0])

        for i in range(jac_matrix.shape[0]):
            sign, det_log = np.linalg.slogdet(jac_matrix[i, :].reshape([nr_func, nr_func]))
            dets[i] = np.exp(det_log)  # always positive (--> absolute value), don't need to use the sign

        if plot:
            plt.figure(), plt.imshow(dets.reshape([50, 300]))

    def imshow_jacobi_det(self, p1, p2):
        p1dp1 = np.gradient(p1, axis=1)
        p1dp2 = np.gradient(p1, axis=0)

        p2dp1 = np.gradient(p2, axis=1)
        p2dp2 = np.gradient(p2, axis=0)

        det_mat = np.zeros_like(p1)

        for i in range(p1.shape[0]):
            for j in range(p1.shape[1]):
                jac = np.array([ [[p1dp1[i, j], p1dp2[i, j]],
                                  [p2dp1[i, j], p2dp2[i, j]]]
                                 ])
                det_mat[i, j] = np.linalg.det(jac)
        return det_mat

    def imshow_proj_jacdet(self, tsnap):
        df_basis = self.realdata_basis()  # use only the basis...
        df_basis = df_basis.reset_index()

        p1t0 = df_basis.loc[:, "level_1"].values.reshape([50, 300])  # time step
        p2t0 = df_basis.loc[:, "level_0"].values.reshape([50, 300])  # trajectory number

        #jacobi_det = self.imshow_jacobi_det(p1t0, p2t0)

        res = self.solve_koopman_system()

        p1all, p2all = res[-2], res[-1]

        plot_initial = True
        if plot_initial:
            fig = plt.figure()
            ax = fig.add_subplot(221)
            ax.set_title("exact p1 (time)")
            ax.imshow(p1t0)
            ax = fig.add_subplot(222)
            ax.imshow(p2t0)
            ax.set_title("exact p2 (traj)")

            ax = fig.add_subplot(223)
            ax.imshow(p1all[:, 0].reshape([50,300]))
            ax.set_title("back and forth p1 @ t=1 (time)")

            ax = fig.add_subplot(224)
            ax.imshow(p2all[:, 0].reshape([50,300]))
            ax.set_title("back and forth p2 @ t=1 (traj)")

        p1tsnap = p1all[:, tsnap].reshape([50, 300])
        p2tsnap = p2all[:, tsnap].reshape([50, 300])

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.imshow(p1tsnap)
        ax.set_title(f"p1(time) @ time={tsnap}")

        ax = fig.add_subplot(212)
        ax.imshow(p2tsnap)
        ax.set_title(f"p2(traj) @ time={tsnap}")


        # density push forward

        # ...at start (by definition)
        rho0 = np.zeros([50, 300])
        rho0[:, 0:5] = 1  # first column = 1

        #idx_old, _ = self._compute_shift_indices()
        #rho0 = self._set_bump()[idx_old].reshape([50, 300])

        # ... at time t
        jacobi_det_flowt = self.imshow_jacobi_det(p1tsnap, p2tsnap)
        #jacobi_det_flowt = self.imshow_jacobi_det(p1t0, p2t0)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f"absolute determinant values at t={tsnap}")
        ish = ax.imshow(np.abs(jacobi_det_flowt), cmap="jet")
        fig.colorbar(ish)

        rhot = 1/np.abs(jacobi_det_flowt) * rho0
        rhot[np.isnan(rhot)] = 0

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.imshow(rho0)
        ax.set_title(f"rho @ t=1")

        ax = fig.add_subplot(212)
        ax.imshow(rhot)
        ax.set_title(f"rho @ t={tsnap}")

        print(jacobi_det_flowt)



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

        kpm.imshow_proj_jacdet(5)
        plt.show()
        exit()


        #kpm.interp_gradient_trajectories(points=None)
        #kpm.interp_det_basis(plot=True)

        #kpm.get_values_for_time(1)
        #kpm.fd_jacobi_det(points=df.iloc[0, :].values)

        #plt.show()
        #exit()

        kdata_list = kpm.solve_koopman_system()
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

