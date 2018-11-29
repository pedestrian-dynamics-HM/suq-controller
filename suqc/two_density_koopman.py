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


def compute_koopman(df, dmap, plot=False):
    phi = dmap.eigenvectors
    phi, sval, _ = np.linalg.svd(phi.T, full_matrices=False)
    #phi, _ = np.linalg.qr(phi.T, mode="reduced")


    # TODO: check if all are the same length, later possibly generalize to different lengths
    len_traj = len(np.unique(df.index.get_level_values(1).values))
    all_idx = np.arange(0, df.shape[0])

    idx_old = all_idx[~(df.index.get_level_values(1) == len_traj)]  # remove last position of trajectory
    idx_new = all_idx[~(df.index.get_level_values(1) == 1)]  # remove first position of trajectory (note starts at 1 not 0)

    #idx_old = list(set(np.arange(7525)).difference(set(np.arange(301, 7525, 301))))[:-1]
    #idx_new = list(set(np.arange(7525)).difference(set(np.arange(0, 7525+1, 301))))

    phi_old = phi[idx_old, :]  # columns are time, rows are space
    phi_new = phi[idx_new, :]

    #idx = pd.IndexSlice
    #phi_old = df.loc[idx[:, np.arange(1, len_traj)], :]
    #phi_new = df.loc[idx[:, np.arange(2, len_traj+1)], :].values

    #K = np.linalg.pinv(phi_old, rcond=1E-13) @ phi_new

    K = np.linalg.lstsq(phi_old, phi_new, rcond=1E-14)[0]

    #Qold, Rold = np.linalg.qr(phi_old, mode="reduced")  # TODO: more stable lstsq, but there is almost no difference...
    #K = np.linalg.solve(Rold, Qold.T@phi_new)

    evK, psiKright = np.linalg.eig(K)
    B = psiKright @ np.diag(evK)  # Can be speed up by multiplying elementwise with row vector
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

#ev2, Sinv2 = np.linalg.eig(K.T)
#Sinv2 = Sinv2.T
#ev, S2 = np.linalg.eig(K)


# eigenvectors are column wise and normed https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eig.html
#evK, psiKleft, psiKright = linalg.eig(K, left=True, right=True)  # K = S \L S^-1  <-> K S = S \L <-> S^-1 K = \L S^-1
#S, Sinv = psiKright, np.linalg.inv(psiKright)

def get_bump_column_old(df):
    idx = pd.IndexSlice
    quantity = "areaVoronoiDensity.2"
    q0 = df.loc[:, idx["QoI_voronoiDensity_scalar", "areaVoronoiDensity.2"]]
    #q0[np.logical_and(20 <= q0, q0 <= 30)] = 1
    #q0[~np.logical_and(5 <= q0, q0 <= 10)] = 0

    from scipy.stats import norm

    weight = norm.pdf(q0, loc=7.5, scale=1)
    q0 = weight
    df.loc[:, idx["QoI_voronoiDensity_scalar", "bump"]] = q0

    plt.figure()
    plt.plot(df.loc[:, idx["QoI_voronoiDensity_scalar", "areaVoronoiDensity.2"]], q0, '*')
    plt.title(quantity)
    return df


def get_bump_column(df):
    point = df.iloc[450*25, :]  # starting value of T1

    #vals = norm.pdf(df.values, loc=point, scale=3)
    #kde = gaussian_kde(df.values.T)
    #vals = kde(df.values.T)

    from scipy.stats import norm, gaussian_kde, multivariate_normal

    vals = multivariate_normal(point, 20).pdf(df.values)*100000
    idx = pd.IndexSlice
    #df.loc[:, idx["QoI_voronoiDensity_scalar", "bump"]] = vals

    df.loc[:, idx["QoI_voronoiDensity_scalar", "bump"]] = df.iloc[:, 1].values

    return df

def compute_func_coeff(df, dmap):

    func_coeffs = []

    #df = get_bump_column_old(df)
    df = get_bump_column(df)

    all_idx = np.arange(0, df.shape[0])
    len_traj = len(np.unique(df.index.get_level_values(1).values))
    idx_old = all_idx[~(df.index.get_level_values(1) == len_traj)]  # remove last position of trajectory
    idx_new = all_idx[~(df.index.get_level_values(1) == 1)]  # remove first position of trajectory (note starts at 1 not 0)

    phi = dmap.eigenvectors
    phi, sval, _ = np.linalg.svd(phi.T, full_matrices=False)
    phi_old = phi[idx_old, :]  # columns are time, rows are space

    idx = all_idx[df.index.get_level_values(1) == 1]
    phi_old_ls = phi[idx, :]
    for i in range(df.shape[1]):
        ck, res = np.linalg.lstsq(phi_old, df.iloc[idx_old, i], rcond=1E-14)[:2]
        print(f"Sums of residuals; squared Euclidean 2-norm least-square parameter {i}: {res}")
        func_coeffs.append(ck)

    return func_coeffs

    # write basis in terms of Koopman basis
    # E = np.zeros_like(phi_old)
    # for j in range(E.shape[1]):
    #     E[:, j] = coefs_orig(basis=psiK, func_vals=phi_old[:, j])


@DeprecationWarning
def animate_function(m):
    import matplotlib.animation as animation

    idx = pd.IndexSlice
    manifold_points = df.iloc[idx_old, :].loc[:, idx["QoI_voronoiDensity_scalar", "areaVoronoiDensity.2"]]

    fig, ax = plt.subplots()
    line, = ax.plot(manifold_points, m[:, 0], '*')

    def update_plots(i):
        line.set_ydata(m[:, i])
        return line,

    line_ani = animation.FuncAnimation(fig, update_plots, range(m.shape[1]), interval=100, blit=True)
    plt.show()


def plot_blocks(m):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.01)

    t_factor = 5
    vmin, vmax = np.min(m[:, 5]), np.max(m[:, 5])

    for i in range(5):
        #vals = m[i * 300:(i + 1) * 300, :]

        t = t_factor * i
        col = m[:, t]
        col = np.reshape(col, [50, 300])

        ax = fig.add_subplot(5, 1, i+1)
        ax.set_title(f"time={t}")

        ax.imshow(col, vmin=vmin, vmax=vmax)

def animate_first_block():
    pass


def solve_koopman_system(phi_old, evK, psiKleft, psiKright, func_coeffs):
    NT = 300
    cg0 = func_coeffs
    ci = np.zeros([cg0.shape[0], NT])

    g0t = np.zeros([phi_old.shape[0], NT])

    #cur_K = np.eye(K.shape[0])
    #np.linalg.matrix_power(K, t)

    for t in range(NT):
        #ci[:, t] = np.linalg.matrix_power(K, t) @ cg0

        # according to equation
        #ci[:, t] = np.real(psiKright @ np.diag(evK**t) @ psiKleft @ cg0)  # TODO: add check how large imag part is

        # speed up (make n**2 instead of n**3):
        vec = psiKleft @ cg0
        vec = evK**t * vec
        ci[:, t] = np.real(psiKright @ vec)
        g0t[:, t] = phi_old @ ci[:, t]
    return g0t


if __name__ == "__main__":

    df = load_data(FILE_ACCUM)
    dmap = DMAPWrapper.load_pickle()

    all_idx = np.arange(0, df.shape[0])
    len_traj = len(np.unique(df.index.get_level_values(1).values))
    idx_old = all_idx[~(df.index.get_level_values(1) == len_traj)]  # remove last position of trajectory
    idx_new = all_idx[~(df.index.get_level_values(1) == 1)]  # remove first position of trajectory (note starts at 1 not 0)

    phi_old, evK, psiKleft, psiKright = compute_koopman(df, dmap, plot=True)
    func_coeffs = compute_func_coeff(df, dmap)

    for i in range(df.shape[1]):
        kdata = solve_koopman_system(phi_old, evK, psiKleft, psiKright, func_coeffs[i])
        kdata = kdata[np.arange(0, 15000, 300), :]

        #kdata = kdata[:, 0].reshape([50, 300])  # TODO: can be used to check how well the transformation in ck and back worked

        #kdata = solve_koopman_system(i)[:, 0].reshape([50, 300])

        edata = df.iloc[idx_old, i].values  # the entire observation function (as from VADERE)
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

    plot_blocks(solve_koopman_system(phi_old, evK, psiKleft, psiKright, func_coeffs[4]))

    plt.show()

