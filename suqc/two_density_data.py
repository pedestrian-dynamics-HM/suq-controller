#!/usr/bin/env python3 



import os

import suqc
from suqc.parameter.postchanges import PostScenarioChange

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


ENV_MAN = suqc.EnvironmentManager("two_density")

FILE_PARLOOKUP = "par_lookup.csv"
FILE_SIMULATION_NAME = "simresult"
FILE_SIMULATION = lambda _id: f"{FILE_SIMULATION_NAME}{str(_id).zfill(3)}.csv"
FILE_ACCUM = "averaged_results.csv"


def plot_data(df):
    plt.plot(df["QoI_voronoiDensity_scalar"]["areaVoronoiDensity"], df["QoI_voronoiDensity_scalar"]["areaVoronoiDensity.2"], '*')
    plt.xlabel("density bus"), plt.ylabel("density second door")
    plt.show()


def plot3d(df):

    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')

    ax.plot(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], '*', markersize=10)

    ax.set_xlabel("density bus")
    ax.set_ylabel("density door1")
    ax.set_zlabel("density door2")

    plt.show()


def animate_data(df):

    trajectories = [i[1] for i in df.groupby(level=0)]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim3d([0.0, np.max(df.iloc[:, 0])])
    ax.set_xlabel('count inside bus')

    ax.set_ylim3d([0.0, np.max(df.iloc[:, 1])])
    ax.set_ylabel('count bus door')

    ax.set_zlim3d([0.0, np.max(df.iloc[:, 2])])
    ax.set_zlabel('count second door')

    timesteps = list(map(lambda x: x[1], trajectories[0].index.get_values()))

    lines = [ax.plot(dat.iloc[0:1, 0],
                     dat.iloc[0:1, 1],
                     dat.iloc[0:1, 2], '-*')[0]
             for dat in trajectories]

    def update_lines(num, traj, lines):
        for line, data in zip(lines, traj):

            # NOTE: there is no .set_data() for 3 dim data...
            #print(data.iloc[:num, 0:2].values.T)

            line.set_data(data.iloc[:num, 0:2].values.T)
            line.set_3d_properties(data.iloc[:num, 2].values.T)
        return lines

    line_ani = animation.FuncAnimation(fig, update_lines, timesteps, fargs=(trajectories, lines), interval=100, blit=False)

    plt.show()


def get_hist_at_time(time, traj):
    hist_data = pd.concat(
        [t.reset_index(level=0, drop=True).loc[time, :] for t in traj],
        # go through all trajectories and get the row for the current time
        ignore_index=True, axis=1).reset_index(
        level=0, drop=True)

    hist_time = []

    bins = 100

    for par in range(4):
        n, bins = np.histogram(hist_data.iloc[par, :].values, bins=bins, range=(0, 100), density=True)

        binsd = np.diff(bins)
        width = binsd[0]
        x = np.cumsum(binsd) - width/2
        y = n

        from scipy.interpolate import interp1d
        inp = interp1d(x, y, kind="nearest", fill_value="extrapolate")
        xi = np.linspace(0, 100, 500)
        yi = inp(xi)

        hist_time.append((xi, yi))
    return hist_time


def initialize_hist_data(ax, hist_data):
    import matplotlib.path as path
    import matplotlib.patches as patches

    n, bins = hist_data

    # According to https://matplotlib.org/examples/animation/histogram.html

    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n
    nrects = len(left)


    nverts = nrects * (1 + 3 + 1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    verts[0::5, 0] = left
    verts[0::5, 1] = bottom
    verts[1::5, 0] = left
    verts[1::5, 1] = top
    verts[2::5, 0] = right
    verts[2::5, 1] = top
    verts[3::5, 0] = right
    verts[3::5, 1] = bottom

    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(
        barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
    ax.add_patch(patch)

    return ax, verts, bottom, patch


def animate_data_full(df):

    trajectories = [i[1] for i in df.groupby(level=0)]

    fig = plt.figure(figsize=[15, 9])
    fig.subplots_adjust(left=0.04, right=0.95, top=0.93, wspace=0.1, hspace=0.34)
    ax = fig.add_subplot(1, 2, 1, projection="3d")

    ax.set_xlim(0.0, np.max(df.iloc[:, 0]))
    ax.set_xlabel('count inside bus')

    ax.set_ylim(0.0, np.max(df.iloc[:, 1]))
    ax.set_ylabel('count bus door')

    ax.set_zlim(0.0, np.max(df.iloc[:, 2]))
    ax.set_zlabel('count second door')

    left_side_plots = list()
    lsverts = list()

    hist_data = get_hist_at_time(1, trajectories)

    # ax1 = fig.add_subplot(4, 2, 2)
    # ax1, verts1, bottom, patch = initialize_hist_data(ax1, hist_data[0])
    # left_side_plots.append(ax1)
    # lsverts.append(verts1)
    # ax2 = fig.add_subplot(4, 2, 4)
    # left_side_plots.append(initialize_hist_data(ax2, hist_data[1]))
    #
    # ax3 = fig.add_subplot(4, 2, 6)
    # left_side_plots.append(initialize_hist_data(ax3, hist_data[2]))
    #
    # ax4 = fig.add_subplot(4, 2, 8)
    # left_side_plots.append(initialize_hist_data(ax4, hist_data[3]))

    x, y = hist_data[0]
    ax1 = fig.add_subplot(4, 2, 2)
    ax1.set_xlim(0, 100), ax1.set_ylim(0, 1), ax1.set_xlabel("(0) #ped bus"), ax1.set_ylabel("density")
    left_side_plots.append(ax1.plot(x, y, '-')[0])

    x, y = hist_data[1]
    ax2 = fig.add_subplot(4, 2, 4)
    ax2.set_xlim(0, 100), ax2.set_ylim(0, 1), ax2.set_xlabel("(1) #ped door bus"), ax2.set_ylabel("density")
    left_side_plots.append(ax2.plot(x, y, '-')[0])

    x, y = hist_data[2]
    ax3 = fig.add_subplot(4, 2, 6)
    ax3.set_xlim(0, 100), ax3.set_ylim(0, 1), ax3.set_xlabel("(2) #ped door exit"), ax3.set_ylabel("density")
    left_side_plots.append(ax3.plot(x, y, '-')[0])

    x, y = hist_data[3]
    ax4 = fig.add_subplot(4, 2, 8)
    ax4.set_xlim(0, 100), ax4.set_ylim(0, 1), ax4.set_xlabel("(3) #ped(t=0) - (0) - (1) - (2)"), ax4.set_ylabel("density")
    left_side_plots.append(ax4.plot(x, y, '-')[0])


    timesteps = list(map(lambda x: x[1], trajectories[0].index.get_values()))

    lines = [ax.plot(dat.iloc[0:1, 0],
                     dat.iloc[0:1, 1],
                     dat.iloc[0:1, 2], '-*')[0]
             for dat in trajectories]

    def update_plots(time, traj, lines, lsplots):

        for line, data in zip(lines, traj):
            # NOTE: there is no .set_data() for 3 dim data...
            #print(data.iloc[:num, 0:2].values.T)

            line.set_data(data.iloc[:time, 0:2].values.T)
            line.set_3d_properties(data.iloc[:time, 2].values.T)

        hist_time = get_hist_at_time(time=time, traj=traj)

        for i, hist in enumerate(hist_time):
            _, y = hist
            left_side_plots[i].set_ydata(y)

        #for i, lsp in enumerate(lsplots):
        # n, bins = hist_time[0]
        # top = bottom + n
        #
        # print(n)
        #
        # verts1[1::5, 1] = top
        # verts1[2::5, 1] = top

        return lines, lsplots,

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=3600)

    line_ani = animation.FuncAnimation(fig, update_plots, timesteps, fargs=(trajectories, lines, left_side_plots),
                                       interval=100, blit=False)

    line_ani.save('lines.mp4', writer=writer)

    plt.show()


def remove_existing_files():
    for i in os.listdir(ENV_MAN.env_path):
        if i.startswith(FILE_SIMULATION_NAME) or i.startswith(FILE_PARLOOKUP):
            os.remove(os.path.join(ENV_MAN.env_path, i))


class TwoDensityChanges(PostScenarioChange):

    def __init__(self, name):
        self.name = name
        super(TwoDensityChanges, self).__init__(name=name)

    def get_changes_dict(self, scenario, par_id, par_var):
        from suqc.utils.dict_utils import deep_dict_lookup
        # set maxSpawnNumberTotal to the same as spawnNumber
        spnval, _ = deep_dict_lookup(scenario, key="sources.[id==1].spawnNumber")

        return {"sources.[id==1].maxSpawnNumberTotal": spnval}

def load_data(filename):
    return pd.read_csv(os.path.join(ENV_MAN.env_path, filename), header=[0, 1], index_col=[0, 1])


def create_postchanges(i):
    pc = suqc.ScenarioChanges(apply_default=False)

    pc.add_scenario_change(suqc.parameter.postchanges.ChangeScenarioName())
    pc.add_scenario_change(suqc.parameter.postchanges.ChangeRealTimeSimTimeRatio())
    pc.add_scenario_change(suqc.parameter.postchanges.ChangeDescription())

    def rngfunc(scenario, par_id, par_var, **kwargs):
        return kwargs["avrun"] * 100 + par_id

    rand_change = suqc.parameter.postchanges.ChangeRandomNumber(vfunc=True, **{"avrun": i+1})
    rand_change.set_vfunc(func=rngfunc)

    pc.add_scenario_change(rand_change)
    pc.add_scenario_change(TwoDensityChanges("max_spawn"))

    return pc


def simulate(peds, av_runs):
    remove_existing_files()

    for i in range(av_runs):

        print(f"Running {i} of {av_runs} average runs.")

        pv = suqc.FullGridSampling()
        pv.add_dict_grid({"sources.[id==1].spawnNumber": peds})

        q1 = suqc.AreaDensityVoronoiProcessor(ENV_MAN)
        #q1 = suqc.CombinedTwoDensityProcessor(ENV_MAN, p1=[0, 0], p2=[20, 20])

        pc = create_postchanges(i)

        #with suqc.ServerConnection() as sc:
        #    ss = suqc.ServerSimulation(sc)
        #    parlu, qois = ss.run(env_man=ENV_MAN, par_var=pv, qoi=q1, sc=pc)

        parlu, qois = suqc.Query(ENV_MAN, pv, q1, pc).run(njobs=1)
        if i == 0:
            parlu.to_csv(os.path.join(ENV_MAN.env_path, FILE_PARLOOKUP))
        qois.to_csv(os.path.join(ENV_MAN.env_path, FILE_SIMULATION(i)))

    df = average_data()
    df = add_parameter(df)
    save_final_df(df)

def save_final_df(df):
    df.to_csv(os.path.join(ENV_MAN.env_path, FILE_ACCUM))

def add_parameter(df):

    def const_new_parameter(df):
        initial_val = df.iloc[0, 0]
        new_par = initial_val - df.iloc[:, 0] - df.iloc[:, 1] - df.iloc[:, 2]
        df.loc[:, ("QoI_voronoiDensity_scalar", "combined_par")] = new_par
        return df
    return df.groupby(level=0).apply(const_new_parameter)

def average_data():
    data = list()
    for i in os.listdir(ENV_MAN.env_path):
        if i.startswith(FILE_SIMULATION_NAME):
            data.append(load_data(i))

    accum = np.zeros_like(data[0].values)

    for i in data:
        accum += i
    accum /= len(data)
    return pd.DataFrame(data=accum, index=data[0].index, columns=data[0].columns)


if __name__ == "__main__":
    SIM = True

    if SIM:
        peds = np.linspace(10, 100, 50).astype(np.int)
        avruns = 100
        simulate(peds=peds, av_runs=avruns)
    else:
        df = load_data(FILE_ACCUM)

        print(df)

        #plot_data(df)
        #plot3d(df)
        animate_data_full(df)



