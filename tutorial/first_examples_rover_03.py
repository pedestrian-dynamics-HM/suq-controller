#!/usr/bin/env python3
# !/usr/bin/python3

import sys
from tutorial.imports import *

# This is just to make sure that the systems path is set up correctly, to have correct imports, it can be ignored:

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))

run_local = True
###############################################################################################################
# Usecase: Set yourself the parameters you want to change. Do this by defining a list of dictionaries with the
# corresponding parameter. Again, the Vadere output is deleted after all scenarios run.


if __name__ == "__main__":

    if os.environ["CROWNET_HOME"] is None:
        raise SystemError(
            "Please add CROWNET_HOME to your system variables to run a rover simulation."
        )

    output_folder = os.path.join(os.getcwd(), "first_examples_rover_03")

    ## Define the simulation to be used
    # A rover simulation is defined by an "omnetpp.ini" file and its corresponding directory.
    # Use following *.ini file:

    path2ini = os.path.join(
        os.environ["CROWNET_HOME"],
        "crownet/simulations/mucFreiheitLte/omnetpp.ini",
    )

    # Currently no qoi supported only raw output.

    setup = CrownetSumoRequest.create(
        ini_path=path2ini,
        config="mucSumo_120_maptypes",
        parameter_dict_list=[
            {"omnet": {
                "sim-time-limit": "10s",
                "*.pNode[*].app[2].app.mapTypeLog": '"ymf"',
                "*.pNode[*].app[2].app.mapType": '"ymf"'}},
            {"omnet": {
                "sim-time-limit": "10s",
                "*.pNode[*].app[2].app.mapTypeLog": '"mean"',
                "*.pNode[*].app[2].app.mapType": '"mean"'}},
            {"omnet": {
                "sim-time-limit": "10s",
                "*.pNode[*].app[2].app.mapTypeLog": '"median"',
                "*.pNode[*].app[2].app.mapType": '"median"'}},
            {"omnet": {
                "sim-time-limit": "10s",
                "*.pNode[*].app[2].app.mapTypeLog": '"invSourceDist"',
                "*.pNode[*].app[2].app.mapType": '"invSourceDist"'}},
        ],
        output_path="/home/vm-sts/repos/crownet/analysis/suq-controller/tutorial/",
        output_folder=output_folder,
        repeat=1,
        seed_config={"sumo": "fixed", "omnet": "random"},
        debug=True
    )

    _, _ = setup.run(njobs=1)
    # _, _ = setup.run(njobs=-1)

    print("All simulation runs completed.")
