#!/usr/bin/env python3
# !/usr/bin/python3

import sys
import re
import warnings
from io import StringIO

from suqc.CommandBuilder.VadereOppCommand import VadereOppCommand

from suqc.utils.SeedManager.OmnetSeedManager import OmnetSeedManager
from tutorial.imports import *

# This is just to make sure that the systems path is set up correctly, to have correct imports, it can be ignored:

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))

run_local = True
###############################################################################################################
# Usecase: Set yourself the parameters you want to change. Do this by defining a list of dictionaries with the
# corresponding parameter. Again, the Vadere output is deleted after all scenarios run.


if __name__ == "__main__":
    output_folder = os.path.join(os.getcwd(), "first_examples_rover_06")

    ## Define the simulation to be used
    # A rover simulation is defined by an "omnetpp.ini" file and its corresponding directory.
    # Use following *.ini file:

    path2ini = os.path.join(
        os.environ["CROWNET_HOME"],
        "crownet/simulations/simple_detoure_suqc_traffic/omnetpp.ini",
    )

    ## Define parameters and sampling method
    # parameters
    # number of repetitions for each sample
    reps = 1

    # sampling
    par_var = [
        {'vadere': {'sources.[id==1].distributionParameters.numberPedsPerSecond': 0.0375},
         'omnet': {'*.misc[0].app[0].incidentTime': '10s',
                   '*.radioMedium.obstacleLoss.typename': 'DielectricObstacleLoss'}},
        {'vadere': {'sources.[id==1].distributionParameters.numberPedsPerSecond': 0.05},
         'omnet': {'*.misc[0].app[0].incidentTime': '10s',
                   '*.radioMedium.obstacleLoss.typename': 'DielectricObstacleLossXXX'}},
        {'vadere': {'sources.[id==1].distributionParameters.numberPedsPerSecond': 0.01},
         'omnet': {'*.misc[0].app[0].incidentTime': '10s',
                   '*.radioMedium.obstacleLoss.typename': 'DielectricObstacleLoss'}},
    ]
    par_var = OmnetSeedManager(par_variations=par_var, rep_count=reps, vadere_fixed=False, omnet_fixed=False) \
        .get_new_seed_variation()

    ## Define the quantities of interest (simulation output variables)
    # Make sure that corresponding post processing methods exist in the run_script2.py file

    qoi = [
        "degree_informed_extract.txt",
        "poisson_parameter.txt",
        "time_95_informed.txt",
    ]

    # define tag of omnet and vadere docker images, see https://sam-dev.cs.hm.edu/rover/rover-main/container_registry/

    model = VadereOppCommand() \
        .create_vadere_container() \
        .vadere_tag("latest") \
        .omnet_tag("latest") \
        .qoi(qoi) \
        .experiment_label("out") \

    setup = CoupledDictVariation(
        ini_path=path2ini,
        config="final",
        parameter_dict_list=par_var,
        qoi=qoi,
        model=model,
        post_changes=None,
        output_path=path2tutorial,
        output_folder=output_folder,
        remove_output=False,
        env_remote=None,
    )

    if os.environ["CROWNET_HOME"] is None:
        raise SystemError(
            "Please add ROVER_MAIN to your system variables to run a rover simulation."
        )

    if run_local:
        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            par_var, data = setup.run(1)
            # check for the amount of retries
            output = out.getvalue().strip().split("\n")
            regex = re.compile(r"Retry attempt:")
            retry_string = filter(lambda text: regex.match(text), output)
            if len(list(retry_string)) == 6:
                print("Do something for the expected amount of retries.")
            else:
                warnings.warn("Did not found the 6 expected retries.")
        finally:
            sys.stdout = saved_stdout

    else:
        par_var, data = setup.remote(-1)

    # Save results
    summary = output_folder + "_df"
    if os.path.exists(summary):
        shutil.rmtree(summary)

    os.makedirs(summary)

    par_var.to_csv(os.path.join(summary, "parameters.csv"))
    for q in qoi:
        data[q].to_csv(os.path.join(summary, f"{q}"))

    print("All simulation runs completed.")
