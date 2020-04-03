#!/usr/bin/env python3

import os
import sys

from suqc.parameter.parameter import LatinHyperCubeSampling, Parameter
from tutorial.imports import *

# This is just to make sure that the systems path is set up correctly, to have correct imports, it can be ignored:
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))


run_local = True


###############################################################################################################
# Usecase: Set yourself the parameters you want to change. Do this by defining a list of dictionaries with the
# corresponding parameter. Again, the Vadere output is deleted after all scenarios run.


if __name__ == "__main__":  # main required by Windows to run in parallel

    ## following code will be in a separate script
    # create sampling for vadere only
    parameter_vadere_only = [
        Parameter("speedDistributionMean", range=[1.3, 1.6]),
        Parameter("maximumSpeed", range=[2.3, 2.6]),
    ]
    par_var_vadere = LatinHyperCubeSampling(parameter_vadere_only).get_dictionary()

    # create sampling for rover
    parameter = [
        Parameter("speedDistributionMean", simulator="vadere", range=[1.3, 1.6]),
        Parameter("maximumSpeed", simulator="vadere", range=[2.3, 2.6]),
        Parameter("omnet", unit="m", simulator="omnet", range=[0.5, 1.5]),
    ]

    par_var = LatinHyperCubeSampling(parameter).get_dictionary(5)

    setup = DictVariation(
        scenario_path=path2scenario,
        parameter_dict_list=par_var,
        qoi="density.txt",
        model=path2model,
        scenario_runs=1,
        post_changes=PostScenarioChangesBase(apply_default=True),
        output_path=None,
        output_folder=None,
        remove_output=False,
    )

    if run_local:
        par_var, data = setup.run(
            -1
        )  # -1 indicates to use all cores available to parallelize the scenarios
    else:
        par_var, data = setup.remote(-1)

    print("\n \n ---------------------------------------\n \n")
    print("ALL USED PARAMETER:")
    print(par_var)

    print("COLLECTED DATA:")
    print(data)
