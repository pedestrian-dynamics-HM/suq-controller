#!/usr/bin/env python3

import sys

from tutorial.imports import *

# This is just to make sure that the systems path is set up correctly, to have correct imports.
# (It can be ignored)
sys.path.append(os.path.abspath("."))   # in case tutorial is called from the root directory
sys.path.append(os.path.abspath(".."))  # in tutorial directly

run_local = True

###############################################################################################################
# Usecase: One parameter in the scenario is changed, for every independent the data is collected and returned.
# The Vadere output is deleted after all scenarios run.

# Example where the values of 'speedDistributionMean' are set between 0.1 and 1.5 in 5 equidistant points

if __name__ == "__main__":  # main required by Windows to run in parallel

    setup = SingleKeyVariation(scenario_path=path2scenario,  # path to the Vadere .scenario file to vary
                               key="speedDistributionMean",  # parameter key to change
                               values=np.linspace(0.7, 1.5, 3),  # values to set for the parameter
                               qoi="density.txt",  # output file name to collect
                               model=path2model,  # path to Vadere console jar file to use for simulation
                               scenario_runs=1,  # specify how often each scenario should run
                               # post changes can be used to apply changes to the scenario that are not part of the
                               # sampling -- especially random seed setting. It is easy to define user based changes.
                               post_changes=PostScenarioChangesBase(apply_default=True),
                               output_path=os.path.abspath("."),  # specify the path, where the results are written
                               output_folder="testfolder",  # specify the folder in the 'output_path'
                               remove_output=False  # flag whether to remove the output_folder or not
                               )

    if run_local:
        par_var, data = setup.run(njobs=-1)
    else:
        par_var, data = setup.remote(njobs=-1)

    print("---------------------------------------\n \n")
    print("ALL USED PARAMETER:")
    print(par_var)

    print("COLLECTED DATA:")
    print(data)
