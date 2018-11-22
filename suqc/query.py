#!/usr/bin/env python3

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

import subprocess
import multiprocessing

from suqc.qoi import *
from suqc.configuration import EnvironmentManager
from suqc.parameter.sampling import ParameterVariation, FullGridSampling, RandomSampling, BoxSamplingUlamMethod
from suqc.parameter.postchanges import ScenarioChanges
from suqc.parameter.create import VadereScenarioCreation
from suqc.resultformat import ParameterResult

from typing import List, Union

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


class Query(object):

    def __init__(self, env_man: EnvironmentManager, par_var: ParameterVariation, qoi: QoIProcessor,
                 sc_change: ScenarioChanges=None):

        self._env_man = env_man
        self._par_var = par_var
        self._qoi = qoi
        self._sc_change = sc_change
        self._result_df = None

    @property
    def result(self):
        return self._result_df

    def _run_vadere_simulation(self, scenario_fp, output_path):
        model_path = self._env_man.get_model_path()

        loglvl = "OFF"  # TODO: possibly parametrize...
        subprocess.call(["java", "-jar", model_path, "--loglevel", loglvl, "suq", "-f", scenario_fp, "-o", output_path])

    def _single_query(self, kwargs):
        output_path = self._env_man.get_output_path(kwargs["par_id"], create=True)
        self._run_vadere_simulation(kwargs["scenario_path"], output_path)
        result = self._qoi.read_and_extract_qoi(kwargs["par_id"], output_path)
        return result  # because of the multi-processor, don't try to already add the results here to _results_df

    def _sp_query(self, query_list):
        # enumerate returns tuple(par_id, scenario filepath) see ParameterVariation.generate_vadere_scenarios and
        # ParameterVariation._vars_object()
        results = list()
        for arg in query_list:
            results.append(self._single_query(arg))
        self._result_df = ParameterResult.concat_parameter_results(results)

    def _mp_query(self, query_list, njobs):
        pool = multiprocessing.Pool(processes=njobs)
        results = pool.map(self._single_query, query_list)
        self._result_df = ParameterResult.concat_parameter_results(results)

    def run(self, njobs: int=-1):

        assert njobs != 0, "njobs cannot be zero"

        nr_simulations = self._par_var.points.shape[0]  # nr of rows = nr of parameter settings = #simulations

        if njobs == -1:
            avail_cpus = multiprocessing.cpu_count()
            njobs = min(avail_cpus, nr_simulations)
            print(f"INFO: Available cpus: {avail_cpus}. Nr. of simulation {nr_simulations}. "
                  f"Setting to {njobs} processes.")

        vadcreate = VadereScenarioCreation(self._env_man, self._par_var, self._sc_change)
        query_list = vadcreate.generate_vadere_scenarios(njobs)

        if njobs == 1:
            self._sp_query(query_list=query_list)
        else:
            self._mp_query(query_list=query_list, njobs=njobs)
        return self._par_var.points, self.result


if __name__ == "__main__":

    # from suqc.qoi import *
    # em = EnvironmentManager("two_density")
    #
    # par = FullGridSampling()
    # par.add_dict_grid({})
    #
    # q = AreaDensityVoronoiProcessor(em)

    em = EnvironmentManager("two_density")
    pv = FullGridSampling()
    pv.add_dict_grid({"sources.[id==1].spawnNumber": [10, 20]})
    q0 = CombinedTwoDensityProcessor(em=em)

    one, two = Query(em, pv, q0).run(njobs=1)
    print(two)
    exit()

    em = EnvironmentManager("corner")
    pv = FullGridSampling()
    pv.add_dict_grid({"speedDistributionStandardDeviation": [0.0, 0.1, 0.2, 0.3], "speedDistributionMean": [1.2, 1.3]})
    q0 = CountInArea(em, p1=[0, 14], p2=[13, 24])

    print(Query(em, pv, q0).run(njobs=-1))
    exit()

    #q1 = PedestrianDensityGaussianProcessor(em) # TODO: need to check if qoi-processor is available in basis file!


    pv = BoxSamplingUlamMethod()
    pv.create_grid("speedDistributionStandardDeviation", 0, 0.5, 3, 3)

    sc = ScenarioChanges(apply_default=True)

    q = Query(em, pv, q0, sc).run(njobs=-1)
    print(q)

    # q = Query(em, pv, AreaDensityVoronoiProcessor(em)).run(njobs=1)
    # print(q)
