#!/usr/bin/env python3

# TODO: """ << INCLUDE DOCSTRING (one-line or multi-line) >> """

import pickle
import os
import pandas as pd

from fabric import Connection

from suqc.parameter import ParameterVariation
from suqc.qoi import QoIProcessor
from suqc.configuration import EnvironmentManager

# --------------------------------------------------
# people who contributed code
__authors__ = "Daniel Lehmberg"
# people who made suggestions or reported bugs but didn't contribute code
__credits__ = ["n/a"]
# --------------------------------------------------


class SimulationDefinition(object):

    def __init__(self, env_man: EnvironmentManager, par_var: ParameterVariation, qoi: QoIProcessor):
        self.name = env_man.name
        self.basis_file = env_man.get_vadere_scenario_basis_file()
        self.model = env_man.get_cfg_value("model")
        self.par_var = par_var
        self.qoi = qoi


class ServerConnection(object):

    READ_VERSION = "python3 -c 'import suqc; print(suqc.__version__)'"

    def __init__(self):
        self._con = None

    @property
    def con(self):
        if self._con is None:
            raise RuntimeError("Server not initialized")
        return self._con

    def __enter__(self):
        self._connect_server()
        return self

    def __exit__(self, type, value, traceback):
        self.con.close()
        print("INFO: Server connection closed")

    def _connect_server(self):
        self._con: Connection = Connection("minimuc.cs.hm.edu", user="dlehmberg", port=5022)
        version = self.read_terminal_stdout(ServerConnection.READ_VERSION)
        print(f"INFO: Connection established. Detected suqc version {version} on server side.")

    def read_terminal_stdout(self, s: str) -> str:
        r = self._con.run(s)
        return r.stdout.rstrip()  # rstrip -> remove trailing whitespaces and new lines

    def send_file2server(self, local_fp, server_fp):
        self._con.put(local_fp, server_fp)



class ServerSimulation(object):

    FILENAME_PICKLE_RESULTS = "results.p"

    def __init__(self, server: ServerConnection):
        self._server = server

    @classmethod
    def _create_remote_environment(cls, fp):
        from suqc.configuration import create_environment
        simdef = pickle.load(fp)
        create_environment(simdef.name, simdef.basis_file, simdef.model, replace=True)

    @classmethod
    def _remote_run_env(cls, fp):
        import suqc.query
        import suqc.configuration
        simdef = pickle.load(fp)

        env_man = suqc.configuration.EnvironmentManager(simdef.name)
        ret = suqc.query.Query(env_man, simdef.qoi).query_simulate_all_new(simdef.par_var, njobs=-1)
        path = os.path.join(env_man.env_path, ServerSimulation.FILENAME_PICKLE_RESULTS)
        ret.to_pickle(path)
        print(path)  # Is read from console (from local). This allows to transfer the file back.

    @classmethod
    def remote_simulate(cls, fp):
        ServerSimulation._create_remote_environment(fp)
        ServerSimulation._remote_run_env(fp)

    def _setup_server_env(self, local_pickle_path, simdef):
        with open(local_pickle_path, "wb") as f:
            pickle.dump(simdef, f)

        rem_con_path = self._server.read_terminal_stdout(
            "python3 -c 'import suqc.configuration as c; print(c.get_container_path())'")
        rem_env_path = os.path.join(rem_con_path, simdef.name)
        self._server.send_file2server("INVALID", rem_env_path)

        return rem_env_path

    def _local_submit_request(self, fp):
        s = f"""python3 -c 'import suqc.rem; rem.ServerProcedure.remove_simulate({fp})' """
        result = self._server.con.run(s)
        last_line = result.stdout.rstrip().split("\n")[-1]  # last line to get the last 'print(path)' statement
        return last_line

    def _read_results(self, local_path, remote_path):
        self._server.con.get(remote_path, local_path)

        with open(local_path, "rb") as f:
            df = pickle.load(f)
            isinstance(df, pd.DataFrame)
        return df

    def run(self, env_man: EnvironmentManager, par_var: ParameterVariation, qoi: QoIProcessor):
        simdef = SimulationDefinition(env_man, par_var, qoi)

        local_pickle_path = os.path.join(env_man.env_path, "simdef.p")
        fp_env = self._setup_server_env(local_pickle_path=local_pickle_path, simdef=simdef)
        fp_rem_results = self._local_submit_request(fp_env)
        fp_loc_results = os.path.join(env_man.env_path, ServerSimulation.FILENAME_PICKLE_RESULTS)
        results = self._read_results(fp_loc_results, fp_rem_results)
        return results


if __name__ == "__main__":

    from suqc.qoi import PedestrianEvacuationTimeProcessor

    env_man = EnvironmentManager("corner")
    par_var = ParameterVariation(env_man)
    par_var.add_dict_grid({"speedDistributionStandardDeviation": [0.0, 0.1, 0.2, 0.3]})
    qoi = PedestrianEvacuationTimeProcessor(env_man)

    with ServerConnection() as sc:
        server_sim = ServerSimulation(sc)
        result = server_sim.run(env_man, par_var, qoi)

    print(result)
