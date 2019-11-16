#!/usr/bin/env python3

import json
import glob
import subprocess
import time
import os

from shutil import rmtree
from typing import *

from suqc.configuration import SuqcConfig
from suqc.utils.general import user_query_yes_no, get_current_suqc_state, create_folder, str_timestamp


# configuration of the suq-controller
DEFAULT_SUQ_CONFIG = {"default_vadere_src_path": "TODO",   # TODO Feature: compile Vadere before using the jar file
                      "server": {
                          "host": "",
                          "user": "",
                          "port": -1
                      }}


@DeprecationWarning
def get_suq_config():
    assert os.path.exists(SuqcConfig.path_suq_config_file()), "Config file does not exist."

    with open(SuqcConfig.path_suq_config_file(), "r") as f:
        config_file = f.read()
    return json.loads(config_file)


class VadereConsoleWrapper(object):

    # Current log level choices, requires to manually add, if there are changes
    ALLOWED_LOGLVL = ["OFF", "FATAL", "TOPOGRAPHY_ERROR", "TOPOGRAPHY_WARN", "INFO", "DEBUG", "ALL"]

    def __init__(self, model_path: str, loglvl="INFO"):

        self.jar_path = os.path.abspath(model_path)
        if not os.path.exists(self.jar_path):
            raise ValueError(f"Path to jar file {self.jar_path} does not exist.")

        if not self.jar_path.endswith(".jar"):
            raise ValueError(f"Path to file {self.jar_path} is not a jar file.")

        self.loglvl = loglvl.upper()

        if self.loglvl not in self.ALLOWED_LOGLVL:
            raise ValueError(f"parameter 'loglvl={self.loglvl}' not contained in allowed: \n{self.ALLOWED_LOGLVL}")

        if not os.path.exists(self.jar_path):
            raise FileExistsError(f"Vadere console file {self.jar_path} does not exist.")

    def run_simulation(self, scenario_filepath, output_path):
        start = time.time()
        return_code = subprocess.call(["java",
                                       "-jar", self.jar_path,
                                       "--loglevel", self.loglvl,
                                       "suq", "-f", scenario_filepath, "-o", output_path])
        return return_code, time.time() - start

    def run_floorfield_cache(self, scenario_filepath, output_path):
        start = time.time()
        return_code = subprocess.call(["java",
                                       "-jar", self.jar_path, "--loglevel", self.loglvl,
                                       "utils", "-i", scenario_filepath, "-o", output_path, "-m", "binCache"])
        return return_code, time.time() - start

    def run_cache_hash(self, scenario_filepath):
        start = time.time()
        return_code = subprocess.call(["java",
                                       "-jar", self.jar_path, "--loglevel", self.loglvl,
                                       "utils", "-i", scenario_filepath, "-m", "getHash"])
        return return_code, time.time() - start

    @classmethod
    def from_default_models(cls, model):
        if not model.endswith(".jar"):
            model = ".".join([model, "jar"])
        return cls(os.path.join(SuqcConfig.path_models_folder(), model))

    @classmethod
    def from_model_path(cls, model_path):
        return cls(model_path)

    @classmethod
    def from_new_compiled_package(cls, src_path=None):
        pass  # TODO: use default src_path

    @classmethod
    def infer_model(cls, model):
        if isinstance(model, str):
            if os.path.exists(model):
                return VadereConsoleWrapper.from_model_path(os.path.abspath(model))
            else:
                return VadereConsoleWrapper.from_default_models(model)
        elif isinstance(model, VadereConsoleWrapper):
            return model
        else:
            raise ValueError("Failed to infer Vadere model.")


class EnvironmentManager(object):

    vadere_output_folder = "vadere_output"
    vadere_cache_path = os.path.join(vadere_output_folder, "__cache__")

    def __init__(self, base_path, env_name: str):

        self.base_path, self.env_name = self.handle_path_and_env_input(base_path, env_name)

        self.env_name = env_name
        self.env_path = self.environment_folder_path(self.base_path, self.env_name)

        # output is usually of the following format:
        # 000001_000002 for variation 1 and run_id 2 (default 6 digits)
        # change these variables externally, if less digits are required to have shorter paths
        self.nr_digits_variation = 6
        self.nr_digits_runs = 6

        print(f"INFO: Set environment path to {self.env_path}")
        if not os.path.exists(self.env_path):
            raise FileNotFoundError(f"Environment {self.env_path} does not exist. Use function "
                                    f"'EnvironmentManager.create_new_environment'")
        self._scenario_basis = None

    @property
    def basis_scenario(self):
        if self._scenario_basis is None:
            path_basis_scenario = self.path_basis_scenario

            with open(path_basis_scenario, "r") as f:
                basis_file = json.load(f)
            self._scenario_basis = basis_file

        return self._scenario_basis

    @property
    def path_basis_scenario(self):
        sc_files = glob.glob(os.path.join(self.env_path, "*.scenario"))
        assert len(sc_files) == 1, "None or too many .scenario files found in environment."
        return sc_files[0]

    @classmethod
    def from_full_path(cls, env_path):
        assert os.path.isdir(env_path)
        base_path = os.path.dirname(env_path)

        if env_path.endswith(os.pathsep):
            env_path = env_path.rstrip(os.path.sep)
        env_name = os.path.basename(env_path)

        cls(base_path=base_path, env_name=env_name)

    @classmethod
    @DeprecationWarning
    def create_variation_if_not_exist(cls, basis_scenario: Union[str, dict], base_path=None, env_name=None):
        target_path = cls.environment_folder_path(base_path, env_name)
        if os.path.exists(target_path):
            existing = cls(base_path, env_name)

            # TODO: maybe it is good to compare if the handled file is the same as the existing
            # exist_basis_file = existing.get_vadere_scenario_basis_file()
            return existing
        else:
            return cls.create_variation_env(basis_scenario=basis_scenario, base_path=base_path, env_name=env_name)

    @classmethod
    def create_new_environment(cls, base_path=None, env_name=None, handle_existing="ask_user_replace"):

        output_folder_path = cls.environment_folder_path(base_path, env_name)

        # TODO: Refactor, make 'handle_existing' an Enum
        assert handle_existing in ["ask_user_replace", "force_replace", "write_in_if_exist_else_create", "write_in"]

        abort = False
        env_man = None

        env_exists = os.path.exists(output_folder_path)

        if handle_existing == "ask_user_replace" and env_exists:
            if not cls.remove_environment(base_path, env_name):
                abort = True
        elif handle_existing == "force_replace" and env_exists:
            if env_exists:
                cls.remove_environment(base_path, env_name, force=True)
        elif handle_existing == "write_in":
            assert env_exists, f"base_path={base_path} env_name={env_name} does not exist"
            env_man = cls(base_path=base_path, env_name=env_name)
        elif handle_existing == "write_in_if_exist_else_create":
            if env_exists:
                env_man = cls(base_path=base_path, env_name=env_name)

        if abort:
            raise ValueError("Could not create new environment.")

        if env_man is None:
            # Create new environment folder
            os.mkdir(output_folder_path)
            env_man = cls(base_path=base_path, env_name=env_name)

        return env_man

    @classmethod
    def create_variation_env(cls, basis_scenario: Union[str, dict], base_path=None, env_name=None,
                             handle_existing="ask_user_replace"):

        # Check if environment already exists
        env_man = cls.create_new_environment(base_path=base_path, env_name=env_name, handle_existing=handle_existing)
        path_output_folder = env_man.env_path

        # Add basis scenario used for the variation (i.e. sampling!)
        ##################

        if isinstance(basis_scenario, str):  # assume that this is a path
            assert os.path.isfile(basis_scenario), "Filepath to .scenario does not exist"
            assert basis_scenario.split(".")[-1] == "scenario", "File has to be a Vadere '*.scenario' file"

            with open(basis_scenario, "r") as file:
                basis_scenario = file.read()

        basis_fp = os.path.join(path_output_folder, f"BASIS_{env_name}.scenario")

        # FILL IN THE STANDARD FILES IN THE NEW SCENARIO:
        with open(basis_fp, "w") as file:
            if isinstance(basis_scenario, dict):
                json.dump(basis_scenario, file, indent=4)
            else:
                file.write(basis_scenario)

        # Create and store the configuration file to the new folder
        cfg = dict()

        if not SuqcConfig.is_package_paths():  # TODO it may be good to write the git hash / version number in the file
            cfg["suqc_state"] = get_current_suqc_state()

            with open(os.path.join(path_output_folder, "suqc_commit_hash.json"), 'w') as outfile:
                s = "\n".join(["commit hash at creation", cfg["suqc_state"]["git_hash"]])
                outfile.write(s)

        # Create the folder where all output is stored
        os.mkdir(os.path.join(path_output_folder, EnvironmentManager.vadere_output_folder))

        return cls(base_path, env_name)

    @classmethod
    def remove_environment(cls, base_path, name, force=False):
        target_path = cls.environment_folder_path(base_path, name)

        if force or user_query_yes_no(question=f"Are you sure you want to remove the current environment? Path: \n "
        f"{target_path}"):
            try:
                rmtree(target_path)
            except FileNotFoundError:
                print(f"INFO: Tried to remove environment {name}, but did not exist.")
            return True
        return False

    @staticmethod
    def handle_path_and_env_input(base_path, env_name):
        if env_name is None:
            env_name = "_".join(["output", str_timestamp()])

        if base_path is None:
            base_path = SuqcConfig.path_container_folder()

        return base_path, env_name

    @staticmethod
    def environment_folder_path(base_path, env_name):
        base_path, env_name = EnvironmentManager.handle_path_and_env_input(base_path, env_name)
        assert os.path.isdir(base_path)
        output_folder_path = os.path.join(base_path, env_name)
        return output_folder_path

    def vadere_result_folder_path(self):
        rel_path = os.path.join(self.env_path, EnvironmentManager.vadere_output_folder)
        return os.path.abspath(rel_path)

    def single_run_output_folder(self, parameter_id, run_id):
        scenario_filename = self._scenario_variation_filename(parameter_id=parameter_id, run_id=run_id)
        scenario_filename = scenario_filename.replace(".scenario", "")
        return os.path.join(self.vadere_result_folder_path(), "".join([scenario_filename, "_output"]))

    def _scenario_variation_filename(self, parameter_id, run_id):
        digits_parameter_id = str(parameter_id).zfill(self.nr_digits_variation)
        digits_run_id = str(run_id).zfill(self.nr_digits_variation)
        numbered_scenario_name = "_".join([digits_parameter_id, digits_run_id])

        return "".join([numbered_scenario_name, ".scenario"])

    def scenario_variation_path(self, par_id, run_id):
        return os.path.join(self.vadere_result_folder_path(), self._scenario_variation_filename(par_id, run_id))

    def save_scenario_variation(self, par_id, run_id, content):
        scenario_path = self.scenario_variation_path(par_id, run_id)
        assert not os.path.exists(scenario_path), f"File {scenario_path} already exists!"

        with open(scenario_path, "w") as outfile:
            json.dump(content, outfile, indent=4)
        return scenario_path