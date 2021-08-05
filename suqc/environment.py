#!/usr/bin/env python3
import glob
import json
import os
import platform
import shutil
import subprocess
import time
from shutil import copytree, rmtree
from typing import *
import xml.etree.ElementTree as et
import re

import pandas as pd

from suqc.requestitem import RequestItem
from suqc.configuration import SuqcConfig
from omnetinireader.config_parser import OppConfigFileBase, OppConfigType, OppParser
from suqc.utils.general import (
    get_current_suqc_state,
    str_timestamp,
    user_query_yes_no,
    include_patterns,
    removeEmptyFolders,
    get_info_vadere_repo, check_simulator,
)

# configuration of the suq-controller
DEFAULT_SUQ_CONFIG = {
    "default_vadere_src_path": "TODO",
    "server": {"host": "", "user": "", "port": -1},
}


class VadereJarFile(object):
    def __init__(self, path, git_commit_hash, branch="master"):
        self.git_commit_hash = git_commit_hash
        self.branch = branch
        self.path = path

    def get_path(self):
        jar_file_name = f"vadere-console_{self.branch}_{self.git_commit_hash}.jar"
        return os.path.join(self.path, jar_file_name)

    def get_jar_file_from_vadere_repo(self):
        # checkout commit and make a package
        # copy the generated jar-file to the user-defined path

        print(f"Branch: {self.branch}, commit hash: {self.git_commit_hash}")

        if platform.system() != "Linux":
            raise NotImplemented

        get_info_vadere_repo()

        vadere = os.environ["VADERE"]
        vadere_jar_file = self.get_path()
        vadere_commit = self.git_commit_hash
        if os.path.exists(vadere_jar_file) is False:

            return_code = subprocess.check_call(
                ["git", "-C", vadere, "pull", "origin", self.branch]
            )
            if vadere_commit:
                return_code = subprocess.check_call(
                    ["git", "-C", vadere, "checkout", vadere_commit]
                )

            command = ["mvn", "clean", "-f", os.path.join(vadere, "pom.xml")]
            return_code = subprocess.check_call(command)
            command = [
                "mvn",
                "package",
                "-f",
                os.path.join(vadere, "pom.xml"),
                "-Dmaven.test.skip=true",
            ]
            return_code = subprocess.check_call(command)

            jar_file = os.path.join(vadere, "VadereSimulator/target/vadere-console.jar")

            shutil.copyfile(jar_file, vadere_jar_file)


class AbstractConsoleWrapper(object):
    @classmethod
    def infer_model(cls, model) -> "AbstractConsoleWrapper":

        if isinstance(model, str):
            if model == "Coupled":
                return CoupledConsoleWrapper(model)
            else:
                return VadereConsoleWrapper.infer_model(model)
        elif isinstance(model, VadereConsoleWrapper) or \
                isinstance(model, CoupledConsoleWrapper) or \
                isinstance(model, CrownetSumoWrapper):
            return model
        else:
            raise ValueError(
                f"Model must be of type string or VadereConsoleWrapper or CoupledConsoleWrapper. Got type {type(model)}."
            )


class CoupledConsoleWrapper(AbstractConsoleWrapper):
    def __init__(self, model, vadere_tag="latest", omnetpp_tag="lastest", additional_settings = None):
        self.simulator = model
        self.vadere_tag = vadere_tag
        self.omnetpp_tag = omnetpp_tag
        self.add_settings = None
        if additional_settings is not None:
            self.set_additional_arguements(additional_settings)

    def set_additional_arguements(self, add_settings):
        if isinstance( add_settings, str):
            add_settings = list(add_settings)

        for i in add_settings:
            if isinstance(i, str) is False:
                raise ValueError("Please provide a string or list of strings.")
        self.add_settings = add_settings


    def run_simulation(
        self, dirname, start_file, required_files: Union[str, List[str]]
    ):

        terminal_command = ["python3", start_file, "--qoi"]
        terminal_command.extend(required_files)
        terminal_command.extend(["--run-name", os.path.basename(dirname)])
        terminal_command.extend(["--create-vadere-container"])
        terminal_command.extend(["--delete-existing-containers"])
        terminal_command.extend(["--vadere-tag", self.vadere_tag])
        terminal_command.extend(["--omnet-tag", self.omnetpp_tag])

        if self.add_settings is not None:
            terminal_command.extend(self.add_settings)

        time_started = time.time()
        t = time.strftime("%H:%M:%S", time.localtime(time_started))
        print(f"{t}\t Call {os.path.basename(dirname)}/{start_file} ")

        return_code = subprocess.check_call(
            terminal_command,
            env=os.environ,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=dirname,
            timeout=15000,  # stop simulation after 15000s
        )

        process_duration = time.time() - time_started
        output_subprocess = None

        return return_code, process_duration, output_subprocess


class VadereConsoleWrapper(AbstractConsoleWrapper):

    # Current log level choices, requires to manually add, if there are changes in Vadere
    ALLOWED_LOGLVL = [
        "OFF",
        "FATAL",
        "TOPOGRAPHY_ERROR",
        "TOPOGRAPHY_WARN",
        "INFO",
        "DEBUG",
        "ALL",
    ]

    def __init__(
        self,
        model_path: Union[str, VadereJarFile],
        loglvl="INFO",
        jvm_flags: Optional[List] = None,
        timeout_sec=None,
    ):

        if not isinstance(model_path, str):
            if not os.path.exists(model_path.get_path()):
                model_path.get_jar_file_from_vadere_repo()

            model_path = model_path.get_path()

        self.jar_path = os.path.abspath(model_path)

        if not os.path.exists(self.jar_path):
            raise FileNotFoundError(
                f"Vadere console .jar file {self.jar_path} does not exist."
            )

        loglvl = loglvl.upper()
        if loglvl not in self.ALLOWED_LOGLVL:
            raise ValueError(
                f"set loglvl={loglvl} not contained "
                f"in allowed: {self.ALLOWED_LOGLVL}"
            )

        if jvm_flags is not None and not isinstance(jvm_flags, list):
            raise TypeError(
                f"jvm_flags are required to be a list. Got: {type(jvm_flags)}"
            )

        if timeout_sec is None:
            pass  # do nothing, no timeout
        elif not isinstance(timeout_sec, int) or timeout_sec <= 0:
            raise TypeError(
                "vadere_run_timeout_sec must be of type int and positive " "value"
            )

        self.loglvl = loglvl
        # Additional Java Virtual Machine options / flags
        self.jvm_flags = jvm_flags if jvm_flags is not None else []
        self.timeout_sec = timeout_sec

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
            raise ValueError(f"Failed to infer Vadere model. \n {model}")

    def run_simulation(self, scenario_fp, output_path):
        start = time.time()

        subprocess_cmd = ["java"]
        subprocess_cmd += self.jvm_flags
        subprocess_cmd += ["-jar", self.jar_path]
        # Vadere console commands
        subprocess_cmd += ["--loglevel", self.loglvl]
        subprocess_cmd += ["suq", "-f", scenario_fp, "-o", output_path]

        output_subprocess = dict()

        try:
            subprocess.check_output(
                subprocess_cmd, timeout=self.timeout_sec, stderr=subprocess.PIPE
            )
            process_duration = time.time() - start

            # if return_code != 0 a subprocess.CalledProcessError is raised
            return_code = 0
            output_subprocess = None
        except subprocess.TimeoutExpired as exception:
            return_code = 1
            process_duration = self.timeout_sec
            output_subprocess["stdout"] = exception.stdout
            output_subprocess["stderr"] = None
        except subprocess.CalledProcessError as exception:
            return_code = exception.returncode
            process_duration = time.time() - start
            output_subprocess["stdout"] = exception.stdout
            output_subprocess["stderr"] = exception.stderr

        return return_code, process_duration, output_subprocess

    @classmethod
    def from_default_models(cls, model):
        if not model.endswith(".jar"):
            model = ".".join([model, "jar"])
        return cls(os.path.join(SuqcConfig.path_models_folder(), model))

    @classmethod
    def from_model_path(cls, model_path):
        return cls(model_path)


class AbstractEnvironmentManager(object):
    VADERE_SCENARIO_FILE_TYPE = ".scenario"

    def __init__(self, base_path, env_name: str):

        self.base_path, self.env_name = self.handle_path_and_env_input(
            base_path, env_name
        )

        self.env_name = env_name
        self.env_path = self.output_folder_path(self.base_path, self.env_name)

        # output is usually of the following format:
        # 000001_000002 for variation 1 and run_id 2
        # Change these attributes externally, if less digits are required to have
        # shorter/longer paths.
        self.nr_digits_variation = 6
        self.nr_digits_runs = 6

        print(f"INFO: Set environment path to {self.env_path}")
        if not os.path.exists(self.env_path):
            raise FileNotFoundError(
                f"Environment {self.env_path} does not exist. Use function "
                f"'EnvironmentManager.create_new_environment'"
            )
        self._vadere_scenario_basis = None

    @property
    def vadere_basis_scenario(self):
        if self._vadere_scenario_basis is None:
            path_basis_scenario = self.vadere_path_basis_scenario

            with open(path_basis_scenario, "r") as f:
                basis_file = json.load(f)
            self._vadere_scenario_basis = basis_file

        return self._vadere_scenario_basis

    @property
    def vadere_path_basis_scenario(self):
        sc_files = glob.glob(
            os.path.join(self.env_path, f"*{self.VADERE_SCENARIO_FILE_TYPE}")
        )

        if len(sc_files) != 1:
            raise RuntimeError(
                f"None or too many '{self.VADERE_SCENARIO_FILE_TYPE}' files "
                "found in environment."
            )
        return sc_files[0]

    @classmethod
    def create_variation_env_from_info_file(cls, path_info_file):
        raise NotImplemented

    @classmethod
    def create_new_environment(
        cls, base_path=None, env_name=None, handle_existing="ask_user_replace"
    ):

        base_path, env_name = cls.handle_path_and_env_input(base_path, env_name)

        # TODO: Refactor, make handle_existing an Enum
        assert handle_existing in [
            "ask_user_replace",
            "force_replace",
            "write_in_if_exist_else_create",
            "write_in",
        ]

        # set to True if env already exists, and it shouldn't be overwritten
        about_creating_env = False
        env_man = None

        env_exists = os.path.exists(cls.output_folder_path(base_path, env_name))

        if handle_existing == "ask_user_replace" and env_exists:
            if not cls.remove_environment(base_path, env_name):
                about_creating_env = True
        elif handle_existing == "force_replace" and env_exists:
            if env_exists:
                cls.remove_environment(base_path, env_name, force=True)
        elif handle_existing == "write_in":
            assert (
                env_exists
            ), f"base_path={base_path} env_name={env_name} does not exist"
            env_man = cls(base_path=base_path, env_name=env_name)
        elif handle_existing == "write_in_if_exist_else_create":
            if env_exists:
                env_man = cls(base_path=base_path, env_name=env_name)

        if about_creating_env:
            raise ValueError("Could not create new environment.")

        if env_man is None:
            # Create new environment folder
            os.mkdir(cls.output_folder_path(base_path, env_name))
            env_man = cls(base_path=base_path, env_name=env_name)

        return env_man

    @classmethod
    def remove_environment(cls, base_path, name, force=False):
        target_path = cls.output_folder_path(base_path, name)

        if force or user_query_yes_no(
            question=f"Are you sure you want to remove the current environment? Path: \n "
            f"{target_path}"
        ):
            try:
                rmtree(target_path)
            except FileNotFoundError:
                print(f"INFO: Tried to remove environment {name}, but did not exist.")
            return True
        return False

    @classmethod
    def from_full_path(cls, env_path):
        assert os.path.isdir(env_path)
        base_path = os.path.dirname(env_path)

        if env_path.endswith(os.pathsep):
            env_path = env_path.rstrip(os.path.sep)
        env_name = os.path.basename(env_path)

        cls(base_path=base_path, env_name=env_name)

    @staticmethod
    def handle_path_and_env_input(base_path, env_name):
        if env_name is None:
            env_name = "_".join(["output", str_timestamp()])

        if base_path is None:
            base_path = SuqcConfig.path_container_folder()

        return base_path, env_name

    @staticmethod
    def output_folder_path(base_path, env_name):
        base_path, env_name = AbstractEnvironmentManager.handle_path_and_env_input(
            base_path, env_name
        )
        assert os.path.isdir(base_path)
        output_folder_path = os.path.join(base_path, env_name)
        return output_folder_path

    def scenario_variation_path(self, par_id, run_id):
        return os.path.join(
            self.get_env_outputfolder_path(),
            self._scenario_variation_filename(par_id, run_id),
        )

    def save_scenario_variation(self, par_id, run_id, content):
        scenario_path = self.scenario_variation_path(par_id, run_id)
        assert not os.path.exists(
            scenario_path
        ), f"File {scenario_path} already exists!"

        with open(scenario_path, "w") as outfile:
            json.dump(content, outfile, indent=4)
        return scenario_path

    def get_temp_folder(self):
        raise NotImplemented

    def get_env_outputfolder_path(self):
        raise NotImplemented

    def get_variation_output_folder(self, parameter_id, run_id):
        scenario_filename = self._scenario_variation_filename(
            parameter_id=parameter_id, run_id=run_id
        )
        scenario_filename = scenario_filename.replace(
            self.VADERE_SCENARIO_FILE_TYPE, ""
        )
        return os.path.join(
            self.get_env_outputfolder_path(), "".join([scenario_filename, "_output"])
        )

    def _scenario_variation_filename(self, parameter_id, run_id):
        digits_parameter_id = str(parameter_id).zfill(self.nr_digits_variation)
        digits_run_id = str(run_id).zfill(self.nr_digits_variation)
        numbered_scenario_name = "_".join([digits_parameter_id, digits_run_id])

        return "".join([numbered_scenario_name, self.VADERE_SCENARIO_FILE_TYPE])

    def get_env_info(self):
        return self.env_info_df

    @classmethod
    def set_env_info(cls, basis_scenario, base_path, env_name, ini_scenario):

        info = {
            "basis_scenario": basis_scenario,
            "ini_path": ini_scenario,
            "base_path": base_path,
            "env_name": env_name,
        }

        info = pd.DataFrame(data=info, index=[0])
        cls.env_info_df = info

    def write_parameter_info(self, parameter):
        file_path = os.path.join(self.get_temp_folder(), "parameter.pkl")
        parameter.to_pickle(file_path)


class VadereEnvironmentManager(AbstractEnvironmentManager):

    PREFIX_BASIS_SCENARIO = "BASIS_"
    VADERE_SCENARIO_FILE_TYPE = ".scenario"
    simulation_runs_output_folder = "vadere_output"

    def __init__(self, base_path, env_name: str):
        super().__init__(base_path, env_name)

    @classmethod
    def create_variation_env(
        cls,
        basis_scenario: Union[str, dict],
        base_path=None,
        env_name=None,
        handle_existing="ask_user_replace",
    ):

        cls.set_env_info(
            basis_scenario=basis_scenario,
            base_path=base_path,
            env_name=env_name,
            ini_scenario="",
        )

        # Check if environment already exists
        env_man = cls.create_new_environment(
            base_path=base_path, env_name=env_name, handle_existing=handle_existing
        )
        path_output_folder = env_man.env_path

        # Add basis scenario used for the variation (i.e. sampling)
        if isinstance(basis_scenario, str):  # assume that this is a path
            if not os.path.isfile(basis_scenario):
                raise FileExistsError("Filepath to .scenario does not exist")
            elif basis_scenario.split(".")[-1] != cls.VADERE_SCENARIO_FILE_TYPE[1:]:
                raise ValueError(
                    "basis_scenario has to be a Vadere '*"
                    f"{cls.VADERE_SCENARIO_FILE_TYPE}' file"
                )

            with open(basis_scenario, "r") as file:
                basis_scenario = file.read()

        # add prefix to scenario file:
        basis_fp = os.path.join(
            path_output_folder, f"{cls.PREFIX_BASIS_SCENARIO}{env_name}.scenario"
        )

        # FILL IN THE STANDARD FILES IN THE NEW SCENARIO:
        with open(basis_fp, "w") as file:
            if isinstance(basis_scenario, dict):
                json.dump(basis_scenario, file, indent=4)
            else:
                file.write(basis_scenario)

        # Create and store the configuration file to the new folder
        cfg = dict()

        # TODO it may be good to write the git hash / version number in the file
        if not SuqcConfig.is_package_paths():
            cfg["suqc_state"] = get_current_suqc_state()

            with open(
                os.path.join(path_output_folder, "suqc_commit_hash.json"), "w"
            ) as outfile:
                s = "\n".join(
                    ["commit hash at creation", cfg["suqc_state"]["git_hash"]]
                )
                outfile.write(s)

        # Create the folder where all output is stored
        os.mkdir(
            os.path.join(
                path_output_folder,
                VadereEnvironmentManager.simulation_runs_output_folder,
            )
        )

        return cls(base_path, env_name)

    def get_env_outputfolder_path(self):
        rel_path = os.path.join(
            self.env_path, VadereEnvironmentManager.simulation_runs_output_folder
        )
        return os.path.abspath(rel_path)


class CoupledEnvironmentManager(AbstractEnvironmentManager):

    PREFIX_BASIS_SCENARIO = ""
    simulation_runs_output_folder = "simulation_runs"
    simulation_runs_single_folder_name = "Sample_"
    run_file = "run_script.py"
    temp_folder_rover = "temp"
    simulators = {
        "vadere": {
            "subdir": "vadere/scenarios",
        },
        "omnet": {
            "subdir": "",
        },
        "sumo": {
            "subdir": "sumo",
        }
    }

    def __init__(self, base_path, env_name: str):
        super().__init__(base_path, env_name)
        self._omnet_ini_basis = None

    @property
    def omnet_basis_ini(self):
        if self._omnet_ini_basis is None:
            path_basis_ini = self.omnet_path_ini
            ini_file = OppConfigFileBase.from_path(
                ini_path=path_basis_ini,
                config="final",
                cfg_type=OppConfigType.EXT_DEL_LOCAL,
            )
            self._omnet_ini_basis = ini_file
        return self._omnet_ini_basis

    @property
    def omnet_path_ini(self):
        sc_files = glob.glob(os.path.join(self.env_path, "*ini"))

        if len(sc_files) != 1:
            raise RuntimeError(f"None or too many 'ini' files " "found in environment.")
        return sc_files[0]

    @classmethod
    def create_variation_env_from_info_file(cls, path_info_file):

        d = pd.read_pickle(path_info_file)

        env = cls.create_variation_env(
            basis_scenario=d["basis_scenario"].values[0],
            ini_scenario=d["ini_path"].values[0],
            base_path=d["base_path"].values[0],
            env_name=d["env_name"].values[0],
            handle_existing="write_in",
        )

        return env

    @classmethod
    def create_variation_env(
        cls,
        basis_scenario: Union[str, dict],
        ini_scenario: Union[str, dict],
        base_path=None,
        env_name=None,
        handle_existing="ask_user_replace",
    ):

        cls.set_env_info(
            basis_scenario=basis_scenario,
            base_path=base_path,
            env_name=env_name,
            ini_scenario=ini_scenario,
        )

        cls.basis_scenario_name = os.path.basename(basis_scenario)
        # Check if environment already exists

        env_man = cls.create_new_environment(
            base_path=base_path, env_name=env_name, handle_existing=handle_existing
        )
        path_output_folder = env_man.env_path

        ini_path = os.path.dirname(ini_scenario)

        new_path = os.path.join(path_output_folder, "additional_rover_files")

        if os.path.exists(new_path) is False:
            copytree(ini_path, new_path, ignore=include_patterns("*.py", "*.xml"))
            removeEmptyFolders(new_path)

        # Add vadere basis scenario used for the variation (i.e. sampling)
        if isinstance(basis_scenario, str):  # assume that this is a path
            if not os.path.isfile(basis_scenario):
                raise FileExistsError("Filepath to .scenario does not exist")
            elif basis_scenario.split(".")[-1] != cls.VADERE_SCENARIO_FILE_TYPE[1:]:
                raise ValueError(
                    "basis_scenario has to be a Vadere '*"
                    f"{cls.VADERE_SCENARIO_FILE_TYPE}' file"
                )

            with open(basis_scenario, "r") as file:
                basis_scenario = file.read()

        # add prefix to scenario file:
        basis_fp = os.path.join(
            path_output_folder,
            f"{cls.PREFIX_BASIS_SCENARIO}{env_man.get_scenario_name()}.scenario",
        )

        # FILL IN THE STANDARD FILES IN THE NEW SCENARIO:
        with open(basis_fp, "w") as file:
            if isinstance(basis_scenario, dict):
                json.dump(basis_scenario, file, indent=4)
            else:
                file.write(basis_scenario)  # this is where the scenario is defined

        # Add omnet basis scenario used for the variation (i.e. sampling)
        if isinstance(ini_scenario, str):  # assume that this is a path
            if not os.path.isfile(ini_scenario):
                raise FileExistsError("Filepath to .ini does not exist")
            elif ini_scenario.split(".")[-1] != "ini":
                raise ValueError("omnet ini has to be a ini file")

        # omnetpp output file at new location
        basis_fp = os.path.join(path_output_folder, "omnetpp.ini")
        # copy ini file but resolve all include directives before hand.
        # After the copy operation some the include directives might be invalid.
        OppParser.resolve_includes(
            ini_path=ini_scenario,
            output_path=basis_fp
        )

        # Create and store the configuration file to the new folder
        cfg = dict()

        # TODO it may be good to write the git hash / version number in the file
        if not SuqcConfig.is_package_paths():
            cfg["suqc_state"] = get_current_suqc_state()

            with open(
                os.path.join(path_output_folder, "suqc_commit_hash.json"), "w"
            ) as outfile:
                s = "\n".join(
                    ["commit hash at creation", cfg["suqc_state"]["git_hash"]]
                )
                outfile.write(s)

        # Create the folder where all output is stored

        path_output_folder_rover = os.path.join(
            path_output_folder, CoupledEnvironmentManager.simulation_runs_output_folder,
        )

        if os.path.exists(path_output_folder_rover) is False:
            os.mkdir(path_output_folder_rover)

        temp_folder_rover = os.path.join(
            path_output_folder, CoupledEnvironmentManager.temp_folder_rover,
        )

        if handle_existing != "write_in":
            if os.path.exists(temp_folder_rover):
                shutil.rmtree(temp_folder_rover)
            os.mkdir(temp_folder_rover)

        df = cls.env_info_df
        df.to_pickle(os.path.join(temp_folder_rover, "env_info.pkl"))

        return cls(base_path, env_name)

    def get_temp_folder(self):
        rel_path = os.path.join(
            self.env_path, CoupledEnvironmentManager.temp_folder_rover
        )
        return os.path.abspath(rel_path)

    def get_env_outputfolder_path(self):
        rel_path = os.path.join(
            self.env_path, CoupledEnvironmentManager.simulation_runs_output_folder
        )
        return os.path.abspath(rel_path)

    def get_variation_output_folder(self, parameter_id, run_id):
        scenario_filename = self._scenario_variation_filename(
            parameter_id=parameter_id, run_id=run_id
        )
        scenario_filename = scenario_filename.replace(
            self.VADERE_SCENARIO_FILE_TYPE, ""
        )

        variation_output_folder = os.path.join(
            self.get_env_outputfolder_path(), "".join([scenario_filename, "_output"])
        )

        return variation_output_folder

    def _scenario_variation_filename(self, parameter_id, run_id):
        digits_parameter_id = str(parameter_id).zfill(self.nr_digits_variation)
        digits_run_id = str(run_id).zfill(self.nr_digits_variation)
        numbered_scenario_name = "_".join([digits_parameter_id, digits_run_id])

        return "".join([numbered_scenario_name, self.VADERE_SCENARIO_FILE_TYPE])

    def get_original_name(self, simulator="vadere"):
        if simulator == "vadere":
            return os.path.basename(self.vadere_path_basis_scenario)
        elif simulator == "omnet":
            return "omnetpp.ini"
        else:
            return ""

    def scenario_variation_path(self, par_id, run_id, simulator="vadere"):

        subdir = self.simulators[simulator]["subdir"]
        original_name_scenario = self.get_original_name(simulator)

        sim_name = self.get_simulation_directory(par_id, run_id)
        sim_path = os.path.join(self.get_env_outputfolder_path(), sim_name)
        sim_path = os.path.join(sim_path, subdir)

        os.makedirs(sim_path, exist_ok=True)

        scenario_variation_path = os.path.join(sim_path, original_name_scenario)

        return scenario_variation_path

    def get_scenario_name(self):
        scenario_name = self.basis_scenario_name
        scenario_name = os.path.splitext(scenario_name)[0]
        return scenario_name

    def get_name_run_script_file(self):
        return CoupledEnvironmentManager.run_file

    def get_simulation_directory(self, par_id, run_id):
        prefix = CoupledEnvironmentManager.simulation_runs_single_folder_name
        return f"{prefix}_{par_id}_{run_id}"


class CrownetSumoEnvironmentManager(CoupledEnvironmentManager):

    def __init__(self,
                 base_path,
                 env_name: str,
                 opp_config: str = "final",
                 opp_basename: str = "omnetpp.ini",
                 output_folder="simulation_runs",
                 run_prefix="Sample_",
                 temp_folder="temp",
                 run_file="run_script.py",
                 mobility_sim=("sumo", "latest"),
                 communication_sim=("omnet", "latest"),
                 debug: bool = False):
        super().__init__(base_path, env_name)
        self._output_folder = output_folder
        self.run_prefix = run_prefix
        self._temp_folder = temp_folder
        self._omnet_path_ini = os.path.join(self.env_path, opp_basename)
        self._opp_config = opp_config
        self._omnet_ini_basis = None  # e.g. omnetpp.ini
        self.mobility_sim = mobility_sim
        self.communication_sim = communication_sim
        self.run_file = run_file
        self.debug = debug

    @property
    def omnet_path_ini(self):
        return self._omnet_path_ini

    def set_ini_filename(self, name):
        self._omnet_path_ini = os.path.join(self.env_path, name)

    @property
    def omnet_basis_ini(self):
        if self._omnet_ini_basis is None:
            _file = OppConfigFileBase.from_path(
                ini_path=self.omnet_path_ini,
                config=self._opp_config,
                cfg_type=OppConfigType.EXT_DEL_LOCAL,
            )
            self._omnet_ini_basis = _file
        return self._omnet_ini_basis

    @property
    def ini_config(self):
        return self._opp_config

    def set_ini_config(self, config):
        self._opp_config = config

    def get_variation_output_folder(self, parameter_id, run_id):

        return os.path.join(
            self.get_env_outputfolder_path(),
            "outputs",
            f"{self.run_prefix}{parameter_id}_{run_id}"
        )

    def get_env_outputfolder_path(self):
        rel_path = os.path.join(
            self.env_path, self._output_folder
        )
        return os.path.abspath(rel_path)

    def get_temp_folder(self):
        return os.path.join(
            self.env_path, self._temp_folder
        )

    def get_simulation_directory(self, par_id, run_id):
        return f"{self.run_prefix}_{par_id}_{run_id}"

    def copy_data(self, ini_path: str):
        """
        Copy environment *only* for selected configuration.
        Workflow:
          1.) set source_base as the directory where the ini-file (ini_path) resides
          2.) parse ini-file with given config (at source location to ensure relative paths are resolved correctly)
          3.) copy ini-file after resolving include directives to environment (String based. Will preserver comments).
          4.) add run_script to set of files to copy -> add to files
          5.) find all additional omnet files mentioned in the ini-file -> add to files
          6.) find sumo configuration from ini-file and parse xml for all files needed for sumo -> add to files
          7.) copy files to "additional_rover_files" in the current enviroment

        """
        # 1.) set source_base as the directory where the ini-file (ini_path) resides
        source_base = os.path.dirname(ini_path)
        # 2.) read base ini file to ensure paths are correct.
        opp_cfg = OppConfigFileBase.from_path(
            ini_path=ini_path, config=self.ini_config, cfg_type=OppConfigType.EXT_DEL_LOCAL,
        )
        sumo_cfg, simulator = check_simulator(opp_cfg)
        if self.mobility_sim[0] != simulator:
            raise RuntimeError("config missmatch. Selected configuration in ini file "
                               f" does not match DockerEnvironmentManager setup. {self.mobility_sim[0]}!={simulator}")

        # set ini file info
        self.set_ini_filename(os.path.basename(ini_path))

        # 3.) copy ini file manually and ensure includes are resoved.
        OppParser.resolve_includes(ini_path=ini_path, output_path=self.omnet_path_ini)

        # 4.) copy additional files (run_file, ...) and place in files set to copy into environment
        files = set()
        files.add(os.path.join(source_base, self.run_file))

        # 5.) check selected config for used files
        pattern = re.compile('.*(absFilePath|xmldoc)\((?P<val>.*)\)')
        for k, v in opp_cfg:
            match = pattern.match(v)
            if match is not None:
                file_name = match.group("val").strip('"')
                files.add(os.path.join(source_base, file_name))

        # 6.) add sumo specific files by parsing the sumo cfg file.
        files.add(sumo_cfg)
        sumo_xml = et.parse(sumo_cfg)
        inputs = sumo_xml.find("input")
        for i in inputs:
            files.add(os.path.join(os.path.dirname(sumo_cfg), i.get("value")))

        # 7.) copy files
        for f in files:
            if os.path.isfile(f):
                f_dir = os.path.dirname(f)
                f_name = os.path.basename(f)
                rel_path = os.path.relpath(f_dir, source_base)
                dest_path = os.path.join(self.env_path, "additional_rover_files", rel_path)
                os.makedirs(dest_path, exist_ok=True)
                shutil.copy(src=f, dst=dest_path)


class CrownetSumoWrapper(AbstractConsoleWrapper):

    def __init__(self):
        pass

    def build_args(self, env_man: CrownetSumoEnvironmentManager, r_item: RequestItem, required_files):
        """helper to create arguments for the given request item"""
        args = [
            "--run-name",
            f"{env_man.run_prefix}_{r_item.parameter_id}_{r_item.run_id}",
            "--opp.-f",
            os.path.basename(env_man.omnet_path_ini),
            "--opp.-c",
            env_man.ini_config,
            "--sumo-exec",
            "single-run",
            "--sumo.bind",
            "0.0.0.0",
            "--sumo.port",
            "9999",
            "--override-host-config",
            f"--create-{env_man.mobility_sim[0]}-container",
            "--delete-existing-containers",
            f"--{env_man.mobility_sim[0]}-tag",
            env_man.mobility_sim[1],
            f"--{env_man.communication_sim[0]}-tag",
            env_man.communication_sim[1],
            "--resultdir",
            r_item.output_path
        ]
        if len(required_files):
            args.append("--qoi")
            args.extend(required_files)

        if env_man.debug:
            args.append("--write-container-log")

        return args

    def run_simulation(self, dirname, start_file, args):
        terminal_command = ["python3", start_file]
        terminal_command.extend(args)

        print(" ".join(terminal_command))
        time_started = time.time()
        t = time.strftime("%H:%M:%S", time.localtime(time_started))
        print(f"{t}\t Call {os.path.basename(dirname)}/{start_file} ")

        try:
            return_code = subprocess.check_call(
                terminal_command,
                env=os.environ,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=dirname,
                timeout=15000,  # stop simulation after 15000s
            )
        except Exception as e:
            return_code = 255
            print(e)

        process_duration = time.time() - time_started
        output_subprocess = None

        return return_code, process_duration, output_subprocess
