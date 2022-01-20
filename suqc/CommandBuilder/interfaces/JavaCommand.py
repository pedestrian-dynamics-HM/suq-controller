import abc
import subprocess
import time
from typing import Tuple, List, Iterator, Union

from suqc.CommandBuilder.interfaces.Command import Command
from suqc.requestitem import RequestItem


class JavaCommand(Command):
    _options: List[str] = []
    _sub_command: str = ""
    _main_class: str = ""
    ALLOWED_LOGLVL = [
        "OFF",
        "FATAL",
        "TOPOGRAPHY_ERROR",
        "TOPOGRAPHY_WARN",
        "INFO",
        "DEBUG",
        "ALL",
    ]

    def __init__(self):
        super().__init__()
        self._set_sub_command()

    @abc.abstractmethod
    def _set_sub_command(self) -> None:
        pass

    def __str__(self):
        return f"{self._executable} " \
               f"{' '.join(self._options)} " \
               f"{self._sub_command} " \
               f"{self._main_class} " \
               f"{self._arguments}"\
            .strip()

    def arg_list(self) -> List[str]:
        args =[]
        if self._options:
            args.extend(self._options)
        args.extend(self._sub_command.split(" "))
        if self._main_class:
            args.append(self._main_class)
        if self._arguments:
            args.extend(list(self._arguments))
        return args
    
    def __iter__(self) -> Iterator[str]:
        run_command: List[str] = [self._executable]
        run_command.extend(self.arg_list())
        return run_command.__iter__()

    def _set_executable(self) -> None:
        self._executable = "java"

    def add_option(self, option: str) -> "JavaCommand":
        self._options.append(option)
        return self

    def main_class(self, main_class: str) -> "JavaCommand":
        self._main_class = main_class
        return self

    def add_argument(self, key: str, value: str) -> "JavaCommand":
        self._arguments[key] = value
        return self

    def log_level(self, log_level: str) -> "JavaCommand":
        log_level = log_level.upper()
        if log_level not in self.ALLOWED_LOGLVL:
            raise ValueError(
                f"set loglvl={log_level} not contained "
                f"in allowed: {self.ALLOWED_LOGLVL}"
            )
        self.add_argument(key="--loglevel", value=log_level)
        return self

    # def run(self, cwd: str, file_name: str) -> Tuple[int, float]:
    def run(self, timeout_sec: int = None) -> Tuple[int, float, dict]:
        start = time.time()
        output_subprocess = dict()
        run_command = list(self)

        try:
            subprocess.check_output(
                run_command, timeout=timeout_sec, stderr=subprocess.PIPE
            )
            process_duration = time.time() - start

            # if return_code != 0 a subprocess.CalledProcessError is raised
            return_code = 0
            output_subprocess = None
        except subprocess.TimeoutExpired as exception:
            return_code = 1
            process_duration = timeout_sec
            output_subprocess["stdout"] = exception.stdout
            output_subprocess["stderr"] = None
        except subprocess.CalledProcessError as exception:
            return_code = exception.returncode
            process_duration = time.time() - start
            output_subprocess["stdout"] = exception.stdout
            output_subprocess["stderr"] = exception.stderr

        return return_code, process_duration, output_subprocess

    def write_context(self, ctx_path: str, cwd: str, r_item: Union[RequestItem, None] = None):
        raise NotImplemented()