import abc
import os
import subprocess
import json
import time
from abc import ABC
from typing import Tuple, List, Union

from suqc.CommandBuilder.interfaces.Command import Command
from suqc.requestitem import RequestItem


class Python3Command(Command, ABC):
    _sub_command: str = None
    _script: str = None

    def __init__(self):
        super().__init__()
        self._set_sub_command()
        self.timeout = 15000 # stop simulation after 15000s ~4h

    @abc.abstractmethod
    def _set_sub_command(self) -> None:
        pass

    def __str__(self):
        return f"{self._executable} {self._script} {self._sub_command} {self._arguments}"

    def set_script(self, file_name):
        self._script = file_name

    def _set_executable(self) -> None:
        self._executable = "python3"

    def force_set_run_name(self, run_name: str):
        self._arguments["--run-name"] = run_name
        return self

    def arg_list(self) -> List[str]:
        return [self._sub_command, *list(self._arguments)]

    def run(self, cwd: str, file_name: Union[str, None] = None) -> Tuple[int, float]:
        if file_name is not None:
            self.set_script(file_name)
        time_started = time.time()
        t: str = time.strftime("%H:%M:%S", time.localtime(time_started))
        #print(f"{t}\t Call {str(self)}")
        print(f"Call {self._arguments['--run-name']}.")

        run_command: List[str] = [self._executable, self._script, *self.arg_list()]
        try:
            return_code: int = subprocess.check_call(
                run_command,
                env=os.environ,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=cwd,
                timeout=self.timeout,  
            )
        except subprocess.CalledProcessError:
            print(f"Simulation failed: {run_command}")
            return_code = -1

        process_duration = time.time() - time_started
        return return_code, process_duration

    def write_context(self, ctx_path: str, cwd, r_item: Union[RequestItem, None] = None):
        ctx_dir, ctx = super().write_context(ctx_path, cwd=cwd, r_item=r_item)
        ctx["script"] = self._script
        ctx_json = json.dumps(ctx, indent=2, sort_keys=False)
        os.makedirs(ctx_dir, exist_ok=True)
        with open(ctx_path, "w", encoding="utf-8") as fd:
            fd.write(ctx_json)