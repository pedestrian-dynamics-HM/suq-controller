import abc
from typing import Tuple, Union, List

from suqc.CommandBuilder.interfaces.CommandArguments import CommandArguments
from suqc.requestitem import RequestItem
from os.path import abspath, dirname
import datetime

class Command(abc.ABC):
    _executable: str = None
    _arguments: CommandArguments = None

    def __init__(self):
        self._set_executable()
        self._arguments = CommandArguments()

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def _set_executable(self) -> None:
        pass

    @abc.abstractmethod
    def run(self, cwd: str, file_name: str) -> Tuple[int, float]:
        pass
    
    @abc.abstractclassmethod
    def arg_list(self) -> List[str]:
        pass

    def write_context(self, ctx_path: str, cwd: str, r_item: Union[RequestItem, None] = None):
        ctx = {}
        ctx_path = abspath(ctx_path)
        ctx_dir = dirname(ctx_path)

        ctx["description"] = "SUQ-Controler model context used for execution."
        ctx["created_at"] = datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        ctx["cmd_args"] = self.arg_list()
        ctx["cwd"] = cwd
        if r_item is not None:
            ctx["request_item"] = vars(r_item)
        
        ctx["comand_arguments"] = self._arguments.store
        return ctx_dir, ctx


