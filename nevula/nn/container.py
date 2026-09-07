from collections import OrderedDict
from typing import Any, Iterable, Union
from nevula.nn.module import Module


class Sequential(Module):
    """
    A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    The `forward()` method of `Sequential` accepts any input and cascades it
    through all contained modules in sequence.
    """

    def __init__(self, *args: Union[Module, OrderedDict, list]):
        super().__init__()
        self._module_list: list[Module] = []

        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            for idx, module in enumerate(args[0]):
                self.add_module(str(idx), module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def add_module(self, name: str, module: Module) -> None:
        """Adds a child module to the container."""
        if not isinstance(module, Module):
            raise TypeError(f"Module {type(module)} is not a Module subclass")
        self.register_module(name, module)
        self._module_list.append(module)

    def __len__(self) -> int:
        return len(self._module_list)

    def __getitem__(self, idx: int) -> Module:
        return self._module_list[idx]

    def __iter__(self):
        return iter(self._module_list)

    def forward(self, x: Any) -> Any:
        for module in self._module_list:
            x = module(x)
        return x
