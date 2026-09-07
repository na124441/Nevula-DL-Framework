from typing import Any, Iterator, Optional, Union
from nevula.nn.parameter import Parameter


class Module:
    """
    Base class for all neural network modules in Nevula.
    Your models should also subclass this class.
    Modules can contain other Modules, allowing nested tree structures.
    """

    def __init__(self):
        # We bypass standard __setattr__ during initialization of internal dicts
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")

        if isinstance(value, Parameter):
            if params is not None:
                params[name] = value
            if modules is not None and name in modules:
                del modules[name]
        elif isinstance(value, Module):
            if modules is not None:
                modules[name] = value
            if params is not None and name in params:
                del params[name]
        else:
            # If replacing an existing parameter or module with a non-module value
            if params is not None and name in params:
                del params[name]
            if modules is not None and name in modules:
                del modules[name]

        super().__setattr__(name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Defines the forward computation performed at every call."""
        raise NotImplementedError(f"Module [{type(self).__name__}] must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """Adds a parameter to the module."""
        if param is None:
            self._parameters.pop(name, None)
        elif not isinstance(param, Parameter):
            raise TypeError(f"Cannot register {type(param)} as parameter (must be Parameter or None)")
        else:
            self._parameters[name] = param

    def register_module(self, name: str, module: Optional['Module']) -> None:
        """Adds a child module to the current module."""
        if module is None:
            self._modules.pop(name, None)
        elif not isinstance(module, Module):
            raise TypeError(f"Cannot register {type(module)} as child module (must be Module or None)")
        else:
            self._modules[name] = module

    def parameters(self, recurse: bool = True) -> list[Parameter]:
        """Returns an iterator over module parameters."""
        params: list[Parameter] = []
        seen = set()
        for _, p in self.named_parameters(recurse=recurse):
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
        return params

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> list[tuple[str, Parameter]]:
        """Returns an iterator over module parameters, yielding both name and parameter."""
        named_params: list[tuple[str, Parameter]] = []
        
        # Current module's immediate parameters
        for name, param in self._parameters.items():
            if param is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                named_params.append((full_name, param))

        # Check in other attributes for un-registered parameters (e.g. lists or dicts)
        for attr_name, attr_val in self.__dict__.items():
            if attr_name not in self._parameters and attr_name not in ("_parameters", "_modules"):
                if isinstance(attr_val, Parameter):
                    full_name = f"{prefix}.{attr_name}" if prefix else attr_name
                    named_params.append((full_name, attr_val))

        # Recurse into child modules
        if recurse:
            for mod_name, module in self._modules.items():
                if module is not None:
                    child_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                    named_params.extend(module.named_parameters(prefix=child_prefix, recurse=True))

        return named_params

    def modules(self) -> list['Module']:
        """Returns a list of all modules in the network."""
        return [m for _, m in self.named_modules()]

    def named_modules(self, prefix: str = "", recurse: bool = True) -> list[tuple[str, 'Module']]:
        """Returns a list of all modules, yielding both name and module."""
        mods: list[tuple[str, 'Module']] = [(prefix, self)]
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    mods.extend(module.named_modules(prefix=child_prefix, recurse=True))
        return mods

    def train(self, mode: bool = True) -> 'Module':
        """Sets the module in training mode."""
        self.training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        return self

    def eval(self) -> 'Module':
        """Sets the module in evaluation mode."""
        return self.train(False)

    def zero_grad(self) -> None:
        """Sets gradients of all model parameters to None."""
        for p in self.parameters():
            p.zero_grad()

    def __repr__(self) -> str:
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _add_indent(mod_str, 2)
            child_lines.append(f"({key}): {mod_str}")

        lines = child_lines
        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str


def _add_indent(s_: str, num_spaces: int) -> str:
    s = s_.split("\n")
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s
