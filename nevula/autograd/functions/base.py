from typing import Any
from nevula.autograd.engine import GradMode
from nevula.autograd.node import Context, Node


class Function:
    """
    Base class for all differentiable operations in Nevula.
    Subclasses must implement static methods:
        forward(ctx, *args, **kwargs)
        backward(ctx, *grad_outputs)
    """

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        from nevula.core.tensor import Tensor

        # Check if any input Tensor requires grad and gradient mode is enabled
        needs_grad = GradMode.is_enabled() and any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )

        ctx = Context()

        # Execute forward pass
        output = cls.forward(ctx, *args, **kwargs)

        if needs_grad:
            next_functions = []
            for arg in args:
                if isinstance(arg, Tensor):
                    next_functions.append((arg.grad_fn, arg))
                else:
                    next_functions.append((None, arg))

            node = Node(
                name=f"{cls.__name__}Backward",
                ctx=ctx,
                next_functions=next_functions,
                backward_fn=cls.backward
            )

            if isinstance(output, Tensor):
                output.grad_fn = node
                output.requires_grad = True
            elif isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, Tensor):
                        item.grad_fn = node
                        item.requires_grad = True

        return output

    @staticmethod
    def forward(ctx: Context, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, *grad_outputs: Any) -> Any:
        raise NotImplementedError
