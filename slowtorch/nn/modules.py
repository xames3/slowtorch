"""\
SlowTorch Modules API
=====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Thursday, January 16 2025
Last updated on: Friday, January 17 2025

This module provides a foundational framework for building and training
neural networks, inspired by PyTorch's flexible and dynamic design. It
enables users to create custom models by leveraging a modular and
extensible architecture where components like layers and parameters are
automatically registered and optimized during training. 

The `Module` class serves as the base class for all neural network
components. It automatically registers submodules and learnable
parameters when assigned as attributes. It also provides mechanisms for
forward computation, parameter iteration, and mode switching (training
vs. evaluation). Whereas, the `Parameter` class is a specialized
subclass of `Tensor` designed to represent learnable parameters, such as
weights and biases. It automatically tracks gradients and integrates
seamlessly with the `Module` hierarchy.

This module provides key important features like::

    1. Automatic Registration: Submodules (`Module`) and learnable
       parameters (`Parameter`) are automatically registered when
       assigned to attributes of a `Module`. This eliminates the need
       for explicit registration methods and ensures consistency across
       the framework.
    2. Dynamic Forward Computation: The `forward` method in the `Module`
       class is designed to be overridden by subclasses to define the
       specific computation for the module. Calling the module object
       directly invokes this method, enabling an intuitive interface.
    3. Gradient Handling: The API provides utilities for resetting
       gradients (`zero_grad`) and managing backpropagation seamlessly.
    4. Training and Evaluation Modes: Modules and their submodules can
       be easily switched between training and evaluation modes using
       the `train` and `eval` methods.
    5. Iterative Access: The API supports hierarchical iteration over
       submodules and parameters, making it easy to traverse complex
       model architectures.
"""

from __future__ import annotations

import typing as t
from collections import OrderedDict

from slowtorch import empty
from slowtorch._tensor import Tensor

__all__: list[str] = [
    "Flatten",
    "Module",
    "Parameter",
]


class Parameter(Tensor):
    """A specialized subclass of `Tensor` designed to represent
    parameters in modules. A `Parameter` is typically used to define
    learnable weights within a neural network or other computational
    models.

    Unlike regular `Tensor` objects, `Parameter` is automatically
    registered as part of a model when assigned as an attribute of a
    module. This registration facilitates optimization and gradient
    tracking during training.

    :param data: The underlying tensor data. If not provided, an
        uninitialized tensor of shape `(1,)` is created, defaults to
        `None`.
    :param requires_grad: A flag indicating whether gradients should
        be computed for this parameter during backpropagation, defaults
        to `True`.
    """

    def __init__(
        self,
        data: None | Tensor = None,
        requires_grad: bool = True,
    ) -> None:
        """Initialize a `Parameter` instance with optional data."""
        if data is None:
            data = empty((1,), requires_grad=requires_grad)
        data.requires_grad = requires_grad
        self._param = self = data

    def __repr__(self) -> str:
        """Return a string representation of `Parameter` object."""
        return f"Parameter containing:\n{self._param}"


class Module:
    """Base class for all neural network modules.

    All models should subclass this class. This class provides
    fundamental building blocks such as parameter registration, forward
    computation, and gradient handling. Submodules and parameters are
    automatically registered when assigned as attributes.
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialize `Module` instance with optional arguments."""
        self.args = args
        self.kwargs = kwargs
        self.training: bool = True
        self._modules: dict[str, None | Module] = OrderedDict()
        self._parameters: dict[str, None | Parameter] = OrderedDict()

    def __repr__(self) -> str:
        """Return a string representation of the `Module` object."""
        modules = "\n"
        for key, module in vars(self).items():
            if key in (
                "args",
                "kwargs",
                "training",
                "_modules",
                "_parameters",
            ):
                continue
            modules += f"  ({key}): {module}\n"
        return f"{type(self).__name__}({modules})"

    def __setattr__(self, name: str, value: Module | Parameter) -> None:
        """Override to automatically register submodules and parameters.

        :param name: Attribute name.
        :param value: Attribute value to be set.
        """
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def forward(self, input: Tensor) -> Tensor:
        """Define the computation performed at every call.

        Should be overridden by all subclasses.

        :raises NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError(
            f"Module: {type(self).__name__} is missing the"
            ' required "forward" function'
        )

    __call__: t.Callable[[Tensor], Tensor] = forward

    def children(self) -> t.Iterator[None | Module]:
        """Yield an iterator over immediate child modules."""
        yield from self._modules.values()

    def modules(self) -> t.Iterator[None | Module]:
        """Yield an iterator over all modules in the network."""
        yield self
        for module in self.children():
            yield from module.modules()

    def parameters(self, recurse: bool = True) -> t.Iterator[None | Parameter]:
        """Return an iterator over module parameters.

        :param recurse: If True, include parameters of submodules,
            defaults to `True`.
        :yield: The parameters of the module.
        """
        yield from self._parameters.values()
        if recurse:
            for module in self.children():
                yield from module.parameters()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset the gradients of all parameters in the module and its
        children.

        This ensures no leftover gradients from a previous backward
        pass.

        :param set_to_none: Bool to set grad to `None` instead of
            setting to 0.0.
        """
        for param in self.parameters():
            if param is not None and param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad = empty((1,))

    def train(self, mode: bool = True) -> Module:
        """Set the module and its submodules to training mode.

        :param mode: Whether to set to training mode or evaluation mode,
            defaults to `False`.
        :return: Itself.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> Module:
        """Set the module and its submodules to evaluation mode."""
        return self.train(False)


class Flatten(Module):
    """Flatten a contiguous range of dims into a tensor."""

    def __init__(self) -> None:
        """Initialize `Flatten` instance with `Module` base class."""
        super().__init__()

    def __repr__(self) -> str:
        """Return a string representation of the `Flatten` object."""
        return f"{type(self).__name__}()"

    def forward(self, input: Tensor) -> Tensor:
        """Compute the forward pass."""
        return input.flatten()
