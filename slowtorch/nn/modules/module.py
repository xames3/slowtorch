"""\
SlowTorch Module
================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, June 01 2025
Last updated on: Sunday, June 01 2025

Base module object.

This module provides classes like `Module` and `Sequential` which allows
building neural network models. These classes mimic the API and
semantics of PyTorch's `nn.Module` and `nn.Sequential` objects.
"""

from __future__ import annotations

import abc
import typing as t
from collections import OrderedDict

from slowtorch.internal.tensor import Tensor
from slowtorch.nn.modules import modules_dispatch
from slowtorch.nn.modules.parameter import Parameter
from slowtorch.utils import set_module

if t.TYPE_CHECKING:
    from slowtorch.types import BoolLikeType
    from slowtorch.types import ModuleType


@set_module("slowtorch.nn.modules.module")
@modules_dispatch
class Module(abc.ABC):
    """Base class for all neural network modules.

    All models should subclass this class. This class provides
    fundamental building blocks such as parameter registration, forward
    computation, and gradient handling. Submodules and parameters are
    automatically registered when assigned as attributes.
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialise `Module` instance with optional arguments."""
        self.args = args
        self.kwargs = kwargs
        self.training: BoolLikeType = True
        self._modules: dict[str, None | Module] = OrderedDict()
        self._parameters: dict[str, None | Parameter] = OrderedDict()

    def __repr__(self) -> str:
        """Return a string representation of the `Module` class."""
        modules = "\n"
        for key, module in self._modules.items():
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

    def __call__(self, *input: Tensor, **kwargs: t.Any) -> Tensor:
        """Invoke the `forward` method."""
        return self.forward(*input, **kwargs)

    @t.overload
    def forward(self, input: Tensor) -> Tensor: ...
    @t.overload
    def forward(self, input: Tensor, target: Tensor) -> Tensor: ...

    @abc.abstractmethod
    def forward(self, *input: Tensor, **kwargs: t.Any) -> Tensor:
        """Define the computation performed at every call.

        Should be overridden by all subclasses.

        :raises NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError(
            f"Module: {type(self).__name__} is missing the"
            ' required "forward" function'
        )

    def children(self) -> ModuleType:
        """Yield an iterator over immediate child modules."""
        yield from self._modules.values()

    def modules(self) -> ModuleType:
        """Yield an iterator over all modules in the network."""
        yield self
        for module in self.children():
            yield from module.modules()

    def parameters(
        self, recurse: BoolLikeType = True
    ) -> t.Iterator[None | Parameter]:
        """Return an iterator over module parameters.

        :param recurse: If True, include parameters of submodules,
            defaults to `True`.
        :yield: The parameters of the module.
        """
        yield from self._parameters.values()
        if recurse:
            for module in self.children():
                yield from module.parameters()

    def zero_grad(self, set_to_none: BoolLikeType = True) -> None:
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
                    param.grad = Tensor(1)

    def train(self, mode: BoolLikeType = True) -> Module:
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


@set_module("slowtorch.nn.modules.container")
@modules_dispatch
class Sequential(Module):
    """A container module that sequentially applies a list of
    sub-modules.

    Modules in `Sequential` are executed in the order they are added.
    This class is particularly useful for building neural network layers
    that can be executed one after another without requiring manual
    forwarding.

    Modules can be added either as positional arguments or as a single
    ordered dictionary.
    """

    def __init__(self, *args: Module) -> None:
        """Initialise the `Sequential` container with modules."""
        super().__init__()
        self._modules = OrderedDict()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            iterator = args[0].items()
        else:
            iterator = enumerate(args)
        for name, module in iterator:
            self.__setattr__(str(name), module)

    def __len__(self) -> int:
        """Return the number of sub-modules in `Sequential` container.

        :return: The count of sub-modules.
        """
        return len(self._modules)

    def __iter__(self) -> ModuleType:
        """Return an iterator over the sub-modules in `Sequential`
        container.

        :return: An iterator over the sub-modules.
        """
        return iter(self._modules.values())

    def forward(self, input: Tensor) -> Tensor:
        """Compute the forward pass by applying each sub-module in
        sequence.

        :param input: The input tensor to the `Sequential` container.
        :return: The output tensor after applying all sub-modules.
        """
        for module in self:
            if module is not None:
                input = module(input)
        return input
