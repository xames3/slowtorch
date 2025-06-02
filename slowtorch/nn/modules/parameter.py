"""\
SlowTorch Parameter
===================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, June 01 2025
Last updated on: Sunday, June 01 2025

A tensor parameter object.

This module defined a `Parameter` which is typically used to define
learnable weights within a neural network or other computational models.
These module mimic the API and semantics of PyTorch's parameter object.
"""

from __future__ import annotations

import typing as t

from slowtorch.internal.tensor import Tensor
from slowtorch.nn.modules import modules_dispatch
from slowtorch.ops.random import randn
from slowtorch.utils import set_module

if t.TYPE_CHECKING:
    from slowtorch.types import BoolLikeType


@set_module("slowtorch.nn.parameter")
@modules_dispatch
class Parameter(Tensor):
    """A specialised subclass of `Tensor` designed to represent
    parameters in modules. A `Parameter` is typically used to define
    learnable weights within a neural network or other computational
    models.

    Unlike regular `Tensor` objects, `Parameter` is automatically
    registered as part of a model when assigned as an attribute of a
    module. This registration facilitates optimisation and gradient
    tracking during training.

    :param data: The underlying tensor data. If not provided, an
        uninitialised tensor of shape `(1,)` is created, defaults to
        `None`.
    :param requires_grad: A flag indicating whether gradients should
        be computed for this parameter during backpropagation, defaults
        to `True`.
    """

    def __init__(
        self,
        data: None | Tensor = None,
        requires_grad: BoolLikeType = True,
    ) -> None:
        """Initialise a `Parameter` instance with optional data."""
        if data is None:
            data = randn(1, requires_grad=requires_grad)
        else:
            data = data.clone()
        data.requires_grad = requires_grad
        for key, value in data.__dict__.items():
            setattr(self, key, value)

    @property
    def data(self) -> Tensor:
        """Return the underlying tensor data."""
        return self

    @data.setter
    def data(self, value: Tensor) -> None:
        """Ensure that setting `data` updates the parameter in-place."""
        if not isinstance(value, Tensor):
            raise TypeError("Parameter data must be a tensor")
        self.storage[:] = value.storage

    def __repr__(self) -> str:
        """Return a string representation of `Parameter` object."""
        return f"Parameter containing:\n{super().__repr__()}"
