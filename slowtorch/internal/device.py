"""\
SlowTorch Device
================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, 01 June 2025
Last updated on: Sunday, 24 August 2025

Mock device wrapper.

This module provides `device` class, a minimal, Pythonic representation
of a computation target (e.g. CPU). This abstraction allows students to
explicitly specify where tensor operations should be executed, enabling
device-aware computations in a manner similar to `torch.device`.

.. note::

    SlowTorch supports computation on CPU only.
"""

from __future__ import annotations

import typing as t

from slowtorch import function_dispatch
from slowtorch.utils import set_module

if t.TYPE_CHECKING:
    from slowtorch.types import IntLikeType


@set_module("slowtorch")
@function_dispatch
class device:
    """Class to represent a computational device.

    The `device` class encapsulates the concept of a computation
    backend, such as `cpu`. It provides a way to specify the target
    device where Tensor computations will occur.

    This abstraction allows students to explicitly manage computational
    resources, mimicking PyTorch's `torch.device` behavior.

    :param type: The type of the device, defaults to `cpu`.
    :param index: An optional index representing the device number,
        defaults to 0.
    :raises RuntimeError: If an unsupported device is specified.
    """

    def __init__(self, type: str = "cpu", index: IntLikeType = 0) -> None:
        """Initialise a new `device` object with default index."""
        if type != "cpu":
            raise RuntimeError("SlowTorch not compiled for other backends")
        self.type = type
        self.index = index

    def __repr__(self) -> str:
        """Return a string representation of the `device` object."""
        return f"{type(self).__name__}(type={self.type!r}, index={self.index})"

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"{self.type}:{self.index}"
