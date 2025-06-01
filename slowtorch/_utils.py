"""\
SlowTorch Utilities API
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, January 07 2025
Last updated on: Saturday, May 31 2025

This module provides utility classes, functions, and objects that are
essential to the core operations of SlowTorch. It is intended to
encapsulate common functionality and helpers that simplify the
development of the overall library, promoting code reuse and modular
design. The `utils` module includes utility classes and objects that are
frequently used across the SlowTorch framework, such as custom
decorators and helper functions. By centralising frequently used
utilities, the module aims to maintain a clean and modular structure,
enabling easy maintenance, extensibility, and testability.

These utilities are not limited to specific aspects of the SlowTorch
framework but are general-purpose, allowing them to support multiple
components, such as tensor manipulations, mathematical operations, and
type systems.

This module is critical to the SlowTorch project's overarching goals of
simplicity and modularity. By leveraging only Python's standard library,
SlowTorch intentionally forgoes third-party dependencies, making the
utility functions, classes, and objects within `utils.py` essential for
achieving flexibility, performance, and code clarity. The utilities
housed here support the core computational structures of SlowTorch,
aiding in data type management, object construction, and internal
operations.

The `utils` module is intended for developers contributing to SlowTorch,
as well as any users who wish to understand the underlying mechanics of
the framework. It is designed with simplicity and transparency in mind,
making it an ideal reference for those learning about deep learning
library construction or looking to extend SlowTorch's capabilities.
"""

from __future__ import annotations

import math
import typing as t
from collections.abc import Iterator

from slowtorch import function_dispatch

if t.TYPE_CHECKING:
    from slowtorch._tensor import Tensor

__all__: list[str] = [
    "e",
    "inf",
    "nan",
    "newaxis",
    "pi",
]

e: float = math.e
inf: float = float("inf")
nan: float = float("nan")
newaxis: t.NoneType = None
pi: float = math.pi


@function_dispatch
def set_module(mod: str) -> t.Callable[..., t.Any]:
    """Decorator for overriding `__module__` on a function or class."""

    def decorator(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
        """Inner function."""
        if mod is not None:
            func.__module__ = mod
        return func

    return decorator


def _fill_tensor(tensor: Tensor, values: Iterator[float]) -> None:
    """Recursively assign values from a flat iterator to a tensor."""
    if hasattr(tensor, "shape") and len(tensor.shape) > 1:
        for dim in range(tensor.shape[0]):
            _fill_tensor(tensor[dim], values)
    else:
        for dim in range(len(tensor)):
            tensor[dim] = next(values)
