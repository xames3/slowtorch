"""\
SlowTorch Utilities
===================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, January 07 2025
Last updated on: Sunday, June 01 2025

General purpose utilities.

This module provides core utility functions, constants, and lightweight
helpers that support the internal workings of SlowTorch. While these
components are not tied to any single feature like tensors, or autograd,
they are however used widely across this library to reduce duplication,
simplify code, and enforce shared conventions.

Some utilities mirror functionality found in PyTorch
(e.g., `set_printoptions`, `save`, `load`), while others are specific to
SlowTorch's philosophy of clarity and minimalism, like custom dtype
inference, module decoration, and recursive fill helpers.
"""

from __future__ import annotations

import builtins
import math
import pickle
import typing as t

import slowtorch
from slowtorch import function_dispatch

if t.TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType

    from slowtorch.types import BoolLikeType
    from slowtorch.types import Dtype
    from slowtorch.types import FileLike
    from slowtorch.types import FloatLikeType
    from slowtorch.types import IntLikeType
    from slowtorch.types import Tensor

__all__: list[str] = [
    "e",
    "inf",
    "nan",
    "newaxis",
    "pi",
]

_T = t.TypeVar("_T")

e: FloatLikeType = math.e
pi: FloatLikeType = math.pi
inf: FloatLikeType = float("inf")
nan: FloatLikeType = float("nan")
newaxis: t.NoneType = None


class PrinterOptions:
    """Printer options to mimic PyTorch's way."""

    precision: IntLikeType = 4


def fill(tensor: Tensor, values: Iterator[float]) -> None:
    """Recursively assign values from a flat iterator to a tensor."""
    if hasattr(tensor, "shape"):
        if len(tensor.shape) > 1:
            for dim in range(tensor.shape[0]):
                fill(tensor[dim], values)
        else:
            for dim in range(len(tensor)):
                tensor[dim] = next(values)


def py_impl_max(arg1: t.Any, arg2: FloatLikeType = 0.0) -> t.Any:
    """Mock function to type safe compute maximum values."""
    return builtins.max(arg1, arg2)


def py_impl_nexp(value: t.Any) -> t.Any:
    """Mock function to type safe compute negative exponentiations."""
    return math.exp(-value)


@function_dispatch
def set_printoptions(precision: None | IntLikeType = None) -> None:
    """Set options for printing."""
    from slowtorch.internal.tensor import Tensor

    if precision is None:
        precision = 4
    Tensor._print_opts.precision = precision


@function_dispatch
def set_module(mod: str) -> t.Callable[..., t.Any]:
    """Decorator for overriding `__module__` on a function or class."""

    def decorator(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
        """Inner function."""
        if mod is not None:
            func.__module__ = mod
        return func

    return decorator


@function_dispatch
def save(
    obj: object,
    f: FileLike,
    pickle_module: ModuleType = pickle,
    pickle_protocol: t.Literal[2] = 2,
) -> None:
    """Save an object to disk file."""
    with open(f, "wb") as opened_file:
        pickle_module.dump(obj, opened_file, protocol=pickle_protocol)


@function_dispatch
def load(
    f: FileLike,
    pickle_module: ModuleType = pickle,
    weights_only: None | BoolLikeType = None,
) -> t.Any:
    """Load an object saved from a file."""
    weights_only = weights_only
    with open(f, "rb") as opened_file:
        output = pickle_module.load(opened_file)
    return output


@function_dispatch
def typename(obj: t.Any) -> str:
    """String representation of the type of an object.

    This function returns a fully qualified string representation of an
    object's type.

    :param obj: The object whose type to represent
    :return: The type of the object.
    """
    if isinstance(obj, Tensor):
        return f"slowtorch.{obj.dtype.typename}"
    module = "slowtorch" or ""
    qualname = ""
    if hasattr(obj, "__qualname__"):
        qualname = obj.__qualname__
    elif hasattr(obj, "__name__"):
        qualname = obj.__name__
    else:
        module = obj.__class__.__module__ or ""
        qualname = obj.__class__.__qualname__
    return f"{module}.{qualname}"


@function_dispatch
def dtypecheck(dtype: None | t.Any) -> Dtype:
    """Return a valid SlowTorch dtype based on the provided value.

    :param dtype: A dtype-like object or Python primitive. If `None`,
        defaults to `slowtorch.float32`.
    :return: A canonical SlowTorch Dtype (`float32`, `int64`, or `bool`).
    """
    if dtype is None:
        return slowtorch.float32
    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        if name.startswith("float"):
            return slowtorch.float32
        if name.startswith("int"):
            return slowtorch.int64
        return slowtorch.bool
    if isinstance(dtype, float):
        return slowtorch.float32
    if isinstance(dtype, int):
        return slowtorch.int64
    return slowtorch.bool
