"""\
SlowTorch Tensor Manipulation
=============================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, June 01 2025
Last updated on: Sunday, June 01 2025

Tensor manipulation operations.

This module provides essential tensor manipulation and mutating
utilities. It contains functions designed to transform the `Tensor`
objects with various patterns and values, mimicking the functionality of
PyTorch's core tensor creation routines.

This module serves as the foundation for manipulating tensors with
specific sizes, patterns, and values.

.. note::

    The implementations in this module are not optimised for performance
    and are intended for learning and exploratory purposes. For
    production-grade numerical computation, consider using PyTorch
    directly.
"""

from __future__ import annotations

import typing as t

from slowtorch import function_dispatch
from slowtorch.internal.tensor import Tensor
from slowtorch.ops.creation import empty

if t.TYPE_CHECKING:
    from slowtorch.types import IntLikeType


@function_dispatch
def cat(
    tensors: t.Sequence[Tensor],
    dim: IntLikeType = 0,
) -> Tensor:
    """Concatenate the given sequence of tensors in the given dimension.

    The `cat` function creates an tensor with the same size as the input
    tensors.

    :param tensors: Sequence of tensors of same type.
    :param dim: Dimension over which the tensors are concatenated,
        defaults to 0.
    :return: A new tensor concatenated over the provided dimension.
    """
    if not all(map(lambda x: x.shape == tensors[0].shape, tensors)):
        raise ValueError("Tensors must have same shapes and dimensions")
    size = list(tensors[0].shape)
    size[dim] = size[dim] * len(tensors)
    new_tensor = empty(*size, dtype=tensors[0].dtype)
    offset = 0
    for tensor in tensors:
        slices = [slice(None)] * len(tensor.shape)
        slices[dim] = slice(offset, offset + tensor.shape[dim])
        new_tensor[tuple(slices)] = tensor
        offset += tensor.shape[dim]
    return new_tensor
