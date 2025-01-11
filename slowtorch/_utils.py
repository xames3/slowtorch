"""\
SlowTorch Utilities API
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, January 07 2025
Last updated on: Saturday, January 11 2025

This module provides utility classes, functions, and objects that are
essential to the core operations of SlowTorch. It is intended to
encapsulate common functionality and helpers that simplify the
development of the overall library, promoting code reuse and modular
design. The `utils` module includes utility classes and objects that are
frequently used across the SlowTorch framework, such as custom
decorators and helper functions. By centralizing frequently used
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

from slowtorch import function_dispatch

if t.TYPE_CHECKING:
    from slowtorch._types import Size
    from slowtorch._types import TensorBase


@function_dispatch
def calc_strides(
    shape: t.Sequence[int],
    itemsize: int,
) -> tuple[int, ...]:
    """Calculate memory strides for traversing a Tensor in row-major
    order.

    Strides represent the number of bytes required to move in memory
    between successive elements along each dimension of a Tensor. This
    function computes strides assuming a row-major (C-style) memory
    layout, where elements in the last dimension are stored
    contiguously.

    :param shape: A sequence of integers representing the dimensions of
        the Tensor. Each integer specifies the size along a particular
        dimension.
    :param itemsize: An integer specifying the size (in bytes) of each
        Tensor element. This depends on the data type of the Tensor.
    :return: A tuple of integers representing the memory strides for
        each dimension of the Tensor, in bytes.

    .. note::

        [1] Strides are critical for indexing Tensors efficiently and
            correctly.
        [2] Row-major order ensures that the last dimension changes the
            fastest in memory.
    """
    strides: list[int] = []
    stride: int = itemsize
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


@function_dispatch
def calc_size(shape: t.Sequence[int]) -> int:
    """Calculate the total number of elements in a Tensor based on its
    shape.

    The total number of elements in a Tensor is the product of its
    dimensions, determined by multiplying the shape along each axis.

    :param shape: A sequence of integers representing the dimensions of
        the Tensor. Each integer specifies the size along a particular
        dimension.
    :return: An integer representing the total number of elements in
        the Tensor.
    """
    return math.prod(shape)


@function_dispatch
def get_step(view: TensorBase) -> int:
    """Calculate the step size for traversing a Tensor along its last
    dimension.

    The step size determines the number of memory elements to skip when
    moving to the next index along the last axis of the Tensor. If the
    Tensor is C-contiguous (row-major layout), the step size is 1.
    Non-contiguous Tensors return a step size of 0.

    :param view: A Tensor object with attributes `shape`, `strides`, and
        `itemsize`, representing its memory layout.
    :return: An integer step size: 1 for C-contiguous Tensors, 0
        otherwise.
    """
    contiguous = calc_strides(view.shape, view.itemsize)
    step = view.strides[-1] // contiguous[-1]
    strides = tuple(stride * step for stride in contiguous)
    return step if view.strides == strides else 0


@function_dispatch
def calc_shape_from_data(data: t.Any) -> Size:
    """Infer the shape of a nested iterable structure and represent it
    as a Tensor.

    This function recursively determines the dimensions of a nested
    structure (e.g., a list of lists) and converts it into a tuple of
    integers representing the corresponding Tensor shape.

    :param data: A nested iterable structure that can be converted
        into a Tensor. Each level of nesting corresponds to a dimension
        in the shape.
    :return: A tuple of integers representing the shape of the input
        data.
    """
    shape: list[int] = []

    def _calc_shape(obj: t.Any, axis: int) -> None:
        """Helper function to calculate shape recursively."""
        if isinstance(obj, t.Sized) and not isinstance(obj, (str, bytes)):
            if len(shape) <= axis:
                shape.append(0)
            length = len(obj)
            if length > shape[axis]:
                shape[axis] = length
            for element in obj:
                _calc_shape(element, axis + 1)

    _calc_shape(data, 0)
    return tuple(shape)


@function_dispatch
def has_uniform_shape(data: TensorBase) -> bool:
    """Check if a nested iterable structure can form a Tensor with a
    uniform shape.

    A structure has a uniform shape if::

        - All elements along the same axis have the same shape.
        - Sub-elements (if any) also follow uniformity.

    :param data: A nested iterable structure to validate for Tensor
        compatibility.
    :return: True if the structure has a uniform shape, otherwise False.
    """
    if not isinstance(data, t.Iterable):
        return True
    return (
        all(has_uniform_shape(idx) for idx in data)
        and len(set(len(idx) for idx in data if isinstance(idx, t.Sized))) <= 1
    )
