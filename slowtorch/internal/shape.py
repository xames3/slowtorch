"""\
SlowTorch Shape
===============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, June 01 2025
Last updated on: Sunday, June 01 2025

Shape utilities.

This module defines shape-related utilities used throughout this library.
It provides functionality to represent, infer, manipulate, and validate
tensor shapes, strides, and memory layouts, all foundational to tensor
operations.

The central class `size` wraps a tuple of integers to semantically
represent tensor dimensions. These utilities form the backbone of
SlowTorch's internal tensor construction and indexing logic, enabling it
to mirror NumPy/PyTorch-style semantics with clean, Pythonic
implementations.
"""

from __future__ import annotations

import math
import typing as t
from collections.abc import Iterable

from slowtorch import function_dispatch
from slowtorch.utils import set_module

if t.TYPE_CHECKING:
    from slowtorch.internal.tensor import Tensor
    from slowtorch.types import BoolLikeType
    from slowtorch.types import Input
    from slowtorch.types import IntLikeType
    from slowtorch.types import ShapeType
    from slowtorch.types import StrideType


@set_module("slowtorch")
@function_dispatch
class size(tuple[int, ...]):
    """Represent the shape (size) of a tensor as a tuple of integers.

    This class extends the built-in tuple to provide a clear and
    descriptive representation for tensor dimensions.

    :param iterable: A tuple representing the dimensions of the tensor.
    """

    def __init__(self, iterable: tuple[IntLikeType, ...]) -> None:
        """Initialise a `size` instance with some iterable."""
        if not all(isinstance(dim, int) and dim >= 0 for dim in iterable):
            raise ValueError("Dimensions must be non-negative")
        self.iterable = iterable

    def __repr__(self) -> str:
        """Return a string representation of the `size` object."""
        return f"slowtorch.Size({list(self.iterable)})"

    def numel(self) -> IntLikeType:
        """Return total number of elements a tensor would contain."""
        return numel(self.iterable)


@function_dispatch
def infer_size_shapes(data: Input) -> size:
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
    shape: ShapeType = []

    def infer_size(input: Input, numel: IntLikeType) -> None:
        """Helper function to calculate shape recursively."""
        if isinstance(input, t.Sized) and not isinstance(input, (str, bytes)):
            if len(shape) <= numel:
                shape.append(0)
            length = len(input)
            if length > shape[numel]:
                shape[numel] = length
            for element in input:
                infer_size(element, numel + 1)

    infer_size(data, 0)
    return size(tuple(shape))


@function_dispatch
def make_contiguous_strides(shape: ShapeType, itemsize: IntLikeType) -> size:
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
    strides: StrideType = []
    stride: IntLikeType = itemsize
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return size(tuple(reversed(strides)))


@function_dispatch
def numel(shape: t.Sequence[IntLikeType] | IntLikeType) -> IntLikeType:
    """Calculate the total number of elements in a Tensor based on its
    shape.

    The total number of elements in a Tensor is the product of its
    dimensions, determined by multiplying the shape along each axis.

    :param shape: An integer or sequence of integers representing the
        dimensions of the Tensor. Each integer specifies the size along a
        particular dimension.
    :return: An integer representing the total number of elements in
        the Tensor.
    """
    if not isinstance(shape, Iterable):
        shape = (shape,)
    return math.prod(shape)


@function_dispatch
def get_step(view: Tensor) -> IntLikeType:
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
    contiguous = make_contiguous_strides(view.shape, view.itemsize)
    step = view._strides[-1] // contiguous[-1]
    strides = tuple(stride * step for stride in contiguous)
    return step if view._strides == strides else 0


@function_dispatch
def check_same_shape(args: Tensor) -> BoolLikeType:
    """Check if a nested iterable structure can form a Tensor with a
    uniform shape.

    A structure has a uniform shape if::

        - All elements along the same axis have the same shape.
        - Sub-elements (if any) also follow uniformity.

    :param args: A nested iterable structure to validate for Tensor
        compatibility.
    :return: True if the structure has a uniform shape, otherwise False.
    """
    if not isinstance(args, t.Iterable):
        return True
    return (
        all(check_same_shape(arg) for arg in args)
        and len(set(len(arg) for arg in args if isinstance(arg, t.Sized))) <= 1
    )


@set_module("slowtorch")
@function_dispatch
def broadcast_shapes(input: size, other: size) -> ShapeType:
    """Calculate the broadcast-compatible shape for two tensors.

    This function aligns the two shapes from the right, padding the
    smaller shape with `1`s on the left. Then, it checks compatibility
    for broadcasting::

        - Each tensor has at least one dimension.
        - Dimension sizes must either be equal, one of them is 1 or
          one of them does not exist.

    :param input: Shape of the first tensor.
    :param other: Shape of the second tensor.
    :return: The broadcast-compatible shape.
    :raises ValueError: If the shapes are incompatible for broadcasting.
    """
    shape: ShapeType = []
    r_input = list(reversed(input))
    r_other = list(reversed(other))
    maximum = max(len(r_input), len(r_other))
    r_input += [1] * (maximum - len(r_input))
    r_other.extend([1] * (maximum - len(r_other)))
    for idx, jdx in zip(r_input, r_other):
        if idx == jdx or idx == 1 or jdx == 1:
            shape.append(max(idx, jdx))
        else:
            raise ValueError(
                f"Operands couldn't broadcast together with shapes {input} "
                f"and {other}"
            )
    return tuple(reversed(shape))


def unravel_index(
    indices: IntLikeType,
    shape: ShapeType,
) -> ShapeType:
    """Convert a tensor of flat indices into a multi-dimensional
    index for a given shape.

    :param indices: Index position to unravel.
    :param shape: The shape of the tensor.
    :return: A tuple representing the multi-dimensional index.
    """
    size: ShapeType = []
    for dim in reversed(shape):
        size.append(indices % dim)
        indices = indices // dim
    return tuple(reversed(size))
