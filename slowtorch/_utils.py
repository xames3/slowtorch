"""\
SlowTorch Utilities API
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, January 07 2025
Last updated on: Thursday, January 16 2025

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
from collections.abc import Iterable

from slowtorch import function_dispatch

if t.TYPE_CHECKING:
    from slowtorch._tensor import Tensor
    from slowtorch._types import Number

__all__: list[str] = [
    "device",
    "dtype",
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
class Device:
    """Represent a computational device for executing Tensor operations.

    The `Device` class in SlowTorch encapsulates the concept of a
    computation backend, such as `cpu`. It provides a way to specify the
    target device where Tensor computations will occur, including
    support for multi-device systems using an optional device index.

    This abstraction allows users to explicitly manage computational
    resources, mimicking PyTorch's `torch.device` behavior.

    :param type: The type of the device, defaults to `cpu`.
    :param index: An optional index representing the device number,
        defaults to 0.
    """

    __module__: str = "slowtorch"
    __qualname__: str = "device"

    def __init__(self, type: str = "cpu", index: int = 0) -> None:
        """Initialize a new `Device` object with default index."""
        self.type = type
        self.index = index

    def __repr__(self) -> str:
        """Return a string representation of the `Device` object."""
        return f"device(type={self.type!r}, index={self.index})"

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"{self.type}:{self.index}"


DeviceType: t.TypeAlias = None | str | int | Device
device = Device


@function_dispatch
class Dtype:
    """Represent data types used in the SlowTorch framework.

    The `Dtype` class encapsulates information about supported datatypes
    for tensor operations and storage in SlowTorch. It provides a way to
    describe the type of data stored in tensors, its size in bytes, and
    associated metadata.

    :param name: The full name of the datatype.
    :param short: A shorthand representation of the datatype, where the
        last character specifies the size in bytes.
    :param data: A representation of the type's data structure or
        internal details.
    :param value: A representative value for the data type, used for
        internal operations or comparisons.
    """

    __module__: str = "slowtorch"
    __qualname__: str = "dtype"

    def __init__(
        self,
        name: str,
        short: str,
        data: t.Any,
        value: Number | bool,
    ) -> None:
        """Initialize a new `Dtype` object with name and value."""
        self.name = name
        self.itemsize = int(short[-1])
        self.data = data
        self.value = value

    def __repr__(self) -> str:
        """Return a string representation of the `Dtype` object."""
        return f"slowtorch.{self.name}"


dtype = Dtype


@function_dispatch
class Size(tuple[int, ...]):
    """Represent the shape of a tensor as a tuple of integers.

    This class extends the built-in tuple to provide a clear and
    descriptive representation for tensor dimensions.

    :param shape: A tuple representing the dimensions of the tensor.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        """Initialize a `Size` instance with some shape."""
        if not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            raise ValueError("Dimensions must be non-negative")
        self.value = shape

    def __repr__(self) -> str:
        """Return a string representation of the `Size` object."""
        return f"slowtorch.{type(self).__qualname__}({list(self.value)})"


@function_dispatch
def calc_strides(
    shape: t.Sequence[int],
    itemsize: int,
) -> Size:
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
    return Size(tuple(reversed(strides)))


@function_dispatch
def calc_size(shape: t.Sequence[int] | int) -> int:
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
def get_step(view: Tensor) -> int:
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
    step = view._strides[-1] // contiguous[-1]
    strides = tuple(stride * step for stride in contiguous)
    return step if view._strides == strides else 0


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
    return Size(tuple(shape))


@function_dispatch
def has_uniform_shape(data: Tensor) -> bool:
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


@function_dispatch
def normal_exp(value: float) -> float:
    """Dummy function to type safe compute exponentiations."""
    return math.exp(value)


@function_dispatch
def safe_exp(value: float) -> float:
    """Dummy function to type safe compute negative exponentiations."""
    return math.exp(-value)


@function_dispatch
def safe_max(arg1: float, arg2: float = 0.0) -> float:
    """Dummy function to type safe compute maximum values."""
    return max(arg1, arg2)
