"""\
SlowTorch Tensor API
====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, January 07 2025
Last updated on: Saturday, May 31 2025

Tensor object.

This module provides the foundational classes and functions for tensor
operations in SlowTorch. It defines the `Tensor` class and supporting
utility like the `tensor` function to create and manipulate tensors.
SlowTorch aims to mimic PyTorch's tensor behavior while implementing key
functionalities from scratch to foster learning and customisation.

The `Tensor` class acts as the primary data structure for
multidimensional arrays and supports features like automatic
differentiation, and flexible data types. Whereas, the `tensor` function
is a factory function to create tensors from nested data structures. It
infers tensor shape, data type, and supports optional device
specification and gradient requirements. Designed with flexibility,
efficiency, and modularity in mind, the `Tensor` class aims to
replicate and innovate upon the core features of PyTorch tensors while
emphasising a Python-standard-library-only approach. Additionally, the
class introduces a Pythonic, educational perspective, making it suitable
for learning and experimentation with tensor mechanics without relying
on external libraries.

The module supports features such as::

    - The primary `Tensor` object supports auto-differentiation.
    - Efficient storage and representation of n-dimensional data.
    - Flexible shape manipulation, including reshaping and broadcasting.
    - Element-wise operations, including arithmetic, logical, and
      comparison operations, via rich operator overloading.
    - Slicing and indexing support for intuitive data access.
    - Conversion utilities to export data to native Python types
      (e.g., lists).
    - Tensor based operations such as unsqueezing, clamping, reshaping,
      transposing, cloning, etc.
    - Linear algebraic operations such as calulating maximum, minimum,
      mean, standard deviation, exponent, log, square root etc.

The `Tensor` implementation draws inspiration from PyTorch's
architecture but deliberately simplifies and reimagines certain aspects
for educational purposes and to meet the constraints of pure Python. By
eschewing C or Cython extensions, the `Tensor` class offers an
accessible implementation that emphasises algorithmic clarity over raw
performance.

In addition to the core `Tensor` functionality, this module
introduces several helper functions to aid in tensor manipulation and
generation. These functions are designed to complement the `Tensor`
class and mimic key functionality found in PyTorch.

While this module implements many fundamental features of `Tensor`,
it does not aim to match PyTorch's performance or breadth. Instead, the
focus is on clarity, usability, and modularity, providing a platform for
learning and experimentation.
"""

from __future__ import annotations

import builtins
import ctypes
import math
import pickle
import types
import typing as t
from collections import OrderedDict
from collections.abc import Iterable
from itertools import product as pdt

import slowtorch
from slowtorch import function_dispatch
from slowtorch._types import BoolLikeType
from slowtorch._types import FileLike
from slowtorch._types import FloatLikeType
from slowtorch._types import IndexLike
from slowtorch._types import IntLikeType
from slowtorch._types import Scalar
from slowtorch._utils import set_module

__all__: list[str] = [
    "bool",
    "double",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "int8",
    "long",
    "short",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
]

py_impl_int = builtins.int
py_impl_bool = builtins.bool
py_impl_min = builtins.min
py_impl_max = builtins.max
py_impl_sorted = builtins.sorted
py_impl_round = builtins.round
py_impl_sum = builtins.sum


class PrinterOptions:
    """Printer options to mimic PyTorch's way."""

    precision: IntLikeType = 4


@set_module("slowtorch")
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

    __qualname__: str = "device"

    def __init__(self, type: str = "cpu", index: IntLikeType = 0) -> None:
        """Initialise a new `Device` object with default index."""
        self.type = type
        self.index = index

    def __repr__(self) -> str:
        """Return a string representation of the `Device` object."""
        return f"device(type={self.type!r}, index={self.index})"

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return f"{self.type}:{self.index}"


@set_module("slowtorch")
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
    :param typename: A `typename` similar to PyTorch tensors.
    """

    __qualname__: str = "dtype"

    def __init__(
        self,
        name: str,
        short: str,
        data: t.Any,
        value: Scalar,
        typename: str,
    ) -> None:
        """Initialise a new `Dtype` object with name and value."""
        self.name = name
        self.itemsize = int(short[-1])
        self.data = data
        self.value = value
        self.typename = typename

    def __repr__(self) -> str:
        """Return a string representation of the `Dtype` object."""
        return f"slowtorch.{self.name}"


@function_dispatch
class Size(tuple[IntLikeType, ...]):
    """Represent the shape of a tensor as a tuple of integers.

    This class extends the built-in tuple to provide a clear and
    descriptive representation for tensor dimensions.

    :param iterable: A tuple representing the dimensions of the tensor.
    """

    def __init__(self, iterable: tuple[IntLikeType, ...]) -> None:
        """Initialise a `Size` instance with some iterable."""
        if not all(isinstance(dim, int) and dim >= 0 for dim in iterable):
            raise ValueError("Dimensions must be non-negative")
        self.iterable = iterable

    def __repr__(self) -> str:
        """Return a string representation of the `Size` object."""
        return f"slowtorch.{type(self).__qualname__}({list(self.iterable)})"

    def numel(self) -> IntLikeType:
        """Return total number of elements a tensor would contain."""
        return numel(self.iterable)


DeviceType: t.TypeAlias = str | IntLikeType | Device
ShapeType: t.TypeAlias = Size | tuple[IntLikeType, ...]
StrideType: t.TypeAlias = list[IntLikeType] | tuple[IntLikeType, ...]
Dim: t.TypeAlias = IntLikeType

supported_dtypes: tuple[Dtype, ...] = (
    (bool := Dtype("bool", "b1", ctypes.c_bool, False, "BoolTensor")),
    (int8 := Dtype("int8", "i1", ctypes.c_int8, 0, "CharTensor")),
    (uint8 := Dtype("uint8", "u1", ctypes.c_uint8, 0, "ByteTensor")),
    (int16 := Dtype("int16", "i2", ctypes.c_int16, 0, "ShortTensor")),
    (uint16 := Dtype("uint16", "u2", ctypes.c_uint16, 0, "UShortTensor")),
    (int32 := Dtype("int32", "i4", ctypes.c_int32, 0, "IntTensor")),
    (uint32 := Dtype("uint32", "u4", ctypes.c_uint32, 0, "UIntTensor")),
    (int64 := Dtype("int64", "i8", ctypes.c_int64, 0, "LongTensor")),
    (uint64 := Dtype("uint64", "u8", ctypes.c_uint64, 0, "ULongTensor")),
    (float32 := Dtype("float32", "f4", ctypes.c_float, 0.0, "FloatTensor")),
    (float64 := Dtype("float64", "f8", ctypes.c_double, 0.0, "DoubleTensor")),
)

double = float64
short = int16
long = int64

for dtype in supported_dtypes:
    globals()[dtype] = dtype


@function_dispatch
def set_printoptions(precision: None | IntLikeType = None) -> None:
    """Set options for printing."""
    from slowtorch._tensor import Tensor

    if precision is None:
        precision = 4
    Tensor._print_opts.precision = precision


@function_dispatch
def infer_size_shapes(data: Input) -> Size:
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
    return Size(tuple(shape))


@function_dispatch
def make_contiguous_strides(shape: ShapeType, itemsize: IntLikeType) -> Size:
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
    return Size(tuple(reversed(strides)))


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
def broadcast_shapes(input: Size, other: Size) -> ShapeType:
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


@set_module("slowtorch.autograd")
class Node:
    """Represent a node in the computational graph for gradient
    computation.

    A `Node` encapsulates the gradient function (`backward`) and its
    associated input tensors. It is responsible for executing the
    gradient computation during backpropagation.

    :param backward: A callable function that computes the gradient for
        this node, defaults to `None`.
    :var inputs: Tuple of input tensors connected to this node in the
        graph, defaults to `()`.
    """

    inputs: tuple[Tensor] = ()

    def __init__(self, backward: None | t.Callable[..., t.Any] = None) -> None:
        """Initialise a `Node` instance."""
        self.backward = backward

    def __call__(self) -> None:
        """Execute the gradient function for this node, if defined.

        This method propagates the gradient through the inputs of the
        node by invoking the `backward`.
        """
        if self.backward:
            self.backward()

    def name(self) -> str:
        """Return the name."""
        return self.backward.__name__ if self.backward else ""


PRINT_OPTS = PrinterOptions()


@set_module("slowtorch")
@function_dispatch
class Tensor:
    """Represent a multi-dimensional tensor object.

    `Tensor` is the core data structure in SlowTorch, supporting
    multi-dimensional data storage and computation. Tensors are defined
    by their shape, data type, device, and additional metadata like
    strides and offsets for memory layout.

    :param shape: The dimensions of the tensor.
    :param dtype: The data type of the tensor elements, defaults to
        `float32`.
    :param device: The device on which the tensor resides, defaults to
        `None`.
    :param requires_grad: Whether gradients should be tracked for this
        tensor, defaults to `False`.
    :param storage:  An optional storage buffer to use for tensor data,
        defaults to `None`.
    :param offset: The starting position within the data storage buffer,
        defaults to 0.
    :param strides: Memory strides for each dimension, defaults
        to `None`.
    :raises RuntimeError: If an unsupported device is specified.
    :raises ValueError: If invalid strides or offsets are provided.
    """

    _print_opts = PRINT_OPTS
    __slots__ = (
        "_base",
        "_cached_sizes_strides_offsets",
        "_dtype",
        "_itemsize",
        "_shape",
        "_storage_offset",
        "_strides",
        "data",
        "device",
        "grad",
        "grad_fn",
        "requires_grad",
        "storage",
    )

    def __init__(
        self,
        shape: ShapeType | IntLikeType,
        dtype: None | Dtype = float32,
        device: None | DeviceType = None,
        requires_grad: BoolLikeType = False,
        storage: None | str | ctypes._CData | Tensor = None,
        offset: IntLikeType = 0,
        strides: None | StrideType = None,
    ) -> None:
        """Initialise a `tensor` object from provided shape."""
        if device is not None and device.type != "cpu":
            raise RuntimeError(
                f"{type(self).__qualname__} supports only device type: cpu"
            )
        self.device = device or Device()
        self.requires_grad = requires_grad
        if not isinstance(shape, Iterable):
            shape = (shape,)
        self._shape = tuple(py_impl_int(dim) for dim in shape)
        if dtype is None:
            dtype = float32
        elif isinstance(dtype, type):
            dtype = globals()[
                f"{dtype.__name__}{'32' if dtype == float else ''}"
            ]
        else:
            dtype = globals()[dtype]
        self._dtype = dtype
        self._itemsize = self._dtype.itemsize
        self._storage_offset = offset
        if storage is None:
            self._base = None
            if self._storage_offset != 0:
                raise ValueError("Offset must be 0 when storage is None")
            if strides is not None:
                raise ValueError("Strides must be None when storage is None")
            self._strides = make_contiguous_strides(
                self._shape, self._itemsize
            )
        else:
            if isinstance(storage, Tensor) and storage.base is not None:
                storage = storage.base
            self._base = storage
            if isinstance(storage, Tensor):
                storage = storage.buffer
            if self._storage_offset < 0:
                raise ValueError("Offset must be non-negative")
            if strides is None:
                strides = make_contiguous_strides(self._shape, self._itemsize)
            elif not (
                isinstance(strides, tuple)
                and all(isinstance(stride, py_impl_int) for stride in strides)
                and len(strides) == len(self._shape)
            ):
                raise ValueError("Invalid strides provided")
            self._strides = tuple(strides)
        storage_offset = self._strides[0] * self._shape[0] // self._itemsize
        storage_offset += self._storage_offset
        Storage = self._dtype.data * storage_offset
        if storage is None:
            if not isinstance(Storage, str):
                self.storage = Storage()
        elif isinstance(storage, ctypes.Array):
            self.storage = Storage.from_address(ctypes.addressof(storage))
        else:
            self.storage = Storage.from_buffer(storage)
        self.data = self
        self.grad_fn: Node = Node()
        self.grad: Tensor = None

    def format_repr(
        self,
        formatted: str,
        dimension: IntLikeType,
        storage_offset: IntLikeType,
        padding: IntLikeType = 0,
        whitespace: IntLikeType = 0,
        is_scalar: BoolLikeType = False,
        precision: IntLikeType = 4,
    ) -> str:
        """Method to mimic PyTorch's tensor as close as possible."""
        if is_scalar:
            value = self.storage[storage_offset]
            if isinstance(value, float):
                element = f"{value:.{precision}f}".rstrip("0").rstrip(".")
                if "." not in element:
                    element += "."
            else:
                element = str(value)
            return element.rjust(whitespace)
        indent = py_impl_min(2, py_impl_max(0, (self.ndim - dimension - 1)))
        if dimension < len(self._shape):
            formatted += "["
            for idx in range(self._shape[dimension]):
                if idx > 0:
                    formatted += (
                        "\n " + " " * padding + " " * dimension
                    ) * indent
                current = (
                    storage_offset
                    + idx * self._strides[dimension] // self._itemsize
                )
                formatted = self.format_repr(
                    formatted,
                    dimension + 1,
                    current,
                    padding,
                    whitespace,
                    False,
                    precision,
                )
                if idx < self._shape[dimension] - 1:
                    formatted += ", "
            formatted += "]"
        else:
            value = self.storage[storage_offset]
            if isinstance(value, float):
                element = f"{value:.{precision}f}".rstrip("0").rstrip(".")
                if "." not in element:
                    element += "."
            else:
                element = str(value)
            formatted += element.rjust(whitespace)
        return formatted

    def __repr__(self) -> str:
        """Return a string representation of `Tensor` object."""
        precision = getattr(self._print_opts, "precision", 4)

        def fmt_data(data: t.Any) -> str:
            if isinstance(data, float):
                out = f"{data:.{precision}f}".rstrip("0").rstrip(".")
                return out + "." if "." not in out else out
            return str(data)

        whitespace = 0
        if self.storage:
            whitespace = py_impl_max(
                len(fmt_data(value)) for value in self.storage
            )
        is_scalar = len(self.storage) == 1
        formatted = self.format_repr(
            formatted="",
            dimension=0,
            storage_offset=self._storage_offset,
            padding=7,
            whitespace=whitespace,
            is_scalar=is_scalar,
            precision=precision,
        )
        extra_repr = ""
        if self.requires_grad:
            try:
                if self.grad_fn.name():
                    extra_repr = f", grad_fn=<{self.grad_fn.name()}>"
                else:
                    extra_repr = ", requires_grad=True"
            except AttributeError:
                extra_repr = ", requires_grad=True"
        if self.dtype not in (float32, float64, int64, bool):
            return f"tensor({formatted}, dtype={self.dtype}{extra_repr})"
        return f"tensor({formatted}{extra_repr})"

    def __float__(self) -> FloatLikeType:
        """Convert the tensor to a scalar float if it has exactly one
        element.

        This method attempts to convert a tensor instance to a scalar
        float. The conversion is only possible if the tensor contains
        exactly one element.

        :raises TypeError: If tensor is not of size 1.
        """
        if self.nelement() == 1:
            return float(self.storage[self._storage_offset])
        else:
            raise TypeError("Only tensor of size 1 can be converted to scalar")

    def __int__(self) -> IntLikeType:
        """Convert the tensor to a scalar int if it has exactly one
        element.

        This method attempts to convert a tensor instance to a scalar
        int. The conversion is only possible if the tensor contains
        exactly one element.

        :raises TypeError: If tensor is not of size 1.
        """
        if self.nelement() == 1:
            return py_impl_int(self.storage[self._storage_offset])
        else:
            raise TypeError("Only tensor of size 1 can be converted to scalar")

    def __bool__(self) -> BoolLikeType:
        """Convert the tensor to a scalar bool if it has exactly one
        element.

        This method attempts to convert a tensor instance to a scalar
        bool. The conversion is only possible if the tensor contains
        exactly one element.

        :raises TypeError: If tensor is not of size 1.
        """
        if self.nelement() == 1:
            return py_impl_bool(self.storage[self._storage_offset])
        else:
            raise TypeError("Only tensor of size 1 can be converted to scalar")

    def __len__(self) -> IntLikeType:
        """Return the size of the first dimension of the tensor.

        This implements the behavior of `len()` for the tensor object,
        providing the number of elements in the first axis.

        :return: Size of the first dimension.
        :raises IndexError: If the tensor has no dimensions.
        """
        if not self._shape:
            raise IndexError("Tensor has no dimensions")
        return self._shape[0]

    def __iter__(self) -> t.Generator[Scalar]:
        """Flatten the tensor and yield its elements one by one.

        This property allows you to iterate over all elements in the
        tensor, regardless of its shape or dimensionality, in a flattened
        order. It yields the elements one by one, similar to Python's
        built-in `iter()` function, and handles both contiguous and
        non-contiguous memory layouts.

        :yield: The elements of the tensor in row-major (C-style)
            order.
        """
        itemsize = self._itemsize
        strides = self._strides
        shape = self._shape
        storage_offset = self._storage_offset
        storage = self.storage
        if not shape:
            yield storage[storage_offset]
            return
        stride_units = [stride // itemsize for stride in strides]
        for index in pdt(*[range(dim) for dim in shape]):
            offset = storage_offset + py_impl_sum(
                idx * sdx for idx, sdx in zip(index, stride_units)
            )
            yield storage[offset]

    def __getitem__(self, indices: IndexLike | Tensor) -> t.Any | Tensor:
        """Retrieve a scalar or a sub-tensor based on the specified index
        or slice.

        This method provides support for advanced indexing, including
        single indices, slices, or tuples of indices and slices,
        allowing flexible access to tensor elements or sub-tensors. For
        scalar values, a single element is returned. For subarrays, a
        new tensor object is created and returned.

        :param indices: Index or slice object, or tuple of them.
        :return: Scalar or sub-array as per the indexing operation.
        :raises IndexError: For invalid indexing and unsupported indices
            types.
        """
        if isinstance(indices, Tensor):
            if indices.dtype != slowtorch.int64:
                raise IndexError(
                    "tensors used as indices must be long or int tensors"
                )
            shape = indices._shape
            new_tensor = Tensor(
                shape + self._shape[1:],
                dtype=self.dtype,
                requires_grad=self.requires_grad,
            )
            for index, element in enumerate(indices.storage):
                dimensions = unravel_index(index, shape)
                if len(self._shape) == 1:
                    new_tensor[dimensions] = self[element]
                else:
                    for dim in range(self._shape[1]):
                        new_tensor[dimensions + (dim,)] = self[element][dim]
            return new_tensor
        if isinstance(indices, list):
            indices = tuple(indices)
        if not isinstance(indices, tuple):
            indices = (indices,)
        for index in range(indices.count(Ellipsis)):
            pre = indices[:index]
            post = indices[index + 1 :]
            count = len(self._shape) - len(pre) - len(post)
            if count < 0:
                raise IndexError("Too many indices for tensor")
            indices = pre + (slice(None),) * count + post
        if len(indices) < len(self._shape):
            indices += (slice(None),) * (len(self._shape) - len(indices))
        if any(isinstance(kdx, py_impl_bool) for kdx in indices):
            indices = tuple(
                (
                    py_impl_int(index)
                    if isinstance(index, py_impl_bool)
                    else index
                )
                for index in indices
            )
        size, strides, storage_offset = (
            self.compute_sizes_strides_storage_offset(indices)
        )
        if all(dim == 1 for dim in size):
            return self.storage[storage_offset]
        return Tensor(
            shape=size,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            storage=self,
            offset=storage_offset,
            strides=strides,
        )

    def __setitem__(self, indices: IndexLike, value: t.Any) -> None:
        """Assign a value to a specific element or subarray within the
        tensor.

        This method supports element-wise or block-wise assignment using
        indexing or slicing. The assigned value can be a scalar, a
        sequence (e.g., list or tuple), or another tensor. If assigning
        to a subarray, the value must match the shape of the subarray.

        :param indices: Index or slice to identify the element or
            subarray to update.
        :param value: The value to assign to the selected element or
            subarray.
        :raises ValueError: If the number of element in the value does
            not match the size of selected subarray.

        .. note::

            The value can be a single scalar (float or int), a list, or a
            tuple, but must match the shape and size of the subarray
            being updated.
        """
        if indices == Ellipsis:
            indices = ()
        if not isinstance(indices, tuple):
            indices = (indices,)
        size, strides, storage_offset = (
            self.compute_sizes_strides_storage_offset(indices)
        )
        nelement = numel(size)
        if not size:
            self.storage[storage_offset] = value
            return
        if isinstance(value, Scalar):
            source = [value] * nelement
        elif isinstance(value, Iterable) and not isinstance(value, Tensor):
            source = list(value)
        else:
            if not isinstance(value, Tensor):
                value = Tensor(
                    shape=value,
                    dtype=self.dtype,
                    device=self.device,
                    requires_grad=self.requires_grad,
                )
            source = list(value)
        if nelement != len(source):
            raise ValueError(
                "Scalar of elements in the value doesn't match the size"
            )
        current_offset = 1
        for stride in reversed(strides):
            if stride != 0:
                current_offset = stride // self.itemsize
                break
        if current_offset == 1:
            layout = [
                (
                    py_impl_int(element)
                    if not self.dtype.name.startswith(("float", "bool"))
                    else py_impl_round(element, self._print_opts.precision)
                )
                for element in source
            ]
            self.storage[
                slice(
                    storage_offset,
                    storage_offset + nelement * current_offset,
                    current_offset,
                )
            ] = layout
        else:
            self.set_(source, storage_offset, size, strides)

    def compute_sizes_strides_storage_offset(
        self,
        key: IndexLike,
    ) -> tuple[ShapeType, StrideType, IntLikeType]:
        """Compute shape (size), strides, and storage offset for
        indexing operation.

        This helper method computes the tensor metadata required for
        retrieving a sub-array or value based on the provided key.
        It handles `Tensor`, integers, slices, `Ellipsis`, and `None`
        indexing.

        :param key: Indexing specification (IntLikeType, slice, tuple, etc.).
        :return: Tuple of (size, strides, storage offset).
        :raises IndexError: For invalid axis indexing or bounds errors.
        :raises TypeError: For unsupported key types.
        """
        if isinstance(key, list):
            key = tuple(key)
        if not hasattr(self, "_cached_sizes_strides_offsets"):
            self._cached_sizes_strides_offsets: dict[
                tuple[IntLikeType, str],
                tuple[ShapeType, StrideType, IntLikeType],
            ] = {}
        cache = (id(self), repr(key))
        if cache in self._cached_sizes_strides_offsets:
            return self._cached_sizes_strides_offsets[cache]
        axis: IntLikeType = 0
        sizes: ShapeType = []
        strides: StrideType = []
        storage_offset: IntLikeType = self._storage_offset
        for dimension in key:
            if axis >= len(self._shape) and dimension is not None:
                raise IndexError("Too many indices for tensor")
            axissize = self._shape[axis] if axis < len(self._shape) else None
            if isinstance(dimension, int) and axissize is not None:
                if not (-axissize <= dimension < axissize):
                    raise IndexError(
                        f"Index {dimension} of tensor is out of bounds for "
                        f"dimension {axis}"
                    )
                dimension = (
                    dimension + axissize if dimension < 0 else dimension
                )
                storage_offset += (
                    dimension * self._strides[axis] // self.itemsize
                )
                axis += 1
            elif isinstance(dimension, slice) and axissize is not None:
                start, stop, step = dimension.indices(axissize)
                sizes.append(-(-(stop - start) // step) if step != 0 else 0)
                strides.append(step * self._strides[axis])
                storage_offset += start * self._strides[axis] // self.itemsize
                axis += 1
            elif dimension is None:
                sizes.append(1)
                strides.append(0)
            else:
                raise TypeError(
                    f"Invalid index type: {type(dimension).__name__!r}"
                )
        sizes.extend(self._shape[axis:])
        strides.extend(self._strides[axis:])
        strided = (tuple(sizes), tuple(strides), storage_offset)
        self._cached_sizes_strides_offsets[cache] = strided
        return strided

    def __add__(self, other: Input) -> Tensor:
        """Perform element-wise addition of the tensor with a scalar or
        another tensor.

        This method supports addition with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for addition. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            addition.
        """
        return slowtorch.nn.functional.add(self, other)

    def __radd__(self, other: Input) -> Tensor:
        """Perform reverse addition, delegating to `__add__`.

        :param other: The left-hand operand.
        :return: The result of the addition.
        """
        return self.__add__(other)

    def __sub__(self, other: Input) -> Tensor:
        """Perform element-wise subtraction of the tensor with a scalar
        or another tensor.

        This method supports subtraction with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for subtraction. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            subtraction.
        """
        return slowtorch.nn.functional.sub(self, other)

    def __rsub__(self, other: Input) -> Tensor:
        """Perform reverse subtraction, delegating to `__sub__`.

        :param other: The left-hand operand.
        :return: The result of the subtraction.
        """
        return self.__sub__(other)

    def __mul__(self, other: Input) -> Tensor:
        """Perform element-wise multiplication of the tensor with a
        scalar or another tensor.

        This method supports multiplication with scalars (int or float)
        and other tensors of the same shape. The resulting tensor is of
        the same shape and dtype as the input.

        :param other: The operand for multiplication. Can be a scalar or
            an tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            multiplication.
        """
        return slowtorch.nn.functional.mul(self, other)

    def __rmul__(self, other: Input) -> Tensor:
        """Perform reverse multiplication, delegating to `__mul__`.

        :param other: The left-hand operand.
        :return: The result of the multiplication.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Input) -> Tensor:
        """Perform element-wise division of the tensor with a scalar or
        another tensor.

        This method supports division with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for division. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            division.
        """
        return slowtorch.nn.functional.div(self, other)

    def __rtruediv__(self, other: Input) -> Tensor:
        """Perform element-wise right-hand division of the tensor with a
        scalar or another tensor.

        This method supports division with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for division. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            division.
        """
        return slowtorch.nn.functional.div(other, self)

    def __floordiv__(self, other: Input) -> Tensor:
        """Perform element-wise division of the tensor with a scalar or
        another tensor.

        This method supports division with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for division. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            division.
        """
        return slowtorch.nn.functional.div(self, other, rounding_mode="floor")

    def __matmul__(self, other: Tensor) -> Tensor:
        """Perform element-wise matrix multiplication of the tensor with
        another tensor.

        This method supports matrix multiplication with other tensors of
        the same shape. The resulting tensor is of the same shape and
        dtype as the input.

        :param other: The operand for matrix multiplication.
        :return: A new tensor containing the result of the element-wise
            matrix multiplication.
        :raises TypeError: If `other` is not a tensor.
        :raises ValueError: If `other` is a tensor but its shape doesn't
            match `self.shape`.
        """
        return slowtorch.nn.functional.matmul(self, other)

    def __mod__(self, other: Input) -> Tensor:
        """Perform element-wise modulo operation of the tensor with a
        scalar or another tensor.

        This method supports modulo operation with scalars (int or float)
        and other tensors of the same shape. The resulting tensor is of
        the same shape and dtype as the input.

        :param other: The operand for modulo operation. Can be a scalar
            or a tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            modulo operation.
        """
        return slowtorch.nn.functional.remainder(self, other)

    def __pow__(self, other: Input) -> Tensor:
        """Perform element-wise exponentiation of the tensor with a
        scalar or another tensor.

        This method supports exponentiation with scalars (int or float)
        and other tensors of the same shape. The resulting tensor is of
        the same shape and dtype as the input.

        :param other: The operand for exponentiation. Can be a scalar or
            a tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            exponentiation.
        """
        return slowtorch.nn.functional.pow(self, other)

    def __rpow__(self, other: Input) -> Tensor:
        """Perform reverse exponentiation, delegating to `__pow__`.

        :param other: The left-hand operand.
        :return: The result of the exponentiation.
        """
        return self.__pow__(other)

    def __neg__(self) -> Tensor:
        """Perform negation on the input tensor.

        :return: The result of the negation.
        """
        return slowtorch.nn.functional.neg(self)

    def __abs__(self) -> Tensor:
        """Perform element-wise absolute value conversion of the tensor.

        This method performs absolute value conversion. The resulting
        tensor is of the same shape and dtype as the input.

        :return: A new tensor containing the result of the element-wise
            absolute value conversion.
        """
        return slowtorch.nn.functional.abs(self)

    def __lt__(self, other: Input) -> Tensor:
        """Perform element-wise less-than operation of the tensor with a
        scalar or another tensor.

        This method supports comparison with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of
        the same shape and dtype as the input.

        :param other: The operand for comparison. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            comparison.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        new_tensor = Tensor(self._shape, bool)
        if isinstance(other, Scalar):
            new_tensor[:] = (x < other for x in self.storage)
        elif isinstance(other, Tensor):
            if self._shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self._shape} {other.shape}"
                )
            new_tensor[:] = (x < y for x, y in zip(self, other))
        else:
            raise TypeError(
                f"Unsupported operand type(s) for <: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return new_tensor

    def __gt__(self, other: Input) -> Tensor:
        """Perform element-wise greater-than operation of the tensor
        with a scalar or another tensor.

        This method supports comparison with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of
        the same shape and dtype as the input.

        :param other: The operand for comparison. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            comparison.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        new_tensor = Tensor(self._shape, bool)
        if isinstance(other, Scalar):
            new_tensor[:] = (x > other for x in self.storage)
        elif isinstance(other, Tensor):
            if self._shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self._shape} {other.shape}"
                )
            new_tensor[:] = (x > y for x, y in zip(self, other))
        else:
            raise TypeError(
                f"Unsupported operand type(s) for >: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return new_tensor

    def __le__(self, other: Input) -> Tensor:
        """Perform element-wise less-than-equal operation of the tensor
        with a scalar or another tensor.

        This method supports comparison with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of
        the same shape and dtype as the input.

        :param other: The operand for comparison. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            comparison.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        new_tensor = Tensor(self._shape, bool)
        if isinstance(other, Scalar):
            new_tensor[:] = (x <= other for x in self.storage)
        elif isinstance(other, Tensor):
            if self._shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self._shape} {other.shape}"
                )
            new_tensor[:] = (x <= y for x, y in zip(self, other))
        else:
            raise TypeError(
                f"Unsupported operand type(s) for <=: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return new_tensor

    def __ge__(self, other: Input) -> Tensor:
        """Perform element-wise greater-than-equal operation of the
        tensor with a scalar or another tensor.

        This method supports comparison with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of
        the same shape and dtype as the input.

        :param other: The operand for comparison. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            comparison.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        new_tensor = Tensor(self._shape, bool)
        if isinstance(other, Scalar):
            new_tensor[:] = (x >= other for x in self.storage)
        elif isinstance(other, Tensor):
            if self._shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self._shape} {other.shape}"
                )
            new_tensor[:] = (x >= y for x, y in zip(self, other))
        else:
            raise TypeError(
                f"Unsupported operand type(s) for >=: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )
        return new_tensor

    def set_(
        self,
        source: StorageWeakRef,
        storage_offset: IntLikeType,
        size: ShapeType,
        strides: StrideType,
        index: IntLikeType = 0,
    ) -> IntLikeType:
        """Sets the underlying storage, size, and strides."""
        if len(size) == 1:
            offset = strides[0] // self.itemsize
            for dimension in range(size[0]):
                element = source[index + dimension]
                self.storage[storage_offset + dimension * offset] = element
            return index + size[0]
        else:
            stride = strides[0] // self.itemsize
            for dimension in range(size[0]):
                index = self.set_(
                    source,
                    storage_offset + dimension * stride,
                    size[1:],
                    strides[1:],
                    index,
                )
            return index

    def broadcast_to(self, size: Size) -> Tensor:
        """Broadcast the tensor to the target shape."""
        shape = self._shape
        if shape == size:
            return self
        inner = len(shape)
        outer = len(size)
        if outer < inner or (outer - inner < 0):
            raise ValueError(f"Cannot broadcast {shape} to {size}")
        padded = (1,) * (outer - inner) + shape
        for dimension, (input, target) in enumerate(zip(padded, size)):
            if input != target and input != 1:
                raise ValueError(
                    f"Cannot broadcast {shape} to {size} at dimension "
                    f"{dimension}"
                )
        storage: StorageWeakRef = []
        for index in pdt(*(range(dim) for dim in size)):
            indices = tuple(0 if p == 1 else j for j, p in zip(index, padded))
            storage_offset = sum(
                bdx * sdx for bdx, sdx in zip(indices, self.stride())
            )
            storage.append(self.storage[storage_offset])
        new_tensor = Tensor(size, self.dtype, requires_grad=self.requires_grad)
        new_tensor.storage = storage
        return new_tensor

    def backward(
        self,
        gradient: None | Tensor = None,
        inputs: None | tuple[Tensor, ...] = None,
        retain_graph: BoolLikeType = False,
    ) -> None:
        """Compute the gradient of current tensor w.r.t graph leaves.

        :param gradient: The gradient of the output with respect to the
            tensor, defaults to `None`.
        :param inputs: Tuple of inputs w.r.t which the gradients will
            be accumulated into `.grad`.
        :param retain_graph: Whether to retain the computation graph
            after backward.
        :raises RuntimeError: If the tensor does not require gradients.
        """
        if not self.requires_grad:
            raise RuntimeError(
                "Tensors does not require grad and does not have a grad_fn"
            )
        graph: list[TensorOrTensors] = []
        seen: set[TensorOrTensors] = set()
        if gradient is None:
            gradient = Tensor(1, float32)
            gradient[:] = 1.0
        self.grad = gradient

        def iter_graph(inputs: TensorOrTensors) -> None:
            """Recursive function to traverse the computation graph."""
            if isinstance(inputs, Tensor) and inputs not in seen:
                seen.add(inputs)
                if hasattr(inputs.grad_fn, "inputs"):
                    for input in inputs.grad_fn.inputs:
                        iter_graph(input)
                graph.append(inputs)

        iter_graph(inputs if inputs else self)
        for node in reversed(graph):
            if node.grad_fn is not None and callable(node.grad_fn):
                node.grad_fn()
        self.grad = None
        if not retain_graph:
            self.grad_fn = None

    def render(self, show_dtype: BoolLikeType = False) -> None:
        """Render the backward computation graph of the tensor as an
        ASCII tree.

        This method recursively traverses the autograd graph in
        post-order, assigns a unique identifier to each node, and prints
        out an aligned, human-readable tree with tensor's metadata and
        nodes.

        Leaf tensors are denoted explicitly, and previously visited
        nodes are marked to indicate reuse in shared computation paths.

        :param show_dtype: Flag to display dtype, defaults to `False`.

        .. note::

            - Nodes are uniquely numbered using post-order traversal,
              ensuring a consistent inside-out visual structure.
        """
        seen: set[IntLikeType] = set()
        shown: set[IntLikeType] = set()
        counter: list[IntLikeType] = [1]
        Id: t.TypeAlias = list[IntLikeType] | IntLikeType
        tensors: OrderedDict[IntLikeType, Id] = OrderedDict()

        def set_id(input: Tensor) -> None:
            """Set unique ID to each node using post-order traversal."""
            if input is None or id(input) in seen:
                return
            seen.add(id(input))
            if (
                hasattr(input, "grad_fn")
                and input.grad_fn
                and hasattr(input.grad_fn, "inputs")
            ):
                for _input in input.grad_fn.inputs:
                    set_id(_input)
            tensors[id(input)] = counter[0]
            counter[0] += 1

        set_id(self)

        def get_id(input: Tensor) -> Id:
            """Return ID for a tensor, or -1 if unregistered."""
            return tensors.get(id(input), -1)

        def is_leaf(input: Tensor) -> BoolLikeType:
            """Determine if a tensor is a leaf in the autograd graph."""
            return not (
                hasattr(input, "grad_fn")
                and input.grad_fn
                and hasattr(input.grad_fn, "inputs")
                and input.grad_fn.inputs
            )

        def iter_graph(
            input: Tensor,
            indent: str = "",
            is_last: BoolLikeType = True,
            is_root: BoolLikeType = True,
        ) -> None:
            """
            Recursively render the computation graph as an ASCII tree.

            :param input: Tensor for which the backward graph is to be
                visualised.
            :param indent: indent string used for indentation and
                continuation lines, defaults to ``.
            :param is_last: Flag indicating whether this node is the
                last among its siblings, defaults to `True`.
            :param is_root: Flag indicating if this node is the root of
                the tree, defaults to `True`.
            """
            if input is None:
                return
            branch = "└──► " if is_last or is_leaf(input) else "├──► "
            if not is_last and get_id(input) == 1:
                branch = "├──► "
            prefix = indent + branch
            if is_root:
                prefix = ""
            dtype = f", dtype={input.dtype}" if show_dtype else ""
            shape = getattr(input, "_shape", "?")
            label = f"Tensor.{get_id(input)}(shape={shape}{dtype})"
            if id(input) in shown:
                print(prefix + label + " [already seen]")
                return
            else:
                print(prefix + label)
            shown.add(id(input))
            if not is_leaf(input):
                outer = indent + ("     " if is_last else "│    ")
                print(outer + f"{input.grad_fn.name()}")
                inputs = input.grad_fn.inputs
                for idx, _input in enumerate(inputs):
                    is_last_input = idx == len(inputs) - 1
                    inner = indent + ("     " if is_last else "│    ")
                    iter_graph(_input, inner, is_last_input, is_root=False)

        iter_graph(self)
        print()

    @property
    def buffer(self) -> t.Any:
        """Return the memory buffer holding the tensor elements."""
        return self.storage

    @property
    def base(self) -> t.Any:
        """Return underlying buffer (if any)."""
        return self._base

    @property
    def dtype(self) -> Dtype:
        """Return the data type of the tensor elements (mainly str)."""
        return self._dtype

    @property
    def is_cuda(self) -> t.Literal[False]:
        """Return True if tensor is stored on GPU else False."""
        return False

    @property
    def is_quantized(self) -> t.Literal[False]:
        """Return True if tensor is quantised else False."""
        return False

    @property
    def is_meta(self) -> t.Literal[False]:
        """Return True if tensor is a meta tensor else False."""
        return False

    def dim(self) -> IntLikeType:
        """Return the number of dimensions of the tensor."""
        return len(self._shape)

    ndim = property(dim)

    @property
    def nbytes(self) -> IntLikeType:
        """Return number of byte size of a tensor."""
        return self.nelement() * self.element_size()

    def element_size(self) -> IntLikeType:
        """Return the size, in bytes, of each tensor element."""
        return py_impl_int(self._itemsize)

    itemsize = property(element_size)

    @property
    def shape(self) -> Size:
        """Return shape of the tensor."""
        return Size(self._shape)

    @shape.setter
    def shape(self, value: Size) -> None:
        """Set a new shape for the tensor."""
        if value == self._shape:
            return
        if self.nelement() != numel(value):
            raise ValueError("New shape is incompatible with the current size")
        if get_step(self) == 1:
            self._shape = value
            self._strides = make_contiguous_strides(self._shape, self.itemsize)
            return
        shape = [dim for dim in self._shape if dim > 1]
        strides = [
            stride
            for dim, stride in zip(self._shape, self._strides)
            if dim > 1
        ]
        new_shape = [dim for dim in value if dim > 1]
        if new_shape != shape:
            raise AttributeError(
                "New shape is incompatible with the current memory layout"
            )
        new_strides: StrideType = []
        shape.append(1)
        strides.append(strides[-1])
        idx = len(shape) - 1
        for dim in reversed(value):
            if dim == 1:
                new_strides.append(strides[idx] * shape[idx])
            else:
                idx -= 1
                new_strides.append(strides[idx])
        if idx != -1:
            raise AttributeError(
                "New shape is incompatible with the current memory layout"
            )
        self._shape = value
        self._strides = tuple(reversed(new_strides))

    def flat(self) -> list[Scalar]:
        """Flatten the tensor and return all its elements in a list.

        This method traverses through the tensor and collects its
        elements into a single list, regardless of its shape or
        dimensionality. It handles contiguous memory layouts and
        non-contiguous slices, ensuring that all elements of the tensor
        are included in the returned list.

        :return: A list containing all elements in the tensor.
        """
        return list(self)

    def nelement(self) -> IntLikeType:
        """Return total number of elements in a tensor."""
        return numel(self._shape)

    numel = nelement

    def size(self, dim: None | Dim = None) -> ShapeType | IntLikeType:
        """Returns the size of the tensor."""
        if dim is not None:
            return self._shape[dim]
        return Size(self._shape)

    def stride(self) -> StrideType:
        """Return the strides for traversing the tensor dimensions."""
        return tuple(idx // self.itemsize for idx in self._strides)

    def to(self, dtype: Dtype) -> Tensor:
        """Return a copy of the tensor cast to a specified data type.

        This method creates a new `Tensor` with the same shape and data
        as the original tensor but cast to the specified data type. The
        original tensor remains unmodified.

        :param dtype: The desired data type for the output tensor.
        :return: A new tensor with the specified data type and the same
            shape as the original tensor.

        .. note::

            [1] This operation creates a copy of the data, even if the
                requested data type is the same as the original.
        """
        new_tensor = Tensor(
            shape=self.shape,
            dtype=dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        new_tensor[:] = self
        return new_tensor

    type = to

    def float(self) -> Tensor:
        """Return tensor with floating dtype."""
        return self.to(float32)

    float64 = float32 = half = double = float

    def int(self) -> Tensor:
        """Return tensor with integer dtype."""
        return self.to(int64)

    int64 = int32 = int16 = int8 = long = char = int

    def bool(self) -> Tensor:
        """Return tensor with bool dtype."""
        return self.to(bool)

    def _view(self) -> Tensor:
        """Create a new view of the tensor.

        This method allows creating a new tensor view. The method
        supports efficient reinterpretation of the data buffer and
        respects the shape and strides of the original tensor. For 1D
        tensors, the dtype can differ if the total number of bytes
        remains consistent.

        :return: A new tensor view with the specified dtype. Returns
            `None` if the view cannot be created.
        :raises RuntimeError: If the tensor is a scalar.
        """
        if self.ndim == 0:
            raise RuntimeError("cannot create a view of a scalar")
        return Tensor(
            shape=self._shape,
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
            storage=self,
            offset=self._storage_offset,
            strides=self._strides,
        )

    def view(self, *shape: IntLikeType) -> Tensor:
        """Return a new view of the tensor with the specified shape.

        This method attempts to reshape the tensor while keeping the
        data layout intact. If the new shape is incompatible with the
        current memory layout, a copy of the data is made to achieve the
        desired shape.

        :param shape: The desired shape for the tensor.
        :return: A reshaped view of the tensor if possible; otherwise, a
            reshaped copy.
        """
        if len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)
        if shape.count(-1) > 1:
            raise RuntimeError("only one dimension can be inferred")
        nelement = self.nelement()
        possible = numel([dim for dim in shape if dim != -1])
        if -1 in shape:
            if possible == 0 or nelement % possible != 0:
                raise RuntimeError(
                    f"shape '{shape}' is invalid for input of size {nelement}"
                )
            index = shape.index(-1)
            dimension = nelement // possible
            shape = shape[:index] + (dimension,) + shape[index + 1 :]
        if numel(shape) != nelement:
            raise RuntimeError(
                f"shape '{shape}' is invalid for input of size {nelement}"
            )
        new_tensor = self._view()
        new_tensor.shape = shape
        return new_tensor

    reshape = view

    def tolist(self) -> t.Any:
        """Convert the tensor to a nested Python list.

        This method recursively iterates over the dimensions of the
        tensor to construct a nested list that mirrors the shape and
        contents of the tensor.

        :return: A nested Python list representation of the tensor's
            data.
        """
        if self.ndim == 0:
            return self.item()

        def flatten(prefix: t.Any = ()) -> t.Any:
            dimensions = len(prefix)
            return [
                (
                    flatten(prefix + (index,))
                    if dimensions + 1 < self.ndim
                    else self[prefix + (index,)].item()
                )
                for index in range(self._shape[dimensions])
            ]

        return flatten()

    def item(self) -> t.Any:
        """Return standard scalar Python object for tensor object."""
        if self.nelement() == 1:
            return self.view(-1).storage[0]
        else:
            raise RuntimeError(
                f"Tensor with {self.nelement()} elements cannot be"
                " converted to scalar"
            )

    def ravel(self) -> Tensor:
        """Return a copy of the tensor collapsed into one dimension."""
        return slowtorch.nn.functional.ravel(self)

    flatten = ravel

    def as_strided(self, size: ShapeType, strides: StrideType) -> Tensor:
        """Create a new view of the tensor with the specified shape and
        strides.

        :param shape: The shape of the new view.
        :param strides: The strides of the new view.
        :return: A new tensor view.
        """
        new_tensor = self.__class__.__new__(self.__class__)
        new_tensor._base = self._base
        new_tensor._dtype = self._dtype
        new_tensor._itemsize = self._itemsize
        new_tensor._shape = size
        new_tensor._storage_offset = self._storage_offset
        new_tensor._strides = strides
        new_tensor.device = self.device
        new_tensor.grad = self.grad
        new_tensor.grad_fn = self.grad_fn
        new_tensor.requires_grad = self.requires_grad
        new_tensor.storage = self.storage
        return new_tensor

    def transpose(self, dim0: IntLikeType, dim1: IntLikeType) -> Tensor:
        """Transpose the tensor by permuting its dimensions.

        This method returns a view of the tensor with its dimensions
        permuted. If no dimensions are specified, the dimensions are
        reversed (i.e., equivalent to a full transpose).

        :param dim0: First dimension to be transposed.
        :param dim1: Second dimension to be transposed.
        :return: A new tensor view with transposed dimensions.
        """
        return slowtorch.nn.functional.transpose(self, dim0, dim1)

    swapaxes = swapdims = transpose

    def t(self) -> Tensor:
        """Transpose dimensions 0 and 1."""
        return self.transpose(0, 1)

    T = property(t)

    def unique(self, sorted: BoolLikeType = True) -> Tensor:
        """Return unique elements from the tensor.

        :param sorted: Whether to sort the unique elements before
            returning the output, defaults to `True`.
        :return: Tensor with list of unique elements.
        """
        storage = list(self)
        unique = set(storage)
        values = py_impl_sorted(unique) if sorted else storage
        new_tensor = Tensor(
            shape=len(values),
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        new_tensor[:] = values
        return new_tensor

    def neg(self) -> Tensor:
        """Compute negative of the elements.

        :return: Tensor with negative of the input elements.
        """
        return self.__mul__(-1)

    negative = neg

    def fill_(self, value: Scalar) -> Tensor:
        """Fill the entire tensor with a scalar value.

        This method assigns the given scalar value to all elements in
        the tensor. The operation modifies the tensor in place and
        supports both integers and floating-point numbers as input.

        :param value: The scalar value to fill the tensor with.
        :raises ValueError: If the provided `value` is not an integer
            or floating-point number.

        .. note::

            [1] This method modifies the tensor in place.
            [2] The method uses slicing (`self[:] = value`) to
                efficiently set all elements to the specified value.
        """
        if not isinstance(value, Scalar):
            raise ValueError("Value must be an integer or a float")
        self[:] = value
        return self

    def unsqueeze(self, dim: IntLikeType) -> Tensor:
        """Return a new tensor with a singleton dimension inserted at
        the specified position.

        :param dim: The position at which to insert the new singleton
            dimension.
        :return: A new tensor with the updated shape.
        :raises ValueError: If `dim` is not within the valid range.
        """
        if dim < 0:
            dim += self.ndim + 1
        if not (0 <= dim <= self.ndim):
            raise ValueError(
                f"Dimension {dim} out of range for tensor of shape "
                f"{self._shape} with {self.ndim} dimensions"
            )
        shape = self._shape[:dim] + (1,) + self._shape[dim:]
        new_tensor = Tensor(
            shape,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
        )
        new_tensor[:] = self[:]
        return new_tensor

    def add(self, other: Input, *, alpha: Scalar = 1) -> Tensor:
        """Perform element-wise addition of the tensor with a scalar or
        another tensor, scaled by alpha.

        This method supports addition with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for addition. Can be a scalar or an
            tensor of the same shape.
        :param alpha: The multiplier for other, defaults to 1.
        :return: A new tensor containing the result of the element-wise
            addition.
        """
        return self.__add__(alpha * other)

    def sub(self, other: Input, *, alpha: Scalar = 1) -> Tensor:
        """Perform element-wise subtraction of the tensor with a scalar
        or another tensor, scaled by alpha.

        This method supports subtraction with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for subtraction. Can be a scalar or an
            tensor of the same shape.
        :param alpha: The multiplier for other, defaults to 1.
        :return: A new tensor containing the result of the element-wise
            subtraction.
        """
        return self.__sub__(alpha * other)

    def mul(self, other: Input, *, alpha: Scalar = 1) -> Tensor:
        """Perform element-wise multiplication of the tensor with a
        scalar or another tensor, scaled by alpha.

        This method supports multiplication with scalars (int or float)
        and other tensors of the same shape. The resulting tensor is of
        the same shape and dtype as the input.

        :param other: The operand for multiplication. Can be a scalar or
            a tensor of the same shape.
        :param alpha: The multiplier for other, defaults to 1.
        :return: A new tensor containing the result of the element-wise
            multiplication.
        """
        return self.__mul__(alpha * other)

    def div(self, other: Input, *, rounding_mode: None | str = None) -> t.Any:
        """Perform element-wise division of the tensor with a scalar or
        another tensor.

        This method supports division with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for division. Can be a scalar or an
            tensor of the same shape.
        :param rounding_mode: Type of rounding to apply to the result,
            defaults to `None`.
        :return: A new tensor containing the result of the element-wise
            division.
        """
        if rounding_mode is not None:
            raise RuntimeError("Rounding mode is not supported")
        return self.__truediv__(other)

    true_divide = divide = div
    matmul = __matmul__
    pow = __pow__
    abs = __abs__
    less = lt = __lt__
    greater = gt = __gt__
    less_equal = le = __le__
    greater_equal = ge = __ge__

    def detach(self) -> Tensor:
        """Return a new tensor without requiring gradients."""
        new_tensor = Tensor(
            shape=self._shape,
            dtype=self._dtype,
            device=self.device,
            requires_grad=False,
            storage=self.storage,
            offset=self._storage_offset,
            strides=self._strides,
        )
        new_tensor._base = self
        return new_tensor

    def log(self) -> Tensor:
        """Return a new tensor with the natural log of the elements of
        input.

        This method creates a new `Tensor` instance with the same shape
        and as the original tensor but with natural log calculated.

        :return: A new tensor with the log calculated data and shape.
        """
        return slowtorch.nn.functional.log(self)

    def clone(self) -> Tensor:
        """Return a deep copy of the tensor.

        This method creates a new `Tensor` instance with the same data,
        shape, and type as the original tensor. The copy is independent
        of the original, meaning changes to the copy do not affect the
        original tensor.

        :return: A new tensor with the same data, shape, and type as the
            original tensor.

        .. note::

            [1] This method ensures that both the data and metadata of
                the tensor are duplicated.
            [2] The `to` method is used internally for copying, ensuring
                consistency and type fidelity.
        """
        return slowtorch.nn.functional.clone(self)

    def sum(
        self,
        dim: None | Dim = None,
        keepdim: BoolLikeType = False,
    ) -> Tensor:
        """Compute the sum of elements in the tensor across a specified
        dimension.

        This method computes the sum of all elements in the tensor if no
        dimension is provided. If a dimension is specified, the method
        reduces the tensor along the given dimension while optionally
        retaining the reduced dimensions.

        :param dim: The dimension along which to compute the sum,
            defaults to `None`. For `None`, the sum is computed over all
            elements of the tensor.
        :param keepdim: A boolean indicating whether to retain the
            reduced dimensions in the resulting tensor, defaults to
            `False`.
        :return: A new tensor containing the sum of the specified
            elements.
        """
        return slowtorch.nn.functional.sum(self, dim, keepdim)

    def max(
        self,
        dim: None | Dim = None,
        keepdim: BoolLikeType = False,
    ) -> Tensor:
        """Return the maximum of elements in the tensor across a
        specified dimension.

        This method returns the maximum of all elements in the tensor if
        no dimension is provided. If a dimension is specified, the method
        reduces the tensor along the given dimension while optionally
        retaining the reduced dimensions.

        :param dim: The dimension along which to compute the maximum,
            defaults to `None`. For `None`, the maximum is computed over
            all elements of the tensor.
        :param keepdim: A boolean indicating whether to retain the
            reduced dimensions in the resulting tensor, defaults to
            `False`.
        :return: A new tensor containing the maximum of the specified
            elements.
        """
        return slowtorch.nn.functional.max(self, dim, keepdim)

    def min(
        self,
        dim: None | Dim = None,
        keepdim: BoolLikeType = False,
    ) -> Tensor:
        """Return the minimum of elements in the tensor across a
        specified dimension.

        This method returns the minimum of all elements in the tensor if
        no dimension is provided. If a dimension is specified, the method
        reduces the tensor along the given dimension while optionally
        retaining the reduced dimensions.

        :param dim: The dimension along which to compute the minimum,
            defaults to `None`. For `None`, the minimum is computed over
            all elements of the tensor.
        :param keepdim: A boolean indicating whether to retain the
            reduced dimensions in the resulting tensor, defaults to
            `False`.
        :return: A new tensor containing the minimum of the specified
            elements.
        """
        return slowtorch.nn.functional.min(self, dim, keepdim)

    def mean(
        self,
        dim: None | Dim = None,
        keepdim: BoolLikeType = False,
    ) -> Tensor:
        """Compute the mean of elements in the tensor across a specified
        dimension.

        This method computes the mean of all elements in the tensor if
        no dimension is provided. If a dimension is specified, the method
        reduces the tensor along the given dimension while optionally
        retaining the reduced dimensions.

        :param dim: The dimension along which to compute the mean,
            defaults to `None`. For `None`, the mean is computed over
            all elements of the tensor.
        :param keepdim: A boolean indicating whether to retain the
            reduced dimensions in the resulting tensor, defaults to
            `False`.
        :return: A new tensor containing the mean of the specified
            elements.
        """
        return slowtorch.nn.functional.mean(self, dim, keepdim)

    def std(
        self,
        dim: None | Dim = None,
        keepdim: BoolLikeType = False,
    ) -> Tensor:
        """Compute the standard deviation of elements in the tensor
        across a specified dimension.

        This method computes the standard deviation of all elements in
        the tensor if no dimension is provided. If a dimension is
        specified, the method reduces the tensor along the given
        dimension while optionally retaining the reduced dimensions.

        :param dim: The dimension along which to compute the standard
            deviation, defaults to `None`. For `None`, the standard
            deviation is computed over all elements of the tensor.
        :param keepdim: A boolean indicating whether to retain the
            reduced dimensions in the resulting tensor, defaults to
            `False`.
        :return: A new tensor containing the standard deviation of the
            specified elements.
        """
        return slowtorch.nn.functional.std(self, dim, keepdim)

    def exp(self) -> Tensor:
        """Perform element-wise exponentiation of the tensor.

        This method supports exponentiation. The resulting tensor is of
        the same shape and dtype as the input. The exponentiation
        function is defined as::

            exp(x) = math.exp(x)

        :return: A new tensor containing the result of the element-wise
            exponentiation.
        """
        return slowtorch.nn.functional.exp(self)

    def sqrt(self) -> Tensor:
        """Perform element-wise square root of a tensor.

        This function computes the square root of each element in the
        input tensor. The result is returned as a new tensor, and
        gradients are properly propagated during backpropagation.

        :return: A new tensor containing the square root of each element
            in the input tensor.
        """
        return slowtorch.nn.functional.sqrt(self)

    def relu(self) -> Tensor:
        """Apply the Rectified Linear Unit (ReLU) function element-wise.

        ReLU sets all negative values in the tensor to zero and keeps
        positive values unchanged. This operation is differentiable, and
        gradients are propagated only for positive elements. The relu
        function is defined as::

            relu(x) = max(x, 0)

        :return: Output tensor after applying the ReLU function, with
            gradients linked for backpropagation.
        """
        return slowtorch.nn.functional.relu(self)

    def elu(self, alpha: FloatLikeType = 1.0) -> Tensor:
        """Apply the Exponential Linear Unit (ELU) function element-
        wise.

        ELU is a function that tend to converge cost to zero faster and
        produce more accurate results. This operation is differentiable,
        and gradients are propagated only for positive elements. The elu
        function is defined as::

            elu(x) = x if x >- 0 else alpha * (exp(x) - 1)

        :param alpha: Value for the ELU formulation, defaults to 1.0.
        :return: Output tensor after applying the ELU function, with
            gradients linked for backpropagation.
        """
        return slowtorch.nn.functional.elu(self, alpha)

    def tanh(self) -> Tensor:
        """Apply the Hyperbolic Tangent (Tanh) function element-wise.

        Tanh squashes all the values between the range of -1 to 1. This
        operation is differentiable, and gradients are propagated. The
        tanh function is defined as::

            tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

        :return: Output tensor after applying the Tanh function, with
            gradients linked for backpropagation.
        """
        return slowtorch.nn.functional.tanh(self)

    def sigmoid(self) -> Tensor:
        """Apply the Sigmoid function element-wise.

        Sigmoid function squashes between 0 and 1. This operation is
        differentiable, and gradients are propagated. The sigmoid
        function is defined as::

            sigmoid(x) = 1 / (1 + exp(-x))

        :return: Output tensor after applying the Sigmoid function, with
            gradients linked for backpropagation.
        """
        return slowtorch.nn.functional.sigmoid(self)

    def softmax(
        self,
        dim: None | Dim = None,
        dtype: None | Dtype = None,
    ) -> Tensor:
        """Apply the Softmax function element-wise.

        Softmax function squashes between 0 and 1, along the provided
        dimension and sum to 1. This operation is differentiable, and
        gradients are propagated. The softmax function is defined as::

            softmax(x) = exp(x) / exp(x).sum()

        :param dim: A dimension along which softmax will be computed,
            defaults to `None`.
        :return: Output tensor after applying the Softmax function, with
            gradients linked for backpropagation.
        """
        return slowtorch.nn.functional.softmax(self, dim=dim, dtype=dtype)

    def log_softmax(
        self,
        dim: None | Dim = None,
        dtype: None | Dtype = None,
    ) -> Tensor:
        """Apply the Softmax function followed by Logarithm.

        This is mathematically equivalent to applying softmax function
        followed by logarith. This operation is differentiable, and
        gradients are propagated.

        :param dim: A dimension along which softmax will be computed,
            defaults to `None`.
        :return: Output tensor after applying the Log softmax function,
            with gradients linked for backpropagation.
        """
        return slowtorch.nn.functional.log_softmax(self, dim=dim, dtype=dtype)


Input: t.TypeAlias = Scalar | Tensor
StorageWeakRef: t.TypeAlias = t.Sequence[Input]
TensorOrTensors: t.TypeAlias = tuple[Tensor, ...] | Tensor


@function_dispatch
def save(
    obj: object,
    f: FileLike,
    pickle_module: types.ModuleType = pickle,
    pickle_protocol: t.Literal[2] = 2,
) -> None:
    """Save an object to disk file."""
    with open(f, "wb") as opened_file:
        pickle_module.dump(obj, opened_file, protocol=pickle_protocol)


@function_dispatch
def load(
    f: FileLike,
    pickle_module: types.ModuleType = pickle,
    weights_only: None | BoolLikeType = None,
) -> t.Any:
    """Load an object saved from a file."""
    weights_only = weights_only
    with open(f, "rb") as opened_file:
        output = pickle_module.load(opened_file)
    return output


@function_dispatch
def typename(obj: t.Any) -> t.Any:
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
    if module in {"", "builtins"}:
        return qualname
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
