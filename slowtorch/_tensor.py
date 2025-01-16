"""\
SlowTorch Tensor API
====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, January 07 2025
Last updated on: Thursday, January 16 2025

Tensor object.

This module provides the foundational classes and functions for tensor
operations in SlowTorch. It defines the `Tensor` class and supporting
utility like the `tensor` function to create and manipulate tensors.
SlowTorch aims to mimic PyTorch's tensor behavior while implementing key
functionalities from scratch to foster learning and customization.

The `Tensor` class acts as the primary data structure for
multidimensional arrays and supports features like automatic
differentiation, and flexible data types. Whereas, the `tensor` function
is a factory function to create tensors from nested data structures. It
infers tensor shape, data type, and supports optional device
specification and gradient requirements. Designed with flexibility,
efficiency, and modularity in mind, the `Tensor` class aims to replicate
and innovate upon the core features of PyTorch tensors while emphasizing
a Python-standard-library-only approach. Additionally, the class
introduces a Pythonic, educational perspective, making it suitable for
learning and experimentation with tensor mechanics without relying on
external libraries.

As of now, the module supports features such as::

    - Efficient storage and representation of n-dimensional data.
    - Flexible shape manipulation, including reshaping and broadcasting.
    - Element-wise operations, including arithmetic, logical, and
      comparison operations, via rich operator overloading.
    - Slicing and indexing support for intuitive data access.
    - Conversion utilities to export data to native Python types
      (e.g., lists).

The `Tensor` implementation draws inspiration from PyTorch's
architecture but deliberately simplifies and reimagines certain aspects
for educational purposes and to meet the constraints of pure Python. By
eschewing C or Cython extensions, the `Tensor` class offers an
accessible implementation that emphasizes algorithmic clarity over raw
performance.

In addition to the core `Tensor` functionality, this module introduces
several helper functions to aid in tensor manipulation and generation.
These functions are designed to complement the `Tensor` class and mimic
key functionality found in PyTorch.

While this module implements many fundamental features of `Tensor`, it
does not aim to match PyTorch's performance or breadth. Instead, the
focus is on clarity, usability, and modularity, providing a platform for
learning and experimentation.
"""

from __future__ import annotations

import builtins
import ctypes
import typing as t
from collections.abc import Iterable

from slowtorch import function_dispatch
from slowtorch._types import ArrayLike
from slowtorch._types import Number
from slowtorch._utils import Device
from slowtorch._utils import DeviceType
from slowtorch._utils import Dtype
from slowtorch._utils import Size
from slowtorch._utils import calc_shape_from_data
from slowtorch._utils import calc_size
from slowtorch._utils import calc_strides
from slowtorch._utils import get_step
from slowtorch._utils import has_uniform_shape

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

py_min = min
py_max = max
py_sorted = sorted

supported_dtypes: tuple[Dtype, ...] = (
    (bool := Dtype("bool", "b1", ctypes.c_bool, False)),
    (int8 := Dtype("int8", "i1", ctypes.c_int8, 0)),
    (uint8 := Dtype("uint8", "u1", ctypes.c_uint8, 0)),
    (int16 := Dtype("int16", "i2", ctypes.c_int16, 0)),
    (uint16 := Dtype("uint16", "u2", ctypes.c_uint16, 0)),
    (int32 := Dtype("int32", "i4", ctypes.c_int32, 0)),
    (uint32 := Dtype("uint32", "u4", ctypes.c_uint32, 0)),
    (int64 := Dtype("int64", "i8", ctypes.c_int64, 0)),
    (uint64 := Dtype("uint64", "u8", ctypes.c_uint64, 0)),
    (float32 := Dtype("float32", "f4", ctypes.c_float, 0.0)),
    (float64 := Dtype("float64", "f8", ctypes.c_double, 0.0)),
)

double = float64
short = int16
long = int64

for dtype in supported_dtypes:
    globals()[dtype] = dtype


class PrinterOptions:
    """Printer options to mimic PyTorch's way."""

    precision: int = 4


PRINT_OPTS = PrinterOptions()


@function_dispatch
def set_printoptions(precision: None | int = None) -> None:
    """Set options for printing."""
    from slowtorch._tensor import Tensor

    if precision is None:
        precision = 4
    Tensor._print_opts.precision = precision


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
        """Initialize a `Node` instance."""
        self.backward = backward
        self.name = backward.__name__ if backward else None

    def __call__(self) -> None:
        """Execute the gradient function for this node, if defined.

        This method propagates the gradient through the inputs of the
        node by invoking the `backward`.
        """
        if self.backward:
            self.backward()


@function_dispatch
class Tensor:
    """Represent a multi-dimensional tensor object.

    A `Tensor` is the core data structure in SlowTorch, supporting
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
    :param buffer:  An optional buffer to use for tensor data, defaults
        to `None`.
    :param offset: The starting position within the data buffer,
        defaults to 0.
    :param strides: Memory strides for each dimension, defaults
        to `None`.
    :raises RuntimeError: If an unsupported device is specified.
    :raises TypeError: If the shape is not an iterable of integers.
    :raises ValueError: If invalid strides or offsets are provided.
    """

    __module__: str = "slowtorch"
    __slots__: tuple[str, ...] = (
        "_base",
        "_cdata",
        "_dtype",
        "_itemsize",
        "_offset",
        "_shape",
        "_strides",
        "data",
        "device",
        "grad",
        "grad_fn",
        "requires_grad",
    )
    _print_opts = PRINT_OPTS

    def __init__(
        self,
        shape: Size | tuple[int, ...] | int,
        dtype: None | Dtype = float64,
        device: DeviceType = None,
        requires_grad: builtins.bool = False,
        buffer: None | t.Any = None,
        offset: t.SupportsIndex = 0,
        strides: None | Size | tuple[int, ...] = None,
    ) -> None:
        """Initialize a `tensor` object from provided shape."""
        if device is not None and device.type != "cpu":
            raise RuntimeError(
                f"{type(self).__qualname__} supports only device type: cpu"
            )
        self.device = device or Device()
        self.requires_grad = requires_grad
        if not isinstance(shape, Iterable):
            shape = (shape,)
        self._shape = tuple(int(dim) for dim in shape)
        if dtype is None:
            dtype = float64
        elif isinstance(dtype, type):
            dtype = globals()[
                f"{dtype.__name__}{'32' if dtype != builtins.bool else ''}"
            ]
        else:
            dtype = globals()[dtype]
        self._dtype = dtype
        self._itemsize = self._dtype.itemsize
        self._offset = int(offset)
        if buffer is None:
            self._base = None
            if self._offset != 0:
                raise ValueError("Offset must be 0 when buffer is None")
            if strides is not None:
                raise ValueError("Strides must be None when buffer is None")
            self._strides = calc_strides(self._shape, self._itemsize)
        else:
            if isinstance(buffer, Tensor) and buffer.base is not None:
                buffer = buffer.base
            self._base = buffer
            if isinstance(buffer, Tensor):
                buffer = buffer.buffer
            if self._offset < 0:
                raise ValueError("Offset must be non-negative")
            if strides is None:
                strides = calc_strides(self._shape, self._itemsize)
            elif not (
                isinstance(strides, tuple)
                and all(isinstance(stride, int) for stride in strides)
                and len(strides) == len(self._shape)
            ):
                raise ValueError("Invalid strides provided")
            self._strides = tuple(strides)
        buffersize = self._strides[0] * self._shape[0] // self._itemsize
        buffersize += self._offset
        Buffer = self._dtype.data * buffersize
        if buffer is None:
            if not isinstance(Buffer, str):
                self._cdata = Buffer()
        elif isinstance(buffer, ctypes.Array):
            self._cdata = Buffer.from_address(ctypes.addressof(buffer))
        else:
            self._cdata = Buffer.from_buffer(buffer)
        self.grad: Tensor = None
        self.grad_fn: Node = Node()
        self.data = self

    def format_repr(
        self,
        formatted: str,
        axis: int,
        offset: int,
        pad: int = 0,
        whitespace: int = 0,
        only: builtins.bool = False,
    ) -> str:
        """Method to mimic PyTorch's tensor as close as possible."""
        if only:
            if not self.requires_grad:
                return self._cdata[0]
        indent = min(2, max(0, (self.ndim - axis - 1)))
        if axis < len(self.shape):
            formatted += "["
            for idx in range(self.shape[axis]):
                if idx > 0:
                    formatted += ("\n " + " " * pad + " " * axis) * indent
                current = offset + idx * self._strides[axis] // self._itemsize
                formatted = self.format_repr(
                    formatted, axis + 1, current, pad, whitespace
                )
                if idx < self.shape[axis] - 1:
                    formatted += ", "
            formatted += "]"
        else:
            element = repr(self._cdata[offset])
            if "." in element and element.endswith(".0"):
                element = f"{element[:-1]:>{whitespace}}"
            else:
                element = f"{element:>{whitespace}}"
            formatted += element
        return formatted

    def __repr__(self) -> str:
        """Return a string representation of `Tensor` object."""
        whitespace = max(
            len(str(self._cdata[idx])) for idx in range(self.nelement())
        )
        only = len(self._cdata) == 1
        formatted = self.format_repr("", 0, self._offset, 7, whitespace, only)
        extra: str = ""
        if self.requires_grad:
            extra = ", requires_grad=True"
            if self.grad_fn.name:
                extra = f", grad_fn=<{self.grad_fn.name}>"
        if self.dtype not in (float64, int64, bool):
            return f"tensor({formatted}, dtype={self.dtype}{extra})"
        else:
            return f"tensor({formatted}{extra})"

    def calculate_offset_shape_strides(
        self, key: int | slice | tuple[None | int | slice, ...] | t.Ellipsis
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        """Calculate offset, shape, and strides for an indexing
        operation.

        This helper method computes the tensor metadata required for
        retrieving a sub-array or value based on the provided key.
        It handles integers, slices, `Ellipsis`, and `None` indexing.

        :param key: Indexing specification (int, slice, tuple, etc.).
        :return: Tuple of (offset, shape, strides).
        :raises IndexError: For invalid axis indexing or bounds errors.
        :raises TypeError: For unsupported key types.
        """
        axis: int = 0
        offset: int = self._offset
        shape: list[int] = []
        strides: list[int] = []
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            ellipsis = key.index(Ellipsis)
            pre = key[:ellipsis]
            post = key[ellipsis + 1 :]
            count = len(self.shape) - len(pre) - len(post)
            if count < 0:
                raise IndexError("Too many indices for tensor")
            key = pre + (slice(None),) * count + post
        for dim in key:
            if axis >= len(self._shape) and dim is not None:
                raise IndexError("Too many indices for tensor")
            axissize = self._shape[axis] if axis < len(self._shape) else None
            if isinstance(dim, int) and axissize is not None:
                if not (-axissize <= dim < axissize):
                    raise IndexError(
                        f"Index {dim} new_tensor of bounds for axis {axis}"
                    )
                dim = dim + axissize if dim < 0 else dim
                offset += dim * self._strides[axis] // self.itemsize
                axis += 1
            elif isinstance(dim, slice) and axissize is not None:
                start, stop, step = dim.indices(axissize)
                shape.append(-(-(stop - start) // step))
                strides.append(step * self._strides[axis])
                offset += start * self._strides[axis] // self.itemsize
                axis += 1
            elif dim is None:
                shape.append(1)
                strides.append(0)
            else:
                raise TypeError(f"Invalid index type: {type(dim).__name__!r}")
        shape.extend(self.shape[axis:])
        strides.extend(self._strides[axis:])
        return offset, tuple(shape), tuple(strides)

    def __float__(self) -> None | float:
        """Convert the tensor to a scalar float if it has exactly one
        element.

        This method attempts to convert a tensor instance to a scalar
        float. The conversion is only possible if the tensor contains
        exactly one element.
        """
        if self.nelement() == 1:
            return float(self.buffer[self._offset])
        else:
            raise TypeError("Only tensor of size 1 can be converted to scalar")

    def __int__(self) -> None | float:
        """Convert the tensor to a scalar int if it has exactly one
        element.

        This method attempts to convert a tensor instance to a scalar
        int. The conversion is only possible if the tensor contains
        exactly one element.
        """
        if self.nelement() == 1:
            return builtins.int(self.buffer[self._offset])
        else:
            raise TypeError("Only tensor of size 1 can be converted to scalar")

    def __bool__(self) -> None | float:
        """Convert the tensor to a scalar bool if it has exactly one
        element.

        This method attempts to convert a tensor instance to a scalar
        bool. The conversion is only possible if the tensor contains
        exactly one element.
        """
        if self.nelement() == 1:
            return builtins.bool(self.buffer[self._offset])
        else:
            raise TypeError("Only tensor of size 1 can be converted to scalar")

    def __len__(self) -> int:
        """Return the size of the first dimension of the tensor.

        This implements the behavior of `len()` for the tensor object,
        providing the number of elements in the first axis.

        :return: Size of the first dimension.
        :raises IndexError: If the tensor has no dimensions.
        """
        if not self.shape:
            raise IndexError("Tensor has no dimensions")
        return self.shape[0]

    def __getitem__(
        self,
        key: builtins.int | slice | tuple[builtins.int | slice | None, ...],
    ) -> t.Any | "Tensor":
        """Retrieve a scalar or a sub-tensor based on the specified index
        or slice.

        This method provides support for advanced indexing, including
        single indices, slices, or tuples of indices and slices,
        allowing flexible access to tensor elements or sub-tensors. For
        scalar values, a single element is returned. For subarrays, a
        new tensor object is created and returned.

        :param key: Index or slice object, or tuple of them.
        :return: Scalar or sub-array as per the indexing operation.
        :raises IndexError: For invalid indexing.
        :raises TypeError: For unsupported key types.
        """
        offset, shape, strides = self.calculate_offset_shape_strides(key)
        if not shape:
            return self._cdata[offset]
        return Tensor(
            shape,
            self._dtype,
            self.device,
            self.requires_grad,
            buffer=self,
            offset=offset,
            strides=strides,
        )

    def __setitem__(
        self,
        key: builtins.int | slice | tuple[None | builtins.int | slice, ...],
        value: builtins.float | builtins.int | t.Sequence[Number] | Tensor,
    ) -> None:
        """Assign a value to a specific element or subarray within the
        tensor.

        This method supports element-wise or block-wise assignment using
        indexing or slicing. The assigned value can be a scalar, a
        sequence (e.g., list or tuple), or another tensor. If assigning
        to a subarray, the value must match the shape of the subarray.

        :param key: Index or slice to identify the element or subarray
            to update.
        :param value: The value to assign to the selected element or
            subarray.
        :raises ValueError: If the number of element in the value does
            not match the size of selected subarray.

        .. note::

            The value can be a single scalar (float or int), a list, or a
            tuple, but must match the shape and size of the subarray
            being updated.
        """
        offset, shape, strides = self.calculate_offset_shape_strides(key)
        if not shape:
            self._cdata[offset] = value
            return
        new_tensor = Tensor(
            shape,
            self._dtype,
            self.device,
            self.requires_grad,
            buffer=self,
            offset=offset,
            strides=strides,
        )
        if isinstance(value, Number):
            array_like = [value] * new_tensor.nelement()
        elif isinstance(value, Iterable):
            array_like = list(value)
        else:
            if not isinstance(value, Tensor):
                value = Tensor(  # type: ignore
                    value,
                    self._dtype,
                    self.device,
                    self.requires_grad,
                )
            array_like = value.flat()
        if new_tensor.nelement() != len(array_like):
            raise ValueError(
                "Number of elements in the value doesn't match the shape"
            )
        sub_tensors = [new_tensor]
        idx = 0
        while sub_tensors:
            sub_tensor = sub_tensors.pop(0)
            if step_size := get_step(sub_tensor):
                block = array_like[idx : idx + sub_tensor.nelement()]
                converted: list[Number] = []
                for element in block:
                    if not self._dtype.name.startswith(("float", "bool")):
                        converted.append(int(element))
                    else:
                        element = round(element, self._print_opts.precision)
                        converted.append(element)
                sub_tensor._cdata[
                    slice(
                        sub_tensor._offset,
                        sub_tensor._offset + sub_tensor.nelement() * step_size,
                        step_size,
                    )
                ] = converted
                idx += sub_tensor.nelement()
            else:
                for dim in range(sub_tensor.shape[0]):
                    sub_tensors.append(sub_tensor[dim])
        assert idx == len(array_like)

    def __add__(self, other: Number | Tensor) -> Tensor:
        """Perform element-wise addition of the tensor with a scalar or
        another tensor.

        This method supports addition with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for addition. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            addition.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        new_tensor: Tensor
        if isinstance(other, int):
            new_tensor = Tensor(
                self.shape, self._dtype, requires_grad=self.requires_grad
            )
            new_tensor[:] = [x + other for x in self._cdata]
        elif isinstance(other, float):
            new_tensor = Tensor(
                self.shape, float64, requires_grad=self.requires_grad
            )
            new_tensor[:] = [x + other for x in self._cdata]
        elif isinstance(other, Tensor):
            dtype = (
                int64
                if all(
                    map(
                        lambda x: not x.name.startswith(("float", "bool")),
                        (self.dtype, other.dtype),
                    )
                )
                else float64
            )
            requires_grad = self.requires_grad or other.requires_grad
            new_tensor = Tensor(self.shape, dtype, requires_grad=requires_grad)
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            new_tensor[:] = [x + y for x, y in zip(self._flat, other._flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

        def AddBackward0() -> None:
            """Backpropagation implementation for addition.

            Computes gradients for `self` and `other` and propagates
            them.
            """
            if None in (self.grad, other.grad):
                self.grad = other.grad = Tensor((1,), dtype)
            self.grad += new_tensor.grad
            other.grad += new_tensor.grad

        new_tensor.grad_fn = Node(AddBackward0)
        new_tensor.grad_fn.inputs = (self, other)
        return new_tensor

    def __radd__(self, other: Number | Tensor) -> Tensor:
        """Perform reverse addition, delegating to `__add__`.

        :param other: The left-hand operand.
        :return: The result of the addition.
        """
        return self.__add__(other)

    def __sub__(self, other: Number | Tensor) -> Tensor:
        """Perform element-wise subtraction of the tensor with a scalar or
        another tensor.

        This method supports subtraction with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for subtraction. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            subtraction.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        new_tensor: Tensor
        if isinstance(other, int):
            new_tensor = Tensor(
                self.shape, self._dtype, requires_grad=self.requires_grad
            )
            new_tensor[:] = [x - other for x in self._cdata]
        elif isinstance(other, float):
            new_tensor = Tensor(
                self.shape, float64, requires_grad=self.requires_grad
            )
            new_tensor[:] = [x - other for x in self._cdata]
        elif isinstance(other, Tensor):
            dtype = (
                int64
                if all(
                    map(
                        lambda x: not x.name.startswith(("float", "bool")),
                        (self.dtype, other.dtype),
                    )
                )
                else float64
            )
            requires_grad = self.requires_grad or other.requires_grad
            new_tensor = Tensor(self.shape, dtype, requires_grad=requires_grad)
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            new_tensor[:] = [x - y for x, y in zip(self._flat, other._flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for -: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

        def SubBackward0() -> None:
            """Backpropagation implementation for subtraction.

            Computes gradients for `self` and `other` and propagates
            them.
            """
            if None in (self.grad, other.grad):
                self.grad = other.grad = Tensor((1,), dtype)
            self.grad += new_tensor.grad
            other.grad -= new_tensor.grad

        new_tensor.grad_fn = Node(SubBackward0)
        new_tensor.grad_fn.inputs = (self, other)
        return new_tensor

    def __rsub__(self, other: Number | Tensor) -> Tensor:
        """Perform reverse subtraction, delegating to `__sub__`.

        :param other: The left-hand operand.
        :return: The result of the subtraction.
        """
        return self.__sub__(other)

    def __mul__(self, other: Number | Tensor) -> Tensor:
        """Perform element-wise multiplication of the tensor with a scalar or
        another tensor.

        This method supports multiplication with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for multiplication. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            multiplication.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        new_tensor: Tensor
        if isinstance(other, int):
            new_tensor = Tensor(
                self.shape, self._dtype, requires_grad=self.requires_grad
            )
            new_tensor[:] = [x * other for x in self._cdata]
        elif isinstance(other, float):
            new_tensor = Tensor(
                self.shape, float64, requires_grad=self.requires_grad
            )
            new_tensor[:] = [x * other for x in self._cdata]
        elif isinstance(other, Tensor):
            dtype = (
                int64
                if all(
                    map(
                        lambda x: not x.name.startswith(("float", "bool")),
                        (self.dtype, other.dtype),
                    )
                )
                else float64
            )
            requires_grad = self.requires_grad or other.requires_grad
            new_tensor = Tensor(self.shape, dtype, requires_grad=requires_grad)
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            new_tensor[:] = [x * y for x, y in zip(self._flat, other._flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

        def MulBackward0() -> None:
            """Backpropagation implementation for multiplication.

            Computes gradients for `self` and `other` and propagates
            them.
            """
            if None in (self.grad, other.grad):
                self.grad = other.grad = Tensor((1,), dtype)
            self.grad += other * new_tensor.grad
            other.grad += self * new_tensor.grad

        new_tensor.grad_fn = Node(MulBackward0)
        new_tensor.grad_fn.inputs = (self, other)
        return new_tensor

    def __rmul__(self, other: Number | Tensor) -> Tensor:
        """Perform reverse multiplication, delegating to `__mul__`.

        :param other: The left-hand operand.
        :return: The result of the multiplication.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Number | Tensor) -> Tensor:
        """Perform element-wise division of the tensor with a scalar or
        another tensor.

        This method supports division with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for division. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            division.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        new_tensor: Tensor
        if isinstance(other, Number):
            data = []
            for idx in self._data:
                try:
                    data.append(idx / other)
                except ZeroDivisionError:
                    data.append(builtins.float("inf"))
            new_tensor = Tensor(
                self.shape, self._dtype, requires_grad=self.requires_grad
            )
            new_tensor[:] = data
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            data = []
            for x, y in zip(self._flat, other._flat):
                try:
                    data.append(x / y)
                except ZeroDivisionError:
                    data.append(builtins.float("inf"))
            requires_grad = self.requires_grad or other.requires_grad
            new_tensor = Tensor(self.shape, dtype, requires_grad=requires_grad)
            new_tensor[:] = data
        else:
            raise TypeError(
                f"Unsupported operand type(s) for /: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

        def DivBackward0() -> None:
            """Backpropagation implementation for division.

            Computes gradients for `self` and `other` and propagates
            them.
            """
            if None in (self.grad, other.grad):
                self.grad = other.grad = Tensor((1,), dtype)
            self.grad += new_tensor.grad / other
            other.grad -= new_tensor.grad * (self / (other**2))

        new_tensor.grad_fn = Node(DivBackward0)
        new_tensor.grad_fn.inputs = (self, other)
        return new_tensor

    def __floordiv__(self, other: Number | Tensor) -> Tensor:
        """Perform element-wise division of the tensor with a scalar or
        another tensor.

        This method supports division with scalars (int or float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param other: The operand for division. Can be a scalar or an
            tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            division.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        new_tensor: Tensor
        if isinstance(other, Number):
            data = []
            for idx in self._data:
                try:
                    data.append(idx // other)
                except ZeroDivisionError:
                    data.append(builtins.float("inf"))
            new_tensor = Tensor(
                self.shape, self._dtype, requires_grad=self.requires_grad
            )
            new_tensor[:] = data
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            data = []
            for x, y in zip(self._flat, other._flat):
                try:
                    data.append(x // y)
                except ZeroDivisionError:
                    data.append(builtins.float("inf"))
            requires_grad = self.requires_grad or other.requires_grad
            new_tensor = Tensor(self.shape, dtype, requires_grad=requires_grad)
            new_tensor[:] = data
        else:
            raise TypeError(
                f"Unsupported operand type(s) for //: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

        def DivBackward0() -> None:
            """Backpropagation implementation for division.

            Computes gradients for `self` and `other` and propagates
            them.
            """
            if None in (self.grad, other.grad):
                self.grad = other.grad = Tensor((1,), dtype)
            self.grad += new_tensor.grad / other
            other.grad -= new_tensor.grad * (self / (other**2))

        new_tensor.grad_fn = Node(DivBackward0)
        new_tensor.grad_fn.inputs = (self, other)
        return new_tensor

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
        new_tensor: Tensor
        if isinstance(other, Tensor):
            dtype = (
                int64
                if all(
                    map(
                        lambda x: not x.name.startswith(("float", "bool")),
                        (self.dtype, other.dtype),
                    )
                )
                else float64
            )
            requires_grad = self.requires_grad or other.requires_grad
            if self.ndim == 1 and other.ndim == 1:
                if self.shape[0] != other.shape[0]:
                    raise ValueError(
                        "Shapes of 1D tensors must be the same for dot product"
                    )
                new_tensor = Tensor((1,), dtype, requires_grad=requires_grad)
                new_tensor[:] = sum(
                    self[idx] * other[idx] for idx in range(self.shape[0])
                )
            elif self.ndim == 2 and other.ndim == 2:
                if self.shape[1] != other.shape[0]:
                    raise ValueError(
                        "Shapes are not aligned for matrix multiplication"
                    )
                new_tensor = Tensor(
                    self.shape, dtype, requires_grad=requires_grad
                )
                for idx in range(self.shape[0]):
                    for jdx in range(other.shape[1]):
                        new_tensor[idx, jdx] = sum(
                            self[idx, kdx] * other[kdx, jdx]
                            for kdx in range(self.shape[1])
                        )
            elif self.ndim > 2 or other.ndim > 2:
                raise ValueError(
                    "Higher-dimensional dot product is not supported"
                )
            else:
                raise ValueError("Invalid shapes for dot product")
        else:
            raise TypeError(
                f"Unsupported operand type(s) for @: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

        def DotBackward0() -> None:
            """Backpropagation implementation for matrix multiplication.

            Computes gradients for `self` and `other` and propagates
            them.
            """
            if None in (self.grad, other.grad):
                self.grad = other.grad = Tensor((1,), dtype)
            self.grad += new_tensor.grad @ other
            other.grad += self @ new_tensor.grad

        new_tensor.grad_fn = Node(DotBackward0)
        new_tensor.grad_fn.inputs = (self, other)
        return new_tensor

    def __mod__(self, other: Number | Tensor) -> Tensor:
        """Perform element-wise modulo operation of the tensor with a
        scalar or another tensor.

        This method supports modulo operation with scalars (int or float)
        and other tensors of the same shape. The resulting tensor is of
        the same shape and dtype as the input.

        :param other: The operand for modulo operation. Can be a scalar
            or a tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            modulo operation.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        if isinstance(other, int):
            new_tensor = Tensor(self.shape, self._dtype)
            new_tensor[:] = [x % other for x in self._cdata]
            return new_tensor
        elif isinstance(other, float):
            new_tensor = Tensor(self.shape, float64)
            new_tensor[:] = [x % other for x in self._cdata]
            return new_tensor
        elif isinstance(other, Tensor):
            dtype = (
                int64
                if all(
                    map(
                        lambda x: not x.name.startswith(("float", "bool")),
                        (self.dtype, other.dtype),
                    )
                )
                else float64
            )
            new_tensor = Tensor(self.shape, dtype)
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            new_tensor[:] = [x % y for x, y in zip(self._flat, other._flat)]
            return new_tensor
        else:
            raise TypeError(
                f"Unsupported operand type(s) for %: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

    def __pow__(self, other: Number | Tensor) -> Tensor:
        """Perform element-wise exponentiation of the tensor with a
        scalar or another tensor.

        This method supports exponentiation with scalars (int or float)
        and other tensors of the same shape. The resulting tensor is of
        the same shape and dtype as the input.

        :param other: The operand for exponentiation. Can be a scalar or
            a tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            exponentiation.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        new_tensor: Tensor
        if isinstance(other, int):
            new_tensor = Tensor(
                self.shape, self._dtype, requires_grad=self.requires_grad
            )
            new_tensor[:] = [x**other for x in self._cdata]
        elif isinstance(other, float):
            new_tensor = Tensor(
                self.shape, float64, requires_grad=self.requires_grad
            )
            new_tensor[:] = [x**other for x in self._cdata]
        elif isinstance(other, Tensor):
            dtype = (
                int64
                if all(
                    map(
                        lambda x: not x.name.startswith(("float", "bool")),
                        (self.dtype, other.dtype),
                    )
                )
                else float64
            )
            requires_grad = self.requires_grad or other.requires_grad
            new_tensor = Tensor(self.shape, dtype, requires_grad=requires_grad)
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            new_tensor[:] = [x**y for x, y in zip(self._flat, other._flat)]
        else:
            raise TypeError(
                f"Unsupported operand type(s) for **: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

        def PowBackward0() -> None:
            """Backpropagation implementation for exponentiation.

            Computes gradient for `self` and propagate it.
            """
            if self.nelement() > 1:
                raise RuntimeError("grad can be created only for scalars")
            if None in (self.grad,):
                self.grad = Tensor((1,), self._dtype)
            self.grad += (other * self ** (other - 1)) * new_tensor.grad

        new_tensor.grad_fn = Node(PowBackward0)
        new_tensor.grad_fn.inputs = (self,)
        return new_tensor

    def __rpow__(self, other: Number | Tensor) -> Tensor:
        """Perform reverse exponentiation, delegating to `__pow__`.

        :param other: The left-hand operand.
        :return: The result of the exponentiation.
        """
        return self.__pow__(other)

    def __lt__(self, other: Number | Tensor) -> Tensor:
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
        new_tensor = Tensor(self.shape, bool)
        if isinstance(other, Number):
            new_tensor[:] = [x < other for x in self._cdata]
            return new_tensor
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            new_tensor[:] = [x < y for x, y in zip(self._flat, other._flat)]
            return new_tensor
        else:
            raise TypeError(
                f"Unsupported operand type(s) for <: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

    def __gt__(self, other: Number | Tensor) -> Tensor:
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
        new_tensor = Tensor(self.shape, bool)
        if isinstance(other, Number):
            new_tensor[:] = [x > other for x in self._cdata]
            return new_tensor
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            new_tensor[:] = [x > y for x, y in zip(self._flat, other._flat)]
            return new_tensor
        else:
            raise TypeError(
                f"Unsupported operand type(s) for >: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

    def __le__(self, other: Number | Tensor) -> Tensor:
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
        new_tensor = Tensor(self.shape, bool)
        if isinstance(other, Number):
            new_tensor[:] = [x <= other for x in self._cdata]
            return new_tensor
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            new_tensor[:] = [x <= y for x, y in zip(self._flat, other._flat)]
            return new_tensor
        else:
            raise TypeError(
                f"Unsupported operand type(s) for <=: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

    def __ge__(self, other: Number | Tensor) -> Tensor:
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
        new_tensor = Tensor(self.shape, bool)
        if isinstance(other, Number):
            new_tensor[:] = [x >= other for x in self._cdata]
            return new_tensor
        elif isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(
                    "Operands couldn't broadcast together with shapes "
                    f"{self.shape} {other.shape}"
                )
            new_tensor[:] = [x >= y for x, y in zip(self._flat, other._flat)]
            return new_tensor
        else:
            raise TypeError(
                f"Unsupported operand type(s) for >=: {type(self).__name__!r} "
                f"and {type(other).__name__!r}"
            )

    @property
    def buffer(self) -> t.Any:
        """Return the memory buffer holding the tensor elements."""
        return self._cdata

    @property
    def base(self) -> None | t.Any:
        """Return underlying buffer (if any)."""
        return self._base

    @property
    def dtype(self) -> t.Any | str:
        """Return the data type of the tensor elements (mainly str)."""
        return self._dtype

    @property
    def is_cuda(self) -> t.Literal[False]:
        """Return True if tensor is stored on GPU else False."""
        return False

    @property
    def is_quantized(self) -> t.Literal[False]:
        """Return True if tensor is quantized else False."""
        return False

    @property
    def is_meta(self) -> t.Literal[False]:
        """Return True if tensor is a meta tensor else False."""
        return False

    def dim(self) -> int:
        """Return the number of dimensions of the tensor."""
        return len(self._shape)

    ndim = property(dim)

    @property
    def nbytes(self) -> t.Any:
        """Return number of byte size of a tensor."""
        return self.nelement() * self.itemsize

    def element_size(self) -> int:
        """Return the size, in bytes, of each tensor element."""
        return self._itemsize

    itemsize = property(element_size)

    @property
    def shape(self) -> Size | tuple[int, ...]:
        """Return shape of the tensor."""
        return self._shape

    @shape.setter
    def shape(self, value: Size) -> None:
        """Set a new shape for the tensor."""
        if value == self.shape:
            return
        if self.nelement() != calc_size(value):
            raise ValueError("New shape is incompatible with the current size")
        if get_step(self) == 1:
            self._shape = tuple(value)
            self._strides = calc_strides(self._shape, self.itemsize)
            return
        shape = [dim for dim in self.shape if dim > 1]
        strides = [
            stride for dim, stride in zip(self.shape, self._strides) if dim > 1
        ]
        new_shape = [dim for dim in value if dim > 1]
        if new_shape != shape:
            raise AttributeError(
                "New shape is incompatible with the current memory layout"
            )
        shape.append(1)
        strides.append(strides[-1])
        new_strides = []
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
        self._shape = tuple(value)
        self._strides = tuple(reversed(new_strides))

    @property
    def _flat(self) -> t.Generator[Number]:
        """Flatten the tensor and yield its elements one by one.

        This property allows you to iterate over all elements in the
        tensor, regardless of its shape or dimensionality, in a flattened
        order. It yields the elements one by one, similar to Python's
        built-in `iter()` function, and handles both contiguous and
        non-contiguous memory layouts.

        :yield: The elements of the tensor in row-major (C-style)
            order.
        """
        sub_tensors = [self]
        while sub_tensors:
            sub_tensor = sub_tensors.pop(0)
            step_size = get_step(sub_tensor)
            if step_size:
                for dim in self._cdata[
                    slice(
                        sub_tensor._offset,
                        sub_tensor._offset + sub_tensor.nelement() * step_size,
                        step_size,
                    )
                ]:
                    yield dim
            else:
                for dim in range(sub_tensor.shape[0]):
                    sub_tensors.append(sub_tensor[dim])

    def flat(self) -> list[Number]:
        """Flatten the tensor and return all its elements in a list.

        This method traverses through the tensor and collects its
        elements into a single list, regardless of its shape or
        dimensionality. It handles contiguous memory layouts and
        non-contiguous slices, ensuring that all elements of the tensor
        are included in the returned list.

        :return: A list containing all elements in the tensor.
        """
        array_like: list[Number] = []
        sub_tensors = [self]
        while sub_tensors:
            sub_tensor = sub_tensors.pop(0)
            step_size = get_step(sub_tensor)
            if step_size:
                array_like += self._cdata[
                    slice(
                        sub_tensor._offset,
                        sub_tensor._offset + sub_tensor.nelement() * step_size,
                        step_size,
                    )
                ]
            else:
                for dim in range(sub_tensor.shape[0]):
                    sub_tensors.append(sub_tensor[dim])
        return array_like

    def nelement(self) -> int:
        """Return total number of elements in a tensor."""
        return calc_size(self._shape)

    numel = nelement

    def size(self, dim: None | int = None) -> Size | tuple[int, ...] | int:
        """Returns the size of the tensor."""
        if dim is not None:
            return self._shape[dim]
        return Size(self._shape)

    def stride(self) -> tuple[int, ...]:
        """Return the strides for traversing the tensor dimensions."""
        return tuple(idx // self.itemsize for idx in self._strides)

    def to(self, dtype: t.Any) -> Tensor:
        """Return a copy of the tensor cast to a specified data type.

        This method creates a new `Tensor` with the same shape and data
        as the original tensor but cast to the specified data type. The
        original tensor remains unmodified.

        :param dtype: The desired data type for the output tensor.
        :return: A new tensor with the specified data type and the same
            shape as the original tensor.
        :raises ValueError: If `dtype` is invalid or cannot be applied
            to the tensor.

        .. note::

            [1] This operation creates a copy of the data, even if the
                requested data type is the same as the original.
        """
        new_tensor = Tensor(
            self.shape,
            dtype,
            self.device,
            self.requires_grad,
        )
        new_tensor[:] = self
        return new_tensor

    type = to

    def float(self) -> Tensor:
        """Return tensor with floating dtype."""
        return self.to(float64)

    float64 = float32 = half = double = float

    def int(self) -> Tensor:
        """Return tensor with integer dtype."""
        return self.to(int64)

    int64 = int32 = int16 = int8 = long = char = int

    def bool(self) -> Tensor:
        """Return tensor with bool dtype."""
        return self.to(bool)

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
        return self.to(self.dtype)

    def _view(self) -> None | Tensor:
        """Create a new view of the tensor.

        This method allows creating a new tensor view. The method
        supports efficient reinterpretation of the data buffer and
        respects the shape and strides of the original tensor. For 1D
        tensors, the dtype can differ if the total number of bytes
        remains consistent.

        :return: A new tensor view with the specified dtype. Returns
            `None` if the view cannot be created.
        :raises ValueError: If the tensor is multidimensional.
        """
        if self.ndim == 1:
            itemsize = self._dtype.itemsize
            size = (self.nbytes // itemsize,)
            offset = (self._offset * self.itemsize) // itemsize
            return Tensor(
                size,
                self._dtype,
                self.device,
                self.requires_grad,
                buffer=self,
                offset=offset,
            )
        elif self.ndim > 1:
            return Tensor(
                self.shape,
                self._dtype,
                self.device,
                self.requires_grad,
                buffer=self,
                offset=self._offset,
                strides=self._strides,
            )
        else:
            raise ValueError("Tensors can only be viewed with the same dtype")

    def view(self, shape: Size | builtins.int = -1) -> None | Tensor:
        """Return a new view of the tensor with the specified shape.

        This method attempts to reshape the tensor while keeping the
        data layout intact. If the new shape is incompatible with the
        current memory layout, a copy of the data is made to achieve the
        desired shape.

        :param shape: The desired shape for the tensor, defaults to -1.
        :return: A reshaped view of the tensor if possible; otherwise, a
            reshaped copy.
        """
        if shape == -1:
            new_tensor = Tensor((self.nelement(),), self._dtype)
            new_tensor[:] = self
            return new_tensor
        new_tensor = self._view()
        try:
            new_tensor._shape = shape
        except AttributeError:
            new_tensor = self.clone()
            new_tensor._shape = shape
        return new_tensor

    reshape = view

    def tolist(self) -> list[t.Any]:
        """Convert the tensor to a nested Python list.

        This method recursively iterates over the dimensions of the
        tensor to construct a nested list that mirrors the shape and
        contents of the tensor.

        :return: A nested Python list representation of the tensor's
            data.
        """
        comprehensions = 0
        shape = list(self.shape).copy()
        comprehension = list(self._cdata).copy()
        skip = self.nelement() // shape[-1]
        while comprehensions < len(self.shape) - 1:
            comprehension = [
                comprehension[idx * shape[-1] : idx * shape[-1] + shape[-1]]
                for idx in range(skip)
            ]
            shape.pop()
            skip = len(comprehension) // shape[-1]
            comprehensions += 1
        return comprehension

    def clamp(
        self,
        min: Number | Tensor,
        max: Number | Tensor,
        out: None | Tensor = None,
    ) -> Tensor:
        """Clamp (limit) the values in the tensor.

        Given an input tensor, this method returns a tensor where values
        are limited to a specified range. All values less than `min`
        are set to `min`, and all values greater than `max` are set
        to `max`.

        :param min: Minimum value to clamp to. Can be a scalar or a
            tensor.
        :param max: Maximum value to clamp to. Can be a scalar or a
            tensor.
        :param out: Optional output tensor to store the result, defaults
            to `None`.
        :return: A new tensor with values clamped to the specified range.
        :raises TypeError: If either `min` or `max` are not either
            of type `int`, `float` or `tensor`.
        :raises ValueError: If output shape doesn't match as the input
            tensor.
        """
        if not isinstance(min, (Number, Tensor)):
            raise TypeError("min must be a scalar or a tensor")
        if not isinstance(max, (Number, Tensor)):
            raise TypeError("max must be a scalar or a tensor")
        if isinstance(min, Tensor) and min.shape != self.shape:
            raise ValueError("min must have same shape as the input tensor")
        if isinstance(max, Tensor) and max.shape != self.shape:
            raise ValueError("max must have same shape as the input tensor")
        if out is None:
            out = Tensor(self._shape, self._dtype)
        F = self.flat()
        R = range(len(F))
        if isinstance(min, Tensor) and isinstance(max, Tensor):
            L = [py_min(max._flat[_], py_max(min._flat[_], F[_])) for _ in R]
        elif isinstance(min, Tensor):
            L = [py_min(max, py_max(min._flat[_], F[_])) for _ in R]
        elif isinstance(max, Tensor):
            L = [py_min(max._flat[_], py_max(min, F[_])) for _ in R]
        else:
            L = [py_min(max, py_max(min, _)) for _ in F]
        out[:] = L
        return out

    clip = clamp

    def item(self) -> t.Any:
        """Return standard scalar Python object for tensor object."""
        if self.nelement() == 1:
            return self.view()[0]
        else:
            raise RuntimeError(
                f"Tensor with {self.nelement()} elements cannot be"
                " converted to scalar"
            )

    def flatten(self) -> Tensor:
        """Return a copy of the tensor collapsed into one dimension."""
        new_tensor = Tensor(
            (self.nelement(),),
            self._dtype,
            self.device,
            self.requires_grad,
        )
        new_tensor[:] = self
        return new_tensor

    ravel = flatten

    def view_(self, shape: Size, strides: Size) -> Tensor:
        """Create a new view of the tensor with the specified shape and
        strides.

        :param shape: The shape of the new view.
        :param strides: The strides of the new view.
        :return: A new tensor view.
        """
        new_tensor = self.__class__.__new__(self.__class__)
        new_tensor._shape = shape
        new_tensor._strides = strides
        new_tensor._cdata = self._cdata
        new_tensor._dtype = self._dtype
        new_tensor._offset = self._offset
        new_tensor._itemsize = self._itemsize
        new_tensor.device = self.device
        new_tensor.requires_grad = self.requires_grad
        return new_tensor

    def transpose(self, dim0: builtins.int, dim1: builtins.int) -> Tensor:
        """Transpose the tensor by permuting its dimensions.

        This method returns a view of the tensor with its dimensions
        permuted. If no dimensions are specified, the dimensions are
        reversed (i.e., equivalent to a full transpose).

        :param dim0: First dimension to be transposed.
        :param dim1: Second dimension to be transposed.
        :return: A new tensor view with transposed dimensions.
        :raises ValueError: If the provided dimensions are invalid.
        """
        if sorted((dim0, dim1)) != list(range(self.ndim)):
            raise ValueError("Invalid dimensions permutation")
        dims = tuple(reversed(sorted((dim0, dim1))))
        shape = Size(tuple(self._shape[dim] for dim in dims))
        strides = Size(tuple(self._strides[dim] for dim in dims))
        return self.view_(shape, strides)

    def t(self) -> Tensor:
        """Transpose dimensions 0 and 1."""
        return self.transpose(0, 1)

    def unique(self, sorted: builtins.bool = True) -> Tensor:
        """Return unique elements from the tensor.

        :param sorted: Whether to sort the unique elements before
            returning the output, defaults to `True`.
        :return: Tensor with list of unique elements.
        """
        size = len((unique := set(self._cdata)))
        new_tensor = Tensor(
            (size,),
            self._dtype,
            self.device,
            self.requires_grad,
        )
        new_tensor[:] = tuple(py_sorted(unique) if sorted else unique)
        return new_tensor

    def lt(self, other: Number | Tensor) -> Tensor:
        """Compute input < other element-wise.

        :param other: Tensor or Scalar to compare.
        :return: A boolean tensor that is True where input is less than
            other and False elsewhere.
        """
        return self.__lt__(other)

    less = lt

    def gt(self, other: Number | Tensor) -> Tensor:
        """Compute input > other element-wise.

        :param other: Tensor or Scalar to compare.
        :return: A boolean tensor that is True where input is greater
            than other and False elsewhere.
        """
        return self.__gt__(other)

    greater = gt

    def le(self, other: Number | Tensor) -> Tensor:
        """Compute input <= other element-wise.

        :param other: Tensor or Scalar to compare.
        :return: A boolean tensor that is True where input is less than
            or equal to other and False elsewhere.
        """
        return self.__le__(other)

    less_equal = le

    def ge(self, other: Number | Tensor) -> Tensor:
        """Compute input >= other element-wise.

        :param other: Tensor or Scalar to compare.
        :return: A boolean tensor that is True where input is greater
            than or equal to other and False elsewhere.
        """
        return self.__ge__(other)

    greater_equal = ge

    def neg(self) -> Tensor:
        """Compute negative of the elements.

        :return: Tensor with negative of the input elements.
        """
        return self.__mul__(-1)

    negative = neg

    def fill_(self, value: Number) -> Tensor:
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
        if not isinstance(value, Number):
            raise ValueError("Value must be an integer or a float")
        self[:] = value
        return self

    def add(self, other: Number | Tensor, *, alpha: Number = 1) -> Tensor:
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
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        return self.__add__(alpha * other)

    def sub(self, other: Number | Tensor, *, alpha: Number = 1) -> Tensor:
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
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        return self.__sub__(alpha * other)

    def mul(self, other: Number | Tensor, *, alpha: Number = 1) -> Tensor:
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
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        return self.__mul__(alpha * other)

    def div(
        self, other: Number | Tensor, *, rounding_mode: None | str = None
    ) -> t.Any:
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
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        if rounding_mode is not None:
            raise RuntimeError("Rounding mode is not supported")
        return self.__div__(other)

    divide = div

    def matmul(self, other: Tensor) -> Tensor:
        """Perform matrix multiplication of the tensor with another
        tensor, scaled by alpha.

        This method supports matrix multiplication with other tensors of
        the same shape. The resulting tensor is of the same shape and
        dtype as the input.

        :param other: The operand for matrix multiplication.
        :return: A new tensor containing the result of the element-wise
            matrix multiplication.
        :raises TypeError: If `other` is not a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        return self.__matmul__(other)

    def pow(self, exponent: Number | Tensor) -> Tensor:
        """Perform element-wise exponentiation of the tensor with a
        scalar or another tensor.

        This method supports exponentiation with scalars (float) and
        other tensors of the same shape. The resulting tensor is of the
        same shape and dtype as the input.

        :param exponent: The operand for exponentiation. Can be a scalar
            or a tensor of the same shape.
        :return: A new tensor containing the result of the element-wise
            exponentiation.
        :raises TypeError: If `other` is neither a scalar nor a tensor.
        :raises ValueError: If `other` is a tensor but its shape
            doesn't match `self.shape`.
        """
        return self.__pow__(exponent)

    def backward(
        self,
        gradient: None | Tensor = None,
        inputs: None | tuple[Tensor, ...] = None,
    ) -> None:
        """Compute the gradient of current tensor w.r.t graph leaves.

        :param gradient: The gradient of the output with respect to the
            tensor, defaults to `None`.
        :param inputs: Tuple of inputs w.r.t which the gradients will
            be accumulated into `.grad`.
        :raises RuntimeError: If the tensor does not require gradients.
        """
        if not self.requires_grad:
            raise RuntimeError(
                "Tensors does not require grad and does not have a grad_fn"
            )
        graph: list[tuple[Tensor, ...] | Tensor] = []
        seen: set[tuple[Tensor, ...] | Tensor] = set()
        if gradient is None:
            gradient = Tensor((1,), float64)
            gradient[:] = 1.0
        self.grad = gradient

        def iter_graph(inputs: tuple[Tensor, ...] | Tensor) -> None:
            """Recursive function to traverse the computation graph."""
            if inputs not in seen:
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

    def max(self) -> Tensor:
        """Return the maximum value of all elements along a given
        dimension.

        :return: Return the maximum value as a tensor.
        """
        new_tensor = Tensor((1,), self.dtype)
        new_tensor[:] = max(self._flat)
        return new_tensor


@function_dispatch
def tensor(
    data: ArrayLike,
    *,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: builtins.bool = False,
) -> Tensor:
    """Create a tensor from the input data.

    This function initializes a tensor with a given data array,
    optionally specifying its data type, device, and gradient
    requirement.

    :param data: The data from which to create the tensor. Must have
        uniform shape.
    :param dtype: The desired data type of the tensor. If None, inferred
        automatically, defaults to `None`.
    :param device: The device where the tensor will reside, defaults to
        `None`.
    :param requires_grad: Whether the tensor requires gradients for
        backpropagation, defaults to `False`.
    :return: A new instance of the Tensor object.
    :raises ValueError: If the input data does not have a uniform shape.
    """
    if not has_uniform_shape(data):
        raise ValueError("Input data is not uniformly nested")
    size = calc_shape_from_data(data)
    array_like: list[t.Any] = []

    def chain_from_iterable(object: ArrayLike) -> None:
        """Recursively flatten the input iterable."""
        if isinstance(object, Iterable) and not isinstance(data, (str, bytes)):
            for idx in object:
                chain_from_iterable(idx)
        else:
            array_like.append(object)

    chain_from_iterable(data)
    if dtype is None:
        dtype = (
            int64
            if all(isinstance(idx, int) for idx in array_like)
            else float64
        )
    new_tensor = Tensor(size, dtype, device, requires_grad)
    new_tensor[:] = array_like
    return new_tensor
