"""\
SlowTorch Tensor
================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, January 07 2025
Last updated on: Sunday, June 01 2025

Tensor object.

This module provides the foundational classes and functions for tensor
operations in this library. It defines the `Tensor` class and supporting
utility like the `tensor` function to create and manipulate tensors.
SlowTorch aims to mimic PyTorch's tensor behavior while implementing key
functionalities from scratch to foster learning and customisation.

The central `Tensor` class acts as the primary data structure for
multidimensional tensors and supports features like automatic
differentiation, and flexible data types. Designed with flexibility,
efficiency, and modularity in mind, the `Tensor` class aims to replicate
and innovate upon the core features of PyTorch tensors while emphasising
a Python-standard-library-only approach. Additionally, the class
introduces a Pythonic, educational perspective, making it suitable
for learning and experimentation with tensor mechanics without relying
on external libraries.

The module supports features such as::

    - The primary `Tensor` object supports auto-differentiation.
    - Efficient storage and representation of n-dimensional data.
    - Flexible shape manipulation, including reshaping and broadcasting.
    - Slicing and indexing support for intuitive data access.
    - Element-wise operations, including arithmetic, logical, and
      comparison operations, via rich operator overloading.
    - Conversion utilities to export data to native Python types
      (e.g., lists).
    - Tensor based operations such as unsqueezing, reshaping, cloning,
      transposing, etc.
    - Linear algebraic operations such as calulating maximum, minimum,
      mean, standard deviation, exponent, log, square root etc.

The `Tensor` implementation draws inspiration from PyTorch's
architecture but deliberately simplifies and reimagines certain aspects
for educational purposes and to meet the constraints of pure Python. By
eschewing C or Cython extensions, the `Tensor` class offers an
accessible implementation that emphasises algorithmic clarity over raw
performance.

While this module implements many fundamental features of `Tensor`,
it does not aim to match PyTorch's performance or breadth. Instead, the
focus is on clarity, usability, and modularity, providing a platform for
learning and experimentation.
"""

from __future__ import annotations

import builtins
import ctypes
import functools
import typing as t
from collections import OrderedDict
from collections.abc import Iterable
from itertools import product as pdt

import slowtorch
from slowtorch import function_dispatch
from slowtorch.internal.device import device as Device
from slowtorch.internal.dtype import bool as bool_
from slowtorch.internal.dtype import float32
from slowtorch.internal.dtype import float64
from slowtorch.internal.dtype import int64
from slowtorch.internal.shape import get_step
from slowtorch.internal.shape import make_contiguous_strides
from slowtorch.internal.shape import numel
from slowtorch.internal.shape import size as Size
from slowtorch.internal.shape import unravel_index
from slowtorch.utils import PrinterOptions
from slowtorch.utils import set_module

if t.TYPE_CHECKING:
    from slowtorch.types import BoolLikeType
    from slowtorch.types import DeviceType
    from slowtorch.types import Dim
    from slowtorch.types import Dtype
    from slowtorch.types import FloatLikeType
    from slowtorch.types import Id
    from slowtorch.types import IndexLike
    from slowtorch.types import Input
    from slowtorch.types import IntLikeType
    from slowtorch.types import Scalar
    from slowtorch.types import ShapeType
    from slowtorch.types import StorageWeakRef
    from slowtorch.types import StrideType
    from slowtorch.types import TensorOrTensors


PRINT_OPTS = PrinterOptions()


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
    :raises ValueError: If invalid strides or offsets are provided.
    """

    _print_opts = PRINT_OPTS

    def __init__(
        self,
        shape: ShapeType | IntLikeType,
        dtype: None | Dtype = None,
        device: None | DeviceType = None,
        requires_grad: BoolLikeType = False,
        storage: None | str | ctypes._CData | Tensor = None,
        offset: IntLikeType = 0,
        strides: None | StrideType = None,
    ) -> None:
        """Initialise a `tensor` object from provided shape."""
        self.device = device or Device()
        self.requires_grad = requires_grad
        if not isinstance(shape, Iterable):
            shape = (shape,)
        self._shape = tuple(int(dim) for dim in shape)
        if dtype is None:
            dtype = float32
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
                and all(isinstance(stride, int) for stride in strides)
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
        self._cached_sizes_strides_offsets: dict[
            tuple[IntLikeType, str],
            tuple[ShapeType, StrideType, IntLikeType],
        ] = {}

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
        indent = min(2, max(0, (self.ndim - dimension - 1)))
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
            whitespace = max(len(fmt_data(value)) for value in self.storage)
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
        if self.dtype not in (float32, float64, int64, bool_):
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
            return int(self.storage[self._storage_offset])
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
            return bool(self.storage[self._storage_offset])
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
            offset = storage_offset + sum(
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
        if any(isinstance(kdx, bool) for kdx in indices):
            indices = tuple(
                (int(index) if isinstance(index, bool) else index)
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
        if isinstance(value, (int, bool, float)):
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
                    int(element)
                    if not self.dtype.name.startswith(("float", "bool"))
                    else round(element, self._print_opts.precision)
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

    def __getattr__(self, name: str) -> functools.partial[t.Any]:
        """Lazily resolves and registers missing methods`.

        If an attribute `name` is not found on the current instance,
        this method attempts to fetch a callable with the same name from
        the `slowtorch.nn.functional` module. If such a function exists,
        it is wrapped with `functools.partial` thereby emulating
        method-like behaviour.

        This pattern supports delayed registration of functional-style
        operations, keeping the core tensor class lightweight while
        allowing dynamic extensibility.

        :param name: The name of the attribute being accessed.
        :return: The existing attribute if present, or a lazily-bound
            function if resolved.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            from slowtorch.nn import functional

            return functools.partial(getattr(functional, name), self)

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
        new_tensor = Tensor(self._shape, bool_)
        if isinstance(other, (int, bool, float)):
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
        new_tensor = Tensor(self._shape, bool_)
        if isinstance(other, (int, bool, float)):
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
        new_tensor = Tensor(self._shape, bool_)
        if isinstance(other, (int, bool, float)):
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
        new_tensor = Tensor(self._shape, bool_)
        if isinstance(other, (int, bool, float)):
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
        return int(self._itemsize)

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
        return self.to(bool_)

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
                    else self[prefix + (index,)]
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
        new_tensor._cached_sizes_strides_offsets = (
            self._cached_sizes_strides_offsets
        )
        return new_tensor

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
        values = builtins.sorted(unique) if sorted else storage
        new_tensor = Tensor(
            shape=len(values),
            dtype=self.dtype,
            device=self.device,
            requires_grad=self.requires_grad,
        )
        new_tensor[:] = values
        return new_tensor

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
        if not isinstance(value, (int, bool, float)):
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
