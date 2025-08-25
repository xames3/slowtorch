"""\
SlowTorch Tensor Creation
=========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, 01 June 2025
Last updated on: Sunday, 24 August 2025

Tensor creation operations.

This module provides essential tensor creation and initialisation
utilities. It contains a suite of functions designed to construct and
populate `Tensor` objects with various patterns and values, mimicking
the functionality of PyTorch's core tensor creation routines.

This module serves as the foundation for generating tensors with specific
sizes, patterns, and values. These utilities are essential for
initialising tensors, enabling users to quickly prototype and perform
computations without the need for manual data entry. Inspired by
PyTorch's tensor creation APIs, this module brings similar functionality
to `slowtorch` with a focus on educational clarity, pure Python
implementation, and modular design.

.. note::

    The implementations in this module are not optimised for performance
    and are intended for learning and exploratory purposes. For
    production-grade numerical computation, consider using PyTorch
    directly.
"""

from __future__ import annotations

import math
import typing as t
from collections.abc import Iterable

import slowtorch
from slowtorch import function_dispatch
from slowtorch.internal.shape import check_same_shape
from slowtorch.internal.shape import infer_size_shapes
from slowtorch.internal.tensor import Tensor
from slowtorch.utils import dtypecheck

if t.TYPE_CHECKING:
    from slowtorch.types import ArrayLikeOrScalar
    from slowtorch.types import BoolLikeType
    from slowtorch.types import DeviceType
    from slowtorch.types import Dtype
    from slowtorch.types import IntLikeType
    from slowtorch.types import Scalar
    from slowtorch.types import ShapeType
    from slowtorch.types import StorageWeakRef


@function_dispatch
def tensor(
    data: ArrayLikeOrScalar,
    *,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Create a tensor from the input data.

    This function initialises a tensor with a given data array,
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
    if not check_same_shape(data):
        raise ValueError("Input data is not uniformly nested")
    size = size if (size := infer_size_shapes(data)) else (1,)
    storage: StorageWeakRef = []

    def chain_from_iterable(a: ArrayLikeOrScalar) -> None:
        """Recursively flatten the input iterable."""
        if isinstance(a, Iterable) and not isinstance(data, (str, bytes)):
            for index in a:
                chain_from_iterable(index)
        else:
            storage.append(a)

    chain_from_iterable(data)
    if dtype is None:
        types = set(type(item) for item in storage)
        dtype = (
            bool
            if types <= {bool}
            else (
                slowtorch.int64
                if types <= {int}
                else slowtorch.float32 if types <= {int, float} else None
            )
        )
    new_tensor = Tensor(size, dtype, device, requires_grad)
    new_tensor[:] = storage
    return new_tensor


@function_dispatch
def empty(
    *size: IntLikeType,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Create a new tensor without initialising its values.

    The `empty` function returns a new `Tensor` with the specified
    size, data type, and device. The contents of the tensor are
    uninitialised and contain random data, in theory but practically,
    it fills them with zeros because of `ctypes`. This function is useful
    for performance-critical applications where immediate initialisation
    is not required.

    :param size: Dimensions of the tensor.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: An uninitialised tensor with the specified properties.

    .. note::

        [1] The contents of the returned tensor are random and should
            not be used without proper initialisation.
    """
    if not isinstance(size, (int, Iterable)):
        raise TypeError(
            "Expected a sequence of integers or a single integer, "
            f"got {size!r}"
        )
    shape: ShapeType
    if len(size) == 1 and isinstance(size[0], Iterable):
        shape = size[0]
    else:
        shape = size
    new_tensor = Tensor(shape, dtype, device, requires_grad)
    zero = 0.0 if dtypecheck(dtype) == slowtorch.float32 else 0
    new_tensor.fill_(zero)
    return new_tensor


@function_dispatch
def zeros(
    *size: IntLikeType,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Create a new tensor filled with zeros.

    The `zeros` function creates an tensor with the specified size,
    data type, and device, initialising all its elements to zero.
    This function is particularly useful for scenarios requiring a blank
    tensor with known dimensions and type, where all elements must
    initially be zero.

    :param size: Dimensions of the tensor.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: An initialised tensor with the values set to 0.
    """
    return empty(
        *size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


@function_dispatch
def zeros_like(
    input: Tensor,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Create a new tensor with the same size and type as a given
    tensor, filled with zeros.

    The `zeros_like` function creates an tensor with the same size as
    the input tensor `input`. By default, the new tensor will have the
    same data type as `input`, but this can be overridden with the
    `dtype` parameter.

    :param input: The reference tensor whose size and optionally type
        are used to create the new tensor.
    :param dtype: The desired data type of the new tensor. If `None`,
        the data type of `input` is used, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: A new tensor filled with zeros, matching the size of `input`
        and the specified or inherited data type and device.
    """
    dtype = input.dtype if dtype is None else dtype
    size = input.shape
    return zeros(
        *size, dtype=dtype, device=device, requires_grad=requires_grad
    )


@function_dispatch
def ones(
    *size: IntLikeType,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Create a new tensor filled with ones.

    The `ones` function creates an tensor with the specified size,
    data type, and device, initialising all its elements to one. This
    function is particularly useful for scenarios requiring a blank
    tensor with known dimensions and type, where all elements must
    initially be one.

    :param size: Dimensions of the tensor.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: An initialised tensor with the values set to 1.
    """
    new_tensor = empty(
        *size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    one = 1.0 if dtypecheck(dtype) == slowtorch.float32 else 1
    new_tensor.fill_(one)
    return new_tensor


@function_dispatch
def ones_like(
    input: Tensor,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Create a new tensor with the same size and type as a given
    tensor, filled with ones.

    The `ones_like` function creates an tensor with the same size as
    the input tensor `input`. By default, the new tensor will have the
    same data type as `input`, but this can be overridden with the
    `dtype` parameter.

    :param input: The reference tensor whose size and optionally type
        are used to create the new tensor.
    :param dtype: The desired data type of the new tensor. If `None`,
        the data type of `input` is used, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: A new tensor filled with ones, matching the size of `input`
        and the specified or inherited data type and device.
    """
    dtype = input.dtype if dtype is None else dtype
    size = input.shape
    return ones(*size, dtype=dtype, device=device, requires_grad=requires_grad)


@function_dispatch
def full(
    *size: IntLikeType,
    fill_value: Scalar,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Create a new tensor filled with `fill_value`.

    The `full` function creates an tensor with the specified size,
    data type, and device, initialising all its elements to `fill_value`.
    This function is particularly useful for scenarios requiring a blank
    tensor with known dimensions and type, where all elements must
    initially be `fill_value`.

    :param size: Dimensions of the tensor.
    :param fill_value: Value to fill the output tensor with.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: An initialised tensor with the values set to `fill_value`.
    """
    new_tensor = empty(
        *size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    new_tensor.fill_(fill_value)
    return new_tensor


@function_dispatch
def full_like(
    input: Tensor,
    fill_value: Scalar,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Create a new tensor with the same size and type as a given
    tensor, filled with `fill_value`.

    The `ones_like` function creates an tensor with the same size as
    the input tensor `input`. By default, the new tensor will have the
    same data type as `input`, but this can be overridden with the
    `dtype` parameter.

    :param input: The reference tensor whose size and optionally type
        are used to create the new tensor.
    :param dtype: The desired data type of the new tensor. If `None`,
        the data type of `input` is used, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: A new tensor filled with `fill_value`, matching the size of
        `input` and the specified or inherited data type and device.
    """
    dtype = input.dtype if dtype is None else dtype
    size = input.shape
    return full(
        *size,
        fill_value,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


@function_dispatch
def tril(
    input: Tensor,
    diagonal: IntLikeType = 0,
) -> Tensor:
    """Create a lower triangular matrix (2-D tensor) with elements of
    the input below and on the main diagonal, and zeros elsewhere.

    :param input: The reference tensor whose size and optionally type
        are used to create the new tensor.
    :param diagonal: Diagonal offset, defaults to 0. Possible diagonal
        offsets::
            - `0` for main diagonal
            - `>0` above the main diagonal
            - `<0` below the main diagonal
    :return: A 2-D tensor of shape `(N, M)` where the elements below and
        on the main diagonal are 1, and the elements above the diagonal
        are 0.
    :return: Tensor filled with elements of the input below and on the
        main diagonal, and zeros elsewhere.
    :raises RuntimeError: If input shape is not 2-D.
    """
    if len(input.shape) != 2:
        raise RuntimeError("tril: input tensor must have 2 dimensions")
    M, N = input.shape
    new_tensor = empty(
        (M, N),
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
    )
    for m in range(M):
        for n in range(N):
            if n - m <= diagonal:
                new_tensor[m, n] = input[m, n]
            else:
                new_tensor[m, n] = 0
    return new_tensor


@function_dispatch
def triu(
    input: Tensor,
    diagonal: IntLikeType = 0,
) -> Tensor:
    """Create a upper triangular matrix (2-D tensor) with elements of
    the input above and on the main diagonal, and zeros elsewhere.

    :param input: The reference tensor whose size and optionally type
        are used to create the new tensor.
    :param diagonal: Diagonal offset, defaults to 0. Possible diagonal
        offsets::
            - `0` for main diagonal
            - `>0` above the main diagonal
            - `<0` below the main diagonal
    :return: A 2-D tensor of shape `(N, M)` where the elements above and
        on the main diagonal are 1, and the elements above the diagonal
        are 0.
    :return: Tensor filled with elements of the input above and on the
        main diagonal, and zeros elsewhere.
    :raises RuntimeError: If input shape is not 2-D.
    """
    if len(input.shape) != 2:
        raise RuntimeError("triu: input tensor must have 2 dimensions")
    M, N = input.shape
    new_tensor = empty(
        (M, N),
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
    )
    for m in range(M):
        for n in range(N):
            if n - m < diagonal:
                new_tensor[m, n] = 0
            else:
                new_tensor[m, n] = input[m, n]
    return new_tensor


@function_dispatch
def arange(
    start: Scalar = 0,
    end: Scalar = float("inf"),
    step: Scalar = 1,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Return evenly spaced values within a given range.

    The `arange` function generates a 1-D tensor containing evenly
    spaced values over a specified interval. The interval is defined by
    the `start`, `end`, and `step` arguments. It mimics the behavior of
    Python's built-in `range` function but returns an `Tensor` instead.

    :param start: Starting value for the set of points, defaults to 0.
    :param end: Ending value for the set of points, defaults to `nan`.
    :param step: Gap betwen each pair of adjacent points, defaults to 1.
    :param dtype: The desired data type of the new tensor. If `None`,
        the data type of `input` is used, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: A 1-D tensor of evenly spaced values.
    """
    if end == float("inf"):
        start, end = 0, start
    if step == 0:
        raise ValueError("Step size must not be zero")
    size = max(0, math.ceil((end - start) / step))
    if dtype is None:
        dtype = (
            slowtorch.int64
            if all(isinstance(index, int) for index in (start, end, step))
            else slowtorch.float32
        )
    new_tensor = empty(
        size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    new_tensor[:] = (start + index * step for index in range(size))
    return new_tensor


@function_dispatch
def linspace(
    start: Scalar,
    end: Scalar,
    steps: IntLikeType,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Creates a 1-D tensor of `steps` equally spaced points between
    `start` and `end` (inclusive).

    :param start: Starting value for the set of points.
    :param end: Ending value for the set of points`.
    :param steps: Size of the constructed tensor.
    :param dtype: The desired data type of the new tensor. If `None`,
        the data type of `input` is used, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: A 1-D tensor of evenly spaced values.
    :raises ValueError: If `steps` is not a positive integer.
    """
    if steps <= 0:
        raise ValueError("Scalar of steps must be a positive integer")
    if dtype is None:
        dtype = slowtorch.float32
    new_tensor = empty(
        steps, dtype=dtype, device=device, requires_grad=requires_grad
    )
    jump = (end - start) / (steps - 1) if steps > 1 else 0
    new_tensor[:] = (start + index * jump for index in range(steps))
    return new_tensor
