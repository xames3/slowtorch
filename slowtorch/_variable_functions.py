"""\
SlowTorch Functions API
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, January 13 2025
Last updated on: Wednesday, May 28 2025

This module provides essential tensor creation and initialisation
utilities for the `slowtorch` package. It contains a suite of functions
designed to construct and populate `Tensor` objects with various
patterns and values, mimicking the functionality of PyTorch's core
tensor creation routines.

This module serves as the foundation for generating tensors with specific
sizes, patterns, and values. These utilities are essential for
initialising tensors, enabling users to quickly prototype and perform
computations without the need for manual data entry. Inspired by
PyTorch's tensor creation APIs, this module brings similar functionality
to `slowtorch` with a focus on educational clarity, pure Python
implementation, and modular design.

The following functions are implemented in this module::

    - Tensor Creation Functions
    - Pattern-Based Tensor Functions
    - Tensor Transformation Functions

This module is designed to balance functionality and clarity, making it
both a practical tool and an educational resource. Key principles
guiding its implementation include::

    - Consistency: Functions follow predictable naming conventions and
      parameter usage, ensuring a seamless experience for users familiar
      with PyTorch.
    - Flexibility: Support for multiple data types, sizes, and device
      layouts.
    - Simplicity: Implementations prioritise readability and modularity,
      enabling users to explore and extend functionality with ease.
    - Educational Value: As part of the `slowtorch` project, this module
      emphasises the learning of tensor mechanics and API design.

The tensor creation functions in this module are ideal for::

    - Initialising tensors for numerical computations.
    - Creating test datasets for algorithm development.
    - Prototyping applications that require structured numerical data.
    - Exploring the mechanics of multidimensional tensor creation in
      Python.

The implementations in this module are not optimised for performance and
are intended for learning and exploratory purposes. For production-grade
numerical computation, consider using PyTorch directly.
"""

from __future__ import annotations

import itertools
import math
import typing as t
from collections.abc import Iterable

import slowtorch
from slowtorch import function_dispatch
from slowtorch._random import Generator
from slowtorch._random import default_generator
from slowtorch._tensor import DeviceType
from slowtorch._tensor import Tensor
from slowtorch._types import Number
from slowtorch._utils import Dtype
from slowtorch._utils import Size
from slowtorch._utils import _fill_tensor
from slowtorch._utils import dtypecheck


@function_dispatch
def randn(
    *size: int,
    generator: None | Generator = None,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a tensor filled with random numbers from a normal
    distribution with mean 0 and variance 1.

    :param size: Dimensions of the tensor.
    :param generator: A pseudorandom number generator for sampling,
        defaults to `None`.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: Tensor filled with random numbers.
    """
    if generator is None:
        generator = default_generator
    if not isinstance(size, (int, Iterable)):
        raise TypeError(
            "Expected a sequence of integers or a single integer, "
            f"got {size!r}"
        )
    dtype = dtypecheck(dtype)
    if not dtype == slowtorch.float32:
        raise RuntimeError(
            f"'normal' not implemented for {dtype.typename[:-6]!r}"
        )
    shape: tuple[int, ...]
    if len(size) == 1 and isinstance(size[0], Iterable):
        shape = tuple(size[0])
    else:
        shape = tuple(size)
    new_tensor = Tensor(shape, dtype, device, requires_grad)
    numel = 1
    for dim in shape:
        numel *= dim
    values = (generator.internal.gauss(0, 1) for _ in range(numel))
    _fill_tensor(new_tensor, values)
    return new_tensor


@function_dispatch
def rand(
    *size: int,
    generator: None | Generator = None,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a tensor filled with random numbers from a uniform
    distribution on the interval [0, 1).

    :param size: Dimensions of the tensor.
    :param generator: A pseudorandom number generator for sampling,
        defaults to `None`.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: Tensor filled with random numbers.
    """
    if generator is None:
        generator = default_generator
    if not isinstance(size, (int, Iterable)):
        raise TypeError(
            "Expected a sequence of integers or a single integer, "
            f"got {size!r}"
        )
    dtype = dtypecheck(dtype)
    if not dtype == slowtorch.float32:
        raise RuntimeError(
            f"'uniform' not implemented for {dtype.typename[:-6]!r}"
        )
    shape: tuple[int, ...]
    if len(size) == 1 and isinstance(size[0], Iterable):
        shape = tuple(size[0])
    else:
        shape = tuple(size)
    new_tensor = Tensor(shape, dtype, device, requires_grad)
    numel = 1
    for dim in shape:
        numel *= dim
    values = (generator.internal.uniform(0, 1) for _ in range(numel))
    _fill_tensor(new_tensor, values)
    return new_tensor


@function_dispatch
def randint(
    low: int = 0,
    high: None | int = None,
    size: None | Size | tuple[int, ...] = None,
    generator: None | Generator = None,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a tensor filled with random numbers from a uniform
    distribution between low (inclusive) and high (exclusive).

    :param low: Lower bound (inclusive), defaults to 0.
    :param high: Upper bound (exclusive), defaults to `None`.
    :param size: Dimensions of the tensor, defaults to `None`.
    :param generator: A pseudorandom number generator for sampling,
        defaults to `None`.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: Tensor filled with random integers from range [low, high].
    """
    if generator is None:
        generator = default_generator
    if high is None:
        low, high = 0, low
    if size is None:
        raise ValueError("Size must be tuple of ints, not None")
    if dtype is None:
        dtype = slowtorch.int64
    if isinstance(size, tuple):
        shape: tuple[int, ...]
        if len(size) == 1 and isinstance(size[0], Iterable):
            shape = tuple(size[0])
        else:
            shape = tuple(size)
        new_tensor = Tensor(shape, dtype, device, requires_grad)
        numel = 1
        for dim in shape:
            numel *= dim
        values = (
            generator.internal.randint(low, high - 1) for _ in range(numel)
        )
        _fill_tensor(new_tensor, values)
        return new_tensor
    else:
        raise TypeError(f"Expected a sequence of integers got {size!r}")


@function_dispatch
def randperm(
    n: int,
    generator: None | Generator = None,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
) -> Tensor:
    """Return a tensor filled with random permutation of integers from
    0 to n - 1.

    :param n: The upper bound (exclusive).
    :param generator: A pseudorandom number generator for sampling,
        defaults to `None`.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: Tensor filled with random integers from range [low, high].
    """
    if generator is None:
        generator = default_generator
    if dtype is None:
        dtype = slowtorch.int64
    new_tensor = Tensor((n,), dtype, device, requires_grad)
    data = list(range(n))
    generator.internal.shuffle(data)
    new_tensor[:] = data
    return new_tensor


@function_dispatch
def uniform_(
    tensor: Tensor,
    a: float = 0.0,
    b: float = 1.0,
    generator: None | Generator = None,
) -> Tensor:
    """Fill the input tensor with values drawn from a uniform
    distribution U(a, b).

    The uniform distribution U(a, b) generates values such that::

        U(a, b) = a <= x < b

    :param tensor: The tensor to initialize.
    :param a: The lower bound of the uniform distribution (inclusive),
        defaults to 0.0.
    :param b: The upper bound of the uniform distribution (exclusive),
        defaults to 1.0.
    :return: The modified tensor.
    :raises ValueError: If `a` >= `b`.
    """
    if a >= b:
        raise ValueError(f"Invalid range, a ({a}) must be less than b ({b})")
    if generator is None:
        generator = default_generator
    size = tensor.shape
    N = range(max(size))
    for dim in itertools.product(N, N):
        try:
            tensor[dim] = generator.internal.uniform(a, b)
        except IndexError:
            continue
    return tensor


@function_dispatch
def empty(
    *size: int,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
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
    shape: tuple[int, ...]
    if len(size) == 1 and isinstance(size[0], Iterable):
        shape = tuple(size[0])
    else:
        shape = tuple(size)
    new_tensor = Tensor(shape, dtype, device, requires_grad)
    zero = 0.0 if dtypecheck(dtype) == slowtorch.float32 else 0
    new_tensor.fill_(zero)
    return new_tensor


@function_dispatch
def zeros(
    *size: int,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
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
    device: DeviceType = None,
    requires_grad: bool = False,
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
    *size: int,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
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
    new_tensor.fill_(1.0)
    return new_tensor


@function_dispatch
def ones_like(
    input: Tensor,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
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
    *size: int,
    fill_value: Number,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
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
    fill_value: Number,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
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
    diagonal: int = 0,
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
    m, n = input.shape
    new_tensor = empty(
        (m, n),
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
    )
    for idx in range(m):
        for jdx in range(n):
            if jdx - idx <= diagonal:
                new_tensor[idx, jdx] = input[idx, jdx]
            else:
                new_tensor[idx, jdx] = 0
    return new_tensor


@function_dispatch
def triu(
    input: Tensor,
    diagonal: int = 0,
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
    m, n = input.shape
    new_tensor = empty(
        (m, n),
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
    )
    for idx in range(m):
        for jdx in range(n):
            if jdx - idx < diagonal:
                new_tensor[idx, jdx] = 0
            else:
                new_tensor[idx, jdx] = input[idx, jdx]
    return new_tensor


@function_dispatch
def arange(
    start: Number = 0,
    end: Number = float("inf"),
    step: Number = 1,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
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
            if all(isinstance(idx, int) for idx in (start, end, step))
            else slowtorch.float32
        )
    new_tensor = empty(
        size, dtype=dtype, device=device, requires_grad=requires_grad
    )
    new_tensor[:] = (start + idx * step for idx in range(size))
    return new_tensor


@function_dispatch
def linspace(
    start: Number,
    end: Number,
    steps: int,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
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
        raise ValueError("Number of steps must be a positive integer")
    if dtype is None:
        dtype = slowtorch.float32
    new_tensor = empty(
        steps, dtype=dtype, device=device, requires_grad=requires_grad
    )
    jump = (end - start) / (steps - 1) if steps > 1 else 0
    new_tensor[:] = [start + idx * jump for idx in range(steps)]
    return new_tensor


@function_dispatch
def cat(
    tensors: t.Sequence[Tensor],
    dim: int = 0,
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
    new_tensor = empty(*tuple(size), dtype=tensors[0].dtype)
    offset = 0
    for tensor in tensors:
        slices = [slice(None)] * len(tensor.shape)
        slices[dim] = slice(offset, offset + tensor.shape[dim])
        new_tensor[tuple(slices)] = tensor
        offset += tensor.shape[dim]
    return new_tensor
