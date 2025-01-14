"""\
SlowTorch Functions API
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, January 13 2025
Last updated on: Tuesday, January 14 2025

This module provides essential tensor creation and initialization
utilities for the `slowtorch` package. It contains a suite of functions
designed to construct and populate `Tensor` objects with various
patterns and values, mimicking the functionality of PyTorch's core
tensor creation routines.

This module serves as the foundation for generating tensors with specific
sizes, patterns, and values. These utilities are essential for
initializing tensors, enabling users to quickly prototype and perform
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
    - Simplicity: Implementations prioritize readability and modularity,
      enabling users to explore and extend functionality with ease.
    - Educational Value: As part of the `slowtorch` project, this module
      emphasizes the learning of tensor mechanics and API design.

The tensor creation functions in this module are ideal for::

    - Initializing tensors for numerical computations.
    - Creating test datasets for algorithm development.
    - Prototyping applications that require structured numerical data.
    - Exploring the mechanics of multidimensional tensor creation in
      Python.

The implementations in this module are not optimized for performance and
are intended for learning and exploratory purposes. For production-grade
numerical computation, consider using PyTorch directly.
"""

from __future__ import annotations

import itertools
import math

import slowtorch
from slowtorch import function_dispatch
from slowtorch._random import Generator
from slowtorch._random import default_generator
from slowtorch._tensor import Tensor
from slowtorch._types import Number
from slowtorch._utils import DeviceType
from slowtorch._utils import Dtype
from slowtorch._utils import Size


@function_dispatch
def randn(
    size: Size | tuple[int, ...] | int,
    *,
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
    if isinstance(size, int):
        new_tensor = Tensor((size,), dtype, device, requires_grad)
        new_tensor[:] = [generator.internal.gauss(0, 1) for _ in range(size)]
        return new_tensor
    elif isinstance(size, tuple):
        new_tensor = Tensor(size, dtype, device, requires_grad)
        N = range(max(size))
        for dim in itertools.product(N, N):
            try:
                new_tensor[dim] = generator.internal.gauss(0, 1)
            except IndexError:
                continue
        return new_tensor
    else:
        raise TypeError(
            f"Expected a sequence of integers or a single integer, "
            f"got {size!r}"
        )


@function_dispatch
def rand(
    size: Size | tuple[int, ...] | int,
    *,
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
    if isinstance(size, int):
        new_tensor = Tensor((size,), dtype, device, requires_grad)
        new_tensor[:] = [generator.internal.uniform(0, 1) for _ in range(size)]
        return new_tensor
    elif isinstance(size, tuple):
        new_tensor = Tensor(size, dtype, device, requires_grad)
        N = range(max(size))
        for dim in itertools.product(N, N):
            try:
                new_tensor[dim] = generator.internal.uniform(0, 1)
            except IndexError:
                continue
        return new_tensor
    else:
        raise TypeError(
            f"Expected a sequence of integers or a single integer, "
            f"got {size!r}"
        )


@function_dispatch
def randint(
    low: int = 0,
    high: None | int = None,
    size: None | Size | tuple[int, ...] = None,
    *,
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
        dtype = slowtorch.int32
    if isinstance(size, tuple):
        new_tensor = Tensor(size, dtype, device, requires_grad)
        if len(size) == 1:
            new_tensor[:] = [
                generator.internal.randint(low, high - 1)
                for _ in range(size[0])
            ]
            return new_tensor
        else:
            N = range(max(size))
        for dim in itertools.product(N, N):
            try:
                new_tensor[dim] = generator.internal.randint(low, high - 1)
            except IndexError:
                continue
        return new_tensor
    else:
        raise TypeError(f"Expected a sequence of integers got {size!r}")


@function_dispatch
def randperm(
    n: int,
    *,
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
        dtype = slowtorch.int32
    new_tensor = Tensor((n,), dtype, device, requires_grad)
    data = list(range(n))
    generator.internal.shuffle(data)
    new_tensor[:] = data
    return new_tensor


@function_dispatch
def empty(
    size: Size | tuple[int, ...] | int,
    *,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a new tensor without initializing its values.

    The `empty` function returns a new `Tensor` with the specified
    size, data type, and device. The contents of the tensor are
    uninitialized and contain random data, in theory but practically,
    it fills them with zeros because of `ctypes`. This function is useful
    for performance-critical applications where immediate initialization
    is not required.

    :param size: Dimensions of the tensor.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: An uninitialized tensor with the specified properties.

    .. note::

        [1] The contents of the returned tensor are random and should
            not be used without proper initialization.
    """
    return Tensor(size, dtype, device, requires_grad)


@function_dispatch
def zeros(
    size: Size | tuple[int, ...] | int,
    *,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a new tensor filled with zeros.

    The `zeros` function creates an tensor with the specified size,
    data type, and device, initializing all its elements to zero.
    This function is particularly useful for scenarios requiring a blank
    tensor with known dimensions and type, where all elements must
    initially be zero.

    :param size: Dimensions of the tensor.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: An initialized tensor with the values set to 0.
    """
    return empty(size, dtype=dtype, device=device, requires_grad=requires_grad)


@function_dispatch
def zeros_like(
    input: Tensor,
    *,
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
    return zeros(size, dtype=dtype, device=device, requires_grad=requires_grad)


@function_dispatch
def ones(
    size: Size | tuple[int, ...] | int,
    *,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a new tensor filled with ones.

    The `ones` function creates an tensor with the specified size,
    data type, and device, initializing all its elements to one. This
    function is particularly useful for scenarios requiring a blank
    tensor with known dimensions and type, where all elements must
    initially be one.

    :param size: Dimensions of the tensor.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device where the tensor will be created, defaults to
        `None`.
    :param requires_grad: Boolean if autograd should record operations
        on the returned tensor, defaults to `False`.
    :return: An initialized tensor with the values set to 1.
    """
    new_tensor = empty(
        size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    new_tensor.fill_(1)
    return new_tensor


@function_dispatch
def ones_like(
    input: Tensor,
    *,
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
    return ones(size, dtype=dtype, device=device, requires_grad=requires_grad)


@function_dispatch
def full(
    size: Size | tuple[int, ...] | int,
    fill_value: Number,
    *,
    dtype: None | Dtype = None,
    device: DeviceType = None,
    requires_grad: bool = False,
) -> Tensor:
    """Create a new tensor filled with `fill_value`.

    The `full` function creates an tensor with the specified size,
    data type, and device, initializing all its elements to `fill_value`.
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
    :return: An initialized tensor with the values set to `fill_value`.
    """
    new_tensor = empty(
        size,
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
    *,
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
        size,
        fill_value,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


@function_dispatch
def arange(
    start: Number = 0,
    end: Number = float("inf"),
    step: Number = 1,
    *,
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
    dtype = (
        slowtorch.int32
        if all(isinstance(idx, int) for idx in (start, end, step))
        else slowtorch.float32
    )
    new_tensor = empty(
        (size,), dtype=dtype, device=device, requires_grad=requires_grad
    )
    new_tensor[:] = [start + idx * step for idx in range(size)]
    return new_tensor