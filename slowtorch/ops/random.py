"""\
SlowTorch Random Module
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, 01 June 2025
Last updated on: Sunday, 24 August 2025

Random sampling operations.

This module provides functions to generate random tensors with various
distributions, including normal, uniform, and integer distributions. It
also includes functions for generating random permutations and filling
tensors with uniform random values.
"""

from __future__ import annotations

import itertools
import typing as t
from collections.abc import Iterable

import slowtorch
from slowtorch import function_dispatch
from slowtorch.internal.random import Generator
from slowtorch.internal.random import default_generator
from slowtorch.internal.tensor import Tensor
from slowtorch.utils import dtypecheck
from slowtorch.utils import fill

if t.TYPE_CHECKING:
    from slowtorch.types import BoolLikeType
    from slowtorch.types import DeviceType
    from slowtorch.types import Dtype
    from slowtorch.types import FloatLikeType
    from slowtorch.types import IntLikeType
    from slowtorch.types import ShapeType


@function_dispatch
def randn(
    *size: IntLikeType,
    generator: None | Generator = None,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Return a tensor filled with random numbers from a normal
    distribution with mean 0 and standard deviation 1.

    :param size: Dimensions of the tensor.
    :param generator: Pseudorandom number generator, defaults to `None`.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device on which the tensor is created, defaults
        to `None`.
    :param requires_grad: If autograd should record operations,
        defaults to `False`.
    :return: Tensor of random numbers from a normal distribution.
    :raises TypeError: If `size` is not a sequence of integers or a
        single integer.
    :raises RuntimeError: If `dtype` is not `slowtorch.float32`.

    .. note::

        [1] The normal distribution is defined as N(0, 1), where
            0 is the mean and 1 is the standard deviation.
        [2] The `dtype` must be `slowtorch.float32` as other types
            are not supported for this operation.
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
    shape: ShapeType
    if len(size) == 1 and isinstance(size[0], Iterable):
        shape = size[0]
    else:
        shape = size
    new_tensor = Tensor(shape, dtype, device, requires_grad)
    numel = 1
    for dim in shape:
        numel *= dim
    values = (generator.internal.gauss(0, 1) for _ in range(numel))
    fill(new_tensor, values)
    return new_tensor


@function_dispatch
def rand(
    *size: IntLikeType,
    generator: None | Generator = None,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Return a tensor filled with random numbers drawn from a uniform
    distribution on the interval [0, 1).

    :param size: Dimensions of the tensor.
    :param generator: Pseudorandom number generator, defaults to `None`.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device on which the tensor is created, defaults
        to `None`.
    :param requires_grad: If autograd should record operations,
        defaults to `False`.
    :return: Tensor of random numbers from a uniform distribution.
    :raises TypeError: If `size` is not a sequence of integers or a
        single integer.
    :raises RuntimeError: If `dtype` is not `slowtorch.float32`.

    .. note::

        [1] The uniform distribution U(0, 1) generates values such
            that: U(0, 1) = 0 <= x < 1
        [2] Only `slowtorch.float32` is supported for this operation.
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
    shape: ShapeType
    if len(size) == 1 and isinstance(size[0], Iterable):
        shape = size[0]
    else:
        shape = size
    new_tensor = Tensor(shape, dtype, device, requires_grad)
    numel = 1
    for dim in shape:
        numel *= dim
    values = (generator.internal.uniform(0, 1) for _ in range(numel))
    fill(new_tensor, values)
    return new_tensor


@function_dispatch
def randint(
    low: IntLikeType = 0,
    high: None | IntLikeType = None,
    size: None | ShapeType = None,
    generator: None | Generator = None,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Return a tensor filled with random integers from a uniform
    distribution in the interval [low, high).

    :param low: Lower bound (inclusive), defaults to 0.
    :param high: Upper bound (exclusive), defaults to `None`.
    :param size: Dimensions of the tensor, defaults to `None`.
    :param generator: Pseudorandom number generator, defaults to `None`.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device on which the tensor is created, defaults
        to `None`.
    :param requires_grad: If autograd should record operations,
        defaults to `False`.
    :return: Tensor of random integers in the specified range.
    :raises ValueError: If `size` is `None`.
    :raises TypeError: If `size` is not a sequence of integers or a
        single integer.

    .. note::

        If `high` is `None`, the range becomes [0, low).
        If `low` is greater than or equal to `high`, the range is
        invalid.
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
        shape: ShapeType
        if len(size) == 1 and isinstance(size[0], Iterable):
            shape = size[0]
        else:
            shape = size
        new_tensor = Tensor(shape, dtype, device, requires_grad)
        numel = 1
        for dim in shape:
            numel *= dim
        values = (
            generator.internal.randint(low, high - 1) for _ in range(numel)
        )
        fill(new_tensor, values)
        return new_tensor
    else:
        raise TypeError(f"Expected a sequence of integers got {size!r}")


@function_dispatch
def randperm(
    n: IntLikeType,
    generator: None | Generator = None,
    dtype: None | Dtype = None,
    device: None | DeviceType = None,
    requires_grad: BoolLikeType = False,
) -> Tensor:
    """Return a tensor containing a random permutation of integers from
    0 to n - 1.

    :param n: The upper bound (exclusive).
    :param generator: Pseudorandom number generator, defaults to `None`.
    :param dtype: Data type of the tensor, defaults to `None`.
    :param device: Device on which the tensor is created, defaults
        to `None`.
    :param requires_grad: If autograd should record operations,
        defaults to `False`.
    :return: Tensor with a random permutation of integers.
    """
    if generator is None:
        generator = default_generator
    if dtype is None:
        dtype = slowtorch.int64
    new_tensor = Tensor(n, dtype, device, requires_grad)
    storage = list(range(n))
    generator.internal.shuffle(storage)
    new_tensor[:] = storage
    return new_tensor


@function_dispatch
def uniform_(
    tensor: Tensor,
    a: FloatLikeType = 0.0,
    b: FloatLikeType = 1.0,
    generator: None | Generator = None,
) -> Tensor:
    """Fill the input tensor with values drawn from a uniform
    distribution U(a, b).

    The uniform distribution U(a, b) generates values such that::

        U(a, b) = a <= x < b

    :param tensor: The tensor to fill.
    :param a: Lower bound (inclusive), defaults to 0.0.
    :param b: Upper bound (exclusive), defaults to 1.0.
    :param generator: Pseudorandom number generator, defaults to `None`.
    :return: The modified tensor.
    :raises ValueError: If `a` is not less than `b`.

    .. note::

        This function modifies the input tensor in place and returns it.
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
