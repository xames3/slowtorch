"""\
SlowTorch Random Tensors
========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, June 01 2025
Last updated on: Sunday, June 01 2025

Random sampling operations.

This module implements core random operations used to initialise or
generate tensors with stochastic values. These functions mimic the API
and semantics of PyTorch's random ops (e.g., `randn`, `randint`,
`randperm`), but are implemented using only Python's standard libraries
like `random` and `math`.

This module is essential for testing, weight initialization, and any
operation involving randomness.
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
