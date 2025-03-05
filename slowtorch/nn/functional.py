"""\
SlowTorch Neural Network related Functions API
==============================================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Wednesday, January 15 2025
Last updated on: Monday, March 03 2025

This module in `SlowTorch` offers a comprehensive suite of stateless
functions that perform various tensor operations, mimicking the
functionality of PyTorch's functional API. This module is designed to
provide users with a wide array of operations, including activation
functions, loss computations, and other mathematical transformations,
which are essential building blocks for creating and training neural
network models. Unlike the object-oriented approach of the layers and
losses modules, the functional module delivers these operations in a
purely functional manner.

Activation functions are crucial for introducing non-linearity into
neural networks, enabling them to learn complex patterns in data. The
functional module includes implementations of popular activation
functions. Loss functions measure the difference between the model
outputs and the target values, guiding the optimisation process. The
modular and functional design of the functional module makes it easy to
extend with new operations as needed. Users can contribute additional
functions following the same pattern, enhancing the utility and
flexibility of the `SlowTorch` package.

The functional module is a versatile and essential part of this package,
providing users with a wide range of operations for building and
training neural network models. Its functional API design promotes
clarity and efficiency in code, making advanced neural network
construction accessible to both novice and experienced practitioners in
the field of deep learning.
"""

from __future__ import annotations

import builtins
import itertools
import math
import statistics
import typing as t

import slowtorch
from slowtorch._tensor import Node
from slowtorch._tensor import Tensor
from slowtorch._types import Number
from slowtorch._utils import broadcast_shapes
from slowtorch._utils import normal_exp
from slowtorch._utils import safe_exp
from slowtorch._utils import safe_max
from slowtorch._utils import safe_range


def add(input: Tensor, other: Number | Tensor) -> Tensor:
    """Perform element-wise addition of the tensor with a scalar or
    another tensor.

    This function supports addition with scalars (int or float) and
    other tensors of the same shape. The resulting tensor is of the
    same shape and dtype as the input.

    :param input: Input tensor to be added.
    :param other: The operand for addition. Can be a scalar or an
        tensor of the same shape.
    :return: A new tensor containing the result of the element-wise
        addition.
    :raises TypeError: If `other` is neither a scalar nor a tensor.
    :raises ValueError: If `other` is a tensor but its shape
        doesn't match `input.shape`.
    """
    new_tensor: Tensor
    if isinstance(other, Number):
        dtype = (
            slowtorch.float64
            if isinstance(other, float) or input.dtype.name.startswith("float")
            else slowtorch.int64
        )
        new_tensor = Tensor(
            input.shape,
            dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = [data + other for data in input._cdata]
    elif isinstance(other, Tensor):
        dtype = (
            slowtorch.float64
            if input.dtype.name.startswith("float")
            or other.dtype.name.startswith("float")
            else slowtorch.int64
        )
        shape = broadcast_shapes(input.shape, other.shape)
        input = input.broadcast_to(shape)
        other = other.broadcast_to(shape)
        requires_grad = input.requires_grad or other.requires_grad
        new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
        new_tensor[:] = [x + y for x, y in zip(input._flat, other._flat)]
    else:
        raise TypeError(
            f"Unsupported operand type(s) for +: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )

    def AddBackward0() -> None:
        """Backpropagation implementation for addition.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = other.grad = Tensor(1, dtype)
        input.grad += new_tensor.grad
        other.grad += new_tensor.grad

    new_tensor.grad_fn = Node(AddBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


def sub(input: Tensor, other: Number | Tensor) -> Tensor:
    """Perform element-wise subtraction of the tensor with a scalar or
    another tensor.

    This function supports subtraction with scalars (int or float) and
    other tensors of the same shape. The resulting tensor is of the
    same shape and dtype as the input.

    :param input: Input tensor to be subtracted from.
    :param other: The operand for subtraction. Can be a scalar or an
        tensor of the same shape.
    :return: A new tensor containing the result of the element-wise
        subtraction.
    :raises TypeError: If `other` is neither a scalar nor a tensor.
    :raises ValueError: If `other` is a tensor but its shape
        doesn't match `input.shape`.
    """
    new_tensor: Tensor
    if isinstance(other, Number):
        dtype = (
            slowtorch.float64
            if isinstance(other, float) or input.dtype.name.startswith("float")
            else slowtorch.int64
        )
        new_tensor = Tensor(
            input.shape,
            dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = [data - other for data in input._cdata]
    elif isinstance(other, Tensor):
        dtype = (
            slowtorch.float64
            if input.dtype.name.startswith("float")
            or other.dtype.name.startswith("float")
            else slowtorch.int64
        )
        shape = broadcast_shapes(input.shape, other.shape)
        input = input.broadcast_to(shape)
        other = other.broadcast_to(shape)
        requires_grad = input.requires_grad or other.requires_grad
        new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
        new_tensor[:] = [x - y for x, y in zip(input._flat, other._flat)]
    else:
        raise TypeError(
            f"Unsupported operand type(s) for -: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )

    def SubBackward0() -> None:
        """Backpropagation implementation for subtraction.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = other.grad = Tensor(1, dtype)
        input.grad += new_tensor.grad
        other.grad -= new_tensor.grad

    new_tensor.grad_fn = Node(SubBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


def mul(input: Tensor, other: Number | Tensor) -> Tensor:
    """Perform element-wise multiplication of the tensor with a scalar
    or another tensor.

    This function supports multiplication with scalars (int or float) and
    other tensors of the same shape. The resulting tensor is of the
    same shape and dtype as the input.

    :param input: Input tensor to be multiplied.
    :param other: The operand for multiplication. Can be a scalar or an
        tensor of the same shape.
    :return: A new tensor containing the result of the element-wise
        multiplication.
    :raises TypeError: If `other` is neither a scalar nor a tensor.
    :raises ValueError: If `other` is a tensor but its shape
        doesn't match `input.shape`.
    """
    new_tensor: Tensor
    if isinstance(other, Number):
        dtype = (
            slowtorch.float64
            if isinstance(other, float) or input.dtype.name.startswith("float")
            else slowtorch.int64
        )
        new_tensor = Tensor(
            input.shape,
            dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = [data * other for data in input._cdata]
    elif isinstance(other, Tensor):
        dtype = (
            slowtorch.float64
            if input.dtype.name.startswith("float")
            or other.dtype.name.startswith("float")
            else slowtorch.int64
        )
        shape = broadcast_shapes(input.shape, other.shape)
        input = input.broadcast_to(shape)
        other = other.broadcast_to(shape)
        requires_grad = input.requires_grad or other.requires_grad
        new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
        new_tensor[:] = [x * y for x, y in zip(input._flat, other._flat)]
    else:
        raise TypeError(
            f"Unsupported operand type(s) for *: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )

    def MulBackward0() -> None:
        """Backpropagation implementation for multiplication.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = other.grad = Tensor(1, dtype)
        input.grad += other * new_tensor.grad
        other.grad += input * new_tensor.grad

    new_tensor.grad_fn = Node(MulBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


def div(
    input: Tensor,
    other: Number | Tensor,
    rounding_mode: None | str = None,
) -> Tensor:
    """Perform element-wise division of the tensor with a scalar or
    another tensor.

    This function supports division with scalars (int or float) and
    other tensors of the same shape. The resulting tensor is of the
    same shape and dtype as the input.

    :param input: Input tensor to be divided from.
    :param other: The operand for division. Can be a scalar or an
        tensor of the same shape.
    :param rounding_mode: Type of rounding to apply to the result,
        defaults to `None`.
    :return: A new tensor containing the result of the element-wise
        division.
    :raises TypeError: If `other` is neither a scalar nor a tensor.
    :raises ValueError: If `other` is a tensor but its shape
        doesn't match `input.shape`.
    """
    new_tensor: Tensor
    data: list[Number] = []
    if isinstance(other, Number):
        for idx in input._cdata:
            try:
                data.append(
                    idx // other if rounding_mode == "floor" else idx / other
                )
            except ZeroDivisionError:
                data.append(builtins.float("inf"))
        new_tensor = Tensor(
            input.shape,
            input.dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = data
    elif isinstance(other, Tensor):
        shape = broadcast_shapes(input.shape, other.shape)
        input = input.broadcast_to(shape)
        other = other.broadcast_to(shape)
        requires_grad = input.requires_grad or other.requires_grad
        new_tensor = Tensor(shape, input.dtype, requires_grad=requires_grad)
        for x, y in zip(input._flat, other._flat):
            try:
                data.append(x // y if rounding_mode == "floor" else x / y)
            except ZeroDivisionError:
                data.append(builtins.float("inf"))
        new_tensor[:] = data
    else:
        raise TypeError(
            f"Unsupported operand type(s) for /: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )

    def DivBackward0() -> None:
        """Backpropagation implementation for division.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = other.grad = Tensor(1, slowtorch.float64)
        input.grad += new_tensor.grad / other
        other.grad -= new_tensor.grad * (input / (other**2))

    new_tensor.grad_fn = Node(DivBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


def matmul(input: Tensor, other: Tensor) -> Tensor:
    """Perform element-wise matrix multiplication of the tensor with
    another tensor.

    This function supports matrix multiplication with other tensors of
    the same shape. The resulting tensor is of the same shape and
    dtype as the input.

    :param input: Input tensor to be matrix multiplied by.
    :param other: The operand for matrix multiplication.
    :return: A new tensor containing the result of the element-wise
        matrix multiplication.
    :raises TypeError: If `other` is not a tensor.
    :raises ValueError: If `other` is a tensor but its shape doesn't
        match `input.shape`.
    """
    if not isinstance(other, Tensor):
        raise TypeError(
            f"Unsupported operand type(s) for @: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )
    dtype = (
        slowtorch.float64
        if input.dtype.name.startswith("float")
        or other.dtype.name.startswith("float")
        else slowtorch.int64
    )
    requires_grad = input.requires_grad or other.requires_grad
    if input.ndim == 1 and other.ndim == 1:
        if input.shape[0] != other.shape[0]:
            raise ValueError(
                "Shapes of 1D tensors must be the same for dot product"
            )
        new_tensor = Tensor(1, dtype, requires_grad=requires_grad)
        new_tensor[:] = builtins.sum(
            input[idx] * other[idx] for idx in range(input.shape[0])
        )
    elif input.ndim == 2 or other.ndim == 2:
        if input.shape[1] != other.shape[0]:
            raise ValueError(
                "Shapes are not aligned for matrix multiplication"
            )
        new_tensor = Tensor(
            (input.shape[0], other.shape[1]),
            dtype,
            requires_grad=requires_grad,
        )
        for idx in range(new_tensor.shape[0]):
            for jdx in range(new_tensor.shape[1]):
                new_tensor[idx, jdx] = builtins.sum(
                    input[idx, kdx] * other[kdx, jdx]
                    for kdx in range(input.shape[1])
                )
    elif input.ndim > 2 or other.ndim > 2:
        shape = broadcast_shapes(input.shape[:-2], other.shape[:-2]) + (
            input.shape[-2],
            other.shape[-1],
        )
        input = input.broadcast_to(shape[:-2] + input.shape[-2:])
        other = other.broadcast_to(shape[:-2] + input.shape[-2:])
        requires_grad = input.requires_grad or other.requires_grad
        new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
        for batch in safe_range(new_tensor.shape[:-2]):
            for idx in range(new_tensor.shape[-2]):
                for jdx in range(new_tensor.shape[-1]):
                    new_tensor[batch, idx, jdx] = builtins.sum(
                        input[batch, idx, kdx] * other[batch, kdx, jdx]
                        for kdx in range(input.shape[-1])
                    )
    else:
        raise ValueError("Invalid shapes for dot product")

    def DotBackward0() -> None:
        """Backpropagation implementation for matrix multiplication.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = other.grad = Tensor(1, dtype)
        input.grad += new_tensor.grad @ other
        other.grad += input @ new_tensor.grad

    new_tensor.grad_fn = Node(DotBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


def remainder(input: Tensor, other: Number | Tensor) -> Tensor:
    """Perform element-wise modulo operation of the tensor with a
    scalar or another tensor.

    This function supports modulo operation with scalars (int or float)
    and other tensors of the same shape. The resulting tensor is of
    the same shape and dtype as the input.

    :param input: Input tensor to be performed modulo operation to.
    :param other: The operand for modulo operation. Can be a scalar
        or a tensor of the same shape.
    :return: A new tensor containing the result of the element-wise
        modulo operation.
    :raises TypeError: If `other` is neither a scalar nor a tensor.
    :raises ValueError: If `other` is a tensor but its shape
        doesn't match `input.shape`.
    """
    new_tensor: Tensor
    if isinstance(other, Number):
        dtype = (
            slowtorch.float64
            if isinstance(other, float) or input.dtype.name.startswith("float")
            else slowtorch.int64
        )
        new_tensor = Tensor(
            input.shape,
            dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = [data % other for data in input._cdata]
    elif isinstance(other, Tensor):
        dtype = (
            slowtorch.float64
            if input.dtype.name.startswith("float")
            or other.dtype.name.startswith("float")
            else slowtorch.int64
        )
        shape = broadcast_shapes(input.shape, other.shape)
        input = input.broadcast_to(shape)
        other = other.broadcast_to(shape)
        requires_grad = input.requires_grad or other.requires_grad
        new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
        new_tensor[:] = [x % y for x, y in zip(input._flat, other._flat)]
    else:
        raise TypeError(
            f"Unsupported operand type(s) for %: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )

    def RemainderBackward0() -> None:
        """Backpropagation implementation for modulo operation.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = other.grad = Tensor(1, dtype)
        input.grad += new_tensor.grad
        other.grad += new_tensor.grad

    new_tensor.grad_fn = Node(RemainderBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


def pow(input: Tensor, other: Number | Tensor) -> Tensor:
    """Perform element-wise exponentiation of the tensor with a
    scalar or another tensor.

    This function supports exponentiation with scalars (int or float)
    and other tensors of the same shape. The resulting tensor is of
    the same shape and dtype as the input.

    :param input: Input tensor to be exponentiated.
    :param other: The operand for exponentiation. Can be a scalar or
        a tensor of the same shape.
    :return: A new tensor containing the result of the element-wise
        exponentiation.
    :raises TypeError: If `other` is neither a scalar nor a tensor.
    :raises ValueError: If `other` is a tensor but its shape
        doesn't match `input.shape`.
    """
    new_tensor: Tensor
    if isinstance(other, Number):
        dtype = (
            slowtorch.float64
            if isinstance(other, float) or input.dtype.name.startswith("float")
            else slowtorch.int64
        )
        new_tensor = Tensor(
            input.shape,
            dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = [data**other for data in input._cdata]
    elif isinstance(other, Tensor):
        dtype = (
            slowtorch.float64
            if input.dtype.name.startswith("float")
            or other.dtype.name.startswith("float")
            else slowtorch.int64
        )
        shape = broadcast_shapes(input.shape, other.shape)
        input = input.broadcast_to(shape)
        other = other.broadcast_to(shape)
        requires_grad = input.requires_grad or other.requires_grad
        new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
        new_tensor[:] = [x**y for x, y in zip(input._flat, other._flat)]
    else:
        raise TypeError(
            f"Unsupported operand type(s) for **: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )

    def PowBackward0() -> None:
        """Backpropagation implementation for exponentiation.

        Computes gradient for `input` and propagate it.
        """
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if None in (input.grad,):
            input.grad = Tensor(1, input.dtype)
        input.grad += (other * input ** (other - 1)) * new_tensor.grad

    new_tensor.grad_fn = Node(PowBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def clone(input: Tensor) -> Tensor:
    """Return a deep copy of the tensor.

    This method creates a new `Tensor` instance with the same data,
    shape, and type as the original tensor. The copy is independent
    of the original, meaning changes to the copy do not affect the
    original tensor.

    :param input: Input tensor to be cloned.
    :return: A new tensor with the same data, shape, and type as the
        original tensor.

    .. note::

        [1] This method ensures that both the data and metadata of
            the tensor are duplicated.
        [2] The `to` method is used internally for copying, ensuring
            consistency and type fidelity.
    """
    new_tensor = input.to(input.dtype)

    def CloneBackward0() -> None:
        """Backpropagation implementation for cloning.

        Computes gradient for `input` and propagate it.
        """
        if None in (input.grad,):
            input.grad += new_tensor.grad

    new_tensor.grad_fn = Node(CloneBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def sum(
    input: Tensor,
    dim: None | int = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the sum of elements in the tensor across a specified
    dimension.

    This function computes the sum of all elements in the tensor if no
    dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be summed.
    :param dim: The dimension along which to compute the sum, defaults
        to `None`. For `None`, the sum is computed over all elements of
        the tensor.
    :param keepdims: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the sum of the specified elements.
    :raises ValueError: If the specified dimension is invalid.
    """
    if dim is not None and not (0 <= dim < input.ndim):
        raise ValueError(f"Invalid dimension {dim} for tensor")
    if dim is None:
        shape = (1,) if not keepdims else (1,) * input.ndim
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = builtins.sum(input._flat)
    else:
        shape = tuple(
            (1 if idx == dim and keepdims else input.shape[idx])
            for idx in range(input.ndim)
            if keepdims or idx != dim
        )
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        for idx in range(new_tensor.nelement()):
            dims = Tensor(idx).unravel_index(shape)[0].shape
            indices = list(dims)
            indices.insert(dim, slice(None))  # type: ignore
            value = input[tuple(indices)]
            value = value.sum().item() if isinstance(value, Tensor) else value
            new_tensor[tuple(dims)] = value

    def SumBackward0() -> None:
        """Backward pass for the sum operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if None in (input.grad,):
            input.grad = Tensor(1, input.dtype)
        input.grad += slowtorch.ones(*input.shape)

    new_tensor.grad_fn = Node(SumBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def max(
    input: Tensor,
    dim: None | int = None,
    keepdims: bool = False,
) -> Tensor:
    """Return the maximum of elements in the tensor across a
    specified dimension.

    This function returns the maximum of all elements in the tensor if
    no dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be computed for maximum values.
    :param dim: The dimension along which to compute the maximum,
        defaults to `None`. For `None`, the maximum is computed over
        all elements of the tensor.
    :param keepdims: A boolean indicating whether to retain the
        reduced dimensions in the resulting tensor, defaults to
        `False`.
    :return: A new tensor containing the maximum of the specified
        elements.
    :raises ValueError: If the specified dimension is invalid.
    """
    if dim is not None and not (0 <= dim < input.ndim):
        raise ValueError(f"Invalid dimension {dim} for tensor")
    if dim is None:
        shape = 1 if not keepdims else (1,) * input.ndim
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = builtins.max(input._flat)
    else:
        shape = tuple(
            (1 if idx == dim and keepdims else input.shape[idx])
            for idx in range(input.ndim)
            if keepdims or idx != dim
        )
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        for idx in range(new_tensor.nelement()):
            dims = Tensor(idx).unravel_index(shape)[0].shape
            indices = list(dims)
            indices.insert(dim, slice(None))  # type: ignore
            value = input[tuple(indices)]
            value = value.max().item() if isinstance(value, Tensor) else value
            new_tensor[tuple(dims)] = value

    def MaxBackward0() -> None:
        """Backward pass for the maximum operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if None in (input.grad,):
            input.grad = Tensor(input.shape, input.dtype)
        N = range(builtins.max(input.shape))
        for dim in itertools.product(N, N):
            try:
                input.grad[dim] = (
                    1.0 if input[dim] == new_tensor.item() else 0.0
                )
            except IndexError:
                continue

    new_tensor.grad_fn = Node(MaxBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def min(
    input: Tensor,
    dim: None | int = None,
    keepdims: bool = False,
) -> Tensor:
    """Return the minimum of elements in the tensor across a
    specified dimension.

    This function returns the minimum of all elements in the tensor if
    no dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be computed for minimum values.
    :param dim: The dimension along which to compute the minimum,
        defaults to `None`. For `None`, the minimum is computed over
        all elements of the tensor.
    :param keepdims: A boolean indicating whether to retain the
        reduced dimensions in the resulting tensor, defaults to
        `False`.
    :return: A new tensor containing the minimum of the specified
        elements.
    :raises ValueError: If the specified dimension is invalid.
    """
    if dim is not None and not (0 <= dim < input.ndim):
        raise ValueError(f"Invalid dimension {dim} for tensor")
    if dim is None:
        shape = 1 if not keepdims else (1,) * input.ndim
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = builtins.min(input._flat)
    else:
        shape = tuple(
            (1 if idx == dim and keepdims else input.shape[idx])
            for idx in range(input.ndim)
            if keepdims or idx != dim
        )
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        for idx in range(new_tensor.nelement()):
            dims = Tensor(idx).unravel_index(shape)[0].shape
            indices = list(dims)
            indices.insert(dim, slice(None))  # type: ignore
            value = input[tuple(indices)]
            value = value.min().item() if isinstance(value, Tensor) else value
            new_tensor[tuple(dims)] = value

    def MinBackward0() -> None:
        """Backward pass for the minimum operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if None in (input.grad,):
            input.grad = Tensor(input.shape, input.dtype)
        N = range(builtins.max(input.shape))
        for dim in itertools.product(N, N):
            try:
                input.grad[dim] = (
                    1.0 if input[dim] == new_tensor.item() else 0.0
                )
            except IndexError:
                continue

    new_tensor.grad_fn = Node(MinBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def mean(
    input: Tensor,
    dim: None | int = None,
    keepdims: bool = False,
) -> Tensor:
    """Return the mean of elements in the tensor across a specified
    dimension.

    This function returns the mean of all elements in the tensor if
    no dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be computed for mean values.
    :param dim: The dimension along which to compute the mean,
        defaults to `None`. For `None`, the mean is computed over
        all elements of the tensor.
    :param keepdims: A boolean indicating whether to retain the
        reduced dimensions in the resulting tensor, defaults to
        `False`.
    :return: A new tensor containing the mean of the specified
        elements.
    :raises ValueError: If the specified dimension is invalid.
    """
    if dim is not None and not (0 <= dim < input.ndim):
        raise ValueError(f"Invalid dimension {dim} for tensor")
    if dim is None:
        shape = 1 if not keepdims else (1,) * input.ndim
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = statistics.mean(input._flat)
    else:
        shape = tuple(
            (1 if idx == dim and keepdims else input.shape[idx])
            for idx in range(input.ndim)
            if keepdims or idx != dim
        )
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        for idx in range(new_tensor.nelement()):
            dims = Tensor(idx).unravel_index(shape)[0].shape
            indices = list(dims)
            indices.insert(dim, slice(None))  # type: ignore
            value = input[tuple(indices)]
            value = value.mean().item() if isinstance(value, Tensor) else value
            new_tensor[tuple(dims)] = value

    def MeanBackward0() -> None:
        """Backward pass for the mean operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if None in (input.grad,):
            input.grad = Tensor(input.shape, input.dtype)
        N = range(builtins.max(input.shape))
        for dim in itertools.product(N, N):
            try:
                input.grad[dim] += new_tensor.grad.item() / input.nelement()
            except IndexError:
                continue

    new_tensor.grad_fn = Node(MeanBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def exp(input: Tensor) -> Tensor:
    """Perform element-wise exponentiation of the tensor.

    This function supports exponentiation. The resulting tensor is of the
    same shape and dtype as the input. The exponentiation function is
    defined as::

        exp(x) = math.exp(x)

    :param input: Input tensor to which exponentiation is to be applied.
    :return: A new tensor containing the result of the element-wise
        exponentiation.
    """
    new_tensor = Tensor(
        input.shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    new_tensor[:] = [math.exp(dim) for dim in input._flat]

    def ExpBackward0() -> None:
        """Backpropagation implementation for exponentiation.

        Computes gradient for `input` tensor and propagate it. The exp
        gradient is defined as::

            exp'(x) = math.exp(x)
        """
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if None in (input.grad,):
            input.grad = Tensor(1, input.dtype)
        input.grad += new_tensor.exp() * new_tensor.grad

    new_tensor.grad_fn = Node(ExpBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def sqrt(input: Tensor) -> Tensor:
    """Perform element-wise square root of a tensor.

    This function computes the square root of each element in the input
    tensor. The result is returned as a new tensor, and gradients are
    properly propagated during backpropagation.

    :param input: Input tensor containing the elements for which the
        square root is to be computed.
    :return: A new tensor containing the square root of each element
        in the input tensor.
    :raises ValueError: If the input tensor contains negative values.
    """
    new_tensor = Tensor(
        input.shape,
        slowtorch.float64,
        requires_grad=input.requires_grad,
    )
    data: list[Number] = []
    for idx in input._cdata:
        result = idx**0.5
        data.append(slowtorch.nan if isinstance(result, complex) else result)
    new_tensor[:] = data

    def SqrtBackward0() -> None:
        """Backpropagation implementation for square-roots.

        Computes gradient for `input` and propagate it.
        """
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if None in (input.grad,):
            input.grad = Tensor(1, input.dtype)
        input.grad += (0.5 * input ** (0.5 - 1)) * new_tensor.grad

    new_tensor.grad_fn = Node(SqrtBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def relu(input: Tensor) -> Tensor:
    """Apply the Rectified Linear Unit (ReLU) function element-wise.

    ReLU sets all negative values in the tensor to zero and keeps
    positive values unchanged. This operation is differentiable, and
    gradients are propagated only for positive elements. The relu
    function is defined as::

        relu(x) = max(x, 0)

    :param input: Input tensor to which ReLU is to be applied.
    :return: Output tensor after applying the ReLU function, with
        gradients linked for backpropagation.
    """
    new_tensor = Tensor(
        input.shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    if len(input.shape) == 1:
        new_tensor[:] = (safe_max(input[dim]) for dim in range(input.shape[0]))
    else:
        N = range(builtins.max(input.shape))
        for dim in itertools.product(N, N):
            try:
                new_tensor[dim] = safe_max(input[dim])
            except IndexError:
                continue

    def ReluBackward0() -> None:
        """Backpropagation implementation for ReLU.

        Computes gradients for `input` tensor and propagates them. The
        relu gradient is defined as::

            relu'(x) = 1 if x > 0 else 0

        .. seealso::

            [1] https://ml-cheatsheet.readthedocs.io/en/latest/
                activation_functions.html#relu
        """
        if None in (input.grad,):
            input.grad = Tensor(1, input.dtype)
        input.grad += (new_tensor > 0) * new_tensor.grad

    new_tensor.grad_fn = Node(ReluBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def elu(input: Tensor, alpha: float = 1.0) -> Tensor:
    """Apply the Exponential Linear Unit (ELU) function
    element-wise.

    ELU is a function that tend to converge cost to zero faster and
    produce more accurate results. This operation is differentiable,
    and gradients are propagated only for positive elements. The elu
    function is defined as::

        elu(x) = x if x >- 0 else alpha * (exp(x) - 1)

    :param input: Input tensor to which ELU is to be applied.
    :param alpha: Value for the ELU formulation, defaults to 1.0.
    :return: Output tensor after applying the ELU function, with
        gradients linked for backpropagation.
    """
    new_tensor = Tensor(
        input.shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    data: list[t.Any] = []
    if len(input.shape) == 1:
        iterator = range(input.shape[0])
    else:
        N = range(builtins.max(input.shape))
        iterator = itertools.product(N, N)
    for dim in iterator:
        try:
            if input[dim] <= 0:
                data.append(alpha * (normal_exp(input[dim]) - 1))
            else:
                data.append(input[dim])
        except IndexError:
            continue
    new_tensor[:] = data

    def EluBackward0() -> None:
        """Backpropagation implementation for ELU.

        Computes gradients for `input` tensor and propagates them. The
        elu gradient is defined as::

            elu'(x) = 1 if x > 0 else alpha * exp(x)

        .. seealso::

            [1] https://ml-cheatsheet.readthedocs.io/en/latest/
                activation_functions.html#elu
        """
        if None in (input.grad,):
            input.grad = Tensor(1, input.dtype)
        input.grad += (
            1.0 if new_tensor > 0 else alpha * normal_exp(new_tensor)
        ) * new_tensor.grad

    new_tensor.grad_fn = Node(EluBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def tanh(input: Tensor) -> Tensor:
    """Apply the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh squashes all the values between the range of -1 to 1. This
    operation is differentiable, and gradients are propagated. The
    tanh function is defined as::

        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    :param input: Input tensor to which Tanh is to be applied.
    :return: Output tensor after applying the Tanh function, with
        gradients linked for backpropagation.
    """
    new_tensor = Tensor(
        input.shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    if len(input.shape) == 1:
        new_tensor[:] = [
            ((x := normal_exp(input[dim])) - (y := safe_exp(input[dim])))
            / (x + y)
            for dim in range(input.shape[0])
        ]
    else:
        N = range(builtins.max(input.shape))
        for dim in itertools.product(N, N):
            try:
                new_tensor[dim] = (
                    (x := normal_exp(input[dim])) - (y := safe_exp(input[dim]))
                ) / (x + y)
            except IndexError:
                continue

    def TanhBackward0() -> None:
        """Backpropagation implementation for Tanh.

        Computes gradients for `input` tensor and propagates them. The
        tanh gradient is defined as::

            tanh'(x) = 1 - tanh(x)**2

        .. seealso::

            [1] https://ml-cheatsheet.readthedocs.io/en/latest/
                activation_functions.html#tanh
        """
        if None in (input.grad,):
            input.grad = Tensor(1, input.dtype)
        input.grad -= (1.0 - new_tensor**2) * new_tensor.grad

    new_tensor.grad_fn = Node(TanhBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def sigmoid(input: Tensor) -> Tensor:
    """Apply the Sigmoid function element-wise.

    Sigmoid function squashes between 0 and 1. This operation is
    differentiable, and gradients are propagated. The sigmoid function
    is defined as::

        sigmoid(x) = 1 / (1 + exp(-x))

    :param input: Input tensor to which Sigmoid is to be applied.
    :return: Output tensor after applying the Sigmoid function, with
        gradients linked for backpropagation.
    """
    new_tensor = Tensor(
        input.shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    data = []
    if len(input.shape) == 1:
        iterator = range(input.shape[0])
    else:
        N = range(builtins.max(input.shape))
        iterator = itertools.product(N, N)
    for dim in iterator:
        try:
            data.append(1.0 / (1 + safe_exp(input[dim])))
        except IndexError:
            continue
    new_tensor[:] = data

    def SigmoidBackward0() -> None:
        """Backpropagation implementation for Sigmoid.

        Computes gradients for `input` tensor and propagates them. The
        sigmoid gradient is defined as::

            sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

        .. seealso::

            [1] https://ml-cheatsheet.readthedocs.io/en/latest/
                activation_functions.html#sigmoid
        """
        if None in (input.grad,):
            input.grad = Tensor(1, input.dtype)
        input.grad -= (new_tensor * (1 - new_tensor)) * new_tensor.grad

    new_tensor.grad_fn = Node(SigmoidBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


def linear(
    input: Tensor, weight: Tensor, bias: None | Tensor = None
) -> Tensor:
    """Compute the linear transformation on the input tensor `input` as
    follows::

        output = input @ weight.T + bias

    This function performs a matrix multiplication of the `input` tensor
    and the transpose of the `weight` tensor, followed by adding the
    optional `bias` tensor, if provided. This is the standard operation
    for a dense layer in neural networks.

    :param input: The input tensor to be transformed.
    :param weight: The weights tensor to be transposed and matrix
        multiplied.
    :param bias: Optional bias tensor to add, defaults to `None`.
    :return: The output tensor resulting from the linear transformation.
    :raises ValueError: If `bias` is provided and its shape is
        incompatible with the output shape.
    """
    new_tensor = input @ weight.T
    if bias is not None:
        if bias.shape != (new_tensor.shape[-1],):
            raise ValueError(
                f"Bias {bias.shape} is incompatible with output shape "
                f"{new_tensor.shape}"
            )
        new_tensor += bias

    def AddmmBackward0() -> None:
        """Backpropagation implementation for Linear transformation.

        Computes gradients for the `input`, `weight`, and `bias` tensors
        and propagates them backward through the computational graph.
        """
        if None in (input.grad, weight.grad, bias.grad):
            input.grad = weight.grad = bias.grad = Tensor(1, input.dtype)
        input.grad += new_tensor.grad @ weight
        weight.grad += new_tensor.grad.T @ input
        if bias is not None:
            bias.grad += new_tensor.grad.sum(dim=0)

    new_tensor.grad_fn = Node(AddmmBackward0)
    new_tensor.grad_fn.inputs = (input, weight, bias)
    return new_tensor


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Compute the Mean Squared Error (MSE) loss between two tensors.

    This function calculates the average of the squared differences
    between `input` and `target` tensors. It supports autograd for
    backpropagation.

    :param input: The input tensor representing predictions.
    :param target: The target tensor representing true values.
    :param reduction: The reduction function to apply to the computed
        loss. Options are:
        - `mean`: Return the average of the squared differences.
        - `sum`: Return the sum of the squared differences.
        - `none`: Return the squared differences without reduction.
        Defaults to `mean`.
    :return: A scalar tensor representing the MSE loss.
    """
    loss = (input - target) ** 2
    if reduction == "mean":
        new_tensor = loss.sum() * (1.0 / loss.nelement())
    elif reduction == "sum":
        new_tensor = loss.sum()
    elif reduction == "none":
        new_tensor = loss

    def MSELossBackward0() -> None:
        """Backpropagation implementation for MSE loss.

        Computes gradients for the `input` and `target` tensors and
        propagates them backward through the computational graph.
        """
        if None in (input.grad, target.grad):
            input.grad = target.grad = Tensor(1, input.dtype)
        grad = 2.0 / loss.nelement() if reduction == "mean" else 2.0
        input.grad += grad * (input - target)
        target.grad -= grad * (input - target)

    new_tensor.grad_fn = Node(MSELossBackward0)
    new_tensor.grad_fn.inputs = (input, target)
    return new_tensor
