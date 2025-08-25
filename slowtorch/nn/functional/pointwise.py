"""\
SlowTorch Stateless Pointwise
=============================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, 01 June 2025
Last updated on: Sunday, 24 August 2025

Pointwise operations.

This module offers various stateless functions that perform arithmetic
operations. These functions mimic the API and semantics of PyTorch's
pointwise functions.

.. note::

    All functions support auto-differentiation (backpropagation).
"""

from __future__ import annotations

import builtins
import math
import typing as t
from itertools import product as pdt

import slowtorch
from slowtorch import function_dispatch
from slowtorch.internal.shape import broadcast_shapes
from slowtorch.internal.tensor import Node
from slowtorch.internal.tensor import Tensor
from slowtorch.utils import py_impl_max
from slowtorch.utils import py_impl_nexp

if t.TYPE_CHECKING:
    from slowtorch.types import Dim
    from slowtorch.types import Dtype
    from slowtorch.types import FloatLikeType
    from slowtorch.types import Input
    from slowtorch.types import StorageWeakRef


@function_dispatch
def add(input: Tensor, other: Input) -> Tensor:
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
    """
    if isinstance(other, (int, bool, float)):
        scalar = other
        other = Tensor(1, input.dtype, requires_grad=input.requires_grad)
        other[:] = scalar
    elif not isinstance(other, Tensor):
        raise TypeError(
            f"Unsupported operand type(s) for +: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )
    dtype = (
        slowtorch.float32
        if input.dtype.name.startswith("float")
        or other.dtype.name.startswith("float")
        else slowtorch.int64
    )
    shape = broadcast_shapes(input._shape, other._shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    for offset in range(len(input.storage)):
        new_tensor.storage[offset] = (
            input.storage[offset] + other.storage[offset]
        )

    def AddBackward0() -> None:
        """Backpropagation implementation for addition.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        grad = new_tensor.grad
        if None in (input.grad, other.grad):
            input.grad = Tensor(input._shape, dtype)
            other.grad = Tensor(other._shape, dtype)
        input.grad += grad
        other.grad += grad

    new_tensor.grad_fn = Node(AddBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
def sub(input: Tensor, other: Input) -> Tensor:
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
    """
    if isinstance(other, (int, bool, float)):
        scalar = other
        other = Tensor(1, input.dtype, requires_grad=input.requires_grad)
        other[:] = scalar
    elif not isinstance(other, Tensor):
        raise TypeError(
            f"Unsupported operand type(s) for -: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )
    dtype = (
        slowtorch.float32
        if input.dtype.name.startswith("float")
        or other.dtype.name.startswith("float")
        else slowtorch.int64
    )
    shape = broadcast_shapes(input._shape, other._shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    for offset in range(len(input.storage)):
        new_tensor.storage[offset] = (
            input.storage[offset] - other.storage[offset]
        )

    def SubBackward0() -> None:
        """Backpropagation implementation for subtraction.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        grad = new_tensor.grad
        if None in (input.grad, other.grad):
            input.grad = Tensor(input._shape, dtype)
            other.grad = Tensor(other._shape, dtype)
        input.grad += grad
        other.grad -= grad

    new_tensor.grad_fn = Node(SubBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
def mul(input: Tensor, other: Input) -> Tensor:
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
    """
    if isinstance(other, (int, bool, float)):
        scalar = other
        other = Tensor(1, input.dtype, requires_grad=input.requires_grad)
        other[:] = scalar
    elif not isinstance(other, Tensor):
        raise TypeError(
            f"Unsupported operand type(s) for *: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )
    dtype = (
        slowtorch.float32
        if input.dtype.name.startswith("float")
        or other.dtype.name.startswith("float")
        else slowtorch.int64
    )
    shape = broadcast_shapes(input._shape, other._shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    for offset in range(len(input.storage)):
        new_tensor.storage[offset] = (
            input.storage[offset] * other.storage[offset]
        )

    def MulBackward0() -> None:
        """Backpropagation implementation for multiplication.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        grad = new_tensor.grad
        if None in (input.grad, other.grad):
            input.grad = Tensor(input._shape, dtype)
            other.grad = Tensor(other._shape, dtype)
        input.grad += other * grad
        other.grad += input * grad

    new_tensor.grad_fn = Node(MulBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
def div(
    input: Tensor,
    other: Input,
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
    """
    if isinstance(other, (int, bool, float)):
        scalar = other
        other = Tensor(1, slowtorch.float32, requires_grad=input.requires_grad)
        other[:] = scalar
    elif not isinstance(other, Tensor):
        raise TypeError(
            f"Unsupported operand type(s) for /: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )
    dtype = (
        slowtorch.float32
        if input.dtype.name.startswith("float")
        or other.dtype.name.startswith("float")
        else slowtorch.int64
    )
    shape = broadcast_shapes(input._shape, other._shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    floor = rounding_mode == "floor"
    for offset in range(len(input.storage)):
        try:
            if floor:
                storage = input.storage[offset] // other.storage[offset]
            else:
                storage = input.storage[offset] / other.storage[offset]
        except ZeroDivisionError:
            storage = slowtorch.inf
        new_tensor.storage[offset] = storage

    def DivBackward0() -> None:
        """Backpropagation implementation for division.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        grad = new_tensor.grad
        if input.grad is None:
            input.grad = Tensor(input._shape, slowtorch.float32)
        if other.grad is None:
            other.grad = Tensor(input._shape, slowtorch.float32)
        input.grad += grad / other
        other.grad += (-grad * input) / (other * other)

    new_tensor.grad_fn = Node(DivBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
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
        match `input._shape`.
    """
    if not isinstance(other, Tensor):
        raise TypeError(
            f"Unsupported operand type(s) for @: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )
    dtype = (
        slowtorch.float32
        if input.dtype.name.startswith("float")
        or other.dtype.name.startswith("float")
        else slowtorch.int64
    )
    requires_grad = input.requires_grad or other.requires_grad
    if input.ndim == 1 and other.ndim == 1:
        if input._shape[0] != other._shape[0]:
            raise ValueError(
                "Shapes of 1D tensors must be the same for dot product"
            )
        accumulated_storage = 0
        input_storage, other_storage = input.storage, other.storage
        input_stride = input._strides[0] // input.itemsize
        other_stride = other._strides[0] // other.itemsize
        input_storage_offset = input._storage_offset
        other_storage_offset = other._storage_offset
        for index in range(input._shape[0]):
            accumulated_storage += (
                input_storage[input_storage_offset + index * input_stride]
                * other_storage[other_storage_offset + index * other_stride]
            )
        new_tensor = Tensor(1, dtype, requires_grad=requires_grad)
        new_tensor[...] = accumulated_storage
    elif input.ndim == 2 and other.ndim == 2:
        if input._shape[1] != other._shape[0]:
            raise ValueError(
                "Shapes are not aligned for matrix multiplication"
            )
        M, K = input._shape
        _, N = other._shape
        new_tensor = Tensor((M, N), dtype, requires_grad=requires_grad)
        input_storage, other_storage, new_storage = (
            input.storage,
            other.storage,
            new_tensor.storage,
        )
        input_stride_outer, input_stride_inner = (
            input._strides[0] // input.itemsize,
            input._strides[1] // input.itemsize,
        )
        other_stride_outer, other_stride_inner = (
            other._strides[0] // other.itemsize,
            other._strides[1] // other.itemsize,
        )
        new_stride_outer, new_stride_inner = (
            new_tensor._strides[0] // new_tensor.itemsize,
            new_tensor._strides[1] // new_tensor.itemsize,
        )
        input_storage_offset, other_storage_offset, new_storage_offset = (
            input._storage_offset,
            other._storage_offset,
            new_tensor._storage_offset,
        )
        for m in range(M):
            for n in range(N):
                accumulated_storage = 0
                for k in range(K):
                    accumulated_storage += (
                        input_storage[
                            input_storage_offset
                            + m * input_stride_outer
                            + k * input_stride_inner
                        ]
                        * other_storage[
                            other_storage_offset
                            + k * other_stride_outer
                            + n * other_stride_inner
                        ]
                    )
                new_storage[
                    new_storage_offset
                    + m * new_stride_outer
                    + n * new_stride_inner
                ] = accumulated_storage
    elif input.ndim > 2 or other.ndim > 2:
        shape = broadcast_shapes(input._shape[:-2], other._shape[:-2]) + (
            input._shape[-2],
            other._shape[-1],
        )
        input = input.broadcast_to(shape[:-2] + input._shape[-2:])
        other = other.broadcast_to(shape[:-2] + other._shape[-2:])
        new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)

        for batch in pdt(*[range(index) for index in new_tensor._shape[:-2]]):
            for idx in range(new_tensor._shape[-2]):
                for jdx in range(new_tensor._shape[-1]):
                    accumulated_storage = 0
                    for kdx in range(input._shape[-1]):
                        accumulated_storage += (
                            input[(*batch, idx, kdx)]
                            * other[(*batch, kdx, jdx)]
                        )
                    new_tensor[(*batch, idx, jdx)] = accumulated_storage
    else:
        raise ValueError("Invalid shapes for dot product")

    def DotBackward0() -> None:
        """Backpropagation implementation for matrix multiplication.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        grad = new_tensor.grad
        if input.grad is None:
            input.grad = Tensor(input._shape, dtype)
        if other.grad is None:
            other.grad = Tensor(other._shape, dtype)
        if input.ndim == 1 and other.ndim == 1:
            input.grad += grad * other
            other.grad += grad * input
        elif input.ndim == 2 and other.ndim == 2:
            input.grad += grad @ other.transpose(1, 0)
            other.grad += input.transpose(1, 0) @ grad
        else:
            if other.ndim >= 2:
                _other = other.transpose(-2, -1)
            else:
                _other = other
            if input.ndim >= 2:
                _input = input.transpose(-2, -1)
            else:
                _input = input
            input.grad += grad @ _other
            other.grad += _input @ grad

    new_tensor.grad_fn = Node(DotBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
def neg(input: Tensor) -> Tensor:
    """Perform element-wise negation of the tensor.

    This function negates the input tensor and returns the tensor with
    same shape and dtype but negated.

    :param input: Input tensor to be multiplied.
    :return: A new tensor containing the result of the element-wise
        negation.
    """
    new_tensor = Tensor(
        input._shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    new_tensor[:] = (data * -1 for data in input)

    def NegBackward0() -> None:
        """Backpropagation implementation for negation.

        Computes gradients for `input` and propagates them.
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        input.grad += -1 * new_tensor.grad

    new_tensor.grad_fn = Node(NegBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


negative = neg


@function_dispatch
def remainder(input: Tensor, other: Input) -> Tensor:
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
    """
    if isinstance(other, (int, bool, float)):
        scalar = other
        other = Tensor(1, input.dtype, requires_grad=input.requires_grad)
        other[:] = scalar
    elif not isinstance(other, Tensor):
        raise TypeError(
            f"Unsupported operand type(s) for %: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )
    dtype = (
        slowtorch.float32
        if input.dtype.name.startswith("float")
        or other.dtype.name.startswith("float")
        else slowtorch.int64
    )
    shape = broadcast_shapes(input._shape, other._shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    new_tensor[:] = (x % y for x, y in zip(input, other))

    def RemainderBackward0() -> None:
        """Backpropagation implementation for modulo operation.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        grad = new_tensor.grad
        if None in (input.grad, other.grad):
            input.grad = Tensor(input._shape, dtype)
            other.grad = Tensor(other._shape, dtype)
        input.grad += grad
        other.grad -= (input // other) * grad

    new_tensor.grad_fn = Node(RemainderBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
def pow(input: Tensor, other: Input) -> Tensor:
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
    """
    if isinstance(other, (int, bool, float)):
        scalar = other
        other = Tensor(1, input.dtype, requires_grad=input.requires_grad)
        other[:] = scalar
    elif not isinstance(other, Tensor):
        raise TypeError(
            f"Unsupported operand type(s) for **: {type(input).__name__!r} "
            f"and {type(other).__name__!r}"
        )
    dtype = (
        slowtorch.float32
        if input.dtype.name.startswith("float")
        or other.dtype.name.startswith("float")
        else slowtorch.int64
    )
    shape = broadcast_shapes(input._shape, other._shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    new_tensor[:] = (x**y for x, y in zip(input, other))

    def PowBackward0() -> None:
        """Backpropagation implementation for exponentiation.

        Computes gradient for `input` and propagate it.
        """
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        input.grad += (other * input ** (other - 1)) * new_tensor.grad

    new_tensor.grad_fn = Node(PowBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def abs(input: Tensor) -> Tensor:
    """Return a new tensor with the absolute value of the elements of
    input.

    This function creates a new `Tensor` instance with the same shape
    and absolute values of the original tensor's elements.

    :param input: Input tensor.
    :return: A new tensor with the absolute values of the data and
        shape.
    """
    shape, requires_grad = input._shape, input.requires_grad
    new_tensor = Tensor(shape, slowtorch.float32, requires_grad=requires_grad)
    new_tensor[:] = (builtins.abs(x) for x in input)

    def AbsBackward0() -> None:
        """Backpropagation implementation for absolute value.

        Computes gradient for `input` and propagate it.
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, slowtorch.float32)
        storage: StorageWeakRef = []
        for x in input:
            if x > 0:
                storage.append(1.0)
            elif x < 0:
                storage.append(-1.0)
            else:
                storage.append(0.0)
        abs_tensor = Tensor(input._shape, slowtorch.float32)
        abs_tensor[:] = storage
        input.grad += new_tensor.grad * abs_tensor

    new_tensor.grad_fn = Node(AbsBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def log(input: Tensor) -> Tensor:
    """Return a new tensor with the natural logarithm of the elements
    of input.

    This function creates a new `Tensor` instance with the same shape
    and as the original tensor but with natural logarithm calculated.

    :param input: Input tensor.
    :return: A new tensor with the log calculated data and shape.
    """
    shape, requires_grad = input._shape, input.requires_grad
    new_tensor = Tensor(shape, slowtorch.float32, requires_grad=requires_grad)
    storage: StorageWeakRef = []
    for index in input:
        try:
            storage.append(math.log(index))
        except ValueError:
            storage.append(slowtorch.nan)
    new_tensor[:] = storage

    def LogBackward0() -> None:
        """Backpropagation implementation for logarithm.

        Computes gradient for `input` and propagate it.
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, slowtorch.float32)
        input.grad += new_tensor.grad / input

    new_tensor.grad_fn = Node(LogBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
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
        input._shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    new_tensor[:] = (math.exp(index) for index in input)

    def ExpBackward0() -> None:
        """Backpropagation implementation for exponentiation.

        Computes gradient for `input` tensor and propagate it. The exp
        gradient is defined as::

            exp'(x) = math.exp(x)
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        for idx in pdt(*[range(dim) for dim in input._shape]):
            input.grad[idx] += new_tensor[idx] * new_tensor.grad[idx]

    new_tensor.grad_fn = Node(ExpBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def sqrt(input: Tensor) -> Tensor:
    """Perform element-wise square root of a tensor.

    This function computes the square root of each element in the input
    tensor. The result is returned as a new tensor, and gradients are
    properly propagated during backpropagation.

    :param input: Input tensor containing the elements for which the
        square root is to be computed.
    :return: A new tensor containing the square root of each element
        in the input tensor.
    """
    new_tensor = Tensor(
        input._shape,
        slowtorch.float32,
        requires_grad=input.requires_grad,
    )
    storage: StorageWeakRef = []
    for index in input:
        root = index**0.5
        storage.append(slowtorch.nan if isinstance(root, complex) else root)
    new_tensor[:] = storage

    def SqrtBackward0() -> None:
        """Backpropagation implementation for square-roots.

        Computes gradient for `input` and propagate it.
        """
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        input.grad += (0.5 * input ** (0.5 - 1)) * new_tensor.grad

    new_tensor.grad_fn = Node(SqrtBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
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
        input._shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    if len(input._shape) == 1:
        iterator = range(input._shape[0])
    else:
        iterator = pdt(*[range(index) for index in input._shape])
    for index in iterator:
        try:
            new_tensor[index] = py_impl_max(input[index])
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
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        input.grad += (new_tensor > 0) * new_tensor.grad

    new_tensor.grad_fn = Node(ReluBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def elu(input: Tensor, alpha: FloatLikeType = 1.0) -> Tensor:
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
        input._shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    storage: StorageWeakRef = []
    if len(input._shape) == 1:
        iterator = range(input._shape[0])
    else:
        iterator = pdt(*[range(index) for index in input._shape])
    for index in iterator:
        try:
            if input[index] <= 0:
                storage.append(alpha * (math.exp(input[index]) - 1))
            else:
                storage.append(input[index])
        except IndexError:
            continue
    new_tensor[:] = storage

    def EluBackward0() -> None:
        """Backpropagation implementation for ELU.

        Computes gradients for `input` tensor and propagates them. The
        elu gradient is defined as::

            elu'(x) = 1 if x > 0 else alpha * exp(x)

        .. seealso::

            [1] https://ml-cheatsheet.readthedocs.io/en/latest/
                activation_functions.html#elu
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        grad = Tensor(input._shape, slowtorch.float32)
        for idx in pdt(*[range(dim) for dim in input._shape]):
            grad[idx] = 1.0 if input[idx] > 0 else alpha * math.exp(input[idx])
        input.grad += grad * new_tensor.grad

    new_tensor.grad_fn = Node(EluBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
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
        input._shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    if len(input._shape) == 1:
        iterator = range(input._shape[0])
    else:
        iterator = pdt(*[range(index) for index in input._shape])
    for index in iterator:
        try:
            new_tensor[index] = (
                (x := math.exp(input[index]))
                - (y := py_impl_nexp(input[index]))
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
        grad = new_tensor.grad
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        input.grad -= (1.0 - new_tensor**2) * grad

    new_tensor.grad_fn = Node(TanhBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
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
        input._shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    storage: StorageWeakRef = []
    if len(input._shape) == 1:
        iterator = range(input._shape[0])
    else:
        iterator = pdt(*[range(index) for index in input._shape])
    for index in iterator:
        try:
            storage.append(1.0 / (1.0 + py_impl_nexp(input[index])))
        except IndexError:
            continue
    new_tensor[:] = storage

    def SigmoidBackward0() -> None:
        """Backpropagation implementation for Sigmoid.

        Computes gradients for `input` tensor and propagates them. The
        sigmoid gradient is defined as::

            sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

        .. seealso::

            [1] https://ml-cheatsheet.readthedocs.io/en/latest/
                activation_functions.html#sigmoid
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        input.grad -= (new_tensor * (1 - new_tensor)) * new_tensor.grad

    new_tensor.grad_fn = Node(SigmoidBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def softmax(
    input: Tensor,
    dim: None | Dim = None,
    dtype: None | Dtype = None,
) -> Tensor:
    """Apply the Softmax function element-wise.

    Softmax function squashes between 0 and 1, along the provided
    dimension and sum to 1. This operation is differentiable, and
    gradients are propagated. The softmax function is defined as::

        softmax(x) = exp(x) / exp(x).sum()

    :param input: Input tensor to which Softmax is to be applied.
    :param dim: A dimension along which softmax will be computed,
        defaults to `None`.
    :return: Output tensor after applying the Softmax function, with
        gradients linked for backpropagation.
    """
    new_tensor = Tensor(
        input._shape,
        dtype or input.dtype,
        requires_grad=input.requires_grad,
    )
    numerator = (input - input.max(dim=dim, keepdim=True)).exp()
    denominator = numerator.sum(dim=dim, keepdim=True)
    new_tensor[:] = numerator / denominator

    def SoftmaxBackward0() -> None:
        """Backpropagation implementation for Softmax.

        Computes gradients for `input` tensor and propagates them. The
        sigmoid gradient is defined as::

            softmax'(x) = (
                softmax(x_i) * (1 - softmax(x_j))
                    if i == j
                    else
                -softmax(x_j) * softmax(x_1)
            )

        .. seealso::

            [1] https://eli.thegreenplace.net/2016/
                the-softmax-function-and-its-derivative/
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        input.grad += new_tensor * (
            new_tensor.grad
            - (new_tensor.grad * new_tensor).sum(dim=dim, keepdim=True)
        )

    new_tensor.grad_fn = Node(SoftmaxBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def log_softmax(
    input: Tensor,
    dim: None | Dim = None,
    dtype: None | Dtype = None,
) -> Tensor:
    """Apply the Softmax function followed by Logarithm.

    This is mathematically equivalent to applying softmax function
    followed by logarith. This operation is differentiable, and
    gradients are propagated.

    :param input: Input tensor to which Log + Softmax is to be applied.
    :param dim: A dimension along which softmax will be computed,
        defaults to `None`.
    :return: Output tensor after applying the Log softmax function, with
        gradients linked for backpropagation.
    """
    x = softmax(input, dim, dtype)
    new_tensor = log(x)

    def LogSoftmaxBackward0() -> None:
        """Backpropagation implementation for Logsoftmax.

        Computes gradients for `input` tensor and propagates them.
        """
        grad = new_tensor.grad - x * new_tensor.grad.sum(dim=dim, keepdim=True)
        if input.grad is None:
            input.grad = grad
        else:
            input.grad += grad

    new_tensor.grad_fn = Node(LogSoftmaxBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor
