"""\
SlowTorch Neural Network related Functions API
==============================================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Wednesday, January 15 2025
Last updated on: Saturday, May 24 2025

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
from slowtorch import function_dispatch
from slowtorch._tensor import Node
from slowtorch._tensor import Tensor
from slowtorch._types import Number
from slowtorch._utils import Dtype
from slowtorch._utils import broadcast_shapes
from slowtorch._utils import normal_exp
from slowtorch._utils import safe_exp
from slowtorch._utils import safe_max
from slowtorch._utils import unravel_index


@function_dispatch
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
    if isinstance(other, Number):
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
    shape = broadcast_shapes(input.shape, other.shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    new_tensor[:] = (x + y for x, y in zip(input._flat, other._flat))

    def AddBackward0() -> None:
        """Backpropagation implementation for addition.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = Tensor(input.shape, dtype)
            other.grad = Tensor(input.shape, dtype)
        grad = new_tensor.grad
        input.grad += grad
        other.grad += grad

    new_tensor.grad_fn = Node(AddBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
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
    if isinstance(other, Number):
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
    shape = broadcast_shapes(input.shape, other.shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    new_tensor[:] = (x - y for x, y in zip(input._flat, other._flat))

    def SubBackward0() -> None:
        """Backpropagation implementation for subtraction.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = Tensor(input.shape, dtype)
            other.grad = Tensor(input.shape, dtype)
        grad = new_tensor.grad
        input.grad += grad
        other.grad -= grad

    new_tensor.grad_fn = Node(SubBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
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
    if isinstance(other, Number):
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
    shape = broadcast_shapes(input.shape, other.shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    new_tensor[:] = (x * y for x, y in zip(input._flat, other._flat))

    def MulBackward0() -> None:
        """Backpropagation implementation for multiplication.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = Tensor(input.shape, dtype)
            other.grad = Tensor(input.shape, dtype)
        grad = new_tensor.grad
        input.grad += other * grad
        other.grad += input * grad

    new_tensor.grad_fn = Node(MulBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
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
    data: list[Number] = []
    if isinstance(other, Number):
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
    shape = broadcast_shapes(input.shape, other.shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    for x, y in zip(input._flat, other._flat):
        try:
            data.append(x // y if rounding_mode == "floor" else x / y)
        except ZeroDivisionError:
            data.append(builtins.float("inf"))
    new_tensor[:] = data

    def DivBackward0() -> None:
        """Backpropagation implementation for division.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = Tensor(input.shape, slowtorch.float32)
            other.grad = Tensor(input.shape, slowtorch.float32)
        grad = new_tensor.grad
        input.grad += grad / other
        other.grad -= (input / (other * other)) * grad

    new_tensor.grad_fn = Node(DivBackward0)
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
        input.shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    new_tensor[:] = (data * -1 for data in input._flat)

    def NegBackward0() -> None:
        """Backpropagation implementation for negation.

        Computes gradients for `input` and propagates them.
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        input.grad += -1 * new_tensor.grad

    new_tensor.grad_fn = Node(NegBackward0)
    new_tensor.grad_fn.inputs = (input,)
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
        match `input.shape`.
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
        if input.shape[0] != other.shape[0]:
            raise ValueError(
                "Shapes of 1D tensors must be the same for dot product"
            )
        new_tensor = Tensor(input.shape, dtype, requires_grad=requires_grad)
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
        other = other.broadcast_to(shape[:-2] + other.shape[-2:])
        requires_grad = input.requires_grad or other.requires_grad
        new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
        for batch in itertools.product(
            *[range(dim) for dim in new_tensor.shape[:-2]]
        ):
            for idx in range(new_tensor.shape[-2]):
                for jdx in range(new_tensor.shape[-1]):
                    new_tensor[(*batch, idx, jdx)] = builtins.sum(
                        input[(*batch, idx, kdx)] * other[(*batch, kdx, jdx)]
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
            input.grad = Tensor(input.shape, dtype)
            other.grad = Tensor(input.shape, dtype)
        grad = new_tensor.grad
        input.grad += grad @ other
        other.grad += input @ grad

    new_tensor.grad_fn = Node(DotBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
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
    if isinstance(other, Number):
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
    shape = broadcast_shapes(input.shape, other.shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    new_tensor[:] = (x % y for x, y in zip(input._flat, other._flat))

    def RemainderBackward0() -> None:
        """Backpropagation implementation for modulo operation.

        Computes gradients for `input` and `other` and propagates
        them.
        """
        if None in (input.grad, other.grad):
            input.grad = Tensor(input.shape, dtype)
            other.grad = Tensor(input.shape, dtype)
        grad = new_tensor.grad
        input.grad += grad
        other.grad -= (input // other) * grad

    new_tensor.grad_fn = Node(RemainderBackward0)
    new_tensor.grad_fn.inputs = (input, other)
    return new_tensor


@function_dispatch
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
    if isinstance(other, Number):
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
    shape = broadcast_shapes(input.shape, other.shape)
    input = input.broadcast_to(shape)
    other = other.broadcast_to(shape)
    requires_grad = input.requires_grad or other.requires_grad
    new_tensor = Tensor(shape, dtype, requires_grad=requires_grad)
    new_tensor[:] = (x**y for x, y in zip(input._flat, other._flat))

    def PowBackward0() -> None:
        """Backpropagation implementation for exponentiation.

        Computes gradient for `input` and propagate it.
        """
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        input.grad += (other * input ** (other - 1)) * new_tensor.grad

    new_tensor.grad_fn = Node(PowBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def abs(input: Tensor) -> Tensor:
    """Return a new tensor with the absolute value of the elements of
    input.

    This method creates a new `Tensor` instance with the same shape and
    absolute values of the original tensor's elements.

    :param input: Input tensor.
    :return: A new tensor with the absolute values of the data and
        shape.
    """
    shape, requires_grad = input.shape, input.requires_grad
    new_tensor = Tensor(shape, slowtorch.float32, requires_grad=requires_grad)
    data: list[Number] = [builtins.abs(x) for x in input._flat]
    new_tensor[:] = data

    def AbsBackward0() -> None:
        """Backpropagation implementation for absolute value.

        Computes gradient for `input` and propagate it.
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, slowtorch.float32)
        data: list[Number] = []
        for x in input._flat:
            if x > 0:
                data.append(1.0)
            elif x < 0:
                data.append(-1.0)
            else:
                data.append(0.0)
        sign = Tensor(input.shape, slowtorch.float32)
        sign[:] = data
        input.grad += new_tensor.grad * sign

    new_tensor.grad_fn = Node(AbsBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def log(input: Tensor) -> Tensor:
    """Return a new tensor with the natural logarithm of the elements
    of input.

    This method creates a new `Tensor` instance with the same shape and
    as the original tensor but with natural logarithm calculated.

    :param input: Input tensor.
    :return: A new tensor with the log calculated data and shape.
    """
    shape, requires_grad = input.shape, input.requires_grad
    new_tensor = Tensor(shape, slowtorch.float32, requires_grad=requires_grad)
    data: list[Number] = []
    for idx in input._flat:
        try:
            data.append(math.log(idx))
        except ValueError:
            data.append(slowtorch.nan)
    new_tensor[:] = data

    def LogBackward0() -> None:
        """Backpropagation implementation for logarithm.

        Computes gradient for `input` and propagate it.
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, slowtorch.float32)
        input.grad += new_tensor.grad / input

    new_tensor.grad_fn = Node(LogBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
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
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = new_tensor.grad

    new_tensor.grad_fn = Node(CloneBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def ravel(input: Tensor) -> Tensor:
    """Return a contiguous flattened tensor.

    This method creates a copy of the tensor collapsed into one
    dimension.

    :param input: Input tensor to be flattened.
    :return: A new tensor with the same data, and type as the original
        tensor but in 1-D.
    """
    new_tensor = input.ravel()

    def ViewBackward0() -> None:
        """Backpropagation implementation for cloning.

        Computes gradient for `input` and propagate it.
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = new_tensor.grad

    new_tensor.grad_fn = Node(ViewBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    """Transpose the tensor by permuting its dimensions.

    This method returns a view of the tensor with its dimensions
    permuted. If no dimensions are specified, the dimensions are
    reversed (i.e., equivalent to a full transpose).

    :param input: Input tensor to be transposed.
    :param dim0: First dimension to be transposed.
    :param dim1: Second dimension to be transposed.
    :return: A new tensor view with transposed dimensions.
    :raises ValueError: If the provided dimensions are invalid.
    """
    if sorted((dim0, dim1)) != list(range(input.ndim)):
        raise ValueError("Invalid dimensions permutation")
    dims = tuple(reversed(sorted((dim0, dim1))))
    shape = tuple(input.shape[dim] for dim in dims)
    strides = tuple(input._strides[dim] for dim in dims)
    new_tensor = input.view_(tuple(shape), tuple(strides))

    def PermuteBackward0() -> None:
        """Backpropagation implementation for transpose.

        Computes gradients for `input` tensor and propagates them.
        This is achieved by reversing the transpose operation during
        the backward pass by swapping the same dimensions (dim0,
        dim1) in the gradient tensor.
        """
        if new_tensor.grad is not None:
            input.grad = new_tensor.grad.transpose(dim0, dim1)

    new_tensor.grad_fn = Node(PermuteBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def sum(
    input: Tensor,
    dim: None | int = None,
    keepdim: bool = False,
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
    :param keepdim: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the sum of the specified elements.
    :raises ValueError: If the specified dimension is invalid.
    """
    ndim = input.ndim
    if dim is not None:
        if not (-ndim <= dim < ndim):
            raise IndexError(
                "Dimension out of range (expected to be in the range "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim
    if dim is None:
        shape = 1 if not keepdim else (1,) * ndim
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = builtins.sum(input._flat)
    else:
        shape = tuple(
            (1 if idx == dim and keepdim else input.shape[idx])
            for idx in range(ndim)
            if keepdim or idx != dim
        )
        new_tensor = Tensor(
            shape if shape else (1,),
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        for idx in range(new_tensor.nelement()):
            unravelled = unravel_index(idx, shape)
            indices = list(unravelled)
            if keepdim:
                indices = tuple(
                    indices[idx] if idx != dim else slice(None)
                    for idx in range(len(indices))
                )
            else:
                indices = indices[:dim] + [slice(None)] + indices[dim:]
            new_tensor[unravelled] = input[tuple(indices)].sum().item()

    def SumBackward0() -> None:
        """Backward pass for the sum operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        input.grad += slowtorch.ones(*input.shape)

    new_tensor.grad_fn = Node(SumBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def max(
    input: Tensor,
    dim: None | int = None,
    keepdim: bool = False,
) -> Tensor:
    """Return the maximum of elements in the tensor across a
    specified dimension.

    This function returns the maximum of all elements in the tensor if
    no dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be computed for maximum values.
    :param dim: The dimension along which to compute the maximum,
        defaults to `None`. For `None`, the maximum is computed over all
        elements of the tensor.
    :param keepdim: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the maximum of the specified
        elements.
    :raises ValueError: If the specified dimension is invalid.
    """
    ndim = input.ndim
    if dim is not None:
        if not (-ndim <= dim < ndim):
            raise IndexError(
                "Dimension out of range (expected to be in the range "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim
    if dim is None:
        shape = 1 if not keepdim else (1,) * ndim
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = builtins.max(input._flat)
    else:
        shape = tuple(
            (1 if idx == dim and keepdim else input.shape[idx])
            for idx in range(ndim)
            if keepdim or idx != dim
        )
        new_tensor = Tensor(
            shape if shape else (1,),
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        for idx in range(new_tensor.nelement()):
            unravelled = unravel_index(idx, shape)
            indices = list(unravelled)
            if keepdim:
                indices = tuple(
                    indices[idx] if idx != dim else slice(None)
                    for idx in range(len(indices))
                )
            else:
                indices = indices[:dim] + [slice(None)] + indices[dim:]
            new_tensor[unravelled] = input[tuple(indices)].max().item()

    def MaxBackward0() -> None:
        """Backward pass for the maximum operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        if dim is None:
            for idx in itertools.product(*[range(dim) for dim in input.shape]):
                if input[idx] == new_tensor.item():
                    input.grad[idx] += new_tensor.grad
                else:
                    input.grad[idx] += 0.0
        else:
            for idx in itertools.product(
                *[range(dim) for dim in new_tensor.shape]
            ):
                indices = list(idx)
                if keepdim:
                    indices[dim] = slice(None)
                else:
                    indices.insert(dim, slice(None))  # type: ignore
                indices = tuple(indices)
                slices = input[indices]
                for jdx, maximum in enumerate(slices._flat):
                    if maximum == slices.max().item():
                        dims = list(idx)
                        if keepdim:
                            dims[dim] = jdx
                        else:
                            dims.insert(dim, jdx)
                        input.grad[tuple(dims)] += new_tensor.grad[idx]

    new_tensor.grad_fn = Node(MaxBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def min(
    input: Tensor,
    dim: None | int = None,
    keepdim: bool = False,
) -> Tensor:
    """Return the minimum of elements in the tensor across a
    specified dimension.

    This function returns the minimum of all elements in the tensor if
    no dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be computed for minimum values.
    :param dim: The dimension along which to compute the minimum,
        defaults to `None`. For `None`, the minimum is computed over all
        elements of the tensor.
    :param keepdim: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the minimum of the specified
        elements.
    :raises ValueError: If the specified dimension is invalid.
    """
    ndim = input.ndim
    if dim is not None:
        if not (-ndim <= dim < ndim):
            raise IndexError(
                "Dimension out of range (expected to be in the range "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim
    if dim is None:
        shape = 1 if not keepdim else (1,) * ndim
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = builtins.min(input._flat)
    else:
        shape = tuple(
            (1 if idx == dim and keepdim else input.shape[idx])
            for idx in range(ndim)
            if keepdim or idx != dim
        )
        new_tensor = Tensor(
            shape if shape else (1,),
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        for idx in range(new_tensor.nelement()):
            unravelled = unravel_index(idx, shape)
            indices = list(unravelled)
            if keepdim:
                indices = tuple(
                    indices[idx] if idx != dim else slice(None)
                    for idx in range(len(indices))
                )
            else:
                indices = indices[:dim] + [slice(None)] + indices[dim:]
            new_tensor[unravelled] = input[tuple(indices)].min().item()

    def MinBackward0() -> None:
        """Backward pass for the minimum operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        if dim is None:
            for idx in itertools.product(*[range(dim) for dim in input.shape]):
                if input[idx] == new_tensor.item():
                    input.grad[idx] += new_tensor.grad
                else:
                    input.grad[idx] += 0.0
        else:
            for idx in itertools.product(
                *[range(dim) for dim in new_tensor.shape]
            ):
                indices = list(idx)
                if keepdim:
                    indices[dim] = slice(None)
                else:
                    indices.insert(dim, slice(None))  # type: ignore
                indices = tuple(indices)
                slices = input[indices]
                for jdx, minimum in enumerate(slices._flat):
                    if minimum == slices.min().item():
                        dims = list(idx)
                        if keepdim:
                            dims[dim] = jdx
                        else:
                            dims.insert(dim, jdx)
                        input.grad[tuple(dims)] += new_tensor.grad[idx]

    new_tensor.grad_fn = Node(MinBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def mean(
    input: Tensor,
    dim: None | int = None,
    keepdim: bool = False,
) -> Tensor:
    """Return the mean of elements in the tensor across a specified
    dimension.

    This function returns the mean of all elements in the tensor if
    no dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be computed for mean values.
    :param dim: The dimension along which to compute the mean, defaults
        to `None`. For `None`, the mean is computed over all elements of
        the tensor.
    :param keepdim: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the mean of the specified elements.
    :raises ValueError: If the specified dimension is invalid.
    """
    ndim = input.ndim
    if dim is not None:
        if not (-ndim <= dim < ndim):
            raise IndexError(
                "Dimension out of range (expected to be in the range "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim
    if dim is None:
        shape = 1 if not keepdim else (1,) * ndim
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = statistics.mean(input._flat)
    else:
        shape = tuple(
            (1 if idx == dim and keepdim else input.shape[idx])
            for idx in range(ndim)
            if keepdim or idx != dim
        )
        new_tensor = Tensor(
            shape if shape else (1,),
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        for idx in range(new_tensor.nelement()):
            unravelled = unravel_index(idx, shape)
            indices = list(unravelled)
            if keepdim:
                indices = tuple(
                    indices[idx] if idx != dim else slice(None)
                    for idx in range(len(indices))
                )
            else:
                indices = indices[:dim] + [slice(None)] + indices[dim:]
            new_tensor[unravelled] = input[tuple(indices)].mean().item()

    def MeanBackward0() -> None:
        """Backward pass for the mean operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        if dim is None:
            for idx in itertools.product(*[range(dim) for dim in input.shape]):
                input.grad[idx] += new_tensor.grad / input.nelement()
        else:
            for idx in itertools.product(
                *[range(dim) for dim in new_tensor.shape]
            ):
                indices = list(idx)
                if keepdim:
                    indices[dim] = slice(None)
                else:
                    indices.insert(dim, slice(None))  # type: ignore
                indices = tuple(indices)
                slices = input[indices]
                grad = new_tensor.grad[idx] / slices.nelement()
                for jdx, _ in enumerate(slices._flat):
                    dims = list(idx)
                    if keepdim:
                        dims[dim] = jdx
                    else:
                        dims.insert(dim, jdx)
                    input.grad[tuple(dims)] += grad

    new_tensor.grad_fn = Node(MeanBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def std(
    input: Tensor,
    dim: None | int = None,
    correction: int = 1,
    keepdim: bool = False,
) -> Tensor:
    """Return the standard deviation of elements in the tensor across a
    specified dimension.

    This function returns the standard deviation of all elements in the
    tensor if no dimension is provided. If a dimension is specified, the
    function reduces the tensor along the given dimension while
    optionally retaining the reduced dimensions.

    :param input: Input tensor to be computed for standard deviation
        values.
    :param dim: The dimension along which to compute the standard
        deviation, defaults to `None`. For `None`, the standard
        deviation is computed over all elements of the tensor.
    :param correction: Difference between the sample size and sample
        degrees of freedom, defaults to `1`.
    :param keepdim: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the standard deviation of the
        specified elements.
    :raises ValueError: If the specified dimension is invalid.
    """
    ndim = input.ndim
    if dim is not None:
        if not (-ndim <= dim < ndim):
            raise IndexError(
                "Dimension out of range (expected to be in the range "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim
    if dim is None:
        shape = 1 if not keepdim else (1,) * ndim
        new_tensor = Tensor(
            shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        new_tensor[:] = statistics.stdev(input._flat)
    else:
        shape = tuple(
            (1 if idx == dim and keepdim else input.shape[idx])
            for idx in range(ndim)
            if keepdim or idx != dim
        )
        new_tensor = Tensor(
            shape if shape else (1,),
            dtype=input.dtype,
            requires_grad=input.requires_grad,
        )
        for idx in range(new_tensor.nelement()):
            unravelled = unravel_index(idx, shape)
            indices = list(unravelled)
            if keepdim:
                indices = tuple(
                    indices[idx] if idx != dim else slice(None)
                    for idx in range(len(indices))
                )
            else:
                indices = indices[:dim] + [slice(None)] + indices[dim:]
            new_tensor[unravelled] = input[tuple(indices)].std().item()

    def StdBackward0() -> None:
        """Backward pass for the standard deviation operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        if dim is None:
            mu = input.mean()
            N = input.nelement()
            for idx in itertools.product(*[range(dim) for dim in input.shape]):
                input.grad[idx] += (
                    new_tensor.grad / ((N - correction) * new_tensor)
                ) * (input[idx] - mu)
        else:
            for idx in itertools.product(
                *[range(dim) for dim in new_tensor.shape]
            ):
                indices = list(idx)
                if keepdim:
                    indices[dim] = slice(None)
                else:
                    indices.insert(dim, slice(None))  # type: ignore
                indices = tuple(indices)
                slices = input[indices]
                mu = slices.mean().item()
                x = new_tensor[idx]
                N = slices.nelement()
                scale = new_tensor.grad[idx] / ((N - correction) * x)
                for jdx, value in enumerate(slices._flat):
                    dims = list(idx)
                    if keepdim:
                        dims[dim] = jdx
                    else:
                        dims.insert(dim, jdx)
                    input.grad[tuple(dims)] += scale * (value - mu)

    new_tensor.grad_fn = Node(StdBackward0)
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
        input.shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    new_tensor[:] = (math.exp(dim) for dim in input._flat)

    def ExpBackward0() -> None:
        """Backpropagation implementation for exponentiation.

        Computes gradient for `input` tensor and propagate it. The exp
        gradient is defined as::

            exp'(x) = math.exp(x)
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        for idx in itertools.product(*[range(dim) for dim in input.shape]):
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
    :raises ValueError: If the input tensor contains negative values.
    """
    new_tensor = Tensor(
        input.shape,
        slowtorch.float32,
        requires_grad=input.requires_grad,
    )
    data: list[Number] = []
    for idx in input._flat:
        result = idx**0.5
        data.append(slowtorch.nan if isinstance(result, complex) else result)
    new_tensor[:] = data

    def SqrtBackward0() -> None:
        """Backpropagation implementation for square-roots.

        Computes gradient for `input` and propagate it.
        """
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
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
        input.shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    if len(input.shape) == 1:
        iterator = range(input.shape[0])
    else:
        iterator = itertools.product(*[range(dim) for dim in input.shape])
    for dim in iterator:
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
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        input.grad += (new_tensor > 0) * new_tensor.grad

    new_tensor.grad_fn = Node(ReluBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
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
        iterator = itertools.product(*[range(dim) for dim in input.shape])
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
            input.grad = Tensor(input.shape, input.dtype)
        input.grad += (
            1.0 if new_tensor > 0 else alpha * normal_exp(new_tensor)
        ) * new_tensor.grad

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
        input.shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    if len(input.shape) == 1:
        iterator = range(input.shape[0])
    else:
        iterator = itertools.product(*[range(dim) for dim in input.shape])
    for dim in iterator:
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
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        input.grad -= (1.0 - new_tensor**2) * new_tensor.grad

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
        input.shape,
        input.dtype,
        requires_grad=input.requires_grad,
    )
    data: list[t.Any] = []
    if len(input.shape) == 1:
        iterator = range(input.shape[0])
    else:
        iterator = itertools.product(*[range(dim) for dim in input.shape])
    for dim in iterator:
        try:
            data.append(1.0 / (1.0 + safe_exp(input[dim])))
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
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
        input.grad -= (new_tensor * (1 - new_tensor)) * new_tensor.grad

    new_tensor.grad_fn = Node(SigmoidBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def softmax(
    input: Tensor,
    dim: None | int = None,
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
        input.shape,
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
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input.shape, input.dtype)
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
    dim: None | int = None,
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


@function_dispatch
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
        if None in (input.grad, weight.grad):
            input.grad = weight.grad = Tensor(1, input.dtype)
        input.grad += new_tensor.grad @ weight
        weight.grad += new_tensor.grad.T @ input
        if bias is not None:
            bias.grad = Tensor(bias.shape, bias.dtype)
            bias.grad += new_tensor.grad.sum(dim=0)

    new_tensor.grad_fn = Node(AddmmBackward0)
    new_tensor.grad_fn.inputs = (input, weight, bias)
    return new_tensor


@function_dispatch
def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Compute the Mean Squared Error (MSE) loss between two tensors.

    This function calculates the average of the squared differences
    between `input` and `target` tensors. It supports autograd for
    backpropagation.

    :param input: The input tensor representing predictions.
    :param target: The target tensor representing true values.
    :param reduction: The reduction function to apply to the computed
        loss. Options are::
            - `mean`: Return the average of the squared differences.
            - `sum`: Return the sum of the squared differences.
            - `none`: Return the squared differences without reduction.
        Defaults to `mean`.
    :return: A scalar tensor representing the MSE loss.
    """
    loss = (input - target) ** 2
    if reduction == "mean":
        new_tensor = loss.sum() / loss.nelement()
    elif reduction == "sum":
        new_tensor = loss.sum()
    elif reduction == "none":
        new_tensor = loss

    def MseLossBackward0() -> None:
        """Backpropagation implementation for MSE loss.

        Computes gradients for the `input` and `target` tensors and
        propagates them backward through the computational graph.
        """
        if None in (input.grad, target.grad):
            input.grad = target.grad = Tensor(input.shape, input.dtype)
        grad = 2.0 / loss.nelement() if reduction == "mean" else 2.0
        input.grad += grad * (input - target)
        target.grad -= grad * (input - target)

    new_tensor.grad_fn = Node(MseLossBackward0)
    new_tensor.grad_fn.inputs = (input, target)
    return new_tensor


@function_dispatch
def l1_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Compute the Mean Absolute Error (MAE) loss between two tensors.

    This function calculates the average of the absolute differences
    between `input` and `target` tensors. It supports autograd for
    backpropagation.

    :param input: The input tensor representing predictions.
    :param target: The target tensor representing true values.
    :param reduction: The reduction function to apply to the computed
        loss. Options are::
            - `mean`: Return the average of the absolute differences.
            - `sum`: Return the sum of the absolute differences.
            - `none`: Return the absolute differences without reduction.
        Defaults to `mean`.
    :return: A scalar tensor representing the MAE loss.
    """
    loss = abs(input - target)
    if reduction == "mean":
        new_tensor = loss.sum() / loss.nelement()
        grad_fn = "MeanBackward0"
    elif reduction == "sum":
        new_tensor = loss.sum()
        grad_fn = "SumBackward0"
    elif reduction == "none":
        new_tensor = loss
        grad_fn = "AbsBackward0"

    def L1LossBackward0() -> None:
        """Backpropagation implementation for MAE loss.

        Computes gradients for the `input` and `target` tensors and
        propagates them backward through the computational graph.
        """
        if None in (input.grad, target.grad):
            input.grad = target.grad = Tensor(input.shape, input.dtype)
        grad = 2.0 / loss.nelement() if reduction == "mean" else 2.0
        input.grad += grad * loss
        target.grad -= grad * loss

    L1LossBackward0.__name__ = grad_fn
    new_tensor.grad_fn = Node(L1LossBackward0)
    new_tensor.grad_fn.inputs = (input, target)
    return new_tensor


@function_dispatch
def nll_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Compute the Negative Log Likelihood (NLL) loss.

    This loss is useful to train a classification problem with `C`
    classes. The input is expected to contain log-probabilities
    (typically output of `log_softmax`), and the target contains the
    class indices. It supports autograd for backpropagation.

    :param input: The input tensor of shape (N, C) where C = number of
        classes.
    :param target: The target of shape (N,) with class indices in
        [0, C-1].
    :param reduction: The reduction function to apply to the computed
        loss. Options are::
            - `mean`: Return the average of the absolute differences.
            - `sum`: Return the sum of the absolute differences.
            - `none`: Return the absolute differences without reduction.
        Defaults to `mean`.
    :return: A scalar tensor representing the NLL loss.
    """
    loss = Tensor(
        shape=input.shape[0],
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
    )
    loss[:] = (-input[dim, target[dim]] for dim in range(input.shape[0]))
    if reduction == "mean":
        new_tensor = loss.sum() / loss.nelement()
    elif reduction == "sum":
        new_tensor = loss.sum()
    elif reduction == "none":
        new_tensor = loss

    def NllLossBackward0() -> None:
        """Backpropagation implementation for NLL loss.

        Computes gradients for the `input` and `target` tensors and
        propagates them backward through the computational graph.
        """
        grad = Tensor(input.shape, input.dtype)
        for dim in range(input.shape[0]):
            grad[dim, target[dim]] = -1.0
        if reduction == "mean":
            for dim in range(input.shape[0]):
                grad[dim, target[dim]] /= input.shape[0]
        if input.grad is None:
            input.grad = grad
        else:
            input.grad += grad

    new_tensor.grad_fn = Node(NllLossBackward0)
    new_tensor.grad_fn.inputs = (input, target)
    return new_tensor
