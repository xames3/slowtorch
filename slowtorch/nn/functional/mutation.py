"""\
SlowTorch Stateless Mutation
============================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, June 01 2025
Last updated on: Sunday, June 01 2025

Mutating operations.

This module offers various stateless functions that perform manipulating
operations on the input tensor. These functions mimic the API and
semantics of PyTorch's mutating functions.

.. note::

    All functions support auto-differentiation (backpropagation).
"""

from __future__ import annotations

from slowtorch import function_dispatch
from slowtorch.internal.tensor import Node
from slowtorch.internal.tensor import Tensor


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
    new_tensor = Tensor(
        shape=input._shape,
        dtype=input._dtype,
        device=input.device,
        requires_grad=input.requires_grad,
    )
    new_tensor[...] = input[...]

    def CloneBackward0() -> None:
        """Backpropagation implementation for cloning.

        Computes gradient for `input` and propagate it.
        """
        if input.grad is None:
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
    new_tensor = Tensor(
        input.nelement(), input.dtype, input.device, input.requires_grad
    )
    new_tensor[:] = input

    def ViewBackward0() -> None:
        """Backpropagation implementation for cloning.

        Computes gradient for `input` and propagate it.
        """
        if input.grad is None:
            input.grad = new_tensor.grad

    new_tensor.grad_fn = Node(ViewBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


flatten = ravel


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
    indices = tuple(reversed(sorted((dim0, dim1))))
    shape = tuple(input._shape[index] for index in indices)
    strides = tuple(input._strides[index] for index in indices)
    new_tensor = input.as_strided(tuple(shape), tuple(strides))

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


swapaxes = swapdims = transpose
