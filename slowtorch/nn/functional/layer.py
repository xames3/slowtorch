"""\
SlowTorch Stateless Layer
=========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, 01 June 2025
Last updated on: Sunday, 24 August 2025

Layer operations.

This module offers stateless functions that perform layer-specific
operations. These functions mimic the API and semantics of PyTorch's
neural network layers functions.

.. note::

    All functions support auto-differentiation (backpropagation).
"""

from __future__ import annotations

import typing as t
from itertools import product as pdt

from slowtorch import function_dispatch
from slowtorch.internal.tensor import Node
from slowtorch.internal.tensor import Tensor


@function_dispatch
def linear(
    input: Tensor,
    weight: Tensor,
    bias: None | Tensor = None,
) -> t.Any:
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
        if bias._shape != (new_tensor._shape[-1],):
            raise ValueError(
                f"Bias {bias._shape} is incompatible with output shape "
                f"{new_tensor._shape}"
            )
        new_tensor += bias

    def AddmmBackward0() -> None:
        """Backpropagation implementation for Linear transformation.

        Computes gradients for the `input`, `weight`, and `bias` tensors
        and propagates them backward through the computational graph.
        """
        grad = new_tensor.grad
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        if weight.grad is None:
            weight.grad = Tensor(weight._shape, weight.dtype)
        input.grad += grad @ weight
        weight.grad += grad.T @ input
        if bias is not None:
            if bias.grad is None:
                bias.grad = Tensor(bias._shape, bias.dtype)
            bias.grad += grad.sum(dim=0)

    new_tensor.grad_fn = Node(AddmmBackward0)
    new_tensor.grad_fn.inputs = (input, weight, bias)
    return new_tensor


@function_dispatch
def embedding(input: Tensor, weight: Tensor) -> Tensor:
    """Generate an embedding lookup from a weight matrix using input
    indices.

    The embedding function retrieves rows from the `weight` matrix
    based on integer indices from `input`. Each entry in `input`
    corresponds to a vector in `weight`. If `padding_idx` is set,
    gradients for that index will be masked (i.e., not updated during
    backpropagation).

    :param input: Tensor of indices.
    :param weight: Embedding matrix.
    :return: Tensor plucked out of the embedding matrix represented by
        the tensor of indices.
    """
    new_tensor = weight[input]

    def EmbeddingBackward0() -> None:
        """Backpropagation implementation for embedding lookup.

        Accumulates gradients into the `weight` tensor for all positions
        gathered from during the forward pass. Skips `padding_idx`.
        """
        if not hasattr(weight, "grad") or weight.grad is None:
            weight.grad = Tensor(weight._shape, weight.dtype)
        for index, storage in enumerate(input.storage):
            indices = list(pdt(*[range(dim) for dim in input._shape]))[index]
            for idx in range(weight._shape[1]):
                weight.grad[storage, idx] += new_tensor.grad[indices + (idx,)]

    new_tensor.grad_fn = Node(EmbeddingBackward0)
    new_tensor.grad_fn.inputs = (weight,)
    return new_tensor
