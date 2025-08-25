"""\
SlowTorch Stateless Loss
========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, 01 June 2025
Last updated on: Sunday, 24 August 2025

Loss function operations.

This module offers a comprehensive suite of stateless functions that
calculate losses using various loss function implementations. These
functions mimic the API and semantics of PyTorch's loss functions.
"""

from __future__ import annotations

from slowtorch import function_dispatch
from slowtorch.internal.tensor import Node
from slowtorch.internal.tensor import Tensor
from slowtorch.nn.functional.pointwise import log_softmax


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
            input.grad = target.grad = Tensor(input._shape, input.dtype)
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
            input.grad = target.grad = Tensor(input._shape, input.dtype)
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
        shape=input._shape[0],
        dtype=input.dtype,
        device=input.device,
        requires_grad=input.requires_grad,
    )
    loss[:] = (
        -input[index, target[index]] for index in range(input._shape[0])
    )
    if reduction == "mean":
        new_tensor = loss.sum() / loss.nelement()
    elif reduction == "sum":
        new_tensor = loss.sum()
    elif reduction == "none":
        new_tensor = loss

    def NllLossBackward0() -> None:
        """Backpropagation implementation for NLL loss.

        Computes gradients for the `input` tensor and propagate them
        backward through the computational graph.
        """
        grad = Tensor(input._shape, input.dtype)
        for index in range(input._shape[0]):
            grad[index, target[index]] = -1.0
        if reduction == "mean":
            for index in range(input._shape[0]):
                grad[index, target[index]] /= input._shape[0]
        if input.grad is None:
            input.grad = grad
        else:
            input.grad += grad

    new_tensor.grad_fn = Node(NllLossBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def cross_entropy(
    input: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    """Compute the Cross Entropy loss between input and target tensor.

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
    :return: A scalar tensor representing the Cross Entropy loss.
    """
    return nll_loss(log_softmax(input, dim=1), target, reduction)
