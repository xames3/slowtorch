"""\
SlowTorch Activations
=====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, June 01 2025
Last updated on: Sunday, June 01 2025

Activation functions.

This module provides various activation function (non-linearity)
implementations in their class form. These classes mimic the API and
semantics of PyTorch's activation functions (class).
"""

from __future__ import annotations

import typing as t

from slowtorch.internal.tensor import Tensor
from slowtorch.nn import functional as F
from slowtorch.nn.modules import modules_dispatch
from slowtorch.nn.modules.module import Module
from slowtorch.utils import set_module

if t.TYPE_CHECKING:
    from slowtorch.types import Dim
    from slowtorch.types import FloatLikeType


@set_module("slowtorch.nn.modules.activation")
@modules_dispatch
class ReLU(Module):
    """Represents a Rectified Linear Unit (ReLU) activation layer.

    The ReLU activation function applies an element-wise transformation
    to the input tensor, defined as::

        relu(x) = max(x, 0)

    This operation zeroes out all negative elements in the input tensor
    while retaining positive elements. The ReLU activation function is
    widely used in neural networks due to its simplicity and effective
    gradient propagation for positive values.

    This module can be added to a computation graph, and its gradient
    is computed automatically during backpropagation.
    """

    def __init__(self) -> None:
        """Initialise the `ReLU` module."""
        super().__init__()

    def __repr__(self) -> str:
        """Return a string representation of the `ReLU` module."""
        return f"{type(self).__name__}()"

    def forward(self, input: Tensor) -> Tensor:
        """Perform the forward pass of the ReLU activation layer.

        The forward pass applies the ReLU function to the input tensor,
        zeroing out negative values and retaining positive values.

        :param input: The input tensor to be transformed, of arbitrary
            shape.
        :return: A new tensor where each element is the result of the
            ReLU operation applied to the corresponding element of the
            input tensor.
        """
        return F.relu(input)


@set_module("slowtorch.nn.modules.activation")
@modules_dispatch
class ELU(Module):
    """Represents a Exponential Linear Unit (ELU) activation layer.

    The ELU activation function applies an element-wise transformation
    to the input tensor, defined as::

        elu(x) = x if x >- 0 else alpha * (exp(x) - 1)

    ELU is a function that tend to converge cost to zero faster and
    produce more accurate results. This operation is differentiable,
    and gradients are propagated only for positive elements.
    """

    def __init__(self, alpha: FloatLikeType = 1.0) -> None:
        """Initialise the `ELU` module with an alpha value."""
        super().__init__()
        self.alpha = alpha

    def __repr__(self) -> str:
        """Return a string representation of the `ELU` module."""
        return f"{type(self).__name__}(alpha={self.alpha})"

    def forward(self, input: Tensor) -> Tensor:
        """Perform the forward pass of the ELU activation layer.

        The forward pass applies the ELU function to the input tensor,
        zeroing out negative values and retaining positive values.

        :param input: The input tensor to be transformed, of arbitrary
            shape.
        :return: A new tensor where each element is the result of the
            ELU operation applied to the corresponding element of the
            input tensor.
        """
        return F.elu(input, self.alpha)


@set_module("slowtorch.nn.modules.activation")
@modules_dispatch
class Tanh(Module):
    """Represents a Hyperbolic Tangent (Tanh) activation layer.

    The Tanh activation function applies an element-wise transformation
    to the input tensor, defined as::

        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Tanh squashes all the values between the range of -1 to 1. This
    operation is differentiable, and gradients are propagated.
    """

    def __init__(self) -> None:
        """Initialise the `Tanh` module."""
        super().__init__()

    def __repr__(self) -> str:
        """Return a string representation of the `Tanh` module."""
        return f"{type(self).__name__}()"

    def forward(self, input: Tensor) -> Tensor:
        """Perform the forward pass of the Tanh activation layer.

        The forward pass applies the Tanh function to the input tensor,
        squashing out all the values between the range of -1 to 1.

        :param input: The input tensor to be transformed, of arbitrary
            shape.
        :return: A new tensor where each element is the result of the
            Tanh operation applied to the corresponding element of the
            input tensor.
        """
        return F.tanh(input)


@set_module("slowtorch.nn.modules.activation")
@modules_dispatch
class Sigmoid(Module):
    """Represents a Sigmoid activation layer.

    The Sigmoid activation function applies an element-wise
    transformation to the input tensor, defined as::

        sigmoid(x) = 1 / (1 + exp(-x))

    Sigmoid squashes all the values between the range of 0 to 1. This
    operation is differentiable, and gradients are propagated.
    """

    def __init__(self) -> None:
        """Initialise the `Sigmoid` module."""
        super().__init__()

    def __repr__(self) -> str:
        """Return a string representation of the `Sigmoid` module."""
        return f"{type(self).__name__}()"

    def forward(self, input: Tensor) -> Tensor:
        """Perform the forward pass of the Sigmoid activation layer.

        The forward pass applies the Sigmoid function to the input
        tensor, squashing out all the values between the range of
        0 to 1.

        :param input: The input tensor to be transformed, of arbitrary
            shape.
        :return: A new tensor where each element is the result of the
            Sigmoid operation applied to the corresponding element of
            the input tensor.
        """
        return F.sigmoid(input)


@set_module("slowtorch.nn.modules.activation")
@modules_dispatch
class Softmax(Module):
    """Represents a Softmax activation layer.

    The Softmax activation function applies an element-wise
    transformation to the input tensor, defined as::

        softmax(x) = exp(x) / exp(x).sum()

    Sigmoid squashes all the values between the range of 0 to 1, along
    the provided dimension and sum to 1. This operation is
    differentiable, and gradients are propagated.

    :param dim: A dimension along which softmax will be computed,
        defaults to `None`.
    """

    def __init__(self, dim: None | Dim = None) -> None:
        """Initialise the `Softmax` module."""
        super().__init__()
        self.dim = dim

    def __repr__(self) -> str:
        """Return a string representation of the `Softmax` module."""
        return f"{type(self).__name__}(dim={self.dim})"

    def forward(self, input: Tensor) -> Tensor:
        """Perform the forward pass of the Softmax activation layer.

        The forward pass applies the Softmax function to the input
        tensor, squashing out all the values between the range of
        0 to 1, along the provided dimension and sum to 1.

        :param input: The input tensor to be transformed, of arbitrary
            shape.
        :return: A new tensor where each element is the result of the
            Softmax operation applied to the corresponding element of
            the input tensor.
        """
        return F.softmax(input, self.dim)


@set_module("slowtorch.nn.modules.activation")
@modules_dispatch
class LogSoftmax(Module):
    """Represents a LogSoftmax activation layer.

    This is mathematically equivalent to applying softmax function
    followed by logarith. This operation is differentiable, and
    gradients are propagated.

    :param dim: A dimension along which log softmax will be computed,
        defaults to `None`.
    """

    def __init__(self, dim: None | Dim = None) -> None:
        """Initialise the `LogSoftmax` module."""
        super().__init__()
        self.dim = dim

    def __repr__(self) -> str:
        """Return a string representation of the `LogSoftmax` module."""
        return f"{type(self).__name__}(dim={self.dim})"

    def forward(self, input: Tensor) -> Tensor:
        """Perform the forward pass of the LogSoftmax activation layer.

        The forward pass applies the Softmax function to the input
        tensor, squashing out all the values between the range of
        0 to 1, along the provided dimension and sum to 1. This
        operation is followed by a logarithm.

        :param input: The input tensor to be transformed, of arbitrary
            shape.
        :return: A new tensor where each element is the result of the
            LogSoftmax operation applied to the corresponding element of
            the input tensor.
        """
        return F.log_softmax(input, self.dim)
