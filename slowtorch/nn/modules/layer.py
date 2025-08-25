"""\
SlowTorch Layers
================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, 01 June 2025
Last updated on: Sunday, 24 August 2025

Layer objects.

This module provides various layers to be considered in a neural network
model. These classes mimic the API and semantics of PyTorch's layers.
"""

from __future__ import annotations

import math
import typing as t

from slowtorch.internal.tensor import Tensor
from slowtorch.nn import functional as F
from slowtorch.nn.modules import modules_dispatch
from slowtorch.nn.modules.module import Module
from slowtorch.nn.modules.module import Parameter
from slowtorch.ops.random import randn
from slowtorch.ops.random import uniform_
from slowtorch.utils import set_module

if t.TYPE_CHECKING:
    from slowtorch.types import BoolLikeType
    from slowtorch.types import DeviceType
    from slowtorch.types import Dtype
    from slowtorch.types import IntLikeType


@set_module("slowtorch.nn.modules.flatten")
@modules_dispatch
class Flatten(Module):
    """Flatten a contiguous range of dims into a tensor."""

    def __init__(self) -> None:
        """Initialise `Flatten` instance with `Module` base class."""
        super().__init__()

    def __repr__(self) -> str:
        """Return a string representation of the `Flatten` module."""
        return f"{type(self).__name__}()"

    def forward(self, input: Tensor) -> Tensor:
        """Perform the forward pass of the flatten layer."""
        return input.flatten()


@set_module("slowtorch.nn.modules.linear")
@modules_dispatch
class Identity(Module):
    """A module that performs the identity operation.

    It returns the input tensor unchanged. This is particularly useful
    as a placeholder in model architectures or for debugging.
    """

    def __init__(self) -> None:
        """Initialise `Identity` instance."""
        super().__init__()

    def __repr__(self) -> str:
        """Return a string representation of the `Identity` module."""
        return f"{type(self).__name__}()"

    def forward(self, input: Tensor) -> Tensor:
        """Perform the identity operation.

        This method simply returns the input tensor without modification.
        It is useful when an operation is required in a pipeline but no
        transformation of the input is needed.

        :param input: The input tensor.
        :return: The same tensor as the input.
        """
        return input


@set_module("slowtorch.nn.modules.linear")
@modules_dispatch
class Linear(Module):
    """Represent a fully connected (linear) layer, a key component in
    neural networks, which performs a linear transformation on the
    input data. The mathematical operation is defined as::

        y = x @ w.T + b

    Here::

        - `x` is the input tensor of shape.
        - `w` is the weight matrix of shape.
        - `b` is the bias vector of shape.

    This module is often used as the primary building block in neural
    networks for tasks such as regression, classification, and feature
    transformation. The weights and biases are trainable parameters, and
    gradients for these parameters are tracked automatically during
    backpropagation.

    :param in_features: The number of input features for the layer. This
        corresponds to the dimensionality of the input tensor.
    :param out_features: The number of output features for the layer.
        This determines the dimensionality of the transformed output
        tensor.
    :param bias: Whether to include a bias term in the transformation,
        defaults to `True`.
    :param device: Specifies the device (e.g., CPU or GPU) on which the
        parameters will be allocated, defaults to `None`.
    :param dtype: Specifies the data type for the parameters, defaults
        to `None`.
    """

    in_features: IntLikeType
    out_features: IntLikeType
    weight: Tensor

    def __init__(
        self,
        in_features: IntLikeType,
        out_features: IntLikeType,
        bias: BoolLikeType = True,
        device: None | DeviceType = None,
        dtype: None | Dtype = None,
    ) -> None:
        """Initialise the `Linear` module with the specified input and
        output feature sizes, and optionally include a bias term.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            randn(out_features, in_features, dtype=dtype, device=device)
        )
        if bias:
            self.bias = Parameter(
                randn(out_features, dtype=dtype, device=device)
            )
        else:
            self.bias = None
        self.reset_parameters()

    def __repr__(self) -> str:
        """Return a string representation of the `Linear` module."""
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )

    def reset_parameters(self) -> None:
        """Reset the parameters of the layer using uniform
        initialisation.

        Weights and Bias, if present are initialised from::

            U(-k, k), where k = 1 / sqrt(in_features)
        """
        k = 1.0 / math.sqrt(self.in_features)
        uniform_(self.weight, -k, k)
        if self.bias is not None:
            uniform_(self.bias, -k, k)

    def forward(self, input: Tensor) -> t.Any:
        """Perform the forward pass of the linear layer.

        The forward pass computes the linear transformation on the
        input tensor `input` as follows::

            output = input @ weight.T + bias

        :param input: The input tensor to be transformed, with shape.
        :return: The output tensor resulting from the linear
            transformation, with shape.
        :raises ValueError: If the input tensor does not have the
            expected shape.
        """
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected input with {self.in_features} features, but "
                f"got {input.shape[-1]}"
            )
        return F.linear(input, self.weight, self.bias)


@set_module("slowtorch.nn.modules.sparse")
@modules_dispatch
class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary
    and size.

    This module is often used to store word embeddings and retrieve them
    using indices. The input to the module is a list of indices, and the
    output is the corresponding word embeddings.

    :param num_embeddings: Size of the dictionary of embeddings.
    :param embedding_dim: The size of each embedding vector.
    :param _weight: The learnable weights (embedding matrix) created
        from a normal distribution, defaults to `None`.
    :param _freeze: Boolean flag for freeze gradients during the backward
        pass, defaults to `False`.
    :param device: Specifies the device (e.g., CPU or GPU) on which the
        parameters will be allocated, defaults to `None`.
    :param dtype: Specifies the data type for the parameters, defaults
        to `None`.
    """

    num_embeddings: IntLikeType
    embedding_dim: IntLikeType
    weight: Tensor

    def __init__(
        self,
        num_embeddings: IntLikeType,
        embedding_dim: IntLikeType,
        _weight: None | Tensor = None,
        _freeze: BoolLikeType = False,
        device: None | DeviceType = None,
        dtype: None | Dtype = None,
    ) -> None:
        """Initialise the `Embedding` module with the specified number of
        embeddings and embedding dimensions, and optionally include
        weights (embedding tensor).
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if _weight is None:
            self.weight = Parameter(
                randn(
                    embedding_dim, num_embeddings, dtype=dtype, device=device
                ),
                requires_grad=not _freeze,
            )
        else:
            self.weight = Parameter(_weight, requires_grad=not _freeze)
        self.weight = self.weight.T

    def __repr__(self) -> str:
        """Return a string representation of the `Embedding` module."""
        return (
            f"{type(self).__name__}({self.num_embeddings}, "
            f"{self.embedding_dim})"
        )

    def forward(self, input: Tensor) -> Tensor:
        """Perform the forward pass of the embedding layer.

        The forward pass pops off and returns the corresponding word
        embeddings at the respective input indices.

        :param input: The input tensor representing indices.
        :return: Tensor plucked out of the `weight` (embedding matrix)
            represented by the input.
        """
        return F.embedding(input, self.weight)
