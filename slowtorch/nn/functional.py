"""\
SlowTorch Neural Network related Functions API
==============================================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Wednesday, January 15 2025
Last updated on: Friday, January 17 2025

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
outputs and the target values, guiding the optimization process. The
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

from slowtorch._tensor import Tensor


def relu(input: Tensor) -> Tensor:
    """Apply the rectified linear unit (ReLU) function element-wise.

    ReLU sets all negative values in the input tensor to zero and keeps
    positive values unchanged. This operation is differentiable, and
    gradients are propagated only for positive elements. The relu
    function is defined as::

        relu(x) = max(x, 0)

    :param input: Input tensor for which ReLU is to be applied.
    :return: Output tensor after applying the ReLU function, with
        gradients linked for backpropagation.
    """
    return input.relu()


def elu(input: Tensor, alpha: float = 1.0) -> Tensor:
    """Apply the Exponential Linear Unit (ELU) function
    element-wise.

    ELU is a function that tend to converge cost to zero faster and
    produce more accurate results. This operation is differentiable, and
    gradients are propagated only for positive elements. The elu
    function is defined as::

        elu(x) = x if x >- 0 else alpha * (exp(x) - 1)

    :param input: Input tensor for which Tanh is to be applied.
    :param alpha: Value for the ELU formulation, defaults to 1.0.
    :return: Output tensor after applying the ELU function, with
        gradients linked for backpropagation.
    """
    return input.elu(alpha)


def tanh(input: Tensor) -> Tensor:
    """Apply the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh squashes all the values between the range of -1 to 1. This
    operation is differentiable, and gradients are propagated. The
    tanh function is defined as::

        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    :param input: Input tensor for which Tanh is to be applied.
    :return: Output tensor after applying the Tanh function, with
        gradients linked for backpropagation.
    """
    return input.tanh()


def sigmoid(input: Tensor) -> Tensor:
    """Apply the Sigmoid function element-wise.

    Sigmoid function squashes between 0 and 1. This operation is
    differentiable, and gradients are propagated. The sigmoid function
    is defined as::

        sigmoid(x) = 1 / (1 + exp(-x))

    :param input: Input tensor for which Sigmoid is to be applied.
    :return: Output tensor after applying the Sigmoid function, with
        gradients linked for backpropagation.
    """
    return input.sigmoid()
