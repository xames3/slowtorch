"""\
SlowTorch Neural Network related Functions API
==============================================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Wednesday, January 15 2025
Last updated on: Tuesday, January 21 2025

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

import itertools
import math
import typing as t

from slowtorch._tensor import Node
from slowtorch._tensor import Tensor
from slowtorch._utils import normal_exp
from slowtorch._utils import safe_exp
from slowtorch._utils import safe_max


def exp(input: Tensor) -> Tensor:
    """Perform element-wise exponentiation of the tensor.

    This method supports exponentiation. The resulting tensor is of the
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
            input.grad = Tensor((1,), input.dtype)
        input.grad += new_tensor.exp() * new_tensor.grad

    new_tensor.grad_fn = Node(ExpBackward0)
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
        N = range(max(input.shape))
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
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if None in (input.grad,):
            input.grad = Tensor((1,), input.dtype)
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
        N = range(max(input.shape))
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
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if None in (input.grad,):
            input.grad = Tensor((1,), input.dtype)
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
            ((x := normal_exp(input[dim])) - (y := safe_exp(-input[dim])))
            / (x + y)
            for dim in range(input.shape[0])
        ]
    else:
        N = range(max(input.shape))
        for dim in itertools.product(N, N):
            try:
                new_tensor[dim] = (
                    (x := normal_exp(input[dim]))
                    - (y := safe_exp(-input[dim]))
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
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if None in (input.grad,):
            input.grad = Tensor((1,), input.dtype)
        input.grad += (1.0 - new_tensor**2) * new_tensor.grad

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
        N = range(max(input.shape))
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
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if None in (input.grad,):
            input.grad = Tensor((1,), input.dtype)
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
        if input.nelement() > 1:
            raise RuntimeError("grad can be created only for scalars")
        if None in (input.grad, weight.grad, bias.grad):
            input.grad = weight.grad = bias.grad = Tensor((1,), input.dtype)
        input.grad += new_tensor.grad @ weight
        weight.grad += new_tensor.grad.T @ input
        if bias is not None:
            bias.grad += new_tensor.grad

    new_tensor.grad_fn = Node(AddmmBackward0)
    new_tensor.grad_fn.inputs = (input, weight, bias)
    return new_tensor
