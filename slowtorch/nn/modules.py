"""\
SlowTorch Modules API
=====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Thursday, January 16 2025
Last updated on: Monday, March 03 2025

This module provides a foundational framework for building and training
neural networks, inspired by PyTorch's flexible and dynamic design. It
enables users to create custom models by leveraging a modular and
extensible architecture where components like layers and parameters are
automatically registered and optimised during training.

The `Module` class serves as the base class for all neural network
components. It automatically registers submodules and learnable
parameters when assigned as attributes. It also provides mechanisms for
forward computation, parameter iteration, and mode switching (training
vs. evaluation). Whereas, the `Parameter` class is a specialised
subclass of `Tensor` designed to represent learnable parameters, such as
weights and biases. It automatically tracks gradients and integrates
seamlessly with the `Module` hierarchy.

This module provides key important features like::

    1. Automatic Registration: Submodules (`Module`) and learnable
       parameters (`Parameter`) are automatically registered when
       assigned to attributes of a `Module`. This eliminates the need
       for explicit registration methods and ensures consistency across
       the framework.
    2. Dynamic Forward Computation: The `forward` method in the `Module`
       class is designed to be overridden by subclasses to define the
       specific computation for the module. Calling the module object
       directly invokes this method, enabling an intuitive interface.
    3. Gradient Handling: The API provides utilities for resetting
       gradients (`zero_grad`) and managing backpropagation seamlessly.
    4. Training and Evaluation Modes: Modules and their submodules can
       be easily switched between training and evaluation modes using
       the `train` and `eval` methods.
    5. Iterative Access: The API supports hierarchical iteration over
       submodules and parameters, making it easy to traverse complex
       model architectures.
"""

from __future__ import annotations

import math
import typing as t
from collections import OrderedDict

import slowtorch
from slowtorch import empty
from slowtorch._tensor import Tensor
from slowtorch._utils import DeviceType
from slowtorch._utils import Dtype
from slowtorch._utils import set_module
from slowtorch._variable_functions import uniform_

__all__: list[str] = [
    "ELU",
    "Flatten",
    "Identity",
    "Linear",
    "MSELoss",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Tanh",
    "_Loss",
]


@set_module("slowtorch.nn.parameter")
class Parameter(Tensor):
    """A specialised subclass of `Tensor` designed to represent
    parameters in modules. A `Parameter` is typically used to define
    learnable weights within a neural network or other computational
    models.

    Unlike regular `Tensor` objects, `Parameter` is automatically
    registered as part of a model when assigned as an attribute of a
    module. This registration facilitates optimisation and gradient
    tracking during training.

    :param data: The underlying tensor data. If not provided, an
        uninitialised tensor of shape `(1,)` is created, defaults to
        `None`.
    :param requires_grad: A flag indicating whether gradients should
        be computed for this parameter during backpropagation, defaults
        to `True`.
    """

    def __init__(
        self,
        data: None | Tensor = None,
        requires_grad: bool = True,
    ) -> None:
        """Initialise a `Parameter` instance with optional data."""
        if data is None:
            data = empty(1, requires_grad=requires_grad)
        data.requires_grad = requires_grad
        for key, value in data.__dict__.items():
            setattr(self, key, value)

    @property
    def data(self) -> Tensor:
        """Return the underlying tensor data."""
        return self

    @data.setter
    def data(self, value: Tensor) -> None:
        """Ensure that setting `data` updates the parameter in-place."""
        if not isinstance(value, Tensor):
            raise TypeError("Parameter data must be a tensor")
        self._cdata[:] = value._cdata

    def __repr__(self) -> str:
        """Return a string representation of `Parameter` object."""
        return f"Parameter containing:\n{super().__repr__()}"


@set_module("slowtorch.nn.modules.module")
class Module:
    """Base class for all neural network modules.

    All models should subclass this class. This class provides
    fundamental building blocks such as parameter registration, forward
    computation, and gradient handling. Submodules and parameters are
    automatically registered when assigned as attributes.
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialise `Module` instance with optional arguments."""
        self.args = args
        self.kwargs = kwargs
        self.training: bool = True
        self._modules: dict[str, None | Module] = OrderedDict()
        self._parameters: dict[str, None | Parameter] = OrderedDict()

    def __repr__(self) -> str:
        """Return a string representation of the `Module` object."""
        modules = "\n"
        for key, module in self._modules.items():
            modules += f"  ({key}): {module}\n"
        return f"{type(self).__name__}({modules})"

    def __setattr__(self, name: str, value: Module | Parameter) -> None:
        """Override to automatically register submodules and parameters.

        :param name: Attribute name.
        :param value: Attribute value to be set.
        """
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def __call__(self, *input: t.Any, **kwargs: t.Any) -> Tensor:
        """Invoke the `forward` method."""
        return self.forward(*input, **kwargs)

    def forward(self, *input: t.Any, **kwargs: t.Any) -> Tensor:
        """Define the computation performed at every call.

        Should be overridden by all subclasses.

        :raises NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError(
            f"Module: {type(self).__name__} is missing the"
            ' required "forward" function'
        )

    def children(self) -> t.Iterator[None | Module]:
        """Yield an iterator over immediate child modules."""
        yield from self._modules.values()

    def modules(self) -> t.Iterator[None | Module]:
        """Yield an iterator over all modules in the network."""
        yield self
        for module in self.children():
            yield from module.modules()

    def parameters(self, recurse: bool = True) -> t.Iterator[None | Parameter]:
        """Return an iterator over module parameters.

        :param recurse: If True, include parameters of submodules,
            defaults to `True`.
        :yield: The parameters of the module.
        """
        yield from self._parameters.values()
        if recurse:
            for module in self.children():
                yield from module.parameters()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset the gradients of all parameters in the module and its
        children.

        This ensures no leftover gradients from a previous backward
        pass.

        :param set_to_none: Bool to set grad to `None` instead of
            setting to 0.0.
        """
        for param in self.parameters():
            if param is not None and param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad = Tensor(1)

    def train(self, mode: bool = True) -> Module:
        """Set the module and its submodules to training mode.

        :param mode: Whether to set to training mode or evaluation mode,
            defaults to `False`.
        :return: Itself.
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> Module:
        """Set the module and its submodules to evaluation mode."""
        return self.train(False)


@set_module("slowtorch.nn.modules.container")
class Sequential(Module):
    """A container module that sequentially applies a list of
    sub-modules.

    Modules in `Sequential` are executed in the order they are added.
    This class is particularly useful for building neural network layers
    that can be executed one after another without requiring manual
    forwarding.

    Modules can be added either as positional arguments or as a single
    ordered dictionary.
    """

    def __init__(self, *args: Module) -> None:
        """Initialise the `Sequential` container with modules."""
        super().__init__()
        self._modules = OrderedDict()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            iterator = args[0].items()
        else:
            iterator = enumerate(args)
        for name, module in iterator:
            self.__setattr__(str(name), module)

    def __len__(self) -> int:
        """Return the number of sub-modules in `Sequential` container.

        :return: The count of sub-modules.
        """
        return len(self._modules)

    def __iter__(self) -> t.Iterator[None | Module]:
        """Return an iterator over the sub-modules in `Sequential`
        container.

        :return: An iterator over the sub-modules.
        """
        return iter(self._modules.values())

    def forward(self, input: Tensor) -> Tensor:
        """Compute the forward pass by applying each sub-module in
        sequence.

        :param input: The input tensor to the `Sequential` container.
        :return: The output tensor after applying all sub-modules.
        """
        for module in self:
            if module is not None:
                input = module(input)
        return input


@set_module("slowtorch.nn.modules.flatten")
class Flatten(Module):
    """Flatten a contiguous range of dims into a tensor."""

    def __init__(self) -> None:
        """Initialise `Flatten` instance with `Module` base class."""
        super().__init__()

    def __repr__(self) -> str:
        """Return a string representation of the `Flatten` object."""
        return f"{type(self).__name__}()"

    def forward(self, input: Tensor) -> Tensor:
        """Perform the forward pass of the flatten layer."""
        return input.flatten()


@set_module("slowtorch.nn.modules.linear")
class Identity(Module):
    """A module that performs the identity operation: it returns the
    input tensor unchanged. This is particularly useful as a placeholder
    in model architectures or for debugging.
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        """Initialise `Identity` instance."""
        super().__init__()

    def __repr__(self) -> str:
        """Return a string representation of the `Identity` object."""
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

    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: DeviceType = None,
        dtype: None | Dtype = None,
    ) -> None:
        """Initialise the `Linear` module with the specified input and
        output feature sizes, and optionally include a bias term.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            empty(
                out_features,
                in_features,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = Parameter(
                empty(
                    out_features,
                    dtype=dtype,
                    device=device,
                )
            )
        else:
            self.bias = None
        self.reset_parameters()

    def __repr__(self) -> str:
        """Return a string representation of the `Linear` object."""
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

    def forward(self, input: Tensor) -> Tensor:
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
        return slowtorch.nn.functional.linear(input, self.weight, self.bias)


@set_module("slowtorch.nn.modules.activation")
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
        """Return a string representation of the `ReLU` object."""
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
        return slowtorch.nn.functional.relu(input)


@set_module("slowtorch.nn.modules.activation")
class ELU(Module):
    """Represents a Exponential Linear Unit (ELU) activation layer.

    The ELU activation function applies an element-wise transformation
    to the input tensor, defined as::

        elu(x) = x if x >- 0 else alpha * (exp(x) - 1)

    ELU is a function that tend to converge cost to zero faster and
    produce more accurate results. This operation is differentiable,
    and gradients are propagated only for positive elements.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialise the `ELU` module with an alpha value."""
        super().__init__()
        self.alpha = alpha

    def __repr__(self) -> str:
        """Return a string representation of the `ELU` object."""
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
        return slowtorch.nn.functional.elu(input, self.alpha)


@set_module("slowtorch.nn.modules.activation")
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
        """Return a string representation of the `Tanh` object."""
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
        return slowtorch.nn.functional.tanh(input)


@set_module("slowtorch.nn.modules.activation")
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
        """Return a string representation of the `Sigmoid` object."""
        return f"{type(self).__name__}()"

    def forward(self, input: Tensor) -> Tensor:
        """Perform the forward pass of the Sigmoid activation layer.

        The forward pass applies the Sigmoid function to the input
        tensor, squashing out all the values between the range of
        0 to 1.

        :param input: The input tensor to be transformed, of arbitrary
            shape.
        :return: A new tensor where each element is the result of the
            Sigmoid operation applied to the corresponding element of the
            input tensor.
        """
        return slowtorch.nn.functional.sigmoid(input)


@set_module("slowtorch.nn.modules.loss")
class _Loss(Module):
    """Base class for all loss functions.

    This class provides a common interface for initialising loss
    functions with configurable reduction strategies. Subclasses can
    implement specific loss computation logic while inheriting the
    reduction handling from this base.

    :param size_average: If `True`, set reduction strategy to `mean`,
        defaults to `None`.
    :param reduce: If `True`, set reduction strategy to `sum`, defaults
        to `None`.
    :param reduction: Specify the reduction method to apply to the loss.
        Acceptable values are `mean`, `sum`, and `none`, defaults to
        `mean`.
    :raises ValueError: If conflicting parameters are provided or if the
        `reduction` value is invalid.
    """

    reduction: str

    def __init__(
        self,
        size_average: bool = None,
        reduce: bool = None,
        reduction: str = "mean",
    ) -> None:
        """Initialise `_Loss` instance with a reduction strategy."""
        super().__init__()
        if size_average is not None and reduce:
            pass
        self.reduction = reduction


@set_module("slowtorch.nn.modules.loss")
class MSELoss(_Loss):
    """Mean Squared Error (MSE) Loss module.

    This class calculates the Mean Squared Error loss between the
    predicted tensor (`input`) and the target tensor (`target`). It
    supports different reduction strategies such as `mean`, `sum`, or
    `none`.

    :param size_average: If `True`, set reduction strategy to `mean`,
        defaults to `None`.
    :param reduce: If `True`, set reduction strategy to `sum`, defaults
        to `None`.
    :param reduction: Specify the reduction method to apply to the loss.
        Acceptable values are `mean`, `sum`, and `none`, defaults to
        `mean`.
    :raises ValueError: If conflicting parameters are provided or if the
        `reduction` value is invalid.
    """

    reduction: str

    def __init__(
        self,
        size_average: bool = None,
        reduce: bool = None,
        reduction: str = "mean",
    ) -> None:
        """Initialise `MSELoss` instance with a reduction strategy."""
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the MSE loss between `input` and `target` tensors.

        This method calculates the squared differences between the input
        and target tensors, and applies the reduction strategy specified
        during initialisation.

        :param input: The predicted tensor.
        :param target: The target tensor.
        :return: A tensor representing the computed MSE loss, with
            reduction applied as per the configured strategy.
        :raises ValueError: If `input` and `target` tensors have
            mismatched shapes.
        """
        if input.shape != target.shape:
            raise ValueError(
                f"Shape of input tensor {input.shape} does not match "
                f"with target shape {target.shape}"
            )
        return slowtorch.nn.functional.mse_loss(input, target, self.reduction)
