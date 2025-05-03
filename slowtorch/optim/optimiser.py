"""\
SlowTorch Optimiser API
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Thursday, May 01 2025
Last updated on: Friday, May 02 2025

This module provides a foundational framework for implementing
gradient-based optimisers in SlowTorch. Inspired by PyTorch's
`torch.optim`, it allows users to define and experiment with custom
optimisation strategies while maintaining clarity and pedagogical
structure. The `Optimiser` class serves as the abstract base for all
optimisers. It manages parameter grouping, gradient resetting, and
shared configuration across parameter groups. It is designed to be
subclassed by concrete optimisation strategies.

This module provides key capabilities such as::

    1. Parameter Group Management: Parameters are organised into groups
       with shared hyperparameters. Groups may be individually
       configured, enabling fine-grained control over learning rates and
       other settings.
    2. Gradient Resetting: A unified method for resetting gradients
       ensures that stale gradients do not accumulate across iterations.
    3. Extendable Design: New optimisers can be built by subclassing
       the `Optimiser` class and implementing the `step` method,
       following familiar interfaces.
"""

from __future__ import annotations

import typing as t

from slowtorch._tensor import Tensor

__all__ = ["Optimiser", "Optimizer", "SGD"]

ParamsT: t.TypeAlias = (
    t.Iterable[Tensor]
    | t.Iterable[dict[str, t.Any]]
    | t.Iterable[tuple[str, Tensor]]
)
ParamGroup: t.TypeAlias = dict[str, t.Any]


class Optimiser:
    """Base class for all optimisers.

    This class defines the common interface and utilities for parameter-
    based optimisers. Subclasses should implement the `step` method to
    perform parameter updates.

    :param params: Iterable of parameters or parameter groups.
    :param defaults: Default hyperparameters applied to each group.
    :raises ValueError: If no parameters are provided.
    """

    def __init__(self, params: ParamsT, defaults: dict[str, t.Any]) -> None:
        """Initialise an `Optimiser` instance with model parameters."""
        self.defaults = defaults
        self.param_groups: list[ParamGroup] = []
        param_group = {"params": (_ := list(params))}
        if len(_) == 0:
            raise ValueError("optimizer got an empty parameter list")
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        self.param_groups.append(param_group)
        self.state: dict[Tensor, dict[str, t.Any]] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset gradients of all parameters in all parameter groups.

        This is essential before computing gradients via `.backward()`,
        ensuring that gradient accumulation is explicitly controlled.

        :param set_to_none: Bool to set grad to `None` instead of
            setting to 0.0.
        """
        for group in self.param_groups:
            for param in group["params"]:
                if param is not None and param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad = Tensor(1)

    def step(self, closure: None | t.Callable[[], float] = None) -> None:
        """Perform a single optimisation step.

        Subclasses must override this method to define parameter update
        logic.

        :param closure: A closure that reevaluates the model and returns
            the loss. This is useful for optimisers that need to
            reevaluate the model multiple times.
        :raises NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError


Optimizer = Optimiser


class SGD(Optimiser):
    """Class to implement stochastic gradient descent.

    This class performs Stochastic Gradient Descent (SGD) with optional
    momentum, dampening, Nesterov acceleration, and weight decay. This
    implementation updates parameters according to the formula::

        velocity = momentum * velocity + (1 - dampening) * grad
        update = grad + momentum * velocity if nesterov else velocity
        param = param - lr * update

    :param params: Parameters to optimise.
    :param lr: Learning rate (must be positive), defaults to 1e-3.
    :param momentum: Momentum factor, defaults to 0.0.
    :param dampening: Dampening for momentum, defaults to 0.0.
    :param weight_decay: L2 regularisation coefficient, defaults to 0.0.
    :param nesterov: Enables Nesterov momentum, defaults to `False`.
    :param maximise: If True, maximises the objective instead of
        minimising, defaults to `False`.
    :param foreach: Placeholder for future foreach-style updates,
        defaults to `None`.
    :param differentiable: Placeholder for autograd-aware variants,
        defaults to `None`.
    :raises ValueError: If any hyperparameter is invalid.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        *,
        maximise: bool = False,
        foreach: None | bool = None,
        differentiable: bool = None,
    ) -> None:
        """Initialise stochastic gradient descent optimiser."""
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be a scalar")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximise=maximise,
            foreach=foreach,
            differentiable=differentiable,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening"
            )
        super().__init__(params, defaults)

    def step(self, closure: None | t.Callable[[], float] = None) -> None:
        """Perform a single parameter update using SGD.

        Applies weight decay (if specified), handles momentum buffers,
        and computes the update direction using either classical or
        Nesterov momentum.

        :param closure: Currently ignored. Placeholder for future use.
        """
        closure = closure or None
        for group in self.param_groups:
            params = group["params"]
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            maximise = group["maximise"]
            for param in params:
                if param.grad is None:
                    continue
                grad = param.grad
                if weight_decay != 0.0:
                    grad = grad + weight_decay * param.data
                if momentum != 0.0:
                    buffer = self.state.setdefault(param, {}).get(
                        "momentum_buffer"
                    )
                    if buffer is None:
                        buffer = grad.clone()
                        self.state[param] = {"momentum_buffer": buffer}
                    else:
                        buffer = momentum * buffer + (1 - dampening) * grad
                        self.state[param]["momentum_buffer"] = buffer
                    if nesterov:
                        grad = grad + momentum * buffer
                    else:
                        grad = buffer
                if maximise:
                    param.data += lr * grad
                else:
                    param.data -= lr * grad
