"""\
SlowTorch Losses
================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, June 01 2025
Last updated on: Sunday, June 01 2025

Losses (in class form).

This module offers collection of all the loss modules implemented in
this library. These classes mimic the API and semantics of PyTorch's
loss functions (classes).
"""

from __future__ import annotations

import typing as t

import slowtorch
from slowtorch.internal.tensor import Tensor
from slowtorch.nn import functional as F
from slowtorch.nn.modules import modules_dispatch
from slowtorch.nn.modules.module import Module
from slowtorch.utils import set_module

if t.TYPE_CHECKING:
    from slowtorch.types import BoolLikeType


@set_module("slowtorch.nn.modules.loss")
@modules_dispatch
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
        size_average: BoolLikeType = None,
        reduce: BoolLikeType = None,
        reduction: str = "mean",
    ) -> None:
        """Initialise `_Loss` instance with a reduction strategy."""
        super().__init__()
        if size_average is not None and reduce:
            pass
        self.reduction = reduction


@set_module("slowtorch.nn.modules.loss")
@modules_dispatch
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
        size_average: BoolLikeType = None,
        reduce: BoolLikeType = None,
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
        return F.mse_loss(input, target, self.reduction)


@set_module("slowtorch.nn.modules.loss")
@modules_dispatch
class L1Loss(_Loss):
    """Mean Absolute Error (MAE) Loss module.

    This class calculates the Mean Absolute Error loss between the
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
        size_average: BoolLikeType = None,
        reduce: BoolLikeType = None,
        reduction: str = "mean",
    ) -> None:
        """Initialise `L1Loss` instance with a reduction strategy."""
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the MAE loss between `input` and `target` tensors.

        This method calculates the absolute differences between the
        input and target tensors, and applies the reduction strategy
        specified during initialisation.

        :param input: The predicted tensor.
        :param target: The target tensor.
        :return: A tensor representing the computed MAE loss, with
            reduction applied as per the configured strategy.
        :raises ValueError: If `input` and `target` tensors have
            mismatched shapes.
        """
        if input.shape != target.shape:
            raise ValueError(
                f"Shape of input tensor {input.shape} does not match "
                f"with target shape {target.shape}"
            )
        return F.l1_loss(input, target, self.reduction)


@set_module("slowtorch.nn.modules.loss")
@modules_dispatch
class NLLLoss(_Loss):
    """Negative Log Likelihood (NLL) Loss module.

    This class calculates the negative log likelihood loss between the
    logit (`input`) and the target tensor (`target`). It supports
    different reduction strategies such as `mean`, `sum`, or `none`.

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
        weight: None | Tensor = None,
        size_average: BoolLikeType = None,
        reduce: BoolLikeType = None,
        reduction: str = "mean",
    ) -> None:
        """Initialise `NLLLoss` instance with a reduction strategy."""
        super().__init__(size_average, reduce, reduction)
        if weight:
            raise RuntimeError("weight is currently not supported")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the NLL loss between `input` and `target` tensors.

        This method calculates the negative log likelihood between the
        input and target tensors, and applies the reduction strategy
        specified during initialisation.

        :param input: The predicted tensor.
        :param target: The target tensor.
        :return: A tensor representing the computed NLL loss, with
            reduction applied as per the configured strategy.
        """
        if target.dtype != slowtorch.int64:
            dtype = target.dtype.typename[:-6]
            raise RuntimeError(f"expected scalar type Long but found {dtype}")
        batch, _ = input.shape
        try:
            N, _ = target.shape
        except ValueError:
            if not all(target >= 0):
                raise IndexError("Target < 0 is out of bounds")
            return F.nll_loss(input, target, self.reduction)
        else:
            raise ValueError(
                f"Expected input batch_size ({batch}) to match target "
                f"batch_size ({N})"
            )


@set_module("slowtorch.nn.modules.loss")
@modules_dispatch
class CrossEntropyLoss(_Loss):
    """Cross Entropy Loss module.

    This class calculates the cross entropy loss between the logit
    (`input`) and the target tensor (`target`). It supports different
    reduction strategies such as `mean`, `sum`, or `none`.

    :param size_average: If `True`, set reduction strategy to `mean`,
        defaults to `None`.
    :param reduce: If `True`, set reduction strategy to `sum`, defaults
        to `None`.
    :param reduction: Specify the reduction method to apply to the loss.
        Acceptable values are `mean`, `sum`, and `none`, defaults to
        `mean`.
    :raises ValueError: If conflicting parameters are provided or if the
        `reduction` value is invalid.

    .. note::

        This is equivalent to applying `LogSoftmax` on an input,
        followed by `NLLLoss`.
    """

    reduction: str

    def __init__(
        self,
        weight: None | Tensor = None,
        size_average: BoolLikeType = None,
        reduce: BoolLikeType = None,
        reduction: str = "mean",
    ) -> None:
        """Initialise `CrossEntropyLoss` instance with a reduction
        strategy.
        """
        super().__init__(size_average, reduce, reduction)
        if weight:
            raise RuntimeError("weight is currently not supported")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute the cross entropy loss between `input` (logits) and
        `target` tensors.

        This method calculates the cross entropy loss between the
        input and target tensors, and applies the reduction strategy
        specified during initialisation.

        :param input: The predicted tensor (logits).
        :param target: The target tensor.
        :return: A tensor representing the computed Cross Entropy loss,
            with reduction applied as per the configured strategy.
        """
        if target.dtype != slowtorch.int64:
            dtype = target.dtype.typename[:-6]
            raise RuntimeError(f"expected scalar type Long but found {dtype}")
        batch, _ = input.shape
        try:
            N, _ = target.shape
        except ValueError:
            if not all(target >= 0):
                raise IndexError("Target < 0 is out of bounds")
            return F.cross_entropy(input, target, self.reduction)
        else:
            raise ValueError(
                f"Expected input batch_size ({batch}) to match target "
                f"batch_size ({N})"
            )
