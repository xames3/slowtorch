"""\
SlowTorch Stateless Reduction
=============================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, June 01 2025
Last updated on: Sunday, June 01 2025

Reduction operations.

This module offers various stateless functions that perform reduction
operations. These functions mimic the API and semantics of PyTorch's
reduction functions.

.. note::

    All functions support auto-differentiation (backpropagation).
"""

from __future__ import annotations

import builtins
import statistics
import typing as t
from itertools import product as pdt

import slowtorch
from slowtorch import function_dispatch
from slowtorch.internal.shape import unravel_index
from slowtorch.internal.tensor import Node
from slowtorch.internal.tensor import Tensor

if t.TYPE_CHECKING:
    from slowtorch.types import BoolLikeType
    from slowtorch.types import Dim
    from slowtorch.types import IntLikeType


@function_dispatch
def sum(
    input: Tensor,
    dim: None | Dim = None,
    keepdim: BoolLikeType = False,
) -> Tensor:
    """Compute the sum of elements in the tensor across a specified
    dimension.

    This function computes the sum of all elements in the tensor if no
    dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be summed.
    :param dim: The dimension along which to compute the sum, defaults
        to `None`. For `None`, the sum is computed over all elements of
        the tensor.
    :param keepdim: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the sum of the specified elements.
    :raises IndexError: If the specified dimension is invalid.
    """
    ndim = input.ndim
    dtype = input.dtype
    requires_grad = input.requires_grad
    if dim is not None:
        if not (-ndim <= dim < ndim):
            raise IndexError(
                "Dimension out of range (expected to be in the range "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim
    if dim is None:
        shape = (1,) * ndim if keepdim else ()
        new_tensor = Tensor(
            shape or (1,),
            dtype=dtype,
            requires_grad=requires_grad,
        )
        new_tensor[...] = builtins.sum(input)
    else:
        shape = tuple(
            (1 if index == dim and keepdim else input._shape[index])
            for index in range(ndim)
            if keepdim or index != dim
        )
        new_tensor = Tensor(
            shape or (1,),
            dtype=dtype,
            requires_grad=requires_grad,
        )
        for index in range(new_tensor.nelement()):
            unravelled = unravel_index(index, shape)
            indices = list(unravelled)
            if keepdim:
                indices[dim] = slice(None)
            else:
                indices = indices[:dim] + [slice(None)] + indices[dim:]
            sliced = input[tuple(indices)]
            storage = sliced if isinstance(sliced, Tensor) else [sliced]
            new_tensor[unravelled] = builtins.sum(storage)

    def SumBackward0() -> None:
        """Backward pass for the sum operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if not hasattr(input, "grad") or input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        input.grad += slowtorch.ones(*input._shape)

    new_tensor.grad_fn = Node(SumBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def max(
    input: Tensor,
    dim: None | Dim = None,
    keepdim: BoolLikeType = False,
) -> Tensor:
    """Return the maximum of elements in the tensor across a
    specified dimension.

    This function returns the maximum of all elements in the tensor if
    no dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be computed for maximum values.
    :param dim: The dimension along which to compute the maximum,
        defaults to `None`. For `None`, the maximum is computed over all
        elements of the tensor.
    :param keepdim: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the maximum of the specified
        elements.
    :raises IndexError: If the specified dimension is invalid.
    """
    ndim = input.ndim
    dtype = input.dtype
    requires_grad = input.requires_grad
    if dim is not None:
        if not (-ndim <= dim < ndim):
            raise IndexError(
                "Dimension out of range (expected to be in the range "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim
    if dim is None:
        shape = (1,) * ndim if keepdim else ()
        new_tensor = Tensor(
            shape or (1,),
            dtype=dtype,
            requires_grad=requires_grad,
        )
        new_tensor[...] = builtins.max(input)
    else:
        shape = tuple(
            (1 if index == dim and keepdim else input._shape[index])
            for index in range(ndim)
            if keepdim or index != dim
        )
        new_tensor = Tensor(
            shape or (1,),
            dtype=dtype,
            requires_grad=requires_grad,
        )
        for index in range(new_tensor.nelement()):
            unravelled = unravel_index(index, shape)
            indices = list(unravelled)
            if keepdim:
                indices[dim] = slice(None)
            else:
                indices = indices[:dim] + [slice(None)] + indices[dim:]
            sliced = input[tuple(indices)]
            storage = sliced if isinstance(sliced, Tensor) else [sliced]
            new_tensor[unravelled] = builtins.max(storage)

    def MaxBackward0() -> None:
        """Backward pass for the maximum operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        if dim is None:
            grad = new_tensor.grad
            storage = input.max().item()
            for index in pdt(*[range(dim) for dim in input._shape]):
                input.grad[index] += grad if input[index] == storage else 0.0
        else:
            for index in pdt(*[range(index) for index in new_tensor._shape]):
                indices = list(index)
                if keepdim:
                    indices[dim] = slice(None)
                else:
                    indices = indices[:dim] + [slice(None)] + indices[dim:]
                slices = input[tuple(indices)]
                if not isinstance(slices, Tensor):
                    slices = [slices]
                else:
                    slices = list(slices)
                storage = builtins.max(slices)
                grad = new_tensor.grad[index]
                if isinstance(grad, Tensor):
                    grad = grad.item()
                for slice_index, slice_storage in enumerate(slices):
                    if slice_storage == storage:
                        slice_indices = list(index)
                        if keepdim:
                            slice_indices[dim] = slice_index
                        else:
                            slice_indices.insert(dim, slice_index)
                        input.grad[tuple(slice_indices)] += grad

    new_tensor.grad_fn = Node(MaxBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def min(
    input: Tensor,
    dim: None | Dim = None,
    keepdim: BoolLikeType = False,
) -> Tensor:
    """Return the minimum of elements in the tensor across a
    specified dimension.

    This function returns the minimum of all elements in the tensor if
    no dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be computed for minimum values.
    :param dim: The dimension along which to compute the minimum,
        defaults to `None`. For `None`, the minimum is computed over all
        elements of the tensor.
    :param keepdim: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the minimum of the specified
        elements.
    :raises IndexError: If the specified dimension is invalid.
    """
    ndim = input.ndim
    dtype = input.dtype
    requires_grad = input.requires_grad
    if dim is not None:
        if not (-ndim <= dim < ndim):
            raise IndexError(
                "Dimension out of range (expected to be in the range "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim
    if dim is None:
        shape = (1,) * ndim if keepdim else ()
        new_tensor = Tensor(
            shape or (1,),
            dtype=dtype,
            requires_grad=requires_grad,
        )
        new_tensor[...] = builtins.min(input)
    else:
        shape = tuple(
            (1 if index == dim and keepdim else input._shape[index])
            for index in range(ndim)
            if keepdim or index != dim
        )
        new_tensor = Tensor(
            shape or (1,),
            dtype=dtype,
            requires_grad=requires_grad,
        )
        for index in range(new_tensor.nelement()):
            unravelled = unravel_index(index, shape)
            indices = list(unravelled)
            if keepdim:
                indices[dim] = slice(None)
            else:
                indices = indices[:dim] + [slice(None)] + indices[dim:]
            sliced = input[tuple(indices)]
            storage = sliced if isinstance(sliced, Tensor) else [sliced]
            new_tensor[unravelled] = builtins.min(storage)

    def MinBackward0() -> None:
        """Backward pass for the minimum operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        if dim is None:
            grad = new_tensor.grad
            storage = input.min().item()
            for index in pdt(*[range(dim) for dim in input._shape]):
                input.grad[index] += grad if input[index] == storage else 0.0
        else:
            for index in pdt(*[range(index) for index in new_tensor._shape]):
                indices = list(index)
                if keepdim:
                    indices[dim] = slice(None)
                else:
                    indices = indices[:dim] + [slice(None)] + indices[dim:]
                slices = input[tuple(indices)]
                if not isinstance(slices, Tensor):
                    slices = [slices]
                else:
                    slices = list(slices)
                storage = builtins.min(slices)
                grad = new_tensor.grad[index]
                if isinstance(grad, Tensor):
                    grad = grad.item()
                for slice_index, slice_storage in enumerate(slices):
                    if slice_storage == storage:
                        slice_indices = list(index)
                        if keepdim:
                            slice_indices[dim] = slice_index
                        else:
                            slice_indices.insert(dim, slice_index)
                        input.grad[tuple(slice_indices)] += grad

    new_tensor.grad_fn = Node(MinBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def mean(
    input: Tensor,
    dim: None | Dim = None,
    keepdim: BoolLikeType = False,
) -> Tensor:
    """Return the mean of elements in the tensor across a specified
    dimension.

    This function returns the mean of all elements in the tensor if
    no dimension is provided. If a dimension is specified, the function
    reduces the tensor along the given dimension while optionally
    retaining the reduced dimensions.

    :param input: Input tensor to be computed for mean values.
    :param dim: The dimension along which to compute the mean, defaults
        to `None`. For `None`, the mean is computed over all elements of
        the tensor.
    :param keepdim: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the mean of the specified elements.
    :raises IndexError: If the specified dimension is invalid.
    """
    ndim = input.ndim
    dtype = input.dtype
    requires_grad = input.requires_grad
    if dim is not None:
        if not (-ndim <= dim < ndim):
            raise IndexError(
                "Dimension out of range (expected to be in the range "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim
    if dim is None:
        shape = (1,) * ndim if keepdim else ()
        new_tensor = Tensor(
            shape or (1,),
            dtype=dtype,
            requires_grad=requires_grad,
        )
        new_tensor[...] = statistics.mean(input)
    else:
        shape = tuple(
            (1 if index == dim and keepdim else input._shape[index])
            for index in range(ndim)
            if keepdim or index != dim
        )
        new_tensor = Tensor(
            shape or (1,),
            dtype=dtype,
            requires_grad=requires_grad,
        )
        for index in range(new_tensor.nelement()):
            unravelled = unravel_index(index, shape)
            indices = list(unravelled)
            if keepdim:
                indices[dim] = slice(None)
            else:
                indices = indices[:dim] + [slice(None)] + indices[dim:]
            sliced = input[tuple(indices)]
            storage = sliced if isinstance(sliced, Tensor) else [sliced]
            new_tensor[unravelled] = statistics.mean(storage)

    def MeanBackward0() -> None:
        """Backward pass for the mean operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        if dim is None:
            grad = new_tensor.grad
            storage = input.max().item()
            for index in pdt(*[range(dim) for dim in input._shape]):
                input.grad[index] += grad if input[index] == storage else 0.0
        else:
            for index in pdt(*[range(index) for index in new_tensor._shape]):
                indices = list(index)
                if keepdim:
                    indices[dim] = slice(None)
                else:
                    indices = indices[:dim] + [slice(None)] + indices[dim:]
                slices = input[tuple(indices)]
                if not isinstance(slices, Tensor):
                    slices = [slices]
                    denominator = len(slices)
                else:
                    denominator = slices.nelement()
                grad = new_tensor.grad[index] / denominator
                if isinstance(grad, Tensor):
                    grad = grad.item()
                for slice_index, _ in enumerate(slices):
                    slice_indices = list(index)
                    if keepdim:
                        slice_indices[dim] = slice_index
                    else:
                        slice_indices.insert(dim, slice_index)
                    input.grad[tuple(slice_indices)] += grad

    new_tensor.grad_fn = Node(MeanBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor


@function_dispatch
def std(
    input: Tensor,
    dim: None | Dim = None,
    correction: IntLikeType = 1,
    keepdim: BoolLikeType = False,
) -> Tensor:
    """Return the standard deviation of elements in the tensor across a
    specified dimension.

    This function returns the standard deviation of all elements in the
    tensor if no dimension is provided. If a dimension is specified, the
    function reduces the tensor along the given dimension while
    optionally retaining the reduced dimensions.

    :param input: Input tensor to be computed for standard deviation
        values.
    :param dim: The dimension along which to compute the standard
        deviation, defaults to `None`. For `None`, the standard
        deviation is computed over all elements of the tensor.
    :param correction: Difference between the sample size and sample
        degrees of freedom, defaults to `1`.
    :param keepdim: A boolean indicating whether to retain the reduced
        dimensions in the resulting tensor, defaults to `False`.
    :return: A new tensor containing the standard deviation of the
        specified elements.
    :raises IndexError: If the specified dimension is invalid.
    """
    ndim = input.ndim
    dtype = input.dtype
    requires_grad = input.requires_grad
    if dim is not None:
        if not (-ndim <= dim < ndim):
            raise IndexError(
                "Dimension out of range (expected to be in the range "
                f"[{-ndim}, {ndim - 1}], but got {dim})"
            )
        if dim < 0:
            dim += ndim
    if dim is None:
        shape = (1,) * ndim if keepdim else ()
        new_tensor = Tensor(
            shape or (1,),
            dtype=dtype,
            requires_grad=requires_grad,
        )
        new_tensor[...] = statistics.stdev(input)
    else:
        shape = tuple(
            (1 if index == dim and keepdim else input._shape[index])
            for index in range(ndim)
            if keepdim or index != dim
        )
        new_tensor = Tensor(
            shape or (1,),
            dtype=dtype,
            requires_grad=requires_grad,
        )
        for index in range(new_tensor.nelement()):
            unravelled = unravel_index(index, shape)
            indices = list(unravelled)
            if keepdim:
                indices[dim] = slice(None)
            else:
                indices = indices[:dim] + [slice(None)] + indices[dim:]
            sliced = input[tuple(indices)]
            storage = sliced if isinstance(sliced, Tensor) else [sliced]
            new_tensor[unravelled] = statistics.stdev(storage)

    def StdBackward0() -> None:
        """Backward pass for the standard deviation operation.

        Distributes the gradient from the resulting tensor back to the
        input tensor. If `dim` was specified, the gradient is
        appropriately expanded to match the original tensor's shape.
        """
        if input.grad is None:
            input.grad = Tensor(input._shape, input.dtype)
        if dim is None:
            mu = input.mean()
            N = input.nelement()
            grad = new_tensor.grad / ((N - correction) * new_tensor)
            for index in pdt(*[range(dim) for dim in input._shape]):
                input.grad[index] += grad * (input[index] - mu)
        else:
            for index in pdt(*[range(dim) for dim in new_tensor._shape]):
                indices = list(index)
                if keepdim:
                    indices[dim] = slice(None)
                else:
                    indices = indices[:dim] + [slice(None)] + indices[dim:]
                slices = input[tuple(indices)]
                mu = slices.mean().item()
                x = new_tensor[index]
                N = slices.nelement()
                grad = new_tensor.grad[index] / ((N - correction) * x)
                for slice_index, slice_storage in enumerate(slices):
                    dims = list(index)
                    if keepdim:
                        dims[dim] = slice_index
                    else:
                        dims.insert(dim, slice_index)
                    input.grad[tuple(dims)] += grad * (slice_storage - mu)

    new_tensor.grad_fn = Node(StdBackward0)
    new_tensor.grad_fn.inputs = (input,)
    return new_tensor
