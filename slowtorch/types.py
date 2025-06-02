"""\
SlowTorch Types
===============

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, January 07 2025
Last updated on: Sunday, June 01 2025

Type aliases.

This module defines a collection of semantic type aliases which are used
throughout the SlowTorch framework. These aliases serve as a lightweight
but expressive type system, improving the readability, maintainability,
and robustness of the codebase, especially in function signatures,
operator dispatch, and tensor creation logic.

All types declared here represent primitive or composite abstractions
such as numerical scalars, sequence of scalars (i.e., array-like), and
data representation of data types (dtype) and file paths. By codifying
these patterns into aliases, the library avoids ad hoc type assumptions
and encourages clear developer intent.
"""

from __future__ import annotations

import os
import types
import typing as t

if t.TYPE_CHECKING:
    from slowtorch.internal.device import device
    from slowtorch.internal.dtype import dtype
    from slowtorch.internal.shape import size
    from slowtorch.internal.tensor import Tensor

__all__: list[str] = [
    "ArrayLike",
    "ArrayLikeOrScalar",
    "BoolLikeType",
    "DType",
    "DeviceType",
    "Dim",
    "FileLike",
    "FloatLikeType",
    "IndexLike",
    "Input",
    "IntLikeType",
    "Scalar",
    "ShapeType",
    "Size",
    "StorageWeakRef",
    "StrideType",
    "TensorOrTensors",
]

BoolLikeType: t.TypeAlias = bool
IntLikeType: t.TypeAlias = int
FloatLikeType: t.TypeAlias = float
DType: t.TypeAlias = dtype
Size: t.TypeAlias = size
Dim: t.TypeAlias = IntLikeType
ShapeType: t.TypeAlias = Size | tuple[IntLikeType, ...]
DeviceType: t.TypeAlias = str | IntLikeType | device
Scalar: t.TypeAlias = IntLikeType | FloatLikeType | BoolLikeType
Input: t.TypeAlias = Scalar | Tensor
ArrayLike: t.TypeAlias = t.Sequence[Scalar]
ArrayLikeOrScalar: t.TypeAlias = ArrayLike | Scalar
StorageWeakRef: t.TypeAlias = t.Sequence[Input]
StrideType: t.TypeAlias = list[IntLikeType] | tuple[IntLikeType, ...]
TensorOrTensors: t.TypeAlias = tuple[Tensor, ...] | Tensor
FileLike: t.TypeAlias = (
    IntLikeType | str | bytes | os.PathLike[str] | os.PathLike[bytes]
)
IndexLike: t.TypeAlias = (
    None
    | IntLikeType
    | slice
    | types.EllipsisType
    | tuple[None | IntLikeType | slice | types.EllipsisType, ...]
)
