"""\
SlowTorch Types API
===================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, January 07 2025
Last updated on: Saturday, May 24 2025

This module defines and provides aliases for all simple and composite
types that are used throughout the SlowTorch framework. The types defined
here are fundamental to the framework's core operations, such as
specifying computation devices and data types for tensors. The purpose
of this module is to standardise the handling of various data types and
device representations, making it easier for developers and users to
specify these parameters in a consistent and flexible manner across the
SlowTorch library.

This module primarily serves as the type system foundation for the
SlowTorch framework, ensuring that users can flexibly and explicitly
define the data types of tensors and the devices on which computations
will occur.
"""

from __future__ import annotations

import os
import typing as t

__all__: list[str] = [
    "ArrayLike",
    "DTypeLike",
    "DTypeLikeNested",
    "FILE_LIKE",
    "Number",
    "VoidDTypeLike",
]

Number: t.TypeAlias = int | float | bool
ArrayLike = t.TypeVar("ArrayLike")
DTypeLikeNested: t.TypeAlias = t.Any
VoidDTypeLike: t.TypeAlias = (
    tuple[DTypeLikeNested, int]
    | tuple[DTypeLikeNested]
    | list[t.Any]
    | tuple[DTypeLikeNested, DTypeLikeNested]
)
DTypeLike: t.TypeAlias = None | type[t.Any] | str | VoidDTypeLike
FILE_LIKE: t.TypeAlias = (
    int | str | bytes | os.PathLike[str] | os.PathLike[bytes]
)
IndexLike: t.TypeAlias = int | slice | tuple[None | int | slice, ...]
