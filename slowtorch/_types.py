"""\
SlowTorch Types API
===================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, January 07 2025
Last updated on: Sunday, January 12 2025

This module defines and provides aliases for all simple and composite
types that are used throughout the SlowTorch framework. The types defined
here are fundamental to the framework's core operations, such as
specifying computation devices and data types for tensors. The purpose
of this module is to standardize the handling of various data types and
device representations, making it easier for developers and users to
specify these parameters in a consistent and flexible manner across the
SlowTorch library.

This module primarily serves as the type system foundation for the
SlowTorch framework, ensuring that users can flexibly and explicitly
define the data types of tensors and the devices on which computations
will occur.
"""

from __future__ import annotations

import typing as t

__all__: list[str] = [
    "DTypeLike",
    "DTypeLikeNested",
    "Number",
    "ArrayLike",
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
