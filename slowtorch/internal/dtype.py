"""\
SlowTorch Dtypes
================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, 01 June 2025
Last updated on: Sunday, 24 August 2025

Dtype objects.

This module defines and registers all supported data types (dtypes) used
throughout this library. These dtypes specify the kind of scalar values
that can be stored within tensors, including various integer,
floating-point, and boolean types.

Each data type is represented using the dtype class, which encapsulates
metadata such as its name, byte size, underlying C representation (via
ctypes), default value, and associated tensor type name. These dtypes
are used internally for memory layout decisions, tensor construction,
type promotion, and computation.
"""

from __future__ import annotations

import ctypes
import typing as t

from slowtorch import function_dispatch
from slowtorch.utils import set_module

if t.TYPE_CHECKING:
    from slowtorch.types import Scalar

__all__: list[str] = [
    "bool",
    "double",
    "float32",
    "float64",
    "int16",
    "int32",
    "int64",
    "int8",
    "long",
    "short",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
]


@set_module("slowtorch")
@function_dispatch
class dtype:
    """Class to represent a data type.

    The `dtype` class encapsulates information about supported datatypes
    for tensor operations and storage. It provides a way to describe the
    type of data stored in tensors, its size in bytes, and associated
    metadata.

    :param name: The full name of the data type.
    :param short: A shorthand representation of the data type, where the
        last character specifies the size in bytes.
    :param data: A representation of the type's data structure or
        internal details.
    :param value: A representative value for the data type, used for
        internal operations or comparisons.
    :param typename: A `typename` similar to PyTorch tensors.
    """

    def __init__(
        self,
        name: str,
        short: str,
        data: t.Any,
        value: Scalar,
        typename: str,
    ) -> None:
        """Initialise a new `dtype` object with name and value."""
        self.name = name
        self.itemsize = int(short[-1])
        self.data = data
        self.value = value
        self.typename = typename

    def __repr__(self) -> str:
        """Return a string representation of the `dtype` object."""
        return f"slowtorch.{self.name}"


bool = dtype("bool", "b1", ctypes.c_bool, False, "BoolTensor")
int8 = dtype("int8", "i1", ctypes.c_int8, 0, "CharTensor")
uint8 = dtype("uint8", "u1", ctypes.c_uint8, 0, "ByteTensor")
short = int16 = dtype("int16", "i2", ctypes.c_int16, 0, "ShortTensor")
uint16 = dtype("uint16", "u2", ctypes.c_uint16, 0, "UShortTensor")
int32 = dtype("int32", "i4", ctypes.c_int32, 0, "IntTensor")
uint32 = dtype("uint32", "u4", ctypes.c_uint32, 0, "UIntTensor")
long = int64 = dtype("int64", "i8", ctypes.c_int64, 0, "LongTensor")
uint64 = dtype("uint64", "u8", ctypes.c_uint64, 0, "ULongTensor")
float32 = dtype("float32", "f4", ctypes.c_float, 0.0, "FloatTensor")
double = float64 = dtype("float64", "f8", ctypes.c_double, 0.0, "DoubleTensor")
