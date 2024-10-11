"""\
SlowTorch Types API
===================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Thursday, October 10 2024
Last updated on: Friday, October 11 2024

This module defines and provides aliases for all simple and composite
types that are used throughout the SlowTorch framework, for instance,
`device` and `dtype` classes. The types defined here are fundamental to
the framework's core operations, such as specifying computation devices
and data types for tensors. The purpose of this module is to standardize
the handling of various data types and device representations, making it
easier for developers and users to specify these parameters in a
consistent and flexible manner across the SlowTorch library.

This module primarily serves as the type system foundation for the
SlowTorch framework, ensuring that users can flexibly and explicitly
define the data types of tensors and the devices on which computations
will occur.
"""

import builtins
import typing as t

from slowtorch._utils import device
from slowtorch._utils import dtype

_bool = builtins.bool
_float = builtins.float
_int = builtins.int
_str = builtins.str

DeviceLikeType: t.TypeAlias = _int | _str | device

_dtype = dtype
half: _dtype = dtype(name="float16", itemsize=2)
float: _dtype = dtype(name="float32", itemsize=4)
double: _dtype = dtype(name="float64", itemsize=8)
float16: _dtype = half
float32: _dtype = float
float64: _dtype = double
uint8: _dtype = dtype(name="uint8", itemsize=1)
uint16: _dtype = dtype(name="uint16", itemsize=2)
uint32: _dtype = dtype(name="uint32", itemsize=4)
uint64: _dtype = dtype(name="uint64", itemsize=8)
int8: _dtype = dtype(name="int8", itemsize=1)
short: _dtype = dtype(name="int16", itemsize=2)
int: _dtype = dtype(name="int32", itemsize=4)
long: _dtype = dtype(name="int64", itemsize=8)
int16: _dtype = short
int32: _dtype = int
int64: _dtype = long
bool: _dtype = dtype(name="bool", itemsize=1)
