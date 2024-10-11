"""\
SlowTorch
=========

Author: Akshay Mestry <xa@mes3.dev>
Created on: Thursday, October 10 2024
Last updated on: Friday, October 11 2024

Yet another small-scale implementation of PyTorch from the ground up.
But this time, for real. To put it simply, SlowTorch is a minimalist
reimplementation of PyTorch, like `NanoTorch` but this time around using
only standard Python libraries.
"""

from __future__ import annotations

from slowtorch._types import bool
from slowtorch._types import double
from slowtorch._types import float
from slowtorch._types import float16
from slowtorch._types import float32
from slowtorch._types import float64
from slowtorch._types import half
from slowtorch._types import int
from slowtorch._types import int8
from slowtorch._types import int16
from slowtorch._types import int32
from slowtorch._types import int64
from slowtorch._types import long
from slowtorch._types import short
from slowtorch._types import uint8
from slowtorch._types import uint16
from slowtorch._types import uint32
from slowtorch._types import uint64
from slowtorch._utils import device
from slowtorch._utils import dtype

version: str = "2024.10.10"
