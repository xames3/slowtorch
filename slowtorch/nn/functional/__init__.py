"""\
SlowTorch Functional
====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, 01 June 2025
Last updated on: Sunday, 24 August 2025

This module offers a comprehensive suite of stateless functions that
perform various operations (albeit in function form).
"""

from .layer import *
from .loss import *
from .mutation import *
from .pointwise import *
from .reduction import *

__all__: list[str] = [
    "layer",
    "loss",
    "pointwise",
    "mutation",
    "reduction",
]
