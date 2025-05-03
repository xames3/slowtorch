"""\
SlowTorch Optimiser APIs
========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Thursday, May 01 2025
Last updated on: Thursday, May 01 2025
"""

from .optimiser import *

__all__: list[str] = [
    "optimiser",
]

__all__ += optimiser.__all__
