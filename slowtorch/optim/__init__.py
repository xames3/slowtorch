"""\
SlowTorch Optimiser
===================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Thursday, 01 May 2025
Last updated on: Sunday, 24 August 2025
"""

from .optimiser import *

__all__: list[str] = [
    "optimiser",
]

__all__ += optimiser.__all__
