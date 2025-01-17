"""\
SlowTorch Neural Network APIs
=============================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Wednesday, January 15 2025
Last updated on: Friday, January 16 2025
"""

from . import functional
from .modules import *

__all__: list[str] = [
    "functional",
    "modules",
]

__all__ += modules.__all__
