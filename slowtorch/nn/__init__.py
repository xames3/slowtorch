"""\
SlowTorch Neural Network
========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Wednesday, January 15 2025
Last updated on: Sunday, June 01 2025
"""

from . import functional
from .modules import *

__all__: list[str] = [
    "functional",
    "modules",
]

__all__ += modules.__all__
