"""\
SlowTorch Neural Network
========================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Wednesday, 15 January 2025
Last updated on: Sunday, 24 August 2025
"""

from . import functional
from .modules import *

__all__: list[str] = [
    "functional",
    "modules",
]

__all__ += modules.__all__
