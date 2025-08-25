"""\
SlowTorch
=========

Author: Akshay Mestry <xa@mes3.dev>
Created on: Tuesday, 07 January 2025
Last updated on: Sunday, 24 August 2025

SlowTorch is yet another personal pet-project of mine where I tried and
implemented the basic and bare-bones functionality of PyTorch just using
pure Python, similar to what I did with ``xsNumPy``. This project is
also a testament to the richness of PyTorch's Tensor-oriented design. By
reimplementing its core features in a self-contained and minimalistic
fashion, this project aims to:

- Provide an educational tool for those seeking to understand tensor and
  automatic gradient (backpropagation) mechanics.
- Encourage developers to explore the intricacies of multidimensional
  array computation.

This project acknowledges the incredible contributions of the PyTorch
team and community over decades of development. While this module
reimagines PyTorch's functionality, it owes its design, inspiration, and
motivation to the pioneering work of the core PyTorch developers. If
that's obvious, this module is not a replacement for PyTorch by any
stretch but an homage to its brilliance and an opportunity to explore
its concepts from the ground up.

SlowTorch is a lightweight, pure-Python library inspired by PyTorch,
designed to mimic essential tensor operations and auto-differentiation
(backpropagation) features. This project is ideal for learning and
experimentation with multidimensional tensor processing.
"""

from __future__ import annotations

import typing as t

import slowtorch.internal as internal

if t.TYPE_CHECKING:
    from slowtorch import types

__all__: list[str] = [
    "function_dispatch",
    "nn",
    "optim",
    "random",
    "types",
    "version",
]

T = t.TypeVar("T")

version: str = "30.6.2025"


def function_dispatch(func: t.Callable[..., T]) -> t.Callable[..., T]:
    """Decorator to register a function in the global namespace.

    This utility allows for automatic exposure of decorated functions to
    module-level imports. It ensures the function is added to both
    the global scope and the `__all__` list for proper namespace
    management.

    :param func: The function to be registered and exposed.
    :return: The original function, unmodified.
    """
    globals()[func.__name__] = func
    __all__.append(func.__name__)
    return func


from .internal.dtype import *
from .internal.tensor import *
from .ops.creation import *
from .ops.mutation import *
from .ops.random import *
from .utils import *

__all__ += internal.dtype.__all__
__all__ += utils.__all__

import slowtorch.internal.random as random
import slowtorch.nn as nn
import slowtorch.optim as optim
