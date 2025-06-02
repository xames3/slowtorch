"""\
SlowTorch Modules
=================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Sunday, June 01 2025
Last updated on: Sunday, June 01 2025

This module unifies all the child modules into one.
"""

import typing as t

__all__: list[str] = []

T = t.TypeVar("T")


def modules_dispatch(klass: t.Callable[..., T]) -> t.Callable[..., T]:
    """Decorator to register a classes from `modules` in the namespace.

    :param klass: The class to be registered and exposed.
    :return: The original class, unmodified.
    """
    globals()[klass.__name__] = klass
    __all__.append(klass.__name__)
    return klass


from .activation import *
from .layer import *
from .loss import *
from .module import *
from .parameter import *
