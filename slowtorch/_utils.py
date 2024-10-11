"""\
SlowTorch Utilities API
=======================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Thursday, October 10 2024
Last updated on: Friday, October 11 2024

This module provides utility classes, functions, and objects that are
essential to the core operations of SlowTorch. It is intended to
encapsulate common functionality and helpers that simplify the
development of the overall library, promoting code reuse and modular
design. The `utils` module includes utility classes and objects that are
frequently used across the SlowTorch framework, such as custom
decorators, helper functions, and type handling tools (e.g., `dtype`).
By centralizing frequently used utilities, the module aims to maintain
a clean and modular structure, enabling easy maintenance, extensibility,
and testability.

These utilities are not limited to specific aspects of the SlowTorch
framework but are general-purpose, allowing them to support multiple
components, such as tensor manipulations, mathematical operations, and
type systems. For instance, this module defines a custom `dtype` class
that mimics PyTorch's `dtype` object, representing various data types
(e.g., `float32`, `int64`, etc.). These data types are central to the
tensor operations within SlowTorch.

This module is critical to the SlowTorch project's overarching goals of
simplicity and modularity. By leveraging only Python's standard library,
SlowTorch intentionally forgoes third-party dependencies, making the
utility functions, classes, and objects within `utils.py` essential for
achieving flexibility, performance, and code clarity. The utilities
housed here support the core computational structures of SlowTorch,
aiding in data type management, object construction, and internal
operations.

The `utils` module is intended for developers contributing to SlowTorch,
as well as any users who wish to understand the underlying mechanics of
the framework. It is designed with simplicity and transparency in mind,
making it an ideal reference for those learning about deep learning
library construction or looking to extend SlowTorch's capabilities.
"""


class device:
    """Class to represent a computational device with a specified type
    and an optional index.

    This class allows the specification of a device that can be used for
    computations in the SlowTorch framework. The `type` of the device
    refers to the computational backend, while the optional `index`
    specifies the device number for multi-device systems.

    :param type: The type of the device, e.g., `cpu`, `mps`, etc.
    :param index: An optional index representing the device number,
        defaults to 0.
    """

    type: str = "cpu"
    index: int = 0

    def __init__(self, type: str = "cpu", index: int = 0) -> None:
        """Initialize a new `device` object with default index."""
        self.type = type
        self.index = index

    def __repr__(self) -> str:
        """Return a string representation of the `device` object."""
        return f"{type(self).__name__}(type={self.type!r}, index={self.index})"

    def __str__(self) -> str:
        """Return a human-readable string representation of the `device`
        object.
        """
        return self.type


class dtype:
    """Class to represents data types used in the SlowTorch framework.

    The `dtype` class encapsulates information about different data
    types that can be used for tensor operations and storage. The `name`
    attribute stores the name of the data type, while `itemsize`
    indicates the size in bytes for each item of this type.

    :param name: The name of the data type.
    :param itemsize: The size in bytes of the data type.
    """

    def __init__(self, name: str, itemsize: int):
        """Initializes a new `dtype` object."""
        self.name = name
        self.itemsize = itemsize

    def __repr__(self) -> str:
        """Return a string representation of the `dtype` object."""
        return f"{type(self).__name__}({self.name})"

    def __str__(self) -> str:
        """Return a human-readable string representation of the `dtype`
        object.
        """
        return self.name
