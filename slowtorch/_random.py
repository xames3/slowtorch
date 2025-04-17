"""\
SlowTorch Random API
====================

Author: Akshay Mestry <xa@mes3.dev>
Created on: Monday, January 13 2025
Last updated on: Tuesday, April 01 2025

This module implements pseudo-random number generators (PRNGs or RNGs)
with ability to draw samples from a variety of probability
distributions.
"""

from __future__ import annotations

import random

from slowtorch import function_dispatch
from slowtorch._tensor import DeviceType
from slowtorch._tensor import Tensor
from slowtorch._tensor import tensor


@function_dispatch
class Generator:
    """A random number generator supporting multiple distributions.

    This generator serves as the backbone for random number generation.
    It can produce uniform and normal distributions.

    :param device: Optional device, defaults to `None`.
    """

    __module__: str = "slowtorch._C"

    def __init__(self, device: DeviceType = None) -> None:
        """Initialise a `Generator` instance with device."""
        self.device = device
        self.internal = random.Random()
        self._seed = self.internal.randint(0, 999)

    def get_state(self) -> Tensor:
        """Retrieve the internal RNG state.

        :return: State of RNG as a `slowtorch.Tensor`.
        """
        return tensor(self.internal.getstate())

    def set_state(self, state: tuple[int, ...]) -> None:
        """Set the internal RNG state.

        :param state: State to restore the RNG to.
        """
        self.internal.setstate(state)

    def initial_seed(self) -> int:
        """Set the seed for RNG to ensure reproducibility."""
        return self._seed

    seed = initial_seed

    def manual_seed(self, seed: int) -> Generator:
        """Set the seed for RNG to ensure reproducibility."""
        self.internal.seed(seed)
        return self


default_generator = Generator()


@function_dispatch
def set_rng_state(new_state: tuple[int, ...]) -> None:
    """Set the RNG state."""
    default_generator.set_state(new_state)


@function_dispatch
def get_rng_state() -> Tensor:
    """Return the RNG state as `slowtorch.Tensor`."""
    return default_generator.get_state()


@function_dispatch
def manual_seed(seed: int) -> Generator:
    """Set the seed for generating random numbers.

    :param seed: The desired seed.
    :return: A `slowtorch.Generator` object.
    """
    return default_generator.manual_seed(seed)


@function_dispatch
def seed() -> int:
    """Set the seed for generating random numbers to a non-deterministic
    random number. Return a 64-bit number used to seed the RNG.
    """
    return default_generator.seed()


initial_seed = function_dispatch(seed)
