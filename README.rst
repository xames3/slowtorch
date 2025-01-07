.. Author: Akshay Mestry <xa@mes3.dev>
.. Created on: Thursday, October 10 2024
.. Last updated on: Tuesday, January 07 2025

===============================================================================
SlowTorch
===============================================================================

**SlowTorch** is another personal pet-project of mine where I tried and
implemented the basic and bare-bones functionality of `PyTorch`_ just using
pure Python, similar to what I did with `xsNumPy`_. This project is also a
testament to the richness of PyTorch's Tensor-oriented design. By
reimplementing its core features in a self-contained and minimalistic fashion,
this project aims to:

- Provide an educational tool for those seeking to understand tensor and
  automatic gradient (backpropagation) mechanics.
- Encourage developers to explore the intricacies of multidimensional
  array computation.

This project acknowledges the incredible contributions of the PyTorch team and
community over decades of development. While this module reimagines PyTorch's
functionality, it owes its design, inspiration, and motivation to the
pioneering work of the core PyTorch developers. If that's obvious, this module
is not a replacement for PyTorch by any stretch but an homage to its
brilliance and an opportunity to explore its concepts from the ground up.

**SlowTorch** is a lightweight, pure-Python library inspired by PyTorch,
designed to mimic essential tensor operations and auto-differentiation
(backpropagation) features. This project is ideal for learning and
experimentation with multidimensional tensor processing.

-------------------------------------------------------------------------------
Installation
-------------------------------------------------------------------------------

.. See more at: https://stackoverflow.com/a/15268990

Install the latest version of SlowTorch using `pip`_:

.. code-block:: bash

    pip install -U git+https://github.com/xames3/slowtorch.git#egg=slowtorch

-------------------------------------------------------------------------------
Usage and Documentation
-------------------------------------------------------------------------------

The codebase is structured to be intuitive and mirrors the design principles
of PyTorch to a significant extent. Comprehensive docstrings are provided for
each module and function, ensuring clarity and ease of understanding. Users
are encouraged to delve into the code, experiment with it, and modify it to
suit their learning curve.

Since, the implementation doesn't rely on any external package, it will work
with any CPython build v3.10 and higher. Technically, it should work on 3.9 and
below as well but due to some syntactical and type-aliasing changes, it might
not support. For instance, the typing module was significantly changed in
3.10, hence some features like `types.GenericAlias` and using native types
like `tuple`, `list`, etc. will not work. If you remove all the typing stuff,
the code will work just fine, at least that's what I hope.

**Note.** SlowTorch cannot and should not be used as an alternative to PyTorch.

-------------------------------------------------------------------------------
Contributions and Feedback
-------------------------------------------------------------------------------

Contributions to this project are warmly welcomed. Whether it's refining the
code, enhancing the documentation, or extending the current feature set, your
input is highly valued. Feedback, whether constructive criticism or 
commendation, is equally appreciated and will be instrumental in the evolution
of this educational tool.

-------------------------------------------------------------------------------
Acknowledgments
-------------------------------------------------------------------------------

This project is inspired by the remarkable work done by the `PyTorch
Development Team`_. It is a tribute to their contributions to the field of
machine learning and the open-source community at large.

-------------------------------------------------------------------------------
License
-------------------------------------------------------------------------------

SlowTorch is licensed under the MIT License. See the `LICENSE`_ file for more
details.

.. _LICENSE: https://github.com/xames3/slowtorch/blob/main/LICENSE
.. _PyTorch Development Team: https://pytorch.org/docs/main/community/
  persons_of_interest.html
.. _PyTorch: https://pytorch.org
.. _pip: https://pip.pypa.io/en/stable/getting-started/
.. _xsNumPy: https://github.com/xames3/xsnumpy
