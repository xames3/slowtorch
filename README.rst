.. Author: Akshay Mestry <xa@mes3.dev>
.. Created on: Thursday, October 10 2024
.. Last updated on: Saturday, January 25 2025

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
Features
-------------------------------------------------------------------------------

As of now, **SlowTorch** offers the following features:

SlowTorch native Tensor object (`slowtorch.Tensor`)
===============================================================================

- **slowtorch.Tensor.** The central data structure representing N-dimensional
  tensors with support for:

  - Arbitrary shapes and data types.
  - Broadcasting\*\* for compatible operations (limited).
  - Arithmetic and comparison operations.

.. code-block:: python

    >>> import slowtorch
    >>>
    >>> a = slowtorch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> b = slowtorch.tensor([[4, 1, 5, 3, 2], [1, 3, 5, 7, 2]])
    >>> 
    >>> a + b
    tensor([[ 5,  3,  8,  7,  7], 
            [ 7, 10, 13, 16, 12]])
    >>> a - b
    tensor([[-3,  1, -2,  1,  3], 
            [ 5,  4,  3,  2,  8]])
    >>> a * b
    tensor([[ 4,  2, 15, 12, 10], 
            [ 6, 21, 40, 63, 20]])
    >>> a / b
    tensor([[  0.25,     2.,    0.6, 1.3333,    2.5], 
            [    6., 2.3333,    1.6, 1.2857,     5.]])
    >>> a // b
    tensor([[ 0.,  2.,  0.,  1.,  2.], 
            [ 6.,  2.,  1.,  1.,  5.]])
    >>> a % b
    tensor([[1, 0, 3, 1, 1], 
            [0, 1, 3, 2, 0]])
    >>> a ** b
    tensor([[      1,       2,     243,      64,      25], 
            [      6,     343,   32768, 4782969,     100]])
    >>> a < b
    tensor([[ True, False,  True, False, False], 
            [False, False, False, False, False]])
    >>> a >= b
    tensor([[False,  True, False,  True,  True], 
            [ True,  True,  True,  True,  True]])

Tensor Creation Ops
===============================================================================

- **slowtorch.tensor.** Create an N-dimensional tensor.

.. code-block:: python

    >>> slowtorch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])
    tensor([[0.1, 1.2], 
            [2.2, 3.1], 
            [4.9, 5.2]])
    >>> slowtorch.tensor([[1, 3], [2, 3]])
    tensor([[1, 3], 
            [2, 3]])
    >>> slowtorch.tensor([[1, 2, 3]], dtype=slowtorch.bool)
    tensor([[True, True, True]])

- **slowtorch.empty.** Create an uninitialized tensor of the given shape.

.. code-block:: python

    >>> slowtorch.empty((2, 3))
    tensor([[ 0.,  0.,  0.], 
            [ 0.,  0.,  0.]])
    >>> slowtorch.empty((3, 3), dtype=slowtorch.int64)
    tensor([[0, 0, 0], 
            [0, 0, 0], 
            [0, 0, 0]])

- **slowtorch.zeros.** Create a tensor filled with zeros.

.. code-block:: python

    >>> slowtorch.zeros((2, 3))
    tensor([[ 0.,  0.,  0.], 
            [ 0.,  0.,  0.]])
    >>> slowtorch.zeros((2,))
    tensor([ 0.,  0.])

- **slowtorch.ones.** Create a tensor filled with ones.

.. code-block:: python

    >>> slowtorch.ones((2, 3))
    tensor([[ 1.,  1.,  1.], 
            [ 1.,  1.,  1.]])
    >>> slowtorch.ones(5)
    tensor([ 1.,  1.,  1.,  1.,  1.])

- **slowtorch.full.** Create a tensor filled with *fill_value*.

.. code-block:: python

    >>> slowtorch.full((2, 3), 3.141592)
    tensor([[3.1416, 3.1416, 3.1416], 
            [3.1416, 3.1416, 3.1416]])

- **slowtorch.arange.** Generate evenly spaced values within a given range.

.. code-block:: python

    >>> slowtorch.arange(5)
    tensor([0, 1, 2, 3, 4])
    >>> slowtorch.arange(1, 4)
    tensor([1, 2, 3])
    >>> slowtorch.arange(1, 2.5, 0.5)
    tensor([ 1., 1.5,  2.])

Tensor class reference
===============================================================================

- **Tensor.device.** Device where the tensor is.

.. code-block:: python

    >>> a = slowtorch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> a.device
    device(type='cpu', index=0)

- **Tensor.grad.** This attribute is `None` by default and becomes a
  `Tensor` the first time a call to `backward()` computes gradients for `self`.

- **Tensor.ndim.** Returns the number of dimensions of `self` tensor.
  Alias for `Tensor.dim()`.

.. code-block:: python

    >>> a = slowtorch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> a.ndim
    2
    >>> b = slowtorch.zeros((2, 3, 4))
    >>> b.ndim
    3

- **Tensor.nbytes.** Total bytes consumed by the elements of the tensor.

.. code-block:: python

    >>> a = slowtorch.zeros((3, 2), dtype=slowtorch.float64)
    >>> a
    tensor([[ 0.,  0.], 
            [ 0.,  0.], 
            [ 0.,  0.]])
    >>> a.nbytes
    48
    >>> b = slowtorch.zeros((1, 3), dtype=slowtorch.int64)
    >>> b
    tensor([[0, 0, 0]])
    >>> b.nbytes
    24

- **Tensor.itemsize.** Length of one tensor element in bytes. Alias for
  `Tensor.element_size()`.

.. code-block:: python

    >>> a = slowtorch.full((2, 3), 2.71253)
    >>> a
    tensor([[2.71253, 2.71253, 2.71253], 
            [2.71253, 2.71253, 2.71253]])
    >>> a.itemsize
    8
    >>> b = slowtorch.tensor([1, 2, 3], dtype=slowtorch.int16)
    >>> b.itemsize
    2

- **Tensor.shape.** Size of the tensor as a tuple.

.. code-block:: python

    >>> a = slowtorch.zeros((1, 3), dtype=slowtorch.int64)
    >>> a
    tensor([[0, 0, 0]])
    >>> a.shape
    (1, 3)
    >>> b = slowtorch.zeros((3, 5, 2), dtype=slowtorch.float64)
    >>> b.shape
    (3, 5, 2)
    >>> b.shape = (3, 10)
    >>> b
    tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], 
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], 
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

- **Tensor.data.** Python buffer object pointing to the start of the tensor's
  data.

.. code-block:: python

    >>> a = slowtorch.ones((2, 7))
    >>> a.data
    tensor([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.], 
            [ 1.,  1.,  1.,  1.,  1.,  1.,  1.]])

- **Tensor.dtype.** Data-type of the tensor's elements.

.. code-block:: python

    >>> a = slowtorch.ones((2, 7))
    >>> a.dtype
    slowtorch.float64
    >>> b = slowtorch.zeros((3, 5, 2), dtype=slowtorch.int16)
    >>> b.dtype
    slowtorch.int16
    >>> type(b.dtype)
    <class 'slowtorch.dtype'>

- **Tensor.is_cuda.** Is `True` if the Tensor is stored on the GPU, `False`
  otherwise.

.. code-block:: python

    >>> a = slowtorch.tensor((1, 2, 3, 4, 5))
    >>> a.is_cuda
    False

- **Tensor.is_quantized.** Is `True` if the Tensor is quantized, `False`
  otherwise.

.. code-block:: python

    >>> a = slowtorch.tensor((1, 2, 3))
    >>> a.is_quantized
    False

- **Tensor.is_meta.** Is `True` if the Tensor is a meta tensor, `False`
  otherwise.

.. code-block:: python

    >>> a = slowtorch.zeros((1, 2, 3))
    >>> a.is_meta
    False

- **Tensor.T.** View of the transposed array.

.. code-block:: python

    >>> a = slowtorch.tensor([[1, 2], [3, 4]])
    >>> a
    tensor([[1, 2], 
            [3, 4]])
    >>> a.T
    tensor([[1, 3], 
            [2, 4]])

Tensor class methods
===============================================================================

- **Tensor.to().** Copies a tensor to a specified data type. Alias for
  `Tensor.type()`

.. code-block:: python

    >>> a = slowtorch.tensor((1, 2, 3, 4, 5))
    >>> a
    tensor([1, 2, 3, 4, 5])
    >>> a.to(slowtorch.float64)
    tensor([ 1.,  2.,  3.,  4.,  5.])
    >>> a.type(slowtorch.bool)
    tensor([True, True, True, True, True])

- **Tensor.size().** Number of elements in the tensor.

.. code-block:: python

    >>> a = slowtorch.tensor((1, 2, 3, 4, 5))
    >>> a.size()
    slowtorch.Size([5])
    >>> b = slowtorch.ones((2, 3))
    >>> b
    tensor([[ 1.,  1.,  1.], 
            [ 1.,  1.,  1.]])
    >>> b.size()
    slowtorch.Size([2, 3])

- **Tensor.stride().** Tuple of bytes to step in each dimension when traversing
  a tensor.

.. code-block:: python

    >>> a = slowtorch.ones((2, 3))
    >>> a.stride()
    (3, 1)

- **Tensor.nelement().** Return total number of elements in a tensor. Alias for
  `Tensor.numel()`.

.. code-block:: python

    >>> a = slowtorch.ones((2, 3))
    >>> a
    tensor([[ 1.,  1.,  1.], 
            [ 1.,  1.,  1.]])
    >>> a.nelement()
    6
    >>> b = slowtorch.tensor((1, 2, 3, 4, 5))
    >>> b.numel()
    5

- **Tensor.clone().** Return a deep copy of the tensor.

.. code-block:: python

    >>> a = slowtorch.tensor((1, 2, 3, 4, 5))
    >>> b = a.clone()
    >>> b
    tensor([1, 2, 3, 4, 5])

- **Tensor.fill_().** Fill the tensor with a scalar value.

.. code-block:: python

    >>> a = slowtorch.tensor([1, 2])
    >>> a.fill_(0)
    >>> a
    tensor([0, 0])

- **Tensor.flatten().** Return a copy of the tensor collapsed into one
  dimension.

.. code-block:: python

    >>> a = slowtorch.tensor([[1, 2], [3, 4]])
    >>> a.flatten()
    tensor([1, 2, 3, 4])

- **Tensor.item().** Copy an element of a tensor to a standard Python scalar
  and return it.

.. code-block:: python

    >>> a = slowtorch.tensor((2,))
    >>> a
    tensor([2])
    >>> a.item()
    2

- **Tensor.view().** Gives a new shape to a tensor without changing its
  data.

.. code-block:: python

    >>> a = slowtorch.arange(6).reshape((3, 2))
    >>> a
    tensor([[0, 1], 
            [2, 3], 
            [4, 5]])
    >>> a = slowtorch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> a.reshape((6,))
    tensor([1, 2, 3, 4, 5, 6])

- **Tensor.transpose().** Returns a tensor with dimensions transposed.

.. code-block:: python

    >>> a = slowtorch.tensor([[1, 2], [3, 4]])
    >>> a
    tensor([[1, 2], 
            [3, 4]])
    >>> a.transpose()
    tensor([[1, 3], 
            [2, 4]])
    >>> a = slowtorch.tensor([1, 2, 3, 4])
    >>> a.transpose()
    tensor([1, 2, 3, 4])
    >>> a = slowtorch.ones((1, 2, 3))
    >>> a.transpose((1, 0, 2)).shape
    (2, 1, 3)

Constants
===============================================================================

- **slowtorch.e.** Euler's constant.

.. code-block:: python

    >>> slowtorch.e
    2.718281828459045

- **slowtorch.inf.** IEEE 754 floating point representation of (positive)
  infinity.

.. code-block:: python

    >>> slowtorch.inf
    inf

- **slowtorch.nan.** IEEE 754 floating point representation of Not a Number
  (NaN).

.. code-block:: python

    >>> slowtorch.nan
    nan

- **slowtorch.newaxis.** A convenient alias for None, useful for indexing
  tensors.

.. code-block:: python

    >>> slowtorch.newaxis is None
    True

- **slowtorch.pi.** Pi...

.. code-block:: python

    >>> slowtorch.pi
    3.141592653589793

-------------------------------------------------------------------------------
Usage and Documentation
-------------------------------------------------------------------------------

The codebase is structured to be intuitive and mirrors the design principles
of PyTorch to a significant extent. Comprehensive docstrings are provided for
each module and function, ensuring clarity and ease of understanding. Users
are encouraged to delve into the code, experiment with it, and modify it to
suit their learning curve.

Since, the implementation doesn't rely on any external packages, it will work
with any CPython build v3.10 and higher. Technically, it should work on 3.9 and
below as well but due to some syntactical and type-aliasing changes, it will
not support it directly. For instance, the typing module was significantly
changed in 3.10, hence some features like ``types.GenericAlias`` and using
native types like ``tuple``, ``list``, etc. will not work. If you choose to
remove all the typing stuff, the code will work just fine, at least that's what
I hope.

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
.. _xsNumPy: https://github.com/xames3/slowtorch
