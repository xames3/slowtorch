.. Author: Akshay Mestry <xa@mes3.dev>
.. Created on: Thursday, October 10 2024
.. Last updated on: Wednesday, August 13 2025

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
  `automatic gradient <https://pytorch.org/docs/stable/notes/autograd.html>`_
  (backpropagation) mechanics.
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

Install the **stable** version:

.. code-block:: bash

    pip install slowtorch

**OR**

Install the latest version of SlowTorch using `pip`_:

.. code-block:: bash

    pip install -U git+https://github.com/xames3/slowtorch.git#egg=slowtorch

-------------------------------------------------------------------------------
Features
-------------------------------------------------------------------------------

As of now, **SlowTorch** offers the following features:

SlowTorch native Tensor object (``slowtorch.Tensor``)
===============================================================================

- **slowtorch.Tensor.** The central data structure representing N-dimensional
  tensors with support for:

  - Arbitrary shapes and data types.
  - Broadcasting for compatible operations.
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

- **slowtorch.empty.** Create an uninitialised tensor of the given shape. In
  case of SlowTorch, it creates a tensor of zeros.

  .. code-block:: python
  
      >>> slowtorch.empty(2, 3)
      tensor([[ 0.,  0.,  0.], 
              [ 0.,  0.,  0.]])

- **slowtorch.zeros.** Create a tensor filled with zeros.

  .. code-block:: python
  
      >>> slowtorch.zeros(3, 2, 4)
      tensor([[[0., 0., 0., 0.], 
               [0., 0., 0., 0.]], 
              
              [[0., 0., 0., 0.], 
               [0., 0., 0., 0.]], 
              
              [[0., 0., 0., 0.], 
               [0., 0., 0., 0.]]])
      >>> slowtorch.zeros(2, 4, dtype=slowtorch.int64)
      tensor([[0, 0, 0, 0], 
              [0, 0, 0, 0]])

- **slowtorch.ones.** Create a tensor filled with ones.

  .. code-block:: python
  
      >>> slowtorch.ones(1, 3)
      tensor([[1., 1., 1.]])
      >>> slowtorch.ones(1, 3, 2, dtype=slowtorch.int16)
      tensor([[[1, 1], 
               [1, 1], 
               [1, 1]]], dtype=slowtorch.int16)

- **slowtorch.full.** Create a tensor filled with *fill_value*.

  .. code-block:: python
  
      >>> slowtorch.full(1, 5, 1, fill_value=3.141592)
      tensor([[[3.1416], 
               [3.1416], 
               [3.1416], 
               [3.1416], 
               [3.1416]]])
      >>> slowtorch.full(3, 4, fill_value=1.414)
      tensor([[1.414, 1.414, 1.414, 1.414], 
              [1.414, 1.414, 1.414, 1.414], 
              [1.414, 1.414, 1.414, 1.414]])

- **slowtorch.tril.** Create a lower triangular matrix (2-D tensor).

  .. code-block:: python
  
      >>> a = slowtorch.rand(4, 4)
      >>> slowtorch.tril(a)
      tensor([[0.9828,     0.,     0.,     0.], 
              [0.9489, 0.7202,     0.,     0.], 
              [0.2738, 0.7278,  0.505,     0.], 
              [0.9273, 0.9899, 0.5368, 0.3605]])
      >>> slowtorch.tril(a, diagonal=1)
      tensor([[0.9828, 0.5995,     0.,     0.], 
              [0.9489, 0.7202, 0.7863,     0.], 
              [0.2738, 0.7278,  0.505, 0.2608], 
              [0.9273, 0.9899, 0.5368, 0.3605]])
      >>> slowtorch.tril(a, diagonal=-1)
      tensor([[    0.,     0.,     0.,     0.], 
              [0.9489,     0.,     0.,     0.], 
              [0.2738, 0.7278,     0.,     0.], 
              [0.9273, 0.9899, 0.5368,     0.]])

- **slowtorch.triu.** Create a upper triangular matrix (2-D tensor).

  .. code-block:: python
  
      >>> a = slowtorch.rand(4, 4)
      >>> slowtorch.triu(a)
      tensor([[ 0.823, 0.5405, 0.9747, 0.3099], 
              [    0., 0.4245, 0.8782, 0.1842], 
              [    0.,     0., 0.9246, 0.9326], 
              [    0.,     0.,     0., 0.8109]])
      >>> slowtorch.triu(a, diagonal=1)
      tensor([[    0., 0.5405, 0.9747, 0.3099], 
              [    0.,     0., 0.8782, 0.1842], 
              [    0.,     0.,     0., 0.9326], 
              [    0.,     0.,     0.,     0.]])
      >>> slowtorch.triu(a, diagonal=-1)
      tensor([[ 0.823, 0.5405, 0.9747, 0.3099], 
              [0.2176, 0.4245, 0.8782, 0.1842], 
              [    0., 0.2348, 0.9246, 0.9326], 
              [    0.,     0., 0.5616, 0.8109]])

- **slowtorch.arange.** Generate evenly spaced values within a given range.

  .. code-block:: python
  
      >>> slowtorch.arange(5)
      tensor([0, 1, 2, 3, 4])
      >>> slowtorch.arange(1, 5)
      tensor([1, 2, 3, 4])
      >>> slowtorch.arange(1, 5, 0.5)
      tensor([ 1., 1.5,  2., 2.5,  3., 3.5,  4., 4.5])

- **slowtorch.linspace.** Generate evenly spaced values from start to end,
  inclusive.

  .. code-block:: python
  
      >>> slowtorch.linspace(3, 10, steps=5)
      tensor([  3., 4.75,  6.5, 8.25,  10.])
      >>> slowtorch.linspace(-10, 10, steps=7)
      tensor([   -10., -6.6667, -3.3333,      0.,  3.3333,  6.6667,     10.])
      >>> slowtorch.linspace(start=-10, end=10, steps=5)
      tensor([-10.,  -5.,   0.,   5.,  10.])
      >>> slowtorch.linspace(start=-10, end=10, steps=1)
      tensor([-10.])

- **slowtorch.cat.** Concatenates the given sequence of tensors in tensors in
  the given dimension.

  .. code-block:: python
  
      >>> a = slowtorch.rand(4)
      >>> a
      tensor([0.6386, 0.0518, 0.6576, 0.3298])
      >>> slowtorch.cat((a, a))
      tensor([0.6386, 0.0518, 0.6576, 0.3298, 0.6386, 0.0518, 0.6576, 0.3298])
      >>>
      >>> b = slowtorch.rand(2, 3)
      >>> b
      tensor([[0.7008, 0.1593, 0.6628], 
              [0.6897, 0.1713,  0.033]])
      >>> slowtorch.cat((b, b), dim=0)
      tensor([[0.7008, 0.1593, 0.6628], 
              [0.6897, 0.1713,  0.033], 
              [0.7008, 0.1593, 0.6628], 
              [0.6897, 0.1713,  0.033]])
      >>> slowtorch.cat((b, b), dim=1)
      tensor([[0.7008, 0.1593, 0.6628, 0.7008, 0.1593, 0.6628], 
              [0.6897, 0.1713,  0.033, 0.6897, 0.1713,  0.033]])

Autograd Mechanics
===============================================================================

- **Automatic Differentiation.** In lieu of mimicking PyTorch's functionality,
  pivotal feature of this project is a simple Pythonic version of automatic
  differentiation, akin to PyTorch's autograd. It allows for the computation
  of gradients automatically, which is essential for training neural networks.

  **Note.** To learn more about **Autograd Mechanics**, see `this <https://
  pytorch.org/docs/stable/notes/autograd.html>`_.

  .. code-block:: python
  
      >>> a = slowtorch.rand(2, 4, requires_grad=True)
      >>> b = slowtorch.rand(2, 4, requires_grad=True)
      >>> c = slowtorch.rand(2, 4, requires_grad=True)
      >>> a
      tensor([[0.6051, 0.7561, 0.3075, 0.5302], 
              [0.0418, 0.4999,  0.384, 0.8388]], requires_grad=True)
      >>> b
      tensor([[0.9355, 0.1261, 0.3961, 0.6106], 
              [0.3666, 0.0411, 0.1435, 0.2961]], requires_grad=True)
      >>> c
      tensor([[0.1592, 0.0854, 0.9256, 0.8058], 
              [0.7389, 0.6664, 0.2368, 0.1064]], requires_grad=True)
      >>> 
      >>> d = (a + b) * c
      >>> d
      tensor([[0.2453, 0.0753, 0.6513, 0.9193], 
              [0.3018, 0.3605, 0.1249, 0.1208]], grad_fn=<MulBackward0>)
      >>> d.backward()
      >>> 
      >>> a.grad
      tensor([[0.1592, 0.0854, 0.9256, 0.8058], 
              [0.7389, 0.6664, 0.2368, 0.1064]], grad_fn=<AddBackward0>)
      >>> b.grad
      tensor([[0.1592, 0.0854, 0.9256, 0.8058], 
              [0.7389, 0.6664, 0.2368, 0.1064]], grad_fn=<AddBackward0>)
      >>> c.grad
      tensor([[1.5406, 0.8822, 0.7036, 1.1408], 
              [0.4084,  0.541, 0.5275, 1.1349]], grad_fn=<AddBackward0>)
      >>> 
      >>> d.render(show_dtype=True)  # custom method for SlowTorch
      Tensor.5(shape=(2, 4), dtype=slowtorch.float32)
           MulBackward0
           ├──► Tensor.3(shape=(2, 4), dtype=slowtorch.float32)
           │    AddBackward0
           │    ├──► Tensor.1(shape=(2, 4), dtype=slowtorch.float32)
           │    └──► Tensor.2(shape=(2, 4), dtype=slowtorch.float32)
           └──► Tensor.4(shape=(2, 4), dtype=slowtorch.float32)

- **Specialised Backward Functions.** Like PyTorch, SlowTorch also implements
  some specialised `backward <https://pytorch.org/docs/stable/generated/torch.
  autograd.backward.html#torch.autograd.backward>`_ functions for
  backpropagation. These functions are mainly for representing the derivative
  or gradient calculations of the said functions.
  
  **Note.** SlowTorch supports a few backward functions when ``requires_grad``
  is ``True``:

  - **AddBackward0.** For addition operations.

  .. code-block:: python

      >>> a = slowtorch.rand(2, 3, requires_grad=True)
      >>> b = slowtorch.rand(2, 3, requires_grad=True)
      >>> a
      tensor([[0.5936, 0.9405, 0.8363], 
              [0.2631, 0.3354, 0.7065]], requires_grad=True)
      >>> b
      tensor([[0.5272, 0.2758, 0.5296], 
              [0.2496, 0.6263, 0.4925]], requires_grad=True)
      >>> a + b
      tensor([[1.1208, 1.2163, 1.3659], 
              [0.5127, 0.9617,  1.199]], grad_fn=<AddBackward0>)

  - **SubBackward0.** For subtraction operations.

  .. code-block:: python

      >>> a - b
      tensor([[ 0.0664,  0.6647,  0.3067], 
              [ 0.0135, -0.2909,   0.214]], grad_fn=<SubBackward0>)

  - **MulBackward0.** For multiplication operations

  .. code-block:: python

      >>> a * b
      tensor([[0.3129, 0.2594, 0.4429], 
              [0.0657, 0.2101,  0.348]], grad_fn=<MulBackward0>)

  - **DivBackward0.** For division operations.

  .. code-block:: python

      >>> a / b
      tensor([[1.1259, 3.4101, 1.5791], 
              [1.0541, 0.5355, 1.4345]], grad_fn=<DivBackward0>)
      >>> a // b
      tensor([[1., 3., 1.], 
              [1., 0., 1.]], grad_fn=<DivBackward0>)

  - **NegBackward0.** For negation operations.

  .. code-block:: python

      >>> a = slowtorch.tensor([2.0, 4.5, -1.7], requires_grad=True)
      >>> -a
      tensor([ -2., -4.5,  1.7], grad_fn=<NegBackward0>)

  - **DotBackward0.** For matrix multiplication operations.

  .. code-block:: python

      >>> a = slowtorch.rand(2, 3, requires_grad=True)
      >>> b = slowtorch.rand(3, 2, requires_grad=True)
      >>> a
      tensor([[0.6469, 0.9099, 0.6677], 
              [ 0.057, 0.6974, 0.2137]], requires_grad=True)
      >>> b
      tensor([[0.5674, 0.4916], 
              [0.5235, 0.3726], 
              [0.2661, 0.3235]], requires_grad=True)
      >>> a @ b
      tensor([[1.0211,  0.873], 
              [0.4543,  0.357]], grad_fn=<DotBackward0>)

  - **PowBackward0.** For exponentiation operations.

  .. code-block:: python

      >>> a = slowtorch.rand(2, 3, requires_grad=True)
      >>> a
      tensor([[0.6465, 0.4454, 0.9289], 
              [0.2837, 0.6275, 0.9291]], requires_grad=True)
      >>> a ** 2
      tensor([[ 0.418, 0.1984, 0.8629], 
              [0.0805, 0.3938, 0.8632]], grad_fn=<PowBackward0>)

  - **AbsBackward0.** For absolute value conversion/operations.

  .. code-block:: python

      >>> a = slowtorch.randn(3, 4, requires_grad=True)
      >>> a
      tensor([[ 0.2375,  0.1546, -0.7126, -0.2146], 
              [ 0.0222,  0.2271,  1.0456, -0.1353], 
              [ 0.3093, -0.2779, -1.0915,  0.7554]], requires_grad=True)
      >>> a.abs()
      tensor([[0.2375, 0.1546, 0.7126, 0.2146], 
              [0.0222, 0.2271, 1.0456, 0.1353], 
              [0.3093, 0.2779, 1.0915, 0.7554]], grad_fn=<AbsBackward0>)

  - **LogBackward0.** For logarithmic operations.

  .. code-block:: python

      >>> a = slowtorch.randn(2, 2, 3, requires_grad=True)
      >>> a
      tensor([[[ 1.1276, -0.6102,  0.1581], 
               [ 1.4331, -0.4444, -0.8745]], 
              
              [[ 0.7818,    1.29,  2.0592], 
               [-0.9721,  1.4584, -0.4874]]], requires_grad=True)
      >>> a.log()
      tensor([[[ 0.1201,    nan., -1.8445], 
               [ 0.3598,    nan.,    nan.]], 
              
              [[-0.2462,  0.2546,  0.7223], 
               [   nan.,  0.3773,    nan.]]], grad_fn=<LogBackward0>)

  - **CloneBackward0.** For clone/copy operation.

  .. code-block:: python

      >>> a = slowtorch.tensor([2.,  4.5, -1.7], requires_grad=True)
      >>> a.clone()
      tensor([2.00,  4.5, -1.7], grad_fn=<CloneBackward0>)

  - **ViewBackward0.** For creating a contiguous flattened tensor.

  .. code-block:: python

      >>> a = slowtorch.randn(3, 1, requires_grad=True)
      >>> a
      tensor([[ 0.3739], 
              [-1.9905], 
              [ 1.0801]], requires_grad=True)
      >>> a.ravel()
      tensor([ 0.3739, -1.9905,  1.0801], grad_fn=<ViewBackward0>)

  - **SumBackward0.** For calculating sum, across dimensions. Also supports
    ``keepdim`` option.

  .. code-block:: python

      >>> a = slowtorch.rand(2, 3, 1, requires_grad=True)
      >>> a
      tensor([[[0.8727], 
               [0.3508], 
               [0.8745]], 
              
              [[0.9042], 
               [0.0037], 
               [0.0996]]], requires_grad=True)
      >>> a.sum()
      tensor(3.1055, grad_fn=<SumBackward0>)
      >>> a.sum(dim=0)
      tensor([[1.7769], 
              [0.3545], 
              [0.9741]], grad_fn=<SumBackward0>)
      >>> a.sum(dim=1)
      tensor([[ 2.098], 
              [1.0075]], grad_fn=<SumBackward0>)
      >>> a.sum(dim=2)
      tensor([[0.8727, 0.3508, 0.8745], 
              [0.9042, 0.0037, 0.0996]], grad_fn=<SumBackward0>)

  - **MaxBackward0.** For calculating maximum, across dimensions. Also supports
    ``keepdim`` option.

  .. code-block:: python

      >>> a = slowtorch.rand(2, 2, 3, requires_grad=True)
      >>> a
      tensor([[[0.6439, 0.4503,  0.085], 
               [0.7339,  0.813, 0.6116]], 
              
              [[0.3679,  0.727, 0.6918], 
               [0.3954,  0.053, 0.9787]]], requires_grad=True)
      >>> a.max()
      tensor(0.9787, grad_fn=<MaxBackward0>)
      >>> a.max(dim=0)
      tensor([[0.6439,  0.727, 0.6918], 
              [0.7339,  0.813, 0.9787]], grad_fn=<MaxBackward0>)
      >>> a.max(dim=1)
      tensor([[0.7339,  0.813, 0.6116], 
              [0.3954,  0.727, 0.9787]], grad_fn=<MaxBackward0>)
      >>> a.max(dim=2)
      tensor([[0.6439,  0.813], 
              [ 0.727, 0.9787]], grad_fn=<MaxBackward0>)

  - **MinBackward0.** For calculating minimum, across dimensions. Also supports
    ``keepdim`` option.

  .. code-block:: python

      >>> a = slowtorch.randn(2, 3, requires_grad=True)
      >>> a
      tensor([[-0.9405, -0.1316,  0.8257], 
              [ 0.0997,  2.0668, -0.1255]], requires_grad=True)
      >>> a.min()
      tensor(-0.9405, grad_fn=<MinBackward0>)
      >>> a.min(dim=0)
      tensor([-0.9405, -0.1316, -0.1255], grad_fn=<MinBackward0>)
      >>> a.min(dim=1)
      tensor([-0.9405, -0.1255], grad_fn=<MinBackward0>)

  - **MeanBackward0.** For calculating mean, across dimensions. Also supports
    ``keepdim`` option.

  .. code-block:: python

      >>> a = slowtorch.randn(3, 4, 1, requires_grad=True)
      >>> a
      tensor([[[-0.2082], 
               [ -0.322], 
               [ 0.9676], 
               [  0.907]], 
              
              [[  0.442], 
               [ 1.1031], 
               [ 0.0456], 
               [ 0.5926]], 
              
              [[ 0.0943], 
               [ 0.0541], 
               [-0.6448], 
               [ 1.3448]]], requires_grad=True)
      >>> a.mean()
      tensor(0.3647, grad_fn=<MeanBackward0>)
      >>> a.mean(dim=0)
      tensor([[0.1094], 
              [0.2784], 
              [0.1228], 
              [0.9481]], grad_fn=<MeanBackward0>)
      >>> a.mean(dim=1)
      tensor([[0.3361], 
              [0.5458], 
              [0.2121]], grad_fn=<MeanBackward0>)
      >>> a.mean(dim=2)
      tensor([[-0.2082,  -0.322,  0.9676,   0.907], 
              [  0.442,  1.1031,  0.0456,  0.5926], 
              [ 0.0943,  0.0541, -0.6448,  1.3448]], grad_fn=<MeanBackward0>)

  - **StdBackward0.** For calculating standard deviation, across dimensions.
    Also supports ``keepdim`` option.

  .. code-block:: python

      >>> a = slowtorch.randn(4, 4, requires_grad=True)
      >>> a
      tensor([[ 0.2558,  0.8182, -0.9906, -1.7467], 
              [ 1.5136, -1.2438,  1.3334, -1.3326], 
              [-0.4245, -1.0178,  0.2653, -1.1246], 
              [-0.2272,  0.2684, -0.0806,  -1.179]], requires_grad=True)
      >>> a.std()
      tensor(0.9907, grad_fn=<StdBackward0>)
      >>> a.std(dim=0)
      tensor([ 0.871, 0.9965, 0.9603, 0.2815], grad_fn=<StdBackward0>)
      >>> a.std(dim=1)
      tensor([1.1655, 1.5677, 0.6395, 0.6189], grad_fn=<StdBackward0>)

  - **PermuteBackward0.** For transposing operations across two dimensions.

  .. code-block:: python

      >>> a = slowtorch.randn(1, 4, requires_grad=True)
      >>> a
      tensor([[ 0.9367, -0.1548,  1.2126,  0.2035]], requires_grad=True)
      >>> a.transpose(1, 0)
      tensor([[ 0.9367], 
              [-0.1548], 
              [ 1.2126], 
              [ 0.2035]], grad_fn=<PermuteBackward0>)

  - **ExpBackward0.** For exponentiation operation with respect to ``e``.

  .. code-block:: python

      >>> a = slowtorch.randn(3, 4, requires_grad=True)
      >>> a
      tensor([[ 0.6569,  0.3495, -0.4328,  1.1279], 
              [ 0.9556, -1.1973, -1.2926,   0.445], 
              [-1.7763,  -0.519, -0.2314,  1.3648]], requires_grad=True)
      >>> a.exp()
      tensor([[1.9288, 1.4184, 0.6487, 3.0892], 
              [2.6002,  0.302, 0.2746, 1.5605], 
              [0.1693, 0.5951, 0.7934, 3.9149]], grad_fn=<ExpBackward0>)

  - **SqrtBackward0.** For calculating square-roots.

  .. code-block:: python

      >>> a = slowtorch.rand(4, 1, requires_grad=True)
      >>> a
      tensor([[0.7565], 
              [0.8221], 
              [0.9183], 
              [0.7055]], requires_grad=True)
      >>> a.sqrt()
      tensor([[0.8698], 
              [0.9067], 
              [0.9583], 
              [0.8399]], grad_fn=<SqrtBackward0>)

  - **ReluBackward0.** When using ReLU non-linearity function.

  .. code-block:: python

      >>> a = slowtorch.randn(3, 4, requires_grad=True)
      >>> a
      tensor([[ 0.0896,  0.6086,  0.2634, -0.3649], 
              [ 0.3574,  -0.372,  1.8573,  0.7114], 
              [ 1.1223,  -0.026,  1.2171,  0.3683]], requires_grad=True)
      >>> a.relu()
      tensor([[0.0896, 0.6086, 0.2634,     0.], 
              [0.3574,     0., 1.8573, 0.7114], 
              [1.1223,     0., 1.2171, 0.3683]], grad_fn=<ReluBackward0>)

  - **EluBackward0.** When using ELU non-linearity function.

  .. code-block:: python

      >>> a = slowtorch.randn(2, 2, requires_grad=True)
      >>> a
      tensor([[ -0.362, -0.4587], 
              [ -0.502,  1.6582]], requires_grad=True)
      >>> a.elu()
      tensor([[-0.3037, -0.3679], 
              [-0.3947,  1.6582]], grad_fn=<EluBackward0>)
      >>> a.elu(alpha=0.7)
      tensor([[-0.2126, -0.2575], 
              [-0.2763,  1.6582]], grad_fn=<EluBackward0>)

  - **TanhBackward0.** When using Tanh non-linearity function.

  .. code-block:: python

      >>> a = slowtorch.randn(4, 3, requires_grad=True)
      >>> a
      tensor([[-0.1646,  2.0795, -1.3697], 
              [ 0.1221,  0.3469, -0.5246], 
              [ -0.836, -0.0565, -1.4846], 
              [ 0.4749, -0.0547,  0.2549]], requires_grad=True)
      >>> a.tanh()
      tensor([[-0.1631,  0.9692, -0.8786], 
              [ 0.1215,  0.3336, -0.4812], 
              [-0.6837, -0.0564, -0.9023], 
              [ 0.4421, -0.0546,  0.2495]], grad_fn=<TanhBackward0>)

  - **SigmoidBackward0.** When using Sigmoid non-linearity function.

  .. code-block:: python

      >>> a = slowtorch.randn(2, 4, requires_grad=True)
      >>> a
      tensor([[ 0.8443,  0.3218, -0.9884,  0.0682], 
              [-0.7883, -0.0273, -0.5722, -0.0114]], requires_grad=True)
      >>> a.sigmoid()
      tensor([[0.6994, 0.5798, 0.2712,  0.517], 
              [0.3125, 0.4932, 0.3607, 0.4972]], grad_fn=<SigmoidBackward0>)

  - **SoftmaxBackward0.** When using Softmax non-linearity function.

  .. code-block:: python

      >>> a = slowtorch.randn(4, 4, requires_grad=True)
      >>> a
      tensor([[-0.3575,  0.3504,  1.1453, -0.5454], 
              [ 0.2965, -1.0726, -0.9012,  0.9281], 
              [ -0.419,  0.3782, -1.5862, -1.0067], 
              [ 0.5482, -0.8483, -0.0119,  0.6324]], requires_grad=True)
      >>> a.softmax()
      tensor([[0.0385, 0.0781, 0.1729, 0.0319], 
              [ 0.074, 0.0188, 0.0223, 0.1391], 
              [0.0362, 0.0803, 0.0113, 0.0201], 
              [0.0952, 0.0235, 0.0544, 0.1035]], grad_fn=<SoftmaxBackward0>)
      >>> a.softmax(dim=0)
      tensor([[0.1578,  0.389, 0.6628, 0.1082], 
              [0.3035, 0.0937, 0.0856, 0.4722], 
              [0.1484,    0.4, 0.0431, 0.0682], 
              [0.3903, 0.1173, 0.2084, 0.3513]], grad_fn=<SoftmaxBackward0>)
      >>> a.softmax(dim=1)
      tensor([[0.1197,  0.243, 0.5381, 0.0992], 
              [ 0.291,  0.074, 0.0878, 0.5472], 
              [0.2447, 0.5432, 0.0762,  0.136], 
              [0.3441, 0.0852, 0.1965, 0.3743]], grad_fn=<SoftmaxBackward0>)

  - **LogSoftmaxBackward0.** When using LogSoftmax non-linearity function. It
    is similar to applying ``softmax`` function followed by log.

  .. code-block:: python

      >>> a = slowtorch.randn(3, 2, requires_grad=True)
      >>> a
      tensor([[-0.5559,  0.4392], 
              [   0.21,  1.6154], 
              [ 0.1543, -0.6819]], requires_grad=True)
      >>> a.log_softmax()
      tensor([[-2.8647, -1.8695], 
              [-2.0988, -0.6933], 
              [-2.1542, -2.9917]], grad_fn=<LogSoftmaxBackward0>)
      >>> a.log_softmax(dim=0)
      tensor([[-1.6461, -1.5191], 
              [  -0.88, -0.3428], 
              [-0.9357, -2.6409]], grad_fn=<LogSoftmaxBackward0>)
      >>> a.log_softmax(dim=1)
      tensor([[-1.3097, -0.3146], 
              [-1.6246, -0.2194], 
              [-0.3601,  -1.196]], grad_fn=<LogSoftmaxBackward0>)

  - **AddmmBackward0.** For calculating ``input @ weight.T + bias`` in Linear
    layer.

  .. code-block:: python

      >>> import slowtorch
      >>> import slowtorch.nn as nn
      >>> 
      >>> xs = slowtorch.tensor(
      ...     [
      ...         [1.5, 6.2, 2.6, 3.1, 5.3, 5.3, 7.9, 2.8],
      ...         [3.9, 2.8, 9.3, 6.4, 8.5, 6.9, 3.8, 3.1],
      ...         [3.4, 6.0, 4.4, 8.7, 9.7, 7.7, 1.6, 7.5],
      ...         [6.7, 8.8, 7.5, 1.8, 3.3, 8.4, 4.7, 5.1],
      ...         [6.8, 0.6, 4.8, 2.9, 6.8, 3.6, 3.5, 5.6],
      ...         [4.3, 4.2, 3.7, 7.0, 3.5, 8.5, 2.4, 2.9],
      ...     ],
      ...     requires_grad=True
      ... )
      >>> ys = slowtorch.tensor(
      ...     [
      ...         [-1.0],
      ...         [+1.0],
      ...         [-1.0],
      ...         [+1.0],
      ...         [-1.0],
      ...         [-1.0],
      ...     ]
      ... )
      >>> 
      >>> class NeuralNetwork(nn.Module):
      ...     def __init__(self, in_features, out_features):
      ...             super().__init__(in_features, out_features)
      ...             self.linear = nn.Linear(in_features, out_features)
      ...             self.out = nn.Linear(out_features, 1)
      ...     def forward(self, x):
      ...             return self.out(self.linear(x))
      ... 
      >>> model = NeuralNetwork(8, 16)
      >>> ypred = model(xs)
      >>> ypred
      tensor([[1.5218], 
              [1.5177], 
              [1.8904], 
              [3.6145], 
              [1.7698], 
              [2.0918]], grad_fn=<AddmmBackward0>)

  **Note.** The above demonstration is just for the forward pass through a
  linear layer without any activation. To get better results, you need to train
  the model with additional activation layer(s).

  - **MSELossBackward0.** When calculating Mean Squared Error loss.

  .. code-block:: python

      >>> criterion = nn.MSELoss()
      >>> loss = criterion(ypred, ys)
      >>> loss
      tensor(6.5081, grad_fn=<MSELossBackward0>)

  - **L1LossBackward0.** When calculating Mean Absolute Error (MAE) loss. This
    varies over different reduction strategies. It supports reducing over
    ``mean`` (default), ``sum``, and ``none``.

  .. code-block:: python

      >>> criterion = nn.L1Loss()
      >>> loss = criterion(ypred, ys)
      >>> loss
      tensor(2.401, grad_fn=<MeanBackward0>)
      >>> criterion = nn.L1Loss(reduction='sum')
      >>> loss = criterion(ypred, ys)
      >>> loss
      tensor(14.406, grad_fn=<SumBackward0>)
      >>> criterion = nn.L1Loss(reduction='none')
      >>> loss = criterion(ypred, ys)
      >>> loss
      tensor([[2.5218], 
              [0.5177], 
              [2.8904], 
              [2.6145], 
              [2.7698], 
              [3.0918]], grad_fn=<AbsBackward0>)

Tensor class reference
===============================================================================

- **Tensor.device.** Device where the tensor is.

  .. code-block:: python
  
      >>> a = slowtorch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
      >>> a.device
      device(type='cpu', index=0)

- **Tensor.grad.** This attribute is ``None`` by default and becomes a
  ``Tensor`` the first time a call to ``backward()`` computes gradients for
  ``self``.

- **Tensor.ndim.** Returns the number of dimensions of ``self`` tensor.
  Alias for ``Tensor.dim()``.

  .. code-block:: python
  
      >>> a = slowtorch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
      >>> a.ndim
      2
      >>> b = slowtorch.zeros(2, 3, 4)
      >>> b.dim()
      3

- **Tensor.nbytes.** Total bytes consumed by the elements of the tensor.

  .. code-block:: python
  
      >>> a = slowtorch.zeros(3, 2, dtype=slowtorch.float64)
      >>> a
      tensor([[ 0.,  0.], 
              [ 0.,  0.], 
              [ 0.,  0.]])
      >>> a.nbytes
      48
      >>> b = slowtorch.zeros(1, 3, dtype=slowtorch.int64)
      >>> b
      tensor([[0, 0, 0]])
      >>> b.nbytes
      24

- **Tensor.itemsize.** Length of one tensor element in bytes. Alias for
  ``Tensor.element_size()``.

  .. code-block:: python
  
      >>> a = slowtorch.full(2, 3, fill_value=2.71253)
      >>> a
      tensor([[2.71253, 2.71253, 2.71253], 
              [2.71253, 2.71253, 2.71253]])
      >>> a.itemsize
      8
      >>> b = slowtorch.tensor([1, 2, 3], dtype=slowtorch.int16)
      >>> b.element_size()
      2

- **Tensor.shape.** Size of the tensor as a tuple.

  .. code-block:: python
  
      >>> a = slowtorch.zeros(1, 3, dtype=slowtorch.int64)
      >>> a
      tensor([[0, 0, 0]])
      >>> a.shape
      slowtorch.Size([1, 3])
      >>> b = slowtorch.zeros(3, 5, 2, dtype=slowtorch.float64)
      >>> b.shape
      slowtorch.Size([3, 5, 2])
      >>> b.shape = (3, 10)
      >>> b
      tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], 
              [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], 
              [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

- **Tensor.data.** Python buffer object pointing to the start of the tensor's
  data.

  .. code-block:: python
  
      >>> a = slowtorch.ones(2, 7)
      >>> a.data
      tensor([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.], 
              [ 1.,  1.,  1.,  1.,  1.,  1.,  1.]])

- **Tensor.dtype.** Data-type of the tensor's elements.

  .. code-block:: python
  
      >>> a = slowtorch.ones(2, 7)
      >>> a.dtype
      slowtorch.float64
      >>> b = slowtorch.zeros(3, 5, 2, dtype=slowtorch.int16)
      >>> b.dtype
      slowtorch.int16
      >>> type(b.dtype)
      <class 'slowtorch.dtype'>

- **Tensor.is_cuda.** Is ``True`` if the Tensor is stored on the GPU, ``False``
  otherwise.

  .. code-block:: python
  
      >>> a = slowtorch.tensor((1, 2, 3, 4, 5))
      >>> a.is_cuda
      False

- **Tensor.is_quantized.** Is ``True`` if the Tensor is quantized, ``False``
  otherwise.

  .. code-block:: python
  
      >>> a = slowtorch.tensor((1, 2, 3))
      >>> a.is_quantized
      False

- **Tensor.is_meta.** Is ``True`` if the Tensor is a meta tensor, ``False``
  otherwise.

  .. code-block:: python
  
      >>> a = slowtorch.zeros(1, 2, 3)
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
  ``Tensor.type()``

  .. code-block:: python
  
      >>> a = slowtorch.tensor((1, 2, 3, 4, 5))
      >>> a
      tensor([1, 2, 3, 4, 5])
      >>> a.to(slowtorch.float64)
      tensor([ 1.,  2.,  3.,  4.,  5.])
      >>> a.type(slowtorch.bool)
      tensor([True, True, True, True, True])

- **Tensor.size().** Number of elements in the tensor. Alias for
  ``Tensor.shape``.

  .. code-block:: python
  
      >>> a = slowtorch.tensor((1, 2, 3, 4, 5))
      >>> a.size()
      slowtorch.Size([5])
      >>> b = slowtorch.ones(2, 3)
      >>> b
      tensor([[ 1.,  1.,  1.], 
              [ 1.,  1.,  1.]])
      >>> b.shape
      slowtorch.Size([2, 3])

- **Tensor.stride().** Tuple of bytes to step in each dimension when traversing
  a tensor.

  .. code-block:: python
  
      >>> a = slowtorch.ones(2, 3)
      >>> a.stride()
      (3, 1)

- **Tensor.nelement().** Return total number of elements in a tensor. Alias for
  ``Tensor.numel()``.

  .. code-block:: python
  
      >>> a = slowtorch.ones(2, 3)
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
  data. Alias for ``Tensor.reshape()``.

  .. code-block:: python
  
      >>> a = slowtorch.arange(6).view(3, 2)
      >>> a
      tensor([[0, 1], 
              [2, 3], 
              [4, 5]])
      >>> a = slowtorch.tensor([[1, 2, 3], [4, 5, 6]])
      >>> a.reshape(6)
      tensor([1, 2, 3, 4, 5, 6])

- **Tensor.transpose().** Returns a tensor with dimensions transposed. Alias
  for ``Tensor.swapaxes`` and ``Tensor.swapdims``.

  .. code-block:: python
  
      >>> a = slowtorch.tensor([[1, 2], [3, 4]])
      >>> a
      tensor([[1, 2], 
              [3, 4]])
      >>> a.transpose()
      tensor([[1, 3], 
              [2, 4]])
      >>> a = slowtorch.tensor([1, 2, 3, 4])
      >>> a.swapaxes()
      tensor([1, 2, 3, 4])
      >>> a = slowtorch.ones((1, 2, 3))
      >>> a.swapdims((1, 0, 2)).shape
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
SlowTorch In Action
-------------------------------------------------------------------------------

Below is a small demonstration of what SlowTorch can do, albeit... slowly.

.. code-block:: python

    >>> import slowtorch
    >>> import slowtorch.nn as snn
    >>> 
    >>> xs = slowtorch.tensor(
    ...     [
    ...         [1.5, 6.2, 2.6, 3.1, 5.3, 5.3, 7.9, 2.8],
    ...         [3.9, 2.8, 9.3, 6.4, 8.5, 6.9, 3.8, 3.1],
    ...         [3.4, 6.0, 4.4, 8.7, 9.7, 7.7, 1.6, 7.5],
    ...         [6.7, 8.8, 7.5, 1.8, 3.3, 8.4, 4.7, 5.1],
    ...         [6.8, 0.6, 4.8, 2.9, 6.8, 3.6, 3.5, 5.6],
    ...         [4.3, 4.2, 3.7, 7.0, 3.5, 8.5, 2.4, 2.9],
    ...     ],
    ...     requires_grad=True
    ... )
    >>> ys = slowtorch.tensor(
    ...     [
    ...         [0.558],
    ...         [0.175],
    ...         [0.152],
    ...         [0.485],
    ...         [0.232],
    ...         [0.0134],
    ...     ]
    ... )
    >>> 
    >>>
    >>> class NeuralNetwork(snn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.l1 = snn.Linear(8, 16)
    ...         self.l2 = snn.Linear(16, 32)
    ...         self.l3 = snn.Linear(32, 16)
    ...         self.l4 = snn.Linear(16, 8)
    ...         self.l5 = snn.Linear(8, 1)
    ...         self.tanh = snn.Tanh()
    ...     def forward(self, x):
    ...         x = self.tanh(self.l1(x))
    ...         x = self.tanh(self.l2(x))
    ...         x = self.tanh(self.l3(x))
    ...         x = self.tanh(self.l4(x))
    ...         x = self.tanh(self.l5(x))
    ...         return x
    ...         
    >>> 
    >>> model = NeuralNetwork()
    >>> print(f"Parameters: {sum(p.nelement() for p in model.parameters())}")
    Parameters: 1361
    >>> 
    >>> epochs = 500
    >>> criterion = snn.MSELoss()
    >>> optimiser = slowtorch.optim.SGD(model.parameters(), 0.1, momentum=0.1)
    >>> 
    >>> for epoch in range(epochs):
    ...     ypred = model(xs)
    ...     loss = criterion(ypred, ys)
    ...     optimiser.zero_grad()
    ...     loss.backward()
    ...     optimiser.step()
    ...     if epoch % 100 == 0:
    ...         print(f"New loss: {loss.item():.7f}")
    ... 
    New loss: 0.0403600
    New loss: 0.0098700
    New loss: 0.0002800
    New loss: 0.0000100
    New loss: 0.0000000
    >>> ypred
    tensor([[0.55807], 
            [0.17516], 
            [0.15148], 
            [ 0.4849], 
            [0.23193], 
            [0.01396]], grad_fn=<TanhBackward0>)

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

**Note.** This project also takes massive inspiration from excellent work done
by `Andrej Karpathy`_ in his `micrograd`_ project.

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
.. _Andrej Karpathy: https://github.com/karpathy
.. _micrograd: https://github.com/karpathy/micrograd
