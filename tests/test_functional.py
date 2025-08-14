import pytest
import torch
from torch.testing import assert_close

import slowtorch

from . import to_torch_tensor


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_add(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.add(torch_tensor, torch_tensor)
    against = slowtorch.add(slowtorch_tensor, slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_sub(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.sub(torch_tensor, torch_tensor)
    against = slowtorch.sub(slowtorch_tensor, slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_mul(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.mul(torch_tensor, torch_tensor)
    against = slowtorch.mul(slowtorch_tensor, slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_div(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.div(torch_tensor, torch_tensor)
    against = slowtorch.div(slowtorch_tensor, slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2, 3),
        (3, 3),
        (4, 2, 2),
        (1, 2, 3, 3),
    ],
    indirect=True,
)
def test_matmul(tensors):
    torch_tensor, slowtorch_tensor = tensors
    try:
        primary = torch.matmul(torch_tensor, torch_tensor)
        against = slowtorch.matmul(slowtorch_tensor, slowtorch_tensor)
        assert_close(
            primary.detach(),
            to_torch_tensor(against),
            rtol=1e-3,
            atol=1e-3,
        )
        if torch_tensor.grad is not None:
            torch_tensor.grad.zero_()
        if slowtorch_tensor.grad is not None:
            slowtorch_tensor.grad = 0.0
        primary.sum().backward()
        against.sum().backward()
        assert_close(
            torch_tensor.grad,
            to_torch_tensor(slowtorch_tensor.grad),
            rtol=1e-3,
            atol=1e-3,
        )
    except RuntimeError:
        with pytest.raises((ValueError, RuntimeError)):
            slowtorch.matmul(slowtorch_tensor, slowtorch_tensor)


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_remainder(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.remainder(torch_tensor, torch_tensor)
    against = slowtorch.remainder(slowtorch_tensor, slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_log(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.log(torch_tensor)
    against = slowtorch.log(slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_clone(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.clone(torch_tensor)
    against = slowtorch.clone(slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_neg(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.neg(torch_tensor)
    against = slowtorch.neg(slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2, 3),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dim0", [0, 1])
@pytest.mark.parametrize("dim1", [0, 1])
def test_transpose(tensors, dim0, dim1):
    torch_tensor, slowtorch_tensor = tensors
    ndim = torch_tensor.ndim
    if sorted((dim0, dim1)) == list(range(ndim)):
        primary = torch.transpose(torch_tensor, dim0=dim0, dim1=dim1)
        against = slowtorch.transpose(slowtorch_tensor, dim0=dim0, dim1=dim1)
        assert_close(
            primary.detach(),
            to_torch_tensor(against),
            rtol=1e-3,
            atol=1e-3,
        )
        if torch_tensor.grad is not None:
            torch_tensor.grad.zero_()
        if slowtorch_tensor.grad is not None:
            slowtorch_tensor.grad = 0.0
        primary.sum().backward()
        against.sum().backward()
        assert_close(
            torch_tensor.grad,
            to_torch_tensor(slowtorch_tensor.grad),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "tensors",
    [
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dim", range(-4, 4))
@pytest.mark.parametrize("keepdim", [True, False])
def test_sum(tensors, dim, keepdim):
    torch_tensor, slowtorch_tensor = tensors
    ndim = torch_tensor.ndim
    if dim in range(-ndim, ndim):
        primary = torch.sum(torch_tensor, dim=dim, keepdim=keepdim)
        against = slowtorch.sum(slowtorch_tensor, dim=dim, keepdim=keepdim)
        assert_close(
            primary.detach(),
            to_torch_tensor(against),
            rtol=1e-3,
            atol=1e-3,
        )
        if torch_tensor.grad is not None:
            torch_tensor.grad.zero_()
        if slowtorch_tensor.grad is not None:
            slowtorch_tensor.grad = 0.0
        primary.sum().backward()
        against.sum().backward()
        assert_close(
            torch_tensor.grad,
            to_torch_tensor(slowtorch_tensor.grad),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "tensors",
    [
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dim", range(-4, 4))
@pytest.mark.parametrize("keepdim", [True, False])
def test_max(tensors, dim, keepdim):
    torch_tensor, slowtorch_tensor = tensors
    ndim = torch_tensor.ndim
    if dim in range(-ndim, ndim):
        primary = torch.max(torch_tensor, dim=dim, keepdim=keepdim)
        against = slowtorch.max(slowtorch_tensor, dim=dim, keepdim=keepdim)
        assert_close(
            primary.values.detach(),
            to_torch_tensor(against),
            rtol=1e-3,
            atol=1e-3,
        )
        if torch_tensor.grad is not None:
            torch_tensor.grad.zero_()
        if slowtorch_tensor.grad is not None:
            slowtorch_tensor.grad = 0.0
        primary.values.sum().backward()
        against.sum().backward()
        assert_close(
            torch_tensor.grad,
            to_torch_tensor(slowtorch_tensor.grad),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "tensors",
    [
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dim", range(-4, 4))
@pytest.mark.parametrize("keepdim", [True, False])
def test_min(tensors, dim, keepdim):
    torch_tensor, slowtorch_tensor = tensors
    ndim = torch_tensor.ndim
    if dim in range(-ndim, ndim):
        primary = torch.min(torch_tensor, dim=dim, keepdim=keepdim)
        against = slowtorch.min(slowtorch_tensor, dim=dim, keepdim=keepdim)
        assert_close(
            primary.values.detach(),
            to_torch_tensor(against),
            rtol=1e-3,
            atol=1e-3,
        )
        if torch_tensor.grad is not None:
            torch_tensor.grad.zero_()
        if slowtorch_tensor.grad is not None:
            slowtorch_tensor.grad = 0.0
        primary.values.sum().backward()
        against.sum().backward()
        assert_close(
            torch_tensor.grad,
            to_torch_tensor(slowtorch_tensor.grad),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "tensors",
    [
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dim", range(-4, 4))
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean(tensors, dim, keepdim):
    torch_tensor, slowtorch_tensor = tensors
    ndim = torch_tensor.ndim
    if dim in range(-ndim, ndim):
        primary = torch.mean(torch_tensor, dim=dim, keepdim=keepdim)
        against = slowtorch.mean(slowtorch_tensor, dim=dim, keepdim=keepdim)
        assert_close(
            primary.detach(),
            to_torch_tensor(against),
            rtol=1e-3,
            atol=1e-3,
        )
        if torch_tensor.grad is not None:
            torch_tensor.grad.zero_()
        if slowtorch_tensor.grad is not None:
            slowtorch_tensor.grad = 0.0
        primary.sum().backward()
        against.sum().backward()
        assert_close(
            torch_tensor.grad,
            to_torch_tensor(slowtorch_tensor.grad),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "tensors",
    [
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dim", range(-4, 4))
@pytest.mark.parametrize("keepdim", [True, False])
def test_std(tensors, dim, keepdim):
    torch_tensor, slowtorch_tensor = tensors
    ndim = torch_tensor.ndim
    if dim in range(-ndim, ndim) and dim not in [-4, 0]:
        primary = torch.std(torch_tensor, dim=dim, keepdim=keepdim)
        against = slowtorch.std(slowtorch_tensor, dim=dim, keepdim=keepdim)
        assert_close(
            primary.detach(),
            to_torch_tensor(against),
            rtol=1e-3,
            atol=1e-3,
        )
        if torch_tensor.grad is not None:
            torch_tensor.grad.zero_()
        if slowtorch_tensor.grad is not None:
            slowtorch_tensor.grad = 0.0
        primary.sum().backward()
        against.sum().backward()
        assert_close(
            torch_tensor.grad,
            to_torch_tensor(slowtorch_tensor.grad),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_exp(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.exp(torch_tensor)
    against = slowtorch.exp(slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_sqrt(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.exp(torch_tensor)
    against = slowtorch.exp(slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
        equal_nan=True,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_relu(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.relu(torch_tensor)
    against = slowtorch.relu(slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_elu(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.nn.functional.elu(torch_tensor)
    against = slowtorch.elu(slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_tanh(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.tanh(torch_tensor)
    against = slowtorch.tanh(slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
def test_sigmoid(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch.sigmoid(torch_tensor)
    against = slowtorch.sigmoid(slowtorch_tensor)
    assert_close(
        primary.detach(),
        to_torch_tensor(against),
        rtol=1e-3,
        atol=1e-3,
    )
    if torch_tensor.grad is not None:
        torch_tensor.grad.zero_()
    if slowtorch_tensor.grad is not None:
        slowtorch_tensor.grad = 0.0
    primary.sum().backward()
    against.sum().backward()
    assert_close(
        torch_tensor.grad,
        to_torch_tensor(slowtorch_tensor.grad),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dim", range(-4, 4))
def test_softmax(tensors, dim):
    torch_tensor, slowtorch_tensor = tensors
    ndim = torch_tensor.ndim
    if dim in range(-ndim, ndim):
        primary = torch.softmax(torch_tensor, dim=dim)
        against = slowtorch.softmax(slowtorch_tensor, dim=dim)
        assert_close(
            primary.detach(),
            to_torch_tensor(against),
            rtol=1e-3,
            atol=1e-3,
        )
        if torch_tensor.grad is not None:
            torch_tensor.grad.zero_()
        if slowtorch_tensor.grad is not None:
            slowtorch_tensor.grad = 0.0
        primary.sum().backward()
        against.sum().backward()
        assert_close(
            torch_tensor.grad,
            to_torch_tensor(slowtorch_tensor.grad),
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize(
    "tensors",
    [
        (2,),
        (2, 3),
        (4, 2, 3),
        (1, 2, 3, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dim", range(-4, 4))
def test_log_softmax(tensors, dim):
    torch_tensor, slowtorch_tensor = tensors
    ndim = torch_tensor.ndim
    if dim in range(-ndim, ndim):
        primary = torch.log_softmax(torch_tensor, dim=dim)
        against = slowtorch.log_softmax(slowtorch_tensor, dim=dim)
        assert_close(
            primary.detach(),
            to_torch_tensor(against),
            rtol=1e-3,
            atol=1e-3,
        )
        if torch_tensor.grad is not None:
            torch_tensor.grad.zero_()
        if slowtorch_tensor.grad is not None:
            slowtorch_tensor.grad = 0.0
        primary.sum().backward()
        against.sum().backward()
        assert_close(
            torch_tensor.grad,
            to_torch_tensor(slowtorch_tensor.grad),
            rtol=1e-3,
            atol=1e-3,
        )
