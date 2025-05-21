import pytest
import torch
import slowtorch

from torch.testing import assert_close

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
def test_shape(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch_tensor.shape
    against = slowtorch_tensor.shape
    assert_close(primary, against)


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
def test_size(tensors, dim):
    torch_tensor, slowtorch_tensor = tensors
    ndim = torch_tensor.ndim
    if dim in range(-ndim, ndim):
        primary = torch_tensor.size(dim)
        against = slowtorch_tensor.size(dim)
        assert_close(primary, against)


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
def test_nelement(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch_tensor.nelement()
    against = slowtorch_tensor.nelement()
    assert_close(primary, against)


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
def test_flatten(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch_tensor.flatten()
    against = slowtorch_tensor.flatten()
    assert_close(
        primary,
        to_torch_tensor(against),
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
@pytest.mark.parametrize("sorted", [True])
def test_unique(tensors, sorted):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch_tensor.unique(sorted)
    against = slowtorch_tensor.unique(sorted)
    assert_close(
        primary,
        to_torch_tensor(against),
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
def test_lt(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch_tensor.lt(torch_tensor + torch_tensor.exp())
    against = slowtorch_tensor.lt(slowtorch_tensor + slowtorch_tensor.exp())
    assert_close(
        primary,
        to_torch_tensor(against, bool),
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
def test_gt(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch_tensor.gt(torch_tensor + torch_tensor.exp())
    against = slowtorch_tensor.gt(slowtorch_tensor + slowtorch_tensor.exp())
    assert_close(
        primary,
        to_torch_tensor(against, bool),
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
def test_le(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch_tensor.le(torch_tensor + torch_tensor.exp())
    against = slowtorch_tensor.le(slowtorch_tensor + slowtorch_tensor.exp())
    assert_close(
        primary,
        to_torch_tensor(against, bool),
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
def test_ge(tensors):
    torch_tensor, slowtorch_tensor = tensors
    primary = torch_tensor.ge(torch_tensor + torch_tensor.exp())
    against = slowtorch_tensor.ge(slowtorch_tensor + slowtorch_tensor.exp())
    assert_close(
        primary,
        to_torch_tensor(against, bool),
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
def test_unsqueeze(tensors, dim):
    torch_tensor, slowtorch_tensor = tensors
    ndim = torch_tensor.ndim
    if dim in range(ndim):
        primary = torch_tensor.unsqueeze(dim)
        against = slowtorch_tensor.unsqueeze(dim)
        assert_close(
            primary,
            to_torch_tensor(against),
            rtol=1e-3,
            atol=1e-3,
        )
