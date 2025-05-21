import pytest
import torch
import slowtorch


@pytest.fixture(autouse=True)
def set_seed():
    seed = 99699523170
    torch.manual_seed(seed)
    slowtorch.manual_seed(seed)


def randn(shape):
    return torch.randn(*shape, dtype=float, requires_grad=True)


@pytest.fixture
def tensors(request):
    shape = request.param
    torch_tensor = randn(shape)
    slowtorch_tensor = slowtorch.tensor(
        torch_tensor.detach().numpy(), requires_grad=torch_tensor.requires_grad
    )
    return torch_tensor, slowtorch_tensor
