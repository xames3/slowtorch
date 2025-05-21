import torch


def to_torch_tensor(tensor, dtype=float):
    """Convert a SlowTorch tensor to a Pytorch.Tensor for comparison."""
    return torch.tensor(
        tensor.flat(), dtype=dtype, requires_grad=tensor.requires_grad
    ).reshape(tensor.shape)
