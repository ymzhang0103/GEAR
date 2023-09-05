import torch
import numpy as np

# Removed name argument from tensor creation functions
def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = torch.empty(shape, dtype=torch.float32)
    torch.nn.init.uniform_(initial, -scale, scale)
    return initial


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = torch.empty(shape, dtype=torch.float32)
    torch.nn.init.uniform_(initial, -init_range, init_range)
    return initial


def zeros(shape, name=None):
    """All zeros."""
    return torch.zeros(shape, dtype=torch.float32)


def ones(shape, name=None):
    """All ones."""
    return torch.ones(shape, dtype=torch.float32)