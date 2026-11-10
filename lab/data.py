"""Datasets, with fixed seeds and splits.

Toy data (generated, float32 tensors on the CPU, depending only on the arguments):
  circle(n, seed)       the uniform distribution on the unit circle

Images:
  mnist(train)          N x 784 floats in [0, 1] and labels, standard split
"""
import math

import torch

from . import DATA

def _rng(seed):
    return torch.Generator().manual_seed(seed)


def circle(n, seed=0):
    """Uniform distribution on the unit circle (a 1-dimensional manifold in the plane)."""
    a = 2 * math.pi * torch.rand(n, generator=_rng(seed))
    return torch.stack([torch.cos(a), torch.sin(a)], 1)


def mnist(train=True):
    """MNIST, standard split: (N x 784 floats in [0, 1], labels). Downloads into data/ on first use."""
    import torchvision
    ds = torchvision.datasets.MNIST(DATA, train=train, download=True)
    return ds.data.flatten(1).float().div(255), ds.targets
