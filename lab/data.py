"""Datasets, with fixed seeds and splits.

Toy data (generated, float32 tensors on the CPU, depending only on the arguments):
  ring(n, seed)         8 Gaussians on a circle of radius 2, standard deviation 0.02
  circle(n, seed)       the uniform distribution on the unit circle
  curve(n, p, seed)     a smooth closed curve in R^p (intrinsic dimension 1)
  gaussian(n, p, seed)  a standard Gaussian in R^p (intrinsic dimension p)

Images:
  mnist(train)          N x 784 floats in [0, 1] and labels, standard split
"""
import math

import torch

from . import DATA

RING_MODES, RING_RADIUS, RING_STD = 8, 2.0, 0.02
RING_CENTERS = torch.tensor([[RING_RADIUS * math.cos(2 * math.pi * k / RING_MODES),
                              RING_RADIUS * math.sin(2 * math.pi * k / RING_MODES)] for k in range(RING_MODES)])


def _rng(seed):
    return torch.Generator().manual_seed(seed)


def ring(n, seed=0):
    """Mixture of 8 Gaussians with equal weights, centres on a circle of radius 2, standard deviation 0.02."""
    g = _rng(seed)
    k = torch.randint(RING_MODES, (n,), generator=g)
    return RING_CENTERS[k] + RING_STD * torch.randn(n, 2, generator=g)


def circle(n, seed=0):
    """Uniform distribution on the unit circle (a 1-dimensional manifold in the plane)."""
    a = 2 * math.pi * torch.rand(n, generator=_rng(seed))
    return torch.stack([torch.cos(a), torch.sin(a)], 1)


def curve(n, p, seed=0):
    """A smooth closed curve in R^p: coordinates (cos 2 pi k t, sin 2 pi k t) / k for k = 1 .. p/2, t uniform.
    Intrinsic dimension 1. p must be even."""
    assert p % 2 == 0, "p must be even"
    g = _rng(seed)
    t = torch.rand(n, 1, generator=g)
    k = torch.arange(1, p // 2 + 1, dtype=torch.float32)
    return torch.cat([torch.cos(2 * math.pi * k * t) / k, torch.sin(2 * math.pi * k * t) / k], 1)


def gaussian(n, p, seed=0):
    """Standard Gaussian in R^p (intrinsic dimension p)."""
    return torch.randn(n, p, generator=_rng(seed))


def mnist(train=True):
    """MNIST, standard split: (N x 784 floats in [0, 1], labels). Downloads into data/ on first use."""
    import torchvision
    ds = torchvision.datasets.MNIST(DATA, train=train, download=True)
    return ds.data.flatten(1).float().div(255), ds.targets
