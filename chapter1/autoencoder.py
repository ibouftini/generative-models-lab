"""The MNIST autoencoder whose decoder draws the digits of the manifold schematic (chapter1/manifold.py).

usage: python -m chapter1.autoencoder train    # about 1 min on a GPU, saves checkpoints/mnist_ae.pt
"""
import argparse

import torch
import torch.nn as nn

from lab import CHECKPOINTS as OUT
from lab.data import mnist

SIZES = [784, 256, 64, 8]
device = "cuda" if torch.cuda.is_available() else "cpu"


def mlp(sizes, last):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers[-1] = last
    return nn.Sequential(*layers)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = mlp(SIZES, nn.Identity())        # encoder: 784 -> 256 -> 64 -> 8
        self.g = mlp(SIZES[::-1], nn.Sigmoid())   # decoder: 8 -> 64 -> 256 -> 784, pixels in [0, 1]

    def forward(self, x):
        return self.g(self.f(x))


def train():
    torch.manual_seed(0)
    OUT.mkdir(parents=True, exist_ok=True)
    x, _ = mnist(True)
    x = x.to(device)
    model = AE().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for epoch in range(20):
        perm = torch.randperm(len(x), device=device)
        total = 0.0
        for idx in perm.split(128):
            loss = ((model(x[idx]) - x[idx]) ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch:2d}  reconstruction error {total / len(x):6.2f}", flush=True)
    torch.save(model.state_dict(), OUT / "mnist_ae.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.parse_args()
    train()
