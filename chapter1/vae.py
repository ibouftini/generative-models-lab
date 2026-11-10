"""The 8-dimensional MNIST VAE whose decoder draws the noisy digit of the manifold schematic (chapter1/manifold.py).

Gaussian decoder with variance s^2 = 0.03, closed-form KL, the architecture of chapter1/autoencoder.py
(784 -> 256 -> 64 -> 8, the encoder outputting a mean and a log-variance).

usage: python -m chapter1.vae train    # a few minutes on a GPU, saves checkpoints/mnist_vae8.pt
"""
import argparse

import torch
import torch.nn as nn

from lab import CHECKPOINTS as OUT
from lab.data import mnist

S2, EPOCHS = 0.03, 30
MODELS = {"mnist_vae8.pt": (True, 8, (256, 64))}
device = "cuda" if torch.cuda.is_available() else "cpu"


def mlp(sizes, last):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers[-1] = last
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self, variational, d, hidden, s2=S2):
        super().__init__()
        self.variational, self.s2 = variational, s2
        self.enc = mlp([784, *hidden, 2 * d if variational else d], nn.Identity())
        self.dec = mlp([d, *hidden[::-1], 784], nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        if not self.variational:
            return h, None
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def loss(self, x):
        mu, logvar = self.encode(x)
        if not self.variational:
            return ((self.dec(mu) - x) ** 2).sum(1).mean()
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)       # reparameterization
        rec = ((self.dec(z) - x) ** 2).sum(1) / (2 * self.s2)
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)
        return (rec + kl).mean()


def train():
    OUT.mkdir(parents=True, exist_ok=True)
    x, _ = mnist(True)
    x = x.to(device)
    for name in MODELS:
        torch.manual_seed(0)
        model = Model(*MODELS[name]).to(device)
        opt = torch.optim.Adam(model.parameters(), 1e-3)
        for epoch in range(EPOCHS):
            perm = torch.randperm(len(x), device=device)
            total = 0.0
            for idx in perm.split(128):
                loss = model.loss(x[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if epoch % 5 == 4:
                print(f"{name}  epoch {epoch + 1:2d}  loss {total / len(x):8.2f}", flush=True)
        torch.save(model.state_dict(), OUT / name)


def load(name):
    m = Model(*MODELS[name])
    m.load_state_dict(torch.load(OUT / name, map_location="cpu"))
    return m.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.parse_args()
    train()
