"""Plain autoencoders and VAEs on MNIST (fully connected, codes of size 2 and 8).

A 2D code can be drawn as it is, without any projection. Autoencoder and VAE share the architecture; the VAE encoder
outputs a mean and a log-variance. The VAE loss is the one derived in the chapter: Gaussian decoder with variance s^2,
closed-form KL.

usage: python -m chapter3.vae train     # about 3 min on a GPU: checkpoints/mnist_ae2.pt, mnist_vae2.pt, ... (skips existing)
       python -m chapter3.vae figures   # figures/chapter3/latent-codes.svg and latent-map.png
"""
import argparse

import torch
import torch.nn as nn

from chapter3.autoencoder import OUT, mnist
from lab import FIGURES

FIG = FIGURES / "chapter3"
D, S2, EPOCHS = 2, 0.03, 30
# 2D models: 784 -> 512 -> 256 -> 2. 8D VAE: the architecture of the autoencoder of chapter3/autoencoder.py.
# The "_kl" models use a decoder variance 100 times larger (equivalently beta = 100): the KL term dominates.
MODELS = {"mnist_ae2.pt": (False, 2, (512, 256)), "mnist_vae2.pt": (True, 2, (512, 256)),
          "mnist_vae8.pt": (True, 8, (256, 64)),
          "mnist_vae2_kl.pt": (True, 2, (512, 256), 100 * S2), "mnist_vae8_kl.pt": (True, 8, (256, 64), 100 * S2)}
device = "cuda" if torch.cuda.is_available() else "cpu"


def mlp(sizes, last):
    layers = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers[-1] = last
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self, variational, d=D, hidden=(512, 256), s2=S2):
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
    # 2D models for the latent plots, 8D models for the comparison of random samples
    for name in MODELS:
        if (OUT / name).exists():
            continue
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
    ap.add_argument("cmd", choices=["train", "figures"])
    a = ap.parse_args()
    if a.cmd == "train":
        train()
    else:
        from chapter3.latent_figures import figures
        figures()
