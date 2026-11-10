"""A small WGAN-GP on MNIST, the generator of the opening figure of Chapter 1.

usage: python -m chapter1.generator train     # checkpoints/mnist_wgan_gp.pt, about one hour on an RTX 3050
       python -m chapter1.generator figure    # figures/chapter1/mnist-real.png and mnist-generated.png

Training follows Gulrajani et al. (2017): Adam(1e-4, betas=(0, 0.9)), 5 critic steps per generator step, gradient penalty
weight 10. Networks are DCGAN-style for 28 x 28 grey images. The published generator ran for 60 minutes, which was
70,220 generator steps, so that is the default number of steps here.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from lab import CHECKPOINTS, FIGURES
from lab.data import mnist

CKPT = CHECKPOINTS / "mnist_wgan_gp.pt"
PREVIEWS = CHECKPOINTS / "mnist_wgan_gp_samples"  # sample grids saved during training
OUT_DIR = FIGURES / "chapter1"
Z_DIM, DIM, BATCH = 128, 64, 64
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_mnist(train=True):
    """MNIST images in [0, 1], N x 1 x 28 x 28 (the training loop rescales them to [-1, 1])."""
    return mnist(train)[0].view(-1, 1, 28, 28)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(Z_DIM, 7 * 7 * 4 * DIM), nn.BatchNorm1d(7 * 7 * 4 * DIM), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 4, 2, 1), nn.BatchNorm2d(2 * DIM), nn.ReLU(True),  # 14 x 14
            nn.ConvTranspose2d(2 * DIM, 1, 4, 2, 1), nn.Tanh(),                                    # 28 x 28
        )

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 4 * DIM, 7, 7))


class Critic(nn.Module):
    # No batch norm: the gradient penalty is computed per sample.
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, DIM, 4, 2, 1), nn.LeakyReLU(0.2, True),            # 14 x 14
            nn.Conv2d(DIM, 2 * DIM, 4, 2, 1), nn.LeakyReLU(0.2, True),      # 7 x 7
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, 1), nn.LeakyReLU(0.2, True),  # 4 x 4
        )
        self.fc = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


def gradient_penalty(critic, real, fake):
    eps = torch.rand(real.size(0), 1, 1, 1, device=device)
    x = (eps * real + (1 - eps) * fake).requires_grad_(True)
    grad, = torch.autograd.grad(critic(x).sum(), x, create_graph=True)
    return ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()


def to_grid(x, rows=8, scale=3):
    """x in [-1, 1], 64 x 1 x 28 x 28 -> PIL image, 2 px transparent gutters (fit both site themes),
    nearest-neighbour upscale."""
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()[:, 0]
    gap, s = 2, 28
    n = rows * s + (rows - 1) * gap
    grid = np.zeros((n, n, 4), np.uint8)
    for k, img in enumerate(x[: rows * rows]):
        i, j = divmod(k, rows)
        grid[i * (s + gap): i * (s + gap) + s, j * (s + gap): j * (s + gap) + s] = np.dstack([img, img, img, np.full((s, s), 255, np.uint8)])
    im = Image.fromarray(grid, "RGBA")
    return im.resize((n * scale, n * scale), Image.NEAREST)


def train(iters):
    torch.manual_seed(0)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    data = load_mnist().to(device) * 2 - 1
    G, D = Generator().to(device), Critic().to(device)
    opt_g = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), 1e-4, betas=(0.0, 0.9))
    fixed_z = torch.randn(64, Z_DIM, device=device)
    start, it = time.time(), 0
    while True:
        for _ in range(5):
            real = data[torch.randint(len(data), (BATCH,), device=device)]
            with torch.no_grad():
                fake = G(torch.randn(BATCH, Z_DIM, device=device))
            loss_d = D(fake).mean() - D(real).mean() + 10 * gradient_penalty(D, real, fake)
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
        loss_g = -D(G(torch.randn(BATCH, Z_DIM, device=device))).mean()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        it += 1
        last = it >= iters
        if it % 500 == 0 or last:
            G.eval()
            with torch.no_grad():
                to_grid(G(fixed_z)).save(PREVIEWS / f"samples_{it:05d}.png")
            G.train()
            torch.save({"G": G.state_dict(), "it": it}, CKPT)
            print(f"it {it:6d}  critic {loss_d.item():8.3f}  gen {loss_g.item():8.3f}  "
                  f"{(time.time() - start) / 60:5.1f} min", flush=True)
        if last:
            break


def load_generator():
    ckpt = torch.load(CKPT, map_location=device)
    G = Generator().to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G, ckpt


def figure():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(1)
    data = load_mnist()
    real = data[torch.randperm(len(data), generator=g)[:64]] * 2 - 1
    to_grid(real).save(OUT_DIR / "mnist-real.png", optimize=True)
    G, _ = load_generator()
    with torch.no_grad():
        fake = G(torch.randn(64, Z_DIM, generator=g).to(device))
    to_grid(fake).save(OUT_DIR / "mnist-generated.png", optimize=True)
    print("wrote", OUT_DIR / "mnist-real.png", "and mnist-generated.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "figure"])
    ap.add_argument("--iters", type=int, default=70220)
    a = ap.parse_args()
    {"train": lambda: train(a.iters), "figure": figure}[a.cmd]()
