"""The MNIST autoencoder of Chapter 3 and the schematic drawn from it.

usage: python -m chapter3.autoencoder train    # about 1 min on a GPU, saves checkpoints/mnist_ae.pt
       python -m chapter3.autoencoder figure   # writes figures/chapter3/autoencoder.svg

The schematic shows a real test digit, the layer sizes of the network, the real 8-number code and the real
reconstruction.
"""
import argparse

import torch
import torch.nn as nn

from lab import CHECKPOINTS as OUT
from lab import FIGURES
from lab.data import mnist
from lab.schematic import tex

SVG = FIGURES / "chapter3" / "autoencoder.svg"
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


def figure(index):
    model = AE()
    model.load_state_dict(torch.load(OUT / "mnist_ae.pt", map_location="cpu"))
    model.eval()
    xs, ys = mnist(False)
    x = xs[index]
    with torch.no_grad():
        z = model.f(x)
        xh = model.g(z)
    err = ((xh - x) ** 2).sum().item()

    W, H = 720, 310
    s = 4                                   # pixel size of the 28 x 28 grids
    G = 28 * s
    cy = 138                                # vertical centre of the network row
    in_x, out_x = 20, W - 20 - G
    svg = [f'<svg class="schematic" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']

    def grid(img, x0):
        y0 = cy - G / 2
        svg.append(f'<rect class="sch-frame" x="{x0 - 4}" y="{y0 - 4}" width="{G + 8}" height="{G + 8}" rx="6"/>')
        for k, v in enumerate(img.tolist()):
            if v > 0.04:
                i, j = divmod(k, 28)
                svg.append(f'<rect class="sch-ink" x="{x0 + j * s}" y="{y0 + i * s}" width="{s}" height="{s}" '
                           f'fill-opacity="{min(v, 1):.2f}"/>')

    grid(x, in_x)
    grid(xh, out_x)

    # Layers: bars whose height grows with the layer size, code cells in the middle
    def bar_h(n):
        return 24 + 110 * (n / 784) ** 0.5

    left, right = in_x + G + 26, out_x - 26
    code_w, code_cell = 16, 9
    code_h = 8 * code_cell
    mid = W / 2
    gap = (mid - code_w / 2 - left) / 3
    enc = [(left + k * gap, SIZES[k]) for k in range(3)]
    dec = [(W - x0 - 12, n) for x0, n in enc[::-1]]
    bars = enc + [(mid - code_w / 2, 8)] + dec
    shapes = []
    for x0, n in bars:
        h = code_h if n == 8 else bar_h(n)
        shapes.append((x0, cy - h / 2, h))
    # Light bands joining consecutive layers (the "funnel")
    for k in range(len(shapes) - 1):
        (xa, ya, ha), (xb, yb, hb) = shapes[k], shapes[k + 1]
        wa = code_w if k == 3 else 12
        svg.append(f'<path class="sch-band" d="M{xa + wa},{ya} L{xb},{yb} L{xb},{yb + hb} L{xa + wa},{ya + ha} Z"/>')
    for k, (x0, y0, h) in enumerate(shapes):
        if k == 3:
            zmax = z.abs().max().item()
            for c, v in enumerate(z.tolist()):
                svg.append(f'<rect class="sch-ink" x="{x0}" y="{y0 + c * code_cell}" width="{code_w}" '
                           f'height="{code_cell - 1}" fill-opacity="{0.15 + 0.85 * abs(v) / zmax:.2f}"/>')
            svg.append(f'<rect class="sch-line" x="{x0 - 2}" y="{y0 - 2}" width="{code_w + 4}" height="{h + 3}" rx="3"/>')
            svg.append(tex(x0 + code_w / 2, y0 + h + 16, "8", "tiny", "middle"))
        else:
            svg.append(f'<rect class="sch-layer" x="{x0}" y="{y0}" width="12" height="{h}" rx="3"/>')
            svg.append(tex(x0 + 6, y0 + h + 14, str(bars[k][1]), "tiny", "middle"))

    # Arrows image -> encoder and decoder -> image
    def arrow(x0, x1):
        svg.append(f'<path class="sch-line" d="M{x0},{cy} L{x1},{cy}"/>')
        svg.append(f'<path class="sch-line" d="M{x1 - 7},{cy - 5} L{x1},{cy} L{x1 - 7},{cy + 5}"/>')

    arrow(in_x + G + 6, left - 4)
    arrow(right - 2 + 12 - 12, out_x - 6)

    # Braces naming the parts
    def brace(x0, x1, y, label, math):
        svg.append(f'<path class="sch-line" d="M{x0},{y} L{x0},{y + 6} L{x1},{y + 6} L{x1},{y}"/>')
        xm = (x0 + x1) / 2
        svg.append(f'<text class="sch-label" x="{xm}" y="{y + 26}" text-anchor="middle">{label}</text>')
        svg.append(tex(xm, y + 48, math, "small", "middle"))

    yb = cy + 88
    brace(in_x - 4, in_x + G + 4, yb, "Image", "x")
    brace(enc[0][0], enc[2][0] + 12, yb, "Encoder", "e")
    brace(mid - 22, mid + 22, yb, "Code", "z = e(x)")
    brace(dec[0][0], dec[2][0] + 12, yb, "Decoder", "g")
    brace(out_x - 4, out_x + G + 4, yb, "Reconstruction", "g(z)")

    # Reconstruction error: arc from output back to input, over the network
    top = cy - G / 2 - 8
    xa, xb = in_x + G / 2, out_x + G / 2
    svg.append(f'<path class="sch-line sch-thin" d="M{xb},{top} C{xb},{top - 40} {xa},{top - 40} {xa},{top}"/>')
    svg.append(f'<path class="sch-line sch-thin" d="M{xa - 5},{top - 7} L{xa},{top} L{xa + 5},{top - 7}"/>')
    svg.append(f'<text class="sch-label" x="{W / 2}" y="{top - 36}" text-anchor="middle">'
               'reconstruction error</text>')
    svg.append("</svg>")

    SVG.parent.mkdir(parents=True, exist_ok=True)
    SVG.write_text("\n".join(svg) + "\n", encoding="utf-8")
    print(f"wrote {SVG}  (test image {index}, digit {ys[index].item()}, reconstruction error {err:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "figure"])
    ap.add_argument("--index", type=int, default=1, help="test image shown in the figure")
    a = ap.parse_args()
    train() if a.cmd == "train" else figure(a.index)
