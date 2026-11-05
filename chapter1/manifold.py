"""Manifold schematic, drawn with the autoencoder of Chapter 3.

Left: the space of all images, drawn as a box, with a curved sheet inside (the manifold M). Along a path on the sheet
sit real digits: the decoder of the trained MNIST autoencoder applied to codes on the straight segment between the codes
of two test images of a 4. Off the sheet floats an image with uniformly random pixels.
Right: the flat code space, with the same codes on a segment. An arrow g (the generator, here the decoder) maps the
codes to the images.

usage: python -m chapter1.manifold     (needs checkpoints/mnist_ae.pt)
"""
import math
import numpy as np
import torch

from chapter3.autoencoder import AE, OUT, mnist
from lab import FIGURES
from lab.schematic import tex

SVG = FIGURES / "chapter1" / "manifold.svg"
PAIR = (1159, 1679)          # two test images of a 4, far apart in code space, well reconstructed
STEPS = 5

model = AE()
model.load_state_dict(torch.load(OUT / "mnist_ae.pt", map_location="cpu"))
model.eval()
xs, _ = mnist(False)
with torch.no_grad():
    za, zb = model.f(xs[PAIR[0]]), model.f(xs[PAIR[1]])
    ts = np.linspace(0, 1, STEPS)
    digits = [model.g(za + (zb - za) * float(t)).view(28, 28).numpy() for t in ts]
noise = np.random.default_rng(0).random((28, 28))

W, H = 720, 345
svg = [f'<svg class="schematic" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']

# ---- Left panel: oblique projection of a box [0,1]^3 and a wavy sheet z = h(u) inside it
OX, OY = 40, 300            # screen position of the box corner (u, v, z) = (0, 0, 0)
SU, SV, SZ = 330, 95, 190   # screen lengths of the u, v (depth) and z axes
DX = 0.75                   # depth axis direction: right and up


def proj(u, v, z):
    return OX + u * SU + v * SV * DX, OY - v * SV * 0.62 - z * SZ


def sheet_z(u, v):
    return 0.42 + 0.16 * math.sin(2 * math.pi * (u - 0.15)) + 0.05 * (v - 0.5)


def pts(path):
    return " L".join(f"{x:.1f},{y:.1f}" for x, y in path)


# box edges: back ones first
corners = {(a, b, c): proj(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)}
for (p, q) in [((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)), ((0, 0, 0), (0, 1, 0))]:
    (x1, y1), (x2, y2) = corners[p], corners[q]
    svg.append(f'<path class="sch-frame" d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}"/>')

# sheet: filled outline, then grid lines along u and v
N = 40
outline = ([proj(u, 0, sheet_z(u, 0)) for u in np.linspace(0, 1, N)]
           + [proj(1, v, sheet_z(1, v)) for v in np.linspace(0, 1, 10)]
           + [proj(u, 1, sheet_z(u, 1)) for u in np.linspace(1, 0, N)]
           + [proj(0, v, sheet_z(0, v)) for v in np.linspace(1, 0, 10)])
svg.append(f'<path class="sch-region" d="M{pts(outline)} Z"/>')
for u in np.linspace(0, 1, 9)[1:-1]:
    svg.append(f'<path class="sch-grid" d="M{pts([proj(u, v, sheet_z(u, v)) for v in np.linspace(0, 1, 10)])}"/>')
for v in np.linspace(0, 1, 5)[1:-1]:
    svg.append(f'<path class="sch-grid" d="M{pts([proj(u, v, sheet_z(u, v)) for u in np.linspace(0, 1, N)])}"/>')

# path of the interpolation on the sheet, and the digits sitting on it
V0 = 0.45
us = np.linspace(0.1, 0.9, STEPS)
svg.append(f'<path class="sch-line" d="M{pts([proj(u, V0, sheet_z(u, V0)) for u in np.linspace(us[0], us[-1], N)])}"/>')
T = 34                       # thumbnail size
s = T / 28


def thumb(img, x0, y0, frame=True):
    if frame:
        svg.append(f'<rect class="sch-card" x="{x0 - 3:.1f}" y="{y0 - 3:.1f}" width="{T + 6}" height="{T + 6}" rx="4"/>')
    for k, val in enumerate(img.flatten().tolist()):
        if val > 0.06:
            i, j = divmod(k, 28)
            svg.append(f'<rect class="sch-ink" x="{x0 + j * s:.2f}" y="{y0 + i * s:.2f}" width="{s:.2f}" '
                       f'height="{s:.2f}" fill-opacity="{min(val, 1):.2f}"/>')


for u, img in zip(us, digits):
    px, py = proj(u, V0, sheet_z(u, V0))
    svg.append(f'<circle class="sch-ink" cx="{px:.1f}" cy="{py:.1f}" r="3"/>')
    svg.append(f'<path class="sch-line sch-thin" d="M{px:.1f},{py - 3:.1f} L{px:.1f},{py - 22:.1f}"/>')
    thumb(img, px - T / 2, py - 22 - T - 3)

# image with random pixels, off the sheet
nx, ny = proj(0.82, 0.9, 1.05)
thumb(noise, nx - T / 2, ny - T / 2)
svg.append(f'<text class="sch-label sch-tiny" x="{nx:.1f}" y="{ny + T / 2 + 16:.1f}" text-anchor="middle">Random pixels</text>')

# front box edges last
for (p, q) in [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),
               ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)), ((0, 1, 1), (1, 1, 1)),
               ((1, 1, 0), (1, 1, 1))]:
    (x1, y1), (x2, y2) = corners[p], corners[q]
    svg.append(f'<path class="sch-frame" d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}"/>')

mx, my = proj(0.02, 0.05, sheet_z(0.02, 0.05))
svg.append(tex(mx + 4, my + 26, r"\mathcal{M}"))

# ---- Right panel: flat code space with the codes on a segment
CX, CY, CS = 560, 190, 130
svg.append(f'<rect class="sch-frame" x="{CX - CS / 2}" y="{CY - CS / 2}" width="{CS}" height="{CS}" rx="8"/>')
for k in range(1, 4):
    o = CS * k / 4 - CS / 2
    svg.append(f'<path class="sch-grid" d="M{CX + o},{CY - CS / 2} L{CX + o},{CY + CS / 2}"/>')
    svg.append(f'<path class="sch-grid" d="M{CX - CS / 2},{CY + o} L{CX + CS / 2},{CY + o}"/>')
a, b = (CX - 40, CY + 30), (CX + 42, CY - 28)
svg.append(f'<path class="sch-line" d="M{a[0]},{a[1]} L{b[0]},{b[1]}"/>')
for t in ts:
    svg.append(f'<circle class="sch-ink" cx="{a[0] + (b[0] - a[0]) * t:.1f}" cy="{a[1] + (b[1] - a[1]) * t:.1f}" r="3"/>')

# ---- Arrow g (codes -> images)
def curved_arrow(x0, y0, x1, y1, bend, label, lx, ly):
    mx_, my_ = (x0 + x1) / 2, (y0 + y1) / 2 + bend
    svg.append(f'<path class="sch-line" d="M{x0},{y0} Q{mx_},{my_} {x1},{y1}"/>')
    ang = math.atan2(y1 - my_, x1 - mx_)
    for d in (2.6, -2.6):
        hx, hy = x1 - 9 * math.cos(ang + d / 6), y1 - 9 * math.sin(ang + d / 6)
        svg.append(f'<path class="sch-line" d="M{hx:.1f},{hy:.1f} L{x1},{y1}"/>')
    svg.append(tex(lx, ly, label, "", "middle"))


curved_arrow(CX - CS / 2 - 8, CY + 30, 430, 240, 40, "g", 462, 268)

# ---- Captions
svg.append(f'<text class="sch-label" x="{OX + SU / 2 + 35}" y="{OY + 30}" text-anchor="middle">Space of all images</text>')
svg.append(f'<text class="sch-label" x="{CX}" y="{CY + CS / 2 + 30}" text-anchor="middle">Codes</text>')
svg.append("</svg>")

SVG.parent.mkdir(parents=True, exist_ok=True)
SVG.write_text("\n".join(svg) + "\n", encoding="utf-8")
print("wrote", SVG)
