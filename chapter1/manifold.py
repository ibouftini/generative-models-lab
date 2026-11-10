"""Manifold schematic for Chapter 1 (learning-a-distribution), drawn with the MNIST autoencoder of chapter1/autoencoder.py.

Right: the space of all images R^d, drawn as a box, with two surfaces inside.
  M        the data manifold (shaded sheet), the plausible digits.
  g(R^k)   the image of the generator (here the decoder of the trained MNIST autoencoder, chapter1/autoencoder.py),
           drawn dashed only where it leaves M: it equals M around the training digits and drifts away at the two
           ends, downwards on the left and upwards on the right. A generator only approximates M.
On both surfaces: the path digits, the decoder applied to the straight segment between the codes of two test images
of a 4. On g(R^k) but off M, marked by white circles: the decoder at two codes on the line through the end
codes, beyond the segment, z(s) = za + s (zb - za). Past zb by 1.25 segments (s = 2.25) it draws a deformed 4 (top
stroke shrunk to a hook, right stroke swollen), at the left end where g(R^k) stays near M; past za by two segments (s = -2) it draws a
swollen shape that is no digit, at the right end where g(R^k) is far from M. The latent panel shows the same line,
dashed beyond the segment, with the two codes as white circles. Off both, floating: a noisy digit
sampled from the 8-dimensional VAE of chapter1/vae.py (decoder noise on, at a prior-improbable code of norm 7) and pure noise
(uniformly random pixels). All images are shown at the ink level of the path digits (display only).
Left: the flat latent space R^k with the codes of the path on a segment. A straight arrow g maps it to the images.

Everything is drawn with the sch-* classes (assets/css/post.css), so the inlined SVG follows the theme.
usage: python -m chapter1.manifold     (needs checkpoints/mnist_ae.pt and checkpoints/mnist_vae8.pt)
"""
import math

import numpy as np
import torch

from chapter1.autoencoder import AE, OUT, mnist
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
# Off-M images from the same decoder, at codes on the line through the two end codes, beyond the segment:
# z(s) = za + s (zb - za), the path being 0 <= s <= 1. Past zb by 1.25 segments: a deformed 4, its top stroke shrunk
# to a hook and its right stroke swollen (Mahalanobis radius 13, training codes at most 6.2). Past za by two
# segments: a swollen shape that is no digit (radius 15.4).
S_NEAR, S_FAR = 2.25, -2.0
with torch.no_grad():
    swollen_four = model.g(za + (zb - za) * S_NEAR).view(28, 28).numpy()
    far_shape = model.g(za + (zb - za) * S_FAR).view(28, 28).numpy()
digits = digits[::-1]        # zb end on the left of the box, so each extension lands beside its own end

# A noisy digit: one sample of the 8-dimensional VAE of chapter1/vae.py, decoder noise included, at an improbable code
from chapter1.vae import load as load_vae  # noqa: E402
vae = load_vae("mnist_vae8.pt")
gen = torch.Generator().manual_seed(1)
u = torch.randn(8, generator=gen)
with torch.no_grad():
    noisy = (vae.dec((7 * u / u.norm()).view(1, 8)).view(784)
             + vae.s2 ** 0.5 * torch.randn(784, generator=gen)).view(28, 28).numpy()
pure_noise = np.random.default_rng(0).random((28, 28))


W, H = 720, 368
svg = [f'<svg class="schematic" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']

# ---- Right panel: oblique projection of a box [0,1]^3 and a wavy sheet z = h(u) inside it
OX, OY = 270, 300           # screen position of the box corner (u, v, z) = (0, 0, 0)
SU, SV, SZ = 310, 95, 190   # screen lengths of the u, v (depth) and z axes
BOX_U, BOX_V, BOX_Z = (-0.05, 1.17), (-0.05, 1.05), (-0.05, 1.05)   # the box: the sheet's [0,1]^3 with a margin
DX = 0.75                   # depth axis direction: right and up


def proj(u, v, z):
    return OX + u * SU + v * SV * DX, OY - v * SV * 0.62 - z * SZ


def sheet_z(u, v):
    return 0.42 + 0.16 * math.sin(2 * math.pi * (u - 0.15)) + 0.05 * (v - 0.5)


DRIFT_UP, DRIFT_DOWN = 0.35, 0.2   # how far g(R^k) leaves M at the right end (upwards) and at the left end (downwards)


def g_z(u, v):
    """The image g(R^k) of the generator: equal to M around the training digits (0.2 <= u <= 0.8), leaving it at the
    two ends, upwards on the right and downwards on the left."""
    if u > 0.8:
        return sheet_z(u, v) + DRIFT_UP * ((u - 0.8) / 0.2) ** 2
    if u < 0.2:
        return sheet_z(u, v) - DRIFT_DOWN * ((0.2 - u) / 0.2) ** 2
    return sheet_z(u, v)


def pts(path):
    return " L".join(f"{x:.1f},{y:.1f}" for x, y in path)


# box edges: back ones first
corners = {(a, b, c): proj(BOX_U[a], BOX_V[b], BOX_Z[c]) for a in (0, 1) for b in (0, 1) for c in (0, 1)}
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

# g(R^k), dashed, only where it leaves M: the two ends of the sheet
DASH = 'style="stroke-dasharray: 4 3"'
for lo, hi, edge in ((0.0, 0.2, 0.0), (0.8, 1.0, 1.0)):
    piece = ([proj(u, 0, g_z(u, 0)) for u in np.linspace(lo, hi, 12)]
             + [proj(hi, v, g_z(hi, v)) for v in np.linspace(0, 1, 10)]
             + [proj(u, 1, g_z(u, 1)) for u in np.linspace(hi, lo, 12)]
             + [proj(lo, v, g_z(lo, v)) for v in np.linspace(1, 0, 10)])
    svg.append(f'<path class="sch-line sch-thin" {DASH} d="M{pts(piece)}"/>')
    for v in (0.33, 0.66):
        svg.append(f'<path class="sch-line sch-thin" {DASH} d="M{pts([proj(u, v, g_z(u, v)) for u in np.linspace(lo, hi, 12)])}"/>')

# path of the interpolation on the sheet, and the digits sitting on it
V0 = 0.45
us = np.linspace(0.2, 0.8, STEPS)    # the path stays where g(R^k) and M agree (0.2 <= u <= 0.8)
svg.append(f'<path class="sch-line" d="M{pts([proj(u, V0, sheet_z(u, V0)) for u in np.linspace(us[0], us[-1], N)])}"/>')
T = 34                       # thumbnail size
s = T / 28


def thumb(img, x0, y0, frame=True, size=T):
    px_ = size / 28
    if frame:
        svg.append(f'<rect class="sch-card" x="{x0 - 3:.1f}" y="{y0 - 3:.1f}" width="{size + 6}" height="{size + 6}" rx="4"/>')
    for k, val in enumerate(img.flatten().tolist()):
        if val > 0.06:
            i, j = divmod(k, 28)
            svg.append(f'<rect class="sch-ink" x="{x0 + j * px_:.2f}" y="{y0 + i * px_:.2f}" width="{px_:.2f}" '
                       f'height="{px_:.2f}" fill-opacity="{min(val, 1):.2f}"/>')


for u, img in zip(us, digits):
    px, py = proj(u, V0, sheet_z(u, V0))
    svg.append(f'<circle class="sch-ink" cx="{px:.1f}" cy="{py:.1f}" r="3"/>')
    svg.append(f'<path class="sch-line sch-thin" d="M{px:.1f},{py - 3:.1f} L{px:.1f},{py - 22:.1f}"/>')
    thumb(img, px - T / 2, py - 22 - T - 3)


# Images produced by g sit on g(R^k), with a stem like the path digits; the others float, off both surfaces
# (image, u, v, stem): stem > 0 puts the box above its point of g(R^k), stem < 0 below it (pixels)
on_g = [(swollen_four, 0.08, 0.2, 26)]                                   # thick 4, near M (left)
SPECKS_AT, SPECKS_BOX = (0.98, 0.9), (1.09, 0.9, 0.84)   # far shape: point on the top of g(R^k)'s right end; box to its right
TG = 28                                                                  # smaller boxes for the images off M
floating = [(noisy, (0.55, 0.0, 0.16)), (pure_noise, (0.72, 0.9, 0.88))]


# Shown at the ink level of the path digits: an image's 99th percentile is scaled to theirs (display only)
INK = float(np.mean([np.percentile(d, 99) for d in digits]))


def shown(img):
    img = np.clip(img, 0, 1)
    return img * INK / max(float(np.percentile(img, 99)), 1e-6)


for img, u, v, stem in on_g:          # a white circle on g(R^k) (black dots are on M), a stem, then the box
    px, py = proj(u, v, g_z(u, v))
    sgn = 1 if stem > 0 else -1
    svg.append(f'<path class="sch-line sch-thin" d="M{px:.1f},{py - sgn * 3.5:.1f} L{px:.1f},{py - stem:.1f}"/>')
    svg.append(f'<circle class="sch-card" style="stroke: var(--text); stroke-width: 1.4" cx="{px:.1f}" cy="{py:.1f}" r="3.5"/>')
    top = py - stem - TG - 3 if stem > 0 else py - stem + 3
    thumb(shown(img), px - TG / 2, top, size=TG)
# far shape: in the margin right of the sheet, linked to the top of g(R^k)'s right end
px, py = proj(*SPECKS_AT, g_z(*SPECKS_AT))
bx, by = proj(*SPECKS_BOX)
svg.append(f'<path class="sch-line sch-thin" d="M{px:.1f},{py:.1f} L{bx - TG / 2 - 3:.1f},{by + TG / 2 + 3:.1f}"/>')
svg.append(f'<circle class="sch-card" style="stroke: var(--text); stroke-width: 1.4" cx="{px:.1f}" cy="{py:.1f}" r="3.5"/>')
thumb(shown(far_shape), bx - TG / 2, by - TG / 2, size=TG)
for img, (u, v, z) in floating:
    nx, ny = proj(u, v, z)
    thumb(shown(img), nx - T / 2, ny - T / 2)

# front box edges last
for (p, q) in [((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),
               ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)), ((0, 1, 1), (1, 1, 1)),
               ((1, 1, 0), (1, 1, 1))]:
    (x1, y1), (x2, y2) = corners[p], corners[q]
    svg.append(f'<path class="sch-frame" d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}"/>')

# labels outside the box, lower left, each beside the end of its own edge (solid for M, dashed for g(R^k))
mx, my = proj(0.0, 0.0, sheet_z(0.0, 0.0))
svg.append(tex(corners[(0, 0, 0)][0] - 10, my + 6, r"\mathcal{M}", "", "end"))
gx, gy = proj(0.0, 0.0, g_z(0.0, 0.0))
svg.append(tex(corners[(0, 0, 0)][0] - 10, gy + 5, r"g(\mathbb{R}^k)", "small", "end"))

# ---- Left panel: flat latent space with the codes on a segment
CX, CY, CS = 85, 190, 130
svg.append(f'<rect class="sch-frame" x="{CX - CS / 2}" y="{CY - CS / 2}" width="{CS}" height="{CS}" rx="8"/>')
for k in range(1, 4):
    o = CS * k / 4 - CS / 2
    svg.append(f'<path class="sch-grid" d="M{CX + o},{CY - CS / 2} L{CX + o},{CY + CS / 2}"/>')
    svg.append(f'<path class="sch-grid" d="M{CX - CS / 2},{CY + o} L{CX + CS / 2},{CY + o}"/>')
# the line through the two codes: zb at the lower left, za at the upper right (as in the box, where zb is on the left)
L_SEG, DIR = 66, np.array([82, -58]) / math.hypot(82, -58)
S_NEAR_SHOWN, S_FAR_SHOWN = 1.28, -0.7   # the panel is schematic: shortened extensions, the far one still the longer


def code_pt(s_):               # screen point of z(s) = za + s (zb - za); the span s in [S_FAR, S_NEAR] is centred
    mid = (S_FAR_SHOWN + S_NEAR_SHOWN) / 2
    return np.array([CX, CY]) - (s_ - mid) * L_SEG * DIR


for lo, hi in ((S_FAR_SHOWN, 0.0), (1.0, S_NEAR_SHOWN)):
    (xa_, ya_), (xb_, yb_) = code_pt(lo), code_pt(hi)
    svg.append(f'<path class="sch-line sch-thin" style="stroke-dasharray: 3 3" d="M{xa_:.1f},{ya_:.1f} L{xb_:.1f},{yb_:.1f}"/>')
(xa_, ya_), (xb_, yb_) = code_pt(0.0), code_pt(1.0)
svg.append(f'<path class="sch-line" d="M{xa_:.1f},{ya_:.1f} L{xb_:.1f},{yb_:.1f}"/>')
for t_ in ts:
    x_, y_ = code_pt(float(t_))
    svg.append(f'<circle class="sch-ink" cx="{x_:.1f}" cy="{y_:.1f}" r="3"/>')
for s_ in (S_NEAR_SHOWN, S_FAR_SHOWN):
    x_, y_ = code_pt(s_)
    svg.append(f'<circle class="sch-card" style="stroke: var(--text); stroke-width: 1.4" cx="{x_:.1f}" cy="{y_:.1f}" r="3.5"/>')

# ---- Arrow g, from the latent space (left) to the images (right): straight, with an open head
x0, x1, ya = CX + CS / 2 + 14, corners[(0, 0, 0)][0] - 14, CY
svg.append(f'<path class="sch-line" d="M{x0},{ya} L{x1},{ya}"/>')
svg.append(f'<path class="sch-line" d="M{x1 - 9},{ya - 6} L{x1},{ya} L{x1 - 9},{ya + 6}"/>')
svg.append(tex((x0 + x1) / 2, ya - 12, "g", "", "middle"))

# ---- Captions
svg.append(f'<text class="sch-label" x="{OX + SU / 2 + 35}" y="{OY + 30}" text-anchor="middle">Space of all images</text>')
svg.append(tex(OX + SU / 2 + 35, OY + 56, r"\mathbb{R}^d", "small", "middle"))
svg.append(f'<text class="sch-label" x="{CX}" y="{OY + 30}" text-anchor="middle">Latent space</text>')
svg.append(tex(CX, OY + 56, r"\mathbb{R}^k", "small", "middle"))
svg.append("</svg>")

SVG.parent.mkdir(parents=True, exist_ok=True)
SVG.write_text("\n".join(svg) + "\n", encoding="utf-8")
print("wrote", SVG)
