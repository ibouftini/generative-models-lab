"""Latent-space figures of Chapter 3 (run through chapter3.vae figures).

latent-codes.svg  Three columns: plain autoencoder (reconstruction only), VAE (both terms), VAE with the KL term dominant
                  (decoder variance x100). Top: the 2D codes of 2000 test digits, with the circles of radius 1, 2, 3 of
                  the prior N(0, I) and a label at the centre of each digit class. Bottom: codes drawn from N(0, I) and
                  decoded by the 8D model of the column (same codes for all).
latent-map.png    The 2D VAE decoder applied to a 15 x 15 grid of codes placed at the quantiles of N(0, 1): black ink on a
                  transparent background.
"""
import numpy as np
import torch
from PIL import Image
from scipy.stats import norm

from chapter3.autoencoder import AE, OUT, mnist
from chapter3.vae import FIG, load
from lab.schematic import tex


def thumbs(svg, imgs, x0, y0, T=34, gap=6):
    """Each image drawn on its 28 x 28 pixel grid (scaled to T): one path per ink level, a unit square per pixel."""
    s = T / 28
    for n, img in enumerate(imgs):
        ox = x0 + n * (T + gap)
        svg.append(f'<rect class="sch-frame" x="{ox - 2:.1f}" y="{y0 - 2:.1f}" width="{T + 4}" height="{T + 4}" rx="3"/>')
        levels = {}
        for k, v in enumerate(img.flatten().tolist()):
            if v > 0.06:
                i, j = divmod(k, 28)
                levels.setdefault(f"{min(v, 1):.2f}", []).append(f"M{j} {i}h1v1h-1z")
        svg.append(f'<g class="sch-ink" transform="translate({ox:.2f} {y0:.2f}) scale({s:.4f})">')
        svg.extend(f'<path fill-opacity="{v}" d="{"".join(d)}"/>' for v, d in sorted(levels.items()))
        svg.append("</g>")


def latent_codes():
    x, y = mnist(False)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), 2000, replace=False)
    W, H, P, GAP = 720, 372, 214, 26
    svg = [f'<svg class="schematic" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    ae8 = AE()
    ae8.load_state_dict(torch.load(OUT / "mnist_ae.pt", map_location="cpu"))
    ae8.eval()
    # title, 2D model, half-width of the plotted square, tick, 8D decoder
    columns = [("Reconstruction only", load("mnist_ae2.pt"), 28, 20, ae8.g),
               ("Both terms (VAE)", load("mnist_vae2.pt"), 4, 2, load("mnist_vae8.pt").dec),
               ("KL term dominant", load("mnist_vae2_kl.pt"), 4, 2, load("mnist_vae8_kl.pt").dec)]
    torch.manual_seed(7)
    zs = torch.randn(5, 8)
    for c, (title, model, half, tick, dec8) in enumerate(columns):
        X0 = 10 + c * (P + GAP)
        Y0 = 30
        cx, cy = X0 + P / 2, Y0 + P / 2
        with torch.no_grad():
            z = model.encode(x[idx])[0].numpy()

        def sx(v):
            return cx + v / half * P / 2

        def sy(v):
            return cy - v / half * P / 2

        svg.append(f'<text class="sch-label" x="{cx}" y="{Y0 - 12}" text-anchor="middle">{title}</text>')
        svg.append(f'<rect class="sch-frame" x="{X0}" y="{Y0}" width="{P}" height="{P}" rx="8"/>')
        svg.append(f'<path class="sch-grid" d="M{X0},{cy} L{X0 + P},{cy} M{cx},{Y0} L{cx},{Y0 + P}"/>')
        for t in (-tick, tick):
            svg.append(tex(sx(t), Y0 + P + 14, str(t), "tiny", "middle"))
        # one circle per code (overlaps darken), the class on their group
        svg.append('<g class="sch-muted">')
        svg.extend(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="1.3"/>' for (a, b) in z
                   if abs(a) < half and abs(b) < half)
        svg.append("</g>")
        for r in (1, 2, 3):
            svg.append(f'<circle class="sch-line sch-thin" cx="{cx}" cy="{cy}" r="{r / half * P / 2:.1f}"/>')
        for digit in range(10):
            m = np.median(z[y[idx].numpy() == digit], axis=0)
            svg.append(f'<text class="sch-label sch-digit" x="{sx(m[0]):.1f}" y="{sy(m[1]) + 6:.1f}" '
                       f'text-anchor="middle">{digit}</text>')
        # the same five codes z ~ N(0, I) in R^8, decoded by the 8D model of this column
        with torch.no_grad():
            imgs = dec8(zs).view(5, 28, 28).numpy()
        yb = Y0 + P + 64
        svg.append(f'<text class="sch-label sch-tiny" x="{cx}" y="{yb - 12}" text-anchor="middle">'
                   f'Five codes from the prior, decoded</text>')
        thumbs(svg, imgs, X0 + (P - (5 * 34 + 4 * 6)) / 2, yb, T=34, gap=6)
    svg.append("</svg>")
    (FIG / "latent-codes.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    print("wrote", FIG / "latent-codes.svg")


def latent_map(n=15):
    model = load("mnist_vae2.pt")
    g = norm.ppf(np.linspace(0.03, 0.97, n))
    z = torch.tensor([[a, b] for b in g[::-1] for a in g], dtype=torch.float32)
    with torch.no_grad():
        imgs = model.dec(z).view(n, n, 28, 28).numpy()
    grid = imgs.transpose(0, 2, 1, 3).reshape(n * 28, n * 28)
    rgba = np.zeros((n * 28, n * 28, 4), np.uint8)
    rgba[..., 3] = (np.clip(grid, 0, 1) * 255).astype(np.uint8)       # black ink, alpha = intensity
    Image.fromarray(rgba, "RGBA").resize((n * 28 * 2, n * 28 * 2), Image.LANCZOS).save(FIG / "latent-map.png", optimize=True)
    print("wrote", FIG / "latent-map.png")


def figures():
    FIG.mkdir(parents=True, exist_ok=True)
    latent_codes()
    latent_map()
