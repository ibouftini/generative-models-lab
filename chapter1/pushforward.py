"""Draw the pushforward schematic of Part 1 (learning-a-distribution), from the circle generator of Example 2.

Latent samples Z ~ N(0, 1) on the real line are pushed through
    g(z) = (cos 2 pi Phi(z), sin 2 pi Phi(z))
onto the unit circle, where they are uniform. The region A is a thin band across the arc of angles [T1, T2], so its
preimage g^{-1}[A] is exactly the interval [Phi^{-1}(T1 / 2 pi), Phi^{-1}(T2 / 2 pi)], and both carry the probability
(T2 - T1) / 2 pi. The left panel draws the Gaussian density above the line and shades its area over the interval.
Samples are drawn as rugs of short ticks, so a one-dimensional distribution stays on its line or curve. The arrow g
goes from the latent panel to the data panel, and a second arrow under it goes back, from a set A to its preimage
g^{-1}[A] (square brackets: g^{-1} is applied to a set, so g need not be invertible).

The SVG uses classes only (sch-*), styled in assets/css/post.css, and is inlined by the build
(![caption](figures/pushforward.svg){.inline}) so it follows the light and dark themes.

usage: python -m chapter1.pushforward
"""
import math
import numpy as np
from scipy.stats import norm

from lab import FIGURES
from lab.schematic import tex

OUT = FIGURES / "chapter1" / "pushforward.svg"

rng = np.random.default_rng(3)
Z = rng.standard_normal(320)
ANGLE = 2 * math.pi * norm.cdf(Z)

# Region A: the arc of angles [T1, T2] (a thin band around it), and its preimage interval
T1, T2 = math.radians(25), math.radians(80)
Z_LO, Z_HI = norm.ppf(T1 / (2 * math.pi)), norm.ppf(T2 / (2 * math.pi))
hit = (Z >= Z_LO) & (Z <= Z_HI)
EXACT = (T2 - T1) / (2 * math.pi)

# Layout: two square panels, arrow in between
W, H = 720, 350
P = 250                     # panel size
LX, RX, TOP = 20, W - 20 - P, 20
ZMAX = 3.2                  # latent panel shows [-ZMAX, ZMAX]
BASE = TOP + 0.66 * P       # latent axis
PEAK = 0.44 * P             # height of the density at 0
CX, CY = RX + P / 2, TOP + P / 2
R = 0.34 * P                # circle radius in pixels
TICK = 6                    # half length of a rug tick
BAND = 13                   # half width of the band A


def f(v):
    return f"{v:.1f}"


def lx(z):
    return LX + P / 2 + z / ZMAX * (P / 2 - 14)


def ly(z):
    return BASE - norm.pdf(z) / norm.pdf(0) * PEAK


def on_circle(t, r):
    return CX + r * math.cos(t), CY - r * math.sin(t)


svg = [f'<svg class="schematic" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
for x0 in (LX, RX):
    svg.append(f'<rect class="sch-frame" x="{x0}" y="{TOP}" width="{P}" height="{P}" rx="10"/>')

# ---- Latent panel: density, shaded preimage, axis, rug
zs = np.linspace(Z_LO, Z_HI, 40)
area = [(lx(Z_LO), BASE)] + [(lx(z), ly(z)) for z in zs] + [(lx(Z_HI), BASE)]
svg.append('<path class="sch-region" d="M' + " L".join(f"{f(x)},{f(y)}" for x, y in area) + ' Z"/>')
zs = np.linspace(-ZMAX + 0.15, ZMAX - 0.15, 160)
svg.append('<path class="sch-line" d="M' + " L".join(f"{f(lx(z))},{f(ly(z))}" for z in zs) + '"/>')
svg.append(f'<path class="sch-grid" d="M{f(lx(-ZMAX + 0.1))},{f(BASE)} L{f(lx(ZMAX - 0.1))},{f(BASE)}"/>')
svg.append(f'<path class="sch-line" d="M{f(lx(Z_LO))},{f(BASE)} L{f(lx(Z_HI))},{f(BASE)}"/>')
for z, h in sorted(zip(Z, hit), key=lambda t: t[1]):        # ink ticks on top
    if abs(z) < ZMAX - 0.1:
        cls = "sch-line sch-thin" if h else "sch-grid"
        svg.append(f'<path class="{cls}" d="M{f(lx(z))},{f(BASE + 4)} L{f(lx(z))},{f(BASE + 4 + 2 * TICK)}"/>')
svg.append(tex(lx(0), BASE + 4 + 2 * TICK + 14, "0", "tiny", "middle"))
mid = (Z_LO + Z_HI) / 2
svg.append(tex(lx(mid) - 6, ly(Z_HI) - 14, r"g^{-1}[A]", "small", "end"))

# ---- Data panel: circle, band A, rug
svg.append(f'<circle class="sch-grid" cx="{f(CX)}" cy="{f(CY)}" r="{f(R)}"/>')
ts = np.linspace(T1, T2, 40)
band = [on_circle(t, R + BAND) for t in ts] + [on_circle(t, R - BAND) for t in ts[::-1]]
svg.append('<path class="sch-region" d="M' + " L".join(f"{f(x)},{f(y)}" for x, y in band) + ' Z"/>')
for t, h in sorted(zip(ANGLE, hit), key=lambda t: t[1]):
    (x0, y0), (x1, y1) = on_circle(t, R - TICK), on_circle(t, R + TICK)
    cls = "sch-line sch-thin" if h else "sch-grid"
    svg.append(f'<path class="{cls}" d="M{f(x0)},{f(y0)} L{f(x1)},{f(y1)}"/>')
ax, ay = on_circle((T1 + T2) / 2, R + BAND + 16)
svg.append(tex(ax, ay + 6, "A", "small", "middle"))

# ---- Arrow g
y = TOP + P / 2
x0, x1 = LX + P + 22, RX - 22
svg.append(f'<path class="sch-line" d="M{x0},{y} C{x0 + 40},{y - 30} {x1 - 40},{y - 30} {x1},{y}"/>')
svg.append(f'<path class="sch-line" d="M{x1 - 9},{y - 8} L{x1},{y} L{x1 - 11},{y + 3}"/>')
svg.append(tex((x0 + x1) / 2, y - 32, "g", "", "middle"))
# ---- Arrow back, from a set to its preimage, arching below the arrow g
y2 = y + 16
svg.append(f'<path class="sch-line" d="M{x1},{y2} C{x1 - 40},{y2 + 30} {x0 + 40},{y2 + 30} {x0},{y2}"/>')
svg.append(f'<path class="sch-line" d="M{x0 + 9},{y2 + 8} L{x0},{y2} L{x0 + 11},{y2 - 3}"/>')
svg.append(tex((x0 + x1) / 2, y2 + 46, r"g^{-1}[\,\cdot\,]", "small", "middle"))

# ---- Panel captions
cy = TOP + P + 34
svg.append(f'<text class="sch-label" x="{LX + P / 2}" y="{cy}" text-anchor="middle">Latent space</text>')
svg.append(f'<text class="sch-label" x="{RX + P / 2}" y="{cy}" text-anchor="middle">Data space</text>')
cy += 26
svg.append(tex(LX + P / 2, cy, r"Z \sim \gamma", "small", "middle"))
svg.append(tex(RX + P / 2, cy, r"g(Z) \sim g_{\#}\gamma", "small", "middle"))
svg.append("</svg>")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(svg) + "\n", encoding="utf-8")
print(f"wrote {OUT}  ({hit.sum()} of {len(Z)} samples in A, empirical mass {hit.mean():.3f}, exact {EXACT:.3f})")
