"""The covering schematic for the section "Generators without a density" of Part 1 (learning-a-distribution).

covering.svg  A quarter of the unit circle, the image of [0, 1] under g(t) = (cos(pi t / 2), sin(pi t / 2)), which is
              Lipschitz with L = pi / 2. For N = 4, 8 and 16 the segment is split into N pieces, and each piece lands
              in the square of side 2L/N centred on the image of its starting point. The total area of the squares
              is N (2L/N)^2 = 4 L^2 / N = pi^2 / N, halved at each doubling of N. All three panels share one scale.

The SVG uses classes only (sch-*), styled by the course pages, and is inlined by the build
(![caption](figures/covering.svg){.inline}) so it follows the light and dark themes.

usage: python -m chapter1.no_density
"""
import math
import numpy as np

from lab import FIGURES as ROOT_FIGURES
from lab.schematic import tex

FIGURES = ROOT_FIGURES / "chapter1"
L = math.pi / 2


def g(t):
    return math.cos(L * t), math.sin(L * t)


def f(v):
    return f"{v:.1f}"


def arc_path(to_px, n=120):
    pts = [to_px(*g(t)) for t in np.linspace(0, 1, n)]
    return "M" + " L".join(f"{f(x)},{f(y)}" for x, y in pts)


# ---------------------------------------------------------------- covering
def covering():
    W, H, P, GAP, TOP = 720, 290, 210, 35, 20
    # data window: x, y in [-0.45, 1.45]; same for every panel
    LO, HI = -0.45, 1.45
    svg = [f'<svg class="schematic" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    for i, n in enumerate((4, 8, 16)):
        x0 = 10 + i * (P + GAP)

        def px(x, y, x0=x0):
            return x0 + (x - LO) / (HI - LO) * P, TOP + (HI - y) / (HI - LO) * P

        svg.append(f'<rect class="sch-frame" x="{x0}" y="{TOP}" width="{P}" height="{P}" rx="10"/>')
        side = 2 * L / n
        for j in range(n):
            cx, cy = g(j / n)
            (ax, ay), (bx, by) = px(cx - side / 2, cy + side / 2), px(cx + side / 2, cy - side / 2)
            svg.append(f'<rect class="sch-region sch-thin" x="{f(ax)}" y="{f(ay)}" width="{f(bx - ax)}" height="{f(by - ay)}"/>')
        svg.append(f'<path class="sch-line" d="{arc_path(px)}"/>')
        for j in range(n):                      # starting points of the pieces
            sx, sy = px(*g(j / n))
            svg.append(f'<circle class="sch-ink" cx="{f(sx)}" cy="{f(sy)}" r="2.2"/>')
        area = 4 * L ** 2 / n
        svg.append(tex(x0 + P / 2, TOP + P + 30, f"N = {n}", "small", "middle"))
        svg.append(f'<text class="sch-label" x="{f(x0 + P / 2)}" y="{TOP + P + 52}" text-anchor="middle">'
                   f'total area {area:.2f}</text>')
    svg.append("</svg>")
    (FIGURES / "covering.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    covering()
    print("wrote", FIGURES / "covering.svg")
