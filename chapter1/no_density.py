"""Two schematics for the section on generators without a density.

covering.svg  A quarter of the unit circle, the image of [0, 1] under g(t) = (cos(pi t / 2), sin(pi t / 2)), which is
              Lipschitz with L = pi / 2. For N = 4, 8 and 16 the segment is split into N pieces, and each piece lands
              in the square of side 2L/N centred on the image of its starting point. The total area of the squares
              is N (2L/N)^2 = 4 L^2 / N = pi^2 / N, halved at each doubling of N. All three panels share one scale.
noise.svg     The same arc and a training point X_i off it. Left, no noise: all the mass is on the arc. Right, Gaussian
              noise spreads the arc into a band whose shading fades with the distance to the arc, and an arrow from
              the arc to X_i shows the arc moving toward it.

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


# ---------------------------------------------------------------- Figure A: covering
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


# ---------------------------------------------------------------- Figure B: noise
# Subscripts follow KaTeX: 0.7 of the size (13 px under 18 px) and lowered by 0.15 em (3 px).
def noise():
    W, H, P, TOP = 720, 290, 250, 36
    GAP = 40                                    # same visual pairing as Figure 4
    LX = (W - 2 * P - GAP) / 2
    RX = LX + P + GAP
    LO, HI = -0.42, 1.42                        # room for the noise band inside the frame
    R_XI = 1.3                                  # training point at distance 0.3 from the arc, at 45 degrees
    XI = (R_XI * math.cos(L / 2), R_XI * math.sin(L / 2))
    svg = [f'<svg class="schematic" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    for x0, noisy in ((LX, False), (RX, True)):

        def px(x, y, x0=x0):
            return x0 + (x - LO) / (HI - LO) * P, TOP + (HI - y) / (HI - LO) * P

        svg.append(f'<rect class="sch-frame" x="{x0}" y="{TOP}" width="{P}" height="{P}" rx="10"/>')
        if noisy:
            # Fading band: wide faint strokes under narrower ones, so the shade falls off with the distance to the arc.
            for w in np.linspace(84, 6, 27):
                svg.append(f'<path class="sch-line" style="stroke-width: {w:.1f}; stroke-opacity: .035" d="{arc_path(px)}"/>')
        svg.append(f'<path class="sch-line" d="{arc_path(px)}"/>')
        xx, xy = px(*XI)
        svg.append(f'<circle class="sch-ink" cx="{f(xx)}" cy="{f(xy)}" r="4"/>')
        svg.append(tex(xx + 9, xy - 7, "X_i", "small"))
        if noisy:
            # The arc moves toward X_i: arrow along the outward normal, from just outside the arc to just short of
            # X_i, drawn like the other arrows of the chapter (sch-line, open head of 9 px in one stroke).
            t = 0.5
            a = px(1.035 * math.cos(L * t), 1.035 * math.sin(L * t))
            b = px((R_XI - 0.07) * math.cos(L * t), (R_XI - 0.07) * math.sin(L * t))
            ang = math.atan2(b[1] - a[1], b[0] - a[0])
            h1 = (b[0] - 9 * math.cos(ang - 0.5), b[1] - 9 * math.sin(ang - 0.5))
            h2 = (b[0] - 9 * math.cos(ang + 0.5), b[1] - 9 * math.sin(ang + 0.5))
            svg.append(f'<path class="sch-line" d="M{f(a[0])},{f(a[1])} L{f(b[0])},{f(b[1])}"/>')
            svg.append(f'<path class="sch-line" d="M{f(h1[0])},{f(h1[1])} L{f(b[0])},{f(b[1])} L{f(h2[0])},{f(h2[1])}"/>')
        title = '<text class="sch-label" style="font-size: 14px" x="{x:.1f}" y="{y}" text-anchor="{a}">{t}</text>'
        if noisy:   # "With noise" in the label font, then the math sigma_dec; the pair is centred on the panel
            svg.append(title.format(x=x0 + P / 2 + 16, y=TOP - 12, a="end", t="With noise"))
            svg.append(tex(x0 + P / 2 + 20, TOP - 12, r"\sigma_{\text{dec}}", "small"))
        else:
            svg.append(title.format(x=x0 + P / 2, y=TOP - 12, a="middle", t="No noise"))
    svg.append("</svg>")
    (FIGURES / "noise.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    covering()
    noise()
    print("wrote", FIGURES / "covering.svg", "and", FIGURES / "noise.svg")
