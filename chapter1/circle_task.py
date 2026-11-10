"""The learning problem of Part 1 (learning-a-distribution) on the circle, before any formalism.

circle-task.svg  Three panels on one scale, the unit circle dashed in each. Left, the training set: the 12 points of
                 circle(12, seed=0), the same draw as the runs of the section "Memorization" (open circles, as there).
                 Middle, 300 samples of the copying model, each a training point drawn at random: they stack on the 12
                 points, and the grey disc around each point has an area proportional to its number of samples. Right,
                 300 samples of the uniform distribution on the circle (grey dots), the model we want.

The SVG uses classes only (sch-*), styled in assets/css/post.css, and is inlined by the build
(![caption](figures/circle-task.svg){.inline}) so it follows the light and dark themes.

usage: python -m chapter1.circle_task
"""
import math

import numpy as np

from lab import FIGURES as ROOT_FIGURES
from lab import data

FIGURES = ROOT_FIGURES / "chapter1"
SAMPLES, SAMPLE_SEED = 300, 1


def f(v):
    return f"{v:.1f}"


def circle_task():
    W, H, P, GAP, TOP = 720, 262, 210, 35, 36
    LO, HI = -1.3, 1.3                           # same window for every panel
    train = data.circle(12, seed=0).numpy()
    rng = np.random.default_rng(SAMPLE_SEED)
    counts = np.bincount(rng.integers(0, len(train), SAMPLES), minlength=len(train))
    angles = 2 * math.pi * rng.random(SAMPLES)
    svg = [f'<svg class="schematic" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    titles = ("Training set", "Copying model", "Model we want")
    for i, title in enumerate(titles):
        x0 = 10 + i * (P + GAP)

        def px(x, y, x0=x0):
            return x0 + (x - LO) / (HI - LO) * P, TOP + (HI - y) / (HI - LO) * P

        cx, cy = px(0, 0)
        r = P / (HI - LO)
        svg.append(f'<rect class="sch-frame" x="{x0}" y="{TOP}" width="{P}" height="{P}" rx="10"/>')
        svg.append(f'<circle class="sch-ring sch-thin" style="stroke-dasharray: 4 4; stroke-opacity: .5" '
                   f'cx="{f(cx)}" cy="{f(cy)}" r="{f(r)}"/>')
        if i == 1:                               # area of the grey disc proportional to the number of samples
            for (x, y), c in zip(train, counts):
                sx, sy = px(x, y)
                svg.append(f'<circle class="sch-muted" cx="{f(sx)}" cy="{f(sy)}" r="{f(1.6 * math.sqrt(c))}"/>')
        if i == 2:
            for a in angles:
                sx, sy = px(math.cos(a), math.sin(a))
                svg.append(f'<circle class="sch-muted" cx="{f(sx)}" cy="{f(sy)}" r="2.2"/>')
        for x, y in train:
            sx, sy = px(x, y)
            svg.append(f'<circle class="sch-ring" style="stroke-width: 1.4" cx="{f(sx)}" cy="{f(sy)}" r="4.5"/>')
        svg.append(f'<text class="sch-label" style="font-size: 14px" x="{f(x0 + P / 2)}" y="{TOP - 12}" '
                   f'text-anchor="middle">{title}</text>')
    svg.append("</svg>")
    (FIGURES / "circle-task.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    return counts


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    counts = circle_task()
    print("wrote", FIGURES / "circle-task.svg", "| copying-model counts per point:", counts.tolist())
