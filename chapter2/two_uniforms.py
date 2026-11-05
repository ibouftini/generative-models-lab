"""Two uniform distributions of width w, P = U[-w/2, w/2] and Q = U[theta - w/2, theta + w/2]: TV, KL, JS and W1 as
functions of the shift theta, drawn for w = 0.5.

  with r = min(|theta| / w, 1):  TV = r,  JS = r log 2,  W1 = |theta|,  KL = +inf for every theta != 0

usage: python -m chapter2.two_uniforms
"""
import math

from lab import FIGURES
from lab.plotstyle import plt, save

out = FIGURES / "chapter2" / "two-uniforms.png"
out.parent.mkdir(parents=True, exist_ok=True)

W = 0.5
LOG2 = math.log(2)
INF = 2.45  # the "+inf" level, above a break of the vertical axis at 2.2 (as in the interactive figure)

plt.rcParams.update({"font.size": 11, "axes.edgecolor": "black",
                     "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})
fig, (strip, ax) = plt.subplots(2, 1, figsize=(8, 5.4), dpi=160, sharex=True,
                                gridspec_kw={"height_ratios": [1, 4.2], "hspace": 0.08})

# Strip: the two densities at the shift drawn
theta = 0.3
strip.fill_between([-W / 2, W / 2], 0, 1, color="black", alpha=0.08, lw=0)
strip.plot([-W / 2, -W / 2, W / 2, W / 2], [0, 1, 1, 0], color="black", lw=1.4)
strip.plot([theta - W / 2, theta - W / 2, theta + W / 2, theta + W / 2], [0, 1, 1, 0], color="black", lw=1.4,
           ls=(0, (5, 3)))
strip.text(-W / 2 - 0.08, 0.15, r"$\mathbb{P}$", ha="right")
strip.text(theta + W / 2 + 0.08, 0.15, r"$\mathbb{Q}$", ha="left")
strip.text(2.28, 0.5, rf"$\Delta = {W:g},\ \theta = {theta:g}$", ha="right", va="center", fontsize=9)
strip.axhline(0, color="black", lw=0.8, alpha=0.5)
strip.set_ylim(0, 1.25)
strip.axis("off")

# Plot: the four distances as functions of theta
ax.plot([-2, 0, 2], [2, 0, 2], color="black", lw=2)
for top, dash in ((1, (0, (6, 4))), (LOG2, (0, (1.5, 3)))):
    ax.plot([-2, -W, 0, W, 2], [top, top, 0, top, top], color="black", lw=1.6, ls=dash)
for side in ((-2, -0.06), (0.06, 2)):
    ax.plot(side, [INF, INF], color="black", lw=1.6, ls=(0, (10, 4, 2, 4)))
ax.plot([0], [INF], "o", mfc="white", mec="black", ms=7, mew=1.5)
ax.plot([0], [0], "o", color="black", ms=6)
# the shift drawn in the strip, with its four values (as the marker of the interactive figure)
ax.axvline(theta, color="black", lw=0.8, ls=(0, (2, 2)), alpha=0.6)
for v in (abs(theta), min(abs(theta) / W, 1), min(abs(theta) / W, 1) * LOG2, INF):
    ax.plot([theta], [v], "o", color="black", ms=5)
for v, name in ((2, r"$W_1$"), (INF, r"$\mathrm{KL}$"), (1, r"$\mathrm{TV}$"), (LOG2, r"$\mathrm{JS}$")):
    ax.text(2.08, v, name, va="center", ha="left")
ax.plot([-2.05, -1.95], [2.17, 2.23], color="black", lw=1.2, clip_on=False)  # axis break
ax.set_xlim(-2.05, 2.3)
ax.set_ylim(-0.05, INF + 0.12)
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_yticks([0, LOG2, 1, 2, INF], ["$0$", r"$\log 2$", "$1$", "$2$", r"$+\infty$"])
ax.set_xlabel(r"Shift $\theta$")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_bounds(-2, 2)
save(fig, out, transparent=True, bbox_inches="tight")
print("written", out)
