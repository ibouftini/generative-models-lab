"""The linear model of Chapter 3, a line in the plane with Gaussian noise around it.

Left: 500 samples of X = aZ + b + sigma_dec * eps around the line az + b, and one point x. Right: the prior N(0, 1) and
the posterior p(z | x) of the linear model, N(a^T (x - b) / (|a|^2 + s2), s2 / (|a|^2 + s2)).

usage: python -m chapter3.linear_model
"""
import numpy as np

from lab import FIGURES
from lab.plotstyle import plt, save

A, B, SIGMA = np.array([1.0, 0.5]), np.zeros(2), 0.15
X0 = np.array([0.9, 0.75])                                              # the point whose posterior is shown


def main():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    x = z[:, None] * A + B + SIGMA * rng.normal(size=(500, 2))
    s2, norm2 = SIGMA ** 2, A @ A
    post_mean, post_var = A @ (X0 - B) / (norm2 + s2), s2 / (norm2 + s2)
    print(f"posterior mean {post_mean:.3f}, sd {np.sqrt(post_var):.3f}, prior sd / posterior sd "
          f"{1 / np.sqrt(post_var):.2f}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3.6), dpi=150)
    t = np.linspace(-3, 3, 2)
    left.plot(t * A[0] + B[0], t * A[1] + B[1], color="black", lw=1.2)
    left.scatter(x[:, 0], x[:, 1], s=4, color="black", alpha=0.35, lw=0)
    left.scatter([X0[0]], [X0[1]], s=60, facecolors="none", edgecolors="black", lw=1.4)
    left.annotate("$x$", X0, textcoords="offset points", xytext=(7, 5))
    left.annotate("$az + b$", (2.4, 1.2), textcoords="offset points", xytext=(-10, 8), ha="right")
    left.set_xlim(-3, 3)
    left.set_ylim(-2, 2)
    left.set_aspect("equal")
    left.set_xticks([])
    left.set_yticks([])
    left.set_title("Samples of $X$", fontsize=9)

    zz = np.linspace(-3.5, 3.5, 801)
    prior = np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi)
    post = np.exp(-(zz - post_mean) ** 2 / (2 * post_var)) / np.sqrt(2 * np.pi * post_var)
    right.plot(zz, prior, color="black", lw=1.2, ls="--", label="Prior $p(z)$")
    right.plot(zz, post, color="black", lw=1.6, label=r"Posterior $p(z \mid x)$")
    right.set_xlabel("Code $z$")
    right.set_yticks([])
    right.legend(frameon=False, fontsize=8, loc="upper left")
    for side in ("top", "right", "left"):
        right.spines[side].set_visible(False)
    right.set_title("Codes that could have produced $x$", fontsize=9)
    fig.tight_layout()
    out = FIGURES / "chapter3" / "linear-model.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save(fig, out, transparent=True)
    print("written", out)


if __name__ == "__main__":
    main()
