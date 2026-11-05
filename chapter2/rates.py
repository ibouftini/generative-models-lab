"""How fast does the Wasserstein distance between two samples shrink with their size n?

For two independent samples of size n from the same distribution P, the exact W1 between their empirical
distributions is an optimal assignment (Birkhoff: with equal weights, an optimal coupling is a permutation),
computed with scipy. Its expectation shrinks at the same rate as E W1(P_n, P), up to a factor 2:
E W1(P_n, P) <= E W1(P_n, P'_n) <= 2 E W1(P_n, P).

Distributions: a closed curve in R^16 (intrinsic dimension 1), a standard Gaussian in R^16 (dimension 16), and MNIST
digits (p = 784; two disjoint random subsets of the training set).

usage: python -m chapter2.rates run       # writes results/chapter2/rates.json
       python -m chapter2.rates figure    # writes figures/chapter2/rates.png
"""
import argparse
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from lab import FIGURES, data, results

SIZES = [50, 100, 200, 400, 800, 1600, 3200]
REPEATS = 5


def w1(a, b):
    cost = torch.cdist(a.double(), b.double()).numpy()
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols].mean()


def run():
    mnist, _ = data.mnist(True)
    samplers = {
        "curve in R^16 (m = 1)": lambda n, s: (data.curve(n, 16, seed=s), data.curve(n, 16, seed=10_000 + s)),
        "Gaussian in R^16 (m = 16)": lambda n, s: (data.gaussian(n, 16, seed=s), data.gaussian(n, 16, seed=10_000 + s)),
        "MNIST digits (p = 784)": lambda n, s: tuple(mnist[torch.randperm(len(mnist), generator=torch.Generator().manual_seed(s))[:2 * n]].split(n)),
    }
    numbers = {}
    for name, sample in samplers.items():
        means = []
        for n in SIZES:
            t0 = time.time()
            vals = [w1(*sample(n, s)) for s in range(REPEATS)]
            means.append(float(np.mean(vals)))
            print(f"{name:28s} n={n:5d}  W1 {means[-1]:.4f}  ({time.time() - t0:.1f} s)", flush=True)
        slope = float(np.polyfit(np.log(SIZES), np.log(means), 1)[0])
        tail = float(np.polyfit(np.log(SIZES[-4:]), np.log(means[-4:]), 1)[0])
        numbers[name] = {"sizes": SIZES, "mean_w1": means, "slope": slope, "slope_last_4": tail}
        print(f"  slope {slope:.3f} (last four sizes {tail:.3f})")
    path = results.save("chapter2", "rates", numbers, seed=0, script=__file__,
                        config={"sizes": SIZES, "repeats": REPEATS, "estimator": "exact W1 between two samples"})
    print("written", path)


def figure():
    from lab.plotstyle import plt, save, tex_number

    nums = results.load("chapter2", "rates")["numbers"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    styles = ["-", (0, (6, 4)), (0, (1.5, 3))]
    markers = ["o", "s", "^"]
    short = {"curve in R^16 (m = 1)": r"curve in $\mathbb{R}^{16}$", "Gaussian in R^16 (m = 16)": r"Gaussian in $\mathbb{R}^{16}$",
             "MNIST digits (p = 784)": "MNIST digits"}
    offsets = [1.0, 1.10, 0.89]  # vertical nudges so that the two flat lines keep separate labels
    for (name, v), ls, mk, dy in zip(nums.items(), styles, markers, offsets):
        sizes, w = np.array(v["sizes"]), np.array(v["mean_w1"])
        rel = w / w[0]
        ax.plot(sizes, rel, color="black", lw=1.8, ls=ls, marker=mk, ms=4.5, mfc="white")
        ax.text(sizes[-1] * 1.12, rel[-1] * dy, f"{short[name]}, slope ${v['slope']:.3f}$", va="center", fontsize=8.5)
    ref = np.array([SIZES[0], SIZES[-1]], float)
    ax.plot(ref, (ref / ref[0]) ** -0.5, color="black", lw=0.8, alpha=0.45)
    ax.text(110, (110 / ref[0]) ** -0.5 * 0.86, "slope $-1/2$", fontsize=8, alpha=0.7, rotation=-24)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(SIZES[0] * 0.8, SIZES[-1] * 12)
    ticks = [0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    ax.set_yticks(ticks, [tex_number(t) for t in ticks])
    ax.set_yticks([], minor=True)
    ax.set_xticks(SIZES, [tex_number(n) for n in SIZES])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("$W_1$ between two samples\n(relative to $n = 50$)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = FIGURES / "chapter2" / "rates.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save(fig, out, transparent=True)
    print("written", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "figure"])
    run() if ap.parse_args().cmd == "run" else figure()
