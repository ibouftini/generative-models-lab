"""The two directions of KL. A single Gaussian q = N(mu, sigma^2) fitted to a two-mode target
p = 0.5 N(-2, 0.5^2) + 0.5 N(2, 0.5^2).

  forward KL(p || q), the direction of maximum likelihood: the minimizer matches the mean and variance of p,
                     mu = 0, sigma^2 = 0.25 + 4 = 4.25, mode covering
  reverse KL(q || p): no closed form; minimized numerically from mu = 1, mode seeking

Integrals are computed on a fine grid.

usage: python -m chapter2.kl_directions run      # writes results/chapter2/kl-directions.json and the figure
       python -m chapter2.kl_directions figure   # the figure only
"""
import math
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from lab import FIGURES, results
from lab.plotstyle import plt, save

X = np.linspace(-12, 12, 48001)
DX = X[1] - X[0]
MEANS, STD = np.array([-2.0, 2.0]), 0.5


def log_p(x):
    return logsumexp(norm.logpdf(x[:, None], MEANS, STD) + np.log(0.5), axis=1)


def kl(log_a, log_b):
    a = np.exp(log_a)
    return float(np.sum(a * (log_a - log_b)) * DX)


lp = log_p(X)
fwd_mu, fwd_sigma = 0.0, math.sqrt(STD ** 2 + 4.0)  # moment matching (mean 0, variance 0.25 + 4)
# numerical check of the closed form: minimize KL(p || q) directly
res_f = minimize(lambda t: kl(lp, norm.logpdf(X, t[0], math.exp(t[1]))), [0.5, 0.0], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-9})
res_r = minimize(lambda t: kl(norm.logpdf(X, t[0], math.exp(t[1])), lp), [1.0, 0.0], method="Nelder-Mead",
                 options={"xatol": 1e-6, "fatol": 1e-9})
rev_mu, rev_sigma = float(res_r.x[0]), float(math.exp(res_r.x[1]))
lq_f, lq_r = norm.logpdf(X, fwd_mu, fwd_sigma), norm.logpdf(X, rev_mu, rev_sigma)
numbers = {
    "forward_fit": {"mu": fwd_mu, "sigma": fwd_sigma, "numerical_mu": float(res_f.x[0]),
                    "numerical_sigma": float(math.exp(res_f.x[1])),
                    "forward_kl": kl(lp, lq_f), "reverse_kl": kl(lq_f, lp)},
    "reverse_fit": {"mu": rev_mu, "sigma": rev_sigma, "forward_kl": kl(lp, lq_r), "reverse_kl": kl(lq_r, lp)},
}
for k, v in numbers.items():
    print(k, {a: round(b, 4) for a, b in v.items()})
if sys.argv[1:] != ["figure"]:
    path = results.save("chapter2", "kl-directions", numbers, script=__file__,
                        config={"target": "0.5 N(-2, 0.5^2) + 0.5 N(2, 0.5^2)", "grid": [-12, 12, 48001],
                                "reverse_start": [1.0, 1.0]})
    print("written", path)

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
ax.fill_between(X, np.exp(lp), color="black", alpha=0.08, lw=0)
ax.plot(X, np.exp(lp), color="black", lw=2)
ax.plot(X, np.exp(lq_f), color="black", lw=1.6, ls=(0, (6, 4)))
ax.plot(X, np.exp(lq_r), color="black", lw=1.6, ls=(0, (1.5, 3)))
ax.text(-2, 0.43, "Target", ha="center")
ax.text(0, 0.215, "Forward $\\mathrm{KL}$ fit\n(mode covering)", ha="center", fontsize=9)
ax.text(rev_mu + 0.95, 0.62, "Reverse $\\mathrm{KL}$ fit\n(mode seeking)", ha="left", fontsize=9)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 0.85)
ax.set_yticks([])
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
out = FIGURES / "chapter2" / "kl-directions.png"
out.parent.mkdir(parents=True, exist_ok=True)
save(fig, out, transparent=True)
print("written", out)
