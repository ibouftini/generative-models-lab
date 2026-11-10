"""Memorization on the unit circle, measured with training loss, test loss and copies.

  training loss  MMD^2 between the generator and the n training points (the objective)
  test loss      the same MMD^2 between the generator and 2,000 fresh points of the circle
  copies         share of samples closer to the training set than 99% of held-out points (lab.metrics.memorization)

Kernel: a fixed mixture of Gaussian kernels, bandwidths 0.02, 0.05, 0.1, 0.2, 0.5, the same for every run and every
evaluation. Diagnostic far_share: share of samples farther than 0.25 from every held-out point. Estimator: the generated
samples enter as a U-statistic (no diagonal terms, unbiased), the reference set as the empirical distribution it is
(V-statistic). The test loss therefore estimates MMD^2(P_g, P*) up to a bias of order 1/2,000 from the held-out set,
reported as test_loss_truth (the test loss of 2,000 fresh circle points).

Training: Adam, 1,024 generated samples per step against all n training points, 100,000 steps: learning rate 1e-3 held
for 90,000 steps, then decayed linearly to 0. plateau_change is the relative change of the training loss on fixed
evaluation samples between steps 70,000 and 90,000. Evaluation every 2,000 steps and at the end.

Data: the training set of seed s is circle(n, seed=s), so each seed draws its own points. The held-out set is
circle(2000, seed=1000) and the reference sample of the truth is circle(2000, seed=1001). train_loss_truth is the training
loss of that reference sample, the score of a perfect generator against the same n training points.

Sweeps (3 seeds each, latent dimension k = 2):
  A  n = 12, widths 4, 16, 64, 256, 1024
  B  width 1024, n = 12, 25, 50, 100, 200
  C  width 4 and width 16, n = 200
  bandwidth check: sweep A, seed 0, kernel scale 0.5 and 2
Latent dimension k = 1: widths 16 and 1024, n = 12 and 200, 3 seeds, everything else unchanged. A generator from R^1 to
R^2 has no density and the MMD needs none.

Both sweeps resume after an interruption, and one run does not depend on the others.

usage:
  python -m chapter1.memorization run      # sweeps -> results/chapter1/memorization-sweeps.json, about 5.5 h on an RTX 3050
  python -m chapter1.memorization latent   # k = 1  -> results/chapter1/memorization-sweeps-k1.json, about 1.5 h
  python -m chapter1.memorization mnist    # the three numbers for the MNIST generator of the opening figure
  python -m chapter1.memorization geometry # median distance to the circle and copy radius, from the saved samples
  python -m chapter1.memorization figures  # the figures of the chapter, from the saved samples and results
"""
import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
import time

import torch
import torch.nn as nn

from lab import CHECKPOINTS, FIGURES, data, metrics, results

BANDWIDTHS = (0.02, 0.05, 0.1, 0.2, 0.5)
BATCH, EVAL, HELDOUT, DATA_SEED = 1024, 2000, 2000, 0
STEPS, DECAY, EVERY = 100_000, 10_000, 2_000
HELDOUT_SEED, TRUTH_SEED = 1000, 1001
LR = {4: 1e-3, 16: 1e-3, 64: 1e-3, 256: 1e-3, 1024: 1e-3}
SEEDS = (0, 1, 2)
SAMPLES = CHECKPOINTS / "memorization_sweeps_samples.pt"
device = metrics.DEVICE


def generator(width, k=2):
    return nn.Sequential(nn.Linear(k, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))


def kernel(a, b, scale=1.0):
    d2 = torch.cdist(a, b).pow(2)
    return sum(torch.exp(-d2 / (2 * (scale * h) ** 2)) for h in BANDWIDTHS)


def mmd2(x, y, scale=1.0):
    """MMD^2 between the distribution of the samples x and the empirical distribution of the reference set y.
    x enters as a U-statistic (off-diagonal mean), y as the empirical distribution (full mean)."""
    m = len(x)
    kxx = kernel(x, x, scale)
    xx = (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
    return xx + kernel(y, y, scale).mean() - 2 * kernel(x, y, scale).mean()


def key(width, n, seed, scale=1.0):
    return f"w{width}-n{n}-s{seed}" + ("" if scale == 1.0 else f"-k{scale:g}")


@torch.no_grad()
def evaluate(g, train, heldout, scale, k=2):
    torch.manual_seed(DATA_SEED + 2)
    s = g(torch.randn(EVAL, k, device=device))
    m = metrics.memorization(s.cpu(), train.cpu(), heldout.cpu())
    far = (metrics.nearest_distance(s.cpu(), heldout.cpu()) > 0.25).float().mean().item()
    return s.cpu(), {"train_loss": mmd2(s, train, scale).item(), "test_loss": mmd2(s, heldout, scale).item(),
                     "copy_share": m["copy_share"], "median_ratio": m["median_ratio"], "far_share": far}


def train_one(width, n, seed, heldout, truth, scale=1.0, lr=None, k=2):
    train = data.circle(n, seed=seed).to(device)
    torch.manual_seed(seed)
    g = generator(width, k).to(device)
    lr = lr or LR[width]
    opt = torch.optim.Adam(g.parameters(), lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (STEPS - t) / DECAY))
    losses, curve, evals = [], [], []
    start = time.time()
    for step in range(1, STEPS + 1):
        loss = mmd2(g(torch.randn(BATCH, k, device=device)), train, scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 100 == 0:
            curve.append([step, sum(losses[-100:]) / 100])
        if step % EVERY == 0:
            with torch.no_grad():
                torch.manual_seed(DATA_SEED + 2)
                s = g(torch.randn(EVAL, k, device=device))
                evals.append([step, mmd2(s, train, scale).item(), mmd2(s, heldout, scale).item()])
            torch.manual_seed(seed * 1_000_003 + step)          # keep training noise independent of evaluation
    samples, nums = evaluate(g, train, heldout, scale, k)
    at = {e[0]: e[1] for e in evals}
    nums.update({"width": width, "n_train": n, "seed": seed, "kernel_scale": scale, "lr": lr, "steps": STEPS, "latent": k,
                 "plateau_change": (at[70_000] - at[90_000]) / at[70_000],
                 "train_loss_truth": mmd2(truth, train, scale).item(), "test_loss_truth": mmd2(truth, heldout, scale).item(),
                 "minutes": round((time.time() - start) / 60, 2), "params": sum(p.numel() for p in g.parameters())})
    return nums, samples, train.cpu(), {"train_minibatch": curve, "eval": evals}


def configs():
    runs = []
    for seed in SEEDS:
        runs += [("A", w, 12, seed, 1.0) for w in (4, 16, 64, 256, 1024)]
        runs += [("B", 1024, n, seed, 1.0) for n in (25, 50, 100, 200)]       # n = 12 is shared with sweep A
        runs += [("C", w, 200, seed, 1.0) for w in (4, 16)]
    runs += [("bandwidth", w, 12, 0, sc) for sc in (0.5, 2.0) for w in (4, 16, 64, 256, 1024)]
    return runs


def run():
    heldout = data.circle(HELDOUT, seed=HELDOUT_SEED).to(device)
    truth = data.circle(EVAL, seed=TRUTH_SEED).to(device)
    numbers, samples, trains, curves = {}, {}, {}, {}
    if SAMPLES.exists():                                        # resume after an interruption
        saved = torch.load(SAMPLES)
        numbers, samples, trains, curves = saved["numbers"], saved["samples"], saved["trains"], saved["curves"]
    total = time.time()
    for sweep, w, n, seed, sc in configs():
        k = key(w, n, seed, sc)
        if k in numbers:
            continue
        nums, s, tr, cv = train_one(w, n, seed, heldout, truth, sc)
        nums["sweep"] = sweep
        numbers[k], samples[k], trains[f"n{n}-s{seed}"], curves[k] = nums, s, tr, cv
        print(f"{k:22s} train {nums['train_loss']:.4f} (truth {nums['train_loss_truth']:.4f})  test {nums['test_loss']:.4f}  "
              f"copies {nums['copy_share']:.4f}  far {nums['far_share']:.3f}  plateau {nums['plateau_change']:+.1%}  "
              f"{nums['minutes']:.1f} min", flush=True)
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        torch.save({"numbers": numbers, "samples": samples, "trains": trains, "curves": curves}, SAMPLES)
    out = results.save("chapter1", "memorization-sweeps", numbers, seed=None, script=__file__,
                       config={"bandwidths": BANDWIDTHS, "estimator": "U-statistic on samples, V-statistic on the reference set",
                               "batch": BATCH, "eval_samples": EVAL, "heldout": f"circle({HELDOUT}, seed={HELDOUT_SEED})",
                               "truth_reference": f"circle({EVAL}, seed={TRUTH_SEED})", "training_sets": "circle(n, seed=s)",
                               "lr": LR, "schedule": f"{STEPS} steps, constant then linear decay to 0 over the last {DECAY}",
                               "plateau_change": "relative change of the evaluation training loss from step 70,000 to 90,000",
                               "total_minutes": round((time.time() - total) / 60, 1)})
    (out.parent / "memorization-sweeps-curves.json").write_text(json.dumps(curves))
    print("written", out)


def latent():
    """The k = 1 runs: same training and evaluation as run(), generator input of dimension 1."""
    heldout = data.circle(HELDOUT, seed=HELDOUT_SEED).to(device)
    truth = data.circle(EVAL, seed=TRUTH_SEED).to(device)
    path = CHECKPOINTS / "memorization_sweeps_k1_samples.pt"
    numbers, samples, trains, curves = {}, {}, {}, {}
    if path.exists():                                           # resume after an interruption
        saved = torch.load(path)
        numbers, samples, trains, curves = saved["numbers"], saved["samples"], saved["trains"], saved["curves"]
    torch.manual_seed(DATA_SEED + 2)                            # the evaluation codes of every run, in increasing order
    order = torch.randn(EVAL, 1, device=device).cpu().squeeze(1).argsort()
    total = time.time()
    for seed in SEEDS:
        for n in (12, 200):
            for w in (16, 1024):
                k = key(w, n, seed) + "-z1"
                if k in numbers:
                    continue
                nums, s, tr, cv = train_one(w, n, seed, heldout, truth, k=1)
                numbers[k], samples[k], trains[f"n{n}-s{seed}"], curves[k] = nums, s, tr, cv
                print(f"{k:24s} train {nums['train_loss']:.4f} (truth {nums['train_loss_truth']:.4f})  "
                      f"test {nums['test_loss']:.4f}  copies {nums['copy_share']:.4f}  far {nums['far_share']:.3f}  "
                      f"plateau {nums['plateau_change']:+.1%}  {nums['minutes']:.1f} min", flush=True)
                CHECKPOINTS.mkdir(parents=True, exist_ok=True)
                torch.save({"numbers": numbers, "samples": samples, "trains": trains, "curves": curves, "order": order}, path)
    out = results.save("chapter1", "memorization-sweeps-k1", numbers, seed=None, script=__file__,
                       config={"latent_dimension": 1, "widths": [16, 1024], "n_train": [12, 200], "seeds": list(SEEDS),
                               "same_as": "memorization-sweeps.json (kernel, estimator, schedule, data, evaluation)",
                               "total_minutes": round((time.time() - total) / 60, 1)})
    (out.parent / "memorization-sweeps-k1-curves.json").write_text(json.dumps(curves))
    print("written", out)


def mnist():
    """Training loss, test loss and copies for the MNIST WGAN-GP of the opening figure. The kernel is the one
    of the circle, rescaled by the ratio of median pairwise distances (MNIST training set / uniform circle), so that
    the bandwidths sit at the same place relative to the spread of the data."""
    from chapter1 import generator as w
    G, ckpt = w.load_generator()
    train = w.load_mnist(True).flatten(1).to(device)
    test = w.load_mnist(False).flatten(1).to(device)
    torch.manual_seed(3)
    with torch.no_grad():
        s = torch.cat([(G(torch.randn(500, w.Z_DIM, device=device)) + 1) / 2 for _ in range(EVAL // 500)]).clamp(0, 1).flatten(1)
    g = torch.Generator().manual_seed(0)
    sub = train[torch.randperm(len(train), generator=g)[:4000].to(device)]
    med_mnist = torch.cdist(sub[:2000], sub[2000:]).median().item()
    circ = data.circle(4000, seed=5).to(device)
    med_circle = torch.cdist(circ[:2000], circ[2000:]).median().item()
    scale = med_mnist / med_circle

    def kmean(a, b, chunk=2000):
        tot = 0.0
        for i in range(0, len(a), chunk):
            for j in range(0, len(b), chunk):
                tot += kernel(a[i:i + chunk], b[j:j + chunk], scale).sum().item()
        return tot / (len(a) * len(b))

    # A reference set enters as an empirical distribution, so its size biases the loss upward by about the mean
    # kernel diagonal over its size (60,000 training images versus 10,000 test images). To compare like with like,
    # 2,000 real test images go through the same computation as the 2,000 generated ones, and the test loss uses
    # the 8,000 remaining test images.
    perm = torch.randperm(len(test), generator=g).to(device)
    real, test_ref = test[perm[:EVAL]], test[perm[EVAL:]]
    yy = kmean(train, train), kmean(test_ref, test_ref)

    def losses(x):
        m = len(x)
        kxx = kernel(x, x, scale)
        xx = (kxx.sum() - kxx.diagonal().sum()).item() / (m * (m - 1))
        return xx + yy[0] - 2 * kmean(x, train), xx + yy[1] - 2 * kmean(x, test_ref)

    nums = {}
    for name, x in (("generator", s), ("real_test_images", real)):
        tr, te = losses(x)
        mem = metrics.memorization(x.cpu(), train.cpu(), test_ref.cpu())
        nums[name] = {"train_loss": tr, "test_loss": te, "copy_share": mem["copy_share"], "median_ratio": mem["median_ratio"]}
    nums.update({"kernel_scale": scale, "median_pairwise_mnist": med_mnist, "median_pairwise_circle": med_circle})
    path = results.save("chapter1", "mnist-wgan-losses", nums, seed=3, script=__file__,
                        config={"samples": EVAL, "space": "pixels in [0, 1]", "train": "60,000 MNIST training images",
                                "test": "8,000 MNIST test images (the other 2,000 are the real-image reference)",
                                "bandwidths": [scale * h for h in BANDWIDTHS], "generator_steps": ckpt["it"]})
    print(nums, "written", path)


def geometry():
    """Distance of the samples to the circle and radius of the copy discs, from the saved samples (2026-11-10).
    circle_distance_median is the median of | ||s|| - 1 | over the 2,000 samples of a run. copy_radius is the copy
    threshold of lab.metrics.memorization, the 1% quantile of the distance from a held-out point to its nearest
    training point, so it depends on n and on the draw of the training points."""
    heldout = data.circle(HELDOUT, seed=HELDOUT_SEED)
    for path, name in ((SAMPLES, "memorization-sweeps"),
                       (CHECKPOINTS / "memorization_sweeps_k1_samples.pt", "memorization-sweeps-k1")):
        saved = torch.load(path)
        res_path = results.RESULTS / "chapter1" / f"{name}.json"
        res = json.loads(res_path.read_text())
        for k, nums in saved["numbers"].items():
            s, train = saved["samples"][k], saved["trains"][f"n{nums['n_train']}-s{nums['seed']}"]
            dist = (s.norm(dim=1) - 1).abs().median().item()
            radius = torch.quantile(metrics.nearest_distance(heldout, train), 0.01).item()
            res["numbers"][k].update({"circle_distance_median": dist, "copy_radius": radius})
            print(f"{k:24s} median distance to circle {dist:.4f}  copy radius {radius:.5f}")
        res.setdefault("config", {})["geometry"] = geometry.__doc__.split(".")[0] + " (2026-11-10)."
        res_path.write_text(json.dumps(res, indent=2))



def figures():
    """The figures of the chapter, in black ink on a transparent background so that {.ink-image} can invert them in
    the dark theme. Series are told apart by line style, marker and direct labels, never by colour."""
    from lab.plotstyle import plt, save, tex_number
    from matplotlib.patches import FancyBboxPatch
    import numpy as np

    out = FIGURES / "chapter1"
    out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(SAMPLES)
    numbers, samples, trains = saved["numbers"], saved["samples"], saved["trains"]
    ink, soft = "#1f1f1f", "#6b6c6e"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": ink, "xtick.color": ink, "ytick.color": ink,
                         "axes.labelcolor": ink, "text.color": ink})

    # Samples: widths 16 and 1024 on 12 and on 200 points. The post leads with one latent dimension (k = 1, draw 2,
    # seed 1; draw 1 holds the narrow run that leaked mass) in an interactive figure (interactive/memorization-samples.js,
    # two tabs) fed by interactive/memorization-samples.json, with a 2 x 2 static fallback (RSS, print, no JavaScript).
    # For k = 1 the image of each generator is a curve. The weights are not saved, so the curve is drawn through the
    # samples in the order of their latent values, regenerated from the evaluation seed (consecutive samples on 12
    # points are a median 0.0005 apart). The runs with two latent dimensions (seed 0) get the same interactive figure
    # without the curve (interactive/memorization-samples-k2.js and .json), with its own 2 x 2 static fallback.
    # Copies are counted over all 2,000 evaluation samples with the threshold of metrics.memorization, per training
    # point (the training point nearest to each copy), so each panel's total is the copies cell of its run. The copy
    # distance (0.0024 on 12 points) is far below a pixel: copies are drawn as one disc per training point, area
    # proportional to their number.
    panels = [(16, 12), (1024, 12), (16, 200), (1024, 200)]
    heldout = data.circle(HELDOUT, seed=HELDOUT_SEED)
    k1 = torch.load(CHECKPOINTS / "memorization_sweeps_k1_samples.pt")
    torch.manual_seed(DATA_SEED + 2)
    latents = torch.randn(EVAL, 1, device=device).cpu().squeeze(1)
    shown = 1200
    order = latents[:shown].argsort()                  # the first 1,200 samples, in the order of their latent values

    def copies_of(smp, tr, share):
        threshold = torch.quantile(metrics.nearest_distance(heldout, tr), 0.01)
        dist, nearest = torch.cdist(smp, tr, compute_mode="donot_use_mm_for_euclid_dist").min(1)
        counts = torch.bincount(nearest[dist < threshold], minlength=len(tr))
        assert abs(counts.sum().item() / len(smp) - share) < 1e-6   # shares are stored as float32
        percent = Decimal(100 * counts.sum().item()) / len(smp)
        return {"counts": counts.tolist(), "total": counts.sum().item(), "of": len(smp),
                "percent": str(percent.quantize(Decimal("0.1"), ROUND_HALF_UP).normalize())}

    def runs_of(samples, trains, numbers, seed, suffix, curve):
        runs = {}
        for w, n in panels:
            k = key(w, n, seed) + suffix
            smp, tr = samples[k], trains[f"n{n}-s{seed}"]
            runs[w, n] = {"train": tr, "copies": copies_of(smp, tr, numbers[k]["copy_share"]),
                          "samples": smp[:shown][order] if curve else smp[:shown], "curve": curve}
        return runs

    def grid(runs, path, title):
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 9.2), dpi=150)
        for ax, (w, n) in zip(axes.flat, panels):
            r = runs[w, n]
            smp, tr = r["samples"], r["train"]
            if r["curve"]:
                ax.plot(smp[:, 0], smp[:, 1], color=soft, lw=0.6, alpha=0.6)
            smp = smp[(smp.abs() < 1.3).all(1)]  # inside the window, as in the interactive version
            ax.scatter(smp[:, 0], smp[:, 1], s=2.5, color=ink, alpha=0.35, lw=0)
            if n <= 12:
                ax.scatter(tr[:, 0], tr[:, 1], s=[3.6 * c for c in r["copies"]["counts"]], color=ink, lw=0, zorder=3)
                ax.scatter(tr[:, 0], tr[:, 1], s=210, facecolor="none", edgecolor=ink, lw=1.1, zorder=3)
            else:
                t = torch.atan2(tr[:, 1], tr[:, 0])
                ax.plot(torch.stack([1.07 * t.cos(), 1.15 * t.cos()]), torch.stack([1.07 * t.sin(), 1.15 * t.sin()]),
                        color=ink, lw=0.6)
            ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls=(0, (2, 3)), lw=0.9, color=ink))
            # rounded frame, as in the interactive version and the schematics of the post
            ax.add_patch(FancyBboxPatch((0.012, 0.012), 0.976, 0.976, boxstyle="round,pad=0,rounding_size=0.035",
                                        transform=ax.transAxes, fill=False, lw=0.8, ec=ink, alpha=0.3, clip_on=False))
            ax.set_title(title(w, n), fontsize=12)
            ax.text(0.5, -0.05, f"Copies: {r['copies']['percent']}% of samples", transform=ax.transAxes, ha="center",
                    va="top", fontsize=11, color=soft)
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.3)                     # room for the copies line under the top panels
        save(fig, path, transparent=True, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    k1_numbers = json.loads((results.RESULTS / "chapter1" / "memorization-sweeps-k1.json").read_text())["numbers"]
    one = runs_of(k1["samples"], k1["trains"], k1_numbers, 1, "-z1", curve=True)
    grid(one, out / "memorization-samples.png", lambda w, n: f"Width {w}, {n} points")
    two = runs_of(samples, trains, numbers, 0, "", curve=False)
    grid(two, out / "memorization-samples-k2.png", lambda w, n: f"Width {w}, {n} points")
    inter = out / "interactive"
    inter.mkdir(exist_ok=True)
    payload = {"panels": [{"width": w, "n": n, "curve": True,
                           "train": [[round(v, 3) for v in q] for q in one[w, n]["train"].tolist()],
                           "copies": one[w, n]["copies"],
                           "samples": [[round(v, 3) for v in q] for q in one[w, n]["samples"].tolist()]}
                          for w, n in panels]}
    (inter / "memorization-samples.json").write_text(json.dumps(payload, separators=(",", ":")))
    # the runs with two latent dimensions get the same interactive figure (interactive/memorization-samples-k2.js)
    payload = {"panels": [{"width": w, "n": n, "curve": False,
                           "train": [[round(v, 3) for v in q] for q in two[w, n]["train"].tolist()],
                           "copies": two[w, n]["copies"],
                           "samples": [[round(v, 3) for v in q] for q in two[w, n]["samples"].tolist()]}
                          for w, n in panels]}
    (inter / "memorization-samples-k2.json").write_text(json.dumps(payload, separators=(",", ":")))

    # Training and test loss over training steps, width 1024 on 12 points with one latent dimension, one panel per seed
    k1_curves = json.loads((results.RESULTS / "chapter1" / "memorization-sweeps-k1-curves.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.3), dpi=200, sharex=True)   # narrow canvas: text stays legible in the column
    for ax, sd in zip(axes, SEEDS):
        ev = np.array(k1_curves[key(1024, 12, sd) + "-z1"]["eval"])
        ax.plot(ev[:, 0], ev[:, 1], color=ink, lw=1.8)
        ax.plot(ev[:, 0], ev[:, 2], color=ink, lw=1.8, ls="--")
        truth = k1_numbers[key(1024, 12, sd) + "-z1"]["train_loss_truth"]
        ax.axhline(truth, color=soft, lw=1, ls=":")
        ax.annotate("true circle, training loss", (STEPS, truth), xytext=(0, 4), textcoords="offset points",
                    ha="right", color=soft, fontsize=10)
        if sd == SEEDS[0]:                       # label each line once, where the two are far apart
            at = int(np.searchsorted(ev[:, 0], 20_000))
            ax.annotate("test loss", (ev[at, 0], ev[at, 2]), xytext=(0, 10), textcoords="offset points", ha="center",
                        fontsize=11)
            low = int(np.searchsorted(ev[:, 0], 4_000))
            ax.annotate("training loss", (ev[low, 0], ev[low, 1]), xytext=(0, -26), textcoords="offset points",
                        ha="center", fontsize=11)
        ax.axvline(6000, color=soft, lw=1, ls=(0, (2, 3)))
        ax.set_xscale("log")
        ax.set_ylim(0, max(ev[:, 2].max(), truth) * 1.18)
        ax.set_title(f"Draw {sd + 1}", loc="left", fontsize=11)
        ax.set_xlabel("Training steps", fontsize=11)
    axes[0].set_ylabel(r"$\mathrm{MMD}^2$", fontsize=11)
    axes[1].annotate("6,000\nsteps", (6000, 0), xytext=(-4, 4), textcoords="offset points",
                     ha="right", va="bottom", color=soft, fontsize=10)
    fig.tight_layout()
    save(fig, out / "memorization-steps.png", transparent=True)
    plt.close(fig)

    # Test loss and copies against the number of training points, width 1024
    ns = [12, 25, 50, 100, 200]
    vals = {f: np.array([[numbers[key(1024, n, sd)][f] for sd in SEEDS] for n in ns]) for f in ("test_loss", "copy_share")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 3.6), dpi=150)
    for a, f, scale, ylabel in ((a1, "test_loss", 1, r"Test loss ($\mathrm{MMD}^2$)"), (a2, "copy_share", 100, "Copies (% of samples)")):
        v = scale * vals[f]
        a.fill_between(ns, v.min(1), v.max(1), color=ink, alpha=0.12, lw=0)
        a.plot(ns, v.mean(1), color=ink, lw=1.8, marker="o", ms=5)
        a.set_xscale("log")
        a.set_xticks(ns, [tex_number(n) for n in ns])
        a.minorticks_off()
        a.set_xlabel("Training points")
        a.set_ylabel(ylabel)
        a.set_ylim(bottom=0)
    a2.axhline(1, color=soft, lw=1, ls="--")
    a2.annotate("fresh points of the circle: 1%", (ns[-1], 1), xytext=(0, 5), textcoords="offset points", ha="right",
                color=soft, fontsize=8)
    floor = numbers[key(1024, 12, 0)]["test_loss_truth"]
    a1.axhline(floor, color=soft, lw=1, ls="--")
    a1.annotate(f"fresh points of the circle: {floor:.3f}", (ns[0], floor), xytext=(0, 5), textcoords="offset points",
                ha="left", color=soft, fontsize=8)
    fig.tight_layout()
    save(fig, out / "memorization-data.png", transparent=True)
    plt.close(fig)
    print("written", out / "memorization-samples.png", out / "memorization-samples-k2.png",
          out / "memorization-steps.png", out / "memorization-data.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "latent", "mnist", "geometry", "figures"])
    {"run": run, "latent": latent, "mnist": mnist, "geometry": geometry, "figures": figures}[ap.parse_args().cmd]()
