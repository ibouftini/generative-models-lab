"""The VAE experiments of Chapter 3 and their figures.

  ring        A VAE on the ring of 8 Gaussians (2D data, 2D latent, Gaussian decoder). Maximum likelihood covers every
              mode; does it also put mass between them? -> results/chapter3/ring.json, the samples, figures/ring-vae.png
  beta        The beta sweep on MNIST (8D latent): loss = reconstruction / (2 sigma^2_dec) + beta * KL. Rate (KL per
              image, nats) and distortion (squared error per image) on the test set.
              -> results/chapter3/beta.json, figures/rate-distortion.png
  holes       Prior samples of the 8D VAE that the shared classifier does not recognize.
              -> results/chapter3/holes.json, figures/holes.png
  export      The 2D VAE decoder for the latent explorer of the chapter: int8 weights with per-row float16 scales, and
              the codes of 2,000 test digits. -> figures/chapter3/interactive/
  linear      The collapse rule of the linear model (a direction is used when its data variance exceeds
              beta * sigma^2_dec) applied to the principal variances of MNIST, against the active units of the beta sweep.
              -> results/chapter3/linear-collapse.json
  calibrate   The optimal shared decoder variance of the 8D VAE, a VAE trained with the variance set to its optimum at
              every step, and blur measures of both. -> results/chapter3/calibrated.json
  estimators  Importance sampling of p(x) with the prior and with the encoder as proposal, and the spread of the
              one-sample ELBO estimate, on test digits. -> results/chapter3/estimators.json
  blur        The blur measure of calibrate for the plain 8D autoencoder and the 8D VAE whose KL term weighs 100 times
              more, next to the 8D VAE. -> results/chapter3/blur.json

  ring-figure, beta-figure, holes-figure redraw a figure from saved files, without training or writing results.

usage: python -m chapter3.experiments <command>
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn

from chapter3.autoencoder import AE, OUT
from chapter3.vae import S2, Model, load, mlp
from lab import CHECKPOINTS, FIGURES, data, metrics, results
from lab.data import mnist
from lab.plotstyle import plt, save

FIG = FIGURES / "chapter3"
RING_SAMPLES = CHECKPOINTS / "ring_samples.pt"
device = metrics.DEVICE
BETAS = [0.1, 0.3, 1, 3, 10, 30, 100]
SEEDED = (0.1, 0.3, 1)  # betas repeated with seeds 1 and 2, to measure the run-to-run spread


# ---------------------------------------------------------------- ring of 8 Gaussians
class RingVAE(nn.Module):
    def __init__(self, k=2, h=256, s2=1e-3):
        super().__init__()
        self.s2 = s2
        self.enc = mlp([2, h, h, 2 * k], nn.Identity())
        self.dec = mlp([k, h, h, 2], nn.Identity())

    def loss(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        rec = ((self.dec(z) - x) ** 2).sum(1) / (2 * self.s2)
        kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)
        return (rec + kl).mean()


def ring():
    torch.manual_seed(0)
    x = data.ring(20000, seed=0).to(device)
    model = RingVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for step in range(20000):
        loss = model.loss(x[torch.randint(len(x), (512,), device=device)])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        torch.manual_seed(1)
        samples = model.dec(torch.randn(5000, 2, device=device)).cpu()
    real = metrics.ring_modes(data.ring(5000, seed=2))
    vae = metrics.ring_modes(samples)
    numbers = {"vae": vae, "real": real, "final_loss": loss.item()}
    path = results.save("chapter3", "ring", numbers, seed=0, script=__file__,
                        config={"steps": 20000, "batch": 512, "latent": 2, "hidden": 256, "decoder_variance": 1e-3})
    print(f"VAE: {vae['modes_covered']} modes, high quality {vae['high_quality']:.3f}; real: {real['high_quality']:.3f}")
    print("written", path)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save(samples, RING_SAMPLES)
    ring_figure()


def ring_figure():
    samples = torch.load(RING_SAMPLES)
    vae = results.load("chapter3", "ring")["numbers"]["vae"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    for ax, pts, title in ((axes[0], data.ring(2000, seed=3), "Data"),
                           (axes[1], samples[:2000], f"VAE samples ({100 * vae['high_quality']:.0f}% near a mode)")):
        ax.scatter(pts[:, 0], pts[:, 1], s=2, color="black", alpha=0.35, lw=0)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "ring-vae.png"
    save(fig, out, transparent=True)
    print("written", out)


# ---------------------------------------------------------------- beta sweep
def beta_run(b, seed, X, xt, epochs):
    torch.manual_seed(seed)
    model = Model(True, 8, (256, 64)).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), device=device).split(128):
            mu, logvar = model.encode(X[idx])
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            rec = ((model.dec(z) - X[idx]) ** 2).sum(1) / (2 * S2)
            kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)
            loss = (rec + b * kl).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(xt.to(device))
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        se = ((model.dec(z) - xt.to(device)) ** 2).sum(1).mean().item()
        rate = (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)).mean().item()
        active = int((mu.var(0) > 0.01).sum().item())
    # distortion in nats: -E log N(x; g(z), S2 I) = SE / (2 S2) + (p / 2) log(2 pi S2), with p = 784
    nats = se / (2 * S2) + 392 * math.log(2 * math.pi * S2)
    return {"rate_nats": rate, "distortion": se, "distortion_nats": nats, "active_units": active}


def beta(epochs=10):
    x, _ = mnist(True)
    xt, _ = mnist(False)
    X = x.to(device)
    numbers = {}
    for b in BETAS:
        numbers[str(b)] = beta_run(b, 0, X, xt, epochs)
        n = numbers[str(b)]
        print(f"beta {b:6g}  rate {n['rate_nats']:7.2f} nats  distortion {n['distortion']:6.2f} "
              f"({n['distortion_nats']:8.1f} nats)  active units {n['active_units']}", flush=True)
        if b in SEEDED:
            n["seeds"] = {str(s): beta_run(b, s, X, xt, epochs) for s in (1, 2)}
            print("   seeds 1, 2 distortion", [round(v["distortion"], 2) for v in n["seeds"].values()], flush=True)
    path = results.save("chapter3", "beta", numbers, seed=0, script=__file__,
                        config={"betas": BETAS, "seeded": SEEDED, "epochs": epochs, "latent": 8, "decoder_variance": S2,
                                "active_unit_threshold": "variance of the encoder mean over the test set > 0.01"})
    print("written", path)
    beta_figure(numbers)


def beta_figure(numbers=None):
    """The rate-distortion figure, from the numbers of beta() (read from results/chapter3/beta.json when not given)."""
    numbers = numbers or results.load("chapter3", "beta")["numbers"]
    fig, ax = plt.subplots(figsize=(6.4, 4), dpi=160)
    r = [numbers[str(b)]["rate_nats"] for b in BETAS]
    d = [numbers[str(b)]["distortion"] for b in BETAS]
    ax.plot(r, d, color="black", lw=1.6, marker="o", ms=4)
    for b, ri, di in zip(BETAS, r, d):
        ax.annotate(rf"$\beta = {b:g}$", (ri, di), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel(r"Rate, $\mathrm{KL}$ to the prior per image (nats)")
    ax.set_ylabel("Distortion, squared error per image")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "rate-distortion.png"
    save(fig, out, transparent=True)
    print("written", out)


# ---------------------------------------------------------------- holes
def holes(n=1000, show=8, write=True):
    clf = metrics.MnistClassifier.load()
    dec = load("mnist_vae8.pt").dec
    torch.manual_seed(1)  # the codes of chapter3.vae_numbers
    with torch.no_grad():
        imgs = dec(torch.randn(n, 8))
    conf = clf.probabilities(imgs).max(1).values
    order = conf.argsort()
    numbers = {"prior_samples": n, "share_below_0.9": (conf <= 0.9).float().mean().item(),
               "share_below_0.5": (conf <= 0.5).float().mean().item(),
               "lowest_confidences": [round(v, 3) for v in conf[order[:show]].tolist()],
               "highest_confidences": [round(v, 3) for v in conf[order[-show:]].tolist()]}
    if write:
        path = results.save("chapter3", "holes", numbers, seed=1, script=__file__, config={"decoder": "mnist_vae8.pt"})
        print({k: v for k, v in numbers.items() if "share" in k}, "written", path)
    fig, axes = plt.subplots(2, show, figsize=(show * 1.1, 2.6), dpi=150)
    for row, idx in enumerate((order[-show:].flip(0), order[:show])):
        for ax, i in zip(axes[row], idx):
            ax.imshow(imgs[i].view(28, 28).numpy(), cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"${conf[i]:.2f}$", fontsize=8)
            ax.axis("off")
    fig.text(0.005, 0.73, "Recognized", fontsize=8, rotation=90, va="center")
    fig.text(0.005, 0.27, "Not recognized", fontsize=8, rotation=90, va="center")
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "holes.png"
    save(fig, out, transparent=True)
    print("written", out)


# ---------------------------------------------------------------- decoder export
def export():
    dec = load("mnist_vae2.pt").dec
    layers = [m for m in dec if isinstance(m, nn.Linear)]
    blob, meta = bytearray(), {"layers": [], "activation": "relu", "output": "sigmoid", "latent": 2, "image": [28, 28]}
    for layer in layers:
        w = layer.weight.detach().numpy().astype(np.float32)
        b = layer.bias.detach().numpy().astype(np.float32)
        scale = np.abs(w).max(1, keepdims=True) / 127
        q = np.round(w / scale).astype(np.int8)
        entry = {"in": w.shape[1], "out": w.shape[0], "weights": len(blob)}
        blob += q.tobytes()
        entry["scales"] = len(blob)
        blob += scale.astype(np.float16).tobytes()
        entry["bias"] = len(blob)
        blob += b.astype(np.float16).tobytes()
        meta["layers"].append(entry)
    out = FIG / "interactive"
    out.mkdir(parents=True, exist_ok=True)
    (out / "decoder.bin").write_bytes(bytes(blob))
    (out / "decoder.json").write_text(json.dumps(meta, indent=1))
    # where 2,000 test digits land in the latent plane (encoder means), with their labels
    xt, yt = mnist(False)
    with torch.no_grad():
        mu = load("mnist_vae2.pt").encode(xt[:2000])[0]
    codes = {"z": [[round(a, 2), round(b, 2)] for a, b in mu.tolist()], "label": yt[:2000].tolist()}
    (out / "codes.json").write_text(json.dumps(codes, separators=(",", ":")))
    print(f"written {out / 'decoder.bin'} ({len(blob) / 1e6:.2f} MB), decoder.json and codes.json")


# ---------------------------------------------------------------- linear collapse rule vs the beta sweep
def linear():
    x, _ = mnist(True)
    nu = torch.linalg.eigvalsh(torch.cov(x.double().T)).flip(0)          # principal variances, largest first
    sweep = results.load("chapter3", "beta")["numbers"]
    numbers = {"principal_variances_top10": [round(v, 4) for v in nu[:10].tolist()],
               "total_variance": round(nu.sum().item(), 3), "by_beta": {}}
    for b in BETAS:
        threshold = b * S2
        above = int((nu > threshold).sum().item())
        numbers["by_beta"][str(b)] = {"threshold": threshold, "directions_above": above,
                                      "predicted_active": min(above, 8),
                                      "measured_active": sweep[str(b)]["active_units"]}
        print(f"beta {b:6g}  threshold {threshold:6.3f}  directions above {above:4d}  "
              f"measured active {sweep[str(b)]['active_units']}")
    path = results.save("chapter3", "linear-collapse", numbers, script=__file__,
                        config={"decoder_variance": S2, "latent": 8, "data": "MNIST training set, pixels in [0, 1]",
                                "measured": "results/chapter3/beta.json"})
    print("written", path)


# ---------------------------------------------------------------- calibrated decoder variance and blur
def high_frequency_power(imgs, cutoff=0.25):
    """Share of the spectral power (mean removed) above `cutoff` cycles per pixel, averaged over the images."""
    f = torch.fft.fft2(imgs.view(-1, 28, 28) - imgs.view(-1, 28, 28).mean((1, 2), keepdim=True)).abs() ** 2
    freq = torch.fft.fftfreq(28)
    radius = (freq[:, None] ** 2 + freq[None, :] ** 2).sqrt()
    return ((f * (radius > cutoff)).sum((1, 2)) / f.sum((1, 2)).clamp_min(1e-12)).mean().item()


def describe(model, xt, clf):
    """Measures of one trained 8D VAE on the test set and on 1,000 prior samples (the codes of chapter3.vae_numbers)."""
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(xt)
        torch.manual_seed(0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        rec = model.dec(z)
        torch.manual_seed(1)
        prior = model.dec(torch.randn(1000, 8))
    se = ((rec - xt) ** 2).sum(1)
    return {"squared_error_per_image": se.mean().item(),
            "optimal_decoder_variance": (se.mean() / 784).item(),
            "rate_nats": (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)).mean().item(),
            "active_units": int((mu.var(0) > 0.01).sum().item()),
            "mean_encoder_std": torch.exp(0.5 * logvar).mean().item(),
            "prior_samples_recognized": clf.confident_share(prior),
            "high_frequency_share_reconstructions": high_frequency_power(rec),
            "high_frequency_share_prior_samples": high_frequency_power(prior),
            "high_frequency_share_data": high_frequency_power(xt)}


def calibrate(epochs=30):
    x, _ = mnist(True)
    xt, _ = mnist(False)
    clf = metrics.MnistClassifier.load()
    fixed = describe(load("mnist_vae8.pt"), xt, clf)
    # Train the same architecture with the decoder variance at its optimum at every step. Plugging
    # sigma^2 = SE / 784 into SE / (2 sigma^2) + (784 / 2) log(2 pi sigma^2) leaves (784 / 2) log(SE / 784) + const,
    # computed on the minibatch.
    torch.manual_seed(0)
    model = Model(True, 8, (256, 64)).to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    X = x.to(device)
    for epoch in range(epochs):
        for idx in torch.randperm(len(X), device=device).split(128):
            mu, logvar = model.encode(X[idx])
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            se = ((model.dec(z) - X[idx]) ** 2).sum(1)
            kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)
            loss = 392 * torch.log(se.mean() / 784) + kl.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % 5 == 4:
            print(f"epoch {epoch + 1:2d}  decoder variance {se.mean().item() / 784:.4f}  KL {kl.mean().item():.2f}",
                  flush=True)
    calibrated = describe(model.cpu(), xt, clf)
    numbers = {"fixed_variance_model": fixed, "calibrated_model": calibrated,
               "effective_beta_of_fixed_model": S2 / fixed["optimal_decoder_variance"]}
    path = results.save("chapter3", "calibrated", numbers, seed=0, script=__file__,
                        config={"fixed_decoder_variance": S2, "epochs": epochs, "latent": 8, "hidden": [256, 64],
                                "high_frequency_cutoff": "0.25 cycles per pixel", "prior_codes_seed": 1})
    for name in ("fixed_variance_model", "calibrated_model"):
        print(name, {k: round(v, 4) for k, v in numbers[name].items()})
    print("effective beta of the fixed model", round(numbers["effective_beta_of_fixed_model"], 3))
    print("written", path)


# ---------------------------------------------------------------- estimators of p(x) and of the ELBO
def estimators(images=100, prior_draws=100_000, encoder_draws=1000):
    xt, _ = mnist(False)
    model = load("mnist_vae8.pt").to(device)
    const = 392 * math.log(2 * math.pi * S2)

    def log_lik(x, z):                                                    # log N(x; g(z), S2 I) for many z
        return -((model.dec(z) - x) ** 2).sum(1) / (2 * S2) - const

    def log_std_normal(z):
        return -0.5 * (z ** 2).sum(1) - 4 * math.log(2 * math.pi)

    rows = []
    torch.manual_seed(0)
    with torch.no_grad():
        for x in xt[:images].to(device):
            lw_prior = torch.cat([log_lik(x, z) for z in torch.randn(prior_draws, 8, device=device).split(20000)])
            mu, logvar = model.encode(x[None])
            sd = torch.exp(0.5 * logvar)
            z = mu + sd * torch.randn(encoder_draws, 8, device=device)
            log_q = (-0.5 * ((z - mu) / sd) ** 2 - torch.log(sd) - 0.5 * math.log(2 * math.pi)).sum(1)
            lw_enc = log_lik(x, z) + log_std_normal(z) - log_q

            def ess(lw):                                                  # effective sample size of the weights
                w = torch.exp(lw - lw.max())
                return (w.sum() ** 2 / (w ** 2).sum()).item()

            log_p = (torch.logsumexp(lw_enc, 0) - math.log(encoder_draws)).item()
            rows.append({"ess_prior": ess(lw_prior), "ess_encoder": ess(lw_enc),
                         "log_p_prior": (torch.logsumexp(lw_prior, 0) - math.log(prior_draws)).item(),
                         "log_p_encoder": log_p, "elbo": lw_enc.mean().item(), "elbo_one_sample_std": lw_enc.std().item(),
                         "best_prior_draw_share": torch.softmax(lw_prior, 0).max().item()})
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    numbers = {"median": med, "gap_lower_bound_nats_median": float(np.median([r["log_p_encoder"] - r["elbo"]
                                                                              for r in rows])),
               "prior_estimate_below_encoder_estimate_nats_median": float(np.median(
                   [r["log_p_encoder"] - r["log_p_prior"] for r in rows]))}
    path = results.save("chapter3", "estimators", numbers, seed=0, script=__file__,
                        config={"model": "mnist_vae8.pt", "test_images": images, "prior_draws": prior_draws,
                                "encoder_draws": encoder_draws, "decoder_variance": S2})
    print(json.dumps(numbers, indent=1))
    print("written", path)


# ---------------------------------------------------------------- blur of the plain autoencoder
def blur():
    xt, _ = mnist(False)
    ae = AE()
    ae.load_state_dict(torch.load(OUT / "mnist_ae.pt", map_location="cpu"))
    ae.eval()
    numbers = {}
    with torch.no_grad():
        rec = ae.g(ae.f(xt))
        torch.manual_seed(1)  # the prior codes of describe()
        prior = ae.g(torch.randn(1000, 8))
        numbers["plain_autoencoder"] = {"high_frequency_share_reconstructions": high_frequency_power(rec),
                                        "high_frequency_share_prior_samples": high_frequency_power(prior),
                                        "squared_error_per_image": ((rec - xt) ** 2).sum(1).mean().item()}
        for key, name in (("vae", "mnist_vae8.pt"), ("vae_kl_x100", "mnist_vae8_kl.pt")):
            m = load(name)
            mu, logvar = m.encode(xt)
            torch.manual_seed(0)
            rec = m.dec(mu + torch.exp(0.5 * logvar) * torch.randn_like(mu))
            torch.manual_seed(1)
            prior = m.dec(torch.randn(1000, 8))
            numbers[key] = {"high_frequency_share_reconstructions": high_frequency_power(rec),
                            "high_frequency_share_prior_samples": high_frequency_power(prior),
                            "squared_error_per_image": ((rec - xt) ** 2).sum(1).mean().item()}
    numbers["data"] = {"high_frequency_share": high_frequency_power(xt)}
    path = results.save("chapter3", "blur", numbers, seed=0, script=__file__,
                        config={"latent": 8, "high_frequency_cutoff": "0.25 cycles per pixel", "prior_codes_seed": 1,
                                "note": "plain autoencoder trained 20 epochs, VAEs 30 epochs"})
    print(json.dumps(numbers, indent=1))
    print("written", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    commands = {"ring": ring, "beta": beta, "holes": holes, "export": export, "linear": linear, "calibrate": calibrate,
                "estimators": estimators, "blur": blur, "beta-figure": beta_figure, "ring-figure": ring_figure,
                "holes-figure": lambda: holes(write=False)}
    ap.add_argument("cmd", choices=list(commands))
    commands[ap.parse_args().cmd]()
