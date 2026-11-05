"""The measured numbers of the opening of Chapter 3, written to results/chapter3/<name>.json.

  confident_*     share of images recognized with more than 90% confidence by the shared MNIST classifier: 1000 codes
                  z ~ N(0, I) decoded by the 8D autoencoder and by the 8D VAE (same codes for both), and real test digits
  sigma_mean_*    average standard deviation of the encoder clouds over the test set (8D VAEs)
  kl_per_image_*  average KL(q(z | x) || N(0, I)) per test image, in nats (8D VAEs)
  *_kl            the VAE whose KL term dominates (decoder variance x100)
  active_units_*  latent coordinates whose encoder mean varies across test images (variance > 0.01)
  ae2_code_sd     standard deviation of each coordinate of the 2D autoencoder's codes (the prior has 1), and the
  ae2_radius_95   radius containing 95% of those codes (2.45 for the prior N(0, I_2))

usage: python -m chapter3.vae_numbers [--name vae-numbers]
"""
import argparse

import torch

from chapter3.autoencoder import AE, OUT, mnist
from chapter3.vae import load
from lab import metrics, results


@torch.no_grad()
def cloud_stats(model, x):
    mu, logvar = model.encode(x)
    kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1)
    return torch.exp(0.5 * logvar).mean().item(), kl.mean().item()


def main(name):
    clf = metrics.MnistClassifier.load()
    xt, _ = mnist(False)
    ae8 = AE()
    ae8.load_state_dict(torch.load(OUT / "mnist_ae.pt", map_location="cpu"))
    ae8.eval()
    numbers = {"classifier_test_accuracy": clf.test_accuracy}
    with torch.no_grad():
        for key, decoder in (("autoencoder", ae8.g), ("vae", load("mnist_vae8.pt").dec)):
            torch.manual_seed(1)  # the same 1000 codes for both decoders
            numbers[f"confident_{key}"] = clf.confident_share(decoder(torch.randn(1000, 8)))
    numbers["confident_real_test_all"] = clf.confident_share(xt)
    numbers["confident_real_test_first_1000"] = clf.confident_share(xt[:1000])
    with torch.no_grad():
        torch.manual_seed(1)
        numbers["confident_vae_kl"] = clf.confident_share(load("mnist_vae8_kl.pt").dec(torch.randn(1000, 8)))
    for key, ckpt in (("vae", "mnist_vae8.pt"), ("vae_kl", "mnist_vae8_kl.pt")):
        sigma, kl = cloud_stats(load(ckpt), xt)
        numbers[f"sigma_mean_{key}"] = sigma
        numbers[f"kl_per_image_{key}"] = kl
        with torch.no_grad():
            mu = load(ckpt).encode(xt)[0]
        numbers[f"active_units_{key}"] = int((mu.var(0) > 0.01).sum().item())
    with torch.no_grad():
        codes = load("mnist_ae2.pt").encode(xt)[0]
    numbers["ae2_code_sd"] = [round(v, 3) for v in codes.std(0).tolist()]
    numbers["ae2_radius_95"] = torch.quantile(codes.norm(dim=1), 0.95).item()
    path = results.save("chapter3", name, numbers, seed=1, script=__file__,
                        config={"prior_samples": 1000, "confidence_threshold": 0.9})
    for k, v in numbers.items():
        print(f"  {k:32s} {v}")
    print("written", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="vae-numbers")
    main(ap.parse_args().name)
