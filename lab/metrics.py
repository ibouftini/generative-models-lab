"""Measures shared by the chapters.

  ring_modes(samples)                 ring of 8 Gaussians: modes covered and share of high-quality samples
  MnistClassifier.load()              a fixed MNIST classifier, trained once
    .confident_share(images)          share of images classified with more than 90% confidence
  memorization(samples, train, held)  data-copying test: are samples closer to the training set than fresh
                                      real images are?
"""
import torch
import torch.nn as nn

from . import CHECKPOINTS
from .data import RING_CENTERS, RING_STD, mnist

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- ring of 8 Gaussians
def ring_modes(samples, min_share=0.01):
    """A sample is high quality when it lies within 3 standard deviations (0.06) of the nearest centre.
    A mode is covered when it receives at least min_share of all samples as high-quality samples.
    Returns {"modes_covered": int, "high_quality": share in [0, 1], "per_mode": [share of samples per mode]}."""
    d = torch.cdist(samples.float().cpu(), RING_CENTERS)
    dist, nearest = d.min(1)
    good = dist <= 3 * RING_STD
    per_mode = [((nearest == k) & good).float().mean().item() for k in range(len(RING_CENTERS))]
    return {"modes_covered": sum(s >= min_share for s in per_mode),
            "high_quality": good.float().mean().item(), "per_mode": per_mode}


# ---------------------------------------------------------------- MNIST classifier
class MnistClassifier:
    """Small convolutional classifier (2 conv layers, 3 epochs, seed 0, about 99% test accuracy), trained once
    and reused by every chapter, so that "recognized as a digit" means the same thing everywhere."""

    PATH = CHECKPOINTS / "mnist_classifier.pt"

    def __init__(self, net, test_accuracy):
        self.net, self.test_accuracy = net.eval(), test_accuracy

    @staticmethod
    def _net():
        return nn.Sequential(
            nn.Unflatten(1, (1, 28, 28)), nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(64 * 49, 128), nn.ReLU(), nn.Linear(128, 10))

    @classmethod
    def train(cls):
        x, y = mnist(True)
        xt, yt = mnist(False)
        torch.manual_seed(0)
        net = cls._net().to(DEVICE)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        X, Y = x.to(DEVICE), y.to(DEVICE)
        for _ in range(3):
            for idx in torch.randperm(len(X), device=DEVICE).split(128):
                loss = nn.functional.cross_entropy(net(X[idx]), Y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(xt.to(DEVICE)).argmax(1).cpu() == yt).float().mean().item()
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        torch.save({"net": net.state_dict(), "test_accuracy": acc}, cls.PATH)
        return cls(net, acc)

    @classmethod
    def load(cls):
        """The shared classifier; trained (about 30 s on a GPU) the first time."""
        if not cls.PATH.exists():
            return cls.train()
        state = torch.load(cls.PATH, map_location=DEVICE)
        net = cls._net().to(DEVICE)
        net.load_state_dict(state["net"])
        return cls(net, state["test_accuracy"])

    @torch.no_grad()
    def probabilities(self, images):
        out = [self.net(chunk.to(DEVICE).flatten(1)).softmax(1).cpu() for chunk in images.split(2000)]
        return torch.cat(out)

    def confident_share(self, images, threshold=0.9):
        return (self.probabilities(images).max(1).values > threshold).float().mean().item()


# ---------------------------------------------------------------- memorization
@torch.no_grad()
def nearest_distance(a, b, chunk=1024):
    """For each row of a, the Euclidean distance to its nearest row of b (computed in chunks on the GPU).
    Exact differences, never the matrix-product form |x|^2 + |y|^2 - 2xy that torch.cdist uses by default for large
    inputs: in float32 that form loses about 3e-4 of absolute accuracy, as large as the copy threshold of the circle
    experiment with 200 points (about 1.5e-4)."""
    a, b = a.flatten(1).float(), b.flatten(1).float().to(DEVICE)
    return torch.cat([torch.cdist(x.to(DEVICE), b, compute_mode="donot_use_mm_for_euclid_dist").min(1).values.cpu()
                      for x in a.split(chunk)])


def memorization(samples, train, heldout, quantile=0.01):
    """Data-copying test (after Meehan, Chaudhuri and Dasgupta, 2020): are generated samples closer to the
    training set than fresh real images are?
        d_gen  distance of each sample to its nearest training image (full training set)
        d_ref  distance of each held-out real image to its nearest training image
    median_ratio = median(d_gen) / median(d_ref): about 1 for samples as new as real data, near 0 for copies,
    above 1 for samples farther from the data than real images (blurry or off the manifold).
    copy_share = share of samples closer to the training set than all but `quantile` of the held-out images
    (about `quantile` for real data, 1 for a model that returns training images)."""
    d_gen = nearest_distance(samples, train)
    d_ref = nearest_distance(heldout, train)
    threshold = torch.quantile(d_ref, quantile)
    return {"median_ratio": (d_gen.median() / d_ref.median()).item(),
            "copy_share": (d_gen < threshold).float().mean().item()}
