"""Measures shared by the chapters.

  memorization(samples, train, held)  data-copying test: are samples closer to the training set than fresh
                                      real images are?
"""
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
