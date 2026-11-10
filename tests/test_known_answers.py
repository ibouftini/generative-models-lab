"""Checks on cases whose answer is known: the data and the measures."""
import pytest
import torch

from lab import data, metrics


# ---------------------------------------------------------------- data and measures
def test_circle_sampler_is_deterministic():
    assert torch.equal(data.circle(500, 3), data.circle(500, 3))


def test_memorization_on_points_of_the_circle():
    train, held = data.circle(200, seed=1), data.circle(2000, seed=2)
    fresh = metrics.memorization(data.circle(2000, seed=3), train, held)
    assert fresh["median_ratio"] == pytest.approx(1, abs=0.15) and fresh["copy_share"] < 0.03
    copies = metrics.memorization(train, train, held)
    assert copies["median_ratio"] == 0 and copies["copy_share"] == 1


def test_nearest_distance_is_exact_below_the_float32_error_of_the_fast_form():
    train = data.circle(200, seed=1)
    offset = train + torch.tensor([1e-4, 0.0])
    assert metrics.nearest_distance(offset, train).max().item() == pytest.approx(1e-4, rel=0.2)


@pytest.fixture(scope="module")
def mnist():
    return data.mnist(True)[0], data.mnist(False)[0]


def test_memorization_on_mnist(mnist):
    train, test = mnist
    real = metrics.memorization(test[:5000], train, test[5000:])
    assert real["median_ratio"] == pytest.approx(1, abs=0.05) and real["copy_share"] < 0.03
    copies = metrics.memorization(train[:1000], train, test)
    assert copies["median_ratio"] < 0.01 and copies["copy_share"] == 1


