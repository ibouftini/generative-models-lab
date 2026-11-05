"""Checks on cases whose answer is known: the data, the measures, and the propositions of Chapter 3 against Monte Carlo,
exact sums or a numerical optimizer."""
import math

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from lab import data, metrics


# ---------------------------------------------------------------- data and measures
def test_ring_real_samples_cover_8_modes():
    r = metrics.ring_modes(data.ring(20000, seed=1))
    assert r["modes_covered"] == 8
    assert r["high_quality"] == pytest.approx(1 - math.exp(-4.5), abs=0.005)      # P(|N(0, I_2)| <= 3)


def test_ring_one_gaussian_covers_one_mode():
    one = data.RING_CENTERS[3] + data.RING_STD * torch.randn(5000, 2, generator=torch.Generator().manual_seed(2))
    assert metrics.ring_modes(one)["modes_covered"] == 1


@pytest.mark.parametrize("sampler", [lambda: data.ring(500, 3), lambda: data.circle(500, 3),
                                     lambda: data.curve(500, 16, 3), lambda: data.gaussian(500, 16, 3)],
                         ids=["ring", "circle", "curve", "gaussian"])
def test_toy_samplers_are_deterministic(sampler):
    assert torch.equal(sampler(), sampler())


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


def test_classifier_recognizes_real_digits(mnist):
    classifier = metrics.MnistClassifier.load()
    assert classifier.test_accuracy >= 0.985
    assert classifier.confident_share(mnist[1]) > 0.9


def test_memorization_on_mnist(mnist):
    train, test = mnist
    real = metrics.memorization(test[:5000], train, test[5000:])
    assert real["median_ratio"] == pytest.approx(1, abs=0.05) and real["copy_share"] < 0.03
    copies = metrics.memorization(train[:1000], train, test)
    assert copies["median_ratio"] < 0.01 and copies["copy_share"] == 1


# ---------------------------------------------------------------- propositions of Chapter 3
@pytest.fixture
def rng():
    return np.random.default_rng(0)


def gauss_logpdf(x, mean, cov):
    diff = x - mean
    _, logdet = np.linalg.slogdet(2 * np.pi * cov)
    return -0.5 * (logdet + diff @ np.linalg.solve(cov, diff))


def test_density_bounds_and_jensen_lower_bound():
    """0 < p(x) <= (2 pi s2)^(-d/2) and log p(x) >= -(d/2) log(2 pi s2) - E||x - g(Z)||^2 / (2 s2), d = 1, g = tanh."""
    s2 = 0.05
    zs = np.linspace(-8, 8, 20001)
    prior = np.exp(-zs ** 2 / 2) / np.sqrt(2 * np.pi)
    dz = zs[1] - zs[0]
    for x0 in (-1.5, 0.0, 0.4, 2.0):
        px = np.sum(np.exp(-(x0 - np.tanh(3 * zs)) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2) * prior) * dz
        lower = -0.5 * np.log(2 * np.pi * s2) - np.sum((x0 - np.tanh(3 * zs)) ** 2 * prior) * dz / (2 * s2)
        assert 0 < px <= (2 * np.pi * s2) ** -0.5 and np.log(px) >= lower


def test_linear_model_posterior_and_marginal(rng):
    """Posterior N(M^-1 A^T (x - b), s2 M^-1), M = A^T A + s2 I, against joint Gaussian conditioning."""
    d, k, s2 = 5, 2, 0.3
    A, b, x0 = rng.normal(size=(d, k)), rng.normal(size=d), rng.normal(size=d)
    C = A @ A.T + s2 * np.eye(d)
    M = A.T @ A + s2 * np.eye(k)
    assert np.abs(np.linalg.solve(M, A.T @ (x0 - b)) - A.T @ np.linalg.solve(C, x0 - b)).max() < 1e-10
    assert np.abs(s2 * np.linalg.inv(M) - (np.eye(k) - A.T @ np.linalg.solve(C, A))).max() < 1e-10
    z = rng.normal(size=(400000, k))
    x = z @ A.T + b + np.sqrt(s2) * rng.normal(size=(400000, d))
    assert np.abs(np.cov(x.T) - C).max() < 0.05


def test_second_moment_of_the_prior_estimator(rng):
    """E_p(z)[p(x|z)^2] / p(x)^2 = prod_j (1+r)/sqrt(1+2r) * exp(x_j^2 r / ((1+r)(1+2r) s2)), A = a [I_k; 0]."""
    a, s2, k, d = 1.0, 0.25, 2, 3
    r = a ** 2 / s2
    A = np.vstack([a * np.eye(k), np.zeros((d - k, k))])
    b, C = np.zeros(d), A @ A.T + s2 * np.eye(d)
    for x0 in (np.zeros(d), np.array([0.8, -0.5, 0.3])):
        z = rng.normal(size=(2_000_000, k))
        logp = -0.5 * (((x0 - z @ A.T) ** 2).sum(1) / s2 + d * np.log(2 * np.pi * s2))
        monte_carlo = np.mean(np.exp(2 * logp)) / np.exp(2 * gauss_logpdf(x0, b, C))
        formula = np.prod((1 + r) / np.sqrt(1 + 2 * r) * np.exp(x0[:k] ** 2 * r / ((1 + r) * (1 + 2 * r) * s2)))
        assert monte_carlo == pytest.approx(formula, rel=0.02)


def test_gaussian_kl_closed_form(rng):
    mu, sd = np.array([0.7, -1.2]), np.array([0.4, 1.5])
    z = mu + sd * rng.normal(size=(1_000_000, 2))
    monte_carlo = np.mean((-np.log(sd) - (z - mu) ** 2 / (2 * sd ** 2) + z ** 2 / 2).sum(1))
    assert monte_carlo == pytest.approx(0.5 * np.sum(mu ** 2 + sd ** 2 - 1 - np.log(sd ** 2)), abs=0.01)


@pytest.fixture
def discrete(rng):
    """A discrete model for the identities of the ELBO: x in 4 values, z in 3 values."""
    px_star, pz = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(3))
    px_given_z = rng.dirichlet(np.ones(4), size=3)                                # rows z, columns x
    q = rng.dirichlet(np.ones(3), size=4)                                         # rows x, columns z: the encoder
    return px_star, pz, px_given_z, q


def test_joint_kl_identity(discrete):
    """KL(joint) = -E[ELBO] - H(P*) = KL(P* || P_theta) + E KL(q || posterior)."""
    px_star, pz, px_given_z, q = discrete
    p_joint = pz[:, None] * px_given_z
    p_x = p_joint.sum(0)
    post = (p_joint / p_x).T
    elbo = (q * (np.log(px_given_z.T) + np.log(pz) - np.log(q))).sum(1)
    q_joint = px_star[:, None] * q
    kl_joint = np.sum(q_joint * np.log(q_joint / p_joint.T))
    entropy = -np.sum(px_star * np.log(px_star))
    gap = np.sum(px_star * (q * np.log(q / post)).sum(1))
    assert kl_joint == pytest.approx(-px_star @ elbo - entropy, abs=1e-12)
    assert kl_joint == pytest.approx(np.sum(px_star * np.log(px_star / p_x)) + gap, abs=1e-12)


def test_rate_split_and_best_prior(discrete, rng):
    """rate = I_q(x; z) + KL(q_bar || p), and the aggregate posterior is the best prior."""
    px_star, pz, px_given_z, q = discrete
    q_bar = px_star @ q
    rate = np.sum(px_star * (q * np.log(q / pz)).sum(1))
    mutual_information = np.sum(px_star * (q * np.log(q / q_bar)).sum(1))
    assert rate == pytest.approx(mutual_information + np.sum(q_bar * np.log(q_bar / pz)), abs=1e-12)

    def average_elbo(prior):
        return px_star @ (q * (np.log(px_given_z.T) + np.log(prior) - np.log(q))).sum(1)

    assert max(average_elbo(rng.dirichlet(np.ones(3))) for _ in range(2000)) < average_elbo(q_bar)


def test_squared_error_decoder_is_the_conditional_mean(discrete, rng):
    px_star, _, _, q = discrete
    q_joint = px_star[:, None] * q
    values = rng.normal(size=4)                                                   # images as numbers
    conditional_mean = (q_joint * values[:, None]).sum(0) / q_joint.sum(0)
    fit = minimize(lambda c: np.sum(q_joint * (values[:, None] - c[None, :]) ** 2), np.zeros(3), options={"gtol": 1e-12})
    assert np.abs(fit.x - conditional_mean).max() < 1e-5


def test_optimal_shared_decoder_variance(rng):
    """The best shared variance is the mean squared error per coordinate."""
    d = 50
    residual = rng.normal(scale=0.2, size=(1000, d)) * rng.uniform(0.5, 1.5, size=d)
    fit = minimize(lambda t: np.mean((residual ** 2).sum(1)) / (2 * np.exp(t[0])) + d / 2 * np.log(2 * np.pi * np.exp(t[0])),
                   [0.0])
    assert np.exp(fit.x[0]) == pytest.approx(np.mean(residual ** 2), rel=1e-4)


def test_gradient_estimator_variances(rng):
    """d/dmu E[z^2], z ~ N(0, sigma^2): the pathwise 2z has variance 4 sigma^2, the score z^3 / sigma^2 has 15 sigma^2."""
    sigma = 0.7
    z = sigma * rng.normal(size=4_000_000)
    assert np.var(2 * z) == pytest.approx(4 * sigma ** 2, rel=0.01)
    assert np.var(z ** 3 / sigma ** 2) == pytest.approx(15 * sigma ** 2, rel=0.02)


def test_linear_model_collapse(rng):
    """k = 1: the maximum-likelihood ||a||^2 is max(nu_1 - s2, 0), along the top principal direction."""
    nu = np.array([2.0, 1.0, 0.5, 0.2])
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    covariance = Q @ np.diag(nu) @ Q.T

    def negative_log_likelihood(a, s2):
        C = np.outer(a, a) + s2 * np.eye(4)
        return 0.5 * (np.linalg.slogdet(C)[1] + np.trace(np.linalg.solve(C, covariance)))

    for s2 in (0.5, 1.5, 2.5):
        fits = [minimize(negative_log_likelihood, rng.normal(size=4), args=(s2,)) for _ in range(5)]
        a = min(fits, key=lambda f: f.fun).x
        assert a @ a == pytest.approx(max(nu[0] - s2, 0.0), abs=1e-3)
        if nu[0] > s2:
            assert abs(a @ Q[:, 0]) / np.sqrt(a @ a) > 0.999
