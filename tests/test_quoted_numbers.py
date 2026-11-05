"""Every measured number quoted in the course, compared with the results file it comes from.

A number passes when the value in the results file, rounded to the digits the chapter prints, equals the printed number.
The tables and sentences below are copied from the chapters, so a change to a result or to the text fails here.
"""
import json
import math
from decimal import ROUND_HALF_UP, Decimal

import pytest
import torch

from lab import CHECKPOINTS, data, metrics, results

SWEEPS = results.load("chapter1", "memorization-sweeps")["numbers"]
LATENT = results.load("chapter1", "memorization-sweeps-k1")["numbers"]
GENERATOR = results.load("chapter1", "mnist-wgan-losses")["numbers"]
KL = results.load("chapter2", "kl-directions")["numbers"]
RATES = results.load("chapter2", "rates")["numbers"]
VAE = results.load("chapter3", "vae-numbers")["numbers"]
VAE_RERUN = results.load("chapter3", "vae-numbers-rerun")["numbers"]
VAE_RERUN_2 = results.load("chapter3", "vae-numbers-rerun-2")["numbers"]
ESTIMATORS = results.load("chapter3", "estimators")["numbers"]
CALIBRATED = results.load("chapter3", "calibrated")["numbers"]
BETA = results.load("chapter3", "beta")["numbers"]
HOLES = results.load("chapter3", "holes")["numbers"]
COLLAPSE = results.load("chapter3", "linear-collapse")["numbers"]
RING = results.load("chapter3", "ring")["numbers"]
BLUR = results.load("chapter3", "blur")["numbers"]

SEEDS = (0, 1, 2)
CURVES = results.RESULTS / "chapter1" / "memorization-sweeps-curves.json"
LATENT_CURVES = results.RESULTS / "chapter1" / "memorization-sweeps-k1-curves.json"


def run(width, n, seed):
    return SWEEPS[f"w{width}-n{n}-s{seed}"]


def latent_run(width, n, seed):
    return LATENT[f"w{width}-n{n}-s{seed}-z1"]


def printed(value, digits):
    """The value as the chapter prints it, rounding halves away from zero."""
    factor = 10 ** digits
    return math.copysign(math.floor(abs(value) * factor + 0.5) / factor, value)


def percent_of_samples(share, digits=1):
    """A copy share as the chapter prints it. Shares are counts out of 2,000 samples stored in float32, so 227 / 2000 reads
    0.11349999. Going back to the count keeps the exact tie 11.35, which rounds to 11.4."""
    exact = Decimal(100 * round(share * 2000)) / 2000
    return float(exact.quantize(Decimal(1).scaleb(-digits), ROUND_HALF_UP))


# ---------------------------------------------------------------- one number, one source
QUOTED = [
    # Chapter 1, the MNIST generator of the opening figure
    ("ch1 copies of the MNIST generator, %", 0.20, 2, 100 * GENERATOR["generator"]["copy_share"]),
    ("ch1 copies of real test images, %", 1.15, 2, 100 * GENERATOR["real_test_images"]["copy_share"]),
    # Chapter 1, the circle
    ("ch1 test loss of fresh points", 0.002, 3, run(1024, 12, 0)["test_loss_truth"]),
    ("ch1 k = 1, far share of the narrow generator, %", 31, 0, 100 * latent_run(16, 12, 0)["far_share"]),
    ("ch1 k = 1, training loss of that generator", 0.205, 3, latent_run(16, 12, 0)["train_loss"]),
    ("ch1 plateau change, draw 2, %", 27, 0, 100 * run(1024, 12, 1)["plateau_change"]),
    ("ch1 plateau change, draw 3, %", 31, 0, 100 * run(1024, 12, 2)["plateau_change"]),
    # Chapter 2
    ("ch2 forward fit, standard deviation", 2.06, 2, KL["forward_fit"]["sigma"]),
    ("ch2 forward fit, variance", 4.25, 2, KL["forward_fit"]["sigma"] ** 2),
    ("ch2 reverse fit, standard deviation", 0.50, 2, KL["reverse_fit"]["sigma"]),
    ("ch2 reverse fit, before rounding", 0.5004, 4, KL["reverse_fit"]["sigma"]),
    ("ch2 forward fit, KL(target || q)", 0.72, 2, KL["forward_fit"]["forward_kl"]),
    ("ch2 forward fit, KL(q || target)", 2.10, 2, KL["forward_fit"]["reverse_kl"]),
    ("ch2 reverse fit, KL(target || q)", 15.28, 2, KL["reverse_fit"]["forward_kl"]),
    ("ch2 reverse fit, KL(q || target)", 0.69, 2, KL["reverse_fit"]["reverse_kl"]),
    ("ch2 numerical forward fit, mean", 0, 3, KL["forward_fit"]["numerical_mu"]),
    ("ch2 numerical forward fit, standard deviation", 2.06, 2, KL["forward_fit"]["numerical_sigma"]),
    ("ch2 slope, curve", -0.457, 3, RATES["curve in R^16 (m = 1)"]["slope"]),
    ("ch2 slope, Gaussian", -0.078, 3, RATES["Gaussian in R^16 (m = 16)"]["slope"]),
    ("ch2 slope, MNIST", -0.084, 3, RATES["MNIST digits (p = 784)"]["slope"]),
    ("ch2 dimension read off the Gaussian slope", 13, 0, -1 / RATES["Gaussian in R^16 (m = 16)"]["slope"]),
    ("ch2 dimension read off the MNIST slope", 12, 0, -1 / RATES["MNIST digits (p = 784)"]["slope"]),
    # Chapter 3, opening
    ("ch3 classifier test accuracy, %", 98.9, 1, 100 * VAE["classifier_test_accuracy"]),
    ("ch3 recognized, plain autoencoder, %", 0.3, 1, 100 * VAE["confident_autoencoder"]),
    ("ch3 recognized, plain autoencoder, of 1,000", 3, 0, 1000 * VAE["confident_autoencoder"]),
    ("ch3 recognized, VAE, %", 46.7, 1, 100 * VAE["confident_vae"]),
    ("ch3 recognized, heavy penalty, %", 15.1, 1, 100 * VAE["confident_vae_kl"]),
    ("ch3 recognized, real test digits, %", 96.8, 1, 100 * VAE["confident_real_test_all"]),
    ("ch3 autoencoder code, first standard deviation", 5.8, 1, VAE["ae2_code_sd"][0]),
    ("ch3 autoencoder code, second standard deviation", 4.3, 1, VAE["ae2_code_sd"][1]),
    ("ch3 autoencoder code, 95% radius", 16.7, 1, VAE["ae2_radius_95"]),
    ("ch3 VAE clouds, standard deviation", 0.05, 2, VAE["sigma_mean_vae"]),
    ("ch3 VAE clouds, KL per image", 24.8, 1, VAE["kl_per_image_vae"]),
    ("ch3 heavy penalty, standard deviation", 0.92, 2, VAE["sigma_mean_vae_kl"]),
    ("ch3 heavy penalty, KL per image", 0.83, 2, VAE["kl_per_image_vae_kl"]),
    ("ch3 heavy penalty, coordinates that stopped varying", 6, 0, 8 - VAE["active_units_vae_kl"]),
    # Chapter 3, estimators
    ("ch3 prior estimate below the encoder estimate, nats", 85, 0,
     ESTIMATORS["prior_estimate_below_encoder_estimate_nats_median"]),
    ("ch3 effective number of encoder codes", 2.4, 1, ESTIMATORS["median"]["ess_encoder"]),
    ("ch3 lower bound on the gap, nats", 14, 0, ESTIMATORS["gap_lower_bound_nats_median"]),
    ("ch3 ELBO of the median digit, nats", 468, 0, ESTIMATORS["median"]["elbo"]),
    ("ch3 standard deviation of the log weights, nats", 6.7, 1, ESTIMATORS["median"]["elbo_one_sample_std"]),
    # Chapter 3, decoder variance
    ("ch3 squared error per image, VAE", 12.7, 1, CALIBRATED["fixed_variance_model"]["squared_error_per_image"]),
    ("ch3 best decoder variance, VAE", 0.016, 3, CALIBRATED["fixed_variance_model"]["optimal_decoder_variance"]),
    ("ch3 best decoder variance, more digits", 0.0162, 4, CALIBRATED["fixed_variance_model"]["optimal_decoder_variance"]),
    ("ch3 effective beta", 1.85, 2, CALIBRATED["effective_beta_of_fixed_model"]),
    ("ch3 calibrated VAE, decoder variance", 0.016, 3, CALIBRATED["calibrated_model"]["optimal_decoder_variance"]),
    ("ch3 calibrated VAE, squared error per image", 12.4, 1, CALIBRATED["calibrated_model"]["squared_error_per_image"]),
    ("ch3 calibrated VAE, KL per image", 28.0, 1, CALIBRATED["calibrated_model"]["rate_nats"]),
    ("ch3 calibrated VAE, recognized, %", 44.0, 1, 100 * CALIBRATED["calibrated_model"]["prior_samples_recognized"]),
    ("ch3 fixed VAE, recognized, %", 46.7, 1, 100 * CALIBRATED["fixed_variance_model"]["prior_samples_recognized"]),
    ("ch3 squared-error term, nats", 212, 0,
     CALIBRATED["fixed_variance_model"]["squared_error_per_image"] / (2 * 0.03)),
    # Chapter 3, beta sweep
    ("ch3 rate at beta = 100, nats", 0.7, 1, BETA["100"]["rate_nats"]),
    ("ch3 rate at beta = 0.1, nats", 51, 0, BETA["0.1"]["rate_nats"]),
    ("ch3 squared error at beta = 100", 47, 0, BETA["100"]["distortion"]),
    ("ch3 slope in nats, beta 3 to 1", -1.9, 1, (BETA["1"]["distortion_nats"] - BETA["3"]["distortion_nats"])
     / (BETA["1"]["rate_nats"] - BETA["3"]["rate_nats"])),
    ("ch3 slope in nats, beta 1 to 0.3", -0.36, 2, (BETA["0.3"]["distortion_nats"] - BETA["1"]["distortion_nats"])
     / (BETA["0.3"]["rate_nats"] - BETA["1"]["rate_nats"])),
    ("ch3 active coordinates at beta = 100", 2, 0, BETA["100"]["active_units"]),
    ("ch3 active coordinates at beta = 30", 6, 0, BETA["30"]["active_units"]),
    # Chapter 3, holes
    ("ch3 prior samples below 90% confidence, %", 53, 0, 100 * HOLES["share_below_0.9"]),
    ("ch3 prior samples below 50% confidence, %", 14, 0, 100 * HOLES["share_below_0.5"]),
    # Chapter 3, linear collapse rule
    ("ch3 largest principal variance", 5.1, 1, COLLAPSE["principal_variances_top10"][0]),
    ("ch3 second principal variance", 3.7, 1, COLLAPSE["principal_variances_top10"][1]),
    ("ch3 third principal variance", 3.3, 1, COLLAPSE["principal_variances_top10"][2]),
    ("ch3 total variance", 52.7, 1, COLLAPSE["total_variance"]),
    # Chapter 3, ring
    ("ch3 ring, modes covered", 8, 0, RING["vae"]["modes_covered"]),
    ("ch3 ring, VAE samples near a mode, %", 61.5, 1, 100 * RING["vae"]["high_quality"]),
    ("ch3 ring, real samples near a mode, %", 99.1, 1, 100 * RING["real"]["high_quality"]),
    # Chapter 3, blur
    ("ch3 high frequencies, real digits, %", 10.5, 1, 100 * BLUR["data"]["high_frequency_share"]),
    ("ch3 high frequencies, VAE reconstructions, %", 4.7, 1, 100 * BLUR["vae"]["high_frequency_share_reconstructions"]),
    ("ch3 high frequencies, calibrated VAE, %", 4.9, 1,
     100 * CALIBRATED["calibrated_model"]["high_frequency_share_reconstructions"]),
    ("ch3 high frequencies, plain autoencoder, %", 4.4, 1,
     100 * BLUR["plain_autoencoder"]["high_frequency_share_reconstructions"]),
    ("ch3 high frequencies, heavy penalty, %", 0.15, 2, 100 * BLUR["vae_kl_x100"]["high_frequency_share_reconstructions"]),
    ("ch3 high frequencies, VAE prior samples, %", 5.6, 1, 100 * BLUR["vae"]["high_frequency_share_prior_samples"]),
    ("ch3 high frequencies, autoencoder prior samples, %", 4.9, 1,
     100 * BLUR["plain_autoencoder"]["high_frequency_share_prior_samples"]),
]


@pytest.mark.parametrize("name, quoted, digits, value", QUOTED, ids=[q[0] for q in QUOTED])
def test_quoted_number(name, quoted, digits, value):
    assert printed(value, digits) == pytest.approx(quoted, abs=10 ** -(digits + 3))


# ---------------------------------------------------------------- the tables of Chapter 1
# width, n: training loss, truth, test loss, copies in % (smallest and largest of the three draws)
TABLE_K2 = {
    (4, 12): ("0.116-0.214", "0.351-0.600", "0.183-0.472", "1.0-4.0"),
    (16, 12): ("0.054-0.106", "0.351-0.600", "0.256-0.518", "3.6-6.6"),
    (64, 12): ("0.016-0.062", "0.351-0.600", "0.261-0.536", "4.5-15.7"),
    (256, 12): ("0.014-0.020", "0.351-0.600", "0.288-0.507", "9.4-23.5"),
    (1024, 12): ("0.010-0.015", "0.351-0.600", "0.277-0.538", "11.4-25.1"),
    (1024, 25): ("0.014-0.018", "0.131-0.241", "0.097-0.230", "1.0-2.5"),
    (1024, 50): ("0.010-0.012", "0.063-0.152", "0.057-0.139", "0.1-0.5"),
    (1024, 100): ("0.007-0.008", "0.035-0.053", "0.030-0.045", "0.1-0.1"),
    (1024, 200): ("0.004-0.007", "0.016-0.025", "0.016-0.029", "0-0.1"),
    (16, 200): ("0.005-0.007", "0.016-0.025", "0.011-0.032", "0-0"),
    (4, 200): ("0.013-0.018", "0.016-0.025", "0.011-0.030", "0-0"),
}
TABLE_K1 = {
    (16, 12): ("0.097-0.205", "0.351-0.600", "0.292-0.490", "1.6-3.2"),
    (1024, 12): ("0.017-0.020", "0.351-0.600", "0.264-0.537", "6.7-24.1"),
    (16, 200): ("0.012-0.015", "0.016-0.025", "0.008-0.017", "0-0.1"),
    (1024, 200): ("0.005-0.006", "0.016-0.025", "0.012-0.026", "0-0.1"),
}
ROWS = [("k2", *key, cells) for key, cells in TABLE_K2.items()] + [("k1", *key, cells) for key, cells in TABLE_K1.items()]


@pytest.mark.parametrize("table, width, n, cells", ROWS, ids=[f"{r[0]}-w{r[1]}-n{r[2]}" for r in ROWS])
def test_circle_table(table, width, n, cells):
    get = run if table == "k2" else latent_run
    rows = [get(width, n, s) for s in SEEDS]
    for field, cell in (("train_loss", cells[0]), ("train_loss_truth", cells[1]), ("test_loss", cells[2])):
        low, high = (float(v) for v in cell.split("-"))
        values = [r[field] for r in rows]
        assert printed(min(values), 3) == pytest.approx(low, abs=1e-9), field
        assert printed(max(values), 3) == pytest.approx(high, abs=1e-9), field
    low, high = (float(v) for v in cells[3].split("-"))
    shares = [r["copy_share"] for r in rows]
    assert percent_of_samples(min(shares)) == low and percent_of_samples(max(shares)) == high


# ---------------------------------------------------------------- the sentences of Chapter 1
CURVE_CACHE = {}


def eval_at(curve_file, key, step):
    """(training loss, test loss) of a run at a given step."""
    if curve_file not in CURVE_CACHE:
        CURVE_CACHE[curve_file] = json.loads(curve_file.read_text())
    row = next(e for e in CURVE_CACHE[curve_file][key]["eval"] if e[0] == step)
    return row[1], row[2]


def test_ch1_every_generator_on_12_points_beats_the_truth():
    widths = (4, 16, 64, 256, 1024)
    assert all(run(w, 12, s)["train_loss"] <= 0.22 for w in widths for s in SEEDS)
    assert all(run(w, 12, s)["train_loss_truth"] >= 0.35 for w in widths for s in SEEDS)


def test_ch1_wider_generator_copies_more_and_has_lower_training_loss():
    for s in SEEDS:
        assert run(1024, 12, s)["copy_share"] > run(16, 12, s)["copy_share"]
        assert run(1024, 12, s)["train_loss"] < run(16, 12, s)["train_loss"]
        assert run(16, 12, s)["test_loss"] > 0.25 and run(1024, 12, s)["test_loss"] > 0.25


def test_ch1_longer_training_raises_the_test_loss_by_005_to_008():
    growth = []
    for s in SEEDS:
        key = f"w1024-n12-s{s}"
        early, late = eval_at(CURVES, key, 6000), eval_at(CURVES, key, 100_000)
        growth.append(late[1] - early[1])
        assert printed(early[0], 1) == 0.1                                         # about 0.1 at 6,000 steps
        assert printed(late[0], 2) == 0.01                                         # about 0.01 at 100,000 steps
    assert printed(min(growth), 2) == 0.05 and printed(max(growth), 2) == 0.08


def test_ch1_more_data_lowers_the_test_loss_at_every_size_in_every_draw():
    for s in SEEDS:
        losses = [run(1024, n, s)["test_loss"] for n in (12, 25, 50, 100, 200)]
        assert losses == sorted(losses, reverse=True)
    mean = lambda n: sum(run(1024, n, s)["test_loss"] for s in SEEDS) / 3     # noqa: E731
    assert printed(mean(12), 1) == 0.4 and printed(mean(200), 2) == 0.02


def test_ch1_on_200_points_test_loss_is_5_to_16_times_the_floor_and_training_loss_beats_truth():
    rows = [run(w, 200, s) for w in (16, 1024) for s in SEEDS]
    ratios = [r["test_loss"] / 0.002 for r in rows]
    assert printed(min(ratios), 0) == 5 and printed(max(ratios), 0) == 16
    assert all(r["train_loss"] < r["train_loss_truth"] for r in rows)


def test_ch1_on_200_points_samples_sit_near_the_circle_far_outside_the_copy_discs():
    saved = torch.load(CHECKPOINTS / "memorization_sweeps_samples.pt")
    medians = [(saved["samples"][f"w{w}-n200-s{s}"].norm(dim=1) - 1).abs().median().item() for w in (16, 1024) for s in SEEDS]
    assert printed(min(medians), 3) == 0.003 and printed(max(medians), 3) == 0.006
    heldout = data.circle(2000, seed=1000)
    radii = [torch.quantile(metrics.nearest_distance(heldout, saved["trains"][f"n200-s{s}"]), 0.01).item() for s in SEEDS]
    assert all(1e-4 < r < 3e-4 for r in radii), radii                            # "about 0.0002"


def test_ch1_width_4_has_the_lowest_test_loss_and_copies_order():
    for s in SEEDS:
        assert all(run(4, 12, s)["test_loss"] <= run(w, 12, s)["test_loss"] for w in (16, 64, 256, 1024))
        assert run(4, 12, s)["copy_share"] < run(16, 12, s)["copy_share"] < run(64, 12, s)["copy_share"]
    for scale in ("0.5", "2"):
        share = lambda w: SWEEPS[f"w{w}-n12-s0-k{scale}"]["copy_share"]        # noqa: E731
        assert share(4) < share(16) < share(64)


def test_ch1_latent_dimension_1_repeats_the_results():
    for s in SEEDS:
        assert all(latent_run(w, 12, s)["train_loss"] < latent_run(w, 12, s)["train_loss_truth"] for w in (16, 1024))
        assert latent_run(1024, 12, s)["train_loss"] < latent_run(16, 12, s)["train_loss"]
        assert latent_run(1024, 12, s)["copy_share"] > latent_run(16, 12, s)["copy_share"]
        assert latent_run(1024, 12, s)["median_ratio"] < 0.05 and run(1024, 12, s)["median_ratio"] < 0.05
        key = f"w1024-n12-s{s}-z1"
        assert eval_at(LATENT_CURVES, key, 6000)[1] < eval_at(LATENT_CURVES, key, 100_000)[1]
        for w in (16, 1024):
            assert percent_of_samples(latent_run(w, 200, s)["copy_share"]) <= 0.1
    others = [v["far_share"] for k, v in LATENT.items() if k != "w16-n12-s0-z1"]
    assert printed(100 * max(others), 1) == 2.4


def test_ch1_stopping_at_6000_steps_gives_the_lower_test_loss():
    for s in SEEDS:
        key = f"w1024-n12-s{s}"
        assert eval_at(CURVES, key, 6000)[1] < eval_at(CURVES, key, 100_000)[1]


def test_ch1_the_mnist_generator_copies_less_than_real_images():
    assert GENERATOR["generator"]["copy_share"] < GENERATOR["real_test_images"]["copy_share"]
    assert GENERATOR["generator"]["median_ratio"] > GENERATOR["real_test_images"]["median_ratio"]


# ---------------------------------------------------------------- the sentences and table of Chapter 3
def test_ch3_two_retrainings_of_the_classifier_move_the_shares_by_about_one_point():
    keys = [k for k in VAE if k.startswith("confident_")]
    spread = max(max(r[k] for r in (VAE, VAE_RERUN, VAE_RERUN_2)) - min(r[k] for r in (VAE, VAE_RERUN, VAE_RERUN_2)) for k in keys)
    assert keys and 0.005 < spread <= 0.01


def test_ch3_prior_sampling_puts_all_the_weight_on_one_code():
    assert ESTIMATORS["median"]["best_prior_draw_share"] > 0.99
    assert ESTIMATORS["median"]["ess_prior"] < 1.01


def test_ch3_squared_errors_of_the_three_repeated_betas():
    errors = [run_["distortion"] for b in ("0.1", "0.3", "1")
              for run_ in (BETA[b], *BETA[b]["seeds"].values())]
    assert printed(min(errors), 1) == 14.0 and printed(max(errors), 1) == 14.6


def test_ch3_collapse_table():
    for beta, threshold, above, predicted, measured in ((10, 0.3, 35, 8, 8), (30, 0.9, 13, 8, 6), (100, 3.0, 3, 3, 2)):
        row = COLLAPSE["by_beta"][str(beta)]
        assert row["threshold"] == pytest.approx(threshold)
        assert (row["directions_above"], row["predicted_active"], row["measured_active"]) == (above, predicted, measured)
    for beta in ("0.1", "0.3", "1", "3", "10"):
        assert COLLAPSE["by_beta"][beta]["measured_active"] == 8


# ---------------------------------------------------------------- numbers computed from formulas in the text
def test_ch2_closed_forms():
    assert printed(0.5 * math.log(2 / 3) + 0.5 * math.log(2), 2) == 0.14
    assert printed(0.75 * math.log(1.5) + 0.25 * math.log(0.5), 2) == 0.13
    assert printed(math.log(2), 3) == 0.693


def test_ch3_linear_model_figure():
    a2, s2 = 1.0 ** 2 + 0.5 ** 2, 0.15 ** 2
    assert printed(s2 / (a2 + s2), 3) == 0.018
    assert printed(math.sqrt((a2 + s2) / s2), 1) == 7.5


def test_ch3_prior_estimator_cost():
    s2, k = 0.03, 8
    r = 1 / s2
    factor = (1 + r) / math.sqrt(1 + 2 * r)
    assert printed(factor, 1) == 4.2
    assert printed(factor ** k / 1e4, 1) == 9.2                                  # 9.2 x 10^4
    assert printed((factor ** k - 1) / 0.1 ** 2 / 1e6, 0) == 9                   # 9 x 10^6 codes
    assert printed(factor * math.exp((1 + s2) * r / ((1 + r) * (1 + 2 * r) * s2)), 1) == 6.8
    assert printed(math.exp(3 / (2 * s2)), 0) == printed(math.exp(50), 0)


def test_ch3_decoder_constant():
    assert printed(-392 * math.log(2 * math.pi * 0.03), 0) == 654
    assert printed(5.1 / 0.03, 0) == 170
