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
    ("ch1 k = 1, training loss still falling between 70,000 and 90,000 steps, draw 1, %", 28, 0,
     100 * latent_run(1024, 12, 0)["plateau_change"]),
    ("ch1 k = 1, the same, draw 2, %", 36, 0, 100 * latent_run(1024, 12, 1)["plateau_change"]),
    ("ch1 k = 1, the same, draw 3, %", 9, 0, 100 * latent_run(1024, 12, 2)["plateau_change"]),

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


def test_ch1_on_200_points_test_loss_is_4_to_13_times_the_floor_and_training_loss_beats_truth():
    rows = [latent_run(w, 200, s) for w in (16, 1024) for s in SEEDS]
    ratios = [r["test_loss"] / 0.002 for r in rows]
    assert printed(min(ratios), 0) == 4 and printed(max(ratios), 0) == 13
    assert all(r["train_loss"] < r["train_loss_truth"] for r in rows)


def test_ch1_on_200_points_samples_sit_near_the_circle_far_outside_the_copy_discs():
    k1 = [latent_run(w, 200, s) for w in (16, 1024) for s in SEEDS]
    k2 = [run(w, 200, s) for w in (16, 1024) for s in SEEDS]
    dist = lambda rows: [r["circle_distance_median"] for r in rows]              # noqa: E731
    assert printed(min(dist(k1)), 4) == 0.0015 and printed(max(dist(k1)), 4) == 0.0043
    assert printed(min(dist(k2)), 4) == 0.0034 and printed(max(dist(k2)), 4) == 0.0058
    radii = [r["copy_radius"] for r in k1]
    assert printed(min(radii), 5) == 0.00013 and printed(max(radii), 5) == 0.00018
    assert all(r["circle_distance_median"] >= 9 * r["copy_radius"] for r in k1)  # "at least nine times"
    # the stored values are those of the saved samples
    saved = torch.load(CHECKPOINTS / "memorization_sweeps_k1_samples.pt")
    heldout = data.circle(2000, seed=1000)
    for w in (16, 1024):
        for s in SEEDS:
            r, key = latent_run(w, 200, s), f"w{w}-n200-s{s}-z1"
            median = (saved["samples"][key].norm(dim=1) - 1).abs().median().item()
            radius = torch.quantile(metrics.nearest_distance(heldout, saved["trains"][f"n200-s{s}"]), 0.01).item()
            assert median == pytest.approx(r["circle_distance_median"]) and radius == pytest.approx(r["copy_radius"])


def test_ch1_on_12_points_test_loss_is_over_a_hundred_times_the_floor():
    assert all(latent_run(w, 12, s)["test_loss"] > 100 * latent_run(w, 12, s)["test_loss_truth"]
               for w in (16, 1024) for s in SEEDS)


def test_ch1_latent_dimension_1_longer_training():
    early_train, late_train, growth, best = [], [], [], []
    for s in SEEDS:
        key = f"w1024-n12-s{s}-z1"
        early, late = eval_at(LATENT_CURVES, key, 6000), eval_at(LATENT_CURVES, key, 100_000)
        early_train.append(early[0])
        late_train.append(late[0])
        growth.append(late[1] - early[1])
        evals = CURVE_CACHE[LATENT_CURVES][key]["eval"]
        best.append(min(evals, key=lambda e: e[2])[0])
    assert printed(min(early_train), 2) == 0.10 and printed(max(early_train), 2) == 0.15
    assert all(printed(v, 2) == 0.02 for v in late_train)                           # "about 0.02"
    assert printed(min(growth), 2) == 0.02 and printed(max(growth), 2) == 0.04
    assert best == [16_000, 8_000, 26_000]


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


