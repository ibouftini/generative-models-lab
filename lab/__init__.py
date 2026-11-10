"""Shared code of the experiments: data, measures, results files and figure style.

Every path can be redirected, which is how a rerun is kept apart from the published files:

    LAB_CHECKPOINTS=/tmp/rerun/checkpoints LAB_RESULTS=/tmp/rerun/results LAB_FIGURES=/tmp/rerun/figures make figures
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"                                                     # MNIST, downloaded on first use
CHECKPOINTS = Path(os.environ.get("LAB_CHECKPOINTS", ROOT / "checkpoints"))  # trained weights and saved samples
RESULTS = Path(os.environ.get("LAB_RESULTS", ROOT / "results"))              # the numbers quoted in the course
FIGURES = Path(os.environ.get("LAB_FIGURES", ROOT / "figures"))              # the figures of the course
