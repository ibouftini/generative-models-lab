# Run from the repository root. Figures and tests need no GPU; the experiments are much faster with one.
PY ?= python

.PHONY: test figures chapter1 all

test:
	$(PY) -m pytest tests -q

# Redraw every figure from the saved results, samples and weights. Trains nothing and writes no results.
figures:
	$(PY) -m chapter1.generator figure
	$(PY) -m chapter1.circle_task
	$(PY) -m chapter1.pushforward
	$(PY) -m chapter1.manifold
	$(PY) -m chapter1.no_density
	$(PY) -m chapter1.memorization figures
	$(PY) -m lab.gallery

# Retrain and recompute. Set LAB_CHECKPOINTS, LAB_RESULTS and LAB_FIGURES to keep the published files untouched.
chapter1:     # about 10 h on an RTX 3050
	$(PY) -m chapter1.autoencoder train
	$(PY) -m chapter1.vae train
	$(PY) -m chapter1.generator train
	$(PY) -m chapter1.memorization run
	$(PY) -m chapter1.memorization latent
	$(PY) -m chapter1.memorization geometry
	$(PY) -m chapter1.memorization mnist

all: chapter1 figures
