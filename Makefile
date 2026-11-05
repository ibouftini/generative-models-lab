# Run from the repository root. Figures and tests need no GPU; the experiments are much faster with one.
PY ?= python

.PHONY: test figures chapter1 chapter2 chapter3 all

test:
	$(PY) -m pytest tests -q

# Redraw every figure from the saved results, samples and weights. Trains nothing and writes no results.
figures:
	$(PY) -m chapter1.generator figure
	$(PY) -m chapter1.pushforward
	$(PY) -m chapter1.manifold
	$(PY) -m chapter1.no_density
	$(PY) -m chapter1.memorization figures
	$(PY) -m chapter2.two_uniforms
	$(PY) -m chapter2.kl_directions figure
	$(PY) -m chapter2.rates figure
	$(PY) -m chapter3.autoencoder figure
	$(PY) -m chapter3.vae figures
	$(PY) -m chapter3.linear_model
	$(PY) -m chapter3.experiments ring-figure
	$(PY) -m chapter3.experiments beta-figure
	$(PY) -m chapter3.experiments holes-figure
	$(PY) -m chapter3.experiments export
	$(PY) -m lab.gallery

# Retrain and recompute. Set LAB_CHECKPOINTS, LAB_RESULTS and LAB_FIGURES to keep the published files untouched.
chapter1:     # about 10 h on an RTX 3050
	$(PY) -m chapter1.generator train
	$(PY) -m chapter1.memorization run
	$(PY) -m chapter1.memorization latent
	$(PY) -m chapter1.memorization mnist

chapter2:     # under 1 min
	$(PY) -m chapter2.kl_directions run
	$(PY) -m chapter2.rates run

chapter3:     # about 12 min on an RTX 3050
	$(PY) -m chapter3.autoencoder train
	$(PY) -m chapter3.vae train
	$(PY) -m chapter3.vae_numbers
	$(PY) -m chapter3.experiments ring
	$(PY) -m chapter3.experiments beta
	$(PY) -m chapter3.experiments holes
	$(PY) -m chapter3.experiments linear
	$(PY) -m chapter3.experiments calibrate
	$(PY) -m chapter3.experiments estimators
	$(PY) -m chapter3.experiments blur

all: chapter1 chapter2 chapter3 figures
