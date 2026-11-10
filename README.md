# Generative models for images: code and results

This repository holds the experiments of the course [Generative Models for Images](https://ibouftini.github.io/courses/generative-models-for-images/). It covers Chapter 1 for now, and later chapters will be added as they are published. Every figure of Chapter 1 comes from a script here, and every number it reports comes from a results file here. A test compares each quoted number with its source.

## Check it

These steps need no GPU and take a few minutes. They use the results, samples and weights stored in the repository.

```bash
git clone https://github.com/ibouftini/generative-models-lab
cd generative-models-lab
pip install -r requirements.txt
make test      # compares the numbers quoted in the chapters with results/, checks known answers
make figures   # redraws every figure into figures/ from the saved files, trains nothing
```

MNIST is downloaded on first use. The figures use the Open Sans font (`apt install fonts-open-sans`).

To retrain, send the output to another folder so the published files stay untouched, then compare the two.

```bash
export LAB_CHECKPOINTS=/tmp/rerun/checkpoints LAB_RESULTS=/tmp/rerun/results LAB_FIGURES=/tmp/rerun/figures
make chapter1    # about 10 h on an RTX 3050
python -m lab.compare /tmp/rerun/results
```

## Chapter 1

[Learning a distribution](https://ibouftini.github.io/courses/generative-models-for-images/learning-a-distribution/) trains generators on points of a circle and measures when they memorize the training points.

![Training and test loss against training steps for three draws of the training points](docs/memorization-steps.png)

| Experiment | Command | Time | Results |
|---|---|---|---|
| Autoencoder on MNIST, the digits of the manifold schematic | `python -m chapter1.autoencoder train` | 1 min | `checkpoints/mnist_ae.pt` |
| VAE on MNIST, the noisy digit of the manifold schematic | `python -m chapter1.vae train` | a few minutes | `checkpoints/mnist_vae8.pt` |
| WGAN-GP on MNIST, the digits of the opening figure | `python -m chapter1.generator train` | 3 h | `checkpoints/mnist_wgan_gp.pt` |
| Circle generators, 43 runs over width and number of points | `python -m chapter1.memorization run` | 5.4 h | `results/chapter1/memorization-sweeps.json` |
| The same with a one-dimensional latent, 12 runs | `python -m chapter1.memorization latent` | 1.6 h | `results/chapter1/memorization-sweeps-k1.json` |
| Distance of the samples to the circle and radius of the copy discs | `python -m chapter1.memorization geometry` | seconds | added to both sweep files |
| Training loss, test loss and copies of the MNIST generator | `python -m chapter1.memorization mnist` | 2 min | `results/chapter1/mnist-wgan-losses.json` |

The schematics come from `chapter1/circle_task.py`, `chapter1/pushforward.py`, `chapter1/manifold.py` and `chapter1/no_density.py`. `chapter1.memorization figures` also writes the data of the two interactive sample figures into `figures/chapter1/interactive/`.

## Reproducibility

Every experiment except the last 14 of the 55 circle runs was retrained from scratch on the machine that produced the published files, an RTX 3050 Laptop GPU (4 GB) with the library versions of `requirements.txt`. The two sets of results were then compared with `python -m lab.compare`, leaving out run times.

| Results | Identical numbers | Cause of the differences |
|---|---:|---|
| Chapter 1, circle generators, 41 of 55 runs | 644 of 644 | None |
| Chapter 1, MNIST generator | 7 of 11 | Retrained WGAN-GP |

The WGAN-GP does not repeat from one training to the next, even with the same seed. It is the only convolutional network in the repository, so GPU convolutions are the likely cause, although this was not tested further. The autoencoder and the VAE came back with identical weights. With the published generator from `checkpoints/`, all 11 numbers of its row match exactly. The retrained generator copies 0.30% of its samples, against 0.20% for the published one. Real test images score 1.15%, so the generator stays below them in both trainings.

Other GPUs, CPUs and library versions were not tested.

## Layout

```
lab/         data, measures, results files and figure style
chapter1/    one script per experiment or figure
tests/       test_quoted_numbers.py compares the chapters with results/, test_known_answers.py checks the code
results/     one JSON per experiment: numbers, configuration, seed, script, library versions, device, date
checkpoints/ trained weights and saved samples, so that figures need no training
figures/     the figures of the course, as the scripts write them
```

The SVG schematics are written for inlining in the course pages, which style their classes and typeset their labels with KaTeX. Opened alone they show raw TeX labels, so read them in the chapters.

## License

The code is under the [MIT License](LICENSE). The text of the course is not part of this repository.
