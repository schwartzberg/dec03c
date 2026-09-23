# Problem 1 — Dialogue Summary

**Paul Schwartzberg — CSCI E-89, HW03**

**Goal:** using Claude Code connected to [schwartzberg/dec03c](https://github.com/schwartzberg/dec03c), incrementally rebuild the "Building an Image Classifier with PyTorch" section of `10_neural_nets_with_pytorch.ipynb` (FashionMNIST, Chapter 10, Geron), and add the one thing that section doesn't do on its own: a plot of training accuracy across epochs.

**Process:** before writing anything, we read the reference notebook's actual cells (111-132) rather than assuming their content, to pin down the exact architecture, hyperparameters, and -- critically -- the exact gap to fill. The book's own version trains the model but discards its per-epoch history (`_ = train2(...)`), so it never plots anything. That became the one deliberate addition on top of the template.

From there, the model was built as eight separate requests, one script per request, each tested by actually running it before being committed and pushed to GitHub -- nothing went into the repository unverified:

1. `01_load_data.py` -- download FashionMNIST, split 55,000 / 5,000 / 10,000 into train / validation / test.
2. `02_data_loaders.py` -- wrap each set in a DataLoader, batch size 32.
3. `03_inspect_sample.py` -- confirm one image's shape, data type and class name.
4. `04_model.py` -- define the classifier (784 to 300 to 100 to 10, ReLU activations), its loss (cross-entropy), optimizer (SGD, lr=0.1) and accuracy metric.
5. `05_training_function.py` -- a training function that records training and validation accuracy every epoch, not just loss.
6. `06_train.py` -- the real 20-epoch run. Final training accuracy 92.9%, validation accuracy 87.9%.
7. `07_plot_accuracy.py` -- the addition this assignment asks for: training and validation accuracy plotted against epoch.
8. `08_evaluate_predictions.py` -- sample predictions on 3 validation images, softmax probabilities, top-4 guesses, and the parameter count (266,610).

**Result:** all eight scripts, tested individually, were reassembled into one notebook, `e89_Schwartzberg_Paul_HW03_Prob1.ipynb`, in execution order, each section labeled with and linked to its source script in the repository. The assembled notebook was then executed top to bottom as one run, to confirm it holds together as a single coherent pipeline and not just as separate working pieces -- it produced the same results either way, since every step is seeded.

**Repository:** https://github.com/schwartzberg/dec03c, branch `claude/exciting-einstein-p34a0s`.
