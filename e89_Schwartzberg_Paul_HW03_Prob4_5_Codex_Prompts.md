# Problems 4 & 5 — Codex Cloud Prompts

**Setup (once, before Problem 4):**
1. Go to `chatgpt.com/codex`.
2. Connect GitHub if not already connected, and grant the Codex GitHub App access to `schwartzberg/dec03c`.
3. Start a new Codex Cloud task on that repository, based off `main`. If Codex lets you name the working branch, call it `codex/hw03-prob4-5`; otherwise note whatever it auto-generates.
4. Requires a ChatGPT plan with Codex cloud access (Plus/Pro/Business/Edu/Enterprise).

These files aren't in the repo yet at this point (`01_load_data.py` etc. all belong to Claude's branch, `claude/exciting-einstein-p34a0s` -- Codex is building its own independent set on its own branch).

---

## Problem 4 — incremental (send these one at a time, in the same task thread, after each finishes)

**Request 1**
```
Goal: Load the FashionMNIST dataset and split it into training, validation and test sets, as the first step of building an image classifier.
Context: This is for a PyTorch image-classification exercise (FashionMNIST). Use torchvision.datasets.FashionMNIST with root="datasets", download=True, and a transform pipeline that converts images to float32 tensors scaled to [0,1] (torchvision.transforms.v2: ToImage() then ToDtype(torch.float32, scale=True)). Split the 60,000-image training set into 55,000 for training and 5,000 for validation using torch.utils.data.random_split with torch.manual_seed(42) set immediately before the split. Keep the 10,000-image test set separate, untouched.
Constraints: Save this as 01_load_data.py in the repo root. Print the resulting sizes of all three sets.
Done when: the script runs cleanly and prints 55,000 / 5,000 / 10,000.
```

**Request 2**
```
Goal: Wrap the three datasets from the previous step in DataLoaders.
Context: batch size 32 for all three; shuffle only the training loader, with torch.manual_seed(42) set immediately before creating it.
Constraints: save as 02_data_loaders.py.
Done when: it runs and prints the batch counts for each loader.
```

**Request 3**
```
Goal: Inspect one training sample.
Context: print its tensor shape, dtype, and class name (FashionMNIST's dataset object exposes a .classes list of the 10 category names).
Constraints: save as 03_inspect_sample.py.
Done when: it prints a shape of [1, 28, 28], dtype float32, and a valid class name.
```

**Request 4**
```
Goal: Define the image classifier model, its loss function, optimizer and accuracy metric.
Context: an MLP: Flatten, then Linear(784, 300), ReLU, Linear(300, 100), ReLU, Linear(100, 10). Use torch.manual_seed(42) immediately before creating the model. Loss: nn.CrossEntropyLoss(). Optimizer: SGD with lr=0.1. Metric: torchmetrics.Accuracy(task="multiclass", num_classes=10). Pick the fastest available device (CUDA, then MPS, then CPU fallback) and move the model and metric onto it.
Constraints: save as 04_model.py.
Done when: it prints the model summary and a total trainable-parameter count.
```

**Request 5**
```
Goal: Write a training function that records training loss, training accuracy, and validation accuracy after every epoch, not just loss.
Context: for each epoch: iterate the training loader, do the usual forward/backward/step/zero_grad, accumulate the loss and update the accuracy metric; after the epoch, compute and store the mean training loss and training accuracy, then evaluate validation accuracy the same way (a separate no_grad pass over the validation loader). Return a history dict with keys train_losses, train_metrics, valid_metrics, each a list with one entry per epoch.
Constraints: save as 05_training_function.py. Don't run training yet, just define the function.
Done when: the file defines the function without executing it.
```

**Request 6**
```
Goal: Actually run training, for real, for 20 epochs.
Context: use the function and objects from the previous steps. Keep the returned history -- don't discard it.
Constraints: save as 06_train.py. Print the final training and validation accuracy.
Done when: it runs to completion (expect a few minutes on CPU) with no errors, and both final accuracy numbers print.
```

**Request 7**
```
Goal: Plot training accuracy across epochs -- this is the one piece of the process that needs to be added on top of a basic training loop.
Context: use the history dict's train_metrics list (and valid_metrics alongside it, for a fuller learning-curve chart) against epoch number, with matplotlib. Label axes, add a legend and a title.
Constraints: save as 07_plot_accuracy.py.
Done when: it produces a chart showing both curves rising and roughly leveling off.
```

**Request 8**
```
Goal: Show the trained model's predictions on a few real examples.
Context: take 3 images from the validation set, predict their classes, print predicted vs true class names, softmax probabilities, and the top-4 most likely classes per image. Also print the model's total parameter count again for confirmation.
Constraints: save as 08_evaluate_predictions.py.
Done when: it runs and the printed predictions look sane (matching or close to the true labels).
```

**Request 9 (wrap-up)**
```
Goal: assemble all 8 scripts above into a single, ordered, tested Jupyter notebook, then give a short summary of this whole conversation.
Context: this repository is public; every script above should already be committed on this task's branch.
Constraints: name the notebook e89_Schwartzberg_Paul_HW03_Prob4.ipynb; each cell or group of cells needs a label (markdown heading or comment) saying which step it's from; execute it top to bottom yourself to confirm it actually works before finishing; export it to HTML via jupyter nbconvert --to html; commit and push both files.
Done when: both files exist, committed and pushed, and you've given a short summary of the requests in this conversation and what each one produced.
```

---

## Problem 5 — one long prompt (start a brand-new, separate Codex Cloud task for this -- don't continue Problem 4's thread)

```
Goal: In one pass, build a complete, tested PyTorch image classifier for FashionMNIST and wrap it into a single Jupyter notebook, adding one specific piece beyond a standard training loop: a plot of training accuracy across epochs.

Context:
- Dataset: FashionMNIST via torchvision.datasets.FashionMNIST, transform pipeline converts images to float32 tensors scaled to [0,1] (ToImage() then ToDtype(torch.float32, scale=True)). Split the 60,000-image training set into 55,000 training / 5,000 validation via torch.utils.data.random_split with torch.manual_seed(42) set immediately before splitting; keep the 10,000-image test set held out separately.
- DataLoaders: batch size 32 for all three sets; shuffle only the training loader (seed 42 immediately before creating it).
- Model: an MLP -- Flatten, Linear(784, 300), ReLU, Linear(300, 100), ReLU, Linear(100, 10) -- with torch.manual_seed(42) set immediately before construction. Move it to the fastest available device (CUDA, then MPS, then CPU fallback).
- Loss: nn.CrossEntropyLoss(). Optimizer: plain SGD, lr=0.1. Metric: torchmetrics.Accuracy(task="multiclass", num_classes=10).
- Training: 20 epochs. Track training loss, training accuracy, and validation accuracy after every epoch (not just loss) in a history dict.
- The specific addition this task requires: after training, plot training accuracy against epoch using matplotlib (validation accuracy alongside it is a nice bonus, not required). Most basic training loops only track/print loss -- this explicitly needs the accuracy curve plotted, not just loss.
- Also show real predictions on a few validation images: predicted vs true class name, softmax probabilities, top-4 most likely classes.

Constraints:
- Student name for the notebook's byline: Paul Schwartzberg.
- Output file: e89_Schwartzberg_Paul_HW03_Prob5.ipynb, plus its HTML export via jupyter nbconvert --to html.
- Every cell or group of cells needs a clear markdown heading or comment saying what it does.
- Commit and push both files to this repository once you've executed the notebook yourself, top to bottom, and confirmed it runs with no errors.
- You may iterate and correct yourself as needed -- that's expected, not a one-shot-or-fail test.

Done when: the notebook and its HTML export are committed and pushed, the notebook runs clean top to bottom, shows a real (not fabricated) training-accuracy plot and real prediction results, and you report back the final training and validation accuracy actually achieved.
```
