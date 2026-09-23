# 07_plot_accuracy.py
# Plot training accuracy by epoch -- the one thing the book's own version
# of this section never does, since it discards the history dict via
# `_ = train2(...)`. Validation accuracy is plotted alongside it, since
# train2 already computes it at no extra cost, matching the "Learning
# curves" style used elsewhere in the book (cell 93): training accuracy
# is an average over the epoch, so it's plotted at the epoch's midpoint;
# validation accuracy is measured after the epoch, so it's plotted at
# the epoch's end.
#
# This only needs the history saved by 06_train.py, not the model or
# data loaders -- in the final notebook, this cell just reuses the
# `history` variable already sitting in memory from the previous cell.

import json
import numpy as np
import matplotlib.pyplot as plt

with open("training_history.json") as f:
    history = json.load(f)

n_epochs = len(history["train_metrics"])

plt.plot(np.arange(n_epochs) + 0.5, history["train_metrics"], ".--",
         label="Training")
plt.plot(np.arange(n_epochs) + 1.0, history["valid_metrics"], ".-",
         label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.title("Image Classifier Learning Curves")
plt.legend()
plt.savefig("training_accuracy.png", dpi=150, bbox_inches="tight")
print("Saved training_accuracy.png")
