# 06_train.py
# Train the classifier for 20 epochs, keeping the per-epoch history
# (the book's own version discards it: `_ = train2(...)`).
#
# This script also saves the history and the trained weights to disk.
# That's only so later standalone scripts (07, 08) can be tested without
# re-running the full 20 epochs each time. Inside the final notebook this
# save/reload isn't needed at all: every cell shares one kernel, so
# `history` and `model` are simply still in memory for the next cell.

exec(open("05_training_function.py").read())

import json

n_epochs = 20
history = train2(model, optimizer, xentropy, accuracy, train_loader,
                  valid_loader, n_epochs)

with open("training_history.json", "w") as f:
    json.dump(history, f)
torch.save(model.state_dict(), "model_weights.pt")

print(f"Final train accuracy: {history['train_metrics'][-1]:.4f}")
print(f"Final valid accuracy: {history['valid_metrics'][-1]:.4f}")
