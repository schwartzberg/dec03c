# 08_evaluate_predictions.py
# Inspect predictions on 3 validation images: predicted vs true class,
# softmax probabilities, top-4 guesses, and the total parameter count.
# Mirrors the book's own inspection cells for this section.
#
# Loads model_weights.pt saved by 06_train.py instead of retraining --
# in the final notebook this cell just reuses the trained `model` still
# sitting in memory from the training cell.

exec(open("04_model.py").read())

model.load_state_dict(torch.load("model_weights.pt", map_location=device))
model.eval()

import torch.nn.functional as F

X_new, y_new = next(iter(valid_loader))
X_new = X_new[:3].to(device)
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1)

predicted_classes = [train_and_valid_data.classes[i] for i in y_pred]
true_classes = [train_and_valid_data.classes[i] for i in y_new[:3]]

print("Predicted:", predicted_classes)
print("True:     ", true_classes)
print("All correct!" if predicted_classes == true_classes
      else "Not all correct.")

y_proba = F.softmax(y_pred_logits, dim=1)
print("Probabilities (rounded):")
print(y_proba.round(decimals=3))

y_top4_values, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_values, dim=1)
print("Top-4 class indices per image:")
print(y_top4_indices)
print("Top-4 probabilities per image:")
print(y_top4_probas.round(decimals=3))

n_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {n_params:,}")
