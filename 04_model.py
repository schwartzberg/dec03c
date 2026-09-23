import torch
from torch import nn
import torchmetrics


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.manual_seed(42)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 300),
    nn.ReLU(),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Device: {device}")
print(model)
print(f"Trainable parameters: {trainable_parameters:,}")
