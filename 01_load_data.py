import torch
import torchvision
from torchvision.transforms import v2


transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=transform
)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=transform
)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000]
)

print(f"Training images:   {len(train_data):,}")
print(f"Validation images: {len(valid_data):,}")
print(f"Test images:       {len(test_data):,}")
