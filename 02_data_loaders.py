from pathlib import Path
import runpy

import torch
from torch.utils.data import DataLoader


datasets = runpy.run_path(str(Path(__file__).with_name("01_load_data.py")))
train_data = datasets["train_data"]
valid_data = datasets["valid_data"]
test_data = datasets["test_data"]

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Training batches:   {len(train_loader):,}")
print(f"Validation batches: {len(valid_loader):,}")
print(f"Test batches:       {len(test_loader):,}")
