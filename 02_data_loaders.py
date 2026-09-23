# 02_data_loaders.py
# Wrap the training, validation and test sets in DataLoaders (batch size 32).
# The exec() line below only stands in for running the previous notebook
# cell first; inside the notebook itself this script becomes its own cell
# and simply runs after 01_load_data.py's cell, sharing the same variables.

exec(open("01_load_data.py").read())

from torch.utils.data import DataLoader

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

print(f"Training batches:   {len(train_loader)}")
print(f"Validation batches: {len(valid_loader)}")
print(f"Test batches:       {len(test_loader)}")
