# 03_inspect_sample.py
# Look at one training image: its tensor shape, data type and class name.

exec(open("02_data_loaders.py").read())

X_sample, y_sample = train_data[0]

print(f"Image shape: {X_sample.shape}")   # [channels, rows, columns]
print(f"Image dtype: {X_sample.dtype}")
print(f"Class name:  {train_and_valid_data.classes[y_sample]}")
