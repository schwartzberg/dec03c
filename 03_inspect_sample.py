from pathlib import Path
import runpy


datasets = runpy.run_path(str(Path(__file__).with_name("01_load_data.py")))
train_data = datasets["train_data"]

image, label = train_data[0]
class_name = train_data.dataset.classes[label]

print(f"Shape: {list(image.shape)}")
print(f"Dtype: {image.dtype}")
print(f"Class: {class_name}")
