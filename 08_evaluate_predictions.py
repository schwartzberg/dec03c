from pathlib import Path
import runpy

import torch


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    model_path = repo_root / "fashion_mnist_model.pt"
    if not model_path.is_file():
        raise FileNotFoundError("Run 06_train.py first to save the trained model weights.")

    data = runpy.run_path(str(repo_root / "01_load_data.py"))
    components = runpy.run_path(str(repo_root / "04_model.py"))
    model, device = components["model"], components["device"]
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    valid_data = data["valid_data"]
    classes = valid_data.dataset.classes
    samples = [valid_data[index] for index in range(3)]
    images = torch.stack([image for image, _ in samples]).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(images), dim=1).cpu()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    for index, (probs, (_, true_label)) in enumerate(zip(probabilities, samples)):
        predicted_label = probs.argmax().item()
        print(f"\nValidation image {index}: predicted {classes[predicted_label]}; true {classes[true_label]}")
        print("Softmax probabilities:")
        for class_name, probability in zip(classes, probs.tolist()):
            print(f"  {class_name}: {probability:.6f}")

        top_probs, top_labels = probs.topk(4)
        print("Top 4 classes:")
        for rank, (label, probability) in enumerate(zip(top_labels.tolist(), top_probs.tolist()), start=1):
            print(f"  {rank}. {classes[label]}: {probability:.6f}")
