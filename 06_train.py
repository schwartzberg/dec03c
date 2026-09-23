import json
from pathlib import Path
import runpy


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    data = runpy.run_path(str(repo_root / "02_data_loaders.py"))
    components = runpy.run_path(str(repo_root / "04_model.py"))
    train = runpy.run_path(str(repo_root / "05_training_function.py"))["train"]

    print("Training for 20 epochs...", flush=True)
    history = train(
        model=components["model"],
        train_loader=data["train_loader"],
        valid_loader=data["valid_loader"],
        loss_fn=components["loss_fn"],
        optimizer=components["optimizer"],
        metric=components["accuracy"],
        device=components["device"],
        epochs=20,
    )

    history_path = repo_root / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    print(f"Final training accuracy: {history['train_metrics'][-1]:.4%}")
    print(f"Final validation accuracy: {history['valid_metrics'][-1]:.4%}")
    print(f"History saved to {history_path.name}")
