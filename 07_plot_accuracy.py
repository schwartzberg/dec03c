import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    history = json.loads((repo_root / "training_history.json").read_text(encoding="utf-8"))
    epochs = range(1, len(history["train_metrics"]) + 1)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(epochs, history["train_metrics"], marker="o", label="Training")
    ax.plot(epochs, history["valid_metrics"], marker="s", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("FashionMNIST: training and validation accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xticks(list(epochs))
    ax.grid(alpha=0.25)
    ax.legend()

    output_path = repo_root / "training_accuracy.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Chart saved to {output_path.name}")
