from pathlib import Path
import matplotlib.pyplot as plt
import csv

def save_history_csv(out_dir: Path, history: dict, filename: str = "history.csv"):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(history.keys())
    lengths = [len(history.get(k, [])) for k in keys]
    max_len = max(lengths) if lengths else 0
    rows = []
    for i in range(max_len):
        row = []
        for k in keys:
            vals = history.get(k, [])
            row.append(vals[i] if i < len(vals) else "")
        rows.append(row)

    csv_path = out_dir / filename
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerows(rows)

def _plot_line(xs, ys, title, xlabel, ylabel, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

# Buat kurva training/validation berdasarkan history dict
def plot_training_curves(out_dir: Path, history: dict, tag: str = "val"):
    epochs = history.get("epoch", None)
    if epochs is None:
        n = len(history.get("train_loss", []))
        epochs = list(range(1, n + 1))

    out_dir.mkdir(parents=True, exist_ok=True)

    if "train_loss" in history:
        _plot_line(epochs, history["train_loss"], "Training Loss", "Epoch", "Loss",
                   out_dir / "curve_train_loss.png")

    if f"{tag}_dice" in history:
        _plot_line(epochs, history[f"{tag}_dice"], f"{tag.upper()} Dice", "Epoch", "Dice",
                   out_dir / f"curve_{tag}_dice.png")
    if f"{tag}_iou" in history:
        _plot_line(epochs, history[f"{tag}_iou"], f"{tag.upper()} IoU", "Epoch", "IoU",
                   out_dir / f"curve_{tag}_iou.png")
    if "lr" in history:
        _plot_line(epochs, history["lr"], "Learning Rate", "Epoch", "LR",
                   out_dir / "curve_lr.png")
