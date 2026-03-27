"""
run_part4.py — Train 3 kiến trúc tokenization (Phần 4)

Sử dụng:
    python scripts/run_part4.py
    python scripts/run_part4.py --epochs 10
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_cifar100_loaders, get_device
from src.models_part4 import CNNTransformerHybrid, SpatialTokenViT, ChannelTokenViT
from src.train import fit, load_best_model
from src.utils import (get_param_count, get_predictions, compute_metrics,
                       plot_multi_curves, plot_comparison_bar, print_results_table,
                       save_metrics_json)


def main():
    parser = argparse.ArgumentParser(description="Train Part 4 models")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    DEVICE = get_device()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(base, "results", "checkpoints")
    plot_dir = os.path.join(base, "results", "plots")
    metric_dir = os.path.join(base, "results", "metrics")
    for d in [ckpt_dir, plot_dir, metric_dir]:
        os.makedirs(d, exist_ok=True)

    print("Loading CIFAR-100...")
    train_loader, val_loader, test_loader, _ = get_cifar100_loaders(batch_size=128)

    # SpatialTokenViT cần batch nhỏ hơn do memory
    train_small, val_small, test_small, _ = get_cifar100_loaders(batch_size=32)

    model_configs = [
        {
            "name": "CNNTransformerHybrid",
            "factory": lambda: CNNTransformerHybrid(100),
            "lr": 1e-3,
            "loaders": (train_loader, val_loader, test_loader),
        },
        {
            "name": "SpatialTokenViT",
            "factory": lambda: SpatialTokenViT(100),
            "lr": 1e-4,
            "loaders": (train_small, val_small, test_small),
            "note": "batch_size=32 vì attention matrix 1024×1024",
        },
        {
            "name": "ChannelTokenViT",
            "factory": lambda: ChannelTokenViT(100),
            "lr": 3e-4,
            "loaders": (train_loader, val_loader, test_loader),
        },
    ]

    histories = {}
    results = {}

    for cfg in model_configs:
        name = cfg["name"]
        note = cfg.get("note", "")
        tr, vl, te = cfg["loaders"]
        ckpt_path = os.path.join(ckpt_dir, f"{name.lower()}.pt")
        history_path = os.path.join(metric_dir, f"{name.lower()}_history.json")

        print(f"\n{'=' * 60}")
        print(f"Training: {name} | Epochs: {args.epochs}")
        if note:
            print(f"Note: {note}")
        model = cfg["factory"]()
        print(f"Params: {get_param_count(model)}")
        print(f"{'=' * 60}")

        model = model.to(DEVICE)
        hist = fit(model, tr, vl, {
            "epochs": args.epochs, "lr": cfg["lr"],
            "device": DEVICE, "save_path": ckpt_path,
        })
        histories[name] = hist
        save_metrics_json(hist, history_path)

        # Evaluate
        model_cls = {"CNNTransformerHybrid": CNNTransformerHybrid,
                     "SpatialTokenViT": SpatialTokenViT,
                     "ChannelTokenViT": ChannelTokenViT}[name]
        m = load_best_model(model_cls(100), ckpt_path, DEVICE)
        preds, labels = get_predictions(m, te, DEVICE)
        metrics = compute_metrics(preds, labels)
        results[name] = {
            "test_acc": metrics["accuracy"],
            "val_acc": max(hist["val_acc"]),
            "f1_macro": metrics["f1_macro"],
            "params": get_param_count(m),
        }
        print(f"Test accuracy: {metrics['accuracy']*100:.2f}%")

    save_metrics_json(results, os.path.join(metric_dir, "part4_results.json"))
    print("\n\n=== KẾT QUẢ PHẦN 4 ===")
    print_results_table(results)

    plot_multi_curves(list(histories.values()), list(histories.keys()),
                      title="So sánh 3 kiến trúc tokenization (Phần 4)",
                      save_path=os.path.join(plot_dir, "part4_comparison_curves.png"))
    plot_comparison_bar(results, metric="test_acc",
                        title="Test Accuracy — Phần 4",
                        save_path=os.path.join(plot_dir, "part4_bar.png"))


if __name__ == "__main__":
    main()
