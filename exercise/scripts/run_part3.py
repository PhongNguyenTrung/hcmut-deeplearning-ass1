"""
run_part3.py — Train Custom ViT và so sánh với PyTorch ViT (Phần 3)

Sử dụng:
    python scripts/run_part3.py
    python scripts/run_part3.py --epochs 10  # test nhanh
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_cifar100_loaders, get_device
from src.models_part1 import SimpleViT
from src.models_part3 import CustomViT
from src.train import fit, load_best_model
from src.utils import (get_param_count, get_predictions, compute_metrics,
                       plot_multi_curves, plot_comparison_bar, print_results_table,
                       save_metrics_json, load_metrics_json)


def main():
    parser = argparse.ArgumentParser(description="Train Part 3 models")
    parser.add_argument("--epochs", type=int, default=100)
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

    histories = {}
    results = {}

    models_to_train = [
        ("SimpleViT (PyTorch)", SimpleViT(100), "vit"),
        ("CustomViT (Tự xây)", CustomViT(100), "custom_vit"),
    ]

    for display_name, model, key in models_to_train:
        ckpt_path = os.path.join(ckpt_dir, f"{key}.pt")
        history_path = os.path.join(metric_dir, f"{key}_history.json")

        # Nếu đã có history (từ Part 2), dùng lại
        if key == "vit" and os.path.exists(history_path):
            print(f"\nLoaded {display_name} history từ Part 2")
            histories[display_name] = load_metrics_json(history_path)
        else:
            print(f"\n{'=' * 60}")
            print(f"Training: {display_name} | Epochs: {args.epochs}")
            print(f"Params: {get_param_count(model)}")
            print(f"{'=' * 60}")

            model = model.to(DEVICE)
            hist = fit(model, train_loader, val_loader, {
                "epochs": args.epochs, "lr": 3e-4,
                "device": DEVICE, "save_path": ckpt_path,
            })
            histories[display_name] = hist
            save_metrics_json(hist, history_path)

        # Evaluate
        model_cls = SimpleViT if key == "vit" else CustomViT
        m = load_best_model(model_cls(100), ckpt_path, DEVICE)
        preds, labels = get_predictions(m, test_loader, DEVICE)
        metrics = compute_metrics(preds, labels)
        results[display_name] = {
            "test_acc": metrics["accuracy"],
            "val_acc": max(histories[display_name]["val_acc"]),
            "f1_macro": metrics["f1_macro"],
            "params": get_param_count(m),
        }

    save_metrics_json(results, os.path.join(metric_dir, "part3_results.json"))
    print("\n\n=== KẾT QUẢ PHẦN 3 ===")
    print_results_table(results)

    plot_multi_curves(list(histories.values()), list(histories.keys()),
                      title="SimpleViT (PyTorch) vs CustomViT (Tự xây)",
                      save_path=os.path.join(plot_dir, "part3_comparison_curves.png"))
    plot_comparison_bar(results, metric="test_acc",
                        title="Test Accuracy — Phần 3",
                        save_path=os.path.join(plot_dir, "part3_bar.png"))


if __name__ == "__main__":
    main()
