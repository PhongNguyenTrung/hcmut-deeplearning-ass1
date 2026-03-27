"""
run_part1_2.py — Train và đánh giá 4 mô hình cơ bản (Phần 1 & 2)

Sử dụng:
    python scripts/run_part1_2.py                     # Train tất cả 4 mô hình
    python scripts/run_part1_2.py --model softmax     # Chỉ train Softmax
    python scripts/run_part1_2.py --model mlp         # Chỉ train MLP
    python scripts/run_part1_2.py --model cnn         # Chỉ train CNN
    python scripts/run_part1_2.py --model vit         # Chỉ train ViT
    python scripts/run_part1_2.py --epochs 5          # Số epochs tùy chỉnh (để test nhanh)
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data import get_cifar100_loaders, get_device
from src.models_part1 import SoftmaxRegression, MLP, SimpleCNN, SimpleViT
from src.train import fit, load_best_model
from src.utils import (get_param_count, get_predictions, compute_metrics,
                       plot_training_curves, plot_comparison_bar, print_results_table,
                       save_metrics_json)

# ── Cấu hình từng mô hình ──────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "softmax": {
        "name": "SoftmaxRegression",
        "factory": lambda: SoftmaxRegression(100),
        "epochs": 30,
        "lr": 0.1,
        "batch_size": 256,
    },
    "mlp": {
        "name": "MLP",
        "factory": lambda: MLP(100),
        "epochs": 50,
        "lr": 1e-3,
        "batch_size": 128,
    },
    "cnn": {
        "name": "SimpleCNN",
        "factory": lambda: SimpleCNN(100),
        "epochs": 50,
        "lr": 1e-3,
        "batch_size": 128,
    },
    "vit": {
        "name": "SimpleViT",
        "factory": lambda: SimpleViT(100),
        "epochs": 100,
        "lr": 3e-4,
        "batch_size": 128,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Train Part 1 & 2 models on CIFAR-100")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        default="all", help="Mô hình cần train")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override số epochs (để test nhanh)")
    args = parser.parse_args()

    DEVICE = get_device()
    print(f"Thiết bị: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    # Thư mục kết quả
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(base, "results", "checkpoints")
    plot_dir = os.path.join(base, "results", "plots")
    metric_dir = os.path.join(base, "results", "metrics")
    for d in [ckpt_dir, plot_dir, metric_dir]:
        os.makedirs(d, exist_ok=True)

    # Chọn mô hình cần train
    to_train = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    # Tải data một lần (dùng chung batch_size mặc định)
    print("\nLoading CIFAR-100...")
    loaders_cache = {}

    results = {}

    for model_key in to_train:
        cfg = MODEL_CONFIGS[model_key]
        name = cfg["name"]
        epochs = args.epochs if args.epochs else cfg["epochs"]
        batch_size = cfg["batch_size"]

        print(f"\n{'=' * 60}")
        print(f"Model: {name} | Epochs: {epochs} | LR: {cfg['lr']} | Batch: {batch_size}")
        print(f"{'=' * 60}")

        # Tải data nếu batch_size khác
        if batch_size not in loaders_cache:
            loaders_cache[batch_size] = get_cifar100_loaders(batch_size=batch_size)
        train_loader, val_loader, test_loader, class_names = loaders_cache[batch_size]

        # Tạo model
        model = cfg["factory"]().to(DEVICE)
        print(f"Parameters: {get_param_count(model)}")

        # Train
        ckpt_path = os.path.join(ckpt_dir, f"{model_key}.pt")
        history = fit(model, train_loader, val_loader, {
            "epochs": epochs,
            "lr": cfg["lr"],
            "device": DEVICE,
            "save_path": ckpt_path,
        })

        # Lưu history
        save_metrics_json(history, os.path.join(metric_dir, f"{model_key}_history.json"))

        # Vẽ training curves
        plot_training_curves(history, title=name,
                             save_path=os.path.join(plot_dir, f"{model_key}_curves.png"))

        # Evaluate trên test set
        model = load_best_model(model, ckpt_path, DEVICE)
        preds, labels = get_predictions(model, test_loader, DEVICE)
        metrics = compute_metrics(preds, labels)

        results[name] = {
            "test_acc": metrics["accuracy"],
            "val_acc": max(history["val_acc"]),
            "f1_macro": metrics["f1_macro"],
            "params": get_param_count(model),
        }

        print(f"\nTest accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"F1-macro:      {metrics['f1_macro']*100:.2f}%")

    # Lưu và in bảng kết quả tổng hợp
    save_metrics_json(results, os.path.join(metric_dir, "part1_2_results.json"))
    print("\n\n=== KẾT QUẢ PHẦN 1 & 2 ===")
    print_results_table(results)

    if len(results) > 1:
        plot_comparison_bar(results, metric="test_acc",
                            title="Test Accuracy — Phần 1 & 2 (CIFAR-100)",
                            save_path=os.path.join(plot_dir, "part1_2_bar.png"))


if __name__ == "__main__":
    main()
