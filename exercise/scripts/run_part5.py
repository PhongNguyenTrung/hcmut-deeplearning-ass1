"""
run_part5.py — Train LSTM/GRU với các cách biểu diễn chuỗi (Phần 5)

Sử dụng:
    python scripts/run_part5.py
    python scripts/run_part5.py --epochs 10
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_sequence_loaders, get_device
from src.models_part5 import ImageLSTM, ImageGRU
from src.train import fit, load_best_model
from src.utils import (get_param_count, get_predictions, compute_metrics,
                       plot_multi_curves, plot_comparison_bar, print_results_table,
                       save_metrics_json)


def main():
    parser = argparse.ArgumentParser(description="Train Part 5 LSTM/GRU models")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    DEVICE = get_device()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(base, "results", "checkpoints")
    plot_dir = os.path.join(base, "results", "plots")
    metric_dir = os.path.join(base, "results", "metrics")
    for d in [ckpt_dir, plot_dir, metric_dir]:
        os.makedirs(d, exist_ok=True)

    # 4 cấu hình: 2 seq modes × 2 RNN types
    configs = [
        ("LSTM-row",    "lstm", "row"),
        ("LSTM-patch4", "lstm", "patch4"),
        ("GRU-row",     "gru",  "row"),
        ("GRU-patch4",  "gru",  "patch4"),
    ]

    histories = {}
    results = {}

    for name, rnn_type, seq_mode in configs:
        print(f"\n{'=' * 60}")
        print(f"Training: {name} | seq_mode={seq_mode} | Epochs: {args.epochs}")
        print(f"{'=' * 60}")

        print(f"Loading {seq_mode} loaders...")
        tr, vl, te, seq_len, input_size = get_sequence_loaders(seq_mode=seq_mode, batch_size=128)
        print(f"seq_len={seq_len}, input_size={input_size}")

        RNNCls = ImageLSTM if rnn_type == "lstm" else ImageGRU
        model = RNNCls(input_size=input_size).to(DEVICE)
        print(f"Params: {get_param_count(model)}")

        ckpt_path = os.path.join(ckpt_dir, f"{name.lower()}.pt")
        history_path = os.path.join(metric_dir, f"{name.lower()}_history.json")

        hist = fit(model, tr, vl, {
            "epochs": args.epochs, "lr": 1e-3,
            "device": DEVICE, "save_path": ckpt_path,
        })
        histories[name] = hist
        save_metrics_json(hist, history_path)

        # Evaluate
        m = load_best_model(RNNCls(input_size), ckpt_path, DEVICE)
        preds, labels = get_predictions(m, te, DEVICE)
        metrics = compute_metrics(preds, labels)
        results[name] = {
            "test_acc": metrics["accuracy"],
            "val_acc": max(hist["val_acc"]),
            "f1_macro": metrics["f1_macro"],
            "params": get_param_count(m),
        }
        print(f"Test accuracy: {metrics['accuracy']*100:.2f}%")

    save_metrics_json(results, os.path.join(metric_dir, "part5_results.json"))
    print("\n\n=== KẾT QUẢ PHẦN 5 ===")
    print_results_table(results)

    plot_multi_curves(list(histories.values()), list(histories.keys()),
                      title="LSTM/GRU × Sequence Mode (Phần 5)",
                      save_path=os.path.join(plot_dir, "part5_comparison_curves.png"))
    plot_comparison_bar(results, metric="test_acc",
                        title="Test Accuracy — Phần 5 (LSTM/GRU)",
                        save_path=os.path.join(plot_dir, "part5_bar.png"))


if __name__ == "__main__":
    main()
