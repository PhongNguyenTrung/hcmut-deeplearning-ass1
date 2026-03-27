"""
utils.py — Metrics, visualizations, và so sánh mô hình

Cung cấp các hàm tiện ích để:
- Tính accuracy và F1-macro
- Vẽ training curves (loss + accuracy theo epoch)
- Vẽ biểu đồ so sánh nhiều mô hình
- In bảng kết quả
- Lưu/tải metrics JSON
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(preds, labels):
    """
    Tính accuracy và F1-macro từ mảng predictions và labels.

    Args:
        preds: List/array predictions (integers 0..99)
        labels: List/array nhãn đúng

    Returns:
        dict: {"accuracy": float, "f1_macro": float}
    """
    preds = np.array(preds)
    labels = np.array(labels)
    accuracy = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": float(accuracy), "f1_macro": float(f1)}


@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Thu thập toàn bộ predictions và labels từ DataLoader.

    Returns:
        preds: numpy array [N]
        labels: numpy array [N]
    """
    model.eval()
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    return np.array(all_preds), np.array(all_labels)


def get_param_count(model):
    """Đếm số tham số của model, trả về chuỗi như '2.3M'."""
    total = sum(p.numel() for p in model.parameters())
    if total >= 1_000_000:
        return f"{total / 1_000_000:.1f}M"
    elif total >= 1_000:
        return f"{total / 1_000:.1f}K"
    return str(total)


# ─── Lưu / tải metrics ────────────────────────────────────────────────────────

def save_metrics_json(metrics_dict, path):
    """Lưu dict metrics ra file JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)


def load_metrics_json(path):
    """Tải metrics từ file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Vẽ training curves ───────────────────────────────────────────────────────

def plot_training_curves(history, title="Training Curves", save_path=None):
    """
    Vẽ 2 biểu đồ: (1) Loss theo epoch, (2) Accuracy theo epoch.

    Args:
        history: dict với keys "train_loss", "val_loss", "train_acc", "val_acc"
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu hình PNG (None = chỉ hiện)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Subplot 1: Loss ──
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Subplot 2: Accuracy ──
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", markersize=3, label="Train Acc")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-o", markersize=3, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Annotation: val_acc tốt nhất
    best_epoch = np.argmax(history["val_acc"]) + 1
    best_acc = max(history["val_acc"]) * 100
    ax2.annotate(
        f"Best: {best_acc:.1f}%\n(epoch {best_epoch})",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch + max(1, len(epochs) // 10), best_acc - 5),
        arrowprops=dict(arrowstyle="->", color="green"),
        color="green", fontsize=9,
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()


def plot_multi_curves(histories, labels, title="Model Comparison", save_path=None):
    """
    Vẽ val_accuracy của nhiều model trên cùng 1 biểu đồ để so sánh trực quan.

    Args:
        histories: List các history dict
        labels: List tên model tương ứng
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    for hist, label, color in zip(histories, labels, colors):
        epochs = range(1, len(hist["val_acc"]) + 1)
        ax.plot(epochs, [a * 100 for a in hist["val_acc"]],
                "-o", markersize=3, label=label, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


# ─── Biểu đồ so sánh nhiều mô hình ───────────────────────────────────────────

def plot_comparison_bar(results_dict, metric="accuracy", title="Model Comparison", save_path=None):
    """
    Vẽ biểu đồ cột ngang so sánh nhiều mô hình.

    Args:
        results_dict: {"ModelName": {"accuracy": 0.55, "f1_macro": 0.50, "params": "2.3M"}, ...}
        metric: "accuracy" hoặc "f1_macro"
        title: Tiêu đề
    """
    names = list(results_dict.keys())
    values = [results_dict[n].get(metric, 0) * 100 for n in names]

    # Sắp xếp theo giá trị tăng dần
    sorted_pairs = sorted(zip(values, names), reverse=False)
    values, names = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.6)))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)

    # Thêm giá trị vào cuối thanh
    for bar, val in zip(bars, values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left", fontsize=9)

    ax.set_xlabel(f"{metric.replace('_', ' ').title()} (%)")
    ax.set_title(title)
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    plt.close()


# ─── Bảng kết quả ─────────────────────────────────────────────────────────────

def print_results_table(results_dict):
    """
    In bảng kết quả dạng text ra màn hình.

    results_dict format:
    {
        "ModelName": {
            "test_acc": 0.55,
            "val_acc": 0.54,
            "f1_macro": 0.50,
            "params": "2.3M",
            "train_time": 120.5  # seconds per epoch (optional)
        }
    }
    """
    header = f"{'Model':<25} | {'Test Acc':>9} | {'Val Acc':>8} | {'F1-macro':>9} | {'Params':>8}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # Sắp xếp theo test_acc giảm dần
    sorted_items = sorted(results_dict.items(),
                          key=lambda x: x[1].get("test_acc", 0), reverse=True)

    for name, m in sorted_items:
        test_acc = m.get("test_acc", 0)
        val_acc = m.get("val_acc", 0)
        f1 = m.get("f1_macro", 0)
        params = m.get("params", "—")
        print(f"{name:<25} | {test_acc*100:>8.2f}% | {val_acc*100:>7.2f}% | {f1*100:>8.2f}% | {params:>8}")

    print("=" * len(header))


# ─── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(preds, labels, class_names=None, top_n=20, save_path=None):
    """
    Vẽ confusion matrix (chỉ top_n lớp có nhiều lỗi nhất nếu > 20 lớp).

    Args:
        top_n: Số lớp hiển thị (để biểu đồ không quá nhỏ với 100 lớp)
    """
    cm = confusion_matrix(labels, preds)
    n_classes = cm.shape[0]

    if n_classes > top_n and class_names is not None:
        # Chọn top_n lớp có nhiều lỗi nhất (nhiều False Positive + False Negative)
        errors = cm.sum(axis=1) - cm.diagonal()
        top_indices = np.argsort(errors)[-top_n:][::-1]
        cm = cm[np.ix_(top_indices, top_indices)]
        selected_names = [class_names[i] for i in top_indices]
        title_suffix = f" (Top {top_n} confused classes)"
    else:
        selected_names = class_names if class_names else [str(i) for i in range(n_classes)]
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=(len(selected_names) <= 20), fmt="d", cmap="Blues",
                xticklabels=selected_names, yticklabels=selected_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix{title_suffix}")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
