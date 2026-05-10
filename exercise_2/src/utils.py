"""
utils.py — Visualization, plots, và tiện ích cho Object Detection

Cung cấp:
- Vẽ bounding boxes lên ảnh
- Biểu đồ so sánh mAP giữa các mô hình
- Biểu đồ speed vs accuracy (FPS vs mAP)
- Biểu đồ per-class AP
- Lưu/tải metrics JSON
- Bảng kết quả đẹp trên terminal
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .data import VOC_CLASSES, IMAGENET_MEAN, IMAGENET_STD


# ─── Palette màu cho 20 lớp ───────────────────────────────────────────────────

_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
]


# ─── JSON utilities ───────────────────────────────────────────────────────────

def save_metrics_json(metrics: dict, path: str) -> None:
    """Lưu metrics dict ra file JSON, tạo thư mục nếu chưa có."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {path}")


def load_metrics_json(path: str) -> dict:
    """Tải metrics từ file JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Model info ───────────────────────────────────────────────────────────────

def get_param_count(model) -> str:
    """Trả về số parameters dạng '25.8M'."""
    try:
        n = sum(p.numel() for p in model.parameters())
    except AttributeError:
        try:
            n = sum(p.numel() for p in model.model.parameters())
        except Exception:
            return "N/A"
    return f"{n / 1e6:.1f}M"


# ─── Image Utilities ──────────────────────────────────────────────────────────

def denormalize_imagenet(img_tensor) -> np.ndarray:
    """
    Đảo ngược ImageNet normalization để visualize.
    img_tensor: Tensor [C, H, W] đã normalize → ndarray [H, W, C] trong [0, 1]
    """
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_detections(
    image,
    boxes: list,
    labels: list,
    scores: list = None,
    class_names: list = None,
    gt_boxes: list = None,
    gt_labels: list = None,
    score_threshold: float = 0.3,
    title: str = "",
    save_path: str = None,
    ax=None,
) -> None:
    """
    Vẽ bounding boxes lên ảnh.

    Predicted boxes: màu theo class, đậm
    Ground-truth boxes (nếu có): đường nét đứt màu đỏ

    Args:
        image: PIL Image, numpy array [H,W,C] [0,1], hoặc Tensor [C,H,W]
        boxes: list of [x1, y1, x2, y2] — predictions
        labels: list of int (1-indexed)
        scores: list of float confidence
        class_names: danh sách tên lớp (mặc định VOC_CLASSES)
        gt_boxes: list of [x1,y1,x2,y2] — ground truth
        gt_labels: list of int
        score_threshold: bỏ qua detection có score < threshold
        title: tiêu đề ảnh
        save_path: lưu file nếu không None
        ax: matplotlib Axes (để dùng trong subplot)
    """
    import torch
    if class_names is None:
        class_names = VOC_CLASSES

    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        img = denormalize_imagenet(image)
    elif hasattr(image, "numpy"):
        img = np.array(image) / 255.0
    else:
        img = np.array(image)
        if img.max() > 1.0:
            img = img / 255.0

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.imshow(img)
    ax.set_title(title or "Detections", fontsize=10)
    ax.axis("off")

    # Vẽ predictions
    for i, (box, label) in enumerate(zip(boxes, labels)):
        score = scores[i] if scores else 1.0
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cls_idx = int(label) - 1
        color = _PALETTE[cls_idx % len(_PALETTE)]
        rect = mpatches.Rectangle((x1, y1), w, h,
                                  linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"cls{label}"
        text = f"{name} {score:.2f}" if scores else name
        ax.text(x1, max(y1 - 2, 0), text,
                fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))

    # Vẽ ground truth
    if gt_boxes:
        for box, label in zip(gt_boxes, gt_labels or []):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = mpatches.Rectangle((x1, y1), w, h,
                                      linewidth=1.5, edgecolor="red",
                                      facecolor="none", linestyle="--")
            ax.add_patch(rect)

    if standalone:
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.show()
        plt.close()


# ─── Comparison Plots ─────────────────────────────────────────────────────────

def plot_map_comparison(results: dict, save_path: str = None) -> None:
    """
    Biểu đồ grouped bar: so sánh mAP@0.5 và mAP@0.5:0.95 giữa các mô hình.

    Args:
        results: {"YOLOv8n": {"mAP_50": 0.65, "mAP_50_95": 0.42}, "Faster R-CNN": {...}}
        save_path: đường dẫn lưu PNG
    """
    models = list(results.keys())
    map50 = [results[m].get("mAP_50", 0) * 100 for m in models]
    map5095 = [results[m].get("mAP_50_95", 0) * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, map50, width, label="mAP@0.5", color="#6366f1")
    bars2 = ax.bar(x + width / 2, map5095, width, label="mAP@0.5:0.95", color="#ec4899")

    ax.set_xlabel("Mô hình")
    ax.set_ylabel("mAP (%)")
    ax.set_title("So sánh mAP giữa YOLOv8n và Faster R-CNN")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_per_class_ap(results: dict, save_path: str = None) -> None:
    """
    Biểu đồ horizontal bar: AP@0.5 từng lớp, mỗi mô hình một màu.

    Chỉ vẽ các model có AP_per_class non-empty — model thiếu data sẽ bị bỏ qua
    để không tạo cột rỗng và legend gây hiểu lầm.

    Args:
        results: {"YOLOv8n": {"AP_per_class": {cls: float}}, "Faster R-CNN": {...}}
    """
    models_with_data = [
        m for m, r in results.items()
        if r.get("AP_per_class") and len(r["AP_per_class"]) > 0
    ]
    if not models_with_data:
        print("[plot_per_class_ap] Không có model nào có AP_per_class — skip plot.")
        return

    classes = VOC_CLASSES
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#ec4899", "#6366f1", "#10b981", "#f59e0b"]
    width = 0.8 / len(models_with_data)
    y = np.arange(n_classes)

    for i, (model_name, color) in enumerate(zip(models_with_data, colors)):
        ap_dict = results[model_name]["AP_per_class"]
        ap_vals = [ap_dict.get(c, 0) * 100 for c in classes]
        offset = (i - len(models_with_data) / 2 + 0.5) * width
        ax.barh(y + offset, ap_vals, width, label=model_name, color=color, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(classes)
    ax.set_xlabel("AP@0.5 (%)")
    ax.set_title("AP per class — Pascal VOC 2012")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    plt.tight_layout()
    _save_or_show(save_path)


def plot_speed_accuracy_tradeoff(results: dict, save_path: str = None) -> None:
    """
    Scatter plot: FPS (trục x) vs mAP@0.5 (trục y).
    Mỗi mô hình là một điểm với label.
    Đây là biểu đồ quan trọng nhất để thể hiện trade-off one-stage vs two-stage.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#6366f1", "#ec4899", "#10b981", "#f59e0b"]

    for (model_name, data), color in zip(results.items(), colors):
        fps = data.get("fps", 0)
        map50 = data.get("mAP_50", 0) * 100
        ax.scatter(fps, map50, s=200, color=color, zorder=5, label=model_name)
        ax.annotate(
            model_name,
            (fps, map50),
            textcoords="offset points", xytext=(8, 4),
            fontsize=9, color=color, fontweight="bold",
        )

    ax.set_xlabel("FPS (Frames per Second) — cao hơn = nhanh hơn")
    ax.set_ylabel("mAP@0.5 (%) — cao hơn = chính xác hơn")
    ax.set_title("Speed vs Accuracy Trade-off\nOne-Stage (YOLO) vs Two-Stage (Faster R-CNN)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_loss_curves(history: dict, title: str = "Training Loss", save_path: str = None) -> None:
    """
    Vẽ train_loss và val_loss theo epoch.

    Args:
        history: {"train_loss": [...], "val_loss": [...]}
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


# ─── Console Table ────────────────────────────────────────────────────────────

def print_detection_results_table(results: dict) -> None:
    """
    In bảng so sánh kết quả ra terminal.

    results: {"Model1": {"mAP_50": float, "mAP_50_95": float, "fps": float, "params": str}}
    """
    header = f"{'Mô hình':<20} {'mAP@0.5':>10} {'mAP@0.5:0.95':>14} {'FPS (CPU)':>10} {'Params':>10}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, data in results.items():
        map50 = data.get("mAP_50", 0) * 100
        map5095 = data.get("mAP_50_95", 0) * 100
        fps = data.get("fps", "N/A")
        params = data.get("params", "N/A")
        fps_str = f"{fps:.1f}" if isinstance(fps, (int, float)) else str(fps)
        print(f"{name:<20} {map50:>9.1f}% {map5095:>13.1f}% {fps_str:>10} {str(params):>10}")
    print(sep)


# ─── Internal Helper ──────────────────────────────────────────────────────────

def _save_or_show(save_path: str) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Saved] {save_path}")
    else:
        plt.show()
    plt.close()
