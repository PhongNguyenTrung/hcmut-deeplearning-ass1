"""
models.py — Khởi tạo mô hình phát hiện đối tượng

Hai mô hình chính:
1. YOLOv8n  — one-stage detector (ultralytics), anchor-free, 3.2M params
2. Faster R-CNN ResNet-50 FPN — two-stage detector (torchvision), 41.8M params

Cả hai đều dùng pretrained weights trên COCO, sau đó fine-tune trên Pascal VOC 2012.
"""

import torch
import torch.nn as nn

from .data import NUM_CLASSES

# ─── Faster R-CNN ─────────────────────────────────────────────────────────────


def get_faster_rcnn(
    num_classes: int = NUM_CLASSES + 1,  # +1 cho background (class 0)
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 3,
) -> nn.Module:
    """
    Tải Faster R-CNN ResNet-50 FPN pretrained trên COCO từ torchvision.
    Thay thế box predictor head để phù hợp với num_classes của VOC.

    Kiến trúc:
        Backbone: ResNet-50 + FPN (Feature Pyramid Network)
        RPN: Region Proposal Network — đề xuất ~300 vùng có thể chứa object
        ROI Head: phân loại và tinh chỉnh bounding box cho từng vùng

    Args:
        num_classes: số lớp bao gồm background (VOC = 20 + 1 = 21)
        pretrained_backbone: dùng ResNet-50 weights pretrained trên ImageNet
        trainable_backbone_layers: số lớp backbone được cập nhật trong fine-tune
            0 = freeze toàn bộ backbone, 5 = train toàn bộ, 3 = train layer3+layer4+fpn

    Returns:
        model: nn.Module sẵn sàng để fine-tune
    """
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # Thay head: giữ nguyên backbone + RPN, chỉ đổi classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# ─── YOLOv8 ───────────────────────────────────────────────────────────────────


def get_yolov8(
    model_size: str = "n",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> "ultralytics.YOLO":  # noqa: F821
    """
    Tải YOLOv8 pretrained (mặc định trên COCO) từ ultralytics.

    Model sizes: n (nano, 3.2M), s (small, 11.2M), m (medium, 25.9M)
    Khuyến nghị dùng 'n' cho demo nhanh, 's' hoặc 'm' nếu có GPU.

    Lưu ý: ultralytics tự động thay đổi số output class khi gọi model.train()
    với tham số data=yaml_path. Hàm này chỉ load model và kiểm tra.

    Args:
        model_size: "n", "s", hoặc "m"
        num_classes: số lớp (dùng để kiểm tra sau khi train)
        pretrained: nếu True, dùng COCO pretrained weights

    Returns:
        ultralytics.YOLO object
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Cần cài ultralytics: pip install ultralytics>=8.0.0"
        )

    model_name = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
    model = YOLO(model_name)
    return model


# ─── Model Info ───────────────────────────────────────────────────────────────


def get_model_info(model, model_name: str) -> dict:
    """
    Trả về thông tin tóm tắt về mô hình: tổng params, trainable params.

    Args:
        model: nn.Module (hoặc ultralytics YOLO)
        model_name: tên hiển thị

    Returns:
        dict {"name", "total_params", "trainable_params"}
    """
    try:
        # Faster R-CNN và các nn.Module thông thường
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except AttributeError:
        # ultralytics YOLO
        try:
            total = sum(p.numel() for p in model.model.parameters())
            trainable = sum(
                p.numel() for p in model.model.parameters() if p.requires_grad
            )
        except Exception:
            return {"name": model_name, "total_params": "N/A", "trainable_params": "N/A"}

    def fmt(n):
        return f"{n / 1e6:.1f}M"

    return {
        "name": model_name,
        "total_params": fmt(total),
        "trainable_params": fmt(trainable),
    }
