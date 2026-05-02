"""
evaluate.py — Đánh giá mô hình Object Detection

Metrics:
- mAP@0.5:        AP tại IoU threshold = 0.5 (tiêu chuẩn VOC)
- mAP@0.5:0.95:   AP tại 10 IoU thresholds từ 0.5 đến 0.95 (tiêu chuẩn COCO)
- FPS:            tốc độ inference (frames per second)
- TP/FP/FN:       phân tích lỗi

Dùng pycocotools (thư viện chuẩn cho detection evaluation).
"""

import time

import torch
import numpy as np
from torch.utils.data import DataLoader

from .data import VOC_CLASSES, NUM_CLASSES


# ─── Faster R-CNN Inference ───────────────────────────────────────────────────

@torch.no_grad()
def predict_frcnn(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    score_threshold: float = 0.05,
    max_batches: int = None,
) -> tuple:
    """
    Chạy inference Faster R-CNN trên toàn bộ loader.

    Args:
        model: Faster R-CNN (eval mode)
        loader: DataLoader trả về (list_images, list_targets)
        device: thiết bị
        score_threshold: chỉ giữ detection có score >= threshold

    Returns:
        all_preds: list of dicts {"boxes": [[x1,y1,x2,y2],...], "scores": [...], "labels": [...], "image_id": int}
        all_targets: list of dicts {"boxes": [...], "labels": [...], "image_id": int}
    """
    model.eval()
    model.to(device)

    all_preds, all_targets = [], []
    image_counter = 0

    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = [img.to(device) for img in images]
        outputs = model(images)

        for pred, tgt in zip(outputs, targets):
            image_id = tgt["image_id"].item() if isinstance(tgt["image_id"], torch.Tensor) \
                else image_counter

            # Filter by score
            keep = pred["scores"] >= score_threshold
            all_preds.append({
                "image_id": image_id,
                "boxes": pred["boxes"][keep].cpu().numpy().tolist(),
                "scores": pred["scores"][keep].cpu().numpy().tolist(),
                "labels": pred["labels"][keep].cpu().numpy().tolist(),
            })
            all_targets.append({
                "image_id": image_id,
                "boxes": tgt["boxes"].cpu().numpy().tolist(),
                "labels": tgt["labels"].cpu().numpy().tolist(),
            })
            image_counter += 1

    return all_preds, all_targets


def predict_yolov8(
    model_path: str,
    image_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
) -> tuple:
    """
    Chạy inference YOLOv8 trên thư mục ảnh.

    Args:
        model_path: đường dẫn đến best.pt
        image_dir:  thư mục chứa ảnh val
        conf_threshold: confidence threshold
        iou_threshold:  NMS IoU threshold

    Returns:
        all_preds, all_targets — cùng format với predict_frcnn
        (Lưu ý: all_targets cần load riêng từ label files hoặc từ VOCDetectionDataset)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Cần cài: pip install ultralytics>=8.0.0")

    model = YOLO(model_path)
    results = model.predict(
        source=image_dir,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    all_preds = []
    for i, r in enumerate(results):
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            all_preds.append({"image_id": i, "boxes": [], "scores": [], "labels": []})
            continue
        all_preds.append({
            "image_id": i,
            "boxes": boxes.xyxy.cpu().numpy().tolist(),
            "scores": boxes.conf.cpu().numpy().tolist(),
            # YOLO labels: 0-indexed → convert to 1-indexed để thống nhất với Faster R-CNN
            "labels": (boxes.cls.cpu().numpy().astype(int) + 1).tolist(),
        })

    return all_preds


# ─── mAP Evaluation via pycocotools ──────────────────────────────────────────

def compute_map_coco(
    preds: list,
    targets: list,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """
    Tính mAP@0.5 và mAP@0.5:0.95 bằng pycocotools.

    Tại sao dùng pycocotools thay vì tự tính?
    - mAP@0.5:0.95 cần evaluate ở 10 IoU thresholds → rất phức tạp nếu tự viết
    - pycocotools là thư viện chuẩn, số liệu so được với paper

    Args:
        preds:   list of {"image_id", "boxes" (xyxy), "scores", "labels" (1-indexed)}
        targets: list of {"image_id", "boxes" (xyxy), "labels" (1-indexed)}
        num_classes: số lớp (không gồm background)

    Returns:
        dict {"mAP_50", "mAP_50_95", "AP_per_class"}
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        raise ImportError("Cần cài: pip install pycocotools>=2.0.6")

    # Xây dựng COCO ground-truth object từ targets
    gt_annotations = []
    ann_id = 0
    gt_images = []
    gt_image_ids = set()

    for tgt in targets:
        img_id = tgt["image_id"]
        if img_id not in gt_image_ids:
            gt_images.append({"id": img_id})
            gt_image_ids.add(img_id)

        for box, label in zip(tgt["boxes"], tgt["labels"]):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            gt_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],  # COCO dùng xywh
                "area": float(w * h),
                "iscrowd": 0,
            })
            ann_id += 1

    gt_coco_dict = {
        "images": gt_images,
        "annotations": gt_annotations,
        "categories": [{"id": i + 1, "name": VOC_CLASSES[i]} for i in range(num_classes)],
    }

    # Xây dựng dt (detection) annotations
    dt_annotations = []
    for pred in preds:
        img_id = pred["image_id"]
        for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            dt_annotations.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score),
            })

    if not dt_annotations:
        print("[Eval] Không có predictions nào! Kiểm tra lại model.")
        return {"mAP_50": 0.0, "mAP_50_95": 0.0, "AP_per_class": {}}

    import io
    coco_gt = COCO()
    coco_gt.dataset = gt_coco_dict
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(dt_annotations)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Suppress output khi summarize
    import sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    coco_eval.summarize()
    sys.stdout = old_stdout

    stats = coco_eval.stats
    # stats[0] = mAP@0.5:0.95, stats[1] = mAP@0.5
    result = {
        "mAP_50": float(stats[1]),
        "mAP_50_95": float(stats[0]),
        "AP_per_class": {},
    }

    # Per-class AP@0.5
    coco_eval_per = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval_per.params.iouThrs = np.array([0.5])

    for cat_id in range(1, num_classes + 1):
        coco_eval_per.params.catIds = [cat_id]
        coco_eval_per.evaluate()
        coco_eval_per.accumulate()

        sys.stdout = io.StringIO()
        coco_eval_per.summarize()
        sys.stdout = old_stdout

        class_name = VOC_CLASSES[cat_id - 1]
        ap = float(coco_eval_per.stats[0]) if coco_eval_per.stats[0] > 0 else 0.0
        result["AP_per_class"][class_name] = ap

    return result


# ─── FPS Benchmark ────────────────────────────────────────────────────────────

def measure_fps(
    model,
    model_type: str,
    image_size: tuple = (640, 640),
    n_warmup: int = 10,
    n_runs: int = 50,
    device: str = "cpu",
) -> dict:
    """
    Benchmark tốc độ inference của mô hình.

    Args:
        model: nn.Module (Faster R-CNN) hoặc ultralytics YOLO
        model_type: "frcnn" hoặc "yolo"
        image_size: (H, W) — kích thước ảnh dummy
        n_warmup: số lần warm-up (không đo)
        n_runs: số lần đo thực tế
        device: thiết bị benchmark

    Returns:
        dict {"fps", "ms_per_image", "device", "image_size"}
    """
    H, W = image_size
    # frcnn expects ImageNet-normalized floats; yolo expects [0,1] range
    dummy_frcnn = torch.randn(1, 3, H, W).to(device)
    dummy_yolo  = torch.rand(1, 3, H, W).to(device)

    if model_type == "frcnn":
        model.eval()
        model.to(device)

        def run_once():
            with torch.no_grad():
                _ = model([dummy_frcnn[0]])

    elif model_type == "yolo":
        def run_once():
            _ = model.predict(dummy_yolo, verbose=False)
    else:
        raise ValueError(f"model_type phải là 'frcnn' hoặc 'yolo', nhận được '{model_type}'")

    # Warm-up
    for _ in range(n_warmup):
        run_once()

    if device == "cuda":
        torch.cuda.synchronize()

    # Đo thời gian
    t0 = time.perf_counter()
    for _ in range(n_runs):
        run_once()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_image = (elapsed / n_runs) * 1000
    fps = n_runs / elapsed

    return {
        "fps": round(fps, 1),
        "ms_per_image": round(ms_per_image, 1),
        "device": device,
        "image_size": f"{H}x{W}",
    }


# ─── Error Analysis ───────────────────────────────────────────────────────────

def compute_iou(box1: list, box2: list) -> float:
    """Tính IoU giữa hai bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def get_detection_errors(
    preds: list,
    targets: list,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> dict:
    """
    Phân tích lỗi detection: đếm TP, FP, FN trên toàn dataset.

    TP: prediction có cùng class với GT và IoU >= threshold
    FP: prediction không match với bất kỳ GT nào (wrong class hoặc IoU < threshold)
    FN: GT box không được detect (bị bỏ sót)

    Returns:
        dict {
            "TP": int, "FP": int, "FN": int,
            "precision": float, "recall": float, "f1": float,
            "fp_per_class": {class_name: int},
            "fn_per_class": {class_name: int},
        }
    """
    TP = FP = FN = 0
    fp_per_class = {c: 0 for c in VOC_CLASSES}
    fn_per_class = {c: 0 for c in VOC_CLASSES}

    # Map image_id → targets
    target_map = {t["image_id"]: t for t in targets}

    for pred in preds:
        img_id = pred["image_id"]
        tgt = target_map.get(img_id, {"boxes": [], "labels": []})

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]
        gt_matched = [False] * len(gt_boxes)

        # Filter predictions by score
        pred_pairs = [
            (b, s, l)
            for b, s, l in zip(pred["boxes"], pred["scores"], pred["labels"])
            if s >= score_threshold
        ]
        # Sort by score descending
        pred_pairs = sorted(pred_pairs, key=lambda x: -x[1])

        for p_box, p_score, p_label in pred_pairs:
            best_iou = 0.0
            best_j = -1
            for j, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_matched[j]:
                    continue
                if int(p_label) != int(g_label):
                    continue
                iou = compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0:
                TP += 1
                gt_matched[best_j] = True
            else:
                FP += 1
                class_name = VOC_CLASSES[int(p_label) - 1] if 1 <= int(p_label) <= NUM_CLASSES else "unknown"
                if class_name in fp_per_class:
                    fp_per_class[class_name] += 1

        # FN: GT boxes chưa được match
        for j, matched in enumerate(gt_matched):
            if not matched:
                FN += 1
                label = gt_labels[j] if j < len(gt_labels) else 0
                class_name = VOC_CLASSES[int(label) - 1] if 1 <= int(label) <= NUM_CLASSES else "unknown"
                if class_name in fn_per_class:
                    fn_per_class[class_name] += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fp_per_class": fp_per_class,
        "fn_per_class": fn_per_class,
    }
