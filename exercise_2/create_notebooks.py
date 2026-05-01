"""
create_notebooks.py — Tạo tất cả notebooks cho exercise_2 (Object Detection)

Chạy:
    python create_notebooks.py
→ Tạo/ghi đè 5 file .ipynb trong thư mục notebooks/
"""

import json
import os

# ─── Helpers ──────────────────────────────────────────────────────────────────


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def code(text, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": text.strip(),
    }


def nb(*cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "deeplearning-ass1",
                "language": "python",
                "name": "deeplearning-ass1",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": list(cells),
    }


# ─── Notebook 1: EDA ──────────────────────────────────────────────────────────

NB1 = nb(
    md("""# Notebook 1 — EDA: Khám phá tập dữ liệu Pascal VOC 2012

**Bài tập lớn số 2 · CO5085 · HCMUT 2025-2026**

## Mục tiêu
Trong notebook này ta sẽ tìm hiểu:
1. Pascal VOC 2012 là gì? Tại sao chọn dataset này?
2. Phân phối các lớp đối tượng (class distribution)
3. Thống kê bounding boxes: số lượng, kích thước, tỷ lệ khung hình
4. Visualize mẫu ảnh với ground-truth boxes

## Pascal VOC 2012
- 20 lớp đối tượng thường gặp trong ảnh thực tế
- ~11,530 ảnh train / ~2,788 ảnh val
- Annotation dạng XML với bounding boxes (xmin, ymin, xmax, ymax)
- Benchmark chuẩn trong nhiều năm cho object detection
"""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import datasets

from src.data import VOC_CLASSES, get_voc_stats, get_device

print("Device:", get_device())
print("VOC Classes:", len(VOC_CLASSES), "classes")
print(VOC_CLASSES)"""),

    md("""## 1. Download và kiểm tra dataset

Torchvision sẽ tự download Pascal VOC 2012 (~2GB) lần đầu tiên chạy.
"""),

    code("""# Download dataset (chỉ cần chạy 1 lần)
voc_train = datasets.VOCDetection(root='../data/voc', year='2012',
                                   image_set='train', download=True)
voc_val   = datasets.VOCDetection(root='../data/voc', year='2012',
                                   image_set='val', download=True)

print(f"Train: {len(voc_train)} ảnh")
print(f"Val:   {len(voc_val)} ảnh")
"""),

    md("## 2. Thống kê dataset"),

    code("""stats = get_voc_stats('../data/voc')

print(f"Train images: {stats['n_train']}")
print(f"Val images:   {stats['n_val']}")
print(f"\\nBoxes per image: mean={stats.get('boxes_per_image_stats',{}).get('mean','?'):.1f}, "
      f"max={stats.get('boxes_per_image_stats',{}).get('max','?')}")
"""),

    md("## 3. Phân phối lớp (Class Distribution)"),

    code("""class_counts = stats['class_counts']
classes = list(class_counts.keys())
counts = list(class_counts.values())

# Sắp xếp theo số lượng giảm dần
sorted_pairs = sorted(zip(counts, classes), reverse=True)
counts_sorted, classes_sorted = zip(*sorted_pairs)

fig, ax = plt.subplots(figsize=(14, 5))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))
bars = ax.bar(classes_sorted, counts_sorted, color=colors)
ax.set_xticklabels(classes_sorted, rotation=45, ha='right')
ax.set_ylabel('Số instances')
ax.set_title('Phân phối các lớp trong Pascal VOC 2012 (Train+Val)')
ax.grid(axis='y', alpha=0.3)

# Thêm số trên mỗi bar
for bar, cnt in zip(bars, counts_sorted):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(cnt), ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('../results/plots/class_distribution.png', dpi=120, bbox_inches='tight')
plt.show()

print("\\nNhận xét: 'person' chiếm đa số (~30%), một số lớp hiếm như 'boat', 'sheep'")
"""),

    md("## 4. Thống kê Bounding Boxes"),

    code("""from xml.etree import ElementTree as ET
from pathlib import Path

voc_base = Path('../data/voc/VOCdevkit/VOC2012')
ann_dir = voc_base / 'Annotations'

all_widths, all_heights, boxes_per_img = [], [], []

for ann_file in list(ann_dir.glob('*.xml'))[:3000]:  # Sample 3000 ảnh cho nhanh
    tree = ET.parse(ann_file)
    root = tree.getroot()
    size = root.find('size')
    img_w = float(size.find('width').text)
    img_h = float(size.find('height').text)

    objs = root.findall('object')
    boxes_per_img.append(len(objs))
    for obj in objs:
        bb = obj.find('bndbox')
        w = float(bb.find('xmax').text) - float(bb.find('xmin').text)
        h = float(bb.find('ymax').text) - float(bb.find('ymin').text)
        all_widths.append(w / img_w)   # normalized
        all_heights.append(h / img_h)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(boxes_per_img, bins=20, color='#6366f1', edgecolor='white')
axes[0].set_title('Số boxes mỗi ảnh')
axes[0].set_xlabel('Số bounding boxes')
axes[0].set_ylabel('Số ảnh')

axes[1].hist(all_widths, bins=50, color='#ec4899', edgecolor='white')
axes[1].set_title('Chiều rộng boxes (normalized)')
axes[1].set_xlabel('Width / Image Width')

axes[2].scatter(all_widths[:1000], all_heights[:1000], alpha=0.3, s=10, color='#10b981')
axes[2].set_title('Width vs Height (normalized)')
axes[2].set_xlabel('Width')
axes[2].set_ylabel('Height')

plt.tight_layout()
plt.savefig('../results/plots/bbox_stats.png', dpi=120, bbox_inches='tight')
plt.show()
"""),

    md("## 5. Visualize mẫu ảnh với Ground-Truth Boxes"),

    code("""from src.utils import visualize_detections
from src.data import VOCDetectionDataset, get_val_transforms

ds = VOCDetectionDataset('../data/voc', year='2012', image_set='val',
                          transforms=get_val_transforms())

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img_t, target = ds[i * 30]
    visualize_detections(
        image=img_t,
        boxes=target['boxes'].numpy().tolist(),
        labels=target['labels'].numpy().tolist(),
        gt_boxes=None,
        title=f"Val image #{i*30}",
        ax=ax,
    )

plt.suptitle('Mẫu ảnh Pascal VOC 2012 với Ground-Truth Boxes', fontsize=12)
plt.tight_layout()
plt.savefig('../results/plots/sample_images.png', dpi=100, bbox_inches='tight')
plt.show()
"""),

    md("""## Tóm tắt

| Đặc điểm | Giá trị |
|-----------|---------|
| Số lớp | 20 |
| Train images | ~11,530 |
| Val images | ~2,788 |
| Boxes/image (trung bình) | ~2.7 |
| Lớp phổ biến nhất | person |
| Lớp ít nhất | (xem biểu đồ) |

**Lý do chọn Pascal VOC 2012:**
- Kích thước vừa phải, phù hợp để fine-tune trong thời gian hợp lý
- 20 lớp đa dạng, đủ khó để đánh giá model
- Benchmark lâu đời, kết quả so sánh được với nhiều paper
- Torchvision hỗ trợ sẵn (không cần tự parse)
"""),
)


# ─── Notebook 2: Data Pipeline ────────────────────────────────────────────────

NB2 = nb(
    md("""# Notebook 2 — Data Pipeline: DataLoader, Augmentation, Format Conversion

**Bài tập lớn số 2 · CO5085 · HCMUT 2025-2026**

## Mục tiêu
1. Hiểu tại sao hai mô hình cần hai format dữ liệu khác nhau
2. Demo VOCDetectionDataset và custom collate_fn
3. Visualize augmentation cho ảnh + bounding boxes
4. Chuyển đổi VOC XML → YOLO txt format
"""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data import (VOCDetectionDataset, get_train_transforms, get_val_transforms,
                      collate_fn, get_frcnn_loaders, prepare_yolo_dataset, VOC_CLASSES)
from src.utils import visualize_detections, denormalize_imagenet
"""),

    md("""## 1. Tại sao cần hai format khác nhau?

| Mô hình | Format ảnh | Format annotation |
|---------|-----------|------------------|
| **Faster R-CNN** | `list[Tensor[C,H,W]]` in [0,1] | `list[dict]` với `boxes` [N,4] tuyệt đối |
| **YOLOv8** | File ảnh JPEG | File `.txt` với `class xc yc w h` normalized |

Faster R-CNN (torchvision) nhận list vì ảnh có kích thước khác nhau — **không thể stack**.
YOLOv8 (ultralytics) có pipeline riêng đọc từ file.
"""),

    code("""# Demo VOCDetectionDataset
ds = VOCDetectionDataset('../data/voc', year='2012', image_set='val',
                          transforms=get_val_transforms())

img, target = ds[0]
print("Image shape:", img.shape)
print("Boxes:", target['boxes'])
print("Labels:", target['labels'])
print("Label names:", [VOC_CLASSES[l-1] for l in target['labels'].tolist()])
"""),

    md("## 2. Custom collate_fn"),

    code("""from torch.utils.data import DataLoader

# Thử tạo DataLoader với collate_fn tùy chỉnh
loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
imgs, targets = next(iter(loader))

print(f"Type imgs: {type(imgs)} | len: {len(imgs)}")
print(f"Type targets: {type(targets)} | len: {len(targets)}")
print(f"Image 0 shape: {imgs[0].shape}")
print(f"Target 0 boxes shape: {targets[0]['boxes'].shape}")
print()
print("Tại sao không dùng torch.stack?")
print("→ Mỗi ảnh VOC có kích thước khác nhau!")
print(f"  img[0]: {imgs[0].shape[1:]}")
print(f"  img[1]: {imgs[1].shape[1:]}")
"""),

    md("## 3. Augmentation Visualization"),

    code("""from torchvision import transforms

# Augmentation áp dụng cho training
train_tf = get_train_transforms()
val_tf = get_val_transforms()

ds_train = VOCDetectionDataset('../data/voc', year='2012', image_set='val',
                                transforms=None)

img_pil, target = ds_train[5]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Ảnh gốc
import numpy as np
axes[0].imshow(img_pil)
axes[0].set_title('Ảnh gốc (PIL)')
axes[0].axis('off')

# Val transform (chỉ normalize)
img_val = val_tf(img_pil)
axes[1].imshow(denormalize_imagenet(img_val))
axes[1].set_title('Val transform\\n(ToTensor + Normalize)')
axes[1].axis('off')

# Train transform (augmentation + normalize)
import random; random.seed(42); torch.manual_seed(42)
img_aug = train_tf(img_pil)
axes[2].imshow(denormalize_imagenet(img_aug))
axes[2].set_title('Train transform\\n(Flip + ColorJitter + Normalize)')
axes[2].axis('off')

plt.suptitle('So sánh Val vs Train Transforms', fontsize=12)
plt.tight_layout()
plt.savefig('../results/plots/augmentation_demo.png', dpi=120)
plt.show()
print("Chú ý: Faster R-CNN xử lý augmentation bounding box separately.")
"""),

    md("## 4. YOLO Format Conversion"),

    code("""# Chuyển VOC XML → YOLO txt
yaml_path = prepare_yolo_dataset(
    voc_root='../data/voc',
    output_dir='../data/voc_yolo',
    splits=['train', 'val'],
)
print(f"\\nYAML config: {yaml_path}")
"""),

    code("""# Xem nội dung file voc.yaml
print(open(yaml_path).read())
"""),

    code("""# Xem ví dụ một file label YOLO
from pathlib import Path
lbl_dir = Path('../data/voc_yolo/labels/val')
sample_lbl = next(lbl_dir.glob('*.txt'))
print(f"File: {sample_lbl.name}")
print("Nội dung (class xc yc w h - tất cả normalized [0,1]):")
print(sample_lbl.read_text()[:500])
print()
print("So sánh với VOC XML (absolute pixels): xmin, ymin, xmax, ymax")
"""),

    md("""## Tóm tắt Data Pipeline

```
Pascal VOC XML (absolute xyxy)
        │
        ├─→ VOCDetectionDataset ──→ list[(img_tensor, target_dict)] ──→ Faster R-CNN
        │
        └─→ prepare_yolo_dataset ──→ images/ + labels/*.txt + voc.yaml ──→ YOLOv8
```

**ImageNet Normalization** (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
- Cả Faster R-CNN lẫn YOLOv8 đều dùng backbone pretrained trên ImageNet
- Normalize giúp input distribution match với distribution lúc pretrain
"""),
)


# ─── Notebook 3: Training ─────────────────────────────────────────────────────

NB3 = nb(
    md("""# Notebook 3 — Huấn luyện & So sánh: YOLOv8 vs Faster R-CNN

**Bài tập lớn số 2 · CO5085 · HCMUT 2025-2026**

## Mục tiêu
1. Hiểu kiến trúc One-Stage (YOLO) vs Two-Stage (Faster R-CNN)
2. Fine-tune cả hai mô hình trên Pascal VOC 2012
3. So sánh quá trình training và kết quả

## Cài đặt flag
Đặt `TRAIN_MODE = True` để chạy training thực sự.
Đặt `TRAIN_MODE = False` để load kết quả đã có và chạy nhanh.
"""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

TRAIN_MODE = False  # Đổi thành True để chạy training thực sự

import torch
from src.data import get_frcnn_loaders, get_device
from src.models import get_faster_rcnn, get_yolov8, get_model_info
from src.train import fit_frcnn, train_yolov8, load_frcnn_checkpoint
from src.utils import print_detection_results_table, plot_loss_curves, load_metrics_json

device = get_device()
print("Device:", device)
"""),

    md("""## 1. Kiến trúc: One-Stage vs Two-Stage

### YOLOv8 (One-Stage)
```
Input image
    ↓
CSPDarknet backbone (feature extraction)
    ↓
FPN neck (multi-scale features)
    ↓
Detection head (đồng thời: classification + localization)
    ↓
Output: [N, 4+num_classes] per anchor-free grid cell
```
**Ưu điểm:** Nhanh (single forward pass)
**Nhược điểm:** Đôi khi kém chính xác hơn với small objects

### Faster R-CNN (Two-Stage)
```
Input image
    ↓
ResNet-50 backbone + FPN (feature extraction)
    ↓
RPN (Region Proposal Network) → ~300 proposal boxes
    ↓
ROI Align (crop features cho từng proposal)
    ↓
Box Head (classification + box regression)
    ↓
Output: detected objects với class và bbox
```
**Ưu điểm:** Chính xác hơn, đặc biệt với overlapping objects
**Nhược điểm:** Chậm hơn (two forward passes)
"""),

    code("""# Khởi tạo và so sánh mô hình
print("=== Model Info ===")
try:
    yolo = get_yolov8('n', pretrained=True)
    yolo_info = get_model_info(yolo, 'YOLOv8n')
    print(f"YOLOv8n:         {yolo_info['total_params']} params")
except Exception as e:
    print(f"YOLOv8n: không load được ({e})")

frcnn = get_faster_rcnn(num_classes=21, pretrained_backbone=True)
frcnn_info = get_model_info(frcnn, 'Faster R-CNN ResNet-50 FPN')
print(f"Faster R-CNN:    {frcnn_info['total_params']} params")
"""),

    md("## 2. Faster R-CNN — Loss Structure"),

    code("""# Demo: Faster R-CNN trả về dict losses (không phải logits)
from src.data import VOCDetectionDataset, get_val_transforms, collate_fn
from torch.utils.data import DataLoader

ds = VOCDetectionDataset('../data/voc', year='2012', image_set='val',
                          transforms=get_val_transforms())
loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

frcnn.train()
frcnn.to(device)
images, targets = next(iter(loader))
images = [img.to(device) for img in images]
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

with torch.no_grad():
    loss_dict = frcnn(images, targets)

print("Loss components từ Faster R-CNN:")
for k, v in loss_dict.items():
    print(f"  {k}: {v.item():.4f}")
print(f"  TOTAL: {sum(v.item() for v in loss_dict.values()):.4f}")
"""),

    md("## 3. Training"),

    code("""if TRAIN_MODE:
    # Fine-tune Faster R-CNN
    train_loader, val_loader = get_frcnn_loaders('../data/voc', batch_size=4, num_workers=2)
    model = get_faster_rcnn(num_classes=21)
    config = {'epochs': 10, 'lr': 0.005, 'device': device,
               'save_path': '../results/checkpoints/frcnn_voc.pth'}
    history = fit_frcnn(model, train_loader, val_loader, config)
    plot_loss_curves(history, 'Faster R-CNN Training Loss',
                     save_path='../results/plots/frcnn_loss.png')
else:
    print("[TRAIN_MODE=False] Bỏ qua training. Load kết quả đã có...")
    try:
        frcnn_results = load_metrics_json('../results/metrics/frcnn_results.json')
        if 'history' in frcnn_results:
            plot_loss_curves(frcnn_results['history'], 'Faster R-CNN Training Loss (loaded)')
    except FileNotFoundError:
        print("  Chưa có kết quả. Chạy: python scripts/run_finetune_frcnn.py")
"""),

    code("""if TRAIN_MODE:
    # Fine-tune YOLOv8
    from src.data import prepare_yolo_dataset
    yaml_path = prepare_yolo_dataset('../data/voc', '../data/voc_yolo')
    best_pt = train_yolov8(yaml_path, model_size='n', epochs=20,
                            project='../results', name='yolov8n_voc')
    print(f"YOLOv8 best checkpoint: {best_pt}")
else:
    print("[TRAIN_MODE=False] Bỏ qua YOLOv8 training.")
"""),

    md("## 4. Kết quả nhanh"),

    code("""# Load và hiển thị kết quả từ file JSON
try:
    yolo_r = load_metrics_json('../results/metrics/yolo_results.json')
    frcnn_r = load_metrics_json('../results/metrics/frcnn_results.json')
    results = {
        yolo_r.get('model', 'YOLOv8n'): yolo_r,
        frcnn_r.get('model', 'Faster R-CNN'): frcnn_r,
    }
    print_detection_results_table(results)
except FileNotFoundError:
    print("Chưa có kết quả. Cần chạy training trước.")
    print("  python scripts/run_finetune_yolo.py")
    print("  python scripts/run_finetune_frcnn.py")
"""),
)


# ─── Notebook 4: Results ──────────────────────────────────────────────────────

NB4 = nb(
    md("""# Notebook 4 — Kết quả Thực nghiệm: Phân tích và Thảo luận

**Bài tập lớn số 2 · CO5085 · HCMUT 2025-2026**

## Nội dung
1. Bảng số liệu tổng hợp
2. Biểu đồ so sánh mAP
3. Biểu đồ Speed vs Accuracy
4. Per-class AP analysis
5. Visualize kết quả định tính (qualitative)
6. Phân tích lỗi (TP/FP/FN)
7. Thảo luận và kết luận
"""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils import (load_metrics_json, print_detection_results_table,
                        plot_map_comparison, plot_speed_accuracy_tradeoff,
                        plot_per_class_ap, visualize_detections)
from src.data import VOCDetectionDataset, get_val_transforms, get_device, VOC_CLASSES

device = get_device()
print("Device:", device)
"""),

    md("## 1. Bảng số liệu tổng hợp"),

    code("""try:
    yolo_r = load_metrics_json('../results/metrics/yolo_results.json')
    frcnn_r = load_metrics_json('../results/metrics/frcnn_results.json')
    results = {
        yolo_r.get('model', 'YOLOv8n'): yolo_r,
        frcnn_r.get('model', 'Faster R-CNN'): frcnn_r,
    }
    print_detection_results_table(results)
    has_results = True
except FileNotFoundError:
    print("[WARN] Chưa có kết quả. Chạy training scripts trước.")
    has_results = False
"""),

    md("## 2. Biểu đồ so sánh mAP"),

    code("""if has_results:
    plot_map_comparison(results, save_path='../results/plots/map_comparison.png')
else:
    print("Cần kết quả training trước khi vẽ biểu đồ.")
"""),

    md("## 3. Speed vs Accuracy Trade-off"),

    code("""if has_results:
    plot_speed_accuracy_tradeoff(results, save_path='../results/plots/speed_accuracy.png')
    print()
    print("Nhận xét quan trọng:")
    print("→ One-stage (YOLO): nhanh hơn nhiều, phù hợp real-time (camera, video)")
    print("→ Two-stage (Faster R-CNN): chính xác hơn, phù hợp khi không cần real-time")
"""),

    md("## 4. Per-Class AP"),

    code("""if has_results and 'AP_per_class' in frcnn_r:
    plot_per_class_ap(results, save_path='../results/plots/per_class_ap.png')

    # Top 5 dễ nhất và khó nhất
    ap_frcnn = frcnn_r.get('AP_per_class', {})
    if ap_frcnn:
        sorted_ap = sorted(ap_frcnn.items(), key=lambda x: x[1], reverse=True)
        print("\\nTop 5 lớp dễ nhất (Faster R-CNN AP@0.5):")
        for cls, ap in sorted_ap[:5]:
            print(f"  {cls:15s}: {ap*100:.1f}%")
        print("\\nTop 5 lớp khó nhất:")
        for cls, ap in sorted_ap[-5:]:
            print(f"  {cls:15s}: {ap*100:.1f}%")
else:
    print("Cần per-class AP trong results JSON.")
"""),

    md("## 5. Visualize kết quả định tính"),

    code("""# Load model và visualize predictions
try:
    frcnn_path = '../results/checkpoints/frcnn_voc.pth'
    from src.models import get_faster_rcnn
    from src.train import load_frcnn_checkpoint

    model = get_faster_rcnn(num_classes=21)
    model = load_frcnn_checkpoint(model, frcnn_path, device)
    model.eval()

    ds = VOCDetectionDataset('../data/voc', year='2012', image_set='val',
                              transforms=get_val_transforms())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    import torch
    with torch.no_grad():
        for i, ax in enumerate(axes):
            img_t, target = ds[i * 50]
            output = model([img_t.to(device)])[0]

            keep = output['scores'] > 0.5
            visualize_detections(
                image=img_t,
                boxes=output['boxes'][keep].cpu().tolist(),
                labels=output['labels'][keep].cpu().tolist(),
                scores=output['scores'][keep].cpu().tolist(),
                gt_boxes=target['boxes'].tolist(),
                gt_labels=target['labels'].tolist(),
                title=f"Val #{i*50} — GT(đỏ/nét đứt) vs Pred(màu)",
                ax=ax,
            )

    plt.suptitle('Faster R-CNN Predictions trên Pascal VOC val', fontsize=12)
    plt.tight_layout()
    plt.savefig('../results/plots/qualitative_frcnn.png', dpi=100)
    plt.show()
except Exception as e:
    print(f"[WARN] Không thể visualize: {e}")
    print("Cần có checkpoint frcnn_voc.pth")
"""),

    md("## 6. Phân tích lỗi (Error Analysis)"),

    code("""try:
    from src.evaluate import predict_frcnn, get_detection_errors
    from src.data import get_frcnn_loaders

    model = get_faster_rcnn(num_classes=21)
    model = load_frcnn_checkpoint(model, '../results/checkpoints/frcnn_voc.pth', device)

    _, val_loader = get_frcnn_loaders('../data/voc', batch_size=4)
    preds, targets = predict_frcnn(model, val_loader, device)
    errors = get_detection_errors(preds, targets, iou_threshold=0.5, score_threshold=0.5)

    print("=== Error Analysis (Faster R-CNN) ===")
    print(f"  TP: {errors['TP']}")
    print(f"  FP: {errors['FP']}  (phát hiện sai)")
    print(f"  FN: {errors['FN']}  (bỏ sót)")
    print(f"  Precision: {errors['precision']*100:.1f}%")
    print(f"  Recall:    {errors['recall']*100:.1f}%")
    print(f"  F1:        {errors['f1']*100:.1f}%")

    # FN cao nhất (bỏ sót nhiều nhất)
    fn_sorted = sorted(errors['fn_per_class'].items(), key=lambda x: -x[1])[:5]
    print("\\nTop 5 lớp bị bỏ sót nhiều nhất (FN):")
    for cls, cnt in fn_sorted:
        if cnt > 0:
            print(f"  {cls}: {cnt}")
except Exception as e:
    print(f"Cần có checkpoint: {e}")
"""),

    md("""## 7. Thảo luận và Kết luận

### Kết quả

*(Điền số liệu thực sau khi chạy training)*

| Mô hình | mAP@0.5 | mAP@0.5:0.95 | FPS (CPU) | Params |
|---------|---------|--------------|-----------|--------|
| YOLOv8n | TBD | TBD | TBD | 3.2M |
| Faster R-CNN ResNet-50 FPN | TBD | TBD | TBD | 41.8M |

### Phân tích

**Độ chính xác:**
- Faster R-CNN có xu hướng chính xác hơn vì two-stage pipeline cho phép tập trung vào từng region
- YOLOv8 có thể kém hơn ở small objects (objects nhỏ) nhưng tốt hơn ở single dominant objects

**Tốc độ:**
- YOLOv8n nhanh hơn Faster R-CNN ~5-10x
- Single forward pass vs RPN + ROI head

**Khi nào dùng mô hình nào?**
- YOLOv8: ứng dụng real-time (camera, video, autonomous driving)
- Faster R-CNN: cần độ chính xác cao, không yêu cầu real-time (medical imaging, surveillance)

### Hạn chế
- Training trên GPU sẽ cho kết quả tốt hơn nhiều (cần ít nhất 8GB VRAM)
- VOC 2012 chỉ có 20 lớp — kết quả trên COCO (80 lớp) sẽ khác
"""),
)


# ─── Notebook 5: Extensions ───────────────────────────────────────────────────

NB5 = nb(
    md("""# Notebook 5 — Mở rộng: FPS Benchmark, Visualization & Demo

**Bài tập lớn số 2 · CO5085 · HCMUT 2025-2026**

## Nội dung mở rộng (40% điểm)
1. FPS Benchmark chi tiết (batch sizes, image resolutions)
2. Confidence threshold sweep — tìm threshold tối ưu
3. GradCAM visualization cho Faster R-CNN backbone
4. Gradio web demo
"""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.data import get_device, VOC_CLASSES, VOCDetectionDataset, get_val_transforms
from src.models import get_faster_rcnn
from src.train import load_frcnn_checkpoint
from src.evaluate import measure_fps
from src.utils import visualize_detections, load_metrics_json

device = get_device()
print("Device:", device)
"""),

    md("## 1. FPS Benchmark: Chi tiết theo Image Size"),

    code("""# Benchmark FPS với các image size khác nhau
image_sizes = [(320, 320), (480, 480), (640, 640), (800, 800)]

try:
    frcnn = get_faster_rcnn(num_classes=21)
    frcnn = load_frcnn_checkpoint(frcnn, '../results/checkpoints/frcnn_voc.pth', device)

    fps_results = {'Faster R-CNN': []}
    for h, w in image_sizes:
        fps_info = measure_fps(frcnn, 'frcnn', image_size=(h, w), n_runs=20, device=device)
        fps_results['Faster R-CNN'].append(fps_info['fps'])
        print(f"  {h}x{w}: {fps_info['fps']:.1f} FPS, {fps_info['ms_per_image']:.1f} ms")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sizes_str = [f"{h}x{w}" for h, w in image_sizes]
    for model_name, fps_list in fps_results.items():
        ax.plot(sizes_str, fps_list, 'o-', label=model_name, linewidth=2)
    ax.set_xlabel('Image Size')
    ax.set_ylabel('FPS')
    ax.set_title('FPS vs Image Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/plots/fps_vs_size.png', dpi=120)
    plt.show()
except Exception as e:
    print(f"Cần có checkpoint: {e}")
"""),

    md("## 2. Confidence Threshold Sweep"),

    code("""# Ảnh hưởng của confidence threshold đến số lượng detections
try:
    from src.data import get_frcnn_loaders

    _, val_loader = get_frcnn_loaders('../data/voc', batch_size=4)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    avg_detections = []

    frcnn.eval()
    images, _ = next(iter(val_loader))
    images = [img.to(device) for img in images]

    with torch.no_grad():
        outputs = frcnn(images)

    for thresh in thresholds:
        total = sum((o['scores'] >= thresh).sum().item() for o in outputs)
        avg_detections.append(total / len(images))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, avg_detections, 'o-', color='#6366f1', linewidth=2)
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Avg Detections per Image')
    ax.set_title('Confidence Threshold vs Số lượng Detections (Faster R-CNN)')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='threshold=0.5')
    ax.legend()
    plt.tight_layout()
    plt.savefig('../results/plots/confidence_sweep.png', dpi=120)
    plt.show()
except Exception as e:
    print(f"Cần có checkpoint: {e}")
"""),

    md("## 3. Feature Map Visualization"),

    code("""# Visualize feature maps từ Faster R-CNN backbone (ResNet FPN)
try:
    activation = {}

    def hook_fn(name):
        def hook(module, input, output):
            activation[name] = output.detach()
        return hook

    # Đăng ký hook vào FPN
    frcnn.backbone.fpn.layer_blocks[-1].register_forward_hook(hook_fn('fpn_last'))

    ds = VOCDetectionDataset('../data/voc', year='2012', image_set='val',
                              transforms=get_val_transforms())
    img_t, target = ds[10]

    frcnn.eval()
    with torch.no_grad():
        _ = frcnn([img_t.to(device)])

    feat = activation.get('fpn_last')
    if feat is not None:
        feat_np = feat[0].cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < feat_np.shape[0]:
                ax.imshow(feat_np[i], cmap='viridis')
                ax.set_title(f'Channel {i}')
            ax.axis('off')
        plt.suptitle('FPN Feature Maps (last layer) — Faster R-CNN')
        plt.tight_layout()
        plt.savefig('../results/plots/feature_maps.png', dpi=100)
        plt.show()
    else:
        print("Hook không capture được feature map.")
except Exception as e:
    print(f"Feature map visualization: {e}")
"""),

    md("## 4. Gradio Web Demo"),

    code("""# Interactive demo với Gradio
# Chạy cell này để khởi động web interface

try:
    import gradio as gr
    from PIL import Image
    import torchvision.transforms as T
    import numpy as np, cv2

    model_loaded = False
    try:
        demo_model = get_faster_rcnn(num_classes=21)
        demo_model = load_frcnn_checkpoint(demo_model, '../results/checkpoints/frcnn_voc.pth', 'cpu')
        demo_model.eval()
        model_loaded = True
        print("Model loaded thành công!")
    except Exception as e:
        print(f"Không load được model: {e}")

    def detect_objects(pil_img, confidence_threshold=0.5):
        if not model_loaded:
            return pil_img, "Model chưa được load. Cần chạy training trước."

        transform = T.Compose([T.ToTensor(),
                                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        img_t = transform(pil_img)

        with torch.no_grad():
            output = demo_model([img_t])[0]

        keep = output['scores'] >= confidence_threshold
        boxes = output['boxes'][keep].numpy().tolist()
        labels = output['labels'][keep].numpy().tolist()
        scores = output['scores'][keep].numpy().tolist()

        # Vẽ boxes lên ảnh
        img_np = np.array(pil_img).copy()
        palette = [(255,50,50), (50,200,50), (50,50,255), (255,165,0), (128,0,128)]

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            color = palette[(label-1) % len(palette)]
            cv2.rectangle(img_np, (x1,y1), (x2,y2), color, 2)
            name = VOC_CLASSES[label-1] if 1 <= label <= 20 else f"cls{label}"
            cv2.putText(img_np, f"{name} {score:.2f}", (x1, max(y1-5,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        summary = f"Phát hiện {len(boxes)} đối tượng:\\n"
        for box, label, score in zip(boxes, labels, scores):
            name = VOC_CLASSES[label-1] if 1 <= label <= 20 else f"cls{label}"
            summary += f"  {name}: {score:.2f}\\n"

        return Image.fromarray(img_np), summary

    demo = gr.Interface(
        fn=detect_objects,
        inputs=[
            gr.Image(type="pil", label="Upload ảnh"),
            gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Confidence Threshold"),
        ],
        outputs=[
            gr.Image(type="pil", label="Kết quả Detection"),
            gr.Textbox(label="Danh sách đối tượng"),
        ],
        title="Object Detection Demo — Faster R-CNN trên Pascal VOC 2012",
        description="Upload ảnh để phát hiện 20 loại đối tượng (VOC classes)",
    )

    demo.launch(share=False, server_port=7860)
    print("Demo đang chạy tại http://localhost:7860")

except ImportError:
    print("Cần cài gradio: pip install gradio")
except Exception as e:
    print(f"Demo error: {e}")
"""),
)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)

    notebooks = [
        ("notebooks/nb1_eda.ipynb", NB1),
        ("notebooks/nb2_data_pipeline.ipynb", NB2),
        ("notebooks/nb3_train_compare.ipynb", NB3),
        ("notebooks/nb4_results.ipynb", NB4),
        ("notebooks/nb5_extensions.ipynb", NB5),
    ]

    for path, notebook in notebooks:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"✓ Created {path}")

    print(f"\nĐã tạo {len(notebooks)} notebooks trong thư mục notebooks/")
    print("Mở VS Code và chọn kernel 'deeplearning-ass1' để chạy.")
