# Bài tập lớn số 2 — Object Detection: One-Stage vs Two-Stage

**CO5085 · Deep Learning & Computer Vision · HCMUT 2025-2026**
**Giảng viên:** Lê Thành Sách

## Chủ đề
**Phát hiện đối tượng (Object Detection)** — So sánh kiến trúc one-stage và two-stage detector:
- **YOLOv8n** (ultralytics) — one-stage, anchor-free, 3.2M params
- **Faster R-CNN ResNet-50 FPN** (torchvision) — two-stage, 41.8M params

## Dataset
**Pascal VOC 2012** — 20 lớp đối tượng, ~11,530 train / ~2,788 val

## Cài đặt

```bash
# Từ thư mục gốc dự án
source .venv/bin/activate

# Cài thêm dependencies cho exercise_2
pip install ultralytics>=8.0.0 pycocotools>=2.0.6 opencv-python>=4.8.0
```

## Chạy Training

```bash
cd hcmut-deeplearning-ass1

# Fine-tune YOLOv8n (khuyến nghị bắt đầu với --dry-run)
python exercise_2/scripts/run_finetune_yolo.py --dry-run
python exercise_2/scripts/run_finetune_yolo.py --epochs 2

# Fine-tune Faster R-CNN
python exercise_2/scripts/run_finetune_frcnn.py --dry-run
python exercise_2/scripts/run_finetune_frcnn.py --epochs 2

# So sánh kết quả (sau khi cả hai xong)
python exercise_2/scripts/run_compare.py
```

## Notebooks

```bash
# Tái tạo notebooks
python exercise_2/create_notebooks.py
```

| Notebook | Nội dung |
|----------|----------|
| [nb1_eda.ipynb](notebooks/nb1_eda.ipynb) | EDA: dataset stats, class distribution, bbox analysis |
| [nb2_data_pipeline.ipynb](notebooks/nb2_data_pipeline.ipynb) | DataLoader, augmentation, YOLO format conversion |
| [nb3_train_compare.ipynb](notebooks/nb3_train_compare.ipynb) | Kiến trúc, training, so sánh cơ bản |
| [nb4_results.ipynb](notebooks/nb4_results.ipynb) | mAP tables, biểu đồ, error analysis |
| [nb5_extensions.ipynb](notebooks/nb5_extensions.ipynb) | FPS benchmark, feature maps, Gradio demo |

## Kết quả (dự kiến)

| Mô hình | mAP@0.5 | mAP@0.5:0.95 | FPS (CPU) | Params |
|---------|---------|--------------|-----------|--------|
| YOLOv8n | ~60-65% | ~35-42% | ~15-25 | 3.2M |
| Faster R-CNN ResNet-50 FPN | ~70-75% | ~45-52% | ~1-3 | 41.8M |

*Điền số liệu thực sau khi chạy training.*

## Cấu trúc thư mục

```
exercise_2/
├── src/
│   ├── data.py       # VOC dataset, YOLO format conversion
│   ├── models.py     # YOLOv8 + Faster R-CNN
│   ├── train.py      # Training loops
│   ├── evaluate.py   # mAP, FPS, error analysis
│   └── utils.py      # Visualization, plots
├── scripts/
│   ├── run_finetune_yolo.py
│   ├── run_finetune_frcnn.py
│   └── run_compare.py
├── notebooks/        # 5 Jupyter notebooks
├── results/
│   ├── checkpoints/  # Model weights
│   ├── plots/        # Biểu đồ
│   └── metrics/      # JSON results
└── create_notebooks.py
```
