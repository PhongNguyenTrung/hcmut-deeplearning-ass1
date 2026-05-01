"""
run_finetune_yolo.py — Fine-tune YOLOv8 trên Pascal VOC 2012

Chạy:
    python scripts/run_finetune_yolo.py
    python scripts/run_finetune_yolo.py --epochs 5 --size n --dry-run

Kết quả lưu vào results/yolo_voc/ và results/metrics/yolo_results.json
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Đảm bảo import được package exercise_2.src từ thư mục gốc dự án
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from exercise_2.src.data import prepare_yolo_dataset, get_device
from exercise_2.src.train import train_yolov8
from exercise_2.src.evaluate import measure_fps
from exercise_2.src.utils import save_metrics_json, print_detection_results_table


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Pascal VOC 2012")
    p.add_argument("--epochs", type=int, default=20, help="Số epoch (default: 20)")
    p.add_argument("--size", type=str, default="n", choices=["n", "s", "m"],
                   help="Model size: n=nano, s=small, m=medium (default: n)")
    p.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    p.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    p.add_argument("--data-dir", type=str, default="data/voc",
                   help="Thư mục VOC data (default: data/voc)")
    p.add_argument("--yolo-dir", type=str, default="data/voc_yolo",
                   help="Thư mục YOLO format output (default: data/voc_yolo)")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ chạy 2 epoch để kiểm tra pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    if args.dry_run:
        print("[DRY RUN] Chỉ chạy 2 epoch với batch=4")
        args.epochs = 2
        args.batch = 4

    print("=" * 60)
    print(f"Fine-tune YOLOv8{args.size} trên Pascal VOC 2012")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch}")
    print("=" * 60)

    # 1. Download VOC nếu chưa có, rồi convert sang YOLO format
    print("\n[1/4] Chuẩn bị dataset YOLO format...")
    from exercise_2.src.data import VOCDetectionDataset
    from pathlib import Path as _Path
    if not (_Path(args.data_dir) / "VOCdevkit" / "VOC2012").exists():
        print("  Download VOC 2012 (~2GB, chỉ cần lần đầu)...")
    VOCDetectionDataset(args.data_dir, year="2012", image_set="train", transforms=None)
    VOCDetectionDataset(args.data_dir, year="2012", image_set="val", transforms=None)

    yaml_path = prepare_yolo_dataset(
        voc_root=args.data_dir,
        output_dir=args.yolo_dir,
        splits=["train", "val"],
    )

    # 2. Fine-tune
    print(f"\n[2/4] Fine-tune YOLOv8{args.size}...")
    best_pt = train_yolov8(
        data_yaml=yaml_path,
        model_size=args.size,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="results",
        name=f"yolov8{args.size}_voc",
        device="auto",
    )

    # 3. Đánh giá mAP bằng ultralytics validate
    print("\n[3/4] Đánh giá mAP trên validation set...")
    try:
        from ultralytics import YOLO
        model = YOLO(best_pt)
        val_results = model.val(data=yaml_path, imgsz=args.imgsz, verbose=False)
        map50 = float(val_results.box.map50)
        map5095 = float(val_results.box.map)
        ap_per_class_list = val_results.box.ap_class_index.tolist() \
            if hasattr(val_results.box, "ap_class_index") else []
        print(f"  mAP@0.5     = {map50*100:.1f}%")
        print(f"  mAP@0.5:0.95 = {map5095*100:.1f}%")
    except Exception as e:
        print(f"  [WARN] Không thể tính mAP từ ultralytics: {e}")
        map50, map5095 = 0.0, 0.0

    # 4. FPS benchmark
    print("\n[4/4] Benchmark FPS...")
    try:
        from ultralytics import YOLO as YOLOModel
        yolo_model = YOLOModel(best_pt)
        fps_info = measure_fps(yolo_model, "yolo", device=device)
        print(f"  FPS ({device}): {fps_info['fps']} | {fps_info['ms_per_image']} ms/image")
    except Exception as e:
        print(f"  [WARN] Không benchmark được FPS: {e}")
        fps_info = {"fps": 0, "ms_per_image": 0, "device": device}

    # 5. Lưu kết quả
    results = {
        "model": f"YOLOv8{args.size}",
        "type": "one-stage",
        "dataset": "Pascal VOC 2012",
        "mAP_50": map50,
        "mAP_50_95": map5095,
        "fps": fps_info["fps"],
        "ms_per_image": fps_info["ms_per_image"],
        "fps_device": fps_info["device"],
        "epochs": args.epochs,
        "checkpoint": best_pt,
    }

    save_metrics_json(results, "results/metrics/yolo_results.json")
    print_detection_results_table({f"YOLOv8{args.size}": results})
    print("\nHoàn tất! Kết quả đã lưu vào results/metrics/yolo_results.json")


if __name__ == "__main__":
    main()
