"""
run_finetune_frcnn.py — Fine-tune Faster R-CNN ResNet-50 FPN trên Pascal VOC 2012

Chạy:
    python scripts/run_finetune_frcnn.py
    python scripts/run_finetune_frcnn.py --epochs 5 --batch 2
    python scripts/run_finetune_frcnn.py --dry-run

Kết quả lưu vào results/checkpoints/frcnn_voc.pth và results/metrics/frcnn_results.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from exercise_2.src.data import get_frcnn_loaders, get_device
from exercise_2.src.models import get_faster_rcnn, get_model_info
from exercise_2.src.train import fit_frcnn, load_frcnn_checkpoint
from exercise_2.src.evaluate import predict_frcnn, compute_map_coco, measure_fps
from exercise_2.src.utils import (
    save_metrics_json,
    print_detection_results_table,
    plot_loss_curves,
    get_param_count,
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on Pascal VOC 2012")
    p.add_argument("--epochs", type=int, default=10, help="Số epoch (default: 10)")
    p.add_argument("--batch", type=int, default=4, help="Batch size (default: 4)")
    p.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    p.add_argument("--workers", type=int, default=2, help="Số data workers (default: 2)")
    p.add_argument("--data-dir", type=str, default="data/voc",
                   help="Thư mục VOC data (default: data/voc)")
    p.add_argument("--checkpoint", type=str, default="results/checkpoints/frcnn_voc.pth",
                   help="Đường dẫn lưu checkpoint")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ chạy 1 epoch với batch=2 để kiểm tra")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    if args.dry_run:
        print("[DRY RUN] Chỉ chạy 1 epoch × 3 batch để kiểm tra pipeline")
        args.epochs = 1
        args.batch = 2

    print("=" * 60)
    print("Fine-tune Faster R-CNN ResNet-50 FPN trên Pascal VOC 2012")
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print("=" * 60)

    # 1. Data
    print("\n[1/5] Tải dataset...")
    train_loader, val_loader = get_frcnn_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch,
        num_workers=args.workers,
    )

    # 2. Model
    print("\n[2/5] Khởi tạo mô hình...")
    model = get_faster_rcnn(num_classes=21, pretrained_backbone=True, trainable_backbone_layers=3)
    info = get_model_info(model, "Faster R-CNN ResNet-50 FPN")
    print(f"  Total params: {info['total_params']} | Trainable: {info['trainable_params']}")

    # 3. Training
    print("\n[3/5] Fine-tuning...")
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "device": device,
        "save_path": args.checkpoint,
        "max_batches": 3 if args.dry_run else None,
    }
    history = fit_frcnn(model, train_loader, val_loader, config)

    # Lưu loss curves
    plot_loss_curves(history, title="Faster R-CNN Training Loss",
                     save_path="results/plots/frcnn_loss.png")

    # 4. Evaluate mAP
    print("\n[4/5] Đánh giá mAP trên validation set...")
    model = load_frcnn_checkpoint(model, args.checkpoint, device)
    max_eval = 3 if args.dry_run else None
    preds, targets = predict_frcnn(model, val_loader, device, score_threshold=0.05,
                                   max_batches=max_eval)
    map_results = compute_map_coco(preds, targets)
    print(f"  mAP@0.5      = {map_results['mAP_50']*100:.1f}%")
    print(f"  mAP@0.5:0.95 = {map_results['mAP_50_95']*100:.1f}%")

    # 5. FPS
    print("\n[5/5] Benchmark FPS...")
    fps_info = measure_fps(model, "frcnn", device=device)
    print(f"  FPS ({device}): {fps_info['fps']} | {fps_info['ms_per_image']} ms/image")

    # Lưu kết quả
    results = {
        "model": "Faster R-CNN ResNet-50 FPN",
        "type": "two-stage",
        "dataset": "Pascal VOC 2012",
        "mAP_50": map_results["mAP_50"],
        "mAP_50_95": map_results["mAP_50_95"],
        "AP_per_class": map_results.get("AP_per_class", {}),
        "fps": fps_info["fps"],
        "ms_per_image": fps_info["ms_per_image"],
        "fps_device": fps_info["device"],
        "epochs": args.epochs,
        "params": info["total_params"],
        "checkpoint": args.checkpoint,
        "history": history,
    }

    save_metrics_json(results, "results/metrics/frcnn_results.json")
    print_detection_results_table({"Faster R-CNN": results})
    print("\nHoàn tất! Kết quả đã lưu vào results/metrics/frcnn_results.json")


if __name__ == "__main__":
    main()
