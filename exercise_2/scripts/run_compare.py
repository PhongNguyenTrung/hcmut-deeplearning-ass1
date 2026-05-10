"""
run_compare.py — So sánh kết quả YOLOv8 và Faster R-CNN

Tải metrics từ results/metrics/*.json và tạo:
- Bảng so sánh trên terminal
- Biểu đồ mAP comparison (grouped bar)
- Biểu đồ Speed vs Accuracy (scatter)
- Biểu đồ per-class AP (horizontal bar)

Chạy sau khi đã chạy xong cả hai script fine-tune:
    python scripts/run_compare.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from exercise_2.src.utils import (
    load_metrics_json,
    print_detection_results_table,
    plot_map_comparison,
    plot_per_class_ap,
    plot_speed_accuracy_tradeoff,
)


def main():
    yolo_path = "exercise_2/results/metrics/yolo_results.json"
    frcnn_path = "exercise_2/results/metrics/frcnn_results.json"

    print("=" * 60)
    print("So sánh kết quả: YOLOv8 vs Faster R-CNN")
    print("=" * 60)

    missing = []
    if not Path(yolo_path).exists():
        missing.append(yolo_path)
    if not Path(frcnn_path).exists():
        missing.append(frcnn_path)

    if missing:
        print("\n[WARN] Chưa có file kết quả:")
        for f in missing:
            print(f"  - {f}")
        print("\nChạy các lệnh sau trước:")
        if yolo_path in missing:
            print("  python scripts/run_finetune_yolo.py")
        if frcnn_path in missing:
            print("  python scripts/run_finetune_frcnn.py")
        return

    yolo_data = load_metrics_json(yolo_path)
    frcnn_data = load_metrics_json(frcnn_path)

    yolo_name = yolo_data.get("model", "YOLOv8n")
    frcnn_name = frcnn_data.get("model", "Faster R-CNN")

    results = {
        yolo_name: yolo_data,
        frcnn_name: frcnn_data,
    }

    # Bảng kết quả
    print_detection_results_table(results)

    # Biểu đồ
    print("\nĐang tạo biểu đồ...")

    plot_map_comparison(results, save_path="exercise_2/results/plots/map_comparison.png")
    plot_speed_accuracy_tradeoff(results, save_path="exercise_2/results/plots/speed_accuracy.png")

    if "AP_per_class" in yolo_data or "AP_per_class" in frcnn_data:
        plot_per_class_ap(results, save_path="exercise_2/results/plots/per_class_ap.png")

    print("\nBiểu đồ đã lưu vào exercise_2/results/plots/")

    # Phân tích
    map50_yolo = yolo_data.get("mAP_50", 0) * 100
    map50_frcnn = frcnn_data.get("mAP_50", 0) * 100
    fps_yolo = yolo_data.get("fps", 0)
    fps_frcnn = frcnn_data.get("fps", 0)

    print("\n── Nhận xét ──────────────────────────────────────────────────")
    if map50_frcnn > map50_yolo:
        diff = map50_frcnn - map50_yolo
        print(f"  Faster R-CNN chính xác hơn YOLOv8 {diff:.1f}% mAP@0.5")
    else:
        diff = map50_yolo - map50_frcnn
        print(f"  YOLOv8 chính xác hơn Faster R-CNN {diff:.1f}% mAP@0.5")

    if fps_yolo > 0 and fps_frcnn > 0:
        ratio = fps_yolo / fps_frcnn
        print(f"  YOLOv8 nhanh hơn Faster R-CNN {ratio:.1f}x (FPS)")
        print("  → Trade-off: YOLOv8 phù hợp real-time; Faster R-CNN cho độ chính xác cao hơn")


if __name__ == "__main__":
    main()
