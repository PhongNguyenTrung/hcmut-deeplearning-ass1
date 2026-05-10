"""extract_yolo_per_class.py — Bổ sung AP_per_class vào yolo_results.json đã có.

Run val lại trên YOLOv8 checkpoint (~30s trên T4) và update file JSON cũ
mà KHÔNG động đến mAP/FPS đã có. Dùng khi bạn đã train xong nhưng quên
extract per-class AP lúc đầu.

Run:
    python exercise_2/scripts/extract_yolo_per_class.py
    python exercise_2/scripts/extract_yolo_per_class.py --ckpt path/to/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from exercise_2.src.data import VOC_CLASSES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="Đường dẫn best.pt (mặc định: đọc từ yolo_results.json)")
    p.add_argument("--yaml", type=str, default="data/voc_yolo/voc.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--json", type=str,
                   default="exercise_2/results/metrics/yolo_results.json")
    args = p.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"[ERROR] {json_path} không tồn tại")
        sys.exit(1)

    data = json.loads(json_path.read_text())
    ckpt = args.ckpt or data.get("checkpoint")
    if not ckpt or not Path(ckpt).exists():
        print(f"[ERROR] Checkpoint không tồn tại: {ckpt}")
        print("        Truyền --ckpt /path/to/best.pt")
        sys.exit(1)

    from ultralytics import YOLO
    print(f"[Eval] Loading {ckpt} ...")
    model = YOLO(ckpt)

    print(f"[Eval] Running val() on {args.yaml} ...")
    val_results = model.val(data=args.yaml, imgsz=args.imgsz, verbose=False)

    ap_per_class = {}
    if hasattr(val_results.box, "ap50") and hasattr(val_results.box, "ap_class_index"):
        ap50_arr = val_results.box.ap50
        class_idx = val_results.box.ap_class_index.astype(int)
        for i, cls_id in enumerate(class_idx):
            if 0 <= cls_id < len(VOC_CLASSES):
                ap_per_class[VOC_CLASSES[cls_id]] = float(ap50_arr[i])

    if not ap_per_class:
        print("[WARN] Không trích xuất được AP_per_class — kiểm tra phiên bản ultralytics")
        sys.exit(1)

    data["AP_per_class"] = ap_per_class
    json_path.write_text(json.dumps(data, indent=2))
    print(f"[Saved] {json_path} — {len(ap_per_class)} class entries")

    sorted_ap = sorted(ap_per_class.items(), key=lambda x: -x[1])
    print("\nTop 5:")
    for c, ap in sorted_ap[:5]:
        print(f"  {c:14s} {ap * 100:.1f}%")
    print("\nBottom 5:")
    for c, ap in sorted_ap[-5:]:
        print(f"  {c:14s} {ap * 100:.1f}%")


if __name__ == "__main__":
    main()
