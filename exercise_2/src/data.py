"""
data.py — Tải và xử lý dữ liệu Pascal VOC 2012 cho Object Detection

Pascal VOC 2012:
- 20 lớp đối tượng, ~11,530 ảnh train / ~2,788 ảnh val
- Annotation dạng XML với bounding boxes (xmin, ymin, xmax, ymax)
- Torchvision hỗ trợ sẵn qua datasets.VOCDetection
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─── Hằng số dataset ──────────────────────────────────────────────────────────

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES = 20

# ImageNet normalization — cả YOLOv8 lẫn Faster R-CNN đều dùng pretrained ImageNet backbone
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device():
    """Tự động chọn thiết bị: CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── EDA Utilities ────────────────────────────────────────────────────────────

def get_voc_stats(data_dir: str) -> dict:
    """
    Thống kê dataset Pascal VOC: số ảnh, số instance, phân phối theo lớp.

    Args:
        data_dir: thư mục chứa VOCdevkit (torchvision sẽ tạo cấu trúc này)

    Returns:
        dict với keys: n_train, n_val, class_counts, boxes_per_image_stats
    """
    stats = {"n_train": 0, "n_val": 0, "class_counts": {c: 0 for c in VOC_CLASSES},
             "boxes_per_image": []}

    voc_root = Path(data_dir) / "VOCdevkit" / "VOC2012"
    if not voc_root.exists():
        print(f"[WARN] VOC data chưa download tại {voc_root}. Chạy get_frcnn_loaders() trước.")
        return stats

    for split, key in [("train", "n_train"), ("val", "n_val")]:
        split_file = voc_root / "ImageSets" / "Main" / f"{split}.txt"
        if not split_file.exists():
            continue
        image_ids = split_file.read_text().strip().splitlines()
        stats[key] = len(image_ids)

        for img_id in image_ids:
            ann_path = voc_root / "Annotations" / f"{img_id.strip()}.xml"
            if not ann_path.exists():
                continue
            tree = ET.parse(ann_path)
            root = tree.getroot()
            boxes = root.findall("object")
            stats["boxes_per_image"].append(len(boxes))
            for obj in boxes:
                name = obj.find("name").text.lower().strip()
                if name in stats["class_counts"]:
                    stats["class_counts"][name] += 1

    bpi = stats["boxes_per_image"]
    if bpi:
        import numpy as np
        stats["boxes_per_image_stats"] = {
            "mean": float(np.mean(bpi)),
            "median": float(np.median(bpi)),
            "max": int(np.max(bpi)),
            "min": int(np.min(bpi)),
        }
    return stats


# ─── Faster R-CNN Dataset ─────────────────────────────────────────────────────

class VOCDetectionDataset(torch.utils.data.Dataset):
    """
    Wrapper torchvision.datasets.VOCDetection cho Faster R-CNN.

    Trả về:
        image: FloatTensor [C, H, W] trong khoảng [0, 1] sau normalize
        target: dict với
            - boxes:    FloatTensor [N, 4] — [x1, y1, x2, y2] pixels tuyệt đối
            - labels:   LongTensor  [N]    — class index 1-indexed (0 = background)
            - image_id: int
            - area:     FloatTensor [N]
            - iscrowd:  LongTensor  [N]    — luôn 0 cho VOC
    """

    def __init__(self, root: str, year: str = "2012",
                 image_set: str = "train", transforms=None):
        from pathlib import Path as _Path
        # Chỉ download nếu VOCdevkit chưa tồn tại — tránh MD5 check mỗi lần
        already_extracted = (_Path(root) / "VOCdevkit" / f"VOC{year}").exists()
        self.voc = datasets.VOCDetection(
            root=root, year=year, image_set=image_set,
            download=not already_extracted,
        )
        self.transforms = transforms
        self._class_to_idx = {c: i + 1 for i, c in enumerate(VOC_CLASSES)}

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, ann = self.voc[idx]

        objs = ann["annotation"].get("object", [])
        if isinstance(objs, dict):
            objs = [objs]

        boxes, labels = [], []
        for obj in objs:
            name = obj["name"].lower().strip()
            if name not in self._class_to_idx:
                continue
            bb = obj["bndbox"]
            x1 = float(bb["xmin"])
            y1 = float(bb["ymin"])
            x2 = float(bb["xmax"])
            y2 = float(bb["ymax"])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(self._class_to_idx[name])

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * \
               (boxes_tensor[:, 2] - boxes_tensor[:, 0]) if len(boxes_tensor) else torch.zeros(0)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def collate_fn(batch):
    """
    Custom collate cho detection: trả về (list_of_images, list_of_targets).
    Torchvision detection models yêu cầu định dạng này vì ảnh có kích thước khác nhau.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_frcnn_loaders(
    data_dir: str = "data/voc",
    batch_size: int = 4,
    num_workers: int = 2,
) -> tuple:
    """
    Tạo DataLoader train/val cho Faster R-CNN.

    Args:
        data_dir: thư mục chứa (hoặc sẽ download) VOCdevkit
        batch_size: số ảnh mỗi batch (4 là hợp lý cho RAM 16GB)
        num_workers: số worker song song

    Returns:
        (train_loader, val_loader)
    """
    train_ds = VOCDetectionDataset(data_dir, year="2012", image_set="train",
                                   transforms=get_train_transforms())
    val_ds = VOCDetectionDataset(data_dir, year="2012", image_set="val",
                                 transforms=get_val_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=False,
    )
    print(f"[Data] Train: {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh")
    return train_loader, val_loader


# ─── Transforms ───────────────────────────────────────────────────────────────

def get_train_transforms():
    """
    Augmentation cho training. Normalize theo ImageNet mean/std vì cả hai mô hình
    đều dùng backbone pretrained trên ImageNet.

    Lưu ý: Faster R-CNN nhận FloatTensor [0,1] (KHÔNG normalize) theo torchvision API,
    nhưng ta normalize ở đây vì ta tự kiểm soát pipeline.
    Với YOLOv8, ta dùng YOLO format riêng (prepare_yolo_dataset).
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """Transforms cho validation: chỉ ToTensor + Normalize, không augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── YOLO Format Conversion ───────────────────────────────────────────────────

def prepare_yolo_dataset(
    voc_root: str = "data/voc",
    output_dir: str = "data/voc_yolo",
    splits: list = None,
) -> str:
    """
    Chuyển đổi Pascal VOC XML → YOLO txt format để dùng với ultralytics.

    YOLO label format (mỗi dòng): <class_id> <xc> <yc> <w> <h>
    Tất cả giá trị normalized về [0, 1] theo kích thước ảnh.

    Args:
        voc_root: thư mục VOCdevkit (chứa VOCdevkit/VOC2012/)
        output_dir: thư mục đích (sẽ tạo images/ và labels/)
        splits: danh sách splits, mặc định ["train", "val"]

    Returns:
        Đường dẫn đến file voc.yaml (cần thiết cho ultralytics trainer)
    """
    if splits is None:
        splits = ["train", "val"]

    voc_base = Path(voc_root) / "VOCdevkit" / "VOC2012"
    out_base = Path(output_dir)

    for split in splits:
        img_out = out_base / "images" / split
        lbl_out = out_base / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        split_file = voc_base / "ImageSets" / "Main" / f"{split}.txt"
        if not split_file.exists():
            print(f"[WARN] Không tìm thấy {split_file}")
            continue

        image_ids = split_file.read_text().strip().splitlines()
        converted, skipped = 0, 0

        for img_id in image_ids:
            img_id = img_id.strip()
            src_img = voc_base / "JPEGImages" / f"{img_id}.jpg"
            ann_file = voc_base / "Annotations" / f"{img_id}.xml"

            if not src_img.exists() or not ann_file.exists():
                skipped += 1
                continue

            # Đọc kích thước ảnh từ XML
            tree = ET.parse(ann_file)
            root = tree.getroot()
            size = root.find("size")
            img_w = float(size.find("width").text)
            img_h = float(size.find("height").text)

            lines = []
            for obj in root.findall("object"):
                name = obj.find("name").text.lower().strip()
                if name not in VOC_CLASSES:
                    continue
                cls_id = VOC_CLASSES.index(name)
                bb = obj.find("bndbox")
                x1 = float(bb.find("xmin").text)
                y1 = float(bb.find("ymin").text)
                x2 = float(bb.find("xmax").text)
                y2 = float(bb.find("ymax").text)

                xc = (x1 + x2) / 2 / img_w
                yc = (y1 + y2) / 2 / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w = max(0.001, min(1.0, w))
                h = max(0.001, min(1.0, h))

                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if not lines:
                skipped += 1
                continue

            # Copy ảnh (symlink để tiết kiệm disk)
            dst_img = img_out / f"{img_id}.jpg"
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(src_img.resolve())
                except OSError:
                    import shutil
                    shutil.copy2(src_img, dst_img)

            (lbl_out / f"{img_id}.txt").write_text("\n".join(lines))
            converted += 1

        print(f"[YOLO Prep] {split}: {converted} ảnh converted, {skipped} skipped")

    # Tạo voc.yaml cho ultralytics
    yaml_path = out_base / "voc.yaml"
    names_str = "\n".join(f"  {i}: {c}" for i, c in enumerate(VOC_CLASSES))
    yaml_content = f"""# Pascal VOC 2012 — YOLO format
path: {out_base.resolve()}
train: images/train
val: images/val

nc: {NUM_CLASSES}
names:
{names_str}
"""
    yaml_path.write_text(yaml_content)
    print(f"[YOLO Prep] Wrote {yaml_path}")
    return str(yaml_path)
